"""Reuse the result of a sync across rollouts.

A sync is the slowest transition of the environment (minutes with Airbyte), and its result depends only on the
connection configuration as long as the sources do not change. ``CachingEngine`` runs each distinct configuration
once, keeps the loaded tables in a snapshot namespace, and serves every later sync of the same configuration by
copying the snapshot inside the warehouse.

Only a sync in which every stream loaded becomes a snapshot. A failure can come from the environment (a timeout,
Airbyte restarting) as well as from the configuration, so it is never served to a later request: the next sync of
that configuration runs again. Requests that were already waiting for the sync that failed get its outcome.

The rollout observes what a real sync would produce: the same tables, rows and row counts, and duplicated rows
when it appends twice. Only the loader's metadata columns differ: they are copied with the rows, so every copy
carries the snapshot's ``_airbyte_raw_id`` and ``_airbyte_extracted_at``. Grading ignores those columns.
"""

import hashlib
import itertools
import json
import threading
from dataclasses import asdict, dataclass, replace

from elt_rlvr.extract_load.base import ConnectionRejectedError, ExtractLoadEngine, StreamResult
from elt_rlvr.pipeline import ConnectionConfig, StreamConfig
from elt_rlvr.warehouse.base import Warehouse
from elt_rlvr.warehouse.state import read_state, write_state

SNAPSHOT_SYNC_MODES = ("full_refresh_overwrite", "full_refresh_append")


@dataclass
class _Outcome:
    """What one sync of a configuration into its snapshot namespace produced."""

    namespace: str
    results: list[StreamResult] | None = None
    rejection: str | None = None
    finished: int = 0
    """Position of the end of a failed sync in the engine's order of events."""

    @property
    def complete(self) -> bool:
        return bool(self.results) and all(result.succeeded for result in self.results)


class CachingEngine(ExtractLoadEngine):
    """Wraps an engine and serves repeated syncs of one configuration from a snapshot."""

    def __init__(self, inner: ExtractLoadEngine) -> None:
        self._inner = inner
        self._snapshots: dict[str, _Outcome] = {}
        self._failures: dict[str, _Outcome] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._events = itertools.count()
        """Orders the arrival of requests and the end of failed syncs."""
        self.hits = 0
        self.misses = 0

    def source_types(self) -> tuple[str, ...]:
        return self._inner.source_types()

    def sync_variant(self, connection: ConnectionConfig) -> str:
        return self._inner.sync_variant(connection)

    def _key(self, connection: ConnectionConfig) -> str:
        payload = {
            "source_type": connection.source_type,
            "configuration": connection.configuration,
            "streams": sorted(stream.name for stream in connection.streams),
            "variant": self._inner.sync_variant(connection),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _run(self, key: str, connection: ConnectionConfig, warehouse: Warehouse) -> _Outcome:
        """Sync ``connection`` into its snapshot namespace. The caller holds the lock of ``key``."""
        outcome = _Outcome(namespace=f"sync_cache_{key}")
        if warehouse.persistent and self._restore(outcome, warehouse):
            self.hits += 1
            self._snapshots[key] = outcome
            return outcome
        self.misses += 1
        warehouse.create_namespace(outcome.namespace)
        overwrite = replace(
            connection,
            streams=tuple(StreamConfig(stream.name, "full_refresh_overwrite") for stream in connection.streams),
        )
        try:
            outcome.results = self._inner.sync(overwrite, warehouse=warehouse, namespace=outcome.namespace)
        except ConnectionRejectedError as error:
            outcome.rejection = str(error)
        if outcome.complete:
            if warehouse.persistent:
                write_state(warehouse, outcome.namespace, {"results": [asdict(result) for result in outcome.results]})
            self._snapshots[key] = outcome
        else:
            with self._lock:
                outcome.finished = next(self._events)
            self._failures[key] = outcome
        return outcome

    def _restore(self, outcome: _Outcome, warehouse: Warehouse) -> bool:
        """Fill ``outcome`` from the snapshot an earlier process left in the warehouse. False if none is usable."""
        state = read_state(warehouse, outcome.namespace)
        if state is None:
            return False
        results = [StreamResult(**result) for result in state["results"]]
        stored = {name.lower() for name in warehouse.list_tables(outcome.namespace)}
        for result in results:
            if not result.succeeded or result.table.lower() not in stored:
                return False
            if warehouse.row_count(outcome.namespace, result.table) != result.rows_loaded:
                return False
        outcome.results = results
        return outcome.complete

    def _serve(
        self, outcome: _Outcome, connection: ConnectionConfig, warehouse: Warehouse, namespace: str
    ) -> list[StreamResult]:
        """Give ``namespace`` what the sync behind ``outcome`` loaded, as a sync of ``connection`` would."""
        if outcome.rejection is not None:
            raise ConnectionRejectedError(outcome.rejection)
        sync_modes = {stream.name: stream.sync_mode for stream in connection.streams}
        results: list[StreamResult] = []
        for cached in outcome.results or []:
            if not cached.succeeded:
                results.append(replace(cached, connection=connection.name))
                continue
            rows = warehouse.copy_table(
                outcome.namespace,
                cached.table,
                namespace,
                append=sync_modes[cached.stream] == "full_refresh_append",
            )
            results.append(replace(cached, connection=connection.name, rows_loaded=rows))
        return results

    def sync(self, connection: ConnectionConfig, *, warehouse: Warehouse, namespace: str) -> list[StreamResult]:
        if any(stream.sync_mode not in SNAPSHOT_SYNC_MODES for stream in connection.streams):
            return self._inner.sync(connection, warehouse=warehouse, namespace=namespace)
        key = self._key(connection)
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
            asked = next(self._events)
        # Concurrent rollouts that ask for the same configuration wait for one sync instead of starting their own.
        with key_lock:
            outcome = self._snapshots.get(key)
            if outcome is not None:
                self.hits += 1
            else:
                failure = self._failures.get(key)
                if failure is not None and failure.finished > asked:
                    # This request arrived before that sync ended and has waited for it.
                    self.hits += 1
                    outcome = failure
                else:
                    outcome = self._run(key, connection, warehouse)
                if not outcome.complete:
                    # Served before the lock is released: the next sync of this configuration replaces the
                    # namespace that holds what this one loaded.
                    return self._serve(outcome, connection, warehouse, namespace)
        return self._serve(outcome, connection, warehouse, namespace)

    def close(self) -> None:
        self._inner.close()
