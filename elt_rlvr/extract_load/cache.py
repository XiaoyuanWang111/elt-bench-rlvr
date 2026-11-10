"""Reuse the result of a sync across rollouts.

A sync is the slowest transition of the environment (minutes with Airbyte), and its result depends only on the
connection configuration as long as the sources do not change. ``CachingEngine`` runs each distinct configuration
once, keeps the loaded tables in a snapshot namespace, and serves every later sync of the same configuration by
copying the snapshot inside the warehouse. Invalid configurations are remembered too, with their error message.

The rollout observes what a real sync would produce: the same tables, rows and row counts, duplicated rows when it
appends twice, and the same error for a bad configuration. Only the loader's metadata columns differ: they are
copied with the rows, so every copy carries the snapshot's ``_airbyte_raw_id`` and ``_airbyte_extracted_at``.
Grading ignores those columns.
"""

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, replace

from elt_rlvr.extract_load.base import ConnectionRejectedError, ExtractLoadEngine, StreamResult
from elt_rlvr.pipeline import ConnectionConfig, StreamConfig
from elt_rlvr.warehouse.base import Warehouse
from elt_rlvr.warehouse.state import read_state, write_state

SNAPSHOT_SYNC_MODES = ("full_refresh_overwrite", "full_refresh_append")


@dataclass
class _Snapshot:
    namespace: str
    results: list[StreamResult] | None = None
    rejection: str | None = None


def _snapshot_key(connection: ConnectionConfig) -> str:
    payload = {
        "source_type": connection.source_type,
        "configuration": connection.configuration,
        "streams": sorted(stream.name for stream in connection.streams),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class CachingEngine(ExtractLoadEngine):
    """Wraps an engine and serves repeated syncs of one configuration from a snapshot."""

    def __init__(self, inner: ExtractLoadEngine) -> None:
        self._inner = inner
        self._snapshots: dict[str, _Snapshot] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def source_types(self) -> tuple[str, ...]:
        return self._inner.source_types()

    def _snapshot(self, connection: ConnectionConfig, warehouse: Warehouse) -> _Snapshot:
        key = _snapshot_key(connection)
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        # Concurrent rollouts that ask for the same configuration wait for one sync instead of starting their own.
        with key_lock:
            snapshot = self._snapshots.get(key)
            if snapshot is not None:
                self.hits += 1
                return snapshot
            snapshot = _Snapshot(namespace=f"sync_cache_{key}")
            if warehouse.persistent and self._restore(snapshot, warehouse):
                self.hits += 1
                self._snapshots[key] = snapshot
                return snapshot
            self.misses += 1
            warehouse.create_namespace(snapshot.namespace)
            overwrite = replace(
                connection,
                streams=tuple(StreamConfig(stream.name, "full_refresh_overwrite") for stream in connection.streams),
            )
            try:
                snapshot.results = self._inner.sync(overwrite, warehouse=warehouse, namespace=snapshot.namespace)
            except ConnectionRejectedError as error:
                snapshot.rejection = str(error)
            if warehouse.persistent:
                results = [asdict(result) for result in snapshot.results or []]
                write_state(warehouse, snapshot.namespace, {"results": results, "rejection": snapshot.rejection})
            self._snapshots[key] = snapshot
            return snapshot

    def _restore(self, snapshot: _Snapshot, warehouse: Warehouse) -> bool:
        """Fill ``snapshot`` from what an earlier process left in the warehouse. False if nothing usable is there."""
        state = read_state(warehouse, snapshot.namespace)
        if state is None:
            return False
        results = [StreamResult(**result) for result in state["results"]]
        stored = {name.lower() for name in warehouse.list_tables(snapshot.namespace)}
        for result in results:
            if not result.succeeded:
                continue
            if result.table.lower() not in stored:
                return False
            if warehouse.row_count(snapshot.namespace, result.table) != result.rows_loaded:
                return False
        snapshot.results, snapshot.rejection = results, state["rejection"]
        return True

    def sync(self, connection: ConnectionConfig, *, warehouse: Warehouse, namespace: str) -> list[StreamResult]:
        if any(stream.sync_mode not in SNAPSHOT_SYNC_MODES for stream in connection.streams):
            return self._inner.sync(connection, warehouse=warehouse, namespace=namespace)
        snapshot = self._snapshot(connection, warehouse)
        if snapshot.rejection is not None:
            raise ConnectionRejectedError(snapshot.rejection)
        sync_modes = {stream.name: stream.sync_mode for stream in connection.streams}
        results: list[StreamResult] = []
        for cached in snapshot.results or []:
            if not cached.succeeded:
                results.append(replace(cached, connection=connection.name))
                continue
            rows = warehouse.copy_table(
                snapshot.namespace,
                cached.table,
                namespace,
                append=sync_modes[cached.stream] == "full_refresh_append",
            )
            results.append(replace(cached, connection=connection.name, rows_loaded=rows))
        return results

    def close(self) -> None:
        self._inner.close()
