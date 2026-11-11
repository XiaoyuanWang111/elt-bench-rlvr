"""The sync cache serves a repeated connection from a snapshot and reproduces what the wrapped engine would load."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from elt_rlvr.extract_load.base import ConnectionRejectedError, ExtractLoadEngine, StreamResult
from elt_rlvr.extract_load.cache import CachingEngine
from elt_rlvr.pipeline import ConnectionConfig, StreamConfig

FIRST, SECOND = "rollout_first", "rollout_second"
OUTAGE = "sync job ended with status failed"


class CountingEngine(ExtractLoadEngine):
    """Loads ``rows`` rows per stream, rejects a configuration with a ``reject`` field, and counts its syncs.

    A stream named ``missing`` fails, as a stream absent from a real source does. The next ``outages`` syncs fail
    on every stream, as a sync job does that the deployment could not finish, and ``hold`` keeps a sync from
    returning until it is set.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.outages = 0
        self.variant = ""
        self.entered = threading.Event()
        self.hold: threading.Event | None = None

    def source_types(self) -> tuple[str, ...]:
        return ("counting",)

    def sync_variant(self, connection) -> str:
        return self.variant

    def sync(self, connection, *, warehouse, namespace) -> list[StreamResult]:
        self.calls += 1
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(10)
        if self.outages:
            self.outages -= 1
            return [
                StreamResult(connection.name, stream.name, stream.name, succeeded=False, error=OUTAGE)
                for stream in connection.streams
            ]
        if "reject" in connection.configuration:
            raise ConnectionRejectedError(connection.configuration["reject"])
        results = []
        for stream in connection.streams:
            if stream.name == "missing":
                results.append(
                    StreamResult(connection.name, stream.name, stream.name, succeeded=False, error="no such stream")
                )
                continue
            frame = pd.DataFrame({"id": range(connection.configuration["rows"]), "stream": stream.name})
            append = stream.sync_mode != "full_refresh_overwrite"
            rows = warehouse.write_frame(namespace, stream.name, frame, append=append)
            results.append(StreamResult(connection.name, stream.name, stream.name, succeeded=True, rows_loaded=rows))
        return results


@pytest.fixture
def inner() -> CountingEngine:
    return CountingEngine()


@pytest.fixture
def cache(inner: CountingEngine) -> CachingEngine:
    return CachingEngine(inner)


@pytest.fixture
def rollouts(warehouse):
    warehouse.create_namespace(FIRST)
    warehouse.create_namespace(SECOND)
    return warehouse


def _connection(
    rows: int = 3,
    streams: tuple[str, ...] = ("orders", "customers"),
    sync_mode: str = "full_refresh_overwrite",
    name: str = "shop",
    **configuration,
) -> ConnectionConfig:
    return ConnectionConfig(
        name, "counting", {"rows": rows, **configuration}, tuple(StreamConfig(stream, sync_mode) for stream in streams)
    )


def _table(warehouse, namespace: str, table: str) -> list[dict]:
    return warehouse.read_frame(namespace, table).sort_values("id").to_dict("records")


def test_cache_repeated_connection_is_served_without_syncing_again(cache, inner, rollouts):
    first = cache.sync(_connection(), warehouse=rollouts, namespace=FIRST)
    second = cache.sync(_connection(), warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 1
    assert second == first
    assert rollouts.list_tables(SECOND) == rollouts.list_tables(FIRST) == ["customers", "orders"]
    for table in ("customers", "orders"):
        assert _table(rollouts, SECOND, table) == _table(rollouts, FIRST, table)


def test_cache_first_sync_loads_what_the_wrapped_engine_loads(cache, rollouts):
    results = cache.sync(_connection(rows=4), warehouse=rollouts, namespace=FIRST)

    assert [(result.table, result.rows_loaded, result.succeeded) for result in results] == [
        ("orders", 4, True),
        ("customers", 4, True),
    ]
    assert rollouts.row_count(FIRST, "orders") == 4


def test_cache_append_run_twice_doubles_the_rows(cache, inner, rollouts):
    connection = _connection(sync_mode="full_refresh_append")

    cache.sync(connection, warehouse=rollouts, namespace=FIRST)
    second = cache.sync(connection, warehouse=rollouts, namespace=FIRST)

    assert inner.calls == 1
    assert [result.rows_loaded for result in second] == [3, 3]
    assert rollouts.row_count(FIRST, "orders") == 6


def test_cache_overwrite_after_append_resets_the_table(cache, rollouts):
    cache.sync(_connection(sync_mode="full_refresh_append"), warehouse=rollouts, namespace=FIRST)
    cache.sync(_connection(sync_mode="full_refresh_append"), warehouse=rollouts, namespace=FIRST)
    cache.sync(_connection(), warehouse=rollouts, namespace=FIRST)

    assert rollouts.row_count(FIRST, "orders") == 3


@pytest.mark.parametrize(
    "other",
    [_connection(rows=5), _connection(streams=("orders",)), _connection(region="eu")],
    ids=["configuration_value", "stream_set", "extra_field"],
)
def test_cache_different_configuration_is_synced_separately(other, cache, inner, rollouts):
    cache.sync(_connection(), warehouse=rollouts, namespace=FIRST)
    results = cache.sync(other, warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 2
    assert [result.rows_loaded for result in results] == [other.configuration["rows"]] * len(other.streams)


def test_cache_same_configuration_under_another_name_reports_its_own_name(cache, inner, rollouts):
    cache.sync(_connection(name="shop"), warehouse=rollouts, namespace=FIRST)
    results = cache.sync(_connection(name="store"), warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 1
    assert {result.connection for result in results} == {"store"}


def test_cache_rejected_configuration_is_not_kept(cache, inner, rollouts):
    connection = _connection(reject="host 'db' is not a source of this task")

    with pytest.raises(ConnectionRejectedError, match="host 'db' is not a source of this task"):
        cache.sync(connection, warehouse=rollouts, namespace=FIRST)
    with pytest.raises(ConnectionRejectedError, match="host 'db' is not a source of this task"):
        cache.sync(connection, warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 2
    assert rollouts.list_tables(SECOND) == []


def test_cache_sync_with_a_failed_stream_is_not_kept(cache, inner, rollouts):
    connection = _connection(streams=("orders", "missing"))

    first = cache.sync(connection, warehouse=rollouts, namespace=FIRST)
    second = cache.sync(connection, warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 2
    assert second == first
    assert second[1] == StreamResult("shop", "missing", "missing", succeeded=False, error="no such stream")
    assert rollouts.list_tables(SECOND) == ["orders"]


def test_cache_sync_that_failed_once_succeeds_for_the_next_request(cache, inner, rollouts):
    inner.outages = 1

    failed = cache.sync(_connection(), warehouse=rollouts, namespace=FIRST)
    loaded = cache.sync(_connection(), warehouse=rollouts, namespace=FIRST)
    served = cache.sync(_connection(), warehouse=rollouts, namespace=SECOND)

    assert [(result.succeeded, result.error) for result in failed] == [(False, OUTAGE), (False, OUTAGE)]
    assert [(result.succeeded, result.rows_loaded) for result in loaded] == [(True, 3), (True, 3)]
    assert served == loaded
    assert inner.calls == 2


def test_cache_requests_waiting_for_a_sync_that_fails_get_its_outcome(cache, inner, rollouts):
    inner.outages = 1
    inner.hold = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(cache.sync, _connection(), warehouse=rollouts, namespace=FIRST)
        assert inner.entered.wait(10)
        waiting = pool.submit(cache.sync, _connection(), warehouse=rollouts, namespace=SECOND)
        time.sleep(0.5)  # long enough for the second request to arrive and wait for the first
        inner.hold.set()
        results = running.result(10), waiting.result(10)

    assert inner.calls == 1
    assert all(not result.succeeded for result in results[0] + results[1])
    inner.hold = None
    assert all(result.succeeded for result in cache.sync(_connection(), warehouse=rollouts, namespace=FIRST))
    assert inner.calls == 2


def test_cache_keeps_apart_what_the_engine_says_differs(cache, inner, rollouts):
    cache.sync(_connection(), warehouse=rollouts, namespace=FIRST)
    inner.variant = "another connector definition"
    cache.sync(_connection(), warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 2


@pytest.mark.parametrize("sync_mode", ["incremental_append", "incremental_deduped_history"])
def test_cache_incremental_stream_bypasses_the_cache(sync_mode, cache, inner, rollouts):
    connection = _connection(streams=("orders",), sync_mode=sync_mode)

    cache.sync(connection, warehouse=rollouts, namespace=FIRST)
    cache.sync(connection, warehouse=rollouts, namespace=FIRST)

    assert inner.calls == 2
    assert rollouts.row_count(FIRST, "orders") == 6


def test_cache_connection_with_one_incremental_stream_bypasses_the_cache(cache, inner, rollouts):
    connection = ConnectionConfig(
        "shop",
        "counting",
        {"rows": 3},
        (StreamConfig("orders", "full_refresh_overwrite"), StreamConfig("customers", "incremental_append")),
    )

    cache.sync(connection, warehouse=rollouts, namespace=FIRST)
    cache.sync(connection, warehouse=rollouts, namespace=SECOND)

    assert inner.calls == 2
