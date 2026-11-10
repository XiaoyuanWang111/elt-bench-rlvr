"""A warehouse that keeps its namespaces lets a later process reuse the reference load and the sync snapshots."""

from support import reference_extract_load

from elt_rlvr.extract_load.base import ExtractLoadEngine
from elt_rlvr.extract_load.cache import CachingEngine
from elt_rlvr.extract_load.local import LocalEngine
from elt_rlvr.grading.reference import TaskReference
from elt_rlvr.pipeline import parse_extract_load
from elt_rlvr.warehouse.duckdb import DuckDBWarehouse
from elt_rlvr.warehouse.state import read_state, write_state


class KeptWarehouse(DuckDBWarehouse):
    """DuckDB standing in for a warehouse whose namespaces outlive one process."""

    persistent = True


class CountingEngine(ExtractLoadEngine):
    """Counts the syncs that reach the real engine."""

    def __init__(self, inner: ExtractLoadEngine) -> None:
        self._inner = inner
        self.syncs = 0

    def source_types(self):
        return self._inner.source_types()

    def sync(self, connection, *, warehouse, namespace):
        self.syncs += 1
        return self._inner.sync(connection, warehouse=warehouse, namespace=namespace)


def _engine(task) -> CountingEngine:
    return CountingEngine(LocalEngine(task.reference_dir.parent / "sources"))


def test_state_record_round_trips_and_is_absent_from_an_unknown_namespace():
    warehouse = KeptWarehouse()
    warehouse.create_namespace("kept")

    write_state(warehouse, "kept", {"note": "it's kept", "rows": 3})

    assert read_state(warehouse, "kept") == {"note": "it's kept", "rows": 3}
    assert read_state(warehouse, "never_created") is None


def test_state_record_keeps_line_breaks_quotes_and_backslashes():
    # An error message of Terraform, as the sync cache stores it for a rejected connection.
    state = {"rejection": 'Error: failure to invoke API\n  on main.tf.json line 43:\n{"status":422} C:\\x'}
    warehouse = KeptWarehouse()
    warehouse.create_namespace("kept")

    write_state(warehouse, "kept", state)
    stored = warehouse.run_query("SELECT state FROM kept.elt_rlvr_state", max_rows=1, timeout_seconds=10).rows[0][0]

    assert read_state(warehouse, "kept") == state
    # Nothing in the stored text that a SQL dialect could read as an escape or a quote.
    assert set(stored) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def test_state_record_that_cannot_be_read_counts_as_absent():
    warehouse = KeptWarehouse()
    warehouse.create_namespace("kept")
    warehouse.create_table_from_query("kept", "elt_rlvr_state", "SELECT '{\"results\": [' AS state", timeout_seconds=10)

    assert read_state(warehouse, "kept") is None


def test_second_process_reuses_the_reference_load_of_a_persistent_warehouse(local_tasks):
    task = local_tasks["storefront"]
    warehouse = KeptWarehouse()
    first, second = _engine(task), _engine(task)

    TaskReference(task, warehouse, first).build()
    later = TaskReference(task, warehouse, second)
    later.build()

    assert first.syncs == 3
    assert second.syncs == 0
    assert later.expected_row_count("customers") == warehouse.row_count("reference_storefront", "customers")
    assert len(later.expected("customers_summary")) == later.expected_row_count("customers")


def test_reference_load_is_rebuilt_when_a_source_table_is_gone(local_tasks):
    task = local_tasks["library"]
    warehouse = KeptWarehouse()
    TaskReference(task, warehouse, _engine(task)).build()
    warehouse.run_query("DROP TABLE reference_library.loans", max_rows=1, timeout_seconds=5)
    engine = _engine(task)

    TaskReference(task, warehouse, engine).build()

    assert engine.syncs == 2
    assert warehouse.row_count("reference_library", "loans") > 0


def test_non_persistent_warehouse_always_loads_the_reference(local_tasks, warehouse):
    task = local_tasks["library"]
    first, second = _engine(task), _engine(task)

    TaskReference(task, warehouse, first).build()
    TaskReference(task, warehouse, second).build()

    assert (first.syncs, second.syncs) == (2, 2)


def test_second_process_serves_a_sync_from_the_snapshot_kept_in_the_warehouse(local_tasks):
    task = local_tasks["library"]
    warehouse = KeptWarehouse()
    connection = parse_extract_load(reference_extract_load(task))[0]
    first, second = _engine(task), _engine(task)
    warehouse.create_namespace("rollout_a")
    warehouse.create_namespace("rollout_b")

    loaded = CachingEngine(first).sync(connection, warehouse=warehouse, namespace="rollout_a")
    served = CachingEngine(second).sync(connection, warehouse=warehouse, namespace="rollout_b")

    assert (first.syncs, second.syncs) == (1, 0)
    assert [(result.table, result.rows_loaded) for result in served] == [
        (result.table, result.rows_loaded) for result in loaded
    ]
    assert warehouse.row_count("rollout_b", "members") == warehouse.row_count("rollout_a", "members")


def test_kept_snapshot_with_missing_rows_is_synced_again(local_tasks):
    task = local_tasks["library"]
    warehouse = KeptWarehouse()
    connection = parse_extract_load(reference_extract_load(task))[0]
    warehouse.create_namespace("rollout_a")
    CachingEngine(_engine(task)).sync(connection, warehouse=warehouse, namespace="rollout_a")
    snapshot = next(name for name in _namespaces(warehouse) if name.startswith("sync_cache_"))
    warehouse.run_query(f"DELETE FROM {snapshot}.members WHERE member_id = 1", max_rows=1, timeout_seconds=5)
    engine = _engine(task)

    CachingEngine(engine).sync(connection, warehouse=warehouse, namespace="rollout_a")

    assert engine.syncs == 1


def _namespaces(warehouse: DuckDBWarehouse) -> list[str]:
    result = warehouse.run_query("SELECT schema_name FROM duckdb_schemas()", max_rows=1000, timeout_seconds=5)
    return [row[0] for row in result.rows]
