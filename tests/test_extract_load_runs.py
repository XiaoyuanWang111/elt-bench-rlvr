"""What `run_extract_load` syncs: the whole file, only what is new, or the named connections again."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from support import edit_extract_load, grade, open_session, reference_models, solve, write_pipeline

from elt_rlvr.extract_load.base import ConnectionRejectedError, ExtractLoadEngine
from elt_rlvr.session import UNCHANGED_NOTE


def _append_everywhere(document):
    for connection in document["connections"]:
        for stream in connection["streams"]:
            stream["sync_mode"] = "full_refresh_append"


def _break_loans(document):
    _append_everywhere(document)
    document["connections"][1]["source"]["configuration"]["url"] = "missing.csv"


def test_named_connection_runs_alone(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})

    report = session.run_extract_load(["loans_file"])

    assert [entry["connection"] for entry in report["connections"]] == ["loans_file"]
    assert list(report["source_tables"]) == ["loans"]


def test_unknown_connection_name_is_reported_with_the_names_in_the_file(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})

    report = session.run_extract_load(["orders_file"])

    assert "no connection named: orders_file" in report["error"]
    assert "members_file, loans_file" in report["error"]
    assert session.raw_tables == set()


def test_applying_the_file_again_after_a_failure_leaves_loaded_streams_alone(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, _break_loans), reference_models(task))
    first = session.run_extract_load()
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})

    second = session.run_extract_load()
    session.run_transform()

    assert [entry["status"] for entry in first["connections"]] == ["succeeded", "failed"]
    assert second["connections"][0] == {"connection": "members_file", "status": "unchanged", "note": UNCHANGED_NOTE}
    assert second["connections"][1]["status"] == "succeeded"
    assert session.connection_syncs == 3
    assert grade(session).reward == 1.0


def test_applying_an_unchanged_file_again_syncs_nothing(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})
    first = session.run_extract_load()

    second = session.run_extract_load()

    assert [entry["status"] for entry in second["connections"]] == ["unchanged", "unchanged"]
    assert second["source_tables"] == first["source_tables"]
    assert session.connection_syncs == 2


def test_naming_a_loaded_connection_appends_its_rows_again(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, _append_everywhere), reference_models(task))
    first = session.run_extract_load()["source_tables"]

    second = session.run_extract_load(["members_file"])["source_tables"]

    assert second == {"loans": first["loans"], "members": 2 * first["members"]}
    assert grade(session).source_tables == {"members": False, "loans": True}


def test_changed_source_configuration_is_synced_again(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, lambda document: None), {})
    session.run_extract_load()

    def overwrite_members(document):
        document["connections"][0]["streams"][0]["sync_mode"] = "full_refresh_overwrite"
        document["connections"][0]["source"]["configuration"]["reader_options"] = {"sep": ","}

    write_pipeline(session, edit_extract_load(task, overwrite_members), {})
    report = session.run_extract_load()

    assert [entry["status"] for entry in report["connections"]] == ["succeeded", "unchanged"]


def test_stream_added_to_a_loaded_connection_is_the_only_one_synced(local_tasks, warehouse, workspace):
    task = local_tasks["storefront"]
    session = open_session(task, warehouse, workspace)

    def one_stream(document):
        _append_everywhere(document)
        database = next(c for c in document["connections"] if len(c["streams"]) > 1)
        database["streams"] = database["streams"][:1]

    write_pipeline(session, edit_extract_load(task, one_stream), reference_models(task))
    session.run_extract_load()
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})

    report = session.run_extract_load()
    session.run_transform()

    statuses = [stream["status"] for entry in report["connections"] for stream in entry.get("streams", [])]
    assert sorted(statuses) == ["succeeded", "unchanged"]
    assert grade(session).reward == 1.0


def test_sync_budget_counts_connections_not_calls(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    write_pipeline(session, edit_extract_load(task, lambda document: None), {})
    for _ in range(7):
        assert "error" not in session.run_extract_load(["loans_file"])

    refused = session.run_extract_load(["loans_file", "members_file"])
    last = session.run_extract_load()

    assert "7 are used and this call would run 2" in refused["error"]
    assert [entry["status"] for entry in last["connections"]] == ["succeeded", "unchanged"]
    assert session.connection_syncs == 8


class _FaultyEngine(ExtractLoadEngine):
    """Fails one connection the way a broken environment does; passes the others to the real engine."""

    def __init__(self, inner: ExtractLoadEngine, faulty: str, error: Exception) -> None:
        self._inner, self._faulty, self._error = inner, faulty, error

    def source_types(self):
        return self._inner.source_types()

    def sync(self, connection, *, warehouse, namespace):
        if connection.name == self._faulty:
            raise self._error
        return self._inner.sync(connection, warehouse=warehouse, namespace=namespace)


def test_environment_fault_in_one_connection_does_not_hide_the_others(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    session.engine = _FaultyEngine(session.engine, "members_file", RuntimeError("airbyte is not reachable"))
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})

    report = session.run_extract_load()

    assert report["connections"][0]["status"] == "failed"
    assert "the environment could not run this connection (RuntimeError: airbyte" in report["connections"][0]["error"]
    assert report["connections"][1]["status"] == "succeeded"
    assert list(report["source_tables"]) == ["loans"]
    assert session.environment_errors == 1


def test_rejected_configuration_is_not_counted_as_an_environment_fault(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    session.engine = _FaultyEngine(session.engine, "members_file", ConnectionRejectedError("no such host"))
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})

    report = session.run_extract_load()

    assert report["connections"][0] == {"connection": "members_file", "status": "failed", "error": "no such host"}
    assert session.environment_errors == 0


def test_grading_waits_for_a_tool_call_that_is_still_running(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    entered, release = threading.Event(), threading.Event()
    inner = session.engine

    class SlowEngine(ExtractLoadEngine):
        def sync(self, connection, *, warehouse, namespace):
            entered.set()
            assert release.wait(10)
            return inner.sync(connection, warehouse=warehouse, namespace=namespace)

    session.engine = SlowEngine()

    # The rollout's time limit cancels the awaiting task, not the worker thread that runs the sync.
    with ThreadPoolExecutor(max_workers=2) as pool:
        call = pool.submit(session.run_extract_load, ["members_file"])
        assert entered.wait(10)
        graded = pool.submit(grade, session)
        time.sleep(0.5)
        waited = not graded.done()
        release.set()
        call.result(10)
        result = graded.result(30)

    assert waited
    assert result.reward == 1.0
