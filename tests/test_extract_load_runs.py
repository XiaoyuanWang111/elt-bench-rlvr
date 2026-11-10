"""Running some or all connections of the extract-load file, and what a repeated run does to the loaded rows."""

from support import edit_extract_load, grade, open_session, reference_models, write_pipeline


def _append_everywhere(document):
    for connection in document["connections"]:
        for stream in connection["streams"]:
            stream["sync_mode"] = "full_refresh_append"


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


def test_retrying_only_the_failed_connection_keeps_append_loads_single(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)

    def break_loans(document):
        _append_everywhere(document)
        document["connections"][1]["source"]["configuration"]["url"] = "missing.csv"

    write_pipeline(session, edit_extract_load(task, break_loans), reference_models(task))
    first = session.run_extract_load()
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})
    session.run_extract_load(["loans_file"])
    session.run_transform()

    assert [entry["status"] for entry in first["connections"]] == ["succeeded", "failed"]
    assert grade(session).reward == 1.0


def test_rerunning_every_connection_after_a_failure_duplicates_the_ones_that_had_succeeded(
    local_tasks, warehouse, workspace
):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)

    def break_loans(document):
        _append_everywhere(document)
        document["connections"][1]["source"]["configuration"]["url"] = "missing.csv"

    write_pipeline(session, edit_extract_load(task, break_loans), reference_models(task))
    session.run_extract_load()
    write_pipeline(session, edit_extract_load(task, _append_everywhere), {})
    session.run_extract_load()

    result = grade(session)

    assert result.source_tables == {"members": False, "loans": True}
