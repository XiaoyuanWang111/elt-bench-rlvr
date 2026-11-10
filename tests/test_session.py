"""Rollout session tools: file listing and windowed reads, model file edits, run errors, queries, submit and close."""

import json

import pytest
from support import open_session, reference_extract_load, reference_models, solve, write_pipeline

from elt_rlvr.pipeline import EXTRACT_LOAD_PATH, MODELS_DIRECTORY

NOTES = f"{MODELS_DIRECTORY}/notes.sql"


@pytest.fixture
def session(local_tasks, warehouse, workspace):
    return open_session(local_tasks["library"], warehouse, workspace)


def _write_lines(session, count: int) -> None:
    assert "error" not in session.write_file(NOTES, "\n".join(f"-- line {index}" for index in range(1, count + 1)))


# ---- files -------------------------------------------------------------------------------------------------------


def test_list_files_shows_the_task_files_and_the_pipeline_directory(session):
    entries = session.list_files()["entries"]

    assert [entry["path"] for entry in entries] == [
        "config.yaml",
        "data_model.yaml",
        "documentation/",
        "pipeline/",
        "schemas/",
    ]
    assert entries[0]["bytes"] == (session.workspace / "config.yaml").stat().st_size


def test_list_files_of_a_subdirectory_gives_workspace_relative_paths(session):
    entries = session.list_files("schemas")["entries"]

    assert [entry["path"] for entry in entries] == ["schemas/loans.csv", "schemas/members.csv"]


def test_list_files_shows_a_written_model_file(session):
    write_pipeline(session, None, {"member_activity": "select 1 as x"})

    entries = session.list_files(MODELS_DIRECTORY)["entries"]

    assert entries == [{"path": f"{MODELS_DIRECTORY}/member_activity.sql", "bytes": len("select 1 as x")}]


def test_list_files_of_a_file_is_an_error(session):
    assert "is not a directory" in session.list_files("config.yaml")["error"]


def test_read_file_returns_the_requested_window_and_the_total_line_count(session):
    _write_lines(session, 250)

    result = session.read_file(NOTES, start_line=10, max_lines=3)

    assert result == {
        "path": NOTES,
        "start_line": 10,
        "end_line": 12,
        "total_lines": 250,
        "text": "-- line 10\n-- line 11\n-- line 12",
    }


@pytest.mark.parametrize("max_lines", [None, 1000])
def test_read_file_returns_at_most_the_line_limit(max_lines, session):
    _write_lines(session, 250)

    result = session.read_file(NOTES, max_lines=max_lines)

    assert (result["start_line"], result["end_line"], result["total_lines"]) == (1, 200, 250)
    assert result["text"].splitlines()[-1] == "-- line 200"


def test_read_file_start_line_below_one_starts_at_the_first_line(session):
    _write_lines(session, 5)

    result = session.read_file(NOTES, start_line=0, max_lines=2)

    assert (result["start_line"], result["end_line"], result["text"]) == (1, 2, "-- line 1\n-- line 2")


def test_read_file_last_window_ends_at_the_last_line(session):
    _write_lines(session, 5)

    result = session.read_file(NOTES, start_line=4, max_lines=10)

    assert (result["end_line"], result["text"]) == (5, "-- line 4\n-- line 5")


def test_read_file_long_line_is_cut_at_the_character_limit(session):
    assert "error" not in session.write_file(NOTES, "x" * 13_000 + "\nsecond line")

    result = session.read_file(NOTES)

    assert len(result["text"]) == session.limits.max_read_chars
    assert (result["end_line"], result["total_lines"]) == (1, 2)


def test_read_file_missing_file_is_an_error(session):
    assert "is not a file" in session.read_file("schemas/payments.csv")["error"]


def test_read_file_of_a_directory_is_an_error(session):
    assert "is not a file" in session.read_file("schemas")["error"]


@pytest.mark.parametrize("empty", ["", "  \n\t"])
def test_write_file_with_empty_content_deletes_the_model_file(empty, session):
    written = session.write_file(NOTES, "select 1 as x")
    assert written == {"path": NOTES, "bytes": len("select 1 as x")}
    assert (session.workspace / NOTES).is_file()

    deleted = session.write_file(NOTES, empty)

    assert deleted == {"path": NOTES, "deleted": True}
    assert not (session.workspace / NOTES).exists()
    assert session.list_files(MODELS_DIRECTORY)["entries"] == []


def test_write_file_deleting_a_missing_model_file_is_not_an_error(session):
    assert session.write_file(NOTES, "") == {"path": NOTES, "deleted": True}


# ---- runs --------------------------------------------------------------------------------------------------------


def test_run_transform_with_an_unknown_model_name_is_an_error(session):
    write_pipeline(session, None, {"member_activity": "select 1 as x"})

    result = session.run_transform(["member_activity", "loan_summary"])

    assert result == {"error": "no model file for: loan_summary"}


def test_run_transform_without_model_files_is_an_error(session):
    assert session.run_transform() == {"error": f"there are no model files in {MODELS_DIRECTORY}/"}


def test_run_extract_load_without_the_config_file_is_an_error(session):
    assert session.run_extract_load() == {"error": f"{EXTRACT_LOAD_PATH} does not exist"}


def test_run_extract_load_with_a_malformed_config_file_is_an_error(session):
    write_pipeline(session, "connections: [", {})

    assert "is not valid YAML" in session.run_extract_load()["error"]


def test_run_extract_load_reports_an_unreadable_source_as_a_failed_connection(session):
    configuration = {"dataset_name": "members", "format": "jsonl", "url": "members.csv"}
    config = {
        "connections": [
            {
                "name": "members_file",
                "source": {"type": "file", "configuration": configuration},
                "streams": [{"name": "members", "sync_mode": "full_refresh_overwrite"}],
            }
        ]
    }
    write_pipeline(session, json.dumps(config), {})

    report = session.run_extract_load()

    assert report["connections"][0]["status"] == "failed"


# ---- queries -----------------------------------------------------------------------------------------------------


def test_query_reads_a_model_only_after_run_transform_built_it(session):
    write_pipeline(session, reference_extract_load(session.task), reference_models(session.task))
    session.run_extract_load()
    before = session.query("select count(*) from member_activity")

    session.run_transform()
    after = session.query("select count(*) as members from member_activity")

    assert "unknown table 'member_activity'" in before["error"]
    assert after["columns"] == ["members"]
    assert after["rows"] == session.query("select count(*) from members")["rows"]
    assert after["rows"][0][0] > 0


def test_query_after_a_failed_rebuild_no_longer_reads_the_model(session):
    solve(session)
    write_pipeline(session, None, {"member_activity": "select * from payments"})

    session.run_transform()

    assert "unknown table 'member_activity'" in session.query("select * from member_activity")["error"]


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("cast(1.50 as decimal(5, 2))", "1.50"),
        ("date '2020-01-02'", "2020-01-02"),
        ("cast('nan' as double)", "nan"),
        ("repeat('x', 500)", "x" * 200 + "..."),
        ("cast(null as integer)", None),
        ("42", 42),
    ],
)
def test_query_returns_json_safe_cells(expression, value, session):
    result = session.query(f"select {expression} as cell")

    assert result["rows"] == [[value]]
    json.dumps(result, allow_nan=False)


# ---- end of the episode ------------------------------------------------------------------------------------------


def test_submit_marks_the_session_submitted(session):
    assert not session.submitted

    assert session.submit() == {"status": "submitted"}
    assert session.submitted


def test_close_removes_the_workspace_and_the_namespace_tables(session, warehouse):
    solve(session)
    assert warehouse.list_tables(session.namespace)

    session.close()

    assert not session.workspace.exists()
    assert warehouse.list_tables(session.namespace) == []
