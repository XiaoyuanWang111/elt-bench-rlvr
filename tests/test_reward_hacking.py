"""Each test is one way to earn reward without building the pipeline, and shows that the environment refuses it."""

import json

import pandas as pd
import pytest
from support import (
    edit_extract_load,
    grade,
    open_session,
    reference_extract_load,
    reference_models,
    solve,
    write_pipeline,
)

from elt_rlvr.grading.compare import OFFICIAL, STRICT, compare_tables
from elt_rlvr.grading.grader import RewardConfig, grade_group
from elt_rlvr.grading.reference import TaskReference
from elt_rlvr.pipeline import EXTRACT_LOAD_PATH
from elt_rlvr.session import SessionLimits
from elt_rlvr.warehouse import WarehouseError


def _literal_model(session, model: str) -> str:
    """A model that returns the current correct rows of ``model`` as constants."""
    result = session.query(f"select * from {model}")
    assert not result["truncated"]
    rows = ", ".join(
        "(" + ", ".join(json.dumps(value).replace('"', "'") for value in row) + ")" for row in result["rows"]
    )
    return f"select * from (values {rows}) as constants({', '.join(result['columns'])})"


# ---- answers that are not computed from the data -----------------------------------------------------------------


def test_hard_coded_answer_table_earns_no_transform_credit(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    write_pipeline(session, None, {"member_activity": _literal_model(session, "member_activity")})
    session.run_transform()

    result = grade(session)

    # The constants reproduce the expected table on the data the policy saw, and nothing on held-out data.
    assert result.transform_score_original == 1.0
    assert result.transform_score == 0.0
    assert result.reward == pytest.approx(RewardConfig().extract_load_weight)


def test_hard_coded_answers_pass_when_held_out_grading_is_off(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    write_pipeline(session, None, {"member_activity": _literal_model(session, "member_activity")})

    assert grade(session, RewardConfig(heldout_samples=0)).reward == 1.0


def test_value_looked_up_once_and_pasted_into_sql_loses_its_column(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    overdue = session.query("select member_id from member_activity where has_overdue_loan = 1")
    pasted_ids = ", ".join(str(row[0]) for row in overdue["rows"])
    models = reference_models(task)
    models["member_activity"] = models["member_activity"].replace(
        "coalesce(max(case when l.days_late > 14 then 1 else 0 end), 0)",
        f"case when m.member_id in ({pasted_ids}) then 1 else 0 end",
    )
    write_pipeline(session, None, models)

    # A pasted flag stays right on a sample only if no flagged member lost every overdue loan; with six samples
    # that cannot plausibly happen on all of them.
    result = grade(session, RewardConfig(heldout_samples=6))

    assert result.transform_score_original == 1.0
    assert result.model_columns["member_activity"]["has_overdue_loan"] is False
    assert result.model_columns["member_activity"]["num_loans"] is True


def test_model_files_are_graded_not_the_tables_left_in_the_warehouse(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    # The correct table from the earlier run stays in the warehouse; the submitted file no longer builds it.
    write_pipeline(session, None, {"member_activity": "select member_id, name from members"})

    result = grade(session)

    assert "member_activity" in session.model_tables
    assert result.transform_score == pytest.approx(2 / 5)


def test_failed_held_out_draw_fails_the_whole_group(local_tasks, warehouse, workspace, monkeypatch):
    task = local_tasks["library"]
    sessions = []
    for name in ("constants", "more_constants", "loads_only"):
        session = open_session(task, warehouse, workspace / name)
        solve(session)
        write_pipeline(session, None, {"member_activity": _literal_model(session, "member_activity")})
        sessions.append(session)
    # The third rollout has no column right on the original data, so it never asks for the samples.
    sessions[2].write_file("pipeline/models/member_activity.sql", "")
    reference = TaskReference(task, warehouse, sessions[0].engine)

    def failing_draw(seed, sample_percent, **options):
        raise WarehouseError("the warehouse did not answer")

    monkeypatch.setattr(reference, "draw_heldout", failing_draw)

    # One thread, so that the second rollout asks for the samples after the draw has failed for the first.
    grades = grade_group(sessions, reference, RewardConfig(grading_threads=1), seed=7)

    # No rollout may earn more than another because it did not need what the environment failed to provide.
    assert [grade.reward for grade in grades] == [0.0, 0.0, 0.0]
    assert all("held-out samples could not be drawn" in grade.grader_error for grade in grades)


def test_value_of_one_row_pasted_per_key_loses_its_column(local_tasks, warehouse, workspace):
    task = local_tasks["clinic"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    names = session.query("select doctor_id, full_name from doctors_summary")
    assert not names["truncated"]
    cases = " ".join(f"when {doctor_id} then '{full_name}'" for doctor_id, full_name in names["rows"])
    models = reference_models(task)
    expression = "d.first_name || ' ' || d.last_name"
    assert expression in models["doctors_summary"]
    models["doctors_summary"] = models["doctors_summary"].replace(expression, f"case d.doctor_id {cases} end")
    write_pipeline(session, None, models)

    result = grade(session)

    # A sample that only drops rows leaves every pasted name right. A held-out sample also gives each doctor_id
    # to another doctor, so the name looked up for an id is the name of someone else.
    assert result.transform_score_original == 1.0
    assert result.model_columns["doctors_summary"]["full_name"] is False
    assert result.model_columns["doctors_summary"]["num_visits"] is True
    assert grade(session, RewardConfig(heldout_edge_cases=False)).model_columns["doctors_summary"]["full_name"] is True


def _reference_of(task, warehouse, workspace) -> TaskReference:
    reference = TaskReference(task, warehouse, open_session(task, warehouse, workspace).engine)
    reference.build()
    return reference


def _rows(warehouse, sql: str) -> list:
    return warehouse.run_query(sql, max_rows=10_000, timeout_seconds=10).rows


def test_held_out_sample_gives_every_parent_key_to_another_row(local_tasks, warehouse, workspace):
    reference = _reference_of(local_tasks["library"], warehouse, workspace)

    heldout = reference.draw_heldout(3, sample_percent=100.0)

    same_member = (
        f"select count(*) from {heldout.namespace}.members h join {reference.namespace}.members r "
        "on r.member_id = h.member_id and r.name = h.name"
    )
    assert _rows(warehouse, same_member) == [(0,)]
    loans_by_name = (
        "select m.name, count(l.loan_id) from {0}.members m left join {0}.loans l on l.member_id = m.member_id "
        "group by m.name"
    )
    original = dict(_rows(warehouse, loans_by_name.format(reference.namespace)))
    sampled = dict(_rows(warehouse, loans_by_name.format(heldout.namespace)))
    # The loans followed their member to the new key; only the member chosen to lose its loans differs.
    changed = [name for name in original if original[name] != sampled[name]]
    assert len(changed) == 1 and sampled[changed[0]] == 0


def test_sample_that_cannot_leave_a_parent_childless_is_drawn_again(local_tasks, warehouse, workspace, monkeypatch):
    reference = _reference_of(local_tasks["library"], warehouse, workspace)
    delete = warehouse.delete_children_of_one_parent
    calls = []

    def nothing_the_first_time(*arguments, **options):
        calls.append(arguments)
        if len(calls) > 1:
            delete(*arguments, **options)

    monkeypatch.setattr(warehouse, "delete_children_of_one_parent", nothing_the_first_time)

    heldout = reference.draw_heldout(3, sample_percent=100.0)

    borrowers = "select count(distinct member_id) from {}.loans"
    assert len(calls) == 2
    assert _rows(warehouse, borrowers.format(heldout.namespace))[0][0] == (
        _rows(warehouse, borrowers.format(reference.namespace))[0][0] - 1
    )


def test_two_references_of_one_task_draw_their_samples_into_different_namespaces(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    # Two processes that grade the same task in one warehouse each have their own reference object.
    first = _reference_of(task, warehouse, workspace / "first")
    second = _reference_of(task, warehouse, workspace / "second")

    assert first.draw_heldout(1, 70.0).namespace != second.draw_heldout(1, 70.0).namespace


def test_grader_builds_only_the_required_models(local_tasks, warehouse, workspace, monkeypatch):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    write_pipeline(session, None, {"scratch": "select * from members", "scratch_too": "select * from scratch"})
    built = set()
    create = warehouse.create_table_from_query

    def recording_create(namespace, name, sql, **options):
        if namespace.startswith("grade_"):
            built.add(name)
        return create(namespace, name, sql, **options)

    monkeypatch.setattr(warehouse, "create_table_from_query", recording_create)

    result = grade(session)

    assert result.reward == 1.0
    assert built == {"member_activity"}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_held_out_sample_has_a_parent_that_lost_its_children(seed, local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    reference = TaskReference(task, warehouse, open_session(task, warehouse, workspace).engine)
    reference.build()

    def borrowers(namespace: str) -> int:
        sql = f"select count(distinct member_id) from {namespace}.loans"
        return warehouse.run_query(sql, max_rows=1, timeout_seconds=10).rows[0][0]

    # With every row kept, the only difference from the original data is the member whose loans are removed.
    heldout = reference.draw_heldout(seed, sample_percent=100.0)

    assert borrowers(heldout.namespace) == borrowers(reference.namespace) - 1
    assert warehouse.row_count(heldout.namespace, "members") == warehouse.row_count(reference.namespace, "members")


def test_model_table_of_the_wrong_size_is_never_fetched(local_tasks, warehouse, workspace, monkeypatch):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    write_pipeline(session, None, {"member_activity": "select a.* from members a cross join members b"})
    fetched = []
    read_frame = warehouse.read_frame

    def recording_read_frame(namespace, table):
        fetched.append(namespace)
        return read_frame(namespace, table)

    monkeypatch.setattr(warehouse, "read_frame", recording_read_frame)

    result = grade(session)

    assert result.transform_score == 0.0
    assert not [namespace for namespace in fetched if namespace.startswith("grade_")]


# ---- raw tables that did not come from the sources -----------------------------------------------------------------


def test_table_loaded_by_a_configuration_that_left_the_file_does_not_count(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)

    def wrong_separator(document):
        members = next(c for c in document["connections"] if c["name"] == "members_file")
        members["source"]["configuration"]["reader_options"] = '{"sep": ";"}'

    def statuses(report):
        return [connection["status"] for connection in report["connections"]]

    wrong = edit_extract_load(task, wrong_separator)
    write_pipeline(session, wrong, {})
    assert statuses(session.run_extract_load()) == ["succeeded", "succeeded"]
    write_pipeline(session, reference_extract_load(task), reference_models(task))
    assert statuses(session.run_extract_load()) == ["succeeded", "unchanged"]
    assert grade(session).source_tables == {"members": True, "loans": True}
    # The file goes back to the configuration that loads one meaningless column. The table still holds the rows
    # that the right configuration loaded.
    write_pipeline(session, wrong, {})

    result = grade(session)
    report = session.run_extract_load()

    assert result.source_tables == {"members": False, "loans": True}
    # Applying the file loads what the file says: the table is not "already loaded with this configuration".
    assert statuses(report) == ["succeeded", "unchanged"]
    assert grade(session).source_tables == {"members": False, "loans": True}


@pytest.mark.parametrize("change", ["remove_connection", "change_sync_mode", "delete_file"])
def test_source_table_counts_only_if_the_submitted_file_is_what_loaded_it(change, local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)

    def edit(document):
        if change == "remove_connection":
            document["connections"] = [c for c in document["connections"] if c["name"] == "members_file"]
        else:
            for connection in document["connections"]:
                if connection["name"] != "members_file":
                    connection["streams"][0]["sync_mode"] = "full_refresh_append"

    # The tables stay in the warehouse as they were loaded; only the file changes.
    session.write_file(EXTRACT_LOAD_PATH, "" if change == "delete_file" else edit_extract_load(task, edit))

    result = grade(session)

    assert result.official_extract_load
    assert result.source_tables == {"members": change != "delete_file", "loans": False}


def test_fabricated_source_table_with_the_right_row_count_is_not_a_load(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    reference = TaskReference(task, warehouse, open_session(task, warehouse, workspace / "probe").engine)
    reference.build()
    loan_count = reference.expected_row_count("loans")
    session = open_session(task, warehouse, workspace / "attack")

    def keep_members(document):
        document["connections"] = [c for c in document["connections"] if c["name"] == "members_file"]

    fabricated = f"select i as loan_id, 1 as member_id, 'x' as book_title, 0 as days_late from range({loan_count}) t(i)"
    write_pipeline(session, edit_extract_load(task, keep_members), {"loans": fabricated})
    session.run_extract_load()
    assert session.run_transform()["models"][0]["status"] == "succeeded"

    result = grade(session)

    # A row-count check, as in ELT-Bench stage 1, accepts the fabricated table.
    assert result.official_extract_load
    assert result.source_tables == {"members": True, "loans": False}


def test_appending_twice_duplicates_rows_and_fails_the_load_check(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)

    def append(document):
        for connection in document["connections"]:
            for stream in connection["streams"]:
                stream["sync_mode"] = "full_refresh_append"

    write_pipeline(session, edit_extract_load(task, append), reference_models(task))
    first = session.run_extract_load()["source_tables"]
    second = session.run_extract_load(["members_file", "loans_file"])["source_tables"]

    assert second == {table: 2 * rows for table, rows in first.items()}
    assert grade(session).extract_load_score == 0.0


@pytest.mark.parametrize(
    "configuration",
    [
        {"dataset_name": "leak", "format": "csv", "url": "../grader/task.yaml"},
        {"dataset_name": "leak", "format": "csv", "url": "../workspace/schemas/loans.csv"},
        {"dataset_name": "leak", "format": "csv", "url": "https://example.com/ground_truth.csv"},
        {"dataset_name": "leak", "format": "csv", "url": "file:///etc/passwd"},
    ],
)
def test_file_source_cannot_load_a_file_that_is_not_a_task_source(configuration, local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    config = {
        "connections": [
            {
                "name": "leak",
                "source": {"type": "file", "configuration": configuration},
                "streams": [{"name": "leak", "sync_mode": "full_refresh_overwrite"}],
            }
        ]
    }
    write_pipeline(session, json.dumps(config), {})

    report = session.run_extract_load()

    assert "not a source of this task" in report["connections"][0]["error"]
    assert session.raw_tables == set()


def test_absolute_path_to_the_reference_models_is_not_a_task_source(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    target = (task.reference_dir / "models" / "member_activity.sql").as_posix()
    config = {
        "connections": [
            {
                "name": "leak",
                "source": {"type": "file", "configuration": {"dataset_name": "leak", "format": "csv", "url": target}},
                "streams": [{"name": "leak", "sync_mode": "full_refresh_overwrite"}],
            }
        ]
    }
    write_pipeline(session, json.dumps(config), {})

    assert "not a source of this task" in session.run_extract_load()["connections"][0]["error"]


@pytest.mark.parametrize(
    ("source_type", "configuration"),
    [
        ("sqlite", {"database": "../grader/task.yaml"}),
        ("documents", {"connection_string": "docstore://..", "database": "grader"}),
        ("documents", {"connection_string": "docstore://docs", "database": "../../grader"}),
    ],
)
def test_database_sources_cannot_leave_the_source_directory(
    source_type, configuration, local_tasks, warehouse, workspace
):
    session = open_session(local_tasks["storefront"], warehouse, workspace)
    config = {
        "connections": [
            {
                "name": "leak",
                "source": {"type": source_type, "configuration": configuration},
                "streams": [{"name": "models", "sync_mode": "full_refresh_overwrite"}],
            }
        ]
    }
    write_pipeline(session, json.dumps(config), {})

    report = session.run_extract_load()

    assert "is not a source of this task" in report["connections"][0]["error"]
    assert session.raw_tables == set()


# ---- files outside the workspace -----------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["../grader/task.yaml", "../grader/models/member_activity.sql", "..", "../../.."])
def test_read_and_list_cannot_leave_the_workspace(path, local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)

    assert "outside the workspace" in session.read_file(path)["error"]
    assert "outside the workspace" in session.list_files(path)["error"]


def test_read_cannot_reach_the_reference_by_absolute_path(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)

    assert "error" in session.read_file(str(task.reference_dir / "models" / "member_activity.sql"))


@pytest.mark.parametrize(
    "path",
    [
        "data_model.yaml",
        "config.yaml",
        "pipeline/models/../../data_model.yaml",
        "../outside.sql",
        "pipeline/models/nested/model.sql",
        "pipeline/models/not a name.sql",
        "pipeline/models/model.py",
    ],
)
def test_write_is_limited_to_the_pipeline_files(path, local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    before = (session.workspace / "data_model.yaml").read_text(encoding="utf-8")

    assert "error" in session.write_file(path, "select 1")
    assert (session.workspace / "data_model.yaml").read_text(encoding="utf-8") == before


# ---- SQL that reaches outside the rollout's own tables ---------------------------------------------------------------


def test_query_cannot_read_the_reference_namespace_or_another_rollout(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    peer = open_session(task, warehouse, workspace / "peer")
    solve(peer)
    grade(peer)
    session = open_session(task, warehouse, workspace / "own")
    solve(session)

    for sql in (
        f"select * from {peer.namespace}.member_activity",
        f"select * from reference_{task.task_id}.member_activity",
        f"select * from memory.{peer.namespace}.members",
        "select * from information_schema.tables",
        "select * from duckdb_tables",
        "select * from duckdb_tables()",
        "select * from pg_catalog.pg_tables",
        f"select * from query_table('{peer.namespace}.members')",
        f"select * from query('select * from {peer.namespace}.members')",
    ):
        assert "error" in session.query(sql), sql


@pytest.mark.parametrize(
    "sql",
    [
        "select * from read_csv('{reference}')",
        "select * from read_text('{reference}')",
        "select * from '{reference}'",
        "select content from read_blob('{reference}')",
        "copy members to 'stolen.csv'",
        "attach 'other.duckdb' as other",
        "install httpfs",
        "set enable_external_access = true",
        "pragma database_list",
        "create table extra as select 1",
        "drop table members",
        "insert into members select * from members",
        "update members set name = 'x'",
        "delete from members",
        "select 1; drop table members",
        "select * into copied from members",
        "select getenv('PATH')",
    ],
)
def test_query_accepts_only_a_single_read_only_select(sql, local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    rows_before = warehouse.row_count(session.namespace, "members")
    reference_file = (task.reference_dir / "models" / "member_activity.sql").as_posix()

    assert "error" in session.query(sql.format(reference=reference_file))
    assert warehouse.row_count(session.namespace, "members") == rows_before


@pytest.mark.parametrize(
    "sql",
    [
        "drop table members",
        "select 1; drop table members",
        "create table x as select 1",
        "select * from read_csv('x.csv')",
    ],
)
def test_model_file_with_a_side_effect_is_not_run(sql, local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    solve(session)
    write_pipeline(session, None, {"sneaky": sql})

    report = {entry["model"]: entry for entry in session.run_transform()["models"]}

    assert report["sneaky"]["status"] == "failed"
    assert warehouse.row_count(session.namespace, "members") > 0


def test_model_cannot_take_the_name_of_a_source_table(local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    solve(session)
    rows_before = warehouse.row_count(session.namespace, "loans")
    write_pipeline(session, None, {"loans": "select 1 as loan_id"})

    report = {entry["model"]: entry for entry in session.run_transform()["models"]}

    assert "already the name of a source table" in report["loans"]["error"]
    assert warehouse.row_count(session.namespace, "loans") == rows_before


def test_duckdb_refuses_file_access_even_without_the_sql_guard(local_tasks, warehouse):
    reference_file = (local_tasks["library"].reference_dir / "task.yaml").as_posix()

    for sql in (f"select * from read_text('{reference_file}')", "copy (select 1) to 'x.csv'", "attach 'x.duckdb' as x"):
        with pytest.raises(WarehouseError):
            warehouse.run_query(sql, max_rows=1, timeout_seconds=5)


# ---- the comparison itself -----------------------------------------------------------------------------------------


def test_strict_rule_rejects_an_off_by_one_count_that_the_benchmark_tolerance_accepts():
    expected = pd.DataFrame({"id": [1, 2], "num_orders": [100, 250]})
    produced = pd.DataFrame({"id": [1, 2], "num_orders": [101, 252]})

    assert compare_tables(expected, produced, key=["id"], rule=OFFICIAL).is_full_match
    assert compare_tables(expected, produced, key=["id"], rule=STRICT).unmatched == ("num_orders",)


def test_strict_rule_accepts_the_same_number_in_another_format():
    expected = pd.DataFrame({"id": ["1", "2"], "share": ["33.333333", "5"], "label": ["a", None]})
    produced = pd.DataFrame({"ID": [2, 1], "share": [5.0, 33.33333301], "label": [None, "a"], "extra": [0, 0]})

    assert compare_tables(expected, produced, key=["id"], rule=STRICT).is_full_match


def test_missing_value_does_not_match_a_present_value():
    expected = pd.DataFrame({"id": [1, 2], "score": [None, 3.0]})
    produced = pd.DataFrame({"id": [1, 2], "score": [0.0, 3.0]})

    assert compare_tables(expected, produced, key=["id"], rule=STRICT).unmatched == ("score",)


def test_extra_or_missing_rows_match_no_column():
    expected = pd.DataFrame({"id": [1, 2], "value": [10, 20]})

    more = compare_tables(expected, pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}), key=["id"])
    fewer = compare_tables(expected, pd.DataFrame({"id": [1], "value": [10]}), key=["id"])

    assert more.matched == () and fewer.matched == ()


def test_wrong_key_values_match_no_column():
    expected = pd.DataFrame({"id": [1, 2], "constant": [0, 0]})
    produced = pd.DataFrame({"id": [1, 3], "constant": [0, 0]})

    assert compare_tables(expected, produced, key=["id"]).matched == ()


def test_columns_that_differ_only_in_case_do_not_count():
    expected = pd.DataFrame({"id": [1], "total": [5]})
    produced = pd.DataFrame({"id": [1], "total": [0], "TOTAL": [5]})

    assert "total" in compare_tables(expected, produced, key=["id"]).missing


# ---- the grader and the limits -------------------------------------------------------------------------------------


def test_grader_failure_gives_zero_reward(local_tasks, warehouse, workspace, monkeypatch):
    session = open_session(local_tasks["library"], warehouse, workspace)
    solve(session)

    def broken(self, model):
        raise RuntimeError("expected table unavailable")

    monkeypatch.setattr(TaskReference, "expected", broken)

    result = grade(session)

    assert result.reward == 0.0
    assert result.grader_error is not None


def test_runaway_query_is_stopped_at_the_statement_limit(local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    session.limits = SessionLimits(statement_timeout_seconds=0.5)
    solve(session)

    result = session.query("select count(*) from range(100000000000) a, range(100000) b")

    assert "second limit" in result["error"]


def test_query_returns_a_bounded_number_of_rows(local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    solve(session)

    result = session.query("select * from loans")

    assert len(result["rows"]) == session.limits.max_query_rows
    assert result["truncated"]


def test_connection_syncs_are_capped(local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)
    write_pipeline(session, reference_extract_load(session.task), {})
    # Two source tables give eight syncs; the file has two connections.
    both = ["members_file", "loans_file"]
    for _ in range(session.limits.max_syncs_per_source_table):
        assert "error" not in session.run_extract_load(both)

    assert "at most 8 connection syncs; 8 are used" in session.run_extract_load(both)["error"]


def test_oversized_file_is_refused(local_tasks, warehouse, workspace):
    session = open_session(local_tasks["library"], warehouse, workspace)

    result = session.write_file(EXTRACT_LOAD_PATH, "x" * (session.limits.max_file_bytes + 1))

    assert "at most" in result["error"]
