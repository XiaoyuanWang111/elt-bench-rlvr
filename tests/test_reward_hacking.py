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
from elt_rlvr.grading.grader import RewardConfig
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


# ---- raw tables that did not come from the sources -----------------------------------------------------------------


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

    assert report["connections"][0]["status"] == "failed"
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
