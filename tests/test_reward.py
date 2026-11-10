"""The reward separates a correct pipeline from incomplete and wrong ones."""

import dataclasses

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

from elt_rlvr.grading.grader import RewardConfig, combine
from elt_rlvr.tasks.local import LOCAL_TASKS


@pytest.mark.parametrize("task_id", list(LOCAL_TASKS))
def test_reference_pipeline_earns_full_reward(task_id, local_tasks, warehouse, workspace):
    session = open_session(local_tasks[task_id], warehouse, workspace)
    solve(session)

    result = grade(session)

    assert result.grader_error is None
    assert result.reward == 1.0
    assert result.official_extract_load
    assert result.official_transform == 1.0


def test_ground_truth_with_upper_case_headers_grades_like_the_reference_tables(
    local_tasks, warehouse, workspace, tmp_path
):
    # ELT-Bench's ground truth has upper-case headers, the reference models on DuckDB produce lower-case names.
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    solve(session)
    for model in task.data_models:
        frame = warehouse.read_frame(session.namespace, model.name)
        frame.rename(columns=str.upper).to_csv(tmp_path / f"{model.name}.csv", index=False)
    session.task = dataclasses.replace(task, ground_truth_dir=tmp_path)

    result = grade(session)

    assert result.reward == 1.0
    assert all(all(columns.values()) for columns in result.model_columns.values())


@pytest.mark.parametrize("task_id", list(LOCAL_TASKS))
def test_untouched_workspace_earns_nothing(task_id, local_tasks, warehouse, workspace):
    session = open_session(local_tasks[task_id], warehouse, workspace)

    assert grade(session).reward == 0.0


def test_load_without_models_earns_extract_load_weight(local_tasks, warehouse, workspace):
    session = open_session(local_tasks["storefront"], warehouse, workspace)
    write_pipeline(session, reference_extract_load(session.task), {})
    session.run_extract_load()

    result = grade(session)

    assert result.extract_load_score == 1.0
    assert result.transform_score == 0.0
    assert result.reward == pytest.approx(RewardConfig().extract_load_weight)


def test_missing_source_table_costs_its_share_and_the_model_that_needs_it(local_tasks, warehouse, workspace):
    task = local_tasks["storefront"]
    session = open_session(task, warehouse, workspace)

    def drop_reviews(document):
        document["connections"] = [c for c in document["connections"] if c["name"] != "feedback_documents"]

    models = reference_models(task)
    write_pipeline(session, edit_extract_load(task, drop_reviews), models)
    session.run_extract_load()
    session.run_transform()

    result = grade(session)

    assert result.source_tables == {"customers": True, "orders": True, "products": True, "reviews": False}
    assert result.extract_load_score == 0.75
    # customers_summary needs no reviews and is fully right; products_summary cannot be built.
    assert result.model_columns["customers_summary"] == dict.fromkeys(task.data_models[0].columns, True)
    assert result.transform_score == 0.5
    assert result.reward == pytest.approx(0.3 * 0.75 + 0.7 * 0.5)


def test_wrong_column_loses_only_that_column(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    models = reference_models(task)
    models["member_activity"] = models["member_activity"].replace("l.days_late > 14", "l.days_late > 0")
    write_pipeline(session, reference_extract_load(task), models)
    session.run_extract_load()
    session.run_transform()

    result = grade(session)

    columns = result.model_columns["member_activity"]
    assert columns == {
        "member_id": True,
        "name": True,
        "num_loans": True,
        "total_days_late": True,
        "has_overdue_loan": False,
    }
    assert result.reward == pytest.approx(0.3 + 0.7 * 0.8)


def test_inner_join_that_drops_rows_earns_no_transform_credit(local_tasks, warehouse, workspace):
    task = local_tasks["library"]
    session = open_session(task, warehouse, workspace)
    models = {"member_activity": reference_models(task)["member_activity"].replace("left join", "join")}
    write_pipeline(session, reference_extract_load(task), models)
    session.run_extract_load()
    session.run_transform()

    result = grade(session)

    assert result.transform_score == 0.0
    assert result.reward == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [("multiplicative", 0.5 * (0.3 + 0.7 * 0.5)), ("additive", 0.3 * 0.5 + 0.7 * 0.5), ("binary", 0.0)],
)
def test_reward_shapes_combine_partial_scores(shape, expected):
    assert combine(0.5, 0.5, RewardConfig(shape=shape)) == pytest.approx(expected)


def test_binary_shape_pays_only_a_complete_pipeline():
    assert combine(1.0, 1.0, RewardConfig(shape="binary")) == 1.0
    assert combine(1.0, 0.99, RewardConfig(shape="binary")) == 0.0
