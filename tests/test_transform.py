"""Running model files on DuckDB: dependency order, failure reports, cycles and partial runs."""

import pandas as pd
import pytest

from elt_rlvr.transform import ModelResult, run_models

RAW = "raw_tables"
TARGET = "model_tables"


@pytest.fixture
def raw_warehouse(warehouse):
    warehouse.create_namespace(RAW)
    warehouse.create_namespace(TARGET)
    orders = pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [7, 7, 8], "amount": [10, 20, 30]})
    warehouse.write_frame(RAW, "orders", orders, append=False)
    return warehouse


def _run(warehouse, models: dict[str, str], **options) -> dict[str, ModelResult]:
    results = run_models(
        warehouse,
        models=models,
        raw_table_namespaces={"orders": RAW},
        target_namespace=TARGET,
        timeout_seconds=30,
        **options,
    )
    return {result.name: result for result in results}


def test_run_models_builds_dependencies_first_whatever_the_file_order(raw_warehouse):
    models = {
        "report": "select count(*) as num_customers, sum(total) as total from customer_totals",
        "customer_totals": "select customer_id, sum(amount) as total from orders group by customer_id",
    }

    results = _run(raw_warehouse, models)

    assert results["report"] == ModelResult("report", succeeded=True, rows=1, columns=("num_customers", "total"))
    assert results["customer_totals"].succeeded
    report = raw_warehouse.read_frame(TARGET, "report")
    assert report.to_dict("records") == [{"num_customers": 2, "total": 60}]


def test_run_models_model_reads_another_model_by_its_bare_name(raw_warehouse):
    models = {
        "doubled": "select order_id, amount * 2 as amount from orders",
        "large": "select order_id from doubled where amount >= 40 order by order_id",
    }

    results = _run(raw_warehouse, models)

    assert results["large"].rows == 2
    assert raw_warehouse.read_frame(TARGET, "large")["order_id"].tolist() == [2, 3]


def test_run_models_failed_model_does_not_stop_independent_models(raw_warehouse):
    models = {
        "broken": "select * from payments",
        "counts": "select count(*) as n from orders",
        "invalid": "select no_such_column from orders",
    }

    results = _run(raw_warehouse, models)

    assert "unknown table 'payments'" in results["broken"].error
    assert not results["invalid"].succeeded
    assert results["counts"] == ModelResult("counts", succeeded=True, rows=1, columns=("n",))
    assert raw_warehouse.list_tables(TARGET) == ["counts"]


@pytest.mark.parametrize(
    "broken_sql",
    ["select * from payments", "select no_such_column from orders"],
    ids=["rejected_by_the_guard", "failed_in_the_warehouse"],
)
def test_run_models_dependent_of_a_failed_model_names_the_failed_dependency(broken_sql, raw_warehouse):
    results = _run(raw_warehouse, {"child": "select * from base", "base": broken_sql})

    assert not results["base"].succeeded
    assert not results["child"].succeeded
    assert results["child"].error == f"depends on model 'base', which failed: {results['base'].error}"
    assert raw_warehouse.list_tables(TARGET) == []


def test_run_models_cycle_is_reported_and_other_models_still_run(raw_warehouse):
    models = {"a": "select * from b", "b": "select * from a", "c": "select 1 as x"}

    results = _run(raw_warehouse, models)

    assert not results["a"].succeeded and not results["b"].succeeded
    assert "models depend on each other in a cycle: a -> b -> a" in results["a"].error
    assert "cycle" in results["b"].error
    assert results["c"].succeeded


def test_run_models_model_cannot_read_itself(raw_warehouse):
    results = _run(raw_warehouse, {"loop": "select * from loop"})

    assert "unknown table 'loop'" in results["loop"].error


@pytest.mark.parametrize("only", [["child"], ["CHILD"]])
def test_run_models_only_runs_the_named_model_and_its_dependencies(only, raw_warehouse):
    models = {
        "base": "select * from orders",
        "child": "select count(*) as n from base",
        "other": "select 1 as x",
    }

    results = _run(raw_warehouse, models, only=only)

    assert set(results) == {"base", "child"}
    assert all(result.succeeded for result in results.values())
    assert raw_warehouse.list_tables(TARGET) == ["base", "child"]


def test_run_models_only_with_an_unknown_name_runs_nothing(raw_warehouse):
    assert _run(raw_warehouse, {"base": "select * from orders"}, only=["missing"]) == {}


@pytest.mark.parametrize("name", ["orders", "ORDERS"])
def test_run_models_model_named_like_a_raw_table_fails(name, raw_warehouse):
    results = _run(raw_warehouse, {name: "select 1 as order_id"})

    assert results[name].error == f"model name {name!r} is already the name of a source table"
    assert raw_warehouse.row_count(RAW, "orders") == 3
    assert raw_warehouse.list_tables(TARGET) == []


def test_run_models_without_describe_results_reports_rows_but_no_columns(raw_warehouse):
    results = _run(raw_warehouse, {"all_orders": "select * from orders"}, describe_results=False)

    assert results["all_orders"] == ModelResult("all_orders", succeeded=True, rows=3, columns=())
