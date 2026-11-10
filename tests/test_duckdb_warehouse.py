"""The Warehouse contract on DuckDB: namespaces, queries, loads, copies, sampling and row digests."""

import pandas as pd
import pytest

from elt_rlvr.warehouse import Column, WarehouseError

NS = "first_ns"
OTHER = "second_ns"


@pytest.fixture
def db(warehouse):
    warehouse.create_namespace(NS)
    warehouse.create_namespace(OTHER)
    return warehouse


def _write(warehouse, namespace: str, table: str, rows: list[tuple], columns=("id", "name")) -> None:
    warehouse.write_frame(namespace, table, pd.DataFrame(rows, columns=list(columns)), append=False)


def _ids(warehouse, namespace: str, table: str) -> list:
    return sorted(warehouse.read_frame(namespace, table)["id"].tolist())


# ---- namespaces --------------------------------------------------------------------------------------------------


def test_create_namespace_replaces_an_existing_namespace(db):
    _write(db, NS, "people", [(1, "a")])

    db.create_namespace(NS)

    assert db.list_tables(NS) == []


def test_drop_namespace_removes_its_tables_and_tolerates_a_missing_namespace(db):
    _write(db, NS, "people", [(1, "a")])

    db.drop_namespace(NS)
    db.drop_namespace(NS)
    db.drop_namespace("never_created")

    assert db.list_tables(NS) == []
    with pytest.raises(WarehouseError):
        db.read_frame(NS, "people")


@pytest.mark.parametrize("name", ["bad-name", "two words", "x; drop schema y", "1st"])
def test_namespace_name_that_is_not_an_identifier_is_refused(name, warehouse):
    with pytest.raises(ValueError, match="not a plain SQL identifier"):
        warehouse.create_namespace(name)


def test_list_tables_is_sorted_and_limited_to_the_namespace(db):
    _write(db, NS, "zeta", [(1, "a")])
    _write(db, NS, "alpha", [(1, "a")])
    _write(db, OTHER, "beta", [(1, "a")])

    assert db.list_tables(NS) == ["alpha", "zeta"]
    assert db.list_tables(OTHER) == ["beta"]


# ---- describe, queries and materialization -----------------------------------------------------------------------


def test_describe_lists_columns_in_order_with_a_numeric_flag(db):
    db.create_table_from_query(
        NS,
        "typed",
        "select 1::integer as i, 2::bigint as b, 3.5::decimal(10, 2) as d, 4.5::double as f, 'x' as s, "
        "date '2020-01-01' as day, true as flag",
        timeout_seconds=10,
    )

    columns = db.describe(NS, "typed")

    assert [(column.name, column.is_numeric) for column in columns] == [
        ("i", True),
        ("b", True),
        ("d", True),
        ("f", True),
        ("s", False),
        ("day", False),
        ("flag", False),
    ]
    assert columns[0] == Column("i", "INTEGER", True)


def test_describe_missing_table_raises_a_warehouse_error(db):
    with pytest.raises(WarehouseError):
        db.describe(NS, "missing")


@pytest.mark.parametrize(("max_rows", "returned", "truncated"), [(2, 2, True), (3, 3, False), (10, 3, False)])
def test_run_query_returns_at_most_max_rows_and_says_when_it_truncated(max_rows, returned, truncated, db):
    _write(db, NS, "people", [(1, "a"), (2, "b"), (3, "c")])

    result = db.run_query(f"select id, name from {NS}.people order by id", max_rows=max_rows, timeout_seconds=10)

    assert result.columns == ("id", "name")
    assert result.rows == [(1, "a"), (2, "b"), (3, "c")][:returned]
    assert result.truncated is truncated


def test_run_query_error_is_raised_as_a_warehouse_error(db):
    with pytest.raises(WarehouseError, match="missing"):
        db.run_query(f"select * from {NS}.missing", max_rows=1, timeout_seconds=10)


def test_create_table_from_query_returns_the_row_count_and_replaces_the_table(db):
    first = db.create_table_from_query(NS, "numbers", "select range as id from range(5)", timeout_seconds=10)
    second = db.create_table_from_query(NS, "numbers", "select range as id from range(2)", timeout_seconds=10)

    assert (first, second) == (5, 2)
    assert _ids(db, NS, "numbers") == [0, 1]


def test_create_table_from_query_with_an_empty_result_returns_zero(db):
    assert db.create_table_from_query(NS, "nothing", "select 1 as id where false", timeout_seconds=10) == 0
    assert db.list_tables(NS) == ["nothing"]


# ---- bulk load ---------------------------------------------------------------------------------------------------


def test_write_frame_replaces_unless_appending(db):
    replaced = db.write_frame(NS, "people", pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}), append=False)
    appended = db.write_frame(NS, "people", pd.DataFrame({"id": [3, 4, 5], "name": list("cde")}), append=True)
    assert (replaced, appended) == (2, 3)
    assert _ids(db, NS, "people") == [1, 2, 3, 4, 5]

    db.write_frame(NS, "people", pd.DataFrame({"id": [9], "name": ["z"]}), append=False)

    assert _ids(db, NS, "people") == [9]


def test_write_frame_append_matches_columns_by_name(db):
    db.write_frame(NS, "people", pd.DataFrame({"id": [1], "name": ["a"]}), append=False)

    db.write_frame(NS, "people", pd.DataFrame({"name": ["b"], "id": [2]}), append=True)

    frame = db.read_frame(NS, "people").sort_values("id")
    assert frame.to_dict("records") == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_write_frame_append_to_a_missing_table_creates_it(db):
    assert db.write_frame(NS, "people", pd.DataFrame({"id": [1, 2]}), append=True) == 2
    assert db.row_count(NS, "people") == 2


def test_read_frame_returns_the_whole_table(db):
    _write(db, NS, "people", [(1, "a"), (2, None)])

    frame = db.read_frame(NS, "people").sort_values("id")

    assert list(frame.columns) == ["id", "name"]
    assert frame["id"].tolist() == [1, 2]
    assert frame["name"].tolist()[0] == "a" and pd.isna(frame["name"].tolist()[1])


# ---- copies and samples ------------------------------------------------------------------------------------------


def test_copy_table_copies_the_rows_and_replaces_the_target(db):
    _write(db, NS, "people", [(1, "a"), (2, "b")])
    _write(db, OTHER, "people", [(7, "x"), (8, "y"), (9, "z")])

    copied = db.copy_table(NS, "people", OTHER)

    assert copied == 2
    assert db.read_frame(OTHER, "people").sort_values("id").to_dict("records") == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]


def test_copy_table_append_adds_the_rows(db):
    _write(db, NS, "people", [(1, "a"), (2, "b")])
    db.copy_table(NS, "people", OTHER)

    added = db.copy_table(NS, "people", OTHER, append=True)

    assert added == 2
    assert _ids(db, OTHER, "people") == [1, 1, 2, 2]


def test_copy_table_append_to_a_missing_target_creates_it(db):
    _write(db, NS, "people", [(1, "a"), (2, "b")])

    assert db.copy_table(NS, "people", OTHER, append=True) == 2
    assert _ids(db, OTHER, "people") == [1, 2]


def test_copy_table_sample_keeps_a_subset_that_is_repeatable_for_one_seed(db):
    db.create_table_from_query(NS, "numbers", "select range as id from range(1000)", timeout_seconds=10)
    db.create_namespace("sample_a")
    db.create_namespace("sample_b")

    count_a = db.copy_table(NS, "numbers", "sample_a", sample_percent=50, seed=7)
    count_b = db.copy_table(NS, "numbers", "sample_b", sample_percent=50, seed=7)

    sample = _ids(db, "sample_a", "numbers")
    assert 0 < count_a < 1000
    assert len(sample) == count_a
    assert set(sample) <= set(range(1000))
    assert count_b == count_a
    assert _ids(db, "sample_b", "numbers") == sample


def test_delete_orphans_drops_children_of_removed_parents_only(db):
    _write(db, NS, "customers", [(1, "a"), (2, "b"), (3, "c")])
    _write(db, OTHER, "customers", [(1, "a"), (3, "c")])
    orders = pd.DataFrame({"order_id": [10, 11, 12, 13, 14], "customer_id": pd.array([1, 2, 3, 4, None], "Int64")})
    db.write_frame(OTHER, "orders", orders, append=False)

    db.delete_orphans(
        OTHER, "orders", "customer_id", parent_table="customers", parent_column="id", original_namespace=NS
    )

    # Customer 2 was removed from the sample; customer 4 never existed and a NULL key has no parent.
    assert sorted(db.read_frame(OTHER, "orders")["order_id"].tolist()) == [10, 12, 13, 14]


def _orders_after_one_customer_lost_its_orders(db, customer_ids: list, seed: int) -> list:
    _write(db, NS, "customers", [(1, "a"), (2, "b"), (3, "c")])
    orders = pd.DataFrame(
        {"order_id": range(10, 10 + len(customer_ids)), "customer_id": pd.array(customer_ids, "Int64")}
    )
    db.write_frame(NS, "orders", orders, append=False)

    db.delete_children_of_one_parent(
        NS, "orders", "customer_id", parent_table="customers", parent_column="id", seed=seed
    )

    frame = db.read_frame(NS, "orders").sort_values("order_id")
    return [None if pd.isna(value) else int(value) for value in frame["customer_id"]]


@pytest.mark.parametrize("seed", range(6))
def test_delete_children_of_one_parent_removes_every_row_of_exactly_one_parent(seed, db):
    remaining = _orders_after_one_customer_lost_its_orders(db, [1, 1, 2, 2, 2, 3, 4, None], seed)

    # Customer 4 is not in the parent table and a NULL key has no parent, so neither can be the one chosen.
    assert remaining.count(None) == 1 and remaining.count(4) == 1
    kept = [value for value in remaining if value in (1, 2, 3)]
    assert kept in ([1, 1, 2, 2, 2], [1, 1, 3], [2, 2, 2, 3])


def test_delete_children_of_one_parent_chooses_by_seed(db):
    chosen = set()
    for seed in range(12):
        remaining = _orders_after_one_customer_lost_its_orders(db, [1, 2, 3], seed)
        chosen |= {1, 2, 3} - set(remaining)

    assert len(chosen) > 1


def test_delete_children_of_one_parent_keeps_the_rows_of_an_only_parent(db):
    assert _orders_after_one_customer_lost_its_orders(db, [2, 2, 4], seed=0) == [2, 2, 4]


# ---- fingerprints ------------------------------------------------------------------------------------------------

ROWS = [(1, "x"), (2, "y"), (3, "z")]


def _fingerprint(warehouse, table: str, numeric=("id",)):
    return warehouse.fingerprint(NS, table, ["id", "name"], list(numeric))


def test_fingerprint_is_equal_for_the_same_rows_in_another_order(db):
    _write(db, NS, "first", ROWS)
    _write(db, NS, "second", list(reversed(ROWS)))

    assert _fingerprint(db, "first") == _fingerprint(db, "second")
    assert _fingerprint(db, "first").row_count == 3


def test_fingerprint_does_not_depend_on_the_order_columns_are_listed(db):
    _write(db, NS, "people", ROWS)

    assert db.fingerprint(NS, "people", ["id", "name"], ["id"]) == db.fingerprint(NS, "people", ["name", "id"], ["id"])


def test_fingerprint_ignores_columns_that_are_not_listed(db):
    _write(db, NS, "first", [(1, "x", "load-1"), (2, "y", "load-1")], columns=("id", "name", "loaded_by"))
    _write(db, NS, "second", [(1, "x", "load-2"), (2, "y", "load-2")], columns=("id", "name", "loaded_by"))

    assert _fingerprint(db, "first") == _fingerprint(db, "second")


def test_fingerprint_is_equal_for_bigint_and_decimal_when_the_column_is_numeric(db):
    db.create_table_from_query(NS, "integers", "select range::bigint as v from range(1, 4)", timeout_seconds=10)
    db.create_table_from_query(NS, "decimals", "select range::decimal(18, 3) as v from range(1, 4)", timeout_seconds=10)

    integers = db.fingerprint(NS, "integers", ["v"], ["v"])
    decimals = db.fingerprint(NS, "decimals", ["v"], ["v"])

    assert integers == decimals


def test_fingerprint_compares_bigint_and_decimal_as_text_when_the_column_is_not_numeric(db):
    db.create_table_from_query(NS, "integers", "select range::bigint as v from range(1, 4)", timeout_seconds=10)
    db.create_table_from_query(NS, "decimals", "select range::decimal(18, 3) as v from range(1, 4)", timeout_seconds=10)

    assert db.fingerprint(NS, "integers", ["v"], []) != db.fingerprint(NS, "decimals", ["v"], [])


@pytest.mark.parametrize(
    "other_rows",
    [
        [(1, "x"), (2, "Y"), (3, "z")],
        [(1, "x"), (2, "y"), (4, "z")],
        [(1, "x"), (2, "y")],
        [(1, "x"), (2, "y"), (3, "z"), (3, "z")],
        [(1, "x"), (2, "y"), (2, "y")],
    ],
    ids=["changed_text", "changed_number", "missing_row", "duplicated_row", "row_duplicated_in_place_of_another"],
)
def test_fingerprint_differs_when_the_rows_differ(other_rows, db):
    _write(db, NS, "expected", ROWS)
    _write(db, NS, "produced", other_rows)

    assert _fingerprint(db, "produced") != _fingerprint(db, "expected")


def test_fingerprint_digest_differs_for_a_duplicated_row_with_the_same_row_count(db):
    _write(db, NS, "expected", ROWS)
    _write(db, NS, "produced", [(1, "x"), (2, "y"), (2, "y")])

    expected, produced = _fingerprint(db, "expected"), _fingerprint(db, "produced")

    assert produced.row_count == expected.row_count
    assert produced.digest != expected.digest


def test_fingerprint_of_an_empty_table_has_no_rows(db):
    db.create_table_from_query(NS, "empty", "select 1 as id, 'x' as name where false", timeout_seconds=10)

    assert _fingerprint(db, "empty").row_count == 0
    assert db.row_count(NS, "empty") == 0
