"""The SQL guard accepts one read-only query over known tables, qualifies every table, and rejects everything else."""

import pytest

from elt_rlvr.sql_guard import MAX_SQL_CHARS, SqlRejectedError, vet_query

TABLES = {"members": "ns_a", "loans": "ns_a", "Orders": "ns_b"}


def _vet(sql: str, dialect: str = "duckdb"):
    return vet_query(sql, dialect=dialect, table_namespaces=TABLES)


# ---- accepted queries --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "tables"),
    [
        ("select * from members", ("members",)),
        ("with active as (select * from members where joined_year > 2020) select * from active", ("members",)),
        ("select member_id from members union select member_id from loans", ("members", "loans")),
        ("select member_id from members union all select member_id from loans", ("members", "loans")),
        ("select member_id from members intersect select member_id from loans", ("members", "loans")),
        ("select member_id from members except select member_id from loans", ("members", "loans")),
        ("select * from members where member_id in (select member_id from loans)", ("members", "loans")),
        ("select * from (select member_id from members) as recent", ("members",)),
        ("select (select count(*) from loans) as loan_count from members", ("members", "loans")),
        (
            "select member_id, row_number() over (partition by city order by member_id) as position from members",
            ("members",),
        ),
        ("select * from unnest([1, 2, 3]) as t(x)", ()),
        ("select * from generate_series(1, 3) as t(x)", ()),
        ("select * from range(3) as t(x)", ()),
        ("(select * from members)", ("members",)),
    ],
)
def test_vet_query_read_only_query_is_accepted_and_every_table_qualified(sql, tables):
    result = _vet(sql)

    assert result.tables == tables
    assert all(f"{TABLES[table]}.{table}" in result.sql for table in tables)


def test_vet_query_tables_in_different_namespaces_are_each_qualified_with_their_own_namespace():
    result = _vet("select * from members m join orders o on o.member_id = m.member_id")

    assert result.tables == ("members", "Orders")
    assert "ns_a.members" in result.sql
    assert "ns_b.orders" in result.sql


@pytest.mark.parametrize("sql", ["select * from MEMBERS", "select * from Members", 'select * from "members"'])
def test_vet_query_table_name_in_another_case_resolves_to_the_known_table(sql):
    result = _vet(sql)

    assert result.tables == ("members",)
    assert "ns_a." in result.sql


def test_vet_query_table_read_twice_is_listed_once():
    result = _vet("select * from members a join members b on a.member_id = b.member_id")

    assert result.tables == ("members",)
    assert result.sql.count("ns_a.members") == 2


@pytest.mark.parametrize(
    "sql",
    [
        "select * from members;",
        "select * from members;;",
        "select * from members; -- done",
        "-- all members\nselect * /* every column */ from members -- end",
    ],
)
def test_vet_query_trailing_semicolon_and_comments_are_accepted(sql):
    assert _vet(sql).tables == ("members",)


def test_vet_query_statement_at_the_length_limit_is_accepted():
    sql = "select '" + "x" * (MAX_SQL_CHARS - len("select ''")) + "'"

    assert len(sql) == MAX_SQL_CHARS
    assert _vet(sql).tables == ()


# ---- common table expressions ------------------------------------------------------------------------------------


def test_vet_query_cte_that_shadows_a_table_name_is_read_as_the_cte():
    result = _vet("with loans as (select 1 as loan_id) select * from loans")

    assert result.tables == ()
    assert "ns_a.loans" not in result.sql


def test_vet_query_cte_name_does_not_hide_a_table_outside_the_cte_scope():
    result = _vet("select * from (with loans as (select 1 as x) select * from loans) as inner_query cross join loans")

    assert result.tables == ("loans",)
    assert "CROSS JOIN ns_a.loans" in result.sql


def test_vet_query_cte_name_cannot_unlock_a_catalog_view_outside_the_cte_scope():
    sql = "select * from duckdb_tables where exists (with duckdb_tables as (select 1) select * from duckdb_tables)"

    with pytest.raises(SqlRejectedError, match="unknown table 'duckdb_tables'"):
        _vet(sql)


def test_vet_query_cte_body_that_reads_the_table_it_shadows_reads_the_real_table():
    # A non-recursive CTE is not visible inside its own body, so the inner reference is the raw table.
    result = _vet("with loans as (select * from loans where days_late > 0) select count(*) from loans")

    assert result.tables == ("loans",)
    assert "FROM ns_a.loans WHERE" in result.sql


def test_vet_query_cte_is_visible_to_the_ctes_declared_after_it():
    result = _vet("with late as (select * from loans), counts as (select count(*) as n from late) select * from counts")

    assert result.tables == ("loans",)


def test_vet_query_cte_is_not_visible_to_the_ctes_declared_before_it():
    with pytest.raises(SqlRejectedError, match="unknown table 'later'"):
        _vet("with first as (select * from later), later as (select 1 as x) select * from first")


def test_vet_query_recursive_cte_reads_itself():
    result = _vet(
        "with recursive steps as (select 1 as n union all select n + 1 from steps where n < 3) select * from steps"
    )

    assert result.tables == ()


def test_vet_query_cte_declared_on_a_union_is_visible_in_both_branches():
    result = _vet("with late as (select member_id from loans) select * from late union all select * from late")

    assert result.tables == ("loans",)


def test_vet_query_snowflake_unquoted_cte_name_matches_in_any_case():
    result = _vet("with Active as (select * from members) select * from ACTIVE", dialect="snowflake")

    assert result.tables == ("members",)


def test_vet_query_snowflake_quoted_cte_name_does_not_cover_an_unquoted_reference():
    # Snowflake reads the unquoted name as SECRETS, which the CTE "secrets" does not declare.
    with pytest.raises(SqlRejectedError, match="unknown table 'secrets'"):
        _vet('with "secrets" as (select 1 as x) select * from secrets', dialect="snowflake")


# ---- rejected queries --------------------------------------------------------------------------------------------


def test_vet_query_unknown_table_is_rejected_with_the_readable_tables():
    with pytest.raises(SqlRejectedError, match="unknown table 'payments'; tables you can read: Orders, loans, members"):
        _vet("select * from payments")


def test_vet_query_with_no_readable_tables_says_so():
    with pytest.raises(SqlRejectedError, match=r"tables you can read: \(none\)"):
        vet_query("select * from members", dialect="duckdb", table_namespaces={})


@pytest.mark.parametrize(
    "sql",
    [
        "select * from ns_a.members",
        "select * from ns_b.orders",
        "select * from memory.ns_a.members",
        "select * from information_schema.tables",
        "select * from pg_catalog.pg_tables",
        "select * from members m join ns_b.orders o on o.member_id = m.member_id",
    ],
)
def test_vet_query_qualified_table_name_is_rejected(sql):
    with pytest.raises(SqlRejectedError, match=r"qualified table name .* is not allowed; use the bare table name"):
        _vet(sql)


@pytest.mark.parametrize(
    "sql",
    ["select 1; select 2", "select * from members; drop table members", "select * from members; select * from loans"],
)
def test_vet_query_two_statements_are_rejected(sql):
    with pytest.raises(SqlRejectedError, match="expected exactly one statement, found 2"):
        _vet(sql)


@pytest.mark.parametrize("sql", ["", "   ", "/* only a comment */"])
def test_vet_query_without_a_statement_is_rejected(sql):
    with pytest.raises(SqlRejectedError, match="expected exactly one statement, found 0"):
        _vet(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "create table copied as select * from members",
        "create or replace temp table copied as select 1",
        "create view everyone as select * from members",
        "create macro answer() as 42",
        "drop table members",
        "alter table members rename to people",
        "insert into members select * from members",
        "with extra as (select * from members) insert into members select * from extra",
        "update members set name = 'x'",
        "delete from members",
        "merge into members using loans on members.member_id = loans.member_id when matched then delete",
        "truncate table members",
        "copy members to 'members.csv'",
        "attach 'other.duckdb' as other",
        "detach other",
        "set threads = 1",
        "use ns_b",
        "pragma database_list",
        "install httpfs",
        "load httpfs",
        "call pragma_version()",
        "checkpoint",
        "vacuum",
        "begin transaction",
        "grant select on members to someone",
        "describe members",
        "show tables",
        "summarize members",
        "explain select * from members",
    ],
)
def test_vet_query_statement_that_is_not_a_select_is_rejected(sql):
    with pytest.raises(SqlRejectedError, match="only a single SELECT query is allowed"):
        _vet(sql)


def test_vet_query_select_into_is_rejected():
    with pytest.raises(SqlRejectedError, match="INTO is not allowed in a query"):
        _vet("select * into copied from members")


@pytest.mark.parametrize(
    ("sql", "function"),
    [
        ("select * from read_csv('members.csv')", "read_csv"),
        ("select * from read_csv_auto('members.csv')", "read_csv_auto"),
        ("select * from read_parquet('members.parquet')", "read_parquet"),
        ("select * from read_json_auto('members.json')", "read_json_auto"),
        ("select * from read_text('notes.txt')", "read_text"),
        ("select content from read_blob('notes.txt')", "read_blob"),
        ("select * from parquet_scan('members.parquet')", "parquet_scan"),
        ("select * from parquet_metadata('members.parquet')", "parquet_metadata"),
        ("select * from sniff_csv('members.csv')", "sniff_csv"),
        ("select * from glob('*')", "glob"),
        ("select * from sqlite_scan('other.db', 'members')", "sqlite_scan"),
        ("select * from postgres_scan('dsn', 'public', 'members')", "postgres_scan"),
        ("select * from iceberg_scan('warehouse/members')", "iceberg_scan"),
        ("select * from delta_scan('warehouse/members')", "delta_scan"),
        ("select (select count(*) from read_csv('members.csv'))", "read_csv"),
        ("select * from members where member_id in (select 1 from read_parquet('x'))", "read_parquet"),
    ],
)
def test_vet_query_file_reading_function_is_rejected(sql, function):
    with pytest.raises(SqlRejectedError, match=rf"function {function}\(\) is not available"):
        _vet(sql)


def test_vet_query_quoted_file_path_as_a_table_is_rejected():
    with pytest.raises(SqlRejectedError, match=r"unknown table 'members\.csv'"):
        _vet("select * from 'members.csv'")


@pytest.mark.parametrize(
    ("sql", "function"),
    [
        ("select * from duckdb_tables()", "duckdb_tables"),
        ("select * from duckdb_columns()", "duckdb_columns"),
        ("select * from duckdb_settings()", "duckdb_settings"),
        ("select * from pragma_table_info('members')", "pragma_table_info"),
        ("select current_setting('threads')", "current_setting"),
        ("select getenv('HOME')", "getenv"),
    ],
)
def test_vet_query_catalog_or_settings_function_is_rejected(sql, function):
    with pytest.raises(SqlRejectedError, match=rf"function {function}\(\) is not available"):
        _vet(sql)


@pytest.mark.parametrize(
    ("sql", "function"),
    [
        ("select * from query('select * from ns_b.orders')", "query"),
        ("select * from members, query_table('members')", "query_table"),
        ("select json_execute_serialized_sql('{}')", "json_execute_serialized_sql"),
        ("select json_serialize_sql('select 1')", "json_serialize_sql"),
        ("select json_deserialize_sql('{}')", "json_deserialize_sql"),
    ],
)
def test_vet_query_dynamic_sql_function_is_rejected(sql, function):
    with pytest.raises(SqlRejectedError, match=rf"function {function}\(\) is not available"):
        _vet(sql)


def test_vet_query_over_long_statement_is_rejected():
    sql = "select '" + "x" * MAX_SQL_CHARS + "'"

    with pytest.raises(SqlRejectedError, match=f"longer than {MAX_SQL_CHARS} characters"):
        _vet(sql)


@pytest.mark.parametrize("sql", ["select * from members where", "selec * from members", "select (1", "select * from"])
def test_vet_query_syntax_error_is_rejected(sql):
    with pytest.raises(SqlRejectedError, match="SQL syntax error") as rejection:
        _vet(sql)

    assert "\x1b" not in str(rejection.value)


# ---- dialects ----------------------------------------------------------------------------------------------------


def test_vet_query_snowflake_lateral_flatten_is_accepted():
    result = _vet("select f.value from members m, lateral flatten(input => m.tags) f", dialect="snowflake")

    assert result.tables == ("members",)
    assert "ns_a.members" in result.sql
    assert "FLATTEN" in result.sql


def test_vet_query_snowflake_identifier_function_is_rejected():
    with pytest.raises(SqlRejectedError):
        _vet("select * from identifier('ns_b.orders')", dialect="snowflake")


@pytest.mark.parametrize("sql", ["select * from @stage/members.csv", "select $1 from @stage", "select * from @~/x.csv"])
def test_vet_query_snowflake_stage_reference_is_rejected(sql):
    with pytest.raises(SqlRejectedError):
        _vet(sql, dialect="snowflake")


@pytest.mark.parametrize(
    "sql",
    ["select result_scan(last_query_id())", "select system$whitelist()", "select get_ddl('table', 'x')"],
)
def test_vet_query_snowflake_catalog_function_is_rejected(sql):
    with pytest.raises(SqlRejectedError, match="is not available"):
        _vet(sql, dialect="snowflake")


@pytest.mark.parametrize(
    "sql",
    [
        "select * from table('ns_b.orders')",
        "select * from table('reference_library.member_activity')",
        "select * from table(members)",
        "select * from table($1)",
        "select * from table(to_query('select * from ns_b.orders'))",
        "select * from table(result_scan(last_query_id()))",
        "select query_text from table(information_schema.query_history(result_limit => 100))",
        "select * from members m join table('ns_b.orders') o on o.member_id = m.member_id",
        "with other as (select * from table('ns_b.orders')) select * from other",
        "select (select count(*) from table('ns_b.orders')) from members",
        "select * from members, lateral to_query('select * from ns_b.orders')",
    ],
)
def test_vet_query_snowflake_table_keyword_cannot_name_a_relation_or_a_catalog_function(sql):
    with pytest.raises(SqlRejectedError, match="accept only these table functions"):
        _vet(sql, dialect="snowflake")


def test_vet_query_snowflake_function_called_through_a_schema_is_rejected():
    with pytest.raises(SqlRejectedError, match="qualified name"):
        _vet("select information_schema.current_task_graphs() from members", dialect="snowflake")


@pytest.mark.parametrize(
    "sql",
    [
        "select f.value from members m, table(flatten(input => m.tags)) f",
        "select seq4() from table(generator(rowcount => 3))",
        "select s.value from members m, table(split_to_table(m.name, ' ')) s",
    ],
)
def test_vet_query_snowflake_table_function_that_reads_no_relation_is_accepted(sql):
    assert _vet(sql, dialect="snowflake").sql


def test_vet_query_function_the_dialect_cannot_express_is_rejected():
    with pytest.raises(SqlRejectedError, match="not supported in the duckdb dialect"):
        _vet("select to_char(joined_on, 'YYYY-MM') from members")


def test_vet_query_function_the_dialect_can_express_is_kept():
    result = _vet("select to_char(joined_on, 'YYYY-MM') from members", dialect="snowflake")

    assert "TO_CHAR(joined_on, 'YYYY-MM')" in result.sql


def test_vet_query_duckdb_alternative_to_an_unsupported_function_is_accepted():
    result = _vet("select strftime(joined_on, '%Y-%m') from members")

    assert result.tables == ("members",)
    assert "STRFTIME(joined_on, '%Y-%m')" in result.sql
