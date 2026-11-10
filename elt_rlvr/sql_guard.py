"""Vet model-written SQL before it reaches a warehouse.

The policy may only read tables in its own rollout namespace. A statement is parsed with sqlglot, checked against
an allowlist, and re-emitted from the syntax tree with every table qualified by that namespace. The warehouse runs
the re-emitted text, so a string that sqlglot and the warehouse would parse differently cannot smuggle anything
past the check.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, SqlglotError, UnsupportedError

MAX_SQL_CHARS = 20_000

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Into,
    exp.Command,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Copy,
    exp.Set,
    exp.Use,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Lock,
    exp.DynamicIdentifier,
)

# Table functions never read a named relation, so none of them can reach another namespace or a file.
ALLOWED_TABLE_FUNCTIONS = frozenset({"unnest", "generate_series", "range", "flatten", "explode"})

# Functions that read files, catalogs, settings or run SQL given as a string.
FORBIDDEN_FUNCTION_NAMES = frozenset(
    {
        "getenv",
        "glob",
        "query",
        "query_table",
        "identifier",
        "dynamic_identifier",
        "result_scan",
        "get_ddl",
        "get_presigned_url",
        "get_stage_location",
        "current_setting",
        "sniff_csv",
        "json_serialize_sql",
        "json_deserialize_sql",
        "json_execute_serialized_sql",
        "parquet_scan",
        "parquet_metadata",
        "parquet_schema",
        "table",
    }
)
FORBIDDEN_FUNCTION_PREFIXES = (
    "read_",
    "system$",
    "duckdb_",
    "pragma_",
    "pg_",
    "sqlite_",
    "postgres_",
    "mysql_",
    "iceberg_",
    "delta_",
)


class SqlRejectedError(ValueError):
    """The statement is not a read-only query over the rollout namespace."""


@dataclass(frozen=True)
class VettedQuery:
    """A query that passed the guard, re-emitted with qualified table names."""

    sql: str
    tables: tuple[str, ...]


def _function_name(function: exp.Func) -> str:
    name = function.name if isinstance(function, exp.Anonymous) else function.sql_name()
    return name.lower()


def _check_function(function: exp.Func) -> None:
    name = _function_name(function)
    if name in FORBIDDEN_FUNCTION_NAMES or name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
        raise SqlRejectedError(f"function {name}() is not available")


def _parse_single_query(sql: str, dialect: str) -> exp.Query:
    if len(sql) > MAX_SQL_CHARS:
        raise SqlRejectedError(f"statement is longer than {MAX_SQL_CHARS} characters")
    try:
        statements = [
            statement
            for statement in sqlglot.parse(sql, read=dialect)
            if statement is not None and not isinstance(statement, exp.Semicolon)
        ]
    except SqlglotError as error:
        raise SqlRejectedError(f"SQL syntax error: {error}") from error
    if len(statements) != 1:
        raise SqlRejectedError(f"expected exactly one statement, found {len(statements)}")
    root = statements[0]
    while isinstance(root, exp.Subquery) and not root.alias:
        root = root.this
    if not isinstance(root, exp.Select | exp.SetOperation):
        raise SqlRejectedError("only a single SELECT query is allowed")
    return root


def vet_query(sql: str, *, dialect: str, table_namespaces: Mapping[str, str]) -> VettedQuery:
    """Check that ``sql`` is one read-only query over known tables and qualify each table with its namespace.

    Args:
        sql: Statement written by the policy.
        dialect: sqlglot dialect of the warehouse.
        table_namespaces: The tables the query may read, mapped to the namespace each one lives in. Table names
            match case-insensitively. Every reference is rewritten to ``namespace.table``.

    Raises:
        SqlRejectedError: The statement is not a single SELECT, names a table outside ``table_namespaces``,
            qualifies a table name, or calls a forbidden function.
    """
    root = _parse_single_query(sql, dialect)

    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlRejectedError(f"{type(node).__name__.upper()} is not allowed in a query")
        if isinstance(node, exp.Func):
            _check_function(node)

    cte_names = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    canonical_names = {name.lower(): name for name in table_namespaces}
    referenced: list[str] = []

    for table in root.find_all(exp.Table):
        relation = table.this
        if isinstance(relation, exp.Func):
            if _function_name(relation) not in ALLOWED_TABLE_FUNCTIONS:
                raise SqlRejectedError(f"table function {_function_name(relation)}() is not available")
            continue
        if not isinstance(relation, exp.Identifier):
            raise SqlRejectedError("a table reference must be a plain table name")
        if table.args.get("catalog") or table.args.get("db"):
            raise SqlRejectedError(
                f"qualified table name {table.sql(dialect=dialect)!r} is not allowed; use the bare table name"
            )
        name = relation.name
        if name.lower() in cte_names:
            continue
        if name.lower() not in canonical_names:
            available = ", ".join(sorted(canonical_names.values())) or "(none)"
            raise SqlRejectedError(f"unknown table {name!r}; tables you can read: {available}")
        canonical_name = canonical_names[name.lower()]
        table.set("db", exp.to_identifier(table_namespaces[canonical_name]))
        if canonical_name not in referenced:
            referenced.append(canonical_name)

    try:
        # Without RAISE, sqlglot drops what it cannot express in the dialect (for example the format argument of
        # TO_CHAR in DuckDB) and the warehouse would run something other than what the policy wrote.
        sql = root.sql(dialect=dialect, unsupported_level=ErrorLevel.RAISE)
    except UnsupportedError as error:
        raise SqlRejectedError(f"not supported in the {dialect} dialect: {error}") from error
    return VettedQuery(sql=sql, tables=tuple(referenced))
