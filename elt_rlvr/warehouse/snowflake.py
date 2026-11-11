"""Snowflake destination.

Every namespace is a schema of one database. An administrative connection creates and drops the schemas, runs the
vetted queries and reads results; ownership of each new schema is granted to the loader role, the role Airbyte
writes with, as in ELT-Bench's own setup.

Identifiers are left unquoted so that Snowflake folds them to upper case the same way for every statement; column
names read back from the catalog are quoted exactly as stored.
"""

from collections.abc import Collection, Sequence
from typing import Any

import pandas as pd
import snowflake.connector
from snowflake.connector.errors import Error as SnowflakeError
from snowflake.connector.pandas_tools import write_pandas

from elt_rlvr.settings import destination_database
from elt_rlvr.warehouse.base import (
    KEY_MAP_NEW,
    KEY_MAP_OLD,
    KEY_MAP_TABLE,
    Column,
    QueryResult,
    TableFingerprint,
    Warehouse,
    WarehouseError,
    require_identifier,
)

# Type codes of the Snowflake connector's cursor description: 0 is FIXED (integers and decimals), 1 is REAL.
NUMERIC_TYPE_CODES = (0, 1)
INTERNAL_SCHEMA = "AIRBYTE_INTERNAL"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class SnowflakeWarehouse(Warehouse):
    """A Snowflake database whose schemas are the namespaces."""

    dialect = "snowflake"
    persistent = True

    def __init__(
        self,
        *,
        account: str,
        user: str,
        password: str | None = None,
        private_key_file: str | None = None,
        role: str | None = None,
        warehouse: str | None = None,
        database: str | None = None,
        loader_role: str | None = "AIRBYTE_ROLE",
        **connect_options: Any,
    ) -> None:
        self._database = require_identifier(database or destination_database())
        self._loader_role = require_identifier(loader_role) if loader_role else None
        credentials: dict[str, Any] = (
            {"private_key_file": private_key_file} if private_key_file else {"password": password}
        )
        self._connection = snowflake.connector.connect(
            account=account, user=user, role=role, warehouse=warehouse, **credentials, **connect_options
        )
        self._run(f"CREATE DATABASE IF NOT EXISTS {self._database}")
        self._run(f"USE DATABASE {self._database}")
        if self._loader_role:
            # Airbyte keeps its raw tables in one internal schema of the destination database.
            self._run(f"GRANT USAGE ON DATABASE {self._database} TO ROLE {self._loader_role}")
            self._run(f"CREATE SCHEMA IF NOT EXISTS {self._database}.{INTERNAL_SCHEMA}")
            self._run(f"GRANT OWNERSHIP ON SCHEMA {self._database}.{INTERNAL_SCHEMA} TO ROLE {self._loader_role}")

    def _run(self, sql: str, *, timeout_seconds: float | None = None) -> Any:
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql, timeout=int(timeout_seconds) if timeout_seconds else None)
        except SnowflakeError as error:
            cursor.close()
            raise WarehouseError(str(error)) from error
        return cursor

    def _schema(self, namespace: str) -> str:
        return f"{self._database}.{require_identifier(namespace)}"

    def _table(self, namespace: str, table: str) -> str:
        return f"{self._schema(namespace)}.{require_identifier(table)}"

    def create_namespace(self, namespace: str) -> None:
        self._run(f"CREATE OR REPLACE SCHEMA {self._schema(namespace)}")
        if self._loader_role:
            self._run(
                f"GRANT OWNERSHIP ON SCHEMA {self._schema(namespace)} TO ROLE {self._loader_role} COPY CURRENT GRANTS"
            )

    def drop_namespace(self, namespace: str) -> None:
        self._run(f"DROP SCHEMA IF EXISTS {self._schema(namespace)} CASCADE")

    def list_tables(self, namespace: str) -> list[str]:
        cursor = self._run(
            f"SELECT table_name FROM {self._database}.information_schema.tables "
            f"WHERE table_schema = '{require_identifier(namespace).upper()}' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        return [row[0] for row in cursor.fetchall()]

    def describe(self, namespace: str, table: str) -> list[Column]:
        cursor = self._run(f"SELECT * FROM {self._table(namespace, table)} LIMIT 0")
        return [
            Column(name=column.name, data_type=str(column.type_code), is_numeric=column.type_code in NUMERIC_TYPE_CODES)
            for column in cursor.description
        ]

    def run_query(self, sql: str, *, max_rows: int, timeout_seconds: float) -> QueryResult:
        cursor = self._run(sql, timeout_seconds=timeout_seconds)
        columns = tuple(column.name for column in cursor.description or ())
        rows = cursor.fetchmany(max_rows + 1)
        return QueryResult(columns=columns, rows=rows[:max_rows], truncated=len(rows) > max_rows)

    def create_table_from_query(self, namespace: str, table: str, sql: str, *, timeout_seconds: float) -> int:
        self._run(f"CREATE OR REPLACE TABLE {self._table(namespace, table)} AS {sql}", timeout_seconds=timeout_seconds)
        return self.row_count(namespace, table)

    def write_frame(self, namespace: str, table: str, frame: pd.DataFrame, *, append: bool) -> int:
        try:
            write_pandas(
                self._connection,
                frame,
                require_identifier(table).upper(),
                database=self._database,
                schema=require_identifier(namespace).upper(),
                auto_create_table=True,
                overwrite=not append,
                quote_identifiers=False,
            )
        except SnowflakeError as error:
            raise WarehouseError(str(error)) from error
        return len(frame)

    def copy_table(
        self,
        source_namespace: str,
        table: str,
        target_namespace: str,
        *,
        append: bool = False,
        sample_percent: float | None = None,
        seed: int | None = None,
    ) -> int:
        source, target = self._table(source_namespace, table), self._table(target_namespace, table)
        if append and table.lower() in {name.lower() for name in self.list_tables(target_namespace)}:
            before = self.row_count(target_namespace, table)
            self._run(f"INSERT INTO {target} SELECT * FROM {source}")
            return self.row_count(target_namespace, table) - before
        if sample_percent is None:
            # A clone shares the source table's storage, so a cached sync is served without copying data.
            self._run(f"CREATE OR REPLACE TABLE {target} CLONE {source}")
        else:
            self._run(
                f"CREATE OR REPLACE TABLE {target} AS "
                f"SELECT * FROM {source} SAMPLE BERNOULLI ({float(sample_percent):g}) SEED ({int(seed or 0)})"
            )
        return self.row_count(target_namespace, table)

    def delete_orphans(
        self,
        namespace: str,
        table: str,
        column: str,
        *,
        parent_table: str,
        parent_column: str,
        original_namespace: str,
    ) -> None:
        key, parent_key = _quote(column), _quote(parent_column)
        kept, original = self._table(namespace, parent_table), self._table(original_namespace, parent_table)
        self._run(
            f"DELETE FROM {self._table(namespace, table)} WHERE {key} IS NOT NULL "
            f"AND {key} IN (SELECT {parent_key} FROM {original} WHERE {parent_key} IS NOT NULL) "
            f"AND {key} NOT IN (SELECT {parent_key} FROM {kept} WHERE {parent_key} IS NOT NULL)"
        )

    def delete_children_of_one_parent(
        self, namespace: str, table: str, column: str, *, parent_table: str, parent_column: str, seed: int
    ) -> None:
        child, parent = self._table(namespace, table), self._table(namespace, parent_table)
        key, parent_key = _quote(column), _quote(parent_column)
        self._run(
            f"DELETE FROM {child} WHERE {key} IN ("
            "SELECT referred FROM ("
            f"SELECT referred, ROW_NUMBER() OVER (ORDER BY HASH(referred, {int(seed)}), referred) AS choice, "
            "COUNT(*) OVER () AS parents "
            f"FROM (SELECT DISTINCT {key} AS referred FROM {child} "
            f"WHERE {key} IN (SELECT {parent_key} FROM {parent} WHERE {parent_key} IS NOT NULL))"
            ") WHERE choice = 1 AND parents > 1)"
        )

    def rotate_keys(
        self, namespace: str, table: str, column: str, *, references: Sequence[tuple[str, str]], seed: int
    ) -> None:
        mapping = self._table(namespace, KEY_MAP_TABLE)
        order = f"ORDER BY HASH(old_key, {int(seed)}), old_key"
        self._run(
            f"CREATE OR REPLACE TABLE {mapping} AS SELECT old_key AS {KEY_MAP_OLD}, "
            f"COALESCE(LEAD(old_key) OVER ({order}), FIRST_VALUE(old_key) OVER ({order})) AS {KEY_MAP_NEW} "
            f"FROM (SELECT DISTINCT {_quote(column)} AS old_key FROM {self._table(namespace, table)} "
            f"WHERE {_quote(column)} IS NOT NULL)"
        )
        for target, key in [*references, (table, column)]:
            self._run(
                f"UPDATE {self._table(namespace, target)} SET {_quote(key)} = m.{KEY_MAP_NEW} "
                f"FROM {mapping} AS m WHERE {_quote(key)} = m.{KEY_MAP_OLD}"
            )
        self._run(f"DROP TABLE {mapping}")

    def fingerprint(
        self, namespace: str, table: str, columns: Sequence[str], numeric_columns: Collection[str]
    ) -> TableFingerprint:
        numeric = {name.lower() for name in numeric_columns}
        terms = [
            f"CAST({_quote(name)} AS {'DOUBLE' if name.lower() in numeric else 'VARCHAR'})"
            for name in sorted(columns, key=str.lower)
        ]
        row_hash = f"HASH({', '.join(terms)})" if terms else "0"
        cursor = self._run(
            "SELECT COUNT(*), COALESCE(BITXOR_AGG(row_hash), 0), COALESCE(SUM(row_hash), 0) "
            f"FROM (SELECT {row_hash} AS row_hash FROM {self._table(namespace, table)})"
        )
        row_count, xor_digest, sum_digest = cursor.fetchone()
        return TableFingerprint(row_count=int(row_count), digest=f"{int(xor_digest):x}-{int(sum_digest):x}")

    def read_frame(self, namespace: str, table: str) -> pd.DataFrame:
        return self._run(f"SELECT * FROM {self._table(namespace, table)}").fetch_pandas_all()

    def close(self) -> None:
        self._connection.close()
