"""DuckDB destination: an in-process warehouse that needs no credentials."""

import threading
from collections.abc import Collection, Sequence

import duckdb
import pandas as pd

from elt_rlvr.warehouse.base import Column, QueryResult, TableFingerprint, Warehouse, WarehouseError, require_identifier

NUMERIC_TYPE_PREFIXES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class DuckDBWarehouse(Warehouse):
    """An in-memory DuckDB database.

    External access is switched off and the configuration is locked, so no SQL statement can read or write a file,
    attach another database or load an extension, whatever the SQL guard let through.
    """

    dialect = "duckdb"

    def __init__(self) -> None:
        self._connection = duckdb.connect(":memory:", config={"enable_external_access": "false"})
        self._connection.execute("SET lock_configuration = true")

    def _execute(self, sql: str, *, timeout_seconds: float | None = None) -> duckdb.DuckDBPyConnection:
        cursor = self._connection.cursor()
        timer = threading.Timer(timeout_seconds, cursor.interrupt) if timeout_seconds else None
        if timer:
            timer.start()
        try:
            cursor.execute(sql)
        except duckdb.InterruptException as error:
            raise WarehouseError(f"statement exceeded the {timeout_seconds:g} second limit") from error
        except duckdb.Error as error:
            raise WarehouseError(str(error)) from error
        finally:
            if timer:
                timer.cancel()
        return cursor

    def create_namespace(self, namespace: str) -> None:
        require_identifier(namespace)
        self.drop_namespace(namespace)
        self._execute(f"CREATE SCHEMA {_quote(namespace)}")

    def drop_namespace(self, namespace: str) -> None:
        require_identifier(namespace)
        self._execute(f"DROP SCHEMA IF EXISTS {_quote(namespace)} CASCADE")

    def list_tables(self, namespace: str) -> list[str]:
        cursor = self._execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{require_identifier(namespace)}' ORDER BY table_name"
        )
        return [row[0] for row in cursor.fetchall()]

    def describe(self, namespace: str, table: str) -> list[Column]:
        cursor = self._execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_schema = '{require_identifier(namespace)}' "
            f"AND lower(table_name) = lower('{require_identifier(table)}') ORDER BY ordinal_position"
        )
        return [
            Column(name=name, data_type=data_type, is_numeric=data_type.upper().startswith(NUMERIC_TYPE_PREFIXES))
            for name, data_type in cursor.fetchall()
        ]

    def run_query(self, sql: str, *, max_rows: int, timeout_seconds: float) -> QueryResult:
        cursor = self._execute(sql, timeout_seconds=timeout_seconds)
        columns = tuple(description[0] for description in cursor.description or ())
        rows = cursor.fetchmany(max_rows + 1)
        return QueryResult(columns=columns, rows=rows[:max_rows], truncated=len(rows) > max_rows)

    def create_table_from_query(self, namespace: str, table: str, sql: str, *, timeout_seconds: float) -> int:
        target = f"{_quote(require_identifier(namespace))}.{_quote(require_identifier(table))}"
        self._execute(f"CREATE OR REPLACE TABLE {target} AS {sql}", timeout_seconds=timeout_seconds)
        return self.row_count(namespace, table)

    def write_frame(self, namespace: str, table: str, frame: pd.DataFrame, *, append: bool) -> int:
        target = f"{_quote(require_identifier(namespace))}.{_quote(require_identifier(table))}"
        cursor = self._connection.cursor()
        cursor.register("incoming_frame", frame)
        try:
            if append and table.lower() in {name.lower() for name in self.list_tables(namespace)}:
                cursor.execute(f"INSERT INTO {target} BY NAME SELECT * FROM incoming_frame")
            else:
                cursor.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM incoming_frame")
        except duckdb.Error as error:
            raise WarehouseError(str(error)) from error
        finally:
            cursor.unregister("incoming_frame")
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
        source = f"{_quote(require_identifier(source_namespace))}.{_quote(require_identifier(table))}"
        target = f"{_quote(require_identifier(target_namespace))}.{_quote(table)}"
        sample = ""
        if sample_percent is not None:
            sample = f" USING SAMPLE {float(sample_percent):g} PERCENT (bernoulli, {int(seed or 0)})"
        select = f"SELECT * FROM {source}{sample}"
        if append and table.lower() in {name.lower() for name in self.list_tables(target_namespace)}:
            before = self.row_count(target_namespace, table)
            self._execute(f"INSERT INTO {target} BY NAME {select}")
            return self.row_count(target_namespace, table) - before
        self._execute(f"CREATE OR REPLACE TABLE {target} AS {select}")
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
        child = f"{_quote(require_identifier(namespace))}.{_quote(require_identifier(table))}"
        kept = f"{_quote(namespace)}.{_quote(require_identifier(parent_table))}"
        original = f"{_quote(require_identifier(original_namespace))}.{_quote(parent_table)}"
        key, parent_key = _quote(column), _quote(parent_column)
        self._execute(
            f"DELETE FROM {child} WHERE {key} IS NOT NULL "
            f"AND {key} IN (SELECT {parent_key} FROM {original} WHERE {parent_key} IS NOT NULL) "
            f"AND {key} NOT IN (SELECT {parent_key} FROM {kept} WHERE {parent_key} IS NOT NULL)"
        )

    def fingerprint(
        self, namespace: str, table: str, columns: Sequence[str], numeric_columns: Collection[str]
    ) -> TableFingerprint:
        source = f"{_quote(require_identifier(namespace))}.{_quote(require_identifier(table))}"
        numeric = {name.lower() for name in numeric_columns}
        terms = [
            f"CAST({_quote(name)} AS {'DOUBLE' if name.lower() in numeric else 'VARCHAR'})"
            for name in sorted(columns, key=str.lower)
        ]
        row_hash = f"hash({', '.join(terms)})" if terms else "0"
        cursor = self._execute(
            "SELECT count(*), coalesce(bit_xor(row_hash), 0), coalesce(sum(row_hash::HUGEINT), 0) "
            f"FROM (SELECT {row_hash} AS row_hash FROM {source})"
        )
        row_count, xor_digest, sum_digest = cursor.fetchone()
        return TableFingerprint(row_count=int(row_count), digest=f"{int(xor_digest):x}-{int(sum_digest):x}")

    def read_frame(self, namespace: str, table: str) -> pd.DataFrame:
        source = f"{_quote(require_identifier(namespace))}.{_quote(require_identifier(table))}"
        return self._execute(f"SELECT * FROM {source}").fetchdf()

    def close(self) -> None:
        self._connection.close()
