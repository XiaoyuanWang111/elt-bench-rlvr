"""The destination interface.

Everything that depends on a particular data warehouse (connection, SQL dialect, identifier rules, bulk load,
sampling, hashing) sits behind :class:`Warehouse`. The environment, the tools and the grader call only this
interface, so a new destination is one new subclass plus one line in ``WAREHOUSES``.

A *namespace* is the unit of isolation: one schema that holds the raw tables and models of one rollout, of the
grader's replay, or of a task's reference load.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
KEY_MAP_TABLE, KEY_MAP_OLD, KEY_MAP_NEW = "_elt_rlvr_key_map", "_elt_rlvr_old_key", "_elt_rlvr_new_key"
"""Scratch table of ``rotate_keys`` and its two columns, named so that no source column has the same name."""


class WarehouseError(RuntimeError):
    """A statement failed in the warehouse. The message is safe to show to the policy."""


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: list[tuple]
    truncated: bool


@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    is_numeric: bool


@dataclass(frozen=True)
class TableFingerprint:
    """Row count and an order-insensitive digest of a table's rows."""

    row_count: int
    digest: str


def require_identifier(name: str) -> str:
    """Return ``name`` if it is a plain SQL identifier, else raise ``ValueError``."""
    if not IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"not a plain SQL identifier: {name!r}")
    return name


class Warehouse(ABC):
    """One open connection to a destination."""

    dialect: str
    """sqlglot dialect name used to parse and print SQL for this destination."""
    persistent: bool = False
    """Whether namespaces outlive this connection. A later process may then reuse what an earlier one loaded."""

    @abstractmethod
    def create_namespace(self, namespace: str) -> None:
        """Create an empty namespace, replacing one with the same name."""

    @abstractmethod
    def drop_namespace(self, namespace: str) -> None:
        """Drop a namespace and everything in it. A missing namespace is not an error."""

    @abstractmethod
    def list_tables(self, namespace: str) -> list[str]:
        """Names of the tables in ``namespace``, sorted."""

    @abstractmethod
    def describe(self, namespace: str, table: str) -> list[Column]:
        """Columns of ``namespace.table`` in table order."""

    @abstractmethod
    def run_query(self, sql: str, *, max_rows: int, timeout_seconds: float) -> QueryResult:
        """Run a vetted, fully qualified SELECT and return at most ``max_rows`` rows."""

    @abstractmethod
    def create_table_from_query(self, namespace: str, table: str, sql: str, *, timeout_seconds: float) -> int:
        """Materialize a vetted SELECT as ``namespace.table`` (replacing it) and return the number of rows written."""

    @abstractmethod
    def write_frame(self, namespace: str, table: str, frame: pd.DataFrame, *, append: bool) -> int:
        """Load ``frame`` into ``namespace.table`` and return the number of rows written.

        With ``append`` the rows are added to an existing table; otherwise the table is replaced.
        """

    @abstractmethod
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
        """Copy a table to another namespace and return the number of rows copied.

        With ``append`` the rows are added to an existing target table; otherwise the target is replaced.
        With ``sample_percent`` only a Bernoulli sample of the rows is copied.
        """

    @abstractmethod
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
        """Delete the rows of ``namespace.table`` whose parent row was removed.

        A row is deleted when its ``column`` value is present in ``original_namespace.parent_table`` and absent
        from ``namespace.parent_table``. Rows that had no parent to begin with are kept.
        """

    @abstractmethod
    def delete_children_of_one_parent(
        self, namespace: str, table: str, column: str, *, parent_table: str, parent_column: str, seed: int
    ) -> None:
        """Delete every row of ``namespace.table`` that refers to one parent row, chosen by ``seed``.

        The parent is chosen among the ``parent_column`` values of ``namespace.parent_table`` that some row of
        ``table`` refers to through ``column``. With fewer than two such values nothing is deleted, so the table
        is never emptied.
        """

    @abstractmethod
    def rotate_keys(
        self, namespace: str, table: str, column: str, *, references: Sequence[tuple[str, str]], seed: int
    ) -> None:
        """Move every value of ``namespace.table.column`` to another row, in an order chosen by ``seed``.

        Each distinct value is replaced by the next one in that order and the last by the first. The same
        replacement is made in every ``(table, column)`` of ``references``, so rows that referred to a row still
        refer to it. Afterwards a key value stands for a different row than before.
        """

    @abstractmethod
    def fingerprint(
        self, namespace: str, table: str, columns: Sequence[str], numeric_columns: Collection[str]
    ) -> TableFingerprint:
        """Digest the multiset of rows of ``columns``.

        Columns in ``numeric_columns`` are compared as floating-point numbers and the others as text, so two
        loads of the same data agree even when one stored an integer column as BIGINT and the other as DECIMAL.
        """

    @abstractmethod
    def read_frame(self, namespace: str, table: str) -> pd.DataFrame:
        """Fetch a whole table."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection."""

    def row_count(self, namespace: str, table: str) -> int:
        return self.fingerprint(namespace, table, [], []).row_count
