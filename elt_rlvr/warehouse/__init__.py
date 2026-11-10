"""Destinations. Add a warehouse by subclassing :class:`Warehouse` and registering it in ``WAREHOUSES``."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from elt_rlvr.warehouse.base import Column, QueryResult, TableFingerprint, Warehouse, WarehouseError
from elt_rlvr.warehouse.duckdb import DuckDBWarehouse


def _open_duckdb(**_: Any) -> Warehouse:
    return DuckDBWarehouse()


def _open_snowflake(*, credential_file: str, **options: Any) -> Warehouse:
    """Open Snowflake with the connection values in ``credential_file`` (ELT-Bench's ``snowflake_credential.json``)."""
    # Imported on use: the Snowflake driver is an optional dependency.
    from elt_rlvr.warehouse.snowflake import SnowflakeWarehouse  # noqa: PLC0415

    credentials = json.loads(Path(credential_file).read_text(encoding="utf-8"))
    return SnowflakeWarehouse(**credentials, **options)


WAREHOUSES: dict[str, Callable[..., Warehouse]] = {
    "duckdb": _open_duckdb,
    "snowflake": _open_snowflake,
}


def open_warehouse(kind: str, **config: Any) -> Warehouse:
    """Open a connection to the destination named ``kind``."""
    if kind not in WAREHOUSES:
        raise ValueError(f"unknown warehouse {kind!r}; choose one of: {', '.join(sorted(WAREHOUSES))}")
    return WAREHOUSES[kind](**config)


__all__ = [
    "WAREHOUSES",
    "Column",
    "QueryResult",
    "TableFingerprint",
    "Warehouse",
    "WarehouseError",
    "open_warehouse",
]
