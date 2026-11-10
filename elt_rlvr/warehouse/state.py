"""A small record stored inside a namespace, so that a later process can tell what the namespace holds.

A warehouse whose namespaces outlive the connection (``Warehouse.persistent``) keeps the reference load of a task
and the snapshots of finished syncs. The record says which inputs produced them; a process reuses a namespace only
when the record matches what it would have built itself.
"""

import base64
import binascii
import json
from typing import Any

from elt_rlvr.warehouse.base import Warehouse, WarehouseError

STATE_TABLE = "elt_rlvr_state"
STATE_TIMEOUT_SECONDS = 60.0


def write_state(warehouse: Warehouse, namespace: str, state: dict[str, Any]) -> None:
    """Store ``state`` in ``namespace``, replacing an earlier record.

    The record is written as base64 text. Inside a SQL string literal, JSON's own escapes are not safe: Snowflake
    reads ``\\n`` in a literal as a line break, which turned a stored error message into invalid JSON.
    """
    literal = base64.b64encode(json.dumps(state, sort_keys=True).encode("utf-8")).decode("ascii")
    warehouse.create_table_from_query(
        namespace, STATE_TABLE, f"SELECT '{literal}' AS state", timeout_seconds=STATE_TIMEOUT_SECONDS
    )


def read_state(warehouse: Warehouse, namespace: str) -> dict[str, Any] | None:
    """The record stored in ``namespace``, or ``None`` when there is no namespace, no record or no readable one."""
    try:
        result = warehouse.run_query(
            f"SELECT state FROM {namespace}.{STATE_TABLE}", max_rows=1, timeout_seconds=STATE_TIMEOUT_SECONDS
        )
    except WarehouseError:
        return None
    if not result.rows:
        return None
    try:
        state = json.loads(base64.b64decode(str(result.rows[0][0]), validate=True))
    except (binascii.Error, ValueError):
        # Written by an earlier version or damaged: the caller builds the namespace again.
        return None
    return state if isinstance(state, dict) else None
