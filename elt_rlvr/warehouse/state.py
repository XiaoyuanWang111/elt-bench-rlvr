"""A small record stored inside a namespace, so that a later process can tell what the namespace holds.

A warehouse whose namespaces outlive the connection (``Warehouse.persistent``) keeps the reference load of a task
and the snapshots of finished syncs. The record says which inputs produced them; a process reuses a namespace only
when the record matches what it would have built itself.
"""

import json
from typing import Any

from elt_rlvr.warehouse.base import Warehouse, WarehouseError

STATE_TABLE = "elt_rlvr_state"
STATE_TIMEOUT_SECONDS = 60.0


def write_state(warehouse: Warehouse, namespace: str, state: dict[str, Any]) -> None:
    """Store ``state`` in ``namespace``, replacing an earlier record."""
    literal = json.dumps(state, sort_keys=True).replace("'", "''")
    warehouse.create_table_from_query(
        namespace, STATE_TABLE, f"SELECT '{literal}' AS state", timeout_seconds=STATE_TIMEOUT_SECONDS
    )


def read_state(warehouse: Warehouse, namespace: str) -> dict[str, Any] | None:
    """The record stored in ``namespace``, or ``None`` when the namespace or the record does not exist."""
    try:
        result = warehouse.run_query(
            f"SELECT state FROM {namespace}.{STATE_TABLE}", max_rows=1, timeout_seconds=STATE_TIMEOUT_SECONDS
        )
    except WarehouseError:
        return None
    return json.loads(result.rows[0][0]) if result.rows else None
