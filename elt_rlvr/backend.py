"""Build the warehouse and the extract-load engine a task runs on."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elt_rlvr.extract_load.base import ExtractLoadEngine
from elt_rlvr.extract_load.cache import CachingEngine
from elt_rlvr.extract_load.local import LocalEngine
from elt_rlvr.task import TaskSpec
from elt_rlvr.warehouse import Warehouse, open_warehouse


@dataclass(frozen=True)
class BackendConfig:
    """Which destination a run uses and how syncs are served."""

    warehouse: str = "duckdb"
    warehouse_options: dict[str, Any] = field(default_factory=dict)
    cache_syncs: bool = True
    """Serve a repeated sync of an identical connection from a snapshot instead of running it again."""


def build_engine(task: TaskSpec, *, cache_syncs: bool) -> ExtractLoadEngine:
    """The engine that reads ``task``'s sources."""
    if task.engine_kind == "local":
        engine: ExtractLoadEngine = LocalEngine(Path(task.engine_options["sources_root"]))
    elif task.engine_kind == "airbyte":
        # Imported on use: only runs against a live Airbyte deployment need it.
        from elt_rlvr.extract_load.airbyte import AirbyteEngine  # noqa: PLC0415

        engine = AirbyteEngine(**task.engine_options)
    else:
        raise ValueError(f"unknown engine kind {task.engine_kind!r}")
    return CachingEngine(engine) if cache_syncs else engine


def build_warehouse(config: BackendConfig) -> Warehouse:
    return open_warehouse(config.warehouse, **config.warehouse_options)
