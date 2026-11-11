"""The trusted side of one rollout.

A ``RolloutSession`` owns the rollout's workspace directory and warehouse namespace and carries out every tool
call. The policy supplies only the arguments. Each method returns a JSON-serializable result and reports a
failure as ``{"error": ...}`` so that the policy can read it and try again.
"""

import json
import logging
import math
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from elt_rlvr.extract_load.base import ConnectionRejectedError, ExtractLoadEngine, table_name_for_stream
from elt_rlvr.pipeline import (
    EXTRACT_LOAD_PATH,
    MODELS_DIRECTORY,
    ConnectionConfig,
    PipelineError,
    StreamConfig,
    parse_extract_load,
    read_models,
)
from elt_rlvr.sql_guard import SqlRejectedError, vet_query
from elt_rlvr.task import TaskSpec
from elt_rlvr.transform import run_models
from elt_rlvr.warehouse.base import IDENTIFIER_PATTERN, Warehouse, WarehouseError

logger = logging.getLogger(__name__)

UNCHANGED_NOTE = "already loaded with this configuration; name the connection in `connections` to load it again"


@dataclass(frozen=True)
class SessionLimits:
    """Bounds on what one rollout may ask of the environment."""

    max_file_bytes: int = 32_000
    max_read_lines: int = 200
    max_read_chars: int = 12_000
    max_query_rows: int = 50
    max_cell_chars: int = 200
    statement_timeout_seconds: float = 60.0
    max_syncs_per_source_table: int = 4
    """An episode may sync this many connections for every source table of its task."""


@dataclass
class ToolTiming:
    tool: str
    seconds: float


class ToolError(Exception):
    """A tool call that cannot be carried out. The message is returned to the policy."""


@dataclass
class RolloutSession:
    task: TaskSpec
    warehouse: Warehouse
    engine: ExtractLoadEngine
    workspace: Path
    limits: SessionLimits = field(default_factory=SessionLimits)
    namespace: str = field(default_factory=lambda: f"rollout_{uuid.uuid4().hex[:12]}")
    raw_tables: set[str] = field(default_factory=set)
    model_tables: set[str] = field(default_factory=set)
    timings: list[ToolTiming] = field(default_factory=list)
    extract_load_runs: int = 0
    connection_syncs: int = 0
    tool_errors: int = 0
    environment_errors: int = 0
    """Connections that failed because of the environment (Airbyte, the network), not the configuration."""
    table_writers: dict[str, set[tuple[str, str, str, str]]] = field(default_factory=dict)
    """For each raw table, by lower-cased name, the streams whose syncs its present rows come from. A sync that
    overwrites the table becomes its only writer; any other sync adds one."""
    submitted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    """Held while a tool call runs. The grader and ``close`` take it, so they wait for a call that is still
    running in its worker thread after the rollout was cut off by its time limit."""

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        shutil.copytree(self.task.workspace_dir, self.workspace, dirs_exist_ok=True)
        (self.workspace / MODELS_DIRECTORY).mkdir(parents=True, exist_ok=True)
        self.warehouse.create_namespace(self.namespace)

    def close(self) -> None:
        """Drop the rollout's namespace and delete its workspace."""
        with self.lock:
            self.warehouse.drop_namespace(self.namespace)
            shutil.rmtree(self.workspace, ignore_errors=True)

    def _call(self, tool: str, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            start = time.perf_counter()
            try:
                return action()
            except (ToolError, PipelineError, SqlRejectedError, WarehouseError) as error:
                self.tool_errors += 1
                return {"error": str(error)}
            finally:
                self.timings.append(ToolTiming(tool, time.perf_counter() - start))

    def _resolve(self, path: str) -> Path:
        if not isinstance(path, str) or "\x00" in path:
            raise ToolError("path must be a string")
        resolved = (self.workspace / path).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ToolError(f"{path!r} is outside the workspace")
        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def list_files(self, path: str = ".") -> dict[str, Any]:
        def action() -> dict[str, Any]:
            directory = self._resolve(path)
            if not directory.is_dir():
                raise ToolError(f"{path!r} is not a directory")
            entries = [
                {"path": self._relative(entry) + ("/" if entry.is_dir() else ""), "bytes": entry.stat().st_size}
                for entry in sorted(directory.iterdir())
            ]
            return {"entries": entries}

        return self._call("list_files", action)

    def read_file(self, path: str, start_line: int = 1, max_lines: int | None = None) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            file = self._resolve(path)
            if not file.is_file():
                raise ToolError(f"{path!r} is not a file")
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            first = max(start_line, 1)
            count = min(max_lines or self.limits.max_read_lines, self.limits.max_read_lines)
            text = "\n".join(lines[first - 1 : first - 1 + count])
            end_line = min(first - 1 + count, len(lines))
            if len(text) > self.limits.max_read_chars:
                text = text[: self.limits.max_read_chars]
                end_line = first - 1 + text.count("\n") + 1
            return {"path": path, "start_line": first, "end_line": end_line, "total_lines": len(lines), "text": text}

        return self._call("read_file", action)

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            file = self._resolve(path)
            relative = self._relative(file)
            is_model = file.parent == self.workspace / MODELS_DIRECTORY and file.suffix == ".sql"
            if relative != EXTRACT_LOAD_PATH and not is_model:
                raise ToolError(f"you can write only {EXTRACT_LOAD_PATH} and {MODELS_DIRECTORY}/<model>.sql")
            if is_model and not IDENTIFIER_PATTERN.match(file.stem):
                raise ToolError("a model file must be named <identifier>.sql")
            if not isinstance(content, str) or len(content.encode("utf-8")) > self.limits.max_file_bytes:
                raise ToolError(f"content must be text of at most {self.limits.max_file_bytes} bytes")
            if not content.strip():
                file.unlink(missing_ok=True)
                return {"path": relative, "deleted": True}
            file.write_text(content, encoding="utf-8")
            return {"path": relative, "bytes": len(content.encode("utf-8"))}

        return self._call("write_file", action)

    @staticmethod
    def _stream_key(connection: ConnectionConfig, stream: StreamConfig) -> tuple[str, str, str, str]:
        source = json.dumps([connection.source_type, connection.configuration], sort_keys=True, default=str)
        return (connection.name, source, stream.name, stream.sync_mode)

    @staticmethod
    def _table_key(stream: StreamConfig) -> str:
        return table_name_for_stream(stream.name).lower()

    def submitted_source_tables(self) -> set[str]:
        """Lower-cased names of the raw tables whose present rows the extract-load file, as it stands, describes.

        A table counts if every stream that wrote its present rows is still in the file with the same connection
        name, source configuration and sync mode. A table loaded by a stream that has since been removed from the
        file or changed in it does not count, whatever the table holds.
        """
        try:
            connections = parse_extract_load((self.workspace / EXTRACT_LOAD_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        described: dict[str, set[tuple[str, str, str, str]]] = {}
        for connection in connections:
            for stream in connection.streams:
                described.setdefault(self._table_key(stream), set()).add(self._stream_key(connection, stream))
        return {
            table for table, writers in self.table_writers.items() if writers and writers <= described.get(table, set())
        }

    def _sync(self, connection: ConnectionConfig, pending: ConnectionConfig) -> dict[str, Any]:
        """Sync the streams of ``pending`` and describe every stream of ``connection`` for the policy."""
        self.connection_syncs += 1
        try:
            results = self.engine.sync(pending, warehouse=self.warehouse, namespace=self.namespace)
        except ConnectionRejectedError as error:
            return {"connection": connection.name, "status": "failed", "error": str(error)}
        except Exception as error:
            # A fault of the environment. The call goes on, so that the policy still learns what the other
            # connections of this call loaded.
            # The exception text is logged and not returned: it may hold values the policy must not see.
            logger.exception("syncing connection %s of %s failed", connection.name, self.namespace)
            self.environment_errors += 1
            return {
                "connection": connection.name,
                "status": "failed",
                "error": f"the environment could not run this connection ({type(error).__name__}); try it again",
            }
        entries: dict[str, dict[str, Any]] = {}
        streams = {stream.name: stream for stream in pending.streams}
        for result in results:
            if result.succeeded:
                self.raw_tables.add(result.table)
                self.model_tables.discard(result.table)
                stream = streams[result.stream]
                key = self._stream_key(connection, stream)
                if stream.sync_mode == "full_refresh_overwrite":
                    self.table_writers[self._table_key(stream)] = {key}
                else:
                    self.table_writers.setdefault(self._table_key(stream), set()).add(key)
            entry = {"stream": result.stream, "table": result.table, "rows_loaded": result.rows_loaded}
            entry["status"] = "succeeded" if result.succeeded else "failed"
            if result.error:
                entry["error"] = result.error
            entries[result.stream] = entry
        for stream in connection.streams:
            if stream.name not in streams:
                entries[stream.name] = {
                    "stream": stream.name,
                    "table": table_name_for_stream(stream.name),
                    "status": "unchanged",
                }
        status = "succeeded" if all(result.succeeded for result in results) else "failed"
        ordered = [entries[stream.name] for stream in connection.streams if stream.name in entries]
        return {"connection": connection.name, "status": status, "streams": ordered}

    def run_extract_load(self, connections: list[str] | None = None) -> dict[str, Any]:
        """Apply the extract-load file.

        Without names, a stream is skipped when the rows its table holds now were written by a sync of that
        stream as the file describes it (same connection name, source configuration and sync mode), the way
        ``terraform apply`` leaves an unchanged resource alone. A retry after one connection failed therefore does
        not append the others a second time. Naming a connection syncs all of its streams again.
        """
        selected = connections

        def action() -> dict[str, Any]:
            config_file = self.workspace / EXTRACT_LOAD_PATH
            if not config_file.is_file():
                raise ToolError(f"{EXTRACT_LOAD_PATH} does not exist")
            connections = parse_extract_load(config_file.read_text(encoding="utf-8"))
            if selected:
                unknown = sorted(set(selected) - {connection.name for connection in connections})
                if unknown:
                    names = ", ".join(connection.name for connection in connections)
                    raise ToolError(f"no connection named: {', '.join(unknown)}; connections in the file: {names}")
                connections = [connection for connection in connections if connection.name in selected]
            plan: list[tuple[ConnectionConfig, ConnectionConfig | None]] = []
            for connection in connections:
                streams = tuple(
                    stream
                    for stream in connection.streams
                    if selected
                    or self._stream_key(connection, stream) not in self.table_writers.get(self._table_key(stream), ())
                )
                plan.append((connection, replace(connection, streams=streams) if streams else None))
            syncs = sum(pending is not None for _, pending in plan)
            budget = self.limits.max_syncs_per_source_table * len(self.task.source_tables)
            if self.connection_syncs + syncs > budget:
                raise ToolError(
                    f"an episode can run at most {budget} connection syncs; {self.connection_syncs} are used "
                    f"and this call would run {syncs}"
                )
            self.extract_load_runs += 1
            report = [
                (
                    {"connection": connection.name, "status": "unchanged", "note": UNCHANGED_NOTE}
                    if pending is None
                    else self._sync(connection, pending)
                )
                for connection, pending in plan
            ]
            row_counts = {table: self.warehouse.row_count(self.namespace, table) for table in sorted(self.raw_tables)}
            return {"connections": report, "source_tables": row_counts}

        return self._call("run_extract_load", action)

    def run_transform(self, models: list[str] | None = None) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            model_sql = read_models(self.workspace / MODELS_DIRECTORY)
            if not model_sql:
                raise ToolError(f"there are no model files in {MODELS_DIRECTORY}/")
            if models:
                unknown = sorted(set(models) - set(model_sql))
                if unknown:
                    raise ToolError(f"no model file for: {', '.join(unknown)}")
            results = run_models(
                self.warehouse,
                models=model_sql,
                raw_table_namespaces={table: self.namespace for table in self.raw_tables},
                target_namespace=self.namespace,
                timeout_seconds=self.limits.statement_timeout_seconds,
                only=models or None,
            )
            report = []
            for result in results:
                entry: dict[str, Any] = {"model": result.name}
                if result.succeeded:
                    self.model_tables.add(result.name)
                    entry.update(status="succeeded", rows=result.rows, columns=list(result.columns))
                else:
                    self.model_tables.discard(result.name)
                    entry.update(status="failed", error=result.error)
                report.append(entry)
            return {"models": report}

        return self._call("run_transform", action)

    def query(self, sql: str) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            if not isinstance(sql, str):
                raise ToolError("sql must be a string")
            tables = {table: self.namespace for table in self.raw_tables | self.model_tables}
            vetted = vet_query(sql, dialect=self.warehouse.dialect, table_namespaces=tables)
            result = self.warehouse.run_query(
                vetted.sql,
                max_rows=self.limits.max_query_rows,
                timeout_seconds=self.limits.statement_timeout_seconds,
            )
            rows = [[self._cell(value) for value in row] for row in result.rows]
            return {"columns": list(result.columns), "rows": rows, "truncated": result.truncated}

        return self._call("query", action)

    def submit(self) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            self.submitted = True
            return {"status": "submitted"}

        return self._call("submit", action)

    def _cell(self, value: Any) -> Any:
        if value is None or isinstance(value, bool | int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, Decimal | datetime | date):
            return str(value)
        text = value if isinstance(value, str) else str(value)
        return text if len(text) <= self.limits.max_cell_chars else text[: self.limits.max_cell_chars] + "..."
