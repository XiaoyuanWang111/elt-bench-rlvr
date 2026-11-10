"""A credential-free extract-load engine over file-backed sources.

It stands in for Airbyte in local tests and fast training runs and keeps the parts of Airbyte's contract that
matter to a pipeline author: connections of typed sources with their own configuration schema, per-stream sync
modes (appending twice duplicates rows), one raw table per stream, and loader metadata columns.

Source types:

- ``sqlite``: a relational database file.
- ``file``: one CSV or JSON Lines file per stream.
- ``documents``: a document store; each collection holds JSON documents whose fields may differ and nest.

Every location in a connector configuration is resolved inside the task's source directory. A configuration that
points anywhere else (another directory, an absolute path, a URL) is rejected, so the engine cannot be used to
load a file that is not one of the task's sources.
"""

import json
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from elt_rlvr.extract_load.base import (
    METADATA_COLUMN_PREFIX,
    ConnectionRejectedError,
    ExtractLoadEngine,
    StreamResult,
    table_name_for_stream,
)
from elt_rlvr.pipeline import ConnectionConfig
from elt_rlvr.warehouse.base import Warehouse, WarehouseError

DOCUMENT_STORE_SCHEME = "docstore://"
SUPPORTED_SYNC_MODES = ("full_refresh_overwrite", "full_refresh_append")

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "sqlite": ("database",),
    "file": ("dataset_name", "format", "url"),
    "documents": ("connection_string", "database"),
}
OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "sqlite": (),
    "file": ("reader_options",),
    "documents": (),
}

StreamReader = Callable[[str], pd.DataFrame]


class StreamNotFoundError(LookupError):
    """The source has no stream of that name."""


def _check_fields(source_type: str, configuration: Mapping[str, Any]) -> None:
    required = REQUIRED_FIELDS[source_type]
    allowed = required + OPTIONAL_FIELDS[source_type]
    missing = [field for field in required if configuration.get(field) in (None, "")]
    if missing:
        raise ConnectionRejectedError(f"{source_type} source: missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(configuration) - set(allowed))
    if unknown:
        raise ConnectionRejectedError(
            f"{source_type} source: unknown field(s): {', '.join(unknown)}; allowed fields: {', '.join(allowed)}"
        )


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a table from JSON objects: one column per top-level field, nested values kept as JSON text."""
    flat = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else value
            for key, value in record.items()
        }
        for record in records
    ]
    return pd.DataFrame(flat).convert_dtypes()


def _read_json_lines(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ConnectionRejectedError(f"{path.name}: every line must be a JSON object")
    return _records_to_frame(records)


class LocalEngine(ExtractLoadEngine):
    """Syncs the file-backed sources under ``sources_root`` into a warehouse."""

    def __init__(self, sources_root: Path) -> None:
        self._sources_root = sources_root.resolve()

    def source_types(self) -> tuple[str, ...]:
        return tuple(REQUIRED_FIELDS)

    def _resolve(self, location: Any, *, what: str) -> Path:
        if not isinstance(location, str):
            raise ConnectionRejectedError(f"{what} must be a string")
        if "://" in location:
            raise ConnectionRejectedError(f"{what} {location!r} is not a source of this task")
        path = (self._sources_root / location).resolve()
        if not path.is_relative_to(self._sources_root):
            raise ConnectionRejectedError(f"{what} {location!r} is not a source of this task")
        return path

    def _sqlite_reader(self, configuration: Mapping[str, Any]) -> StreamReader:
        path = self._resolve(configuration["database"], what="database")
        if not path.is_file():
            raise ConnectionRejectedError(f"sqlite source: database {configuration['database']!r} does not exist")

        def read(stream: str) -> pd.DataFrame:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
                if stream not in tables:
                    raise StreamNotFoundError(f"available streams: {', '.join(sorted(tables))}")
                quoted = '"' + stream.replace('"', '""') + '"'
                return pd.read_sql_query(f"SELECT * FROM {quoted}", connection).convert_dtypes()
            finally:
                connection.close()

        return read

    def _file_reader(self, configuration: Mapping[str, Any]) -> StreamReader:
        path = self._resolve(configuration["url"], what="url")
        file_format = configuration["format"]
        if file_format not in ("csv", "jsonl"):
            raise ConnectionRejectedError("file source: format must be 'csv' or 'jsonl'")
        if not path.is_file():
            raise ConnectionRejectedError(f"file source: url {configuration['url']!r} does not exist")
        reader_options = configuration.get("reader_options") or {}
        if isinstance(reader_options, str):
            try:
                reader_options = json.loads(reader_options)
            except json.JSONDecodeError as error:
                raise ConnectionRejectedError(f"file source: reader_options is not valid JSON: {error}") from error
        if not isinstance(reader_options, Mapping) or set(reader_options) - {"sep"}:
            raise ConnectionRejectedError("file source: reader_options supports only the key 'sep'")
        dataset_name = configuration["dataset_name"]

        def read(stream: str) -> pd.DataFrame:
            if stream != dataset_name:
                raise StreamNotFoundError(f"available streams: {dataset_name}")
            if file_format == "jsonl":
                return _read_json_lines(path)
            return pd.read_csv(path, sep=reader_options.get("sep", ",")).convert_dtypes()

        return read

    def _documents_reader(self, configuration: Mapping[str, Any]) -> StreamReader:
        connection_string = configuration["connection_string"]
        if not isinstance(connection_string, str) or not connection_string.startswith(DOCUMENT_STORE_SCHEME):
            raise ConnectionRejectedError(f"documents source: connection_string must start with {DOCUMENT_STORE_SCHEME}")
        store = self._resolve(connection_string.removeprefix(DOCUMENT_STORE_SCHEME), what="connection_string")
        database = self._resolve(
            f"{store.relative_to(self._sources_root).as_posix()}/{configuration['database']}", what="database"
        )
        if not database.is_dir():
            raise ConnectionRejectedError(f"documents source: database {configuration['database']!r} does not exist")

        def read(stream: str) -> pd.DataFrame:
            collections = sorted(path.stem for path in database.glob("*.jsonl"))
            if stream not in collections:
                raise StreamNotFoundError(f"available streams: {', '.join(collections)}")
            return _read_json_lines(database / f"{stream}.jsonl")

        return read

    def _open_source(self, connection: ConnectionConfig) -> StreamReader:
        if connection.source_type not in REQUIRED_FIELDS:
            raise ConnectionRejectedError(
                f"unknown source type {connection.source_type!r}; choose one of: {', '.join(REQUIRED_FIELDS)}"
            )
        _check_fields(connection.source_type, connection.configuration)
        if connection.source_type == "sqlite":
            return self._sqlite_reader(connection.configuration)
        if connection.source_type == "file":
            return self._file_reader(connection.configuration)
        return self._documents_reader(connection.configuration)

    def sync(self, connection: ConnectionConfig, *, warehouse: Warehouse, namespace: str) -> list[StreamResult]:
        read = self._open_source(connection)
        sync_id = uuid.uuid4().hex
        extracted_at = datetime.now(UTC).replace(tzinfo=None)
        results: list[StreamResult] = []
        for stream in connection.streams:
            table = table_name_for_stream(stream.name)

            def failed(error: str, stream_name: str = stream.name, table_name: str = table) -> StreamResult:
                return StreamResult(connection.name, stream_name, table_name, succeeded=False, error=error)

            if stream.sync_mode not in SUPPORTED_SYNC_MODES:
                results.append(
                    failed(f"sync mode {stream.sync_mode} is not supported by a {connection.source_type} source")
                )
                continue
            try:
                frame = read(stream.name)
            except StreamNotFoundError as error:
                results.append(failed(f"stream {stream.name!r} not found in the source; {error}"))
                continue
            frame[f"{METADATA_COLUMN_PREFIX}raw_id"] = [f"{sync_id}-{index}" for index in range(len(frame))]
            frame[f"{METADATA_COLUMN_PREFIX}extracted_at"] = extracted_at
            try:
                rows = warehouse.write_frame(namespace, table, frame, append=stream.sync_mode == "full_refresh_append")
            except WarehouseError as error:
                results.append(failed(f"load failed: {error}"))
                continue
            results.append(StreamResult(connection.name, stream.name, table, succeeded=True, rows_loaded=rows))
        return results
