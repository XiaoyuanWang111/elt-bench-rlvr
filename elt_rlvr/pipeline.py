"""The pipeline a rollout submits: one extract-load config and one SQL file per transformation model.

```
pipeline/
  extract_load.yaml     connections: source type, connector configuration, streams and sync modes
  models/<name>.sql     one SELECT per model; the environment materializes it as table <name>
```
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elt_rlvr.warehouse.base import IDENTIFIER_PATTERN

EXTRACT_LOAD_PATH = "pipeline/extract_load.yaml"
MODELS_DIRECTORY = "pipeline/models"
SYNC_MODES = ("full_refresh_overwrite", "full_refresh_append", "incremental_append", "incremental_deduped_history")
MAX_CONNECTIONS = 32
MAX_STREAMS_PER_CONNECTION = 64

# dbt-style references, so that a model file also runs unchanged inside a dbt project.
_REF_PATTERN = re.compile(r"\{\{\s*ref\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\)\s*\}\}")
_SOURCE_PATTERN = re.compile(r"\{\{\s*source\(\s*['\"][^'\"]*['\"]\s*,\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\)\s*\}\}")
_CONFIG_PATTERN = re.compile(r"\{\{\s*config\([^}]*\)\s*\}\}")


class PipelineError(ValueError):
    """The submitted pipeline files are malformed. The message is shown to the policy."""


@dataclass(frozen=True)
class StreamConfig:
    name: str
    sync_mode: str


@dataclass(frozen=True)
class ConnectionConfig:
    """One source and the streams to copy from it into the destination."""

    name: str
    source_type: str
    configuration: Mapping[str, Any]
    streams: tuple[StreamConfig, ...]


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineError(f"{where} must be a mapping")
    return value


def _parse_stream(value: Any, where: str) -> StreamConfig:
    stream = _require_mapping(value, where)
    name = stream.get("name")
    if not isinstance(name, str) or not name:
        raise PipelineError(f"{where}.name must be a non-empty string")
    sync_mode = stream.get("sync_mode")
    if sync_mode not in SYNC_MODES:
        raise PipelineError(f"{where}.sync_mode must be one of: {', '.join(SYNC_MODES)}")
    return StreamConfig(name=name, sync_mode=sync_mode)


def _parse_connection(value: Any, where: str) -> ConnectionConfig:
    connection = _require_mapping(value, where)
    name = connection.get("name")
    if not isinstance(name, str) or not IDENTIFIER_PATTERN.match(name):
        raise PipelineError(f"{where}.name must be an identifier (letters, digits, underscore)")
    source = _require_mapping(connection.get("source"), f"{where}.source")
    source_type = source.get("type")
    if not isinstance(source_type, str) or not source_type:
        raise PipelineError(f"{where}.source.type must be a non-empty string")
    configuration = _require_mapping(source.get("configuration"), f"{where}.source.configuration")
    streams = connection.get("streams")
    if not isinstance(streams, list) or not streams:
        raise PipelineError(f"{where}.streams must be a non-empty list")
    if len(streams) > MAX_STREAMS_PER_CONNECTION:
        raise PipelineError(f"{where}.streams has more than {MAX_STREAMS_PER_CONNECTION} entries")
    parsed_streams = tuple(_parse_stream(stream, f"{where}.streams[{index}]") for index, stream in enumerate(streams))
    stream_names = [stream.name.lower() for stream in parsed_streams]
    if len(set(stream_names)) != len(stream_names):
        raise PipelineError(f"{where}.streams lists a stream more than once")
    return ConnectionConfig(name=name, source_type=source_type, configuration=configuration, streams=parsed_streams)


def parse_extract_load(text: str) -> list[ConnectionConfig]:
    """Parse and validate the text of ``pipeline/extract_load.yaml``."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PipelineError(f"{EXTRACT_LOAD_PATH} is not valid YAML: {error}") from error
    connections = _require_mapping(document, EXTRACT_LOAD_PATH).get("connections")
    if not isinstance(connections, list) or not connections:
        raise PipelineError(f"{EXTRACT_LOAD_PATH} must have a non-empty list under 'connections'")
    if len(connections) > MAX_CONNECTIONS:
        raise PipelineError(f"{EXTRACT_LOAD_PATH} has more than {MAX_CONNECTIONS} connections")
    parsed = [_parse_connection(connection, f"connections[{index}]") for index, connection in enumerate(connections)]
    names = [connection.name.lower() for connection in parsed]
    if len(set(names)) != len(names):
        raise PipelineError("connection names must be unique")
    return parsed


def render_model_sql(text: str) -> str:
    """Replace dbt ``ref``/``source`` calls with bare table names and drop ``config`` blocks."""
    text = _CONFIG_PATTERN.sub("", text)
    text = _REF_PATTERN.sub(r"\1", text)
    text = _SOURCE_PATTERN.sub(r"\1", text)
    return text.strip().rstrip(";").strip()


def read_models(directory: Path) -> dict[str, str]:
    """Read every ``<model>.sql`` in ``directory`` and return ``{model name: SELECT text}``."""
    if not directory.is_dir():
        return {}
    models: dict[str, str] = {}
    for path in sorted(directory.glob("*.sql")):
        if not IDENTIFIER_PATTERN.match(path.stem):
            raise PipelineError(f"model file name {path.name!r} must be <identifier>.sql")
        if path.stem.lower() in {name.lower() for name in models}:
            raise PipelineError(f"two model files differ only in case: {path.name!r}")
        models[path.stem] = render_model_sql(path.read_text(encoding="utf-8"))
    return models
