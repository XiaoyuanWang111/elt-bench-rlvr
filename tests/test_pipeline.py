"""Parsing of the submitted pipeline files: the extract-load config and the model SQL files."""

import re
from dataclasses import dataclass

import pytest
import yaml

from elt_rlvr.pipeline import (
    MAX_CONNECTIONS,
    MAX_STREAMS_PER_CONNECTION,
    ConnectionConfig,
    PipelineError,
    StreamConfig,
    parse_extract_load,
    read_models,
    render_model_sql,
)


def _connection(name: str = "shop", streams: list | None = None, source: dict | None = None) -> dict:
    return {
        "name": name,
        "source": source or {"type": "sqlite", "configuration": {"database": "shop.sqlite"}},
        "streams": streams if streams is not None else [{"name": "orders", "sync_mode": "full_refresh_overwrite"}],
    }


def _document(*connections: dict) -> str:
    return yaml.safe_dump({"connections": list(connections)})


# ---- parse_extract_load ------------------------------------------------------------------------------------------


def test_parse_extract_load_valid_document_returns_every_connection():
    text = _document(
        _connection(
            "shop",
            streams=[
                {"name": "orders", "sync_mode": "full_refresh_overwrite"},
                {"name": "customers", "sync_mode": "full_refresh_append"},
            ],
        ),
        _connection(
            "reviews_file",
            streams=[{"name": "reviews", "sync_mode": "incremental_append"}],
            source={"type": "file", "configuration": {"dataset_name": "reviews", "format": "csv", "url": "r.csv"}},
        ),
    )

    assert parse_extract_load(text) == [
        ConnectionConfig(
            name="shop",
            source_type="sqlite",
            configuration={"database": "shop.sqlite"},
            streams=(StreamConfig("orders", "full_refresh_overwrite"), StreamConfig("customers", "full_refresh_append")),
        ),
        ConnectionConfig(
            name="reviews_file",
            source_type="file",
            configuration={"dataset_name": "reviews", "format": "csv", "url": "r.csv"},
            streams=(StreamConfig("reviews", "incremental_append"),),
        ),
    ]


def test_parse_extract_load_accepts_json_text():
    text = '{"connections": [{"name": "shop", "source": {"type": "sqlite", "configuration": {"database": "s.db"}},'
    text += ' "streams": [{"name": "orders", "sync_mode": "full_refresh_overwrite"}]}]}'

    assert [connection.name for connection in parse_extract_load(text)] == ["shop"]


def test_parse_extract_load_accepts_the_maximum_number_of_connections():
    connections = [_connection(f"source_{index}") for index in range(MAX_CONNECTIONS)]

    assert len(parse_extract_load(_document(*connections))) == MAX_CONNECTIONS


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("connections: [", "pipeline/extract_load.yaml is not valid YAML"),
        ("- shop\n- orders\n", "pipeline/extract_load.yaml must be a mapping"),
        ("", "pipeline/extract_load.yaml must be a mapping"),
        ("{}", "must have a non-empty list under 'connections'"),
        ("connections: []", "must have a non-empty list under 'connections'"),
        ("connections: shop", "must have a non-empty list under 'connections'"),
        (_document("shop"), "connections[0] must be a mapping"),
        (_document(_connection("my-shop")), "connections[0].name must be an identifier"),
        (_document(_connection("1shop")), "connections[0].name must be an identifier"),
        (_document(_connection("")), "connections[0].name must be an identifier"),
        (
            _document(_connection(source={"configuration": {"database": "s.db"}})),
            "connections[0].source.type must be a non-empty string",
        ),
        (
            _document(_connection(source={"type": "sqlite", "configuration": "s.db"})),
            "connections[0].source.configuration must be a mapping",
        ),
        (_document(_connection(streams=[])), "connections[0].streams must be a non-empty list"),
        (_document(_connection(streams=[{"sync_mode": "full_refresh_overwrite"}])), "streams[0].name must be"),
        (
            _document(_connection(streams=[{"name": "", "sync_mode": "full_refresh_overwrite"}])),
            "connections[0].streams[0].name must be a non-empty string",
        ),
        (
            _document(_connection(streams=[{"name": "orders", "sync_mode": "incremental"}])),
            "connections[0].streams[0].sync_mode must be one of: full_refresh_overwrite, full_refresh_append",
        ),
        (
            _document(_connection(streams=[{"name": "orders"}])),
            "connections[0].streams[0].sync_mode must be one of",
        ),
        (
            _document(
                _connection(
                    streams=[
                        {"name": "orders", "sync_mode": "full_refresh_overwrite"},
                        {"name": "Orders", "sync_mode": "full_refresh_append"},
                    ]
                )
            ),
            "connections[0].streams lists a stream more than once",
        ),
        (
            _document(
                _connection(
                    streams=[
                        {"name": f"stream_{index}", "sync_mode": "full_refresh_overwrite"}
                        for index in range(MAX_STREAMS_PER_CONNECTION + 1)
                    ]
                )
            ),
            f"connections[0].streams has more than {MAX_STREAMS_PER_CONNECTION} entries",
        ),
        (_document(_connection("shop"), _connection("SHOP")), "connection names must be unique"),
        (
            _document(*[_connection(f"source_{index}") for index in range(MAX_CONNECTIONS + 1)]),
            f"has more than {MAX_CONNECTIONS} connections",
        ),
    ],
)
def test_parse_extract_load_invalid_document_is_rejected_with_its_location(text, message):
    with pytest.raises(PipelineError, match=re.escape(message)):
        parse_extract_load(text)


def test_parse_extract_load_error_names_the_failing_connection_by_position():
    text = _document(_connection("shop"), _connection("orders-file"))

    with pytest.raises(PipelineError, match=re.escape("connections[1].name")):
        parse_extract_load(text)


# ---- render_model_sql --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "rendered"),
    [
        ("select * from {{ ref('orders') }}", "select * from orders"),
        ('select * from {{ref("orders")}}', "select * from orders"),
        ("select * from {{ source('raw', 'loans') }}", "select * from loans"),
        (
            'select * from {{ source("raw", "loans") }} l join {{ ref("members") }} m',
            "select * from loans l join members m",
        ),
        ("{{ config(materialized='table') }}\nselect 1 as x", "select 1 as x"),
        ("select 1 as x;", "select 1 as x"),
        ("select 1 as x;\n", "select 1 as x"),
        ("  select 1 as x ;  ", "select 1 as x"),
        ("select 'a;b' as x", "select 'a;b' as x"),
    ],
)
def test_render_model_sql_replaces_dbt_calls_and_drops_the_trailing_semicolon(text, rendered):
    assert render_model_sql(text) == rendered


# ---- read_models -------------------------------------------------------------------------------------------------


def test_read_models_returns_the_rendered_sql_of_every_model_file(tmp_path):
    (tmp_path / "orders_summary.sql").write_text("select * from {{ ref('orders') }};\n", encoding="utf-8")
    (tmp_path / "customers.sql").write_text("select 1 as id", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a model", encoding="utf-8")

    assert read_models(tmp_path) == {"customers": "select 1 as id", "orders_summary": "select * from orders"}


@pytest.mark.parametrize("file_name", ["1model.sql", "my-model.sql", "my model.sql", "model.v2.sql"])
def test_read_models_file_name_that_is_not_an_identifier_is_rejected(file_name, tmp_path):
    (tmp_path / file_name).write_text("select 1", encoding="utf-8")

    with pytest.raises(PipelineError, match=re.escape(f"model file name {file_name!r} must be <identifier>.sql")):
        read_models(tmp_path)


@dataclass(frozen=True, order=True)
class _MemoryFile:
    name: str
    text: str

    @property
    def stem(self) -> str:
        return self.name.removesuffix(".sql")

    def read_text(self, encoding: str) -> str:
        return self.text


@dataclass(frozen=True)
class _MemoryDirectory:
    files: tuple[_MemoryFile, ...]

    def is_dir(self) -> bool:
        return True

    def glob(self, pattern: str) -> list[_MemoryFile]:
        return list(self.files)


def test_read_models_names_that_differ_only_in_case_are_rejected():
    # Case-insensitive file systems cannot hold both files, so the directory is held in memory.
    directory = _MemoryDirectory((_MemoryFile("Summary.sql", "select 1"), _MemoryFile("summary.sql", "select 2")))

    with pytest.raises(PipelineError, match="two model files differ only in case"):
        read_models(directory)


def test_read_models_missing_directory_has_no_models(tmp_path):
    assert read_models(tmp_path / "models") == {}
