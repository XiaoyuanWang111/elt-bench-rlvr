"""The local extract-load engine on small SQLite, CSV, JSON Lines and document sources."""

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from elt_rlvr.extract_load.base import METADATA_COLUMN_PREFIX, ConnectionRejectedError, StreamResult
from elt_rlvr.extract_load.local import LocalEngine
from elt_rlvr.pipeline import ConnectionConfig, StreamConfig

NAMESPACE = "raw"
RAW_ID = f"{METADATA_COLUMN_PREFIX}raw_id"
EXTRACTED_AT = f"{METADATA_COLUMN_PREFIX}extracted_at"
METADATA_COLUMNS = [RAW_ID, EXTRACTED_AT]

SQLITE = ("sqlite", {"database": "shop.sqlite"})
CSV = ("file", {"dataset_name": "people", "format": "csv", "url": "people.csv"})
JSONL = ("file", {"dataset_name": "events", "format": "jsonl", "url": "events.jsonl"})
DOCUMENTS = ("documents", {"connection_string": "docstore://docs", "database": "store"})


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    root = tmp_path / "sources"
    (root / "docs" / "store").mkdir(parents=True)
    connection = sqlite3.connect(root / "shop.sqlite")
    try:
        connection.execute("create table items (id integer, label text)")
        connection.executemany("insert into items values (?, ?)", [(1, "a"), (2, "b"), (3, None)])
        connection.execute("create table suppliers (supplier_id integer)")
        connection.commit()
    finally:
        connection.close()
    (root / "people.csv").write_text("id,name\n1,Ada\n2,Ben\n", encoding="utf-8")
    (root / "people_semicolon.csv").write_text("id;name\n1;Ada\n2;Ben\n", encoding="utf-8")
    (root / "events.jsonl").write_text('{"id": 1, "kind": "a"}\n{"id": 2, "kind": "b"}\n', encoding="utf-8")
    reviews = [
        {"review_id": 1, "rating": {"verified": True, "stars": 4}, "tags": ["fast", "cheap"]},
        {"review_id": 2, "text": "fine"},
    ]
    (root / "docs" / "store" / "reviews.jsonl").write_text(
        "".join(json.dumps(review) + "\n" for review in reviews), encoding="utf-8"
    )
    return root


@pytest.fixture
def engine(sources: Path) -> LocalEngine:
    return LocalEngine(sources)


@pytest.fixture
def raw_warehouse(warehouse):
    warehouse.create_namespace(NAMESPACE)
    return warehouse


def _connection(source: tuple[str, dict], *streams: str, sync_mode: str = "full_refresh_overwrite") -> ConnectionConfig:
    source_type, configuration = source
    return ConnectionConfig("source", source_type, configuration, tuple(StreamConfig(s, sync_mode) for s in streams))


def _sync(engine: LocalEngine, warehouse, connection: ConnectionConfig) -> list[StreamResult]:
    return engine.sync(connection, warehouse=warehouse, namespace=NAMESPACE)


# ---- loading -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "stream", "rows", "columns"),
    [
        (SQLITE, "items", 3, ["id", "label"]),
        (CSV, "people", 2, ["id", "name"]),
        (JSONL, "events", 2, ["id", "kind"]),
        (DOCUMENTS, "reviews", 2, ["review_id", "rating", "tags", "text"]),
    ],
    ids=["sqlite", "csv", "jsonl", "documents"],
)
def test_sync_each_source_type_loads_one_table_per_stream(source, stream, rows, columns, engine, raw_warehouse):
    results = _sync(engine, raw_warehouse, _connection(source, stream))

    assert results == [StreamResult("source", stream, stream, succeeded=True, rows_loaded=rows)]
    assert raw_warehouse.row_count(NAMESPACE, stream) == rows
    assert [column.name for column in raw_warehouse.describe(NAMESPACE, stream)] == columns + METADATA_COLUMNS


def test_sync_sqlite_source_keeps_values_and_nulls(engine, raw_warehouse):
    _sync(engine, raw_warehouse, _connection(SQLITE, "items"))

    frame = raw_warehouse.read_frame(NAMESPACE, "items").sort_values("id")

    assert frame["id"].tolist() == [1, 2, 3]
    assert frame["label"].tolist()[:2] == ["a", "b"]
    assert pd.isna(frame["label"].tolist()[2])


def test_sync_adds_loader_metadata_to_every_row(engine, raw_warehouse):
    _sync(engine, raw_warehouse, _connection(SQLITE, "items"))

    frame = raw_warehouse.read_frame(NAMESPACE, "items")

    assert frame[RAW_ID].notna().all() and frame[RAW_ID].is_unique
    assert frame[EXTRACTED_AT].notna().all()


@pytest.mark.parametrize(
    ("sync_mode", "rows_after_two_syncs"), [("full_refresh_overwrite", 2), ("full_refresh_append", 4)]
)
def test_sync_overwrite_replaces_and_append_adds(sync_mode, rows_after_two_syncs, engine, raw_warehouse):
    connection = _connection(CSV, "people", sync_mode=sync_mode)

    _sync(engine, raw_warehouse, connection)
    second = _sync(engine, raw_warehouse, connection)

    assert second[0].rows_loaded == 2
    assert raw_warehouse.row_count(NAMESPACE, "people") == rows_after_two_syncs


def test_sync_appended_rows_get_new_raw_ids(engine, raw_warehouse):
    connection = _connection(CSV, "people", sync_mode="full_refresh_append")

    _sync(engine, raw_warehouse, connection)
    _sync(engine, raw_warehouse, connection)

    assert raw_warehouse.read_frame(NAMESPACE, "people")[RAW_ID].nunique() == 4


def test_sync_overwrite_after_append_resets_the_table(engine, raw_warehouse):
    _sync(engine, raw_warehouse, _connection(CSV, "people", sync_mode="full_refresh_append"))
    _sync(engine, raw_warehouse, _connection(CSV, "people", sync_mode="full_refresh_append"))
    _sync(engine, raw_warehouse, _connection(CSV, "people"))

    assert raw_warehouse.row_count(NAMESPACE, "people") == 2


@pytest.mark.parametrize("reader_options", ['{"sep": ";"}', {"sep": ";"}], ids=["json_text", "mapping"])
def test_sync_reader_options_set_the_csv_separator(reader_options, engine, raw_warehouse):
    configuration = {
        "dataset_name": "people",
        "format": "csv",
        "url": "people_semicolon.csv",
        "reader_options": reader_options,
    }

    _sync(engine, raw_warehouse, _connection(("file", configuration), "people"))

    frame = raw_warehouse.read_frame(NAMESPACE, "people").sort_values("id")
    assert frame[["id", "name"]].to_dict("records") == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Ben"}]


def test_sync_documents_store_nested_values_as_json_text_and_missing_fields_as_null(engine, raw_warehouse):
    _sync(engine, raw_warehouse, _connection(DOCUMENTS, "reviews"))

    rows = raw_warehouse.read_frame(NAMESPACE, "reviews").sort_values("review_id").to_dict("records")

    assert json.loads(rows[0]["rating"]) == {"stars": 4, "verified": True}
    assert json.loads(rows[0]["tags"]) == ["fast", "cheap"]
    assert pd.isna(rows[0]["text"])
    assert pd.isna(rows[1]["rating"]) and pd.isna(rows[1]["tags"])
    assert rows[1]["text"] == "fine"


def test_sync_documents_nested_values_can_be_read_with_json_functions(engine, raw_warehouse):
    _sync(engine, raw_warehouse, _connection(DOCUMENTS, "reviews"))

    result = raw_warehouse.run_query(
        f"select json_extract_string(rating, '$.stars') from {NAMESPACE}.reviews where review_id = 1",
        max_rows=1,
        timeout_seconds=10,
    )

    assert result.rows == [("4",)]


# ---- streams that cannot be loaded -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "available"),
    [(SQLITE, "items, suppliers"), (CSV, "people"), (DOCUMENTS, "reviews")],
    ids=["sqlite", "file", "documents"],
)
def test_sync_unknown_stream_fails_and_lists_the_available_streams(source, available, engine, raw_warehouse):
    results = _sync(engine, raw_warehouse, _connection(source, "missing"))

    assert results == [
        StreamResult(
            "source",
            "missing",
            "missing",
            succeeded=False,
            error=f"stream 'missing' not found in the source; available streams: {available}",
        )
    ]
    assert raw_warehouse.list_tables(NAMESPACE) == []


@pytest.mark.parametrize("sync_mode", ["incremental_append", "incremental_deduped_history"])
def test_sync_unsupported_sync_mode_fails_only_that_stream(sync_mode, engine, raw_warehouse):
    connection = ConnectionConfig(
        "source",
        "sqlite",
        {"database": "shop.sqlite"},
        (StreamConfig("items", "full_refresh_overwrite"), StreamConfig("suppliers", sync_mode)),
    )

    loaded, refused = _sync(engine, raw_warehouse, connection)

    assert loaded.succeeded
    assert not refused.succeeded
    assert refused.error == f"sync mode {sync_mode} is not supported by a sqlite source"
    assert raw_warehouse.list_tables(NAMESPACE) == ["items"]


# ---- rejected configurations -------------------------------------------------------------------------------------


def test_sync_unknown_source_type_is_rejected(engine, raw_warehouse):
    connection = _connection(("postgres", {"host": "db"}), "items")

    with pytest.raises(ConnectionRejectedError, match="unknown source type 'postgres'; choose one of: sqlite, file"):
        _sync(engine, raw_warehouse, connection)


@pytest.mark.parametrize(
    ("source_type", "configuration", "missing"),
    [
        ("sqlite", {}, "database"),
        ("sqlite", {"database": ""}, "database"),
        ("file", {"dataset_name": "people", "format": "csv"}, "url"),
        ("file", {"url": "people.csv"}, "dataset_name, format"),
        ("documents", {"connection_string": "docstore://docs"}, "database"),
    ],
)
def test_sync_missing_required_field_is_rejected(source_type, configuration, missing, engine, raw_warehouse):
    with pytest.raises(ConnectionRejectedError, match=re.escape(f"missing required field(s): {missing}")):
        _sync(engine, raw_warehouse, _connection((source_type, configuration), "people"))


@pytest.mark.parametrize(
    ("source", "extra", "allowed"),
    [
        (CSV, {"header": True}, "dataset_name, format, url, reader_options"),
        (SQLITE, {"password": "x", "host": "db"}, "database"),
        (DOCUMENTS, {"collection": "reviews"}, "connection_string, database"),
    ],
)
def test_sync_unknown_field_is_rejected_with_the_allowed_fields(source, extra, allowed, engine, raw_warehouse):
    source_type, configuration = source
    connection = _connection((source_type, {**configuration, **extra}), "people")

    expected = f"unknown field(s): {', '.join(sorted(extra))}; allowed fields: {allowed}"
    with pytest.raises(ConnectionRejectedError, match=re.escape(expected)):
        _sync(engine, raw_warehouse, connection)


@pytest.mark.parametrize(
    ("reader_options", "message"),
    [
        ("not json", "reader_options is not valid JSON"),
        ('{"delimiter": ";"}', "reader_options supports only the key 'sep'"),
        ({"sep": ";", "header": 0}, "reader_options supports only the key 'sep'"),
        ('[";"]', "reader_options supports only the key 'sep'"),
    ],
)
def test_sync_invalid_reader_options_are_rejected(reader_options, message, engine, raw_warehouse):
    configuration = {"dataset_name": "people", "format": "csv", "url": "people.csv", "reader_options": reader_options}

    with pytest.raises(ConnectionRejectedError, match=re.escape(message)):
        _sync(engine, raw_warehouse, _connection(("file", configuration), "people"))


def test_sync_separator_that_is_not_text_is_rejected(engine, raw_warehouse):
    configuration = {"dataset_name": "people", "format": "csv", "url": "people.csv", "reader_options": {"sep": 5}}

    with pytest.raises(ConnectionRejectedError):
        _sync(engine, raw_warehouse, _connection(("file", configuration), "people"))


@pytest.mark.parametrize(
    ("source_type", "configuration", "message"),
    [
        ("file", {"dataset_name": "people", "format": "parquet", "url": "people.csv"}, "format must be 'csv' or"),
        ("file", {"dataset_name": "people", "format": "csv", "url": "absent.csv"}, "url 'absent.csv' does not exist"),
        ("file", {"dataset_name": "people", "format": "csv", "url": "docs"}, "url 'docs' does not exist"),
        ("sqlite", {"database": "absent.sqlite"}, "database 'absent.sqlite' does not exist"),
        ("documents", {"connection_string": "docs", "database": "store"}, "must start with docstore://"),
        ("documents", {"connection_string": "docstore://docs", "database": "absent"}, "database 'absent' does not"),
    ],
)
def test_sync_configuration_that_names_no_source_is_rejected(source_type, configuration, message, engine, warehouse):
    with pytest.raises(ConnectionRejectedError, match=re.escape(message)):
        _sync(engine, warehouse, _connection((source_type, configuration), "people"))


@pytest.mark.parametrize(
    ("source_type", "configuration"),
    [
        ("file", {"dataset_name": "people", "format": "jsonl", "url": "people.csv"}),
        ("file", {"dataset_name": "people", "format": "csv", "url": "shop.sqlite"}),
        ("sqlite", {"database": "people.csv"}),
    ],
    ids=["csv_read_as_jsonl", "sqlite_read_as_csv", "csv_read_as_sqlite"],
)
def test_sync_source_whose_content_does_not_match_its_type_is_rejected(
    source_type, configuration, engine, raw_warehouse
):
    with pytest.raises(ConnectionRejectedError):
        _sync(engine, raw_warehouse, _connection((source_type, configuration), "people"))


def test_sync_variant_tells_apart_two_source_directories(tmp_path):
    connection = _connection(("file", {"dataset_name": "people", "format": "csv", "url": "people.csv"}), "people")

    first, second = LocalEngine(tmp_path / "first"), LocalEngine(tmp_path / "second")

    # The same configuration text names different files, so the sync cache must not serve one task's rows to another.
    assert first.sync_variant(connection) != second.sync_variant(connection)
