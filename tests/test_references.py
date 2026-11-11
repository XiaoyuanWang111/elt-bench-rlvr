"""The packaged reference pipelines of official tasks are well formed: they parse, their SQL passes the guard in the
Snowflake dialect, and their foreign keys name tables they load. A live run checks them against the benchmark."""

from importlib import resources
from pathlib import Path

import pytest
import yaml

from elt_rlvr.extract_load.airbyte import SOURCE_RESOURCES
from elt_rlvr.extract_load.base import table_name_for_stream
from elt_rlvr.pipeline import parse_extract_load, read_models
from elt_rlvr.sql_guard import vet_query
from elt_rlvr.task import parse_foreign_keys

REFERENCES = Path(str(resources.files("elt_rlvr.tasks") / "elt_bench_references"))
TASKS = sorted(path.name for path in REFERENCES.iterdir() if (path / "extract_load.yaml").is_file())


def _source_tables(task: str) -> set[str]:
    connections = parse_extract_load((REFERENCES / task / "extract_load.yaml").read_text(encoding="utf-8"))
    return {table_name_for_stream(stream.name) for connection in connections for stream in connection.streams}


def test_the_four_references_are_packaged():
    assert TASKS == ["app_store", "human_resources", "retails", "trains"]


@pytest.mark.parametrize("task", TASKS)
def test_reference_extract_load_uses_known_sources_and_placeholders_for_secrets(task):
    text = (REFERENCES / task / "extract_load.yaml").read_text(encoding="utf-8")
    connections = parse_extract_load(text)

    assert {connection.source_type for connection in connections} <= set(SOURCE_RESOURCES)
    for line in text.splitlines():
        if "password:" in line or "secret_access_key:" in line:
            assert "<secret:" in line, line


@pytest.mark.parametrize("task", TASKS)
def test_reference_models_pass_the_sql_guard_in_the_snowflake_dialect(task):
    models = read_models(REFERENCES / task / "models")
    tables = {table: "rollout" for table in _source_tables(task)} | {model: "rollout" for model in models}

    for name, sql in models.items():
        readable = {table: namespace for table, namespace in tables.items() if table != name}
        assert vet_query(sql, dialect="snowflake", table_namespaces=readable).sql


@pytest.mark.parametrize("task", [task for task in TASKS if (REFERENCES / task / "task.yaml").is_file()])
def test_reference_foreign_keys_name_loaded_tables(task):
    settings = yaml.safe_load((REFERENCES / task / "task.yaml").read_text(encoding="utf-8"))
    loaded = {table.lower() for table in _source_tables(task)}

    for key in parse_foreign_keys(settings["foreign_keys"]):
        assert {key.table.lower(), key.parent_table.lower()} <= loaded
