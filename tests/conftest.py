"""Fixtures: the local tasks written once per test session, and a fresh DuckDB warehouse per test."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from elt_rlvr.task import TaskSpec
from elt_rlvr.tasks.local import build_local_tasks
from elt_rlvr.warehouse import Warehouse, open_warehouse


@pytest.fixture(autouse=True)
def no_deployment_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the ELT_RLVR_* settings of the shell that runs the tests, so a deployment's values change no result."""
    for name in list(os.environ):
        if name.startswith("ELT_RLVR_"):
            monkeypatch.delenv(name)


@pytest.fixture(scope="session")
def local_tasks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, TaskSpec]:
    return {task.task_id: task for task in build_local_tasks(tmp_path_factory.mktemp("tasks"))}


@pytest.fixture
def warehouse() -> Iterator[Warehouse]:
    connection = open_warehouse("duckdb")
    yield connection
    connection.close()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"
