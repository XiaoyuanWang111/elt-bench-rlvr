"""What one ELT task consists of, split into what the policy sees and what only the grader sees."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATA_MODEL_FILE = "data_model.yaml"
_PLAIN_DESCRIPTION = re.compile(r"^(?P<key>\s*(?:-\s+)?description:\s+)(?P<text>[^\s'\"|>].*?)\s*$")


@dataclass(frozen=True)
class DataModelSpec:
    """A table the pipeline must build."""

    name: str
    columns: tuple[str, ...]
    """Required column names, from ``data_model.yaml``."""
    key: tuple[str, ...]
    """Columns that identify a row. The grader aligns rows on them; the policy is not told."""


@dataclass(frozen=True)
class ForeignKey:
    """``table.column`` refers to ``parent_table.parent_column``.

    The grader uses these to keep held-out samples consistent: when a sample drops a parent row it also drops the
    rows that refer to it, so a pipeline may rely on every reference that holds in the original data.
    """

    table: str
    column: str
    parent_table: str
    parent_column: str


@dataclass(frozen=True)
class TaskSpec:
    """One ELT task.

    ``workspace_dir`` is copied into the rollout's workspace. Every other path stays on the grader's side and is
    never reachable through a tool.
    """

    task_id: str
    workspace_dir: Path
    source_tables: tuple[str, ...]
    """Raw tables a correct pipeline loads."""
    data_models: tuple[DataModelSpec, ...]
    engine_kind: str
    """``local`` or ``airbyte``."""
    engine_options: Mapping[str, Any] = field(default_factory=dict)
    reference_dir: Path | None = None
    """Reference pipeline (``extract_load.yaml`` and ``models/``). Enables content checks of raw tables and
    grading on held-out data."""
    ground_truth_dir: Path | None = None
    """One ``<model>.csv`` per data model. When absent, expected outputs come from the reference pipeline."""
    expected_row_counts: Mapping[str, int] | None = None
    """Row count of each source table, used to check loads when there is no reference pipeline."""
    foreign_keys: tuple[ForeignKey, ...] = ()


def parse_foreign_keys(entries: list[Mapping[str, str]]) -> tuple[ForeignKey, ...]:
    """Parse entries of the form ``{table: loans, column: member_id, references: members.member_id}``."""
    keys = []
    for entry in entries:
        parent_table, parent_column = entry["references"].split(".")
        keys.append(ForeignKey(entry["table"], entry["column"], parent_table, parent_column))
    return tuple(keys)


def _parse_data_model(text: str) -> Any:
    """Parse ``data_model.yaml``, also when a description is not valid YAML.

    ELT-Bench writes descriptions as plain text. One of them (in ``world_development_indicators``) contains
    ``: ``, which YAML reads as a nested mapping and rejects. When the file does not parse, every plain
    description is quoted and the file is parsed again.
    """
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        lines = []
        for line in text.splitlines():
            match = _PLAIN_DESCRIPTION.match(line)
            lines.append(f"{match['key']}{json.dumps(match['text'])}" if match else line)
        return yaml.safe_load("\n".join(lines))


def read_data_models(workspace_dir: Path, keys: Mapping[str, list[str]]) -> tuple[DataModelSpec, ...]:
    """Read ``data_model.yaml`` (ELT-Bench format) and attach each model's row key."""
    document = _parse_data_model((workspace_dir / DATA_MODEL_FILE).read_text(encoding="utf-8"))
    lowered_keys = {name.lower(): tuple(columns) for name, columns in keys.items()}
    return tuple(
        DataModelSpec(
            name=model["name"],
            columns=tuple(column["name"] for column in model["columns"]),
            key=lowered_keys.get(model["name"].lower(), ()),
        )
        for model in document["models"]
    )


def load_local_task(task_dir: Path) -> TaskSpec:
    """Load a task laid out as ``workspace/``, ``sources/`` and ``grader/`` under ``task_dir``."""
    grader_dir = task_dir / "grader"
    settings = yaml.safe_load((grader_dir / "task.yaml").read_text(encoding="utf-8"))
    workspace_dir = task_dir / "workspace"
    return TaskSpec(
        task_id=task_dir.name,
        workspace_dir=workspace_dir,
        source_tables=tuple(settings["source_tables"]),
        data_models=read_data_models(workspace_dir, settings["model_keys"]),
        engine_kind="local",
        engine_options={"sources_root": str(task_dir / "sources")},
        reference_dir=grader_dir,
        foreign_keys=parse_foreign_keys(settings.get("foreign_keys", [])),
    )
