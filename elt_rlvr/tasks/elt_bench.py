"""Official ELT-Bench tasks, read from a checkout of the benchmark.

The checkout must already hold what the benchmark's own setup produces: the per-task input bundles written by
``setup/write_config.py`` (``inputs/<task>/``) and, for grading against the published tables, the ground truth
downloaded from Hugging Face.

A task's workspace is derived from its input bundle. It keeps ``data_model.yaml``, the source schemas and the
source and connection documentation, and it changes ``config.yaml`` in two ways: secrets become placeholders, and
the Airbyte and destination sections are removed because the environment owns the Airbyte client and the
destination.
"""

import json
import shutil
from importlib import resources
from pathlib import Path

import yaml

from elt_rlvr.backend import BackendConfig
from elt_rlvr.extract_load.airbyte import INPUT_DIRECTORIES, redact_secrets
from elt_rlvr.settings import process_directory
from elt_rlvr.task import TaskSpec, parse_foreign_keys, read_data_models

WORKSPACE_DOCUMENTS = (
    "connection.md",
    "source_custom_api.md",
    "source_file.md",
    "source_mongodb_v2.md",
    "source_postgres.md",
    "source_s3.md",
)


def reference_directory(task_id: str) -> Path | None:
    """The packaged reference pipeline of an official task, if this repository ships one."""
    directory = Path(str(resources.files("elt_rlvr.tasks") / "elt_bench_references" / task_id))
    return directory if (directory / "extract_load.yaml").is_file() else None


def build_workspace(inputs_dir: Path, workspace_dir: Path, destination: str) -> None:
    """Write the policy-visible workspace of one official task."""
    shutil.rmtree(workspace_dir, ignore_errors=True)
    (workspace_dir / "documentation").mkdir(parents=True)
    config = yaml.safe_load((inputs_dir / "config.yaml").read_text(encoding="utf-8"))
    visible = {key: value for key, value in config.items() if key not in ("Airbyte", destination)}
    visible = {"destination": {"warehouse": destination}, **redact_secrets(visible)}
    (workspace_dir / "config.yaml").write_text(yaml.safe_dump(visible, sort_keys=False), encoding="utf-8")
    shutil.copy(inputs_dir / "data_model.yaml", workspace_dir / "data_model.yaml")
    shutil.copytree(inputs_dir / "schemas", workspace_dir / "schemas")
    for name in WORKSPACE_DOCUMENTS:
        if (inputs_dir / "documentation" / name).is_file():
            shutil.copy(inputs_dir / "documentation" / name, workspace_dir / "documentation" / name)
    pipeline_guide = resources.files("elt_rlvr.tasks") / "elt_bench_documentation" / "pipeline.md"
    (workspace_dir / "documentation" / "pipeline.md").write_text(
        pipeline_guide.read_text(encoding="utf-8"), encoding="utf-8"
    )


def load_elt_bench_task(
    elt_bench_root: Path,
    task_id: str,
    *,
    destination: str = "snowflake",
    ground_truth_root: Path | None = None,
    workspace_directory: Path | None = None,
) -> TaskSpec:
    """Load one official task.

    Args:
        elt_bench_root: Checkout of ELT-Bench with generated inputs.
        task_id: Task name, e.g. ``human_resources``.
        destination: Warehouse the inputs were generated for.
        ground_truth_root: Directory with ``<task>/<model>.csv``; defaults to
            ``<elt_bench_root>/ground_truth/gt_<destination>``.
        workspace_directory: Where the derived workspaces are written; by default a directory of this process.
    """
    inputs_dir = elt_bench_root / INPUT_DIRECTORIES[destination] / task_id
    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"{inputs_dir} does not exist; run ELT-Bench's setup/write_config.py first")
    workspace_dir = (workspace_directory or process_directory("elt_bench")) / destination / task_id
    build_workspace(inputs_dir, workspace_dir, destination)

    evaluation = elt_bench_root / "evaluation"
    row_counts = json.loads((evaluation / "table.json").read_text(encoding="utf-8"))[task_id]
    sort_keys = json.loads((evaluation / "sort_key.json").read_text(encoding="utf-8")).get(task_id, {})
    ground_truth_dir = (ground_truth_root or elt_bench_root / "ground_truth" / f"gt_{destination}") / task_id
    reference_dir = reference_directory(task_id)
    foreign_keys = []
    if reference_dir is not None and (reference_dir / "task.yaml").is_file():
        foreign_keys = yaml.safe_load((reference_dir / "task.yaml").read_text(encoding="utf-8")).get("foreign_keys", [])

    return TaskSpec(
        task_id=task_id,
        workspace_dir=workspace_dir,
        source_tables=tuple(row_counts),
        data_models=read_data_models(workspace_dir, sort_keys),
        engine_kind="airbyte",
        engine_options={"elt_bench_root": str(elt_bench_root), "task_id": task_id, "destination": destination},
        reference_dir=reference_dir,
        ground_truth_dir=ground_truth_dir if ground_truth_dir.is_dir() else None,
        expected_row_counts=row_counts,
        foreign_keys=parse_foreign_keys(foreign_keys),
    )


def elt_bench_backend(
    elt_bench_root: Path, destination: str = "snowflake", *, cache_syncs: bool = True
) -> BackendConfig:
    """The backend of official tasks: the warehouse whose credentials ELT-Bench keeps under ``setup/destination``.

    Only the path of the credential file is stored, so no secret reaches a logged configuration.
    """
    credential_file = elt_bench_root / "setup" / "destination" / f"{destination}_credential.json"
    return BackendConfig(
        warehouse=destination, warehouse_options={"credential_file": str(credential_file)}, cache_syncs=cache_syncs
    )
