"""Helpers shared by the tests: build a rollout on a local task and act in it the way a policy would."""

from pathlib import Path

import yaml

from elt_rlvr.backend import build_engine
from elt_rlvr.grading.grader import Grade, RewardConfig, grade_group
from elt_rlvr.grading.reference import TaskReference
from elt_rlvr.pipeline import EXTRACT_LOAD_PATH, MODELS_DIRECTORY
from elt_rlvr.session import RolloutSession
from elt_rlvr.task import TaskSpec
from elt_rlvr.warehouse import Warehouse

HELDOUT_SEED = 7


def reference_extract_load(task: TaskSpec) -> str:
    assert task.reference_dir is not None
    return (task.reference_dir / "extract_load.yaml").read_text(encoding="utf-8")


def reference_models(task: TaskSpec) -> dict[str, str]:
    assert task.reference_dir is not None
    return {
        path.stem: path.read_text(encoding="utf-8") for path in sorted((task.reference_dir / "models").glob("*.sql"))
    }


def open_session(task: TaskSpec, warehouse: Warehouse, workspace: Path, *, cache_syncs: bool = False) -> RolloutSession:
    return RolloutSession(
        task=task, warehouse=warehouse, engine=build_engine(task, cache_syncs=cache_syncs), workspace=workspace
    )


def write_pipeline(session: RolloutSession, extract_load: str | None, models: dict[str, str]) -> None:
    """Write pipeline files through the ``write_file`` tool and fail the test if a write is refused."""
    if extract_load is not None:
        assert "error" not in session.write_file(EXTRACT_LOAD_PATH, extract_load)
    for name, sql in models.items():
        assert "error" not in session.write_file(f"{MODELS_DIRECTORY}/{name}.sql", sql)


def solve(session: RolloutSession) -> None:
    """Write and run the reference pipeline, then submit."""
    write_pipeline(session, reference_extract_load(session.task), reference_models(session.task))
    assert all(entry["status"] == "succeeded" for entry in session.run_extract_load()["connections"])
    assert all(entry["status"] == "succeeded" for entry in session.run_transform()["models"])
    session.submit()


def grade(session: RolloutSession, config: RewardConfig | None = None) -> Grade:
    reference = TaskReference(
        task=session.task, warehouse=session.warehouse, engine=build_engine(session.task, cache_syncs=False)
    )
    return grade_group([session], reference, config or RewardConfig(), seed=HELDOUT_SEED)[0]


def edit_extract_load(task: TaskSpec, edit) -> str:
    """The reference extract-load config after ``edit`` changed its parsed document in place."""
    document = yaml.safe_load(reference_extract_load(task))
    edit(document)
    return yaml.safe_dump(document, sort_keys=False)
