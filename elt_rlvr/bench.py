"""Measure how long environment transitions and reward computation take.

    python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8

The reference pipeline is played in ``group_size`` rollouts, first with every sync run against the sources and then
with the sync cache, and the group is graded each time. The script prints the mean seconds per tool call and per
graded rollout and writes the numbers as JSON.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from elt_rlvr.backend import build_engine
from elt_rlvr.grading.grader import RewardConfig, grade_group
from elt_rlvr.grading.reference import TaskReference
from elt_rlvr.pipeline import EXTRACT_LOAD_PATH, MODELS_DIRECTORY
from elt_rlvr.session import RolloutSession
from elt_rlvr.task import TaskSpec
from elt_rlvr.tasks.local import LOCAL_TASKS, build_local_tasks
from elt_rlvr.warehouse import Warehouse, open_warehouse


def _play_reference(session: RolloutSession) -> None:
    task = session.task
    assert task.reference_dir is not None
    session.read_file("config.yaml")
    session.read_file("data_model.yaml")
    session.write_file(EXTRACT_LOAD_PATH, (task.reference_dir / "extract_load.yaml").read_text(encoding="utf-8"))
    session.run_extract_load()
    for path in sorted((task.reference_dir / "models").glob("*.sql")):
        session.write_file(f"{MODELS_DIRECTORY}/{path.name}", path.read_text(encoding="utf-8"))
    session.run_transform()
    for model in task.data_models:
        session.query(f"select * from {model.name}")
    session.submit()


def _measure(task: TaskSpec, warehouse: Warehouse, *, group_size: int, cache_syncs: bool, heldout_samples: int) -> dict:
    engine = build_engine(task, cache_syncs=cache_syncs)
    reference = TaskReference(
        task, warehouse, build_engine(task, cache_syncs=False), namespace=f"reference_{cache_syncs}"
    )
    start = time.perf_counter()
    reference.build()
    reference_seconds = time.perf_counter() - start

    with tempfile.TemporaryDirectory() as directory:
        sessions = [
            RolloutSession(task=task, warehouse=warehouse, engine=engine, workspace=Path(directory) / str(index))
            for index in range(group_size)
        ]
        for session in sessions:
            _play_reference(session)
        start = time.perf_counter()
        grades = grade_group(sessions, reference, RewardConfig(heldout_samples=heldout_samples))
        grading_seconds = time.perf_counter() - start
        tool_seconds: dict[str, list[float]] = {}
        for session in sessions:
            for timing in session.timings:
                tool_seconds.setdefault(timing.tool, []).append(timing.seconds)
            session.close()

    assert all(grade.reward == 1.0 for grade in grades), [grade.reward for grade in grades]
    return {
        "cache_syncs": cache_syncs,
        "reference_build_seconds": round(reference_seconds, 4),
        "grading_seconds_per_rollout": round(grading_seconds / group_size, 4),
        "tool_seconds_mean": {tool: round(statistics.mean(values), 4) for tool, values in tool_seconds.items()},
        "first_extract_load_seconds": round(tool_seconds["run_extract_load"][0], 4),
        "later_extract_load_seconds_mean": round(statistics.mean(tool_seconds["run_extract_load"][1:] or [0.0]), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure transition and reward latency on a local task.")
    parser.add_argument("--task", default="storefront", choices=list(LOCAL_TASKS))
    parser.add_argument("--scale", type=int, default=50, help="multiplier of the source data size")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--heldout-samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/latency.json"))
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        task = build_local_tasks(Path(directory), [arguments.task], scale=arguments.scale)[0]
        warehouse = open_warehouse("duckdb")
        rows = {table: 0 for table in task.source_tables}
        results = []
        for cache_syncs in (False, True):
            results.append(
                _measure(
                    task,
                    warehouse,
                    group_size=arguments.group_size,
                    cache_syncs=cache_syncs,
                    heldout_samples=arguments.heldout_samples,
                )
            )
        for table in rows:
            rows[table] = warehouse.row_count("reference_True", table)
        warehouse.close()

    report = {
        "task": arguments.task,
        "scale": arguments.scale,
        "group_size": arguments.group_size,
        "heldout_samples": arguments.heldout_samples,
        "source_rows": rows,
        "runs": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
