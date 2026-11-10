"""Measure how long environment transitions and reward computation take.

    # a local task on DuckDB, with the source data made 50 times larger
    python -m elt_rlvr.bench --task storefront --scale 50 --group-size 8

    # an official task through Airbyte and Snowflake
    python -m elt_rlvr.bench --task human_resources --elt-bench-root /path/to/ELT-Bench --warehouse snowflake \
        --group-size 3 --direct-rollouts 1

The reference pipeline is played in two rounds of rollouts: first with every sync run against the sources, then
with the sync cache. Each round is graded. The script prints the seconds per tool call and per graded rollout and
writes the numbers as JSON.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from elt_rlvr.backend import build_engine, build_warehouse
from elt_rlvr.extract_load.cache import CachingEngine
from elt_rlvr.grading.grader import RewardConfig, grade_group
from elt_rlvr.grading.reference import TaskReference
from elt_rlvr.pipeline import EXTRACT_LOAD_PATH, MODELS_DIRECTORY
from elt_rlvr.session import RolloutSession
from elt_rlvr.task import TaskSpec
from elt_rlvr.tasks import load_tasks, select_backend
from elt_rlvr.warehouse import Warehouse


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


def _measure(
    task: TaskSpec,
    warehouse: Warehouse,
    reference: TaskReference,
    *,
    rollouts: int,
    cache_syncs: bool,
    heldout_samples: int,
) -> dict:
    engine = build_engine(task, cache_syncs=cache_syncs)
    with tempfile.TemporaryDirectory() as directory:
        sessions = [
            RolloutSession(task=task, warehouse=warehouse, engine=engine, workspace=Path(directory) / str(index))
            for index in range(rollouts)
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

    rewards = [grade.reward for grade in grades]
    if rewards != [1.0] * rollouts:
        raise RuntimeError(f"the reference pipeline did not earn full reward: {rewards}")
    result = {
        "cache_syncs": cache_syncs,
        "rollouts": rollouts,
        "grading_seconds_per_rollout": round(grading_seconds / rollouts, 3),
        "tool_seconds_mean": {tool: round(statistics.mean(values), 4) for tool, values in tool_seconds.items()},
        "extract_load_seconds": [round(seconds, 3) for seconds in tool_seconds["run_extract_load"]],
    }
    if isinstance(engine, CachingEngine):
        result["snapshot_hits"], result["snapshot_misses"] = engine.hits, engine.misses
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure transition and reward latency by playing the reference pipeline."
    )
    parser.add_argument("--task", default="storefront")
    parser.add_argument("--elt-bench-root", help="ELT-Bench checkout; makes --task an official task name")
    parser.add_argument("--warehouse", default="duckdb")
    parser.add_argument("--scale", type=int, default=50, help="multiplier of the source data size of a local task")
    parser.add_argument("--group-size", type=int, default=8, help="rollouts in the round with the sync cache")
    parser.add_argument(
        "--direct-rollouts", type=int, help="rollouts in the round without the cache (default: group size)"
    )
    parser.add_argument("--heldout-samples", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/latency.json"))
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        task = load_tasks(
            [arguments.task],
            elt_bench_root=arguments.elt_bench_root,
            destination=arguments.warehouse,
            task_directory=Path(directory),
            scale=arguments.scale,
        )[0]
        backend = select_backend(
            warehouse=arguments.warehouse, elt_bench_root=arguments.elt_bench_root, cache_syncs=True
        )
        warehouse = build_warehouse(backend)
        reference = TaskReference(task, warehouse, build_engine(task, cache_syncs=False))
        start = time.perf_counter()
        reference.build()
        reference_seconds = time.perf_counter() - start
        runs = [
            _measure(
                task,
                warehouse,
                reference,
                rollouts=arguments.direct_rollouts or arguments.group_size,
                cache_syncs=False,
                heldout_samples=arguments.heldout_samples,
            ),
            _measure(
                task,
                warehouse,
                reference,
                rollouts=arguments.group_size,
                cache_syncs=True,
                heldout_samples=arguments.heldout_samples,
            ),
        ]
        source_rows = {table: reference.expected_row_count(table) for table in task.source_tables}
        warehouse.close()

    report = {
        "task": arguments.task,
        "warehouse": arguments.warehouse,
        "scale": arguments.scale if arguments.elt_bench_root is None else None,
        "heldout_samples": arguments.heldout_samples,
        "source_rows": source_rows,
        "reference_build_seconds": round(reference_seconds, 3),
        "runs": runs,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
