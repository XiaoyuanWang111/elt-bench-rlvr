"""Grade pipeline files written outside a rollout, for example a solution to an official task written by hand.

The files run in a new rollout namespace and are graded with the environment's grader, held-out samples included,
once per held-out seed. Without --extract-load the task's packaged reference extract-load file is used, so a set
of model files can be graded on its own.

    uv run python scripts/grade_pipeline.py --elt-bench-root /path/to/ELT-Bench --task retails \\
        --models /path/to/ELT-Bench/example/retails/customers.sql /path/to/ELT-Bench/example/retails/nations.sql \\
        --drop-qualifier retails.airbyte_schema. --seeds 7 11 13

--drop-qualifier removes a schema prefix from the SQL text, since a model refers to tables by their bare names.
"""

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from elt_rlvr.env import task_runtime
from elt_rlvr.grading.grader import RewardConfig, grade_group
from elt_rlvr.pipeline import EXTRACT_LOAD_PATH, MODELS_DIRECTORY
from elt_rlvr.session import RolloutSession, SessionLimits
from elt_rlvr.tasks import load_tasks, select_backend


def grade_files(elt_bench_root: str, task_name: str, files: Mapping[str, str], seeds: Sequence[int]) -> None:
    """Write ``files`` (workspace path to content) into a new rollout, load and grade it once per seed."""
    task = load_tasks([task_name], elt_bench_root=elt_bench_root, destination="snowflake")[0]
    runtime = task_runtime(task, select_backend(warehouse="snowflake", elt_bench_root=elt_bench_root, cache_syncs=True))
    for seed in seeds:
        session = RolloutSession(
            task=task,
            warehouse=runtime.warehouse,
            engine=runtime.engine,
            workspace=Path(tempfile.mkdtemp(prefix="elt_rlvr_grade_")),
            limits=SessionLimits(),
        )
        try:
            for path, content in files.items():
                written = session.write_file(path, content)
                if "error" in written:
                    raise SystemExit(f"{path}: {written['error']}")
            report = session.run_extract_load()
            print("extract-load:", [(entry["connection"], entry["status"]) for entry in report["connections"]])
            grade = grade_group([session], runtime.reference, RewardConfig(), seed=seed)[0]
            wrong = {
                model: [column for column, right in columns.items() if not right]
                for model, columns in grade.model_columns.items()
            }
            print(
                f"seed {seed}: reward {grade.reward:.4f} EL {grade.extract_load_score} T {grade.transform_score:.4f} "
                f"T on original data {grade.transform_score_original:.4f} error {grade.grader_error}"
            )
            print("  columns not right:", json.dumps(wrong))
        finally:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade pipeline files of an official ELT-Bench task.")
    parser.add_argument("--elt-bench-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="model files, one per data model")
    parser.add_argument("--extract-load", type=Path, help="extract-load file; default: the task's reference")
    parser.add_argument("--drop-qualifier", default="", help="text to remove from the SQL, e.g. a schema prefix")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7], help="held-out seeds, one grading each")
    arguments = parser.parse_args()

    task = load_tasks([arguments.task], elt_bench_root=arguments.elt_bench_root, destination="snowflake")[0]
    extract_load = arguments.extract_load
    if extract_load is None:
        if task.reference_dir is None:
            raise SystemExit(f"task {arguments.task} has no reference pipeline; pass --extract-load")
        extract_load = task.reference_dir / "extract_load.yaml"
    files = {EXTRACT_LOAD_PATH: extract_load.read_text(encoding="utf-8")}
    for path in arguments.models:
        sql = path.read_text(encoding="utf-8")
        if arguments.drop_qualifier:
            sql = sql.replace(arguments.drop_qualifier, "")
        files[f"{MODELS_DIRECTORY}/{path.stem}.sql"] = sql
    print("files:", sorted(files))
    grade_files(arguments.elt_bench_root, arguments.task, files, arguments.seeds)


if __name__ == "__main__":
    main()
