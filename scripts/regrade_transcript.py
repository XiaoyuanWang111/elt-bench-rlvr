"""Grade again the pipeline files that one rollout of a transcript submitted, and print the per-column result.

The files are taken from the write_file calls of the transcript (tool-call arguments only), run in a new rollout
namespace and graded with the environment's grader, held-out samples included.

usage: regrade_transcript.py <elt_bench_root> <task> <transcript.jsonl> <rollout index> [<held-out seed> ...]
"""

import json
import re
import sys
import tempfile
from pathlib import Path

from elt_rlvr.env import task_runtime
from elt_rlvr.grading.grader import RewardConfig, grade_group
from elt_rlvr.session import RolloutSession, SessionLimits
from elt_rlvr.tasks import load_tasks, select_backend

root, name, transcript, rollout = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
seeds = [int(seed) for seed in sys.argv[5:]] or [7]
files: dict[str, str] = {}
for line in transcript.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    if "summary" in record or record.get("rollout") != rollout:
        continue
    for key, value in record["logs"].items():
        match = re.match(r"write_file\((\{.*\})\)$", value, re.S) if key.startswith("tool_call") else None
        if match:
            arguments = json.loads(match.group(1))
            files[arguments["path"]] = arguments["content"]
print("files:", sorted(files))

task = load_tasks([name], elt_bench_root=root, destination="snowflake")[0]
runtime = task_runtime(task, select_backend(warehouse="snowflake", elt_bench_root=root, cache_syncs=True))
for seed in seeds:
    session = RolloutSession(
        task=task,
        warehouse=runtime.warehouse,
        engine=runtime.engine,
        workspace=Path(tempfile.mkdtemp(prefix="elt_rlvr_regrade_")),
        limits=SessionLimits(),
    )
    try:
        for path, content in files.items():
            session.write_file(path, content)
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
