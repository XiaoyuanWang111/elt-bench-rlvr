"""Grade again the pipeline files that one rollout of a transcript submitted, and print the per-column result.

The files are taken from the write_file calls of the transcript (tool-call arguments only), run in a new rollout
namespace and graded with the environment's grader, held-out samples included.

usage: regrade_transcript.py <elt_bench_root> <task> <transcript.jsonl> <rollout index> [<held-out seed> ...]
"""

import json
import re
import sys
from pathlib import Path

from grade_pipeline import grade_files


def submitted_files(transcript: Path, rollout: int) -> dict[str, str]:
    """The last content each file had in the write_file calls of ``rollout``."""
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
    return files


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    root, name, transcript, rollout = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    seeds = [int(seed) for seed in sys.argv[5:]] or [7]
    files = submitted_files(transcript, rollout)
    print("files:", sorted(files))
    grade_files(root, name, files, seeds)


if __name__ == "__main__":
    main()
