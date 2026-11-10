"""Survey the official tasks of an ELT-Bench checkout.

Prints how many tasks the environment's loader reads, how many tasks use each kind of source, and the REST API
tables that ELT-Bench's paginated connector cannot finish (see the setup notes in the README).

    uv run python scripts/survey_tasks.py /path/to/ELT-Bench
"""

import argparse
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from elt_rlvr.tasks.elt_bench import load_elt_bench_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey the official tasks of an ELT-Bench checkout.")
    parser.add_argument("elt_bench_root", type=Path)
    root = parser.parse_args().elt_bench_root

    manifest = (root / "setup" / "elt_snowflake.yaml").read_text(encoding="utf-8")
    page_sizes = sorted({int(size) for size in re.findall(r"page_size:\s*(\d+)", manifest)})
    row_counts = json.loads((root / "evaluation" / "table.json").read_text(encoding="utf-8"))
    tasks = sorted(path.name for path in (root / "inputs").iterdir() if path.is_dir())
    workspaces = Path(tempfile.mkdtemp(prefix="elt_rlvr_survey_"))

    failed: list[str] = []
    kinds: Counter[str] = Counter()
    api_tables = 0
    unfinished: dict[str, list[str]] = {}
    for task_id in tasks:
        try:
            task = load_elt_bench_task(root, task_id, workspace_directory=workspaces)
        except Exception as error:
            failed.append(f"{task_id}: {type(error).__name__}: {error}")
            continue
        config = yaml.safe_load((task.workspace_dir / "config.yaml").read_text(encoding="utf-8"))
        kinds.update(key for key in config if key != "destination")
        counts = {table.lower(): rows for table, rows in row_counts[task_id].items()}
        tables = config.get("custom_api", {}).get("config", {}).get("tables", [])
        api_tables += len(tables)
        large = [f"{table} ({counts[table.lower()]})" for table in tables if counts[table.lower()] >= min(page_sizes)]
        if large:
            unfinished[task_id] = large

    print(f"tasks: {len(tasks)}, read by the loader: {len(tasks) - len(failed)}")
    for line in failed:
        print(f"  not read: {line}")
    print("tasks per source kind: " + ", ".join(f"{kind} {count}" for kind, count in kinds.most_common()))
    print(f"page sizes in the REST API connector manifest: {page_sizes}")
    print(f"REST API tables: {api_tables} in {kinds['custom_api']} tasks")
    large_tables = sum(len(tables) for tables in unfinished.values())
    print(f"REST API tables with at least {min(page_sizes)} rows: {large_tables} in {len(unfinished)} tasks")
    for task_id, tables in unfinished.items():
        print(f"  {task_id}: {', '.join(tables)}")


if __name__ == "__main__":
    main()
