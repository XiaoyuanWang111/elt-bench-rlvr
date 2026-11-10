"""Task sources: the built-in local tasks and official ELT-Bench tasks."""

from pathlib import Path

from elt_rlvr.task import TaskSpec
from elt_rlvr.tasks.local import build_local_tasks

DEFAULT_TASK_DIRECTORY = Path.home() / ".cache" / "elt_rlvr" / "tasks"


def load_tasks(
    names: list[str],
    *,
    elt_bench_root: str | None = None,
    ground_truth_root: str | None = None,
    destination: str = "snowflake",
    task_directory: Path = DEFAULT_TASK_DIRECTORY,
    seed: int = 0,
    scale: int = 1,
) -> list[TaskSpec]:
    """Load tasks by name.

    Without ``elt_bench_root`` the names are local tasks, which are written under ``task_directory``. With it, the
    names are ELT-Bench task names and are read from that checkout.
    """
    if elt_bench_root is None:
        return build_local_tasks(task_directory, names or None, seed=seed, scale=scale)
    # Imported on use: reading ELT-Bench tasks needs a checkout of the benchmark and its generated inputs.
    from elt_rlvr.tasks.elt_bench import load_elt_bench_task  # noqa: PLC0415

    return [
        load_elt_bench_task(
            Path(elt_bench_root),
            name,
            destination=destination,
            ground_truth_root=Path(ground_truth_root) if ground_truth_root else None,
        )
        for name in names
    ]
