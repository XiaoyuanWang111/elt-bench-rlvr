"""Task sources: the built-in local tasks and official ELT-Bench tasks."""

from pathlib import Path

from elt_rlvr.backend import BackendConfig
from elt_rlvr.settings import process_directory
from elt_rlvr.task import TaskSpec
from elt_rlvr.tasks.local import build_local_tasks


def select_backend(*, warehouse: str, elt_bench_root: str | None, cache_syncs: bool) -> BackendConfig:
    """The backend of a run: the benchmark's own warehouse for official tasks, a credential-free one otherwise."""
    if elt_bench_root is None:
        return BackendConfig(warehouse=warehouse, cache_syncs=cache_syncs)
    # Imported on use, like the official task loader below.
    from elt_rlvr.tasks.elt_bench import elt_bench_backend  # noqa: PLC0415

    return elt_bench_backend(Path(elt_bench_root), warehouse, cache_syncs=cache_syncs)


def load_tasks(
    names: list[str],
    *,
    elt_bench_root: str | None = None,
    ground_truth_root: str | None = None,
    destination: str = "snowflake",
    task_directory: Path | None = None,
    seed: int = 0,
    scale: int = 1,
) -> list[TaskSpec]:
    """Load tasks by name.

    Without ``elt_bench_root`` the names are local tasks, which are written under ``task_directory`` (by default
    a directory of this process). With it, the names are ELT-Bench task names and are read from that checkout.
    """
    if elt_bench_root is None:
        directory = task_directory or process_directory("tasks")
        return build_local_tasks(directory, names or None, seed=seed, scale=scale)
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
