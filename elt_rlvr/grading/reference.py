"""The grader's expected state of a task: verified raw tables, expected model outputs, held-out variants.

The reference state is built by running the task's reference pipeline through the same engine and warehouse as
the rollouts. It lives in namespaces that no tool can name, and it is built once per task and reused.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from elt_rlvr.extract_load.base import METADATA_COLUMN_PREFIX, ExtractLoadEngine
from elt_rlvr.pipeline import parse_extract_load, read_models
from elt_rlvr.task import TaskSpec
from elt_rlvr.transform import run_models
from elt_rlvr.warehouse.base import Column, TableFingerprint, Warehouse

REFERENCE_TIMEOUT_SECONDS = 600.0
MAX_HELDOUT_DRAWS = 5


class ReferencePipelineError(RuntimeError):
    """The reference pipeline itself failed: the task or the environment is broken, not the rollout."""


@dataclass(frozen=True)
class HeldOut:
    """A sample of the reference raw tables and the reference models' output on it."""

    namespace: str
    expected: dict[str, pd.DataFrame]


def source_columns(columns: list[Column]) -> dict[str, Column]:
    """The columns that carry source data, keyed by lower-cased name."""
    return {
        column.name.lower(): column for column in columns if not column.name.lower().startswith(METADATA_COLUMN_PREFIX)
    }


def find_table(warehouse: Warehouse, namespace: str, table: str) -> str | None:
    """The stored name of ``table`` in ``namespace``, matched case-insensitively."""
    for name in warehouse.list_tables(namespace):
        if name.lower() == table.lower():
            return name
    return None


@dataclass
class TaskReference:
    """Expected raw tables and model outputs of one task in one warehouse."""

    task: TaskSpec
    warehouse: Warehouse
    engine: ExtractLoadEngine
    namespace: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _built: bool = False
    _expected: dict[str, pd.DataFrame] = field(default_factory=dict)
    _fingerprints: dict[str, TableFingerprint] = field(default_factory=dict)
    _heldout_count: int = 0

    def __post_init__(self) -> None:
        self.namespace = self.namespace or f"reference_{self.task.task_id}"

    @property
    def has_pipeline(self) -> bool:
        return self.task.reference_dir is not None

    def _reference_models(self) -> dict[str, str]:
        assert self.task.reference_dir is not None
        return read_models(self.task.reference_dir / "models")

    def _run_reference_models(self, raw_namespace: str, target_namespace: str) -> dict[str, pd.DataFrame]:
        results = run_models(
            self.warehouse,
            models=self._reference_models(),
            raw_table_namespaces={table: raw_namespace for table in self.task.source_tables},
            target_namespace=target_namespace,
            timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
        )
        failed = [f"{result.name}: {result.error}" for result in results if not result.succeeded]
        if failed:
            raise ReferencePipelineError(f"reference models of {self.task.task_id} failed: {'; '.join(failed)}")
        return {model.name: self.warehouse.read_frame(target_namespace, model.name) for model in self.task.data_models}

    def build(self) -> None:
        """Load the reference raw tables and compute the expected outputs. Safe to call repeatedly."""
        with self._lock:
            if self._built:
                return
            if self.has_pipeline:
                assert self.task.reference_dir is not None
                self.warehouse.create_namespace(self.namespace)
                config = (self.task.reference_dir / "extract_load.yaml").read_text(encoding="utf-8")
                for connection in parse_extract_load(config):
                    results = self.engine.sync(connection, warehouse=self.warehouse, namespace=self.namespace)
                    failed = [f"{result.stream}: {result.error}" for result in results if not result.succeeded]
                    if failed:
                        raise ReferencePipelineError(
                            f"reference load of {self.task.task_id} failed: {'; '.join(failed)}"
                        )
                missing = [t for t in self.task.source_tables if find_table(self.warehouse, self.namespace, t) is None]
                if missing:
                    raise ReferencePipelineError(
                        f"reference load of {self.task.task_id} did not create: {', '.join(missing)}"
                    )
                reference_outputs = self._run_reference_models(self.namespace, self.namespace)
            else:
                reference_outputs = {}
            if self.task.ground_truth_dir is not None:
                self._expected = {
                    model.name: pd.read_csv(self._ground_truth_file(model.name), dtype=str, keep_default_na=True)
                    for model in self.task.data_models
                }
            else:
                self._expected = reference_outputs
            self._built = True

    def _ground_truth_file(self, model: str) -> Path:
        assert self.task.ground_truth_dir is not None
        for path in self.task.ground_truth_dir.glob("*.csv"):
            if path.stem.lower() == model.lower():
                return path
        raise ReferencePipelineError(f"no ground-truth file for model {model!r} of {self.task.task_id}")

    def expected(self, model: str) -> pd.DataFrame:
        return self._expected[model]

    def expected_row_count(self, table: str) -> int | None:
        if self.has_pipeline:
            stored = find_table(self.warehouse, self.namespace, table)
            return self.warehouse.row_count(self.namespace, stored) if stored else None
        counts = {name.lower(): count for name, count in (self.task.expected_row_counts or {}).items()}
        return counts.get(table.lower())

    def raw_table_matches(self, namespace: str, table: str) -> bool:
        """Whether ``namespace.table`` holds exactly the rows of the reference load of ``table``.

        Without a reference pipeline only the row count can be checked.
        """
        produced_table = find_table(self.warehouse, namespace, table)
        if produced_table is None:
            return False
        if not self.has_pipeline:
            return self.warehouse.row_count(namespace, produced_table) == self.expected_row_count(table)
        reference_table = find_table(self.warehouse, self.namespace, table)
        assert reference_table is not None
        produced = source_columns(self.warehouse.describe(namespace, produced_table))
        reference = source_columns(self.warehouse.describe(self.namespace, reference_table))
        if set(produced) != set(reference):
            return False
        if any(produced[name].is_numeric != reference[name].is_numeric for name in reference):
            return False
        numeric = [name for name in reference if reference[name].is_numeric]
        if table not in self._fingerprints:
            self._fingerprints[table] = self.warehouse.fingerprint(
                self.namespace, reference_table, [reference[name].name for name in reference], numeric
            )
        fingerprint = self.warehouse.fingerprint(
            namespace, produced_table, [produced[name].name for name in produced], numeric
        )
        return fingerprint == self._fingerprints[table]

    def _column_name(self, table: str, column: str) -> str:
        stored = find_table(self.warehouse, self.namespace, table)
        assert stored is not None
        for candidate in self.warehouse.describe(self.namespace, stored):
            if candidate.name.lower() == column.lower():
                return candidate.name
        raise ReferencePipelineError(f"{self.task.task_id}: foreign key names a missing column {table}.{column}")

    def _parents_first(self) -> list[str]:
        """Source tables ordered so that every table comes after the tables it refers to."""
        parents: dict[str, set[str]] = {table.lower(): set() for table in self.task.source_tables}
        for key in self.task.foreign_keys:
            if key.table.lower() != key.parent_table.lower():
                parents[key.table.lower()].add(key.parent_table.lower())
        ordered: list[str] = []
        remaining = list(self.task.source_tables)
        while remaining:
            ready = [table for table in remaining if parents[table.lower()] <= {name.lower() for name in ordered}]
            if not ready:
                raise ReferencePipelineError(f"{self.task.task_id}: foreign keys form a cycle")
            ordered.extend(ready)
            remaining = [table for table in remaining if table not in ready]
        return ordered

    def draw_heldout(self, seed: int, sample_percent: float) -> HeldOut:
        """Sample the reference raw tables and run the reference models on the sample.

        Each table keeps a random share of its rows. A row that refers to a dropped parent row is dropped too, so
        the sample satisfies every foreign key the original data satisfies. A draw on which some expected table is
        empty would let an empty model pass, so it is redrawn.
        """
        if not self.has_pipeline:
            raise ReferencePipelineError(
                f"task {self.task.task_id} has no reference pipeline to grade held-out data with"
            )
        with self._lock:
            self._heldout_count += 1
            namespace = f"{self.namespace}_heldout_{self._heldout_count}"
        for attempt in range(MAX_HELDOUT_DRAWS):
            self.warehouse.create_namespace(namespace)
            for table in self._parents_first():
                stored = find_table(self.warehouse, self.namespace, table)
                assert stored is not None
                self.warehouse.copy_table(
                    self.namespace, stored, namespace, sample_percent=sample_percent, seed=seed + attempt
                )
                for key in self.task.foreign_keys:
                    if key.table.lower() != table.lower() or key.parent_table.lower() == table.lower():
                        continue
                    self.warehouse.delete_orphans(
                        namespace,
                        stored,
                        self._column_name(stored, key.column),
                        parent_table=find_table(self.warehouse, self.namespace, key.parent_table) or key.parent_table,
                        parent_column=self._column_name(key.parent_table, key.parent_column),
                        original_namespace=self.namespace,
                    )
            expected = self._run_reference_models(namespace, namespace)
            if all(len(frame) > 0 for frame in expected.values()):
                return HeldOut(namespace=namespace, expected=expected)
        raise ReferencePipelineError(f"could not draw a non-empty held-out sample for {self.task.task_id}")

    def drop_heldout(self, heldout: HeldOut) -> None:
        self.warehouse.drop_namespace(heldout.namespace)
