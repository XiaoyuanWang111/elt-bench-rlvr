"""The grader's expected state of a task: verified raw tables, expected model outputs, held-out variants.

The reference state is built by running the task's reference pipeline through the same engine and warehouse as
the rollouts. It lives in namespaces that no tool can name, and it is built once per task and reused.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from elt_rlvr.extract_load.base import METADATA_COLUMN_PREFIX, ExtractLoadEngine
from elt_rlvr.grading.compare import read_ground_truth, to_text_frame
from elt_rlvr.pipeline import parse_extract_load, read_models
from elt_rlvr.task import TaskSpec
from elt_rlvr.transform import run_models
from elt_rlvr.warehouse.base import Column, TableFingerprint, Warehouse
from elt_rlvr.warehouse.state import read_state, write_state

REFERENCE_TIMEOUT_SECONDS = 600.0
MAX_HELDOUT_DRAWS = 5


class ReferencePipelineError(RuntimeError):
    """The reference pipeline itself failed: the task or the environment is broken, not the rollout."""


@dataclass(frozen=True)
class HeldOut:
    """A sample of the reference raw tables and the reference models' output on it."""

    namespace: str
    expected: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class RawTableCheck:
    """How a loaded source table compares with the reference load."""

    row_count_matches: bool
    content_matches: bool


@dataclass
class _ReferenceTable:
    name: str
    columns: dict[str, Column]
    row_count: int
    fingerprint: TableFingerprint | None = None


SYNC_COLUMN_PREFIXES = (METADATA_COLUMN_PREFIX, "_ab_")
"""Columns that describe a sync and not a source row: the loader's own, and those that Airbyte's file and
change-data-capture sources add (the file's name and modification time, log positions)."""


def source_columns(columns: list[Column]) -> dict[str, Column]:
    """The columns that carry source data, keyed by lower-cased name."""
    return {
        column.name.lower(): column for column in columns if not column.name.lower().startswith(SYNC_COLUMN_PREFIXES)
    }


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
    _tables: dict[str, _ReferenceTable] = field(default_factory=dict)
    _heldout_count: int = 0

    def __post_init__(self) -> None:
        self.namespace = self.namespace or f"reference_{self.task.task_id}"

    @property
    def has_pipeline(self) -> bool:
        return self.task.reference_dir is not None

    def _run_reference_models(self, raw_namespace: str, target_namespace: str) -> dict[str, pd.DataFrame]:
        assert self.task.reference_dir is not None
        results = run_models(
            self.warehouse,
            models=read_models(self.task.reference_dir / "models"),
            raw_table_namespaces={table.name: raw_namespace for table in self._tables.values()},
            target_namespace=target_namespace,
            timeout_seconds=REFERENCE_TIMEOUT_SECONDS,
            describe_results=False,
        )
        failed = [f"{result.name}: {result.error}" for result in results if not result.succeeded]
        if failed:
            raise ReferencePipelineError(f"reference models of {self.task.task_id} failed: {'; '.join(failed)}")
        return {
            model.name: to_text_frame(self.warehouse.read_frame(target_namespace, model.name))
            for model in self.task.data_models
        }

    def _stored_source_tables(self) -> dict[str, str]:
        """Stored names of the source tables present in the reference namespace, keyed by lower-cased name."""
        stored = {name.lower(): name for name in self.warehouse.list_tables(self.namespace)}
        return {table.lower(): stored[table.lower()] for table in self.task.source_tables if table.lower() in stored}

    def _load_reference_tables(self) -> None:
        assert self.task.reference_dir is not None
        config = (self.task.reference_dir / "extract_load.yaml").read_text(encoding="utf-8")
        digest = hashlib.sha256(config.encode()).hexdigest()
        # A persistent warehouse may still hold the load of an earlier process. It is reused when it was made
        # from the same extract-load file and every source table is still there.
        state = read_state(self.warehouse, self.namespace) if self.warehouse.persistent else None
        reusable = (
            state is not None
            and state.get("extract_load") == digest
            and len(self._stored_source_tables()) == len(self.task.source_tables)
        )
        if not reusable:
            self.warehouse.create_namespace(self.namespace)
            for connection in parse_extract_load(config):
                results = self.engine.sync(connection, warehouse=self.warehouse, namespace=self.namespace)
                failed = [f"{result.stream}: {result.error}" for result in results if not result.succeeded]
                if failed:
                    raise ReferencePipelineError(f"reference load of {self.task.task_id} failed: {'; '.join(failed)}")
            if self.warehouse.persistent:
                write_state(self.warehouse, self.namespace, {"extract_load": digest})
        stored_names = self._stored_source_tables()
        missing = [table for table in self.task.source_tables if table.lower() not in stored_names]
        if missing:
            raise ReferencePipelineError(f"reference load of {self.task.task_id} did not create: {', '.join(missing)}")
        for table in self.task.source_tables:
            name = stored_names[table.lower()]
            self._tables[table.lower()] = _ReferenceTable(
                name=name,
                columns=source_columns(self.warehouse.describe(self.namespace, name)),
                row_count=self.warehouse.row_count(self.namespace, name),
            )

    def build(self) -> None:
        """Load the reference raw tables and compute the expected outputs. Safe to call repeatedly."""
        with self._lock:
            if self._built:
                return
            reference_outputs: dict[str, pd.DataFrame] = {}
            if self.has_pipeline:
                self._load_reference_tables()
                reference_outputs = self._run_reference_models(self.namespace, self.namespace)
            if self.task.ground_truth_dir is not None:
                self._expected = {
                    model.name: read_ground_truth(self._ground_truth_file(model.name)) for model in self.task.data_models
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
            return self._tables[table.lower()].row_count
        counts = {name.lower(): count for name, count in (self.task.expected_row_counts or {}).items()}
        return counts.get(table.lower())

    def check_raw_table(self, namespace: str, table: str, produced_table: str | None) -> RawTableCheck:
        """Compare the loaded table ``namespace.produced_table`` with the reference load of ``table``.

        The content matches when both tables have the same source columns and the same multiset of rows. Without
        a reference pipeline only the row count can be checked, and it stands in for the content.
        """
        if produced_table is None:
            return RawTableCheck(row_count_matches=False, content_matches=False)
        if not self.has_pipeline:
            matches = self.warehouse.row_count(namespace, produced_table) == self.expected_row_count(table)
            return RawTableCheck(row_count_matches=matches, content_matches=matches)

        reference = self._tables[table.lower()]
        produced = source_columns(self.warehouse.describe(namespace, produced_table))
        same_columns = set(produced) == set(reference.columns) and all(
            produced[name].is_numeric == column.is_numeric for name, column in reference.columns.items()
        )
        if not same_columns:
            row_count = self.warehouse.row_count(namespace, produced_table)
            return RawTableCheck(row_count_matches=row_count == reference.row_count, content_matches=False)
        numeric = [name for name, column in reference.columns.items() if column.is_numeric]
        with self._lock:
            if reference.fingerprint is None:
                reference.fingerprint = self.warehouse.fingerprint(
                    self.namespace, reference.name, [column.name for column in reference.columns.values()], numeric
                )
        fingerprint = self.warehouse.fingerprint(
            namespace, produced_table, [column.name for column in produced.values()], numeric
        )
        return RawTableCheck(
            row_count_matches=fingerprint.row_count == reference.row_count,
            content_matches=fingerprint == reference.fingerprint,
        )

    def _parents_first(self) -> list[str]:
        """Source tables ordered so that every table comes after the tables it refers to."""
        parents: dict[str, set[str]] = {table.lower(): set() for table in self.task.source_tables}
        for key in self.task.foreign_keys:
            if key.table.lower() != key.parent_table.lower():
                parents[key.table.lower()].add(key.parent_table.lower())
        ordered: list[str] = []
        remaining = [table.lower() for table in self.task.source_tables]
        while remaining:
            ready = [table for table in remaining if parents[table] <= set(ordered)]
            if not ready:
                raise ReferencePipelineError(f"{self.task.task_id}: foreign keys form a cycle")
            ordered.extend(ready)
            remaining = [table for table in remaining if table not in ready]
        return ordered

    def _column_name(self, table: str, column: str) -> str:
        columns = self._tables[table.lower()].columns
        if column.lower() not in columns:
            raise ReferencePipelineError(f"{self.task.task_id}: foreign key names a missing column {table}.{column}")
        return columns[column.lower()].name

    def draw_heldout(self, seed: int, sample_percent: float) -> HeldOut:
        """Sample the reference raw tables and run the reference models on the sample.

        Each table keeps a random share of its rows. A row that refers to a dropped parent row is dropped too, so
        the sample satisfies every foreign key the original data satisfies. A draw on which some expected table is
        empty would let an empty model pass, so it is redrawn.
        """
        if not self.has_pipeline:
            raise ReferencePipelineError(f"task {self.task.task_id} has no reference pipeline for held-out grading")
        with self._lock:
            self._heldout_count += 1
            namespace = f"{self.namespace}_heldout_{self._heldout_count}"
        try:
            for attempt in range(MAX_HELDOUT_DRAWS):
                self.warehouse.create_namespace(namespace)
                for table in self._parents_first():
                    stored = self._tables[table].name
                    self.warehouse.copy_table(
                        self.namespace, stored, namespace, sample_percent=sample_percent, seed=seed + attempt
                    )
                    for key in self.task.foreign_keys:
                        if key.table.lower() != table or key.parent_table.lower() == table:
                            continue
                        self.warehouse.delete_orphans(
                            namespace,
                            stored,
                            self._column_name(key.table, key.column),
                            parent_table=self._tables[key.parent_table.lower()].name,
                            parent_column=self._column_name(key.parent_table, key.parent_column),
                            original_namespace=self.namespace,
                        )
                expected = self._run_reference_models(namespace, namespace)
                if all(len(frame) > 0 for frame in expected.values()):
                    return HeldOut(namespace=namespace, expected=expected)
            raise ReferencePipelineError(f"could not draw a non-empty held-out sample for {self.task.task_id}")
        except Exception:
            self.warehouse.drop_namespace(namespace)
            raise

    def drop_heldout(self, heldout: HeldOut) -> None:
        self.warehouse.drop_namespace(heldout.namespace)
