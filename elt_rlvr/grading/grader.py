"""Turn the state a rollout left behind into a reward.

Extract-load score: the share of required source tables whose loaded rows equal the reference load.

Transform score: the grader re-runs the submitted model files itself, in a fresh namespace, and compares each
required table with the expected one column by column. It does this on the original data and on held-out samples
of the raw tables, where the expected output comes from the reference models. A column counts only if it matches
every time, so SQL that returns hard-coded answers, or that is right on this data by accident, scores nothing.

Any failure inside the grader gives reward 0 for that rollout.
"""

import logging
import random
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from elt_rlvr.grading.compare import OFFICIAL, STRICT, compare_tables
from elt_rlvr.grading.reference import HeldOut, TaskReference
from elt_rlvr.pipeline import MODELS_DIRECTORY, PipelineError, read_models
from elt_rlvr.session import RolloutSession
from elt_rlvr.transform import run_models

logger = logging.getLogger(__name__)

RewardShape = Literal["multiplicative", "additive", "binary"]


@dataclass(frozen=True)
class RewardConfig:
    """How scores combine into the reward."""

    shape: RewardShape = "additive"
    """``additive``: ``w * EL + (1 - w) * T``; ``multiplicative``: ``EL * (w + (1 - w) * T)``, which pays for
    transformations only in proportion to the loads that are right, as the benchmark's two stages do;
    ``binary``: 1 only when every table is right."""
    extract_load_weight: float = 0.3
    heldout_samples: int = 2
    """Held-out samples per group of rollouts; 0 grades on the original data only."""
    heldout_sample_percent: float = 70.0
    turn_cost: float = 0.0
    """Subtracted from the reward for every assistant turn."""
    grading_threads: int = 8
    """Rollouts of a group graded at the same time."""


@dataclass(frozen=True)
class Grade:
    """The reward of one rollout and what it is made of."""

    reward: float
    extract_load_score: float
    transform_score: float
    source_tables: Mapping[str, bool] = field(default_factory=dict)
    model_columns: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)
    transform_score_original: float = 0.0
    """Transform score on the original data only, before the held-out check."""
    official_extract_load: bool = False
    """ELT-Bench stage 1: every source table exists with the expected row count."""
    official_transform: float = 0.0
    """ELT-Bench stage 2: share of models that match under the benchmark's value rule, 0 unless stage 1 passed."""
    grader_error: str | None = None

    def metrics(self) -> dict[str, float]:
        return {
            "reward": self.reward,
            "extract_load_score": self.extract_load_score,
            "transform_score": self.transform_score,
            "transform_score_original": self.transform_score_original,
            "official_extract_load": float(self.official_extract_load),
            "official_transform": self.official_transform,
            "solved": float(self.extract_load_score == 1.0 and self.transform_score == 1.0),
            "grader_error": float(self.grader_error is not None),
        }


def combine(extract_load_score: float, transform_score: float, config: RewardConfig) -> float:
    weight = config.extract_load_weight
    if config.shape == "binary":
        return float(extract_load_score == 1.0 and transform_score == 1.0)
    if config.shape == "additive":
        return weight * extract_load_score + (1 - weight) * transform_score
    return extract_load_score * (weight + (1 - weight) * transform_score)


class _HeldOutSamples:
    """The held-out samples of one group, drawn when the first rollout needs them.

    A group in which no rollout gets a column right on the original data never pays for the samples. If a draw
    fails, every rollout that needs the samples fails with it: none is graded on fewer samples than the others.
    """

    def __init__(self, reference: TaskReference, config: RewardConfig, seed: int | None) -> None:
        self._reference = reference
        self._config = config
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._samples: list[HeldOut] = []
        self._drawn = False
        self._error: Exception | None = None

    def get(self) -> list[HeldOut]:
        with self._lock:
            if not self._drawn:
                self._drawn = True
                try:
                    for _ in range(self._config.heldout_samples if self._reference.has_pipeline else 0):
                        seed = self._rng.randrange(2**31)
                        self._samples.append(self._reference.draw_heldout(seed, self._config.heldout_sample_percent))
                except Exception as error:
                    self._error = error
            if self._error is not None:
                raise RuntimeError(f"the held-out samples could not be drawn: {self._error}") from self._error
            return self._samples

    def drop(self) -> None:
        for sample in self._samples:
            self._reference.drop_heldout(sample)


def _replay_models(
    session: RolloutSession,
    models: Mapping[str, str],
    raw_table_namespaces: Mapping[str, str],
    namespace: str,
    expected_rows: Mapping[str, int],
) -> dict[str, pd.DataFrame]:
    """Run the submitted models in a fresh namespace and return the required tables, by lower-cased name.

    ``expected_rows`` gives the row count of each required table. A table is fetched only if it has that many
    rows: one of another size is wrong whatever it holds, and a model can build one too large to fetch.
    """
    warehouse = session.warehouse
    warehouse.create_namespace(namespace)
    try:
        results = run_models(
            warehouse,
            models=models,
            raw_table_namespaces=raw_table_namespaces,
            target_namespace=namespace,
            timeout_seconds=session.limits.statement_timeout_seconds,
            describe_results=False,
        )
        frames = {}
        for result in results:
            rows = expected_rows.get(result.name.lower())
            if result.succeeded and rows is not None and warehouse.row_count(namespace, result.name) == rows:
                frames[result.name.lower()] = warehouse.read_frame(namespace, result.name)
        return frames
    finally:
        warehouse.drop_namespace(namespace)


def _grade(session: RolloutSession, reference: TaskReference, heldouts: _HeldOutSamples, config: RewardConfig) -> Grade:
    task, warehouse = session.task, session.warehouse

    stored_tables = {name.lower(): name for name in warehouse.list_tables(session.namespace)}
    checks = {
        table: reference.check_raw_table(session.namespace, table, stored_tables.get(table.lower()))
        for table in task.source_tables
    }
    # Only a table the extract-load engine wrote can count as loaded; a model with a source table's name cannot.
    loaded = {table.lower() for table in session.raw_tables}
    source_tables = {table: table.lower() in loaded and check.content_matches for table, check in checks.items()}
    extract_load_score = sum(source_tables.values()) / len(source_tables)
    official_extract_load = all(check.row_count_matches for check in checks.values())

    try:
        models = read_models(session.workspace / MODELS_DIRECTORY)
    except PipelineError:
        models = {}
    replay_namespace = f"grade_{session.namespace}"
    produced = _replay_models(
        session,
        models,
        {table: session.namespace for table in session.raw_tables},
        replay_namespace,
        {model.name.lower(): len(reference.expected(model.name)) for model in task.data_models},
    )

    matched: dict[str, set[str]] = {}
    official_matches = 0
    for model in task.data_models:
        frame = produced.get(model.name.lower())
        if frame is None:
            matched[model.name] = set()
            continue
        expected = reference.expected(model.name)
        strict = compare_tables(expected, frame, key=model.key, required_columns=model.columns, rule=STRICT)
        # Lower case, because the ground truth, the warehouse and data_model.yaml each case names their own way.
        matched[model.name] = {name.lower() for name in strict.matched}
        # Whatever the strict rule accepts, the benchmark's looser rule accepts too.
        official_matches += (
            strict.is_full_match
            or compare_tables(
                expected, frame, key=model.key, required_columns=model.columns, rule=OFFICIAL
            ).is_full_match
        )
    original_score = sum(len(matched[model.name]) / len(model.columns) for model in task.data_models) / len(
        task.data_models
    )

    if any(matched.values()):
        for index, heldout in enumerate(heldouts.get()):
            produced_heldout = _replay_models(
                session,
                models,
                {table: heldout.namespace for table in task.source_tables},
                f"{replay_namespace}_heldout_{index}",
                {model.name.lower(): len(heldout.expected[model.name]) for model in task.data_models},
            )
            for model in task.data_models:
                frame = produced_heldout.get(model.name.lower())
                if frame is None:
                    matched[model.name] = set()
                    continue
                comparison = compare_tables(
                    heldout.expected[model.name], frame, key=model.key, required_columns=model.columns, rule=STRICT
                )
                matched[model.name] &= {name.lower() for name in comparison.matched}

    model_columns = {
        model.name: {column: column.lower() in matched[model.name] for column in model.columns}
        for model in task.data_models
    }
    transform_score = sum(len(matched[model.name]) / len(model.columns) for model in task.data_models) / len(
        task.data_models
    )
    return Grade(
        reward=combine(extract_load_score, transform_score, config),
        extract_load_score=extract_load_score,
        transform_score=transform_score,
        source_tables=source_tables,
        model_columns=model_columns,
        transform_score_original=original_score,
        official_extract_load=official_extract_load,
        official_transform=official_matches / len(task.data_models) if official_extract_load else 0.0,
    )


def _grade_or_zero(
    session: RolloutSession, reference: TaskReference, heldouts: _HeldOutSamples, config: RewardConfig
) -> Grade:
    try:
        return _grade(session, reference, heldouts, config)
    except Exception as error:
        logger.exception("grading failed for %s", session.namespace)
        return Grade(0.0, 0.0, 0.0, grader_error=f"{type(error).__name__}: {error}")


def grade_group(
    sessions: Sequence[RolloutSession],
    reference: TaskReference,
    config: RewardConfig,
    *,
    seed: int | None = None,
) -> list[Grade]:
    """Grade the rollouts of one group against the same held-out samples.

    The rollouts are graded in parallel threads; the warehouse does the heavy work outside the interpreter lock.

    Args:
        sessions: The finished rollouts of one task.
        reference: The task's reference state.
        config: Reward settings.
        seed: Seed of the held-out samples. ``None`` draws a fresh one, so no two groups are graded on the same
            hidden data.
    """
    reference.build()
    heldouts = _HeldOutSamples(reference, config, seed)
    try:
        with ThreadPoolExecutor(max_workers=min(config.grading_threads, max(len(sessions), 1))) as pool:
            return list(pool.map(lambda session: _grade_or_zero(session, reference, heldouts, config), sessions))
    finally:
        heldouts.drop()
