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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from elt_rlvr.grading.compare import OFFICIAL, STRICT, compare_tables
from elt_rlvr.grading.reference import HeldOut, TaskReference, find_table
from elt_rlvr.pipeline import MODELS_DIRECTORY, PipelineError, read_models
from elt_rlvr.session import RolloutSession
from elt_rlvr.transform import run_models

logger = logging.getLogger(__name__)

RewardShape = Literal["multiplicative", "additive", "binary"]


@dataclass(frozen=True)
class RewardConfig:
    """How scores combine into the reward."""

    shape: RewardShape = "multiplicative"
    """``multiplicative``: ``EL * (w + (1 - w) * T)``; ``additive``: ``w * EL + (1 - w) * T``;
    ``binary``: 1 only when every table is right."""
    extract_load_weight: float = 0.3
    heldout_samples: int = 2
    """Held-out samples per group of rollouts; 0 grades on the original data only."""
    heldout_sample_percent: float = 70.0
    turn_cost: float = 0.0
    """Subtracted from the reward for every assistant turn."""
    grading_timeout_seconds: float = 300.0


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


def _replay_models(
    session: RolloutSession,
    models: Mapping[str, str],
    raw_table_namespaces: Mapping[str, str],
    namespace: str,
    timeout_seconds: float,
) -> dict[str, pd.DataFrame]:
    """Run the submitted models in a fresh namespace and return the tables that were built, by lower-cased name."""
    warehouse = session.warehouse
    warehouse.create_namespace(namespace)
    try:
        results = run_models(
            warehouse,
            models=models,
            raw_table_namespaces=raw_table_namespaces,
            target_namespace=namespace,
            timeout_seconds=timeout_seconds,
        )
        return {
            result.name.lower(): warehouse.read_frame(namespace, result.name) for result in results if result.succeeded
        }
    finally:
        warehouse.drop_namespace(namespace)


def _grade(
    session: RolloutSession, reference: TaskReference, heldouts: Sequence[HeldOut], config: RewardConfig
) -> Grade:
    task, warehouse = session.task, session.warehouse

    # Only a table the extract-load engine wrote can count as loaded; a model with a source table's name cannot.
    loaded = {table.lower() for table in session.raw_tables}
    source_tables = {
        table: table.lower() in loaded and reference.raw_table_matches(session.namespace, table)
        for table in task.source_tables
    }
    extract_load_score = sum(source_tables.values()) / len(source_tables)
    official_extract_load = all(
        (stored := find_table(warehouse, session.namespace, table)) is not None
        and warehouse.row_count(session.namespace, stored) == reference.expected_row_count(table)
        for table in task.source_tables
    )

    try:
        models = read_models(session.workspace / MODELS_DIRECTORY)
    except PipelineError:
        models = {}
    timeout = session.limits.statement_timeout_seconds
    replay_namespace = f"grade_{session.namespace}"
    produced = _replay_models(
        session, models, {table: session.namespace for table in session.raw_tables}, replay_namespace, timeout
    )

    matched: dict[str, set[str]] = {}
    official_matches = 0
    for model in task.data_models:
        frame = produced.get(model.name.lower())
        if frame is None:
            matched[model.name] = set()
            continue
        expected = reference.expected(model.name)
        matched[model.name] = set(
            compare_tables(expected, frame, key=model.key, required_columns=model.columns, rule=STRICT).matched
        )
        official = compare_tables(expected, frame, key=model.key, required_columns=model.columns, rule=OFFICIAL)
        official_matches += official.is_full_match
    original_score = sum(len(matched[model.name]) / len(model.columns) for model in task.data_models) / len(
        task.data_models
    )

    for index, heldout in enumerate(heldouts):
        produced_heldout = _replay_models(
            session,
            models,
            {table: heldout.namespace for table in task.source_tables},
            f"{replay_namespace}_heldout_{index}",
            timeout,
        )
        for model in task.data_models:
            frame = produced_heldout.get(model.name.lower())
            if frame is None:
                matched[model.name] = set()
                continue
            comparison = compare_tables(
                heldout.expected[model.name], frame, key=model.key, required_columns=model.columns, rule=STRICT
            )
            matched[model.name] &= set(comparison.matched)

    model_columns = {
        model.name: {column: column in matched[model.name] for column in model.columns} for model in task.data_models
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


def grade_group(
    sessions: Sequence[RolloutSession],
    reference: TaskReference,
    config: RewardConfig,
    *,
    seed: int | None = None,
) -> list[Grade]:
    """Grade the rollouts of one group against the same held-out samples.

    Args:
        sessions: The finished rollouts of one task.
        reference: The task's reference state.
        config: Reward settings.
        seed: Seed of the held-out samples. ``None`` draws a fresh one, so no two groups are graded on the same
            hidden data.
    """
    reference.build()
    rng = random.Random(seed)
    heldouts: list[HeldOut] = []
    if reference.has_pipeline:
        heldouts = [
            reference.draw_heldout(rng.randrange(2**31), config.heldout_sample_percent)
            for _ in range(config.heldout_samples)
        ]
    grades: list[Grade] = []
    try:
        for session in sessions:
            try:
                grades.append(_grade(session, reference, heldouts, config))
            except Exception as error:
                logger.exception("grading failed for %s", session.namespace)
                grades.append(Grade(0.0, 0.0, 0.0, grader_error=f"{type(error).__name__}: {error}"))
    finally:
        for heldout in heldouts:
            reference.drop_heldout(heldout)
    return grades
