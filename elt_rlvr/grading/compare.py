"""Compare a produced table with the expected table, column by column.

Both tables are first written as CSV text and read back as strings, which is how ELT-Bench stores its ground truth,
so a value has one textual form whichever warehouse or driver produced it. Rows are aligned by sorting on the
model's key. A required column matches when every aligned pair of values matches.

A missing value is an empty cell, which is how ELT-Bench's ground truth writes it (the 202 files hold 4.3 million
empty cells and 128 cells with a word such as ``null``). CSV cannot tell an empty string from a missing value, so
the two compare equal.

Two value rules are available:

- ``STRICT`` (used for the reward): integers and text must be equal; other numbers may differ by a relative 1e-6.
  Text such as ``NA`` or ``null`` is text, not a missing value.
- ``OFFICIAL``: the rule of ELT-Bench's evaluator, a relative tolerance of 1e-2 on every number,
  case-insensitive text, and the words pandas reads as missing (``NA``, ``null``, ``None``, ...) counted as
  missing. It is reported as a metric so that numbers stay comparable with the benchmark.
"""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# What pandas.read_csv reads as missing unless told otherwise.
PANDAS_MISSING_TEXT = frozenset(
    {
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "None",
        "n/a",
        "nan",
        "null",
    }
)


@dataclass(frozen=True)
class ValueRule:
    relative_tolerance: float
    absolute_tolerance: float
    integers_exact: bool
    case_sensitive_text: bool
    missing_text: frozenset[str] = frozenset()
    """Text that counts as a missing value, besides the empty cell."""


STRICT = ValueRule(relative_tolerance=1e-6, absolute_tolerance=1e-9, integers_exact=True, case_sensitive_text=True)
OFFICIAL = ValueRule(
    relative_tolerance=1e-2,
    absolute_tolerance=1e-9,
    integers_exact=False,
    case_sensitive_text=False,
    missing_text=PANDAS_MISSING_TEXT,
)


@dataclass(frozen=True)
class TableComparison:
    """Which required columns the produced table got right."""

    matched: tuple[str, ...]
    unmatched: tuple[str, ...]
    missing: tuple[str, ...]
    reason: str | None = None
    """Why nothing matched, when the tables could not be aligned at all."""

    @property
    def is_full_match(self) -> bool:
        return not self.unmatched and not self.missing


TEXT_FRAME_MARK = "elt_rlvr_text_frame"


def _read_text_csv(source: Path | io.StringIO) -> pd.DataFrame:
    """Read a CSV with every cell as a string; only the empty cell is missing."""
    text_frame = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[""])
    text_frame.attrs[TEXT_FRAME_MARK] = True
    return text_frame


def read_ground_truth(path: Path) -> pd.DataFrame:
    """Read a ground-truth CSV file as a text frame."""
    return _read_text_csv(path)


def to_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Round-trip a table through CSV so that every cell is a string or missing.

    The result is marked, and converting a marked frame again returns it unchanged, so an expected table that is
    compared many times is converted once.
    """
    if frame.attrs.get(TEXT_FRAME_MARK):
        return frame
    return _read_text_csv(io.StringIO(frame.to_csv(index=False)))


def _as_numbers(values: pd.Series) -> np.ndarray | None:
    """The column as floats if every present value is a number, else ``None``.

    The text ``NaN`` is the number a warehouse exports for a float NaN, which a data frame read from the warehouse
    holds as a missing value. In a column of numbers it therefore counts as a number; among other text it is text.
    """
    numbers = pd.to_numeric(values, errors="coerce")
    not_a_number = values.astype("string").str.strip().str.lower().isin(["nan"]).to_numpy(dtype=bool, na_value=False)
    unparsed = values.notna().to_numpy() & numbers.isna().to_numpy() & ~not_a_number
    if unparsed.any():
        return None
    return numbers.to_numpy(dtype=float)


def _numbers_match(expected: np.ndarray, produced: np.ndarray, rule: ValueRule) -> bool:
    both_missing = np.isnan(expected) & np.isnan(produced)
    close = np.isclose(produced, expected, rtol=rule.relative_tolerance, atol=rule.absolute_tolerance, equal_nan=False)
    if rule.integers_exact:
        with np.errstate(invalid="ignore"):
            both_integers = (expected == np.floor(expected)) & (produced == np.floor(produced))
        close = np.where(both_integers, expected == produced, close)
    return bool(np.all(both_missing | close))


def _text_matches(expected: pd.Series, produced: pd.Series, rule: ValueRule) -> bool:
    expected_missing, produced_missing = expected.isna().to_numpy(), produced.isna().to_numpy()
    if not np.array_equal(expected_missing, produced_missing):
        return False
    left = expected.astype("string").str.strip()
    right = produced.astype("string").str.strip()
    if not rule.case_sensitive_text:
        left, right = left.str.lower(), right.str.lower()
    present = ~expected_missing
    return bool(np.array_equal(left.to_numpy(dtype=object)[present], right.to_numpy(dtype=object)[present]))


def _column_matches(expected: pd.Series, produced: pd.Series, rule: ValueRule) -> bool:
    if rule.missing_text:
        expected = expected.where(~expected.isin(rule.missing_text))
        produced = produced.where(~produced.isin(rule.missing_text))
    expected_numbers, produced_numbers = _as_numbers(expected), _as_numbers(produced)
    if expected_numbers is not None and produced_numbers is not None:
        return _numbers_match(expected_numbers, produced_numbers, rule)
    return _text_matches(expected, produced, rule)


def _sort_values(column: pd.Series) -> pd.Series:
    numbers = _as_numbers(column) if column.notna().any() else None
    if numbers is not None:
        return pd.Series(numbers, index=column.index)
    return column.astype("string").str.strip()


def _sorted_by(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty or not columns:
        return frame.reset_index(drop=True)
    sort_frame = pd.DataFrame({index: _sort_values(frame[column]) for index, column in enumerate(columns)})
    order = sort_frame.sort_values(by=list(sort_frame.columns), kind="mergesort", na_position="last").index
    return frame.loc[order].reset_index(drop=True)


def compare_tables(
    expected: pd.DataFrame,
    produced: pd.DataFrame,
    *,
    key: Sequence[str],
    required_columns: Sequence[str] | None = None,
    rule: ValueRule = STRICT,
) -> TableComparison:
    """Compare ``produced`` with ``expected`` on the required columns.

    Args:
        expected: The correct table.
        produced: The table the pipeline built. Extra columns are ignored.
        key: Columns of ``expected`` that identify a row; both tables are sorted on them.
        required_columns: Columns to grade; defaults to every column of ``expected``.
        rule: How two values are compared.
    """
    expected, produced = to_text_frame(expected), to_text_frame(produced)
    expected_names = {name.lower(): name for name in expected.columns}
    required = [expected_names.get(name.lower(), name) for name in (required_columns or list(expected.columns))]

    produced_names: dict[str, str | None] = {}
    for name in produced.columns:
        # Two produced columns that differ only in case make the choice ambiguous: neither counts.
        produced_names[name.lower()] = None if name.lower() in produced_names else name

    present = [name for name in required if produced_names.get(name.lower()) and name in expected.columns]
    missing = tuple(name for name in required if name not in present)

    def nothing(reason: str) -> TableComparison:
        return TableComparison(matched=(), unmatched=tuple(present), missing=missing, reason=reason)

    if len(expected) != len(produced):
        return nothing(f"expected {len(expected)} rows, found {len(produced)}")
    key_columns = [expected_names[name.lower()] for name in key if name.lower() in expected_names]
    if any(name not in present for name in key_columns):
        return nothing("a key column is missing")
    # Without a usable key, fall back to ordering both tables by all graded columns they share.
    order_columns = key_columns or present
    expected_sorted = _sorted_by(expected, order_columns)
    produced_sorted = _sorted_by(produced, [produced_names[name.lower()] for name in order_columns])

    matched, unmatched = [], []
    for name in present:
        produced_column = produced_sorted[produced_names[name.lower()]]
        (matched if _column_matches(expected_sorted[name], produced_column, rule) else unmatched).append(name)
    if any(name in unmatched for name in key_columns):
        return nothing("the key values differ")
    return TableComparison(matched=tuple(matched), unmatched=tuple(unmatched), missing=missing)
