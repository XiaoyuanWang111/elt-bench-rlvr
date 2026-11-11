"""Table comparison: row alignment on composite, re-cased and reformatted keys, and the edges of the column rules."""

import pandas as pd

from elt_rlvr.grading.compare import OFFICIAL, STRICT, TableComparison, compare_tables, read_ground_truth


def test_compare_composite_key_aligns_rows_on_every_key_column():
    expected = pd.DataFrame({"store": [1, 1, 2], "day": ["mon", "tue", "mon"], "sales": [10, 20, 30]})
    produced = pd.DataFrame({"day": ["mon", "tue", "mon"], "store": [2, 1, 1], "sales": [30, 20, 10]})

    assert compare_tables(expected, produced, key=["store", "day"]).is_full_match


def test_compare_composite_key_with_one_wrong_key_column_matches_nothing():
    expected = pd.DataFrame({"store": [1, 1], "day": ["mon", "tue"], "sales": [10, 20]})
    produced = pd.DataFrame({"store": [1, 1], "day": ["mon", "wed"], "sales": [10, 20]})

    result = compare_tables(expected, produced, key=["store", "day"])

    assert result.matched == ()
    assert result.reason == "the key values differ"


def test_compare_rows_with_equal_keys_line_up_on_the_other_columns():
    expected = pd.DataFrame({"team": ["a", "a", "b"], "player": ["x", "y", "z"], "goals": [1, 2, 3]})
    produced = expected.iloc[[1, 0, 2]].reset_index(drop=True)

    result = compare_tables(expected, produced, key=["team"])

    assert result.matched == ("team", "player", "goals")


def test_compare_key_given_in_another_case_is_found():
    expected = pd.DataFrame({"id": [1, 2], "total": [5, 7]})
    produced = pd.DataFrame({"id": [2, 1], "total": [7, 5]})

    assert compare_tables(expected, produced, key=["ID"]).is_full_match


def test_compare_numeric_key_written_as_integer_or_decimal_aligns_the_same_rows():
    expected = pd.DataFrame({"id": ["1", "2", "10"], "label": ["a", "b", "c"]})
    produced = pd.DataFrame({"id": [10.0, 1.0, 2.0], "label": ["c", "a", "b"]})

    result = compare_tables(expected, produced, key=["id"])

    assert result.is_full_match
    assert result.matched == ("id", "label")


def test_compare_text_that_looks_like_a_missing_value_matches_the_same_text():
    # "NA" is also the country code of Namibia.
    expected = pd.DataFrame({"country": ["US", "NA", "MX"], "orders": [3, 1, 2]})
    produced = pd.DataFrame({"country": ["MX", "US", "NA"], "orders": [2, 3, 1]})

    assert compare_tables(expected, produced, key=["country"]).is_full_match


def test_compare_missing_value_does_not_match_text_that_looks_missing_under_the_reward_rule():
    expected = pd.DataFrame({"id": [1, 2, 3], "country": ["US", "NA", "MX"]})
    produced = pd.DataFrame({"id": [1, 2, 3], "country": ["US", None, "MX"]})

    assert compare_tables(expected, produced, key=["id"], rule=STRICT).unmatched == ("country",)
    assert compare_tables(expected, produced, key=["id"], rule=OFFICIAL).is_full_match


def test_compare_empty_text_and_missing_value_are_the_same_cell():
    expected = pd.DataFrame({"id": [1, 2], "note": ["a", None]})
    produced = pd.DataFrame({"id": [1, 2], "note": ["a", ""]})

    assert compare_tables(expected, produced, key=["id"]).is_full_match


def test_compare_exported_float_nan_matches_a_missing_number():
    # A warehouse exports a float NaN as the text NaN; a frame read from the warehouse holds it as missing.
    expected = pd.DataFrame({"id": ["1", "2", "3"], "ratio": ["0.5", "NaN", "2"]})
    produced = pd.DataFrame({"id": [1, 2, 3], "ratio": [0.5, float("nan"), 2.0]})

    assert compare_tables(expected, produced, key=["id"]).is_full_match


def test_compare_nan_among_other_text_is_text():
    expected = pd.DataFrame({"id": [1, 2], "name": ["NaN", "Ada"]})
    produced = pd.DataFrame({"id": [1, 2], "name": [None, "Ada"]})

    assert compare_tables(expected, produced, key=["id"]).unmatched == ("name",)


def test_read_ground_truth_keeps_words_that_pandas_reads_as_missing(tmp_path):
    path = tmp_path / "model.csv"
    path.write_text("id,status,note\n1,null,\n2,active,NA\n", encoding="utf-8")

    frame = read_ground_truth(path)

    assert frame["status"].tolist() == ["null", "active"]
    assert frame["note"].isna().tolist() == [True, False]


def test_compare_required_columns_limit_the_graded_columns():
    expected = pd.DataFrame({"id": [1, 2], "total": [5, 7], "note": ["a", "b"]})
    produced = pd.DataFrame({"id": [1, 2], "TOTAL": [5, 7], "note": ["wrong", "wrong"]})

    result = compare_tables(expected, produced, key=["id"], required_columns=["id", "Total"])

    assert result == TableComparison(matched=("id", "total"), unmatched=(), missing=())


def test_compare_required_column_absent_from_the_produced_table_is_missing():
    expected = pd.DataFrame({"id": [1, 2], "total": [5, 7], "note": ["a", "b"]})
    produced = pd.DataFrame({"id": [1, 2], "total": [5, 7]})

    result = compare_tables(expected, produced, key=["id"], required_columns=["id", "note"])

    assert result.matched == ("id",)
    assert result.missing == ("note",)
    assert not result.is_full_match


def test_compare_empty_tables_with_the_same_columns_match():
    expected = pd.DataFrame({"id": pd.Series([], dtype="int64"), "name": pd.Series([], dtype="object")})
    produced = pd.DataFrame({"id": pd.Series([], dtype="float64"), "name": pd.Series([], dtype="object")})

    assert compare_tables(expected, produced, key=["id"]).is_full_match


def test_compare_empty_produced_table_matches_no_column_of_a_non_empty_one():
    expected = pd.DataFrame({"id": [1], "name": ["a"]})
    produced = pd.DataFrame({"id": pd.Series([], dtype="int64"), "name": pd.Series([], dtype="object")})

    result = compare_tables(expected, produced, key=["id"])

    assert result.matched == ()
    assert result.reason == "expected 1 rows, found 0"


def test_compare_without_a_key_aligns_rows_on_all_graded_columns():
    expected = pd.DataFrame({"city": ["Boise", "Austin"], "visits": [4, 9]})
    produced = pd.DataFrame({"city": ["Austin", "Boise"], "visits": [9, 4]})

    assert compare_tables(expected, produced, key=[]).is_full_match


def test_compare_without_a_key_reports_the_wrong_column():
    expected = pd.DataFrame({"city": ["Austin", "Boise"], "visits": [9, 4]})
    produced = pd.DataFrame({"city": ["Austin", "Boise"], "visits": [4, 9]})

    result = compare_tables(expected, produced, key=[])

    assert result.matched == ("city",)
    assert result.unmatched == ("visits",)
