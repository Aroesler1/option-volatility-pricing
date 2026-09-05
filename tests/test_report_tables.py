"""Tests for the README table generator.

Its whole purpose is to stop numbers being transcribed by hand, so the two
things worth pinning are that it does not silently mangle a value and that a
missing statistic renders as an empty cell rather than as "nan".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from report_tables import _fmt, markdown  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    (0.123456, "0.1235"),
    (-2.5, "-2.5000"),
    (3, "3"),
    (np.int64(7), "7"),
    (True, "yes"),
    (False, "no"),
    (np.bool_(True), "yes"),
    ("har_rv_iv", "har_rv_iv"),
    (np.nan, ""),
    (None, ""),
    (np.inf, ""),
])
def test_every_cell_type_renders_the_way_a_reader_expects(value, expected):
    assert _fmt(value) == expected


def test_a_missing_statistic_is_blank_not_the_word_nan():
    frame = pd.DataFrame({"model": ["har"], "p": [np.nan]})
    out = markdown(frame, {"model": "model", "p": "p"})
    assert "nan" not in out.lower()
    assert out.splitlines()[-1] == "| har |  |"


def test_columns_absent_from_the_frame_are_skipped_not_invented():
    frame = pd.DataFrame({"model": ["har"], "qlike_mean": [0.2]})
    out = markdown(frame, {"model": "model", "qlike_mean": "QLIKE",
                           "not_there": "missing"})
    assert "missing" not in out
    assert out.splitlines()[0] == "| model | QLIKE |"


def test_the_header_separator_matches_the_column_count():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    lines = markdown(frame, {"a": "a", "b": "b", "c": "c"}).splitlines()
    assert lines[1] == "|---|---|---|"
    assert lines[2].count("|") == lines[0].count("|")
