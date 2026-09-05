"""Tests for the out-of-sample protocol itself.

The protocol is the part of a forecasting study most likely to be quietly wrong
and least likely to be noticed, so the two properties that matter are asserted
directly: a model never sees a row whose target reaches into the block it is
about to predict, and every forecast is floored rather than allowed to go
negative and blow QLIKE up.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_altdata_benchmark import FLOOR, walk_forward  # noqa: E402


def _frame(n=600):
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"rv": np.linspace(0.1, 0.3, n),
                         "target": np.linspace(0.1, 0.3, n)}, index=idx)


def test_purge_removes_exactly_the_overlapping_rows():
    """With an h-day target, the last h training rows reach into the test block."""
    frame = _frame()
    seen = []

    def fp(train, history):
        seen.append(frame.index.get_loc(train.index[-1]))
        return pd.Series(0.2, index=history.index)

    walk_forward(frame, test_start=500, refit=21, fit_predict=fp,
                 min_train=100, purge=21)
    assert seen[0] == 500 - 21 - 1


def test_without_a_purge_the_training_window_runs_up_to_the_block():
    frame = _frame()
    seen = []

    def fp(train, history):
        seen.append(frame.index.get_loc(train.index[-1]))
        return pd.Series(0.2, index=history.index)

    walk_forward(frame, test_start=500, refit=21, fit_predict=fp,
                 min_train=100, purge=0)
    assert seen[0] == 499


def test_every_refit_sees_strictly_more_data_than_the_last():
    frame = _frame(n=700)
    sizes = []

    def fp(train, history):
        sizes.append(len(train))
        return pd.Series(0.2, index=history.index)

    walk_forward(frame, test_start=400, refit=21, fit_predict=fp,
                 min_train=100, purge=21)
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_forecasts_are_floored_not_left_negative():
    frame = _frame()

    def fp(train, history):
        return pd.Series(-5.0, index=history.index)

    preds = walk_forward(frame, test_start=500, refit=21, fit_predict=fp,
                         min_train=100, purge=21)
    assert (preds.dropna() == FLOOR).all()


def test_only_the_test_region_is_forecast():
    frame = _frame()
    preds = walk_forward(frame, test_start=500, refit=21,
                         fit_predict=lambda t, h: pd.Series(0.2, index=h.index),
                         min_train=100, purge=21)
    assert preds.iloc[:500].isna().all()
    assert preds.iloc[500:].notna().all()


def test_a_training_window_that_is_too_short_is_skipped_rather_than_fitted():
    frame = _frame(n=200)
    calls = []
    preds = walk_forward(frame, test_start=100, refit=21,
                         fit_predict=lambda t, h: calls.append(1) or
                         pd.Series(0.2, index=h.index),
                         min_train=500, purge=21)
    assert not calls
    assert preds.isna().all()
