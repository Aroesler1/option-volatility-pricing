"""Synthetic-data checks for the corrected (purged) DM/MCS recomputation.

`run_integrity_report.build_dm_mcs` reads the committed per-date forecast
paths and calls `diebold_mariano`/`model_confidence_set` from
`vol_forecasting.py` directly, the same functions and parameters
(Newey-West lag h-1, alpha=0.10, 2000 bootstrap draws, seed 0) the
pre-repair README table used. These tests pin the wiring on synthetic data
so a lag or parameter regression would be caught without touching the real
results files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_integrity_report import build_dm_mcs
from run_vol_benchmark import qlike_series
from vol_forecasting import diebold_mariano, model_confidence_set


def _synthetic_frame(n=80, seed=0):
    """A benchmark ("har"), a genuinely better model and a noisier one."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    target = pd.Series(np.abs(rng.normal(0.20, 0.04, n)) + 0.05, index=dates)
    har = target + rng.normal(0, 0.015, n)
    better = target + rng.normal(0, 0.006, n)
    worse = target + rng.normal(0, 0.05, n)
    return pd.DataFrame({"date": dates, "target": target.to_numpy(),
                         "har": har.to_numpy(), "better": better.to_numpy(),
                         "worse": worse.to_numpy()})


def _write_purged_forecasts(root, horizon, frame):
    frame.to_csv(root / f"altdata_forecasts_h{horizon}_purged.csv", index=False)


def test_columns_and_benchmark_row_has_no_self_comparison(tmp_path):
    for horizon in (1, 5, 21):
        _write_purged_forecasts(tmp_path, horizon, _synthetic_frame(seed=horizon))

    result = build_dm_mcs(tmp_path)

    expected_cols = {"horizon", "model", "qlike_mean", "qlike_median",
                     "dm_vs_har", "p_vs_har", "n_obs", "mcs_pvalue", "in_mcs"}
    assert set(result.columns) == expected_cols
    assert set(result["horizon"]) == {1, 5, 21}
    assert set(result["model"]) == {"har", "better", "worse"}

    har_rows = result[result["model"] == "har"]
    assert har_rows["dm_vs_har"].isna().all()
    assert har_rows["p_vs_har"].isna().all()


def test_dm_and_mcs_match_direct_calls_with_the_horizon_specific_lag(tmp_path):
    for horizon in (1, 5, 21):
        _write_purged_forecasts(tmp_path, horizon, _synthetic_frame(seed=horizon))

    result = build_dm_mcs(tmp_path).set_index(["horizon", "model"])

    for horizon in (1, 5, 21):
        frame = _synthetic_frame(seed=horizon).set_index("date")
        loss_har = qlike_series(frame["har"], frame["target"])
        loss_better = qlike_series(frame["better"], frame["target"])
        loss_worse = qlike_series(frame["worse"], frame["target"])
        dm_lag = max(horizon - 1, 1)

        stat_b, p_b = diebold_mariano(loss_better, loss_har, lag=dm_lag)
        stat_w, p_w = diebold_mariano(loss_worse, loss_har, lag=dm_lag)
        assert result.loc[(horizon, "better"), "dm_vs_har"] == pytest.approx(stat_b)
        assert result.loc[(horizon, "better"), "p_vs_har"] == pytest.approx(p_b)
        assert result.loc[(horizon, "worse"), "dm_vs_har"] == pytest.approx(stat_w)
        assert result.loc[(horizon, "worse"), "p_vs_har"] == pytest.approx(p_w)

        # "better" is built to beat HAR at 5%, with a negative DM statistic
        # (lower loss); "worse" is built to lose to it.
        assert stat_b < 0 and p_b < 0.05
        assert stat_w > 0

        losses = pd.DataFrame({"har": loss_har, "better": loss_better,
                               "worse": loss_worse}).dropna()
        mcs = model_confidence_set(losses, alpha=0.10, n_boot=2000, seed=0)
        for model in ("har", "better", "worse"):
            assert result.loc[(horizon, model), "mcs_pvalue"] == pytest.approx(
                mcs.loc[model, "mcs_pvalue"])
            assert bool(result.loc[(horizon, model), "in_mcs"]) == bool(
                mcs.loc[model, "in_mcs"])

        # a coarser lag (ignoring the h-1 convention) must not silently give the
        # same answer, so the test actually pins the lag choice, not just that
        # *a* lag was used
        if dm_lag > 1:
            _, p_wrong_lag = diebold_mariano(loss_worse, loss_har, lag=1)
            assert not np.isclose(p_w, p_wrong_lag)


def test_mismatched_stored_table_fails_the_offline_check(tmp_path):
    """The --check flow compares build_dm_mcs's recomputation against the
    stored integrity_dm_mcs.csv with pd.testing.assert_frame_equal; a stale or
    hand-edited number must fail that comparison rather than pass silently.
    """
    for horizon in (1, 5, 21):
        _write_purged_forecasts(tmp_path, horizon, _synthetic_frame(seed=horizon))
    stored = build_dm_mcs(tmp_path)
    stored.to_csv(tmp_path / "integrity_dm_mcs.csv", index=False)

    tampered = stored.copy()
    tampered.loc[tampered["model"] == "worse", "p_vs_har"] += 0.2
    tampered.to_csv(tmp_path / "integrity_dm_mcs.csv", index=False)

    recomputed = build_dm_mcs(tmp_path)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            recomputed, pd.read_csv(tmp_path / "integrity_dm_mcs.csv"),
            check_exact=False, rtol=1e-10, atol=1e-10)
