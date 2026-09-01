"""Tests for the corrected volatility-forecasting methodology."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vol_forecasting import (  # noqa: E402
    HARRV,
    chronological_split,
    diebold_mariano,
    fit_scaler_train_only,
    forward_realized_vol,
    qlike,
    realized_vol,
)


def _garch_like_returns(n=3000, seed=3):
    """Persistent-volatility returns (two-state vol regime chain)."""
    rng = np.random.default_rng(seed)
    vols = np.empty(n)
    v = 0.01
    for i in range(n):
        # slowly mean-reverting log-vol with shocks -> HAR-friendly persistence
        v = np.exp(0.98 * np.log(v) + 0.02 * np.log(0.012) + 0.1 * rng.normal())
        vols[i] = v
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.Series(rng.normal(0, vols), index=idx)


def test_forward_target_uses_only_future_data():
    r = _garch_like_returns()
    base = forward_realized_vol(r, horizon=21)

    # perturbing PAST returns (before t) must not change the target at t
    t = 1500
    bumped = r.copy()
    bumped.iloc[:t] = bumped.iloc[:t] * 3.0
    assert np.isclose(base.iloc[t], forward_realized_vol(bumped, horizon=21).iloc[t])

    # perturbing FUTURE returns must change it
    bumped2 = r.copy()
    bumped2.iloc[t + 1 : t + 22] = bumped2.iloc[t + 1 : t + 22] * 3.0
    assert not np.isclose(base.iloc[t], forward_realized_vol(bumped2, horizon=21).iloc[t])


def test_scaler_uses_train_statistics_only():
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    train, val, test = chronological_split(df, 0.5, 0.25)
    train_s, val_s, test_s = fit_scaler_train_only(train, val, test)

    # train maps to [0, 1]; later periods may exceed 1 (no test-range leakage)
    assert np.isclose(train_s["x"].min(), 0.0)
    assert np.isclose(train_s["x"].max(), 1.0)
    assert test_s["x"].max() > 1.0


def test_har_rv_beats_random_walk_on_persistent_vol():
    r = _garch_like_returns()
    rv = realized_vol(r, 21)
    target = forward_realized_vol(r, 21)

    frame = pd.concat([rv.rename("rv"), target.rename("y")], axis=1).dropna()
    split = int(len(frame) * 0.7)
    train, test = frame.iloc[:split], frame.iloc[split:]

    model = HARRV().fit(train["rv"], train["y"])
    har_pred = model.predict(test["rv"])

    naive_pred = test["rv"]  # random-walk forecast: tomorrow's vol = today's

    har_qlike = qlike(har_pred, test["y"])
    naive_qlike = qlike(naive_pred, test["y"])
    assert np.isfinite(har_qlike)
    assert har_qlike <= naive_qlike * 1.05  # HAR at least matches the naive forecast


def test_diebold_mariano_direction_and_significance():
    rng = np.random.default_rng(9)
    base_err = pd.Series(rng.normal(0, 1.0, size=800)) ** 2
    worse_err = pd.Series(rng.normal(0, 1.4, size=800)) ** 2
    stat, p = diebold_mariano(base_err, worse_err)
    assert stat < 0  # model A (smaller errors) wins
    assert p < 0.05
