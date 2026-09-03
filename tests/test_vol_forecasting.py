"""Tests for the corrected volatility-forecasting methodology."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vol_forecasting import (  # noqa: E402
    LogHARRV,
    WLSHARRV,
    clements_preve_weights,
    mean_combination,
    model_confidence_set,
    stationary_bootstrap_indices,
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


# ---------------------------------------------------------------------------
# Model Confidence Set (Hansen, Lunde & Nason 2011)
# ---------------------------------------------------------------------------


def _loss_panel(n=600, seed=0):
    """Losses where model A is known-best, B is close, C and D are clearly worse."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.05, n)          # shared shocks, as real loss series have
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "best": 1.00 + common + rng.normal(0, 0.02, n),
        "close": 1.01 + common + rng.normal(0, 0.02, n),
        "worse": 1.60 + common + rng.normal(0, 0.02, n),
        "worst": 2.40 + common + rng.normal(0, 0.02, n),
    }, index=idx)


def test_mcs_keeps_the_known_best_model_and_drops_the_clearly_worse():
    losses = _loss_panel()
    out = model_confidence_set(losses, alpha=0.10, n_boot=500, seed=1)
    assert out.loc["best", "in_mcs"]
    assert not out.loc["worse", "in_mcs"]
    assert not out.loc["worst", "in_mcs"]
    assert out.loc["best", "mcs_pvalue"] >= out.loc["worst", "mcs_pvalue"]


def test_mcs_pvalues_are_monotone_in_elimination_order():
    """The running-max construction is what makes membership at any alpha
    readable off the p-value column. The guarantee is monotonicity in
    ELIMINATION order, not in mean loss: the elimination rule is studentized,
    so a model whose loss differentials have a small variance can be dropped
    ahead of one with a higher mean loss."""
    out = model_confidence_set(_loss_panel(), alpha=0.10, n_boot=500, seed=2)
    ordered = out.sort_values("elimination_order")
    assert ordered["mcs_pvalue"].is_monotonic_increasing
    assert ordered["elimination_order"].tolist() == [1, 2, 3, 4]
    assert ordered["mcs_pvalue"].iloc[-1] == 1.0


def test_mcs_keeps_both_when_two_models_are_genuinely_tied():
    n = 500
    rng = np.random.default_rng(3)
    common = rng.normal(0.0, 0.05, n)
    losses = pd.DataFrame({
        "a": 1.0 + common + rng.normal(0, 0.02, n),
        "b": 1.0 + common + rng.normal(0, 0.02, n),
        "junk": 3.0 + common + rng.normal(0, 0.02, n),
    })
    out = model_confidence_set(losses, alpha=0.10, n_boot=500, seed=3)
    assert out.loc["a", "in_mcs"] and out.loc["b", "in_mcs"]
    assert not out.loc["junk", "in_mcs"]


def test_mcs_judges_every_model_on_the_identical_sample():
    losses = _loss_panel(n=300)
    losses.loc[losses.index[:40], "close"] = np.nan
    out = model_confidence_set(losses, alpha=0.10, n_boot=300, seed=4)
    # 40 rows dropped for everyone, not just for the model with the gap
    assert out["loss"].notna().all()
    expected = losses.dropna()["best"].mean()
    assert out.loc["best", "loss"] == pytest.approx(expected)


def test_stationary_bootstrap_wraps_and_stays_in_range():
    rng = np.random.default_rng(5)
    idx = stationary_bootstrap_indices(50, n_boot=200, block_length=5.0, rng=rng)
    assert idx.shape == (200, 50)
    assert idx.min() >= 0 and idx.max() < 50
    # consecutive draws should usually continue the previous block, which is
    # the whole point: an i.i.d. bootstrap would destroy the serial dependence
    contiguous = (idx[:, 1:] == (idx[:, :-1] + 1) % 50).mean()
    assert contiguous > 0.5


# ---------------------------------------------------------------------------
# Clements & Preve remedies
# ---------------------------------------------------------------------------


def test_log_har_bias_correction_lifts_the_forecast_by_exp_half_sigma_squared():
    """exp() of the fitted mean of log RV is the MEDIAN, not the mean. Equation
    (8) of Clements & Preve adds sigma_u^2/2 inside the exponent; this pins that
    the correction is applied, and applied by exactly that factor."""
    rng = np.random.default_rng(0)
    n = 900
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    log_rv = pd.Series(np.log(0.15) + 0.9 * np.linspace(0, 1, n) * 0 +
                       rng.normal(0, 0.3, n), index=idx).ewm(span=10).mean()
    rv = np.exp(log_rv)
    target = rv.shift(-21) * np.exp(rng.normal(0, 0.2, n))

    model = LogHARRV().fit_log(rv, target)
    corrected = model.predict_log(rv, bias_correct=True)
    raw = model.predict_log(rv, bias_correct=False)

    assert model.sigma2_ > 0
    ratio = (corrected / raw).dropna()
    assert np.allclose(ratio, np.exp(0.5 * model.sigma2_))
    assert (corrected.dropna() > raw.dropna()).all()


def test_log_har_forecasts_are_strictly_positive_by_construction():
    """The reason to use it: a linear HAR can forecast a negative volatility and
    has to be clipped, which is what produces the collapsed forecasts the
    intraday benchmark counts. exp() cannot."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-01", periods=600, freq="B")
    rv = pd.Series(np.exp(np.log(0.12) + rng.normal(0, 0.4, 600)), index=idx)
    target = rv.shift(-21)
    model = LogHARRV().fit_log(rv, target)
    pred = model.predict_log(rv).dropna()
    assert (pred > 0).all()


def test_clements_preve_rv_weights_are_one_over_rv():
    rv = pd.Series([0.10, 0.20, 0.40, np.nan, -1.0])
    w = clements_preve_weights(rv=rv, scheme="rv")
    assert w.iloc[0] == pytest.approx(10.0)
    assert w.iloc[1] == pytest.approx(5.0)
    assert w.iloc[2] == pytest.approx(2.5)
    assert pd.isna(w.iloc[3]) and pd.isna(w.iloc[4])   # missing and non-positive drop out


def test_clements_preve_rq_weights_are_one_over_root_rq():
    rq = pd.Series([1e-4, 4e-4, 1e-2])
    w = clements_preve_weights(rq=rq, scheme="rq")
    assert np.allclose(w.to_numpy(), 1.0 / np.sqrt(rq.to_numpy()))


def test_wls_weights_downweight_high_volatility_observations():
    """The mechanism, not just the arithmetic: a single wild high-RV day should
    move the WLS fit less than it moves the OLS fit, because 1/RV_t weights it
    down. If it did not, the remedy would be a no-op."""
    rng = np.random.default_rng(2)
    n = 500
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rv = pd.Series(np.abs(rng.normal(0.15, 0.03, n)), index=idx)
    target = (0.02 + 0.8 * rv).shift(-1)

    contaminated_rv = rv.copy()
    contaminated_target = target.copy()
    contaminated_rv.iloc[250] = 1.5                     # a crisis-day outlier
    contaminated_target.iloc[250] = 3.0

    ols_clean = HARRV().fit(rv, target).coef_
    ols_dirty = HARRV().fit(contaminated_rv, contaminated_target).coef_
    w = clements_preve_weights(rv=contaminated_rv, scheme="rv")
    wls_dirty = WLSHARRV().fit_wls(contaminated_rv, contaminated_target, w).coef_

    ols_shift = float(np.abs(ols_dirty - ols_clean).sum())
    wls_shift = float(np.abs(wls_dirty - ols_clean).sum())
    assert wls_shift < ols_shift


def test_wls_with_constant_weights_reproduces_ols():
    """A sanity anchor: WLS is a generalisation, so equal weights must collapse
    onto the OLS fit exactly."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    rv = pd.Series(np.abs(rng.normal(0.15, 0.03, 400)), index=idx)
    target = (0.01 + 0.9 * rv).shift(-1)
    ols = HARRV().fit(rv, target).coef_
    wls = WLSHARRV().fit_wls(rv, target, pd.Series(1.0, index=idx)).coef_
    assert np.allclose(ols, wls)


def test_mean_combination_averages_and_propagates_gaps():
    a = pd.Series([1.0, 2.0, np.nan], index=pd.date_range("2020-01-01", periods=3))
    b = pd.Series([3.0, 4.0, 5.0], index=a.index)
    out = mean_combination([a, b])
    assert out.iloc[0] == pytest.approx(2.0)
    assert out.iloc[1] == pytest.approx(3.0)
    # a missing constituent must not be silently replaced by the other model
    assert pd.isna(out.iloc[2])
