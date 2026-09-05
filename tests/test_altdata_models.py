"""Tests for the models added for the alternative-data study.

The linear ones are tested by construction: build data whose relationship to the
target is known exactly, fit, and check the coefficients come back. That is a
stronger test than "the loss went down", because it fails loudly when a
regressor is mis-aligned by a row.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lstm_forecasting import LSTMForecaster, make_sequences  # noqa: E402
from vol_forecasting import (  # noqa: E402
    HARRV,
    HARX,
    SHARRV,
    model_confidence_set,
    rolling_model_confidence_set,
    semivol,
)

TRADING_DAYS = 252


def _semivariance_frame(n=800, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    rs_pos = pd.Series(rng.gamma(2.0, 1e-5, size=n), index=idx)
    rs_neg = pd.Series(rng.gamma(2.0, 1e-5, size=n), index=idx)
    rv_var = rs_pos + rs_neg
    rv = np.sqrt(rv_var * TRADING_DAYS)
    return rv, rs_pos, rs_neg


# ---------------------------------------------------------------------------
# SHAR
# ---------------------------------------------------------------------------


def test_semivol_components_square_back_to_the_daily_regressor():
    rv, rs_pos, rs_neg = _semivariance_frame()
    assert np.allclose(semivol(rs_pos) ** 2 + semivol(rs_neg) ** 2, rv ** 2)


def test_shar_recovers_known_asymmetric_loadings():
    """Down-move variance predicts, up-move variance does not: SHAR must see it."""
    rv, rs_pos, rs_neg = _semivariance_frame()
    har = HARRV._features(rv)
    truth = (0.02 + 0.0 * semivol(rs_pos) + 1.5 * semivol(rs_neg)
             + 0.30 * har["rv_w"] + 0.10 * har["rv_m"])
    model = SHARRV().fit_shar(rv, rs_pos, rs_neg, truth)
    b0, b_pos, b_neg, b_w, b_m = model.coef_
    assert b0 == pytest.approx(0.02, abs=1e-8)
    assert b_pos == pytest.approx(0.0, abs=1e-8)
    assert b_neg == pytest.approx(1.5, abs=1e-8)
    assert b_w == pytest.approx(0.30, abs=1e-8)
    assert b_m == pytest.approx(0.10, abs=1e-8)


def test_shar_predictions_are_nan_before_the_monthly_window_fills():
    rv, rs_pos, rs_neg = _semivariance_frame()
    target = rv.shift(-1)
    model = SHARRV().fit_shar(rv, rs_pos, rs_neg, target)
    pred = model.predict_shar(rv, rs_pos, rs_neg)
    assert pred.iloc[:21].isna().all()
    assert pred.iloc[22:].notna().all()


def test_shar_collapses_to_har_when_the_two_semivariances_are_equal():
    """With RS+ = RS- every day the sign split carries no information.

    The two loadings then sum to what HAR would put on its daily term, which is
    the sanity check that the reparameterisation into volatility units did not
    quietly change the model.
    """
    rng = np.random.default_rng(11)
    n = 600
    idx = pd.bdate_range("2018-01-01", periods=n)
    half = pd.Series(rng.gamma(2.0, 1e-5, size=n), index=idx)
    rv = np.sqrt(2 * half * TRADING_DAYS)
    har = HARRV._features(rv)
    truth = 0.01 + 0.4 * har["rv_d"] + 0.3 * har["rv_w"] + 0.2 * har["rv_m"]
    shar = SHARRV().fit_shar(rv, half, half, truth)
    _, b_pos, b_neg, _, _ = shar.coef_
    # semivol(half) = rv / sqrt(2) for each side, so the two loadings must add to
    # sqrt(2) times HAR's daily loading
    assert (b_pos + b_neg) == pytest.approx(0.4 * np.sqrt(2), abs=1e-6)


# ---------------------------------------------------------------------------
# HAR-X and HAR-RV-IV
# ---------------------------------------------------------------------------


def _harx_frame(n=700, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    rv = pd.Series(0.15 + 0.03 * rng.normal(size=n).cumsum() / np.sqrt(n), index=idx)
    rv = rv.clip(lower=0.05)
    exog = pd.DataFrame({"iv2": (rv * 1.1) ** 2 + 0.001 * rng.normal(size=n)},
                        index=idx)
    return rv, exog


def test_harx_recovers_a_known_exogenous_coefficient():
    rv, exog = _harx_frame()
    har = HARRV._features(rv)
    truth = (0.01 + 0.2 * har["rv_d"] + 0.3 * har["rv_w"] + 0.1 * har["rv_m"]
             + 2.5 * exog["iv2"])
    model = HARX().fit_x(rv, exog, truth)
    b0, b_d, b_w, b_m, b_x = model.coef_
    assert b_x == pytest.approx(2.5, abs=1e-6)
    assert b_d == pytest.approx(0.2, abs=1e-6)
    assert b0 == pytest.approx(0.01, abs=1e-6)


def test_harx_with_a_constant_regressor_matches_plain_har():
    rv, _ = _harx_frame()
    target = rv.shift(-1)
    const = pd.DataFrame({"zero": np.zeros(len(rv))}, index=rv.index)
    har_pred = HARRV().fit(rv, target).predict(rv)
    harx_pred = HARX().fit_x(rv, const, target).predict_x(rv, const)
    both = pd.concat([har_pred, harx_pred], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_harx_is_immune_to_the_column_order_it_is_predicted_with():
    rv, exog = _harx_frame()
    exog = exog.assign(other=np.linspace(0.0, 1.0, len(rv)))
    target = rv.shift(-1)
    model = HARX().fit_x(rv, exog, target)
    a = model.predict_x(rv, exog)
    b = model.predict_x(rv, exog[["other", "iv2"]])
    pd.testing.assert_series_equal(a, b)


def test_harx_refuses_to_predict_before_it_is_fitted():
    rv, exog = _harx_frame()
    with pytest.raises(RuntimeError):
        HARX().predict_x(rv, exog)


# ---------------------------------------------------------------------------
# rolling MCS
# ---------------------------------------------------------------------------


def test_rolling_mcs_tracks_a_regime_change():
    """Model A is best in the first half, B in the second. The rolling MCS has to
    show A alone early, B alone late; a single full-sample MCS would show both."""
    n = 1200
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 0.02, size=(n, 2))
    a = np.where(np.arange(n) < n // 2, 0.10, 0.40) + noise[:, 0]
    b = np.where(np.arange(n) < n // 2, 0.40, 0.10) + noise[:, 1]
    losses = pd.DataFrame({"a": np.abs(a), "b": np.abs(b)}, index=idx)
    out = rolling_model_confidence_set(losses, window=300, step=100, n_boot=200)
    assert not out.empty
    early, late = out.iloc[0], out.iloc[-1]
    assert early["a"] and not early["b"]
    assert late["b"] and not late["a"]


def test_rolling_mcs_returns_an_empty_frame_when_the_window_never_fits():
    idx = pd.bdate_range("2020-01-01", periods=50)
    losses = pd.DataFrame({"a": np.ones(50), "b": np.ones(50) * 2}, index=idx)
    assert rolling_model_confidence_set(losses, window=500, step=21).empty


# ---------------------------------------------------------------------------
# LSTM plumbing
# ---------------------------------------------------------------------------


def test_sequences_are_labelled_by_the_last_row_of_the_window():
    features = np.arange(20, dtype=float).reshape(10, 2)
    target = np.arange(10, dtype=float)
    X, y, pos = make_sequences(features, target, length=3)
    assert X.shape == (8, 3, 2)
    assert list(pos) == list(range(2, 10))
    assert list(y) == [float(i) for i in range(2, 10)]
    assert np.array_equal(X[0], features[0:3])


def test_sequences_are_empty_when_the_sample_is_shorter_than_the_window():
    X, y, pos = make_sequences(np.zeros((2, 3)), np.zeros(2), length=5)
    assert len(X) == 0 and len(y) == 0 and len(pos) == 0


def _lstm_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    x = pd.Series(0.15 + 0.02 * rng.normal(size=n), index=idx)
    return pd.DataFrame({"a": x, "b": x.rolling(5).mean(),
                         "target": x.shift(-1)}, index=idx).dropna()


def test_lstm_scaling_uses_training_statistics_only():
    frame = _lstm_frame()
    train = frame.iloc[:400]
    model = LSTMForecaster(epochs=3, patience=2, val_tail=80).fit(train, ["a", "b"])
    lo_before = model._lo.copy()
    model.predict(frame)                      # predicting must not refit anything
    assert np.allclose(lo_before, model._lo)
    assert np.allclose(lo_before, train[["a", "b"]].to_numpy().min(axis=0))


def test_lstm_is_deterministic_under_a_fixed_seed():
    frame = _lstm_frame()
    train = frame.iloc[:400]
    a = LSTMForecaster(epochs=4, patience=2, val_tail=80, seed=1).fit(
        train, ["a", "b"]).predict(frame)
    b = LSTMForecaster(epochs=4, patience=2, val_tail=80, seed=1).fit(
        train, ["a", "b"]).predict(frame)
    pd.testing.assert_series_equal(a, b)


def test_lstm_refuses_to_predict_before_it_is_fitted():
    with pytest.raises(RuntimeError):
        LSTMForecaster().predict(_lstm_frame())
