"""Regression tests for calibration clocks, evaluation calendars and multiplicity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from option_strategies import StraddleConfig, straddle_signal, straddle_trade
from run_altdata_benchmark import make_hgb_fp, make_lasso_fp
from run_inference_audit import holm_adjust, marginal_audit, headline_reproduction
from vol_forecasting import purged_validation_split


def test_holm_matches_hand_calculated_stepdown_and_preserves_missing_baseline():
    raw = pd.Series([0.04, np.nan, 0.01, 0.03], index=list("abcd"))
    expected = pd.Series([0.06, np.nan, 0.03, 0.06], index=list("abcd"))
    pd.testing.assert_series_equal(holm_adjust(raw), expected)


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.inf])
def test_holm_refuses_invalid_pvalues(bad):
    with pytest.raises(ValueError):
        holm_adjust(pd.Series([0.01, bad]))


def test_holm_handles_ties_and_never_lowers_a_pvalue():
    raw = pd.Series([0.01, 0.01, 0.9])
    adjusted = holm_adjust(raw)
    assert adjusted.tolist() == pytest.approx([0.03, 0.03, 0.9])
    assert (adjusted >= raw).all()


def test_all_committed_feature_horizon_tests_belong_to_the_correction():
    table, summary = marginal_audit(Path(__file__).resolve().parent.parent / "results")
    assert dict(zip(summary.family, summary.n_tests)) == {"har": 48, "rich_har": 45}
    assert table.joint_family_n.eq(93).all()
    assert (table.p_holm_joint >= table.p_holm).all()


def test_committed_headlines_reproduce_from_forecast_paths():
    rows = headline_reproduction(Path(__file__).resolve().parent.parent / "results")
    assert len(rows) == 9
    assert rows.n_forecasts.eq(720).all()


def _signals():
    dates = pd.bdate_range("2020-01-01", periods=100)
    forecast = pd.Series(np.linspace(0.1, 0.3, 100), index=dates)
    implied = pd.Series(0.2, index=dates)
    return dates, forecast, implied


def test_rule_never_trades_calibration_and_starts_strictly_after_freeze():
    dates, forecast, implied = _signals()
    side, rule = straddle_signal(forecast, implied)
    assert side.iloc[:50].eq(0).all()
    assert rule["calibration_end"] == str(dates[49].date())
    assert rule["evaluation_start"] == str(dates[50].date())
    assert side.iloc[50:].ne(0).any()


def test_future_evaluation_data_cannot_change_an_earlier_trade():
    dates, forecast, implied = _signals()
    side, rule = straddle_signal(forecast, implied)
    bumped = implied.copy()
    bumped.iloc[70:] = 5.0
    other, other_rule = straddle_signal(forecast, bumped)
    assert rule == other_rule
    pd.testing.assert_series_equal(side.iloc[:70], other.iloc[:70])


def test_all_straddle_books_share_post_calibration_calendar(monkeypatch):
    import run_option_pnl
    dates, forecast, implied = _signals()
    seen = []

    def fake_backtest(chain, entries, underlying, side, config, long_only=False):
        seen.append(side)
        # Simulate model-dependent trade runoff. Every other book must include
        # the same tail as flat days when scored.
        end = len(underlying) if (side == -1).all() else len(dates)
        book = pd.Series(0.001, index=underlying.index[50:end])
        trades = pd.DataFrame({"trade_return": [0.01], "direction": [1]})
        return book, trades

    monkeypatch.setattr(run_option_pnl, "straddle_backtest", fake_backtest)
    under = pd.Series(100.0, index=pd.bdate_range(dates[0], periods=105))
    forecasts = pd.DataFrame({"har": forecast, "target": forecast})
    table, books, _ = run_option_pnl.run_straddles(
        forecasts, implied, pd.DataFrame(), pd.DataFrame(), under,
        StraddleConfig(), 0.5)
    assert all(side.index.min() == dates[51] for side in seen)
    assert table.n_days.eq(54).all()
    assert all(book.index.equals(next(iter(books.values())).index) for book in books.values())
    assert books["har__both"].iloc[-5:].eq(0.0).all()


def test_future_delta_cannot_establish_entry_hedge():
    dates = pd.bdate_range("2020-01-01", periods=3)
    rows = []
    for j, date in enumerate(dates):
        for cp, delta in (("C", 0.6), ("P", -0.4)):
            rows.append(dict(date=date, exdate=dates[-1], strike=100.0,
                             cp_flag=cp, best_bid=2.0, best_offer=2.2,
                             delta=np.nan if j == 0 else delta))
    path = straddle_trade(pd.DataFrame(rows), pd.Series(100.0, index=dates),
                          dates[0], dates[-1], 100.0, 1, StraddleConfig(hold_days=2))
    assert path is None


@pytest.mark.parametrize("horizon", [1, 5, 21])
def test_inner_training_targets_end_before_validation(horizon):
    frame = pd.DataFrame({"target": np.arange(200)})
    fit, validation = purged_validation_split(frame, 50, horizon)
    assert fit.index[-1] + horizon < validation.index[0]
    assert len(validation) == 50


@pytest.mark.parametrize("kind", ["lasso", "hgb"])
def test_adaptive_estimators_actually_use_the_purged_inner_fit(monkeypatch, kind):
    seen = []

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            seen.append(y.index[-1])
            return self

        def predict(self, X):
            return np.full(len(X), 0.2)

    if kind == "lasso":
        monkeypatch.setattr("sklearn.linear_model.Lasso", Estimator)
        factory = make_lasso_fp
    else:
        monkeypatch.setattr("sklearn.ensemble.HistGradientBoostingRegressor", Estimator)
        factory = make_hgb_fp
    frame = pd.DataFrame({c: np.linspace(0.1, 0.3, 600)
                          for c in ("rv", "rv_w", "rv_m", "target")})
    factory([], val_tail=100, horizon=21)(frame, frame)
    assert all(last + 21 < 500 for last in seen[:-1])
    assert seen[-1] == 599  # final refit may use the whole outer training window
