"""Tests for the realized-variance rebuild.

The committed daily series is the input every other result in this repository
depends on, so the two things that would silently corrupt it are asserted here:
the sampling grid (which session minutes are used, and whether the overnight gap
sneaks in as one enormous return) and the estimator arithmetic.

No licensed data is touched; the bars are synthetic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_intraday_rv import (  # noqa: E402
    TRADING_DAYS,
    daily_estimators,
    five_minute_returns,
)


def _bars(days, minute_price):
    """1-minute OHLCV bars in UTC, covering 04:00 to 20:00 New York each day."""
    rows = []
    for day in days:
        session = pd.date_range(f"{day} 04:00", f"{day} 19:59", freq="1min",
                                tz="America/New_York")
        for i, ts in enumerate(session):
            rows.append({"ts_event": ts.tz_convert("UTC"),
                         "close": minute_price(day, i, ts), "symbol": "SPY"})
    frame = pd.DataFrame(rows).set_index("ts_event")
    return frame


def test_returns_are_sampled_on_the_0935_to_1600_grid():
    bars = _bars(["2024-06-03"], lambda d, i, ts: 100.0 + i * 0.01)
    ret = five_minute_returns(bars)
    stamps = ret.index.strftime("%H:%M").tolist()
    assert stamps[0] == "09:40"          # the first RETURN ends at 09:40
    assert stamps[-1] == "16:00"
    assert len(ret) == 77                # 78 closes on the grid, 77 returns


def test_the_overnight_gap_is_not_counted_as_a_return():
    """Two days with a large overnight jump between them.

    If the session boundary were not handled, the first return of day two would
    be the whole gap and realized variance would be dominated by it.
    """
    def price(day, i, ts):
        return 100.0 if day == "2024-06-03" else 150.0

    bars = _bars(["2024-06-03", "2024-06-04"], price)
    ret = five_minute_returns(bars)
    assert len(ret) == 2 * 77
    assert np.allclose(ret.to_numpy(), 0.0)


def test_pre_and_post_market_minutes_are_excluded():
    def price(day, i, ts):
        # a violent move that happens only outside the regular session
        return 200.0 if ts.strftime("%H:%M") < "09:30" else 100.0

    bars = _bars(["2024-06-03"], price)
    ret = five_minute_returns(bars)
    assert np.allclose(ret.to_numpy(), 0.0)


def _synthetic_returns(values, day="2024-06-03"):
    idx = pd.date_range(f"{day} 09:40", periods=len(values), freq="5min",
                        tz="America/New_York")
    return pd.Series(values, index=idx)


def test_semivariances_split_by_sign_and_add_back_to_realized_variance():
    ret = _synthetic_returns([0.01, -0.02, 0.03, -0.01] * 10)
    out = daily_estimators(ret, min_buckets=5)
    row = out.iloc[0]
    assert row["rs_pos_var"] == pytest.approx(10 * (0.01 ** 2 + 0.03 ** 2))
    assert row["rs_neg_var"] == pytest.approx(10 * (0.02 ** 2 + 0.01 ** 2))
    assert row["rs_pos_var"] + row["rs_neg_var"] == pytest.approx(row["rv5m_var"])


def test_a_zero_return_counts_in_neither_semivariance():
    ret = _synthetic_returns([0.0] * 10 + [0.01] * 10)
    row = daily_estimators(ret, min_buckets=5).iloc[0]
    assert row["rs_neg_var"] == 0.0
    assert row["rs_pos_var"] == pytest.approx(10 * 0.01 ** 2)


def test_quarticity_follows_the_n_over_three_convention():
    ret = _synthetic_returns([0.01] * 30)
    row = daily_estimators(ret, min_buckets=5).iloc[0]
    assert row["rq5m"] == pytest.approx((30 / 3.0) * 30 * 0.01 ** 4)


def test_annualised_volatility_is_the_root_of_252_times_the_variance():
    ret = _synthetic_returns([0.01, -0.01] * 20)
    row = daily_estimators(ret, min_buckets=5).iloc[0]
    assert row["rv5m"] == pytest.approx(np.sqrt(row["rv5m_var"] * TRADING_DAYS))


def test_short_sessions_are_dropped_rather_than_reported_as_calm():
    """A half day carries far fewer buckets, so its RV is mechanically small.

    Keeping it would put a spuriously low realized variance into the series on
    exactly the dates around holidays.
    """
    full = _synthetic_returns([0.01] * 40, day="2024-06-03")
    short = _synthetic_returns([0.01] * 10, day="2024-06-04")
    out = daily_estimators(pd.concat([full, short]), min_buckets=30)
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-06-03")
