"""Tests for the economic evaluation: volatility timing and delta-hedged straddles.

The straddle tests run against a synthetic option chain with hand-set quotes, so
the P&L can be checked against an arithmetic answer rather than against another
run of the same code. No licensed data and no network are involved.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from option_strategies import (  # noqa: E402
    StraddleConfig,
    newey_west_tstat,
    variance_swap_summary,
    max_drawdown,
    paired_block_bootstrap_pvalue,
    sharpe,
    straddle_backtest,
    straddle_signal,
    straddle_trade,
    variance_swap_pnl,
    volatility_managed,
)

DATES = pd.bdate_range("2020-01-01", periods=25)
EXDATE = pd.Timestamp("2020-03-20")
STRIKE = 100.0


def synthetic_chain(call_mid, put_mid, spread=0.0, call_delta=0.5, put_delta=-0.5,
                    dates=DATES) -> pd.DataFrame:
    """A two-leg chain with quotes set by the caller.

    `call_mid` and `put_mid` are arrays over `dates`; `spread` is the quoted
    bid-ask width applied to BOTH legs, so a round trip costs 2 * spread in
    straddle terms at the default spread_fraction of 0.5.
    """
    rows = []
    for cp, mids, delta in (("C", call_mid, call_delta), ("P", put_mid, put_delta)):
        for date, mid in zip(dates, mids):
            rows.append({"date": date, "exdate": EXDATE, "cp_flag": cp,
                         "strike": STRIKE, "best_bid": mid - spread / 2.0,
                         "best_offer": mid + spread / 2.0,
                         "delta": delta, "optionid": 1 if cp == "C" else 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# volatility-managed SPY
# ---------------------------------------------------------------------------


def test_weight_is_earned_on_the_next_day_not_the_same_day():
    idx = pd.bdate_range("2021-01-01", periods=5)
    forecast = pd.Series([0.15, 0.30, 0.15, 0.15, 0.15], index=idx)
    returns = pd.Series([0.0, 0.0, 0.10, 0.0, 0.0], index=idx)
    res = volatility_managed(forecast, returns, target_vol=0.15, cap=10.0, cost_bps=0.0)
    # the 0.30 forecast on day 2 halves the weight, and the halving must show up
    # in day 3's return, which is the only nonzero one
    assert res.loc[idx[2], "gross"] == pytest.approx(0.5 * 0.10)


def test_cap_binds_when_the_forecast_is_tiny():
    idx = pd.bdate_range("2021-01-01", periods=4)
    forecast = pd.Series(0.001, index=idx)
    returns = pd.Series(0.01, index=idx)
    res = volatility_managed(forecast, returns, target_vol=0.15, cap=2.0, cost_bps=0.0)
    assert (res["weight"] <= 2.0 + 1e-12).all()
    assert res["weight"].iloc[-1] == pytest.approx(2.0)


def test_turnover_is_charged_and_reduces_the_net_return():
    idx = pd.bdate_range("2021-01-01", periods=6)
    forecast = pd.Series([0.15, 0.15, 0.30, 0.15, 0.30, 0.15], index=idx)
    returns = pd.Series(0.0, index=idx)
    free = volatility_managed(forecast, returns, cost_bps=0.0)
    charged = volatility_managed(forecast, returns, cost_bps=100.0)
    assert charged["net"].sum() < free["net"].sum()
    assert charged["turnover"].sum() > 0


# ---------------------------------------------------------------------------
# straddles
# ---------------------------------------------------------------------------


def test_long_straddle_with_no_spread_and_no_hedge_is_the_change_in_mid():
    call = np.linspace(5.0, 9.0, len(DATES))
    put = np.full(len(DATES), 5.0)
    chain = synthetic_chain(call, put)
    under = pd.Series(100.0, index=DATES)
    config = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    path = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, +1, config)
    premium = call[0] + put[0]
    expected = ((call[10] + put[10]) - premium) / premium
    assert path.sum() == pytest.approx(expected)
    assert path.iloc[0] == 0.0


def test_short_straddle_is_the_mirror_image_when_costs_are_off():
    call = np.linspace(5.0, 9.0, len(DATES))
    put = np.full(len(DATES), 5.0)
    chain = synthetic_chain(call, put)
    under = pd.Series(100.0, index=DATES)
    config = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    long_path = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, +1, config)
    short_path = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, -1, config)
    assert short_path.sum() == pytest.approx(-long_path.sum())


def test_the_round_trip_pays_the_full_quoted_spread():
    """At spread_fraction = 0.5 the strategy buys at the ask and sells at the bid.

    With both legs quoted `spread` wide, that is a round-trip cost of 2 * spread
    in straddle terms: one full spread on the way in and one on the way out.
    """
    call = np.full(len(DATES), 5.0)
    put = np.full(len(DATES), 5.0)
    spread = 0.20
    under = pd.Series(100.0, index=DATES)
    free = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    paid = StraddleConfig(hold_days=10, spread_fraction=0.5, delta_hedge=False)
    flat = straddle_trade(synthetic_chain(call, put, spread=spread), under,
                          DATES[0], EXDATE, STRIKE, +1, free)
    costed = straddle_trade(synthetic_chain(call, put, spread=spread), under,
                            DATES[0], EXDATE, STRIKE, +1, paid)
    premium = call[0] + put[0]
    assert flat.sum() == pytest.approx(0.0)
    assert costed.sum() == pytest.approx(-2.0 * spread / premium)


def test_delta_hedge_removes_the_directional_leg():
    """A net-positive-delta straddle in a rising market: hedging must cost money.

    Deltas are set to +0.6 and -0.1 so the position carries +0.5 of delta. The
    hedge is short half a share, the underlying rises by 10, so the hedge loses
    5 dollars against an option book that is deliberately marked flat.
    """
    n = len(DATES)
    call, put = np.full(n, 5.0), np.full(n, 5.0)
    chain = synthetic_chain(call, put, call_delta=0.6, put_delta=-0.1)
    under = pd.Series(np.linspace(100.0, 110.0, n), index=DATES)
    unhedged = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    hedged = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=True,
                            hedge_cost_bps=0.0)
    flat = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, +1, unhedged)
    with_hedge = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, +1, hedged)
    premium = 10.0
    move = under.iloc[10] - under.iloc[0]
    assert flat.sum() == pytest.approx(0.0)
    assert with_hedge.sum() == pytest.approx(-0.5 * move / premium)


def test_hedge_transaction_costs_only_ever_subtract():
    n = len(DATES)
    chain = synthetic_chain(np.full(n, 5.0), np.full(n, 5.0),
                            call_delta=0.6, put_delta=-0.1)
    under = pd.Series(np.linspace(100.0, 110.0, n), index=DATES)
    free = StraddleConfig(hold_days=10, spread_fraction=0.0, hedge_cost_bps=0.0)
    charged = StraddleConfig(hold_days=10, spread_fraction=0.0, hedge_cost_bps=50.0)
    a = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, +1, free)
    b = straddle_trade(chain, under, DATES[0], EXDATE, STRIKE, +1, charged)
    assert b.sum() < a.sum()


def test_backtest_divides_the_book_by_the_holding_period():
    """Overlapping entries must not silently lever the strategy up."""
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    chain = synthetic_chain(np.linspace(5.0, 7.0, n), np.full(n, 5.0), dates=dates)
    entries = pd.DataFrame({"date": dates, "exdate": EXDATE, "strike": STRIKE})
    under = pd.Series(100.0, index=dates)
    signal = pd.Series(1, index=dates[:5])
    config = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    book, trades = straddle_backtest(chain, entries, under, signal, config)
    assert len(trades) == 5
    single = straddle_trade(chain, under, dates[0], EXDATE, STRIKE, +1, config)
    assert book.loc[dates[1]] == pytest.approx(single.loc[dates[1]] / 10.0)


def test_long_only_variant_drops_the_short_trades():
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    chain = synthetic_chain(np.linspace(5.0, 7.0, n), np.full(n, 5.0), dates=dates)
    entries = pd.DataFrame({"date": dates, "exdate": EXDATE, "strike": STRIKE})
    under = pd.Series(100.0, index=dates)
    signal = pd.Series([1, -1, 1, -1, 0], index=dates[:5])
    config = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    _, both = straddle_backtest(chain, entries, under, signal, config)
    _, longs = straddle_backtest(chain, entries, under, signal, config, long_only=True)
    assert len(both) == 4 and len(longs) == 2
    assert (longs["direction"] > 0).all()


# ---------------------------------------------------------------------------
# the trading rule and the model-free benchmark
# ---------------------------------------------------------------------------


def test_trading_rule_is_frozen_on_the_first_half():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    forecast = pd.Series(0.15 + 0.01 * rng.normal(size=400), index=idx)
    implied = pd.Series(0.18 + 0.01 * rng.normal(size=400), index=idx)
    _, rule = straddle_signal(forecast, implied)
    tampered = implied.copy()
    tampered.iloc[200:] = 5.0                     # absurd values, second half only
    _, rule2 = straddle_signal(forecast, tampered)
    assert rule["premium"] == pytest.approx(rule2["premium"])
    assert rule["margin"] == pytest.approx(rule2["margin"])


def test_trading_rule_is_two_sided_and_leaves_quiet_days_flat():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    forecast = pd.Series(0.15 + 0.02 * rng.normal(size=400), index=idx)
    implied = pd.Series(0.18 + 0.02 * rng.normal(size=400), index=idx)
    side, _ = straddle_signal(forecast, implied, margin_quantile=0.5)
    assert set(side.unique()).issubset({-1, 0, 1})
    assert (side == 0).mean() > 0.3
    assert (side == 1).any() and (side == -1).any()


def test_variance_swap_is_realized_minus_implied_in_variance_units():
    idx = pd.bdate_range("2020-01-01", periods=3)
    implied = pd.Series([0.20, 0.20, 0.20], index=idx)
    realized = pd.Series([0.10, 0.20, 0.30], index=idx)
    out = variance_swap_pnl(implied, realized)
    assert out.iloc[0] == pytest.approx(0.10 ** 2 - 0.20 ** 2)
    assert out.iloc[1] == pytest.approx(0.0)
    assert out.iloc[2] == pytest.approx(0.30 ** 2 - 0.20 ** 2)


# ---------------------------------------------------------------------------
# performance statistics
# ---------------------------------------------------------------------------


def test_max_drawdown_is_negative_and_finds_the_worst_run():
    r = pd.Series([0.1, -0.2, -0.2, 0.5], index=pd.bdate_range("2020-01-01", periods=4))
    assert max_drawdown(r) == pytest.approx(-0.4)


def test_paired_bootstrap_says_nothing_when_the_series_are_identical():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0004, 0.01, size=500), index=idx)
    diff, p = paired_block_bootstrap_pvalue(r, r, n_boot=400)
    assert diff == pytest.approx(0.0)
    assert p > 0.9


def test_paired_bootstrap_detects_a_real_sharpe_difference():
    idx = pd.bdate_range("2020-01-01", periods=1500)
    rng = np.random.default_rng(3)
    common = rng.normal(0.0, 0.01, size=1500)
    good = pd.Series(common + 0.0012, index=idx)
    bad = pd.Series(common - 0.0012, index=idx)
    diff, p = paired_block_bootstrap_pvalue(good, bad, n_boot=1000)
    assert diff > 0
    assert p < 0.05
    assert sharpe(good) > sharpe(bad)


def test_flat_days_are_zero_returns_not_missing_days():
    """A model that trades rarely must not get a Sharpe computed on trading days only."""
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    chain = synthetic_chain(np.linspace(5.0, 7.0, n), np.full(n, 5.0), dates=dates)
    entries = pd.DataFrame({"date": dates, "exdate": EXDATE, "strike": STRIKE})
    under = pd.Series(100.0, index=dates)
    signal = pd.Series(0, index=dates)
    signal.iloc[0] = 1                       # one trade, 59 flat days
    config = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    book, trades = straddle_backtest(chain, entries, under, signal, config)
    assert len(trades) == 1
    assert len(book) == n
    assert (book.iloc[11:] == 0.0).all()


# ---------------------------------------------------------------------------
# overlapping-payoff statistics
# ---------------------------------------------------------------------------


def test_newey_west_shrinks_the_tstat_of_an_overlapping_series():
    """An overlapping mean has a naive t-statistic inflated by the overlap.

    A 21-day rolling sum of i.i.d. noise is exactly that situation: consecutive
    observations share 20 of their 21 terms. The Newey-West statistic has to
    come out substantially smaller than the naive one.
    """
    rng = np.random.default_rng(9)
    raw = pd.Series(rng.normal(0.5, 1.0, size=2000),
                    index=pd.bdate_range("2018-01-01", periods=2000))
    overlapping = raw.rolling(21).mean().dropna()
    naive = overlapping.mean() / (overlapping.std() / np.sqrt(len(overlapping)))
    corrected = newey_west_tstat(overlapping, lag=20)
    assert corrected > 0
    assert corrected < naive / 2


def test_newey_west_matches_the_naive_tstat_at_zero_lag():
    rng = np.random.default_rng(10)
    x = pd.Series(rng.normal(0.3, 1.0, size=500),
                  index=pd.bdate_range("2018-01-01", periods=500))
    naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, lag=0) == pytest.approx(naive)


def test_newey_west_is_nan_on_a_series_too_short_to_judge():
    x = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=3))
    assert np.isnan(newey_west_tstat(x, lag=1))


def test_variance_swap_summary_separates_overlapping_from_independent_counts():
    idx = pd.bdate_range("2018-01-01", periods=420)
    payoff = pd.Series(-0.02, index=idx)
    out = variance_swap_summary(payoff, horizon=21)
    assert out["n_overlapping"] == 420
    assert out["n_non_overlapping"] == 20
    assert out["mean_variance_points"] == pytest.approx(-0.02)
    assert out["share_positive"] == 0.0
    assert out["worst_observation"] == pytest.approx(-0.02)


def test_variance_swap_summary_annualises_by_periods_per_year_not_by_days():
    """A 21-day payoff compounds about 12 times a year, not 252 times.

    Annualising an overlapping 21-day series with sqrt(252) overstates its
    Sharpe by roughly sqrt(21), which is how a variance-swap payoff ends up
    reported with a Sharpe of -12.
    """
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2018-01-01", periods=42 * 21)
    payoff = pd.Series(rng.normal(-0.02, 0.05, size=len(idx)), index=idx)
    out = variance_swap_summary(payoff, horizon=21)
    non_overlap = payoff.iloc[::21]
    expected = non_overlap.mean() / non_overlap.std() * np.sqrt(252 / 21)
    assert out["sharpe_annualised"] == pytest.approx(expected)
    assert abs(out["sharpe_annualised"]) < 5.0


def test_variance_swap_summary_is_empty_for_an_empty_payoff():
    assert variance_swap_summary(pd.Series(dtype=float)) == {}
