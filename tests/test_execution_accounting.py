"""Independent arithmetic cases for the execution-accounting repair."""
import numpy as np
import pandas as pd
import pytest
from option_strategies import (StraddleConfig, max_drawdown, volatility_managed,
                               straddle_trade, straddle_trade_components)
from test_option_strategies import synthetic_chain, DATES, EXDATE, STRIKE


def test_drawdown_includes_starting_capital_peak():
    assert max_drawdown(pd.Series([-0.1])) == pytest.approx(-0.1)


def test_missing_forecast_does_not_drop_exposed_market_return():
    dates = pd.bdate_range("2020-01-01", periods=5)
    forecast = pd.Series([.15, .15, np.nan, .15, .15], index=dates)
    returns = pd.Series([0, 0, -.1, 0, 0], index=dates)
    result = volatility_managed(forecast, returns, cost_bps=0)
    assert result.loc[dates[2], "gross"] == pytest.approx(-.1)
    assert len(result) == 5


def test_constant_half_weight_still_pays_for_daily_rebalancing():
    dates = pd.bdate_range("2020-01-01", periods=5)
    result = volatility_managed(pd.Series(.30, index=dates),
                                pd.Series(.10, index=dates), cost_bps=0)
    assert result.iloc[3].turnover == pytest.approx(abs(.5 - .55 / 1.05))


def test_fees_reduce_equity_used_for_next_day_holdings_drift():
    dates = pd.bdate_range("2020-01-01", periods=5)
    result = volatility_managed(pd.Series(.30, index=dates),
                                pd.Series(.10, index=dates), cost_bps=100)
    assert result.iloc[2].net == pytest.approx(.05 - .005)
    assert result.iloc[3].turnover == pytest.approx(abs(.5 - .55 / 1.045))


def test_integrity_verifier_rejects_an_unreconciled_component(tmp_path):
    import shutil
    from pathlib import Path
    from run_integrity_report import build
    source = Path(__file__).resolve().parents[1] / "results"
    for path in source.glob("*.csv"):
        shutil.copyfile(path, tmp_path / path.name)
    frame = pd.read_csv(tmp_path / "option_pnl_straddles_purged.csv")
    frame.loc[0, "hedge_cost_total"] += .1
    frame.to_csv(tmp_path / "option_pnl_straddles_purged.csv", index=False)
    with pytest.raises(ValueError, match="components"):
        build(tmp_path)


def test_terminal_hedge_unwinds_prior_shares_without_exit_rebalance():
    dates = DATES[:3]
    chain = synthetic_chain([5]*3, [5]*3, dates=dates, call_delta=.6, put_delta=-.4)
    chain.loc[(chain.date == dates[-1]) & (chain.cp_flag == "C"), "delta"] = .2
    config = StraddleConfig(hold_days=2, spread_fraction=0, hedge_cost_bps=100)
    components = straddle_trade_components(chain, pd.Series(100., index=dates),
                                           dates[0], EXDATE, STRIKE, 1, config)
    assert components.hedge_cost.sum() == pytest.approx(-.04)
    assert components.iloc[-1].hedge_cost == pytest.approx(-.02)
    np.testing.assert_allclose(components.net, components.drop(columns="net").sum(axis=1))


@pytest.mark.parametrize("omitted", [0, 1, 2])
def test_missing_entry_interior_or_exit_quote_rejects_incomplete_path(omitted):
    dates = DATES[:3]
    chain = synthetic_chain([5]*3, [5]*3, dates=dates)
    chain = chain.loc[chain.date != dates[omitted]]
    assert straddle_trade(chain, pd.Series(100., index=dates), dates[0], EXDATE,
                          STRIKE, 1, StraddleConfig(hold_days=2)) is None


def test_sample_end_does_not_silently_shorten_hold():
    dates = DATES[:3]
    chain = synthetic_chain([5]*3, [5]*3, dates=dates)
    assert straddle_trade(chain, pd.Series(100., index=dates), dates[0], EXDATE,
                          STRIKE, 1, StraddleConfig(hold_days=3)) is None


def test_entry_spread_is_paid_on_entry_day():
    dates = DATES[:3]
    chain = synthetic_chain([5]*3, [5]*3, spread=.2, dates=dates)
    path = straddle_trade(chain, pd.Series(100., index=dates), dates[0], EXDATE,
                          STRIKE, 1, StraddleConfig(hold_days=2, delta_hedge=False))
    assert path.iloc[0] == pytest.approx(-.02)
    assert path.iloc[-1] == pytest.approx(-.02)
