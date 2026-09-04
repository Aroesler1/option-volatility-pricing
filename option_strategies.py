"""Turn volatility forecasts into money, or fail to, and measure which.

Two economic tests, plus one model-free check.

1. `volatility_managed` scales SPY exposure to a constant volatility target.
   This is the cheapest possible use of a variance forecast and needs no option
   market at all, so it isolates whether a better forecast is worth anything.

2. `straddle_backtest` trades the forecast against the option market: buy an
   at-the-money straddle when the forecast says realized variance will exceed
   what the option is priced at, sell it when the reverse, delta-hedged daily so
   the position is a bet on variance rather than on direction.

3. `variance_swap_pnl` is the model-free version of test 2: the payoff to being
   long a synthetic variance swap struck at VIX squared. It uses no forecast at
   all, so it measures the variance risk premium itself, which is the thing any
   straddle strategy is really trading around (Carr and Wu, RFS 2009).

Costs are charged in every case and stated, because a volatility-timing
strategy's turnover is exactly where its edge goes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Shared performance statistics
# ---------------------------------------------------------------------------


def sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    sd = r.std()
    if not np.isfinite(sd) or sd == 0 or len(r) < 2:
        return np.nan
    return float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough fall of the cumulative arithmetic P&L path.

    Arithmetic rather than compounded because the straddle P&L is a return on a
    fixed premium budget, not on a compounding equity balance, and the two
    strategies have to be measured the same way to be comparable.
    """
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = r.cumsum()
    return float((equity - equity.cummax()).min())


def worst_month(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return np.nan
    return float(r.groupby([r.index.year, r.index.month]).sum().min())


def performance(returns: pd.Series, turnover: Optional[pd.Series] = None) -> dict:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    out = {
        "n_days": int(len(r)),
        "mean_ann": float(r.mean() * TRADING_DAYS),
        "vol_ann": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "worst_month": worst_month(r),
        "hit_rate": float((r > 0).mean()),
    }
    if turnover is not None:
        t = pd.to_numeric(turnover, errors="coerce").dropna()
        out["turnover_ann"] = float(t.mean() * TRADING_DAYS)
    return out


def paired_block_bootstrap_pvalue(a: pd.Series, b: pd.Series, n_boot: int = 5000,
                                  block_length: Optional[float] = None,
                                  seed: int = 0) -> tuple[float, float]:
    """Two-sided p-value for a difference in Sharpe between two return series.

    PAIRED: both strategies are resampled on the SAME bootstrap indices, so the
    common market move they both sit on cancels and the test is about the
    difference rather than about the market. Blocks are used because daily
    strategy returns are serially dependent, in particular the straddle book,
    whose 21 overlapping positions guarantee dependence out to 21 days.

    Returns (observed Sharpe difference, p-value).
    """
    from vol_forecasting import stationary_bootstrap_indices

    frame = pd.concat([pd.to_numeric(a, errors="coerce").rename("a"),
                       pd.to_numeric(b, errors="coerce").rename("b")],
                      axis=1).dropna()
    if len(frame) < 30:
        return np.nan, np.nan
    observed = sharpe(frame["a"]) - sharpe(frame["b"])
    n = len(frame)
    if block_length is None:
        block_length = max(5.0, float(n) ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, n_boot, block_length, rng)
    values = frame.to_numpy()
    draws = values[idx]                                   # (n_boot, n, 2)
    mu = draws.mean(axis=1)
    sd = draws.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = mu / sd * np.sqrt(TRADING_DAYS)
    diff = sr[:, 0] - sr[:, 1]
    centred = diff - np.nanmean(diff)
    p = float(np.nanmean(np.abs(centred) >= abs(observed)))
    return float(observed), p


# ---------------------------------------------------------------------------
# 1. Volatility-managed SPY
# ---------------------------------------------------------------------------


def volatility_managed(forecast_vol: pd.Series, asset_returns: pd.Series,
                       target_vol: float = 0.15, cap: float = 2.0,
                       cost_bps: float = 5.0) -> pd.DataFrame:
    """Scale exposure to a constant volatility target using each model's forecast.

    weight_t = target / forecast_t, capped at `cap`, applied to the return of
    t+1. The one-day offset is the whole point: a weight set from a forecast
    made at t must be earned on t+1, and setting it on the same day's return
    would be the classic way to manufacture a Sharpe ratio out of nothing.

    Cost is `cost_bps` per unit of turnover, charged on the day the weight
    changes.
    """
    f = pd.to_numeric(forecast_vol, errors="coerce")
    r = pd.to_numeric(asset_returns, errors="coerce")
    frame = pd.concat([f.rename("f"), r.rename("r")], axis=1).dropna()
    weight = (target_vol / frame["f"].where(frame["f"] > 0)).clip(upper=cap)
    turnover = weight.diff().abs().fillna(weight.abs())
    gross = weight.shift(1) * frame["r"]
    cost = (cost_bps / 10000.0) * turnover.shift(1)
    out = pd.DataFrame({
        "weight": weight,
        "turnover": turnover,
        "gross": gross,
        "net": gross - cost,
    }).dropna(subset=["net"])
    return out


# ---------------------------------------------------------------------------
# 2. Delta-hedged straddles
# ---------------------------------------------------------------------------


@dataclass
class StraddleConfig:
    hold_days: int = 21
    spread_fraction: float = 0.5      # half the quoted spread, so entry is at the touch
    hedge_cost_bps: float = 5.0
    delta_hedge: bool = True


def straddle_trade(chain: pd.DataFrame, underlying: pd.Series, entry_date: pd.Timestamp,
                   exdate: pd.Timestamp, strike: float, direction: int,
                   config: StraddleConfig) -> Optional[pd.Series]:
    """Daily P&L of one delta-hedged straddle, as a fraction of the premium paid.

    `direction` is +1 for a long straddle and -1 for a short one. The option
    legs are marked at the mid every day and closed at the touch; the hedge is
    -(delta_call + delta_put) shares, rebalanced daily at the OptionMetrics
    close, charged `hedge_cost_bps` on the traded share notional.

    Returns a daily series of P&L divided by the entry premium, indexed by date,
    or None when the contract has no usable quote path.
    """
    legs = chain[(chain["exdate"] == exdate) & (chain["strike"] == strike)
                 & (chain["date"] >= entry_date)]
    if legs.empty:
        return None
    dates = np.sort(legs["date"].unique())[: config.hold_days + 1]
    if len(dates) < 2:
        return None
    legs = legs[legs["date"].isin(dates)]
    wide_bid = legs.pivot_table(index="date", columns="cp_flag", values="best_bid")
    wide_ask = legs.pivot_table(index="date", columns="cp_flag", values="best_offer")
    wide_delta = legs.pivot_table(index="date", columns="cp_flag", values="delta")
    if not {"C", "P"}.issubset(wide_bid.columns):
        return None
    wide_bid, wide_ask, wide_delta = (w.dropna(how="any") for w in
                                      (wide_bid, wide_ask, wide_delta))
    common = wide_bid.index.intersection(wide_ask.index).intersection(wide_delta.index)
    if len(common) < 2:
        return None
    wide_bid, wide_ask = wide_bid.loc[common], wide_ask.loc[common]
    wide_delta = wide_delta.loc[common]

    mid = (wide_bid + wide_ask) / 2.0
    spread = wide_ask - wide_bid
    straddle_mid = mid["C"] + mid["P"]
    straddle_spread = spread["C"] + spread["P"]
    # entry at the touch, exit at the touch, both against the position's own side
    entry_price = straddle_mid.iloc[0] + direction * config.spread_fraction * straddle_spread.iloc[0]
    exit_price = straddle_mid.iloc[-1] - direction * config.spread_fraction * straddle_spread.iloc[-1]
    premium = float(straddle_mid.iloc[0])
    if not np.isfinite(premium) or premium <= 0:
        return None

    marks = straddle_mid.copy()
    marks.iloc[0] = entry_price
    marks.iloc[-1] = exit_price
    option_pnl = direction * marks.diff()
    option_pnl.iloc[0] = 0.0        # the entry day has a position but no move yet

    hedge_pnl = pd.Series(0.0, index=common)
    if config.delta_hedge:
        spot = pd.to_numeric(underlying.reindex(common), errors="coerce").ffill()
        net_delta = direction * (wide_delta["C"] + wide_delta["P"])
        shares = -net_delta                                   # delta neutral
        price_pnl = (shares.shift(1) * spot.diff()).fillna(0.0)
        # shares traded: putting the hedge on at entry, rebalancing each day, and
        # taking it off at exit. Charging only the rebalances would make a
        # constant-delta position look free to hedge, which it is not.
        traded = shares.diff().abs()
        traded.iloc[0] = abs(float(shares.iloc[0]))
        traded.iloc[-1] = traded.iloc[-1] + abs(float(shares.iloc[-1]))
        cost = (config.hedge_cost_bps / 10000.0) * traded * spot
        hedge_pnl = price_pnl - cost.fillna(0.0)

    return (option_pnl + hedge_pnl.fillna(0.0)) / premium


def straddle_backtest(chain: pd.DataFrame, entries: pd.DataFrame, underlying: pd.Series,
                      signal: pd.Series, config: Optional[StraddleConfig] = None,
                      long_only: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Open one straddle per signalled date and hold it `hold_days`.

    Positions overlap: a new trade starts every signalled day while up to
    `hold_days` earlier ones are still open, so the daily portfolio return
    divides by `hold_days`, which is the same as committing 1/hold_days of the
    premium budget each day. That keeps the book fully invested without letting
    the overlap silently lever the strategy up.

    Returns (daily portfolio return series, per-trade summary).
    """
    config = config or StraddleConfig()
    picks = entries.set_index("date")
    daily: list[pd.Series] = []
    trades = []
    for date, side in signal.dropna().items():
        side = int(np.sign(side))
        if side == 0 or (long_only and side < 0):
            continue
        if date not in picks.index:
            continue
        row = picks.loc[date]
        path = straddle_trade(chain, underlying, date, row["exdate"], row["strike"],
                              side, config)
        if path is None or path.abs().sum() == 0:
            continue
        daily.append(path)
        trades.append({"entry": date, "exit": path.index[-1], "direction": side,
                       "strike": float(row["strike"]), "exdate": row["exdate"],
                       "days_held": len(path) - 1, "trade_return": float(path.sum())})
    if not daily:
        return pd.Series(dtype=float), pd.DataFrame()
    book = pd.concat(daily, axis=1).sum(axis=1, min_count=1) / config.hold_days
    # Days on which the rule said "flat" are zero-return days, not missing days.
    # Dropping them would divide the same total P&L by a smaller day count and
    # inflate every Sharpe by the square root of the share of days traded, and
    # it would do so by a different amount for each model, which is exactly the
    # comparison this table exists to make.
    full = pd.DatetimeIndex(signal.index).union(book.index)
    return book.reindex(full).fillna(0.0).sort_index(), pd.DataFrame(trades)


def straddle_signal(forecast_vol: pd.Series, implied_vol: pd.Series,
                    split: float = 0.5, margin_quantile: float = 0.5
                    ) -> tuple[pd.Series, dict]:
    """Long, short or flat, from forecast variance against implied variance.

    Two numbers are estimated on the FIRST `split` of the sample and then frozen:

      premium  the average gap between implied and forecast variance. Implied
               variance sits above realized because of the variance risk
               premium, so an unadjusted comparison would say "sell" almost
               every day and the strategy would be a premium harvester wearing
               a forecast as a disguise. Adding the historical premium back to
               the forecast makes the comparison about the forecast's
               DEVIATION from its own typical relationship with implied.
      margin   the `margin_quantile` quantile of the absolute adjusted signal.
               At the default 0.5 the strategy trades on roughly half the days.

    Returns (+1/-1/0 series, the frozen parameters).
    """
    f = pd.to_numeric(forecast_vol, errors="coerce") ** 2
    i = pd.to_numeric(implied_vol, errors="coerce") ** 2
    frame = pd.concat([f.rename("f"), i.rename("i")], axis=1).dropna()
    cut = int(len(frame) * split)
    if cut < 30:
        raise ValueError("not enough observations to set the trading rule")
    head = frame.iloc[:cut]
    premium = float((head["i"] - head["f"]).mean())
    raw = frame["f"] + premium - frame["i"]
    margin = float(raw.iloc[:cut].abs().quantile(margin_quantile))
    side = pd.Series(0, index=frame.index, dtype=int)
    side[raw > margin] = 1
    side[raw < -margin] = -1
    return side, {"premium": premium, "margin": margin, "calibration_end":
                  str(frame.index[cut - 1].date()), "n_calibration": cut}


# ---------------------------------------------------------------------------
# 3. Model-free variance swap
# ---------------------------------------------------------------------------


def newey_west_tstat(series: pd.Series, lag: int) -> float:
    """t-statistic for a zero mean, with a Newey-West long-run variance.

    Needed because the variance-swap payoff is observed daily but measures an
    overlapping 21-day window, so consecutive observations share 20 of their 21
    days and a naive t-statistic is inflated by roughly sqrt(21).
    """
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    if n < 10:
        return np.nan
    dev = (x - x.mean()).to_numpy()
    lrv = float((dev ** 2).mean())
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * float((dev[k:] * dev[:-k]).mean())
    if lrv <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(lrv / n))


def variance_swap_summary(payoff: pd.Series, horizon: int = 21) -> dict:
    """Summarise an overlapping variance-swap payoff without pretending it is daily.

    Two views, because either alone misleads. The overlapping series uses every
    day and needs a Newey-West correction. The non-overlapping series takes
    every `horizon`-th observation, which throws away data but gives a clean
    t-statistic and a Sharpe that can be annualised honestly (by sqrt of the
    number of periods per year, not sqrt(252)).
    """
    x = pd.to_numeric(payoff, errors="coerce").dropna()
    if x.empty:
        return {}
    non_overlap = x.iloc[::horizon]
    periods = TRADING_DAYS / horizon
    sd = non_overlap.std()
    return {
        "n_overlapping": int(len(x)),
        "mean_variance_points": float(x.mean()),
        "mean_vol_points_equivalent": float(np.sign(x.mean()) * np.sqrt(abs(x.mean()))),
        "nw_tstat_overlapping": newey_west_tstat(x, lag=horizon - 1),
        "n_non_overlapping": int(len(non_overlap)),
        "mean_non_overlapping": float(non_overlap.mean()),
        "tstat_non_overlapping": float(non_overlap.mean() / (sd / np.sqrt(len(non_overlap))))
        if len(non_overlap) > 2 and sd > 0 else np.nan,
        "sharpe_annualised": float(non_overlap.mean() / sd * np.sqrt(periods))
        if sd > 0 else np.nan,
        "share_positive": float((x > 0).mean()),
        "worst_observation": float(x.min()),
    }


def variance_swap_pnl(implied_vol: pd.Series, realized_forward_vol: pd.Series
                      ) -> pd.Series:
    """Payoff to a long synthetic variance swap struck at implied variance.

    Long variance pays realized minus implied, in annualized variance units.
    With implied taken as VIX squared this is the Carr and Wu (2009)
    variance-swap return, and it needs no forecast: it measures the variance
    risk premium directly, which is the benchmark any forecast-driven straddle
    strategy has to beat to have shown anything.
    """
    i = pd.to_numeric(implied_vol, errors="coerce") ** 2
    r = pd.to_numeric(realized_forward_vol, errors="coerce") ** 2
    return (r - i).dropna()
