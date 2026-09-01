#!/usr/bin/env python3
"""Does implied volatility beat the time series at forecasting realized vol?

The rest of this repo forecasts realized volatility from its own history. The
option market produces a forward-looking estimate of the same quantity every
day, and comparing the two is the classic question in the volatility
literature: does the option market know something the time series does not?

Setup:
  target      realized volatility over the next 21 trading days, built from
              5-minute intraday returns (strictly future observations)
  ATM IV      30-day at-the-money implied volatility from the OptionMetrics
              standardised surface (|delta| = 50, days = 30), lagged one day so
              only information available before the forecast date is used
  HAR-RV      the time-series benchmark from run_intraday_benchmark.py
  combined    OLS on both, refit on an expanding window

Two things this is careful about. Implied volatility is a RISK-NEUTRAL
expectation and sits above realized volatility on average because of the
variance risk premium, so using it raw as a point forecast guarantees a
systematic bias; the combined model is allowed to scale it. And the IV series
is lagged, because same-day IV and same-day realized vol share the trading
session.

Usage:
    python run_iv_benchmark.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from run_intraday_benchmark import forward_vol_from_variance
from run_vol_benchmark import har_forecasts, qlike_series
from vol_forecasting import diebold_mariano

TRADING_DAYS = 252


def load_atm_iv(path: Path, days: int = 30) -> pd.Series:
    """Daily at-the-money implied volatility, averaged across put and call."""
    surf = pd.read_parquet(path)
    surf["date"] = pd.to_datetime(surf["date"])
    atm = surf[(surf["days"] == days) & (surf["delta"].abs() == 50)]
    iv = atm.groupby("date")["impl_volatility"].mean().astype(float)
    return iv.sort_index()


def combined_forecasts(frame: pd.DataFrame, test_start: int, refit: int) -> pd.Series:
    """OLS on HAR components plus lagged ATM IV, expanding window."""
    cols = ["rv", "rv_w", "rv_m", "iv_lag"]
    preds = pd.Series(np.nan, index=frame.index)
    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:block_start]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < 200 or block.empty:
            continue
        A = np.column_stack([np.ones(len(train)), train[cols].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, train["target"].to_numpy(), rcond=None)
        B = np.column_stack([np.ones(len(block)), block[cols].to_numpy()])
        preds.loc[block.index] = np.clip(B @ beta, 1e-4, None)
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intraday", type=Path, default=Path("data/SPY_intraday_rv.csv"))
    parser.add_argument("--surface", type=Path, default=Path("data/SPY_ivsurface.parquet"))
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--refit", type=int, default=21)
    parser.add_argument("--test-frac", type=float, default=0.4)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    intra = pd.read_csv(args.intraday, parse_dates=["date"]).set_index("date").sort_index()
    iv = load_atm_iv(args.surface)

    rv = intra["rv5m"]
    frame = pd.DataFrame({
        "rv": rv,
        "rv_w": rv.rolling(5).mean(),
        "rv_m": rv.rolling(22).mean(),
        # lagged: same-day IV and same-day realized vol share a session
        "iv_lag": iv.reindex(rv.index).shift(1),
        "target": forward_vol_from_variance(intra["rv5m_var"], args.horizon),
    }).dropna()

    test_start = int(len(frame) * (1.0 - args.test_frac))
    test = frame.iloc[test_start:]
    print(f"sample: {len(frame):,} days, {frame.index.min().date()} -> {frame.index.max().date()}")
    print(f"OOS: {len(test):,} days from {test.index[0].date()}\n")

    bias = float((frame["iv_lag"] - frame["target"]).mean())
    print(f"mean(IV - realized) = {bias:+.4f}  "
          f"({'IV sits above realized, the variance risk premium' if bias > 0 else 'IV below realized'})\n")

    forecasts = {
        "persistence": frame["rv"],
        "har_rv": har_forecasts(frame, test_start, args.refit),
        "atm_iv_raw": frame["iv_lag"],
        "har_plus_iv": combined_forecasts(frame, test_start, args.refit),
    }
    losses = {k: qlike_series(v.loc[test.index], test["target"]) for k, v in forecasts.items()}

    rows = []
    dm_lag = max(args.horizon - 1, 1)
    for name, loss in losses.items():
        pred = forecasts[name].loc[test.index]
        stat, p = ((np.nan, np.nan) if name == "har_rv"
                   else diebold_mariano(loss, losses["har_rv"], lag=dm_lag))
        rows.append({
            "model": name,
            "qlike_mean": float(loss.mean()),
            "qlike_median": float(loss.median()),
            "mse": float(((pred - test["target"]) ** 2).mean()),
            "dm_vs_har": stat, "p_vs_har": p,
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "iv_vs_timeseries.csv", index=False)
    print(f"\nsaved -> {args.out_dir / 'iv_vs_timeseries.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
