#!/usr/bin/env python3
"""Rebuild the committed daily realized-variance series from raw Databento bars.

The derived series `data/SPY_intraday_rv.csv` was previously produced by an
uncommitted script, which made the one input the rest of the repo depends on
irreproducible. This rebuilds it from the raw 1-minute bars and adds the two
columns the Patton-Sheppard semivariance HAR needs.

Estimators, all from 5-minute log returns inside the regular session
(09:30-16:00 New York, so the overnight gap is excluded rather than treated as
one enormous 17-hour return):

    RV      = sum_i r_i^2                    realized variance
    RQ      = (n/3) * sum_i r_i^4            realized quarticity (BPQ 2016)
    RS+     = sum_i r_i^2 * 1{r_i > 0}       positive semivariance (BNS 2010)
    RS-     = sum_i r_i^2 * 1{r_i < 0}       negative semivariance

RS+ + RS- = RV by construction, which the tests check.

The raw `.dbn.zst` extract is licensed and gitignored; only the daily aggregate
is committed. Point DATABENTO_RAW_DIR or --raw at the extract to re-run.

Usage:
    python build_intraday_rv.py --raw data/databento/SPY_1m_2018_2026.dbn.zst
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"
FIRST_BUCKET = "09:35"


def five_minute_returns(bars: pd.DataFrame) -> pd.Series:
    """Log returns of 5-minute closes inside the regular session, in New York time.

    `bars` is 1-minute OHLCV indexed by a UTC timestamp. Resampling to 5 minutes
    and taking the last close within each bucket is the standard construction;
    the first return of the day is dropped with `between_time` because it would
    otherwise span the overnight close-to-open gap.
    """
    px = bars["close"].copy()
    px.index = px.index.tz_convert("America/New_York")
    px = px.between_time(SESSION_OPEN, SESSION_CLOSE)
    # closed="left" makes each bucket [t, t+5min), so its last 1-minute bar is
    # the one ending at t+5min; label="right" then stamps that price at t+5min.
    # The grid is therefore 09:35, 09:40, ..., 16:00: 78 closes and 77
    # within-session returns on a full day. This reproduces the committed
    # series `data/SPY_intraday_rv.csv` to floating-point precision, which is
    # the point - the baseline results in the README were computed on it.
    closes = px.resample("5min", label="right", closed="left").last().dropna()
    closes = closes.between_time(FIRST_BUCKET, SESSION_CLOSE)
    ret = np.log(closes / closes.shift(1))
    # drop the return that bridges two sessions
    same_day = closes.index.normalize() == closes.index.normalize().to_series().shift(1).to_numpy()
    return ret[same_day].dropna()


def daily_estimators(ret5m: pd.Series, min_buckets: int = 30) -> pd.DataFrame:
    """Aggregate 5-minute returns into the daily realized measures."""
    day = ret5m.index.normalize().tz_localize(None)
    grouped = ret5m.groupby(day)
    r2 = ret5m.pow(2)
    out = pd.DataFrame({
        "rv5m_var": r2.groupby(day).sum(),
        "n_buckets": grouped.size(),
        "sum_r4": ret5m.pow(4).groupby(day).sum(),
        "rs_pos_var": r2.where(ret5m > 0, 0.0).groupby(day).sum(),
        "rs_neg_var": r2.where(ret5m < 0, 0.0).groupby(day).sum(),
    })
    out.index.name = "date"
    # half days (early closes) carry far fewer buckets; keeping them would make
    # RV mechanically small on those dates rather than genuinely low
    out = out[out["n_buckets"] >= min_buckets]
    out["rq5m"] = (out["n_buckets"] / 3.0) * out["sum_r4"]
    out["rv5m"] = np.sqrt(out["rv5m_var"] * TRADING_DAYS)
    return out[["rv5m_var", "rq5m", "n_buckets", "rv5m", "rs_pos_var", "rs_neg_var"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=Path("data/databento/SPY_1m_2018_2026.dbn.zst"))
    parser.add_argument("--out", type=Path, default=Path("data/SPY_intraday_rv.csv"))
    parser.add_argument("--min-buckets", type=int, default=30)
    args = parser.parse_args()

    import databento as db

    store = db.DBNStore.from_file(args.raw)
    bars = store.to_df()
    bars = bars[bars["symbol"] == "SPY"] if "symbol" in bars else bars
    ret = five_minute_returns(bars)
    daily = daily_estimators(ret, args.min_buckets)
    daily.to_csv(args.out, float_format="%.12g")
    print(f"{len(daily):,} days, {daily.index.min().date()} -> {daily.index.max().date()}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
