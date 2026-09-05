#!/usr/bin/env python3
"""Daily SPY closes and total returns from a public source.

The option P&L needs two things from the underlying: the price the delta hedge
trades shares at, and a total return for the volatility-managed strategy. Both
used to come from OptionMetrics `secprd`, which meant a licensed vendor's rows
sat in the repository for two columns that are public information. This pulls
them from yfinance instead, which `run_vol_benchmark.py` already depends on.

    close   the unadjusted closing price, which is what a hedge actually
            transacts at, and what the option quotes are struck around
    ret     the total return, from the dividend-adjusted close, because a
            volatility-managed equity strategy that ignores dividends
            understates its own Sharpe by the dividend yield

Both come from one download so they cannot drift apart. yfinance returns
MultiIndex columns when given a ticker list and flat ones otherwise, and the
shape has changed between releases, so the column extraction is explicit rather
than positional.

Usage:
    python fetch_spy_daily.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _column(frame: pd.DataFrame, name: str, ticker: str) -> pd.Series:
    """One named price column, whatever shape yfinance returned it in."""
    if isinstance(frame.columns, pd.MultiIndex):
        return frame[(name, ticker)]
    return frame[name]


def fetch(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                      progress=False)
    if raw.empty:
        raise SystemExit(f"yfinance returned no rows for {ticker} from {start}")
    close = pd.to_numeric(_column(raw, "Close", ticker), errors="coerce")
    adjusted = pd.to_numeric(_column(raw, "Adj Close", ticker), errors="coerce")
    out = pd.DataFrame({"close": close, "ret": adjusted.pct_change()})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    return out.dropna(subset=["close"]).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", type=Path, default=Path("data/SPY_daily.csv"))
    args = parser.parse_args()

    frame = fetch(args.ticker, args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, float_format="%.8g")
    print(f"{len(frame):,} rows, {frame.index.min().date()} -> {frame.index.max().date()}")
    print(f"mean daily total return {frame['ret'].mean():+.6f}, "
          f"annualised {frame['ret'].mean() * 252:+.4f}")
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
