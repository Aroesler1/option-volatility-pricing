#!/usr/bin/env python3
"""Pull the SPY straddle legs the option P&L backtest needs, from OptionMetrics.

The strategy in `option_strategies.py` opens one at-the-money straddle per
forecast date and holds it for 21 trading days, delta-hedged daily. That needs
the daily quote path of two specific contracts per entry date, not the whole
chain, so the pull is done in two passes:

  1. For every trading date, choose the expiry whose calendar days to
     expiration is closest to `--target-days` (default 30, which is the
     maturity the 21-trading-day forecast is compared against), and then the
     strike closest to OptionMetrics' own forward price for that expiry. This
     pass returns one row per date.
  2. Pull every daily quote for the distinct (exdate, strike) contracts chosen
     in pass 1, over their whole life, both puts and calls.

Underlying closes come from optionm.secprd so the hedge and the option quotes
are from the same vendor and the same close, rather than mixing a second price
source into the P&L.

LICENCE. This writes option-level rows, which are licensed and must not be
committed. The output goes to a gitignored parquet; only the daily P&L series
derived from it is published.

The WRDS connection goes through `fetch_wrds_features.connect`, so the Duo
guard applies here too: WRDS_DUO_READY=1 must be set for the run, one attempt is
allowed, and a refusal is never retried.

Usage:
    WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_option_chain.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_wrds_features import SPY_SECID, _read, connect


def pick_contracts(con, year: int, target_days: int, lo: int, hi: int) -> pd.DataFrame:
    """One (date, exdate, strike) per trading date: nearest expiry, nearest strike.

    `DISTINCT ON` with an ORDER BY is the Postgres idiom for argmin, and doing
    it server-side means the whole chain never crosses the wire.
    """
    # opprcd's own forward_price column is NULL for SPY across this sample, so
    # the at-the-money strike is chosen against the OptionMetrics closing spot
    # from secprd. Spot-ATM is the conventional straddle strike anyway; with a
    # 30-day maturity the forward differs from spot by well under one strike.
    sql = f"""
        with px as (
            select date, close from optionm.secprd{year} where secid = %s
        ),
        picked_exp as (
            select distinct on (date) date, exdate
            from optionm.opprcd{year}
            where secid = %s and exdate > date and (exdate - date) between %s and %s
            order by date, abs((exdate - date) - %s), exdate
        )
        select distinct on (o.date) o.date, o.exdate, o.strike_price, x.close
        from optionm.opprcd{year} o
        join picked_exp p on o.date = p.date and o.exdate = p.exdate
        join px x on x.date = o.date
        where o.secid = %s and o.best_bid > 0
        order by o.date, abs(o.strike_price / 1000.0 - x.close), o.strike_price
    """
    return _read(con, sql, (SPY_SECID, SPY_SECID, lo, hi, target_days, SPY_SECID))


def fetch_quote_paths(con, year: int, contracts: pd.DataFrame) -> pd.DataFrame:
    """Daily quotes for the chosen contracts, from one year's price table."""
    pairs = contracts[["exdate", "strike_price"]].drop_duplicates()
    if pairs.empty:
        return pd.DataFrame()
    values = ",".join(f"(date '{e}', {int(k)})" for e, k in pairs.itertuples(index=False))
    sql = f"""
        select o.date, o.exdate, o.cp_flag, o.strike_price, o.best_bid, o.best_offer,
               o.delta, o.impl_volatility, o.volume, o.open_interest, o.optionid
        from optionm.opprcd{year} o
        join (values {values}) as s(exdate, strike_price)
          on o.exdate = s.exdate and o.strike_price = s.strike_price
        where o.secid = %s
    """
    return _read(con, sql, (SPY_SECID,))


def fetch_underlying(con, years: range) -> pd.DataFrame:
    """SPY closes and total returns from OptionMetrics.

    `close` is the price the delta hedge trades at, so it has to come from the
    same vendor and the same close as the option quotes. `ret` is OptionMetrics'
    own total return, which includes dividends; the volatility-managed strategy
    needs a total return or its Sharpe is understated by the dividend yield.
    """
    frames = [_read(con, f"select date, close, return from optionm.secprd{y} "
                         f"where secid = %s", (SPY_SECID,)) for y in years]
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "close": pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
        "ret": pd.to_numeric(raw["return"], errors="coerce").to_numpy(),
    }, index=pd.to_datetime(raw["date"]))
    out.index.name = "date"
    return out.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--target-days", type=int, default=30)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--max-days", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("data/option_chain_spy.parquet"))
    parser.add_argument("--underlying-out", type=Path,
                        default=Path("data/SPY_optionmetrics_close.csv"))
    parser.add_argument("--username", default=None)
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    con = connect(args.username)
    try:
        picks = pd.concat(
            [pick_contracts(con, y, args.target_days, args.min_days, args.max_days)
             for y in years], ignore_index=True)
        print(f"selected {len(picks):,} entry dates")
        quotes = []
        for year in years:
            # a contract chosen in December expires in January, so each year's
            # price table is asked for every contract chosen in that year or the
            # one before it
            relevant = picks[picks["exdate"].apply(lambda d: d.year in (year, year + 1))]
            got = fetch_quote_paths(con, year, relevant)
            print(f"  {year}: {len(got):,} quote rows")
            quotes.append(got)
        chain = pd.concat(quotes, ignore_index=True).drop_duplicates(
            subset=["date", "optionid"])
        under = fetch_underlying(con, years)
    finally:
        con.close()

    for col in ["best_bid", "best_offer", "delta", "impl_volatility",
                "volume", "open_interest"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    chain["strike"] = pd.to_numeric(chain["strike_price"], errors="coerce") / 1000.0
    chain["date"] = pd.to_datetime(chain["date"])
    chain["exdate"] = pd.to_datetime(chain["exdate"])

    picks["date"] = pd.to_datetime(picks["date"])
    picks["exdate"] = pd.to_datetime(picks["exdate"])
    picks["strike"] = pd.to_numeric(picks["strike_price"], errors="coerce") / 1000.0
    picks["spot"] = pd.to_numeric(picks["close"], errors="coerce")
    picks = picks[["date", "exdate", "strike", "spot"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(args.out, index=False)
    picks.to_parquet(args.out.with_name(args.out.stem + "_entries.parquet"), index=False)
    under.to_csv(args.underlying_out, float_format="%.6f")
    print(f"chain: {len(chain):,} rows, {chain['date'].min().date()} -> {chain['date'].max().date()}")
    print(f"saved -> {args.out} (gitignored, licensed rows)")
    print(f"saved -> {args.underlying_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
