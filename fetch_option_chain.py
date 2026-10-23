#!/usr/bin/env python3
"""Pull the SPY straddle legs the option P&L backtest needs, from OptionMetrics.

The strategy in `option_strategies.py` opens one at-the-money straddle per
forecast date and holds it for 21 trading days, delta-hedged daily. That needs
the daily quote path of two specific contracts per entry date, not the whole
chain, so the pull is done in two passes:

  1. For every trading date and every expiry between `--min-days` and
     `--max-days` calendar days out, take the strike closest to the
     OptionMetrics closing spot (its forward-price column is empty for SPY).
     Then, in Python, keep per date the expiry closest to `--target-days`
     (default 30) among those that expire on or after the exit session
     `--hold-days` sessions later, so a 21-session hold never runs past expiry
     (`option_strategies.choose_straddle_contracts`).
  2. Pull every daily quote for the distinct (exdate, strike) contracts chosen
     in pass 1, over their whole life, both puts and calls.

Underlying closes are NOT pulled here. They come from `fetch_spy_daily.py`,
which uses yfinance, so no licensed vendor rows are needed for two columns of
public information. On the overlap the two agree to 7 cents on the close at
worst and 4e-05 on average, so the substitution costs nothing the P&L can see.

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
from option_strategies import choose_straddle_contracts


def pick_candidates(con, year: int, lo: int, hi: int) -> pd.DataFrame:
    """One at-the-money (date, exdate, strike) per trading date and expiry.

    `DISTINCT ON` with an ORDER BY is the Postgres idiom for argmin, and doing
    it server-side means the whole chain never crosses the wire. The expiry is
    chosen afterwards, in `select_entries`, because it depends on the trading
    calendar.
    """
    # opprcd's own forward_price column is NULL for SPY across this sample, so
    # the at-the-money strike is chosen against the OptionMetrics closing spot
    # from secprd. Spot-ATM is the conventional straddle strike anyway; with a
    # 30-day maturity the forward differs from spot by well under one strike.
    sql = f"""
        with px as (
            select date, close from optionm.secprd{year} where secid = %s
        ),
        expiries as (
            select distinct date, exdate
            from optionm.opprcd{year}
            where secid = %s and exdate > date and (exdate - date) between %s and %s
        )
        select distinct on (o.date, o.exdate) o.date, o.exdate, o.strike_price, x.close
        from optionm.opprcd{year} o
        join expiries e on o.date = e.date and o.exdate = e.exdate
        join px x on x.date = o.date
        where o.secid = %s and o.best_bid > 0
        order by o.date, o.exdate, abs(o.strike_price / 1000.0 - x.close), o.strike_price
    """
    return _read(con, sql, (SPY_SECID, SPY_SECID, lo, hi, SPY_SECID))


def select_entries(candidates: pd.DataFrame, hold_days: int, target_days: int,
                   lo: int, hi: int) -> pd.DataFrame:
    """Per date, the candidate whose expiry outlives the hold and is nearest target.

    The trading calendar is the set of dates OptionMetrics quoted, which is
    every SPY session in the pulled years.
    """
    frame = pd.DataFrame({
        "date": pd.to_datetime(candidates["date"]),
        "exdate": pd.to_datetime(candidates["exdate"]),
        "strike": pd.to_numeric(candidates["strike_price"], errors="coerce") / 1000.0,
        "spot": pd.to_numeric(candidates["close"], errors="coerce"),
    })
    sessions = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    return choose_straddle_contracts(frame, sessions, hold_days=hold_days,
                                     target_days=target_days, min_days=lo, max_days=hi)


def fetch_quote_paths(con, year: int, contracts: pd.DataFrame) -> pd.DataFrame:
    """Daily quotes for the chosen contracts, from one year's price table."""
    pairs = contracts[["exdate", "strike_price"]].drop_duplicates()
    if pairs.empty:
        return pd.DataFrame()
    values = ",".join(f"(date '{pd.Timestamp(e):%Y-%m-%d}', {int(k)})"
                      for e, k in pairs.itertuples(index=False))
    sql = f"""
        select o.date, o.exdate, o.cp_flag, o.strike_price, o.best_bid, o.best_offer,
               o.delta, o.impl_volatility, o.volume, o.open_interest, o.optionid
        from optionm.opprcd{year} o
        join (values {values}) as s(exdate, strike_price)
          on o.exdate = s.exdate and o.strike_price = s.strike_price
        where o.secid = %s
    """
    return _read(con, sql, (SPY_SECID,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--target-days", type=int, default=30)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--max-days", type=int, default=60)
    parser.add_argument("--hold-days", type=int, default=21,
                        help="sessions each straddle is held; the expiry must outlive it")
    parser.add_argument("--out", type=Path, default=Path("data/option_chain_spy.parquet"))
    parser.add_argument("--username", default=None)
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    con = connect(args.username)
    try:
        candidates = pd.concat(
            [pick_candidates(con, y, args.min_days, args.max_days) for y in years],
            ignore_index=True)
        picks = select_entries(candidates, args.hold_days, args.target_days,
                               args.min_days, args.max_days)
        print(f"selected {len(picks):,} entry dates")
        quotes = []
        for year in years:
            # a contract chosen in December expires in January, so each year's
            # price table is asked for every contract chosen in that year or the
            # one before it
            relevant = picks[picks["exdate"].apply(lambda d: d.year in (year, year + 1))]
            got = fetch_quote_paths(con, year, relevant.assign(
                strike_price=(relevant["strike"] * 1000).round().astype(int)))
            print(f"  {year}: {len(got):,} quote rows")
            quotes.append(got)
        chain = pd.concat(quotes, ignore_index=True).drop_duplicates(
            subset=["date", "optionid"])
    finally:
        con.close()

    for col in ["best_bid", "best_offer", "delta", "impl_volatility",
                "volume", "open_interest"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    chain["strike"] = pd.to_numeric(chain["strike_price"], errors="coerce") / 1000.0
    chain["date"] = pd.to_datetime(chain["date"])
    chain["exdate"] = pd.to_datetime(chain["exdate"])


    args.out.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(args.out, index=False)
    picks.to_parquet(args.out.with_name(args.out.stem + "_entries.parquet"), index=False)
    print(f"chain: {len(chain):,} rows, {chain['date'].min().date()} -> {chain['date'].max().date()}")
    print(f"saved -> {args.out} (gitignored, licensed rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
