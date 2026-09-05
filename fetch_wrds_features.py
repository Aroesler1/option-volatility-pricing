#!/usr/bin/env python3
"""Pull the option-market block of the alternative-data panel from WRDS.

Four daily series, all for SPY (OptionMetrics secid 109820), plus the CBOE
index file:

    atm_iv_30       30-day at-the-money implied volatility, the average of the
                    standardised call and put (optionm.stdopdYYYY, days = 30).
                    Its square, `atm_ivar_30`, is the implied VARIANCE the
                    HAR-RV-IV regression of Busch, Christensen and Nielsen
                    (2011) uses as a regressor.
    skew_25d_30     25-delta put IV minus 25-delta call IV at 30 days
                    (optionm.vsurfdYYYY). The standard risk-reversal measure of
                    the option market's price for downside protection.
    term_slope      30-day ATM IV minus 91-day ATM IV (stdopd). Negative in calm
                    markets (upward-sloping term structure) and positive when
                    the front end spikes, so it carries the market's view of
                    whether current stress is transient.
    spy_put_call    total SPY put contract volume over total SPY call contract
                    volume (optionm.opprcdYYYY). A substitute for the CBOE
                    market-wide put/call ratio, whose file is not reachable
                    programmatically; see `fetch_put_call`.
    vix, vxo        cboe.cboe. VXO is retained because the FEARS and attention
                    literature was written against it, but CBOE stopped
                    publishing it, so it is missing over the later sample and is
                    reported rather than silently filled.

Licence: OptionMetrics and the CBOE index file are licensed through WRDS. Only
the derived daily series is written, never option-level rows, and the output
CSV is the only thing committed.

Credentials come from `~/.pgpass`; set WRDS_USERNAME to your WRDS login. Nothing
in this file contains a secret.

WRDS is behind Duo, so `connect` refuses to do anything unless WRDS_DUO_READY=1
is set for that run, allows exactly one attempt per invocation, and never
retries a refusal. See its docstring for why.

Usage:
    WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_wrds_features.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
SPY_SECID = 109820


class WRDSGuardError(RuntimeError):
    """A refusal to open a WRDS connection. Never caught and retried anywhere."""


DUO_ENV = "WRDS_DUO_READY"
_connection_attempts = 0


def connect(username: str | None = None):
    """Open the single WRDS connection this process is allowed, or fail closed.

    WRDS sits behind Duo two-factor. A script that reconnects, or retries after
    a refusal, turns one declined push into a burst of them, and a burst of
    declined pushes is what gets an account locked out. So the rules here are
    deliberately unforgiving:

    1. Nothing connects unless the operator has set WRDS_DUO_READY=1 for that
       run. That variable means a human is at the device and expecting a push.
       It is not stored anywhere and is not defaulted.
    2. One attempt per process. The counter is incremented BEFORE the attempt,
       so a failure still consumes it. Anything that wants a second connection
       has to be a second invocation.
    3. An authentication or connection failure raises `WRDSGuardError` and is
       never retried. The caller is told what happened and stops.

    The cost of the guard is having to re-run a script. The cost of not having
    it is a locked WRDS account and a support ticket.
    """
    global _connection_attempts

    if os.environ.get(DUO_ENV) != "1":
        raise WRDSGuardError(
            f"refusing to contact WRDS: {DUO_ENV} is not set to 1. WRDS requires a "
            f"Duo push, so set {DUO_ENV}=1 only when you are at the device and "
            f"ready to approve one, for that single run.")

    user = username or os.environ.get("WRDS_USERNAME") or os.environ.get("PGUSER")
    if not user:
        raise WRDSGuardError(
            "refusing to contact WRDS: set WRDS_USERNAME (the password is read "
            "from ~/.pgpass, never from this repository)")

    if _connection_attempts:
        raise WRDSGuardError(
            "refusing to contact WRDS: this process has already attempted a "
            "connection. One attempt per invocation, whether or not it "
            "succeeded. Run the script again rather than reconnecting.")
    _connection_attempts += 1

    try:
        return psycopg2.connect(host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
                                user=user, sslmode="require", connect_timeout=60)
    except psycopg2.Error as exc:
        raise WRDSGuardError(
            f"WRDS refused the connection and it will NOT be retried: {exc}. "
            f"Check that the Duo push was approved and that ~/.pgpass holds a "
            f"current password, then run the script again.") from exc


def _read(con, sql: str, params: tuple) -> pd.DataFrame:
    with con.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_atm_iv(con, years: range) -> pd.DataFrame:
    """Standardised 30-day and 91-day ATM implied volatility, call/put averaged.

    The standardised file interpolates to a constant maturity, which is what
    makes a time series of "the 30-day implied vol" meaningful; the raw chain
    would have a maturity that shortens by a day every day.
    """
    frames = []
    for year in years:
        sql = (f"select date, days, cp_flag, impl_volatility "
               f"from optionm.stdopd{year} where secid = %s and days in (30, 91)")
        frames.append(_read(con, sql, (SPY_SECID,)))
    raw = pd.concat(frames, ignore_index=True)
    raw["impl_volatility"] = pd.to_numeric(raw["impl_volatility"], errors="coerce")
    wide = raw.pivot_table(index="date", columns="days",
                           values="impl_volatility", aggfunc="mean")
    out = pd.DataFrame(index=pd.to_datetime(wide.index))
    out["atm_iv_30"] = wide[30].to_numpy(dtype=float)
    out["atm_iv_91"] = wide[91].to_numpy(dtype=float)
    out["atm_ivar_30"] = out["atm_iv_30"] ** 2
    out["term_slope_30_91"] = out["atm_iv_30"] - out["atm_iv_91"]
    return out


def fetch_skew(con, years: range) -> pd.DataFrame:
    """25-delta put IV minus 25-delta call IV at 30 days, from the surface file."""
    frames = []
    for year in years:
        sql = (f"select date, delta, cp_flag, impl_volatility "
               f"from optionm.vsurfd{year} "
               f"where secid = %s and days = 30 and delta in (-25, 25)")
        frames.append(_read(con, sql, (SPY_SECID,)))
    raw = pd.concat(frames, ignore_index=True)
    raw["impl_volatility"] = pd.to_numeric(raw["impl_volatility"], errors="coerce")
    raw["delta"] = pd.to_numeric(raw["delta"], errors="coerce")
    wide = raw.pivot_table(index="date", columns="delta",
                           values="impl_volatility", aggfunc="mean")
    out = pd.DataFrame(index=pd.to_datetime(wide.index))
    out["skew_25d_30"] = wide[-25.0].to_numpy(dtype=float) - wide[25.0].to_numpy(dtype=float)
    return out


def fetch_put_call(con, years: range) -> pd.DataFrame:
    """SPY put/call option VOLUME ratio, aggregated server-side.

    The literature uses the CBOE market-wide put/call ratio. That file is
    behind a 403 for programmatic requests, so this is the SPY-only substitute:
    total put contract volume over total call contract volume across the whole
    listed chain, from OptionMetrics. It is a narrower measure (one underlying,
    not the market) and is labelled `spy_put_call` rather than `put_call` so the
    substitution is visible wherever the feature is used.
    """
    frames = []
    for year in years:
        sql = (f"select date, cp_flag, sum(volume) as vol "
               f"from optionm.opprcd{year} where secid = %s group by date, cp_flag")
        frames.append(_read(con, sql, (SPY_SECID,)))
    raw = pd.concat(frames, ignore_index=True)
    raw["vol"] = pd.to_numeric(raw["vol"], errors="coerce")
    wide = raw.pivot_table(index="date", columns="cp_flag", values="vol", aggfunc="sum")
    out = pd.DataFrame(index=pd.to_datetime(wide.index))
    out["spy_put_call"] = (wide["P"] / wide["C"].replace(0, np.nan)).to_numpy(dtype=float)
    return out


def fetch_cboe(con, start: str) -> pd.DataFrame:
    raw = _read(con, "select date, vix, vxo from cboe.cboe where date >= %s", (start,))
    out = raw.set_index(pd.to_datetime(raw["date"])).drop(columns=["date"])
    out.index.name = "date"
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=Path("data/features_option_market.csv"))
    parser.add_argument("--username", default=None)
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    con = connect(args.username)
    try:
        iv = fetch_atm_iv(con, years)
        skew = fetch_skew(con, years)
        put_call = fetch_put_call(con, years)
        cboe = fetch_cboe(con, f"{args.start_year}-01-01")
    finally:
        con.close()

    panel = (iv.join(skew, how="outer").join(put_call, how="outer")
             .join(cboe, how="outer").sort_index())
    panel.index.name = "date"
    panel = panel.loc[panel[["atm_iv_30", "vix"]].notna().any(axis=1)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, float_format="%.10g")

    print(f"{len(panel):,} rows, {panel.index.min().date()} -> {panel.index.max().date()}")
    print(panel.notna().mean().round(4).to_string())
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
