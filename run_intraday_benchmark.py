#!/usr/bin/env python3
"""Does the realized-variance ESTIMATOR change the volatility-forecasting answer?

The rest of this repo estimates "realized volatility" as a rolling standard
deviation of DAILY returns. Corsi's HAR-RV, and the HAR literature generally,
is defined on INTRADAY realized variance: the sum of squared intraday returns
within a day. These are different estimators of different things, and the daily
proxy is far noisier because it uses one observation per day rather than 78.

This script holds everything else fixed -- same period, same models, same target
horizon, same out-of-sample protocol -- and varies only the estimator, so the
difference attributable to that choice is measurable rather than assumed.

It also gives HARQ its first fair test. HARQ corrects for time-varying
measurement error in realized variance using realized quarticity, which requires
intraday data; the daily-return proxy for RQ used in run_vol_benchmark.py is
coarse enough to plausibly destroy the signal HARQ exploits.

Intraday data: SPY 1-minute bars from Databento XNAS.ITCH, 2018-05 to 2026-08,
aggregated to 5-minute returns (the standard choice trading off microstructure
noise against sampling error).

Usage:
    python run_intraday_benchmark.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from run_vol_benchmark import (
    har_forecasts,
    har_pdv_forecasts,
    harq_forecasts,
    log_har_forecasts,
    pdv_forecasts,
    qlike_series,
    wls_har_forecasts,
)
from vol_forecasting import (
    diebold_mariano,
    mean_combination,
    model_confidence_set,
    realized_vol,
)

TRADING_DAYS = 252


def forward_vol_from_variance(daily_var: pd.Series, horizon: int) -> pd.Series:
    """Annualised realized vol over the NEXT `horizon` days, from daily variances.

    Uses strictly future observations: the value at t is built from t+1..t+h.
    """
    fwd = daily_var.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))
    return np.sqrt(fwd * TRADING_DAYS / horizon)


def build_frames(intraday_path: Path, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    intra = pd.read_csv(intraday_path, parse_dates=["date"]).set_index("date").sort_index()

    px = yf.download("SPY", start="2017-01-01", progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    ret = np.log(px / px.shift(1))

    # --- estimator A: true intraday realized variance -----------------------
    rv_i = intra["rv5m"]                       # annualised vol from 5-min returns
    var_i = intra["rv5m_var"]                  # daily variance (not annualised)
    a = pd.DataFrame({
        "ret": ret.reindex(rv_i.index),
        "rv": rv_i,
        "rq": intra["rq5m"] * (TRADING_DAYS ** 2),
        "rv_w": rv_i.rolling(5).mean(),
        "rv_m": rv_i.rolling(22).mean(),
        "abs_ret_5": ret.reindex(rv_i.index).abs().rolling(5).mean() * np.sqrt(TRADING_DAYS),
        "neg_ret_21": ret.reindex(rv_i.index).clip(upper=0).rolling(21).std() * np.sqrt(TRADING_DAYS),
        "target": forward_vol_from_variance(var_i, horizon),
    }).dropna()

    # --- estimator B: daily-return proxy, same dates -------------------------
    rv_d = realized_vol(ret, 21)
    daily_var = (ret ** 2)
    b = pd.DataFrame({
        "ret": ret,
        "rv": rv_d,
        "rq": (21.0 / 3.0) * ret.pow(4).rolling(21).sum() * (TRADING_DAYS ** 2),
        "rv_w": rv_d.rolling(5).mean(),
        "rv_m": rv_d.rolling(22).mean(),
        "abs_ret_5": ret.abs().rolling(5).mean() * np.sqrt(TRADING_DAYS),
        "neg_ret_21": ret.clip(upper=0).rolling(21).std() * np.sqrt(TRADING_DAYS),
        # SAME target as A, so only the regressors' estimator differs
        "target": forward_vol_from_variance(var_i, horizon),
    }).dropna()

    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


# The four HAR-family forecasts averaged by the combination. Equal weights, no
# estimation: the combination is meant to be the cheapest possible remedy.
COMBINATION_MEMBERS = ("har_rv", "harq", "log_har", "wls_har_rv")


def evaluate(frame: pd.DataFrame, label: str, refit: int, test_frac: float,
             horizon: int, alpha: float = 0.10, n_boot: int = 2000,
             seed: int = 0) -> pd.DataFrame:
    test_start = int(len(frame) * (1.0 - test_frac))
    test = frame.iloc[test_start:]
    forecasts = {
        "persistence": frame["rv"],
        "har_rv": har_forecasts(frame, test_start, refit),
        "harq": harq_forecasts(frame, test_start, refit),
        "pdv": pdv_forecasts(frame, test_start, refit),
        "har_pdv": har_pdv_forecasts(frame, test_start, refit),
        # Clements & Preve (2021) remedies
        "log_har": log_har_forecasts(frame, test_start, refit),
        "wls_har_rv": wls_har_forecasts(frame, test_start, refit, scheme="rv"),
        "wls_har_rq": wls_har_forecasts(frame, test_start, refit, scheme="rq"),
    }
    forecasts["combination"] = mean_combination(
        [forecasts[m].loc[test.index] for m in COMBINATION_MEMBERS])
    losses = {k: qlike_series(v.loc[test.index], test["target"]) for k, v in forecasts.items()}
    rows = []
    for name, loss in losses.items():
        pred = forecasts[name].loc[test.index]
        # QLIKE punishes under-forecasting without bound, so a handful of
        # near-zero forecasts can dominate the mean entirely. Report the median
        # alongside it, plus how many forecasts collapsed to the floor: the gap
        # between mean and median IS the tail story, not a nuisance to hide.
        rows.append({
            "estimator": label, "model": name,
            "qlike_mean": float(loss.mean()),
            "qlike_median": float(loss.median()),
            "mse": float(((pred - test["target"]) ** 2).mean()),
            "collapsed": int((pred <= 1.01e-4).sum()),
            "n_obs": int(loss.notna().sum()),
        })
    table = pd.DataFrame(rows)
    dm_lag = max(horizon - 1, 1)
    # DM on the loss differentials is itself mean-based; with two exploding
    # observations it is not interpretable, so it is reported only where no
    # forecast collapsed.
    table["dm_vs_har"] = [
        diebold_mariano(losses[m], losses["har_rv"], lag=dm_lag)[0] if m != "har_rv" else np.nan
        for m in table["model"]
    ]
    table["p_vs_har"] = [
        diebold_mariano(losses[m], losses["har_rv"], lag=dm_lag)[1] if m != "har_rv" else np.nan
        for m in table["model"]
    ]

    # Pairwise DM answers "is model X better than the benchmark I nominated".
    # With nine models that is nine tests at nominal size and a benchmark chosen
    # after seeing the data. The MCS asks the question that was actually meant:
    # which models cannot be distinguished from the best one.
    mcs = model_confidence_set(pd.DataFrame(losses).dropna(), alpha=alpha,
                               n_boot=n_boot, seed=seed)
    table = table.merge(
        mcs[["mcs_pvalue", "in_mcs"]].rename_axis("model").reset_index(),
        on="model", how="left")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intraday", type=Path, default=Path("data/SPY_intraday_rv.csv"))
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--refit", type=int, default=21)
    parser.add_argument("--test-frac", type=float, default=0.4)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="1 - alpha is the MCS confidence level (0.10 -> 90%% MCS)")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="stationary-bootstrap replications for the MCS")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    intraday, daily = build_frames(args.intraday, args.horizon)
    print(f"aligned sample: {len(intraday):,} days, "
          f"{intraday.index.min().date()} -> {intraday.index.max().date()}")
    print(f"target: forward {args.horizon}d realized vol from intraday variance (identical in both)\n")

    a = evaluate(intraday, "intraday 5-min RV", args.refit, args.test_frac,
                 args.horizon, args.alpha, args.n_boot, args.seed)
    b = evaluate(daily, "daily-return proxy", args.refit, args.test_frac,
                 args.horizon, args.alpha, args.n_boot, args.seed)
    table = pd.concat([a, b], ignore_index=True)

    for label in table["estimator"].unique():
        sub = table[table["estimator"] == label].sort_values("qlike_median")
        print(f"{label}   ({(1 - args.alpha):.0%} MCS, {args.n_boot} stationary-bootstrap draws)")
        print(sub[["model", "qlike_mean", "qlike_median", "collapsed",
                   "mcs_pvalue", "in_mcs", "dm_vs_har", "p_vs_har"]]
              .to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        members = sub.loc[sub["in_mcs"], "model"].tolist()
        print(f"  MCS_{(1 - args.alpha):.0%} = {{{', '.join(members)}}}")
        print()

    best = table.loc[table.groupby("estimator")["qlike_median"].idxmin()]
    print("best median QLIKE per estimator:")
    print(best[["estimator", "model", "qlike_median"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:0.4f}"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "estimator_comparison.csv", index=False)
    print(f"\nsaved -> {args.out_dir / 'estimator_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
