#!/usr/bin/env python3
"""Volatility forecasting benchmark with the corrected methodology.

Downloads a long daily history (yfinance), builds the forward realized-vol
target from strictly future returns, and compares three forecasters
out-of-sample under QLIKE (the standard robust loss for variance
forecasts), with Diebold-Mariano tests:

1. persistence (random walk): forecast = current trailing RV
2. HAR-RV (Corsi 2009), refit on an expanding window every `refit` days
3. ridge regression on HAR features plus return-based extras, alpha chosen
   on a validation tail of the training window (train-only scaling; no
   leakage)

This is the benchmark any ML volatility model in this repo must beat, per
the methodology note in the README.

Usage:
    python run_vol_benchmark.py --ticker SPY --start 2005-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vol_forecasting import (
    HARPDV,
    HARQ,
    HARRV,
    PDVModel,
    diebold_mariano,
    forward_realized_vol,
    realized_vol,
)

TRADING_DAYS = 252


def qlike_series(forecast_vol: pd.Series, realized: pd.Series) -> pd.Series:
    f = (pd.to_numeric(forecast_vol, errors="coerce") ** 2).replace(0, np.nan)
    r = pd.to_numeric(realized, errors="coerce") ** 2
    ratio = r / f
    out = ratio - np.log(ratio) - 1.0
    return out.where(np.isfinite(out))


def build_frame(ticker: str, start: str, horizon: int) -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    ret = np.log(px / px.shift(1))
    rv = realized_vol(ret, 21)
    # Realized quarticity proxy for HARQ. With daily returns as the base
    # observation this is (n/3) * sum(r^4) over the window; it is a coarser
    # estimate than an intraday RQ but preserves the time variation in
    # measurement error that HARQ exploits.
    rq = (21.0 / 3.0) * ret.pow(4).rolling(21).sum() * (TRADING_DAYS ** 2)
    frame = pd.DataFrame(
        {
            "ret": ret,
            "rv": rv,
            "rq": rq,
            "rv_w": rv.rolling(5).mean(),
            "rv_m": rv.rolling(22).mean(),
            "abs_ret_5": ret.abs().rolling(5).mean() * np.sqrt(TRADING_DAYS),
            "neg_ret_21": ret.clip(upper=0).rolling(21).std() * np.sqrt(TRADING_DAYS),
            "target": forward_realized_vol(ret, horizon),
        }
    ).dropna()
    return frame


def pdv_forecasts(frame: pd.DataFrame, test_start: int, refit: int) -> pd.Series:
    """Guyon-Lekeufack path-dependent volatility, refit on an expanding window.

    Kernel decay is grid-searched on the training window only; the loadings are
    then refit by OLS. Prediction needs the trailing return path, so the model
    is evaluated on history through the block end and sliced to the block.
    """
    preds = pd.Series(np.nan, index=frame.index)
    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:block_start]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < 400 or block.empty:
            continue
        model = PDVModel(max_lag=252).fit_kernels(train["ret"], train["target"])
        history = frame["ret"].iloc[: block_start + len(block)]
        preds.loc[block.index] = model.predict(history).loc[block.index]
    return preds


def har_pdv_forecasts(frame: pd.DataFrame, test_start: int, refit: int) -> pd.Series:
    """HAR augmented with the path-dependent state variables, expanding window.

    Isolates whether path-dependence adds anything HAR does not already capture.
    """
    preds = pd.Series(np.nan, index=frame.index)
    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:block_start]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < 400 or block.empty:
            continue
        model = HARPDV(max_lag=252).fit(train["rv"], train["ret"], train["target"])
        hist_rv = frame["rv"].iloc[: block_start + len(block)]
        hist_ret = frame["ret"].iloc[: block_start + len(block)]
        preds.loc[block.index] = model.predict(hist_rv, hist_ret).loc[block.index]
    return preds


def harq_forecasts(frame: pd.DataFrame, test_start: int, refit: int) -> pd.Series:
    """HARQ (Bollerslev-Patton-Quaedvlieg), refit on an expanding window."""
    preds = pd.Series(np.nan, index=frame.index)
    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:block_start]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < 200 or block.empty:
            continue
        model = HARQ().fit_q(train["rv"], train["rq"], train["target"])
        history_rv = frame["rv"].iloc[: block_start + len(block)]
        history_rq = frame["rq"].iloc[: block_start + len(block)]
        preds.loc[block.index] = model.predict_q(history_rv, history_rq).loc[block.index]
    return preds


def ridge_forecasts(frame: pd.DataFrame, test_start: int, refit: int) -> pd.Series:
    from sklearn.linear_model import Ridge

    feature_cols = ["rv", "rv_w", "rv_m", "abs_ret_5", "neg_ret_21"]
    preds = pd.Series(np.nan, index=frame.index)
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:block_start]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < 200 or block.empty:
            continue
        # validation tail of the training window picks alpha; scaling uses
        # train statistics only
        fit, val = train.iloc[:-63], train.iloc[-63:]
        mu, sd = fit[feature_cols].mean(), fit[feature_cols].std().replace(0, np.nan)

        def scale(df):
            return ((df[feature_cols] - mu) / sd).fillna(0.0)

        best_alpha, best_loss = alphas[0], np.inf
        for alpha in alphas:
            model = Ridge(alpha=alpha).fit(scale(fit), fit["target"])
            val_pred = pd.Series(model.predict(scale(val)), index=val.index).clip(lower=1e-4)
            loss = qlike_series(val_pred, val["target"]).mean()
            if np.isfinite(loss) and loss < best_loss:
                best_alpha, best_loss = alpha, loss

        mu, sd = train[feature_cols].mean(), train[feature_cols].std().replace(0, np.nan)
        model = Ridge(alpha=best_alpha).fit(scale(train), train["target"])
        preds.loc[block.index] = np.clip(model.predict(scale(block)), 1e-4, None)
    return preds


def har_forecasts(frame: pd.DataFrame, test_start: int, refit: int) -> pd.Series:
    preds = pd.Series(np.nan, index=frame.index)
    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:block_start]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < 200 or block.empty:
            continue
        model = HARRV().fit(train["rv"], train["target"])
        # predict needs trailing history for the weekly/monthly components,
        # so evaluate on the full series through the block end and slice
        history = frame["rv"].iloc[: block_start + len(block)]
        preds.loc[block.index] = model.predict(history).loc[block.index].clip(lower=1e-4)
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--test-frac", type=float, default=0.4)
    parser.add_argument("--refit", type=int, default=21)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    frame = build_frame(args.ticker, args.start, args.horizon)
    test_start = int(len(frame) * (1.0 - args.test_frac))
    test = frame.iloc[test_start:]
    print(f"{args.ticker}: {len(frame)} obs, OOS test {test.index[0].date()} -> {test.index[-1].date()} "
          f"({len(test)} obs, horizon {args.horizon}d)")

    forecasts = {
        "persistence": frame["rv"],
        "har_rv": har_forecasts(frame, test_start, args.refit),
        "harq": harq_forecasts(frame, test_start, args.refit),
        "pdv": pdv_forecasts(frame, test_start, args.refit),
        "har_pdv": har_pdv_forecasts(frame, test_start, args.refit),
        "ridge": ridge_forecasts(frame, test_start, args.refit),
    }

    losses = {name: qlike_series(pred.loc[test.index], test["target"]) for name, pred in forecasts.items()}
    rows = []
    for name, loss in losses.items():
        pred = forecasts[name].loc[test.index]
        mse = float(((pred - test["target"]) ** 2).mean())
        rows.append({"model": name, "qlike": float(loss.mean()), "mse": mse, "n_obs": int(loss.notna().sum())})
    table = pd.DataFrame(rows)

    # Forecasts are h-step and OVERLAPPING, so loss differentials are serially
    # correlated out to roughly h-1 lags. The generic n^(1/3) Newey-West default
    # (~13 here) understates the long-run variance and overstates significance;
    # h-1 is the standard choice for h-step forecast comparison.
    dm_lag = max(args.horizon - 1, 1)
    dm_rows = []
    for a, b in (
        ("har_rv", "persistence"),
        ("harq", "har_rv"),
        ("pdv", "har_rv"),
        ("har_pdv", "har_rv"),
        ("ridge", "har_rv"),
    ):
        stat, p = diebold_mariano(losses[a], losses[b], lag=dm_lag)
        dm_rows.append({"comparison": f"{a} vs {b} (QLIKE)", "dm_stat": stat,
                        "p_value": p, "nw_lag": dm_lag})
    dm_table = pd.DataFrame(dm_rows)

    print("\nOut-of-sample losses (lower is better):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nDiebold-Mariano (negative stat = first model better):")
    print(dm_table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"vol_benchmark_{args.ticker}.csv", index=False)
    dm_table.to_csv(out_dir / f"vol_benchmark_{args.ticker}_dm.csv", index=False)
    print(f"\nsaved -> {out_dir}/vol_benchmark_{args.ticker}[_dm].csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
