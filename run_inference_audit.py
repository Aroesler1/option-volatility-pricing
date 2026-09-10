#!/usr/bin/env python3
"""Reproduce the audit from derived CSVs, without a vendor login.

`--rebuild-straddles --chain-dir data` additionally rebuilds daily P&L from
existing licensed option-chain caches. It never fetches data. Historical
results are read and retained; all audit outputs use the `audit_` prefix.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from option_strategies import StraddleConfig, performance
from run_option_pnl import run_straddles
from run_vol_benchmark import qlike_series


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm's step-down adjusted p-values, preserving missing baselines."""
    p = pd.to_numeric(pvalues, errors="raise")
    valid = p.dropna()
    if not np.isfinite(valid).all() or not valid.between(0, 1).all():
        raise ValueError("p-values must be finite and between zero and one")
    result = pd.Series(np.nan, index=p.index, dtype=float)
    ordered = valid.sort_values(kind="stable")
    adjusted = np.maximum.accumulate(ordered.to_numpy() *
                                     (len(ordered) - np.arange(len(ordered))))
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def marginal_audit(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = []
    for rich in (False, True):
        suffix, base = ("_rich", "base") if rich else ("", "har")
        source = pd.read_csv(results / f"altdata_marginal{suffix}.csv")
        source = source[source[f"p_vs_{base}"].notna()].copy()
        table = source.rename(columns={f"p_vs_{base}": "p_raw",
                                        f"qlike_delta_vs_{base}": "qlike_delta"})
        table["family"] = "rich_har" if rich else "har"
        table["family_n"] = len(table)
        table["p_holm"] = holm_adjust(table["p_raw"])
        families.append(table[["family", "family_n", "horizon", "model",
                               "qlike_delta", "p_raw", "p_holm"]])
    table = pd.concat(families, ignore_index=True)
    # The joint sensitivity guards against choosing the more flattering base.
    table["joint_family_n"] = len(table)
    table["p_holm_joint"] = holm_adjust(table["p_raw"])
    rows = []
    for family, group in table.groupby("family", sort=False):
        row = {"family": family, "n_tests": len(group)}
        for label, pcol in (("raw", "p_raw"), ("holm", "p_holm"),
                            ("joint_holm", "p_holm_joint")):
            for direction, sign in (("better", -1), ("worse", 1)):
                row[f"{label}_{direction}"] = int(
                    ((group[pcol] < 0.05) & (sign * group["qlike_delta"] > 0)).sum())
        rows.append(row)
    return table, pd.DataFrame(rows)


def headline_reproduction(results: Path) -> pd.DataFrame:
    reported = pd.read_csv(results / "altdata_models.csv").set_index(["horizon", "model"])
    rows = []
    for horizon in (1, 5, 21):
        forecasts = pd.read_csv(results / f"altdata_forecasts_h{horizon}.csv",
                                parse_dates=["date"]).set_index("date")
        for model in ("har", "har_rv_iv", "har_x_lasso"):
            measured = float(qlike_series(forecasts[model], forecasts["target"]).mean())
            expected = float(reported.loc[(horizon, model), "qlike_mean"])
            if not np.isclose(measured, expected, rtol=1e-11, atol=1e-12):
                raise ValueError(f"headline mismatch: {horizon}, {model}")
            rows.append({"horizon": horizon, "model": model, "n_forecasts": len(forecasts),
                         "start": str(forecasts.index.min().date()),
                         "end": str(forecasts.index.max().date()),
                         "reported_qlike": expected, "reproduced_qlike": measured})
    return pd.DataFrame(rows)


def rebuild_straddles(results: Path, data: Path, chain_dir: Path) -> None:
    forecasts = pd.read_csv(results / "altdata_forecasts_h21.csv",
                            parse_dates=["date"]).set_index("date")
    option_market = pd.read_csv(data / "features_option_market.csv",
                                parse_dates=["date"]).set_index("date")
    underlying = pd.read_csv(data / "SPY_daily.csv",
                             parse_dates=["date"]).set_index("date")
    chain = pd.read_parquet(chain_dir / "option_chain_spy.parquet")
    entries = pd.read_parquet(chain_dir / "option_chain_spy_entries.parquet")
    summary, daily, rules = run_straddles(
        forecasts, option_market["atm_iv_30"], chain, entries,
        underlying["close"], StraddleConfig(), margin_quantile=0.5)
    pd.DataFrame(daily).rename_axis("date").to_csv(results / "audit_straddle_daily.csv")
    # Derived counts and outcomes, with no contracts, strikes, or option quotes.
    cols = ["model", "variant", "n_trades", "trade_hit_rate", "mean_trade_return",
            "long_share", "turnover_ann", "trades_per_year"]
    summary[cols].to_csv(results / "audit_straddle_trade_statistics.csv", index=False)
    pd.DataFrame(rules).T.rename_axis("model").to_csv(results / "audit_straddle_rules.csv")


def summarize_straddles(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(results / "audit_straddle_daily.csv",
                        parse_dates=["date"]).set_index("date")
    if daily.empty or daily.isna().any().any() or not daily.index.is_unique:
        raise ValueError("daily P&L needs a nonempty, complete common calendar")
    rows = []
    for name, series in daily.items():
        model, variant = name.split("__", 1)
        rows.append(dict(model=model, variant=variant, **performance(series),
                         evaluation_start=str(daily.index.min().date()),
                         evaluation_end=str(daily.index.max().date())))
    summary = pd.DataFrame(rows).merge(
        pd.read_csv(results / "audit_straddle_trade_statistics.csv"),
        on=["model", "variant"], validate="one_to_one")
    old = pd.read_csv(results / "option_pnl_straddles.csv")
    cols = ["model", "variant", "sharpe", "mean_ann", "n_days", "n_trades"]
    comparison = old[cols].merge(summary[cols], on=["model", "variant"],
                                 suffixes=("_historical", "_corrected"), validate="one_to_one")
    return summary, comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--chain-dir", type=Path, default=Path("data"))
    parser.add_argument("--rebuild-straddles", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.rebuild_straddles:
        parser.error("--check is read-only and cannot rebuild straddles")
    if args.rebuild_straddles:
        rebuild_straddles(args.results_dir, args.data_dir, args.chain_dir)
    features, feature_summary = marginal_audit(args.results_dir)
    straddles, comparison = summarize_straddles(args.results_dir)
    tables = {"audit_headline_reproduction": headline_reproduction(args.results_dir),
              "audit_marginal_holm": features,
              "audit_marginal_holm_summary": feature_summary,
              "audit_straddle_summary": straddles,
              "audit_straddle_comparison": comparison}
    for name, frame in tables.items():
        path = args.results_dir / f"{name}.csv"
        if args.check:
            pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False,
                                          rtol=1e-10, atol=1e-12)
        else:
            frame.to_csv(path, index=False)
    print(f"{'Verified' if args.check else 'Wrote'} {len(tables)} audit tables from derived CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
