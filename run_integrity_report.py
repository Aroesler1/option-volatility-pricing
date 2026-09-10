"""Verify corrected forecast and premium-budget P&L tables, all offline."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from option_strategies import performance
from run_inference_audit import holm_adjust
from run_vol_benchmark import qlike_series
from vol_forecasting import diebold_mariano, model_confidence_set


def build_dm_mcs(root):
    """Diebold-Mariano vs HAR and the 90% Model Confidence Set on the corrected
    (purged) per-date forecast paths, using the same `diebold_mariano` and
    `model_confidence_set` functions in `vol_forecasting.py`, and the same
    parameters run_altdata_benchmark.py used for the pre-repair table: a
    Newey-West lag of h-1 (floored at 1), alpha=0.10, 2000 bootstrap draws,
    seed 0.
    """
    rows = []
    for horizon in (1, 5, 21):
        forecasts = pd.read_csv(root / f"altdata_forecasts_h{horizon}_purged.csv",
                                parse_dates=["date"]).set_index("date")
        models = [c for c in forecasts.columns if c != "target"]
        losses = {m: qlike_series(forecasts[m], forecasts["target"]) for m in models}
        losses_frame = pd.DataFrame(losses)
        dm_lag = max(horizon - 1, 1)
        mcs = model_confidence_set(losses_frame.dropna(), alpha=0.10, n_boot=2000, seed=0)
        for model in models:
            loss = losses[model]
            if model == "har":
                stat, p = np.nan, np.nan
            else:
                stat, p = diebold_mariano(loss, losses["har"], lag=dm_lag)
            rows.append({
                "horizon": horizon,
                "model": model,
                "qlike_mean": float(loss.mean()),
                "qlike_median": float(loss.median()),
                "dm_vs_har": stat,
                "p_vs_har": p,
                "n_obs": int(loss.notna().sum()),
                "mcs_pvalue": float(mcs.loc[model, "mcs_pvalue"]),
                "in_mcs": bool(mcs.loc[model, "in_mcs"]),
            })
    return pd.DataFrame(rows)


def build(root):
    rows = []
    reported = pd.read_csv(root / "altdata_models_purged.csv").set_index(["horizon", "model"])
    old = pd.read_csv(root / "altdata_models.csv").set_index(["horizon", "model"])
    for horizon in (1, 5, 21):
        forecasts = pd.read_csv(root / f"altdata_forecasts_h{horizon}_purged.csv", parse_dates=["date"]).set_index("date")
        for model in forecasts.columns:
            if model == "target":
                continue
            loss = qlike_series(forecasts[model], forecasts.target)
            value = float(loss.mean())
            if not np.isclose(value, reported.loc[(horizon, model), "qlike_mean"], atol=1e-12):
                raise ValueError("Corrected forecast headline does not reproduce")
            rows.append(dict(horizon=horizon, model=model, n=len(loss),
                             historical_qlike=old.loc[(horizon, model), "qlike_mean"],
                             corrected_qlike=value))
    families = []
    for suffix, base in (("", "har"), ("_rich", "base")):
        f = pd.read_csv(root / f"altdata_marginal{suffix}_purged.csv")
        f = f[f[f"p_vs_{base}"].notna()].copy()
        f = f.rename(columns={f"p_vs_{base}": "p_raw", f"qlike_delta_vs_{base}": "qlike_delta"})
        f["family"] = base
        f["holm_p"] = holm_adjust(f.p_raw)
        families.append(f[["family", "horizon", "model", "qlike_delta", "p_raw", "holm_p"]])
    marginal = pd.concat(families, ignore_index=True)
    marginal["joint_holm_p"] = holm_adjust(marginal.p_raw)
    pnl = pd.read_csv(root / "option_pnl_straddles_purged.csv")
    daily = pd.read_csv(root / "option_pnl_straddle_daily_purged.csv", parse_dates=["date"]).set_index("date")
    historical = pd.read_csv(root / "option_pnl_straddles.csv").set_index(["model", "variant"])
    for row in pnl.itertuples():
        series = daily[f"{row.model}__{row.variant}"]
        if series.isna().any():
            raise ValueError("Corrected books must share the full evaluation calendar")
        stats = performance(series)
        for metric in ("sharpe", "mean_ann", "vol_ann", "max_drawdown"):
            if not np.isclose(stats[metric], getattr(row, metric), atol=1e-10, equal_nan=True):
                raise ValueError(f"Daily P&L does not reproduce {row.model}/{row.variant}/{metric}")
        components = sum(getattr(row, field) for field in
                         ("option_mid_total", "hedge_price_total", "option_spread_total", "hedge_cost_total"))
        if not np.isclose(components, series.sum(), atol=1e-10):
            raise ValueError("Trade components do not sum to daily portfolio P&L")
        if row.attempted_signals != row.n_trades + row.missing_entry + row.rejected_path:
            raise ValueError("Every attempted option signal must be accounted for")
    pnl["historical_sharpe"] = [historical.loc[(r.model, r.variant), "sharpe"]
                                 if (r.model, r.variant) in historical.index else np.nan for r in pnl.itertuples()]
    return {"integrity_forecasts.csv": pd.DataFrame(rows),
            "integrity_marginal_holm.csv": marginal,
            "integrity_pnl_comparison.csv": pnl,
            "integrity_dm_mcs.csv": build_dm_mcs(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "results"
    tables = build(root)
    for name, table in tables.items():
        if args.check:
            pd.testing.assert_frame_equal(table, pd.read_csv(root / name), check_exact=False,
                                          rtol=1e-10, atol=1e-10)
        else:
            table.to_csv(root / name, index=False)
    print("Verified corrected forecasts, family inference, every P&L row, component sums, "
          "signal admission counts, and DM/MCS on the corrected forecast paths")


if __name__ == "__main__":
    main()
