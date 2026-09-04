#!/usr/bin/env python3
"""Did the statistical gains become money? Three economic tests per model.

a. Volatility-managed SPY. Weight = 15% annualized target divided by the model's
   forecast, capped at 2, rebalanced daily, 5 bps per unit of turnover. This is
   the cheapest use of a variance forecast and needs no options, so it isolates
   whether a lower QLIKE is worth anything at all. The QLIKE winner and the
   Sharpe winner are compared by a paired block bootstrap, because they are
   usually not the same model and the difference between them is the whole
   question.

b. Delta-hedged SPY straddles. At each forecast date the 21-day forecast, with
   the variance risk premium added back, is compared to 30-day at-the-money
   implied variance; long a straddle when the forecast is higher by a frozen
   margin, short when lower, flat otherwise. A long-only variant is run
   alongside because a short straddle book carries unbounded crash risk and
   its Sharpe is not comparable to a long one's without saying so.

c. A model-free synthetic variance swap: realized variance over the next 21
   days minus VIX squared (Carr and Wu, RFS 2009). No forecast is involved, so
   it measures the variance risk premium every straddle strategy is trading
   around, and it is the number the forecast-driven strategies have to beat.

Requires `run_altdata_benchmark.py` to have been run first (it reads the saved
per-model forecasts) and `fetch_option_chain.py` for the option quotes.

Usage:
    python run_option_pnl.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from option_strategies import (
    StraddleConfig,
    paired_block_bootstrap_pvalue,
    performance,
    straddle_backtest,
    straddle_signal,
    variance_swap_pnl,
    variance_swap_summary,
    volatility_managed,
)
from run_vol_benchmark import qlike_series


def load_forecasts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return frame


def load_option_market(data_dir: Path) -> pd.DataFrame:
    return (pd.read_csv(data_dir / "features_option_market.csv", parse_dates=["date"])
            .set_index("date").sort_index())


def run_vol_managed(forecasts: pd.DataFrame, spy_ret: pd.Series, target_vol: float,
                    cap: float, cost_bps: float) -> tuple[pd.DataFrame, dict]:
    models = [c for c in forecasts.columns if c != "target"]
    rows, series = [], {}
    for model in models:
        res = volatility_managed(forecasts[model], spy_ret, target_vol, cap, cost_bps)
        series[model] = res["net"]
        stats = performance(res["net"], res["turnover"])
        stats["model"] = model
        stats["qlike_mean"] = float(qlike_series(forecasts[model],
                                                 forecasts["target"]).mean())
        rows.append(stats)
    # the buy-and-hold leg, for scale: a volatility-managed strategy that cannot
    # beat holding the index has timed nothing
    bh = spy_ret.reindex(forecasts.index).dropna()
    bh_stats = performance(bh)
    bh_stats["model"] = "buy_and_hold"
    bh_stats["qlike_mean"] = np.nan
    rows.append(bh_stats)
    series["buy_and_hold"] = bh
    table = pd.DataFrame(rows)
    front = ["model", "qlike_mean", "sharpe", "mean_ann", "vol_ann", "max_drawdown",
             "worst_month", "turnover_ann", "n_days"]
    table = table[[c for c in front if c in table.columns]]
    return table.sort_values("sharpe", ascending=False), series


def run_straddles(forecasts: pd.DataFrame, implied: pd.Series, chain: pd.DataFrame,
                  entries: pd.DataFrame, underlying: pd.Series, config: StraddleConfig,
                  margin_quantile: float) -> tuple[pd.DataFrame, dict, dict]:
    models = [c for c in forecasts.columns if c != "target"]
    rows, series, rules = [], {}, {}
    for model in models:
        side, rule = straddle_signal(forecasts[model], implied,
                                     margin_quantile=margin_quantile)
        rules[model] = rule
        for label, long_only in (("both", False), ("long_only", True)):
            book, trades = straddle_backtest(chain, entries, underlying, side,
                                             config, long_only=long_only)
            if book.empty:
                continue
            stats = performance(book)
            years = max(len(book) / 252.0, 1e-9)
            stats.update({
                "model": model, "variant": label,
                "n_trades": int(len(trades)),
                "trade_hit_rate": float((trades["trade_return"] > 0).mean()),
                "mean_trade_return": float(trades["trade_return"].mean()),
                "long_share": float((trades["direction"] > 0).mean()),
                # each trade commits 1/hold_days of the premium budget, so this
                # is premium turned over per year as a fraction of capital
                "turnover_ann": float(len(trades) / years / config.hold_days),
                "trades_per_year": float(len(trades) / years),
            })
            rows.append(stats)
            series[f"{model}__{label}"] = book
    table = pd.DataFrame(rows)
    front = ["model", "variant", "sharpe", "mean_ann", "vol_ann", "max_drawdown",
             "worst_month", "hit_rate", "trade_hit_rate", "mean_trade_return",
             "turnover_ann", "n_trades", "trades_per_year", "long_share", "n_days"]
    table = table[[c for c in front if c in table.columns]]
    return table.sort_values(["variant", "sharpe"], ascending=[True, False]), series, rules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21])
    parser.add_argument("--straddle-horizon", type=int, default=21)
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--cap", type=float, default=2.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--hold-days", type=int, default=21)
    parser.add_argument("--margin-quantile", type=float, default=0.5)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    under = (pd.read_csv(args.data_dir / "SPY_optionmetrics_close.csv",
                         parse_dates=["date"]).set_index("date").sort_index())
    option_market = load_option_market(args.data_dir)

    # ---- a. volatility-managed SPY, at every horizon -----------------------
    vm_tables, vm_series = [], {}
    for horizon in args.horizons:
        path = args.results_dir / f"altdata_forecasts_h{horizon}{args.tag}.csv"
        if not path.exists():
            print(f"skipping horizon {horizon}: {path} not found")
            continue
        forecasts = load_forecasts(path)
        table, series = run_vol_managed(forecasts, under["ret"], args.target_vol,
                                        args.cap, args.cost_bps)
        table.insert(0, "horizon", horizon)
        vm_tables.append(table)
        vm_series[horizon] = series
        print(f"\n{'=' * 78}\na. volatility-managed SPY, {horizon}-day forecast "
              f"(target {args.target_vol:.0%}, cap {args.cap:g}, {args.cost_bps:g} bps)")
        print(table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))

        forecast_models = [c for c in forecasts.columns if c != "target"]
        qlike_winner = min(forecast_models,
                           key=lambda m: qlike_series(forecasts[m], forecasts["target"]).mean())
        sharpe_winner = table[table["model"] != "buy_and_hold"].iloc[0]["model"]
        # The headline comparison is QLIKE winner against Sharpe winner. When
        # they are the same model that test is vacuous by construction, so the
        # Sharpe winner is also compared against HAR and against holding the
        # index; one of those is always a comparison that can fail.
        comparisons = {
            "qlike_winner": qlike_winner,
            "har": "har",
            "buy_and_hold": "buy_and_hold",
        }
        stats = {"qlike_winner": qlike_winner, "sharpe_winner": sharpe_winner}
        for label, other in comparisons.items():
            if other not in series or other == sharpe_winner:
                stats[f"diff_vs_{label}"] = 0.0 if other == sharpe_winner else np.nan
                stats[f"p_vs_{label}"] = np.nan
                continue
            diff, p = paired_block_bootstrap_pvalue(series[sharpe_winner],
                                                    series[other],
                                                    n_boot=args.n_boot)
            stats[f"diff_vs_{label}"] = diff
            stats[f"p_vs_{label}"] = p
            print(f"  Sharpe winner {sharpe_winner} vs {other}: "
                  f"Sharpe difference {diff:+.3f}, "
                  f"paired block bootstrap p = {p:.3f}")
        if sharpe_winner == qlike_winner:
            print(f"  QLIKE winner and Sharpe winner are the same model "
                  f"({sharpe_winner}), so that comparison is vacuous here")
        vm_tables[-1] = vm_tables[-1].assign(**stats)

    if vm_tables:
        pd.concat(vm_tables, ignore_index=True).to_csv(
            args.results_dir / f"option_pnl_volmanaged{args.tag}.csv", index=False)

    # ---- b. delta-hedged straddles ----------------------------------------
    chain_path = args.data_dir / "option_chain_spy.parquet"
    entries_path = args.data_dir / "option_chain_spy_entries.parquet"
    straddle_path = args.results_dir / f"altdata_forecasts_h{args.straddle_horizon}{args.tag}.csv"
    if chain_path.exists() and entries_path.exists() and straddle_path.exists():
        chain = pd.read_parquet(chain_path)
        entries = pd.read_parquet(entries_path)
        forecasts = load_forecasts(straddle_path)
        implied = option_market["atm_iv_30"].reindex(forecasts.index)
        config = StraddleConfig(hold_days=args.hold_days,
                                hedge_cost_bps=args.cost_bps)
        table, series, rules = run_straddles(forecasts, implied, chain, entries,
                                             under["close"], config,
                                             args.margin_quantile)
        print(f"\n{'=' * 78}\nb. delta-hedged ATM straddles, "
              f"{args.straddle_horizon}-day forecast vs 30-day implied, "
              f"held {args.hold_days} days")
        example = next(iter(rules.values()))
        print(f"  rule frozen on the first half: premium adjustment "
              f"{example['premium']:+.5f} variance, margin {example['margin']:.5f}, "
              f"calibration ends {example['calibration_end']}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        table.to_csv(args.results_dir / f"option_pnl_straddles{args.tag}.csv", index=False)
        pd.DataFrame(rules).T.to_csv(
            args.results_dir / f"option_pnl_straddle_rules{args.tag}.csv")
    else:
        print("\nb. skipped: option chain or forecasts not found")

    # ---- c. model-free variance swap ---------------------------------------
    if straddle_path.exists():
        forecasts = load_forecasts(straddle_path)
        vix = option_market["vix"].reindex(forecasts.index) / 100.0
        swap = variance_swap_pnl(vix, forecasts["target"])
        stats = variance_swap_summary(swap, horizon=args.straddle_horizon)
        stats["model"] = "long_variance_swap_vix"
        print(f"\n{'=' * 78}\nc. synthetic variance swap, long realized against "
              f"VIX squared, {args.straddle_horizon}-day")
        print(pd.Series(stats).to_string())
        pd.DataFrame([stats]).to_csv(
            args.results_dir / f"option_pnl_variance_swap{args.tag}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
