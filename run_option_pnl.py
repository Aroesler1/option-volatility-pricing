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
per-model forecasts), `fetch_option_chain.py` for the option quotes, and
`fetch_spy_daily.py` for the underlying closes and total returns.

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


def rank_agreement(qlike: pd.Series, sharpe: pd.Series) -> tuple[float, float]:
    """Spearman rank correlation between forecast loss and strategy Sharpe.

    This is the whole question in one number. QLIKE is a loss, so a model that
    forecasts better has a LOWER QLIKE and, if the statistics matter
    economically, a HIGHER Sharpe. The correlation should therefore be
    NEGATIVE, and the more negative it is the more the forecast ranking
    survives contact with trading costs. It says nothing about whether the
    strategy makes money, only whether being better at forecasting puts you
    higher up the P&L table.
    """
    from scipy.stats import spearmanr

    frame = pd.concat([pd.to_numeric(qlike, errors="coerce").rename("q"),
                       pd.to_numeric(sharpe, errors="coerce").rename("s")],
                      axis=1).dropna()
    if len(frame) < 4:
        return np.nan, np.nan
    rho, p = spearmanr(frame["q"], frame["s"])
    return float(rho), float(p)


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
    calendar = next(iter(series.values())).index
    bh = spy_ret.reindex(calendar)
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
    # One aligned rule-calibration and evaluation calendar for every model.
    # Selecting it once also prevents an unconditional control from trading
    # dates on which the conditional rules were still being calibrated.
    aligned = forecasts.join(implied.rename("_implied"), how="inner").dropna()
    if not aligned.index.is_unique or not aligned.index.is_monotonic_increasing:
        raise ValueError("forecasts must have unique, increasing dates")
    implied = aligned.pop("_implied")
    forecasts = aligned
    models = [c for c in forecasts.columns if c != "target"]
    rows, series, rules = [], {}, {}
    signals = {}
    for model in models:
        signals[model], rules[model] = straddle_signal(
            forecasts[model], implied, margin_quantile=margin_quantile)
    decision_start = pd.Timestamp(rules[models[0]]["evaluation_start"])
    market_calendar = underlying.index[(underlying.index >= forecasts.index.min())
                                        & (underlying.index <= forecasts.index.max())]
    # A close-t statistic selects a trade at close t+1. Shift on market sessions,
    # not on the surviving aligned forecast rows.
    for model in models:
        signals[model] = signals[model].reindex(market_calendar).shift(1).fillna(0)
        rules[model]["signal_lag_sessions"] = 1
    evaluation_start = market_calendar[market_calendar > decision_start][0]
    evaluation_dates = market_calendar[market_calendar >= evaluation_start]
    unconditional = {
        "always_short": pd.Series(-1, index=evaluation_dates),
        "always_long": pd.Series(1, index=evaluation_dates),
    }
    books, trade_tables = {}, {}
    for model in models + list(unconditional):
        if model in unconditional:
            side, rule = unconditional[model], {"premium": np.nan, "margin": 0.0,
                                                "calibration_end": "none",
                                                "n_calibration": 0,
                                                "evaluation_start": str(evaluation_start.date())}
        else:
            side, rule = signals[model].loc[evaluation_dates], rules[model]
        rules[model] = rule
        for label, long_only in (("both", False), ("long_only", True)):
            if model == "always_short" and long_only:
                continue                    # a long-only filter empties it
            book, trades = straddle_backtest(chain, entries, underlying, side,
                                             config, long_only=long_only)
            books[(model, label)], trade_tables[(model, label)] = book, trades
    # Include flat days and any final positions' runoff for every book. No
    # strategy earns a shorter denominator just because its last signal differs.
    end = max(book.index.max() for book in books.values())
    calendar = underlying.index[(underlying.index >= evaluation_start)
                                & (underlying.index <= end)].union(evaluation_dates)
    for (model, label), book in books.items():
        book = book.reindex(calendar).fillna(0.0)
        trades = trade_tables[(model, label)]
        stats = performance(book)
        years = max(len(book) / 252.0, 1e-9)
        stats.update({
            "model": model, "variant": label,
            "n_trades": int(len(trades)),
            "trade_hit_rate": float((trades["trade_return"] > 0).mean()),
            "mean_trade_return": float(trades["trade_return"].mean()),
            "long_share": float((trades["direction"] > 0).mean()),
            # each trade commits 1/hold_days of the premium budget
            "turnover_ann": float(len(trades) / years / config.hold_days),
            "trades_per_year": float(len(trades) / years),
            "evaluation_start": str(calendar.min().date()),
            "evaluation_end": str(calendar.max().date()),
        })
        for component in ("option_mid", "hedge_price", "option_spread", "hedge_cost"):
            stats[component + "_total"] = (float(trades[component].sum()) / config.hold_days
                                             if component in trades else np.nan)
        stats.update(trades.attrs)
        stats["evaluation_status"] = "retrospective_complete_quote_paths_premium_budget"
        rows.append(stats)
        series[f"{model}__{label}"] = book
    table = pd.DataFrame(rows)
    front = ["model", "variant", "sharpe", "mean_ann", "vol_ann", "max_drawdown",
             "worst_month", "hit_rate", "trade_hit_rate", "mean_trade_return",
             "turnover_ann", "n_trades", "trades_per_year", "long_share", "n_days",
             "evaluation_start", "evaluation_end", "option_mid_total", "hedge_price_total",
             "option_spread_total", "hedge_cost_total", "attempted_signals",
             "missing_entry", "rejected_path", "evaluation_status"]
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

    under = (pd.read_csv(args.data_dir / "SPY_daily.csv",
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
        pd.DataFrame(series).to_csv(args.results_dir / f"option_pnl_vm_daily_h{horizon}{args.tag}.csv",
                                    index_label="date")
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
        traded = table[table["model"] != "buy_and_hold"]
        rho, rho_p = rank_agreement(traded.set_index("model")["qlike_mean"],
                                    traded.set_index("model")["sharpe"])
        print(f"  Spearman(QLIKE, Sharpe) across models = {rho:+.3f}, p = {rho_p:.3f}"
              f"  (negative means the forecast ranking survives into the P&L ranking)")
        stats["qlike_sharpe_spearman"] = rho
        stats["qlike_sharpe_spearman_p"] = rho_p
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
        qlike_by_model = {m: float(qlike_series(forecasts[m], forecasts["target"]).mean())
                          for m in forecasts.columns if m != "target"}
        table["qlike_mean"] = table["model"].map(qlike_by_model)
        for variant in table["variant"].unique():
            # the unconditional books have no forecast, so they are excluded
            # from the rank correlation rather than given a missing rank
            sub = table[(table["variant"] == variant) & table["qlike_mean"].notna()]
            rho, rho_p = rank_agreement(sub["qlike_mean"], sub["sharpe"])
            print(f"  {variant}: Spearman(QLIKE, Sharpe) = {rho:+.3f}, p = {rho_p:.3f}")
        table.to_csv(args.results_dir / f"option_pnl_straddles{args.tag}.csv", index=False)
        pd.DataFrame(series).to_csv(args.results_dir / f"option_pnl_straddle_daily{args.tag}.csv",
                                    index_label="date")
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
