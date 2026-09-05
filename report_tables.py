#!/usr/bin/env python3
"""Print the README's results tables as markdown, straight from the results CSVs.

The numbers in a README are the part most likely to drift from the code that
produced them, because they are copied by hand. This prints them instead, so
regenerating the README after a rerun is one command and a paste rather than
twenty transcriptions.

`--inject README.md` writes the tables into the README in place, between paired
`<!-- RESULTS:NAME -->` and `<!-- END:NAME -->` markers, so the command is
idempotent: run it again after a rerun and the numbers are replaced rather than
duplicated.

Usage:
    python report_tables.py                     # every table, default tag
    python report_tables.py --tag _nonews
    python report_tables.py --which models
    python report_tables.py --inject README.md
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(value, digits=4):
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if not np.isfinite(number):
        return ""            # an infinite loss is a diagnostic, not a table cell
    return f"{number:.{digits}f}"


def markdown(frame: pd.DataFrame, headers: dict[str, str], digits=4) -> str:
    cols = [c for c in headers if c in frame.columns]
    lines = ["| " + " | ".join(headers[c] for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def models_tables(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"altdata_models{tag}.csv")
    headers = {"model": "model", "qlike_mean": "QLIKE mean",
               "qlike_median": "QLIKE median", "dm_vs_har": "DM vs HAR",
               "p_vs_har": "p", "mcs_pvalue": "MCS p", "in_mcs": "in 90% MCS"}
    for horizon in sorted(frame["horizon"].unique()):
        sub = frame[frame["horizon"] == horizon].sort_values("qlike_mean")
        print(f"\n#### Horizon {horizon} day{'s' if horizon > 1 else ''}\n")
        print(markdown(sub, headers))
        members = sub.loc[sub["in_mcs"], "model"].tolist()
        print(f"\nMCS(90%) = {{{', '.join(members)}}}")


def marginal_tables(results: Path, tag: str, rich: bool) -> None:
    name = "altdata_marginal_rich" if rich else "altdata_marginal"
    delta = "qlike_delta_vs_base" if rich else "qlike_delta_vs_har"
    dm, pv = ("dm_vs_base", "p_vs_base") if rich else ("dm_vs_har", "p_vs_har")
    frame = pd.read_csv(results / f"{name}{tag}.csv")
    headers = {"model": "model", "block": "block", "qlike_mean": "QLIKE mean",
               delta: "delta vs base", dm: "DM", pv: "p"}
    for horizon in sorted(frame["horizon"].unique()):
        sub = frame[frame["horizon"] == horizon].sort_values("qlike_mean")
        sub = sub.assign(model=sub["model"].str.replace("^(har_x__|rich_x__)", "",
                                                        regex=True))
        print(f"\n#### Horizon {horizon} day{'s' if horizon > 1 else ''}\n")
        print(markdown(sub, headers))


def regime_table(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"altdata_regime_split{tag}.csv")
    headers = {"model": "model", "qlike_calm": "QLIKE calm",
               "qlike_stressed": "QLIKE stressed",
               "delta_calm_vs_har": "calm vs HAR",
               "delta_stressed_vs_har": "stressed vs HAR"}
    for horizon in sorted(frame["horizon"].unique()):
        sub = frame[frame["horizon"] == horizon].sort_values("qlike_stressed")
        print(f"\n#### Horizon {horizon} day{'s' if horizon > 1 else ''} "
              f"({int(sub['n_stressed'].iloc[0])} stressed of "
              f"{int(sub['n_stressed'].iloc[0] + sub['n_calm'].iloc[0])} days)\n")
        print(markdown(sub, headers))


def rolling_table(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"altdata_rolling_mcs{tag}.csv")
    models = [c for c in frame.columns if c not in ("horizon", "window", "window_end")]
    for horizon in sorted(frame["horizon"].unique()):
        for window in sorted(frame["window"].unique()):
            sub = frame[(frame["horizon"] == horizon) & (frame["window"] == window)]
            share = sub[models].mean().sort_values(ascending=False)
            print(f"\n#### Horizon {horizon}, {window}-observation windows "
                  f"({len(sub)} windows)\n")
            print("| model | share of windows in the 90% MCS |")
            print("|---|---|")
            for model, value in share.items():
                print(f"| {model} | {value:.0%} |")


def volmanaged_table(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"option_pnl_volmanaged{tag}.csv")
    headers = {"model": "model", "qlike_mean": "QLIKE", "sharpe": "Sharpe",
               "mean_ann": "mean p.a.", "vol_ann": "vol p.a.",
               "max_drawdown": "max drawdown", "turnover_ann": "turnover p.a."}
    for horizon in sorted(frame["horizon"].unique()):
        sub = frame[frame["horizon"] == horizon].sort_values("sharpe", ascending=False)
        row = sub.iloc[0]
        print(f"\n#### Horizon {horizon}\n")
        print(markdown(sub, headers))
        print(f"\nQLIKE winner {row['qlike_winner']}, Sharpe winner "
              f"{row['sharpe_winner']}; Spearman(QLIKE, Sharpe) = "
              f"{row['qlike_sharpe_spearman']:+.3f} (p = "
              f"{row['qlike_sharpe_spearman_p']:.3f})")


def straddle_table(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"option_pnl_straddles{tag}.csv")
    headers = {"model": "model", "sharpe": "Sharpe", "mean_ann": "mean p.a.",
               "max_drawdown": "max drawdown", "worst_month": "worst month",
               "trade_hit_rate": "trade hit rate", "turnover_ann": "turnover p.a.",
               "n_trades": "trades"}
    for variant in frame["variant"].unique():
        sub = frame[frame["variant"] == variant].sort_values("sharpe", ascending=False)
        print(f"\n#### {variant}\n")
        print(markdown(sub, headers))


def swap_table(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"option_pnl_variance_swap{tag}.csv")
    print()
    print(frame.T.to_string(header=False))


def shar_table(results: Path, tag: str) -> None:
    frame = pd.read_csv(results / f"altdata_shar_coefficients{tag}.csv")
    headers = {"horizon": "horizon", "shar_b_pos": "b on RS+",
               "shar_b_neg": "b on RS-", "har_b_daily": "HAR b on RV",
               "corr_semivols": "corr(RS+, RS-)"}
    print()
    print(markdown(frame.sort_values("horizon"), headers))


def capture(fn) -> str:
    """Run a section and return what it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"(not generated: {exc.filename} not found)")
    return buffer.getvalue().strip("\n")


def inject(path: Path, blocks: dict[str, str]) -> int:
    """Replace each marked region of `path` with its generated table.

    Markers are paired so the edit is idempotent. On the first run a bare
    `<!-- RESULTS:NAME -->` is expanded into marker, content, `<!-- END:NAME -->`;
    on later runs everything between the pair is replaced. A marker with no
    matching section is left alone rather than silently emptied.
    """
    text = path.read_text()
    written = 0
    for name, body in blocks.items():
        key = name.upper()
        paired = re.compile(rf"<!-- RESULTS:{key} -->.*?<!-- END:{key} -->", re.S)
        block = f"<!-- RESULTS:{key} -->\n\n{body}\n\n<!-- END:{key} -->"
        if paired.search(text):
            text = paired.sub(lambda _: block, text)
            written += 1
        elif f"<!-- RESULTS:{key} -->" in text:
            text = text.replace(f"<!-- RESULTS:{key} -->", block)
            written += 1
    path.write_text(text)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--which", nargs="+", default=["all"])
    parser.add_argument("--inject", type=Path, default=None,
                        help="write the tables into this markdown file in place")
    args = parser.parse_args()

    sections = {
        "models": lambda: models_tables(args.results_dir, args.tag),
        "marginal": lambda: marginal_tables(args.results_dir, args.tag, rich=False),
        "marginal_rich": lambda: marginal_tables(args.results_dir, args.tag, rich=True),
        "regime": lambda: regime_table(args.results_dir, args.tag),
        "rolling": lambda: rolling_table(args.results_dir, args.tag),
        "shar": lambda: shar_table(args.results_dir, args.tag),
        "volmanaged": lambda: volmanaged_table(args.results_dir, args.tag),
        "straddles": lambda: straddle_table(args.results_dir, args.tag),
        "swap": lambda: swap_table(args.results_dir, args.tag),
    }
    wanted = list(sections) if args.which == ["all"] else args.which
    unknown = [n for n in wanted if n not in sections]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; pick from {list(sections)}")

    blocks = {name: capture(sections[name]) for name in wanted}
    if args.inject is None:
        for name in wanted:
            print(f"\n## {name}")
            print(blocks[name])
        return 0
    written = inject(args.inject, blocks)
    print(f"injected {written} of {len(blocks)} sections into {args.inject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
