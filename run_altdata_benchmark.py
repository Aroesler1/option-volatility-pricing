#!/usr/bin/env python3
"""Does alternative data improve HAR forecasts of SPY realized variance, and where?

Three horizons (1, 5 and 21 trading days), one out-of-sample protocol, one loss.
Every model sees the same dates, is refit on the same expanding-window schedule,
and is scored under QLIKE, so a difference between two rows is a difference in
information rather than in machinery.

Structural models (Table A, and the Model Confidence Set is run over these):
    persistence   today's realized volatility, carried forward
    har           HAR-RV (Corsi 2009), the benchmark everything is measured against
    shar          semivariance HAR (Patton and Sheppard 2015), which needs no
                  data beyond the price path and is the literature's strongest
                  horizon-stable improvement
    har_rv_iv     HAR plus 30-day at-the-money implied VARIANCE (Busch,
                  Christensen and Nielsen 2011)
    har_x_lasso   HAR plus every alternative-data feature, with LASSO selecting
                  which of them survive, refit inside the walk-forward so the
                  selection itself is out of sample
    hgb           sklearn HistGradientBoosting on the HAR terms plus every
                  feature, which can find interactions a linear model cannot
    lstm          the notebook's LSTM on the HAR terms alone, on the corrected
                  protocol (forward target, training-window scaling)
    lstm_x        the same network with every feature added, so the neural
                  arm's own answer to the question is separable from the
                  network itself
    combination   equal-weight mean of every model above except persistence

Marginal value (Table B): HAR plus ONE feature, once per feature, so each
feature's incremental contribution is visible instead of being pooled into a
single kitchen-sink number that cannot be attributed.

Marginal value against a strong base (Table C): the same one-feature-at-a-time
exercise, but on top of HAR plus semivariance plus implied variance rather than
on top of bare HAR. Most of the alternative-data literature tests one data type
against a bare HAR benchmark, which overstates what the data adds once the two
cheap improvements are already in the model. Table C is the honest version of
Table B and the two are reported side by side.

Sample. The alternative-data panel is limited at the right-hand end by
OptionMetrics coverage in WRDS, so this study stops there while the baseline
chapter of the README keeps its longer sample. Every model is evaluated on the
identical set of dates, which the MCS requires anyway.

Usage:
    python run_altdata_benchmark.py
    python run_altdata_benchmark.py --extra-lag 1     # one more day of caution
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from typing import Optional

import numpy as np
import pandas as pd

from alt_data import ALL_FEATURES, FEATURE_BLOCK, build_feature_panel
from run_intraday_benchmark import forward_vol_from_variance
from run_vol_benchmark import qlike_series
from vol_forecasting import (
    HARRV,
    HARX,
    SHARRV,
    semivol,
    diebold_mariano,
    mean_combination,
    model_confidence_set,
    rolling_model_confidence_set,
)

FLOOR = 1e-4
MIN_TRAIN = 400
COMBINATION_MEMBERS = ("har", "shar", "har_rv_iv", "har_x_lasso", "hgb",
                       "lstm", "lstm_x")
# HAR plus the two improvements that need no alternative data at all: the
# signed split of the daily term, and what the option market already prices
RICH_BASE = ("rs_pos_vol", "rs_neg_vol", "atm_ivar_30")


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def walk_forward(frame: pd.DataFrame, test_start: int, refit: int, fit_predict,
                 min_train: int = MIN_TRAIN, purge: int = 0) -> pd.Series:
    """Expanding-window refit every `refit` rows, with the training window purged.

    `fit_predict(train, history)` fits on `train` and returns predictions over
    `history`, which runs from the start of the sample to the end of the block.
    Models with trailing regressors (every HAR variant) need that history to
    form the weekly and monthly averages for the block's first rows, so passing
    the block alone would silently drop them.

    PURGING. The target at row t is realized variance over (t, t+h], so the last
    h rows of a training window that ends where the test block begins have
    targets reaching INTO the test block. Dropping those h rows is the standard
    purge for overlapping targets. It costs about 2% of the training rows and
    removes the only channel by which a model here could see its own evaluation
    period. The baseline scripts in this repo (`run_vol_benchmark.py`,
    `run_intraday_benchmark.py`) do not purge; the effect is small and identical
    across their models, but it is a real difference in protocol and is stated
    in the README rather than left for a reader to find.
    """
    preds = pd.Series(np.nan, index=frame.index)
    for block_start in range(test_start, len(frame), refit):
        train = frame.iloc[:max(block_start - purge, 0)]
        block = frame.iloc[block_start:block_start + refit]
        if len(train) < min_train or block.empty:
            continue
        history = frame.iloc[: block_start + len(block)]
        preds.loc[block.index] = fit_predict(train, history).loc[block.index]
    return preds.clip(lower=FLOOR)


def har_fp(train, history):
    model = HARRV().fit(train["rv"], train["target"])
    return model.predict(history["rv"])


def shar_fp(train, history):
    model = SHARRV().fit_shar(train["rv"], train["rs_pos_var"], train["rs_neg_var"],
                              train["target"])
    return model.predict_shar(history["rv"], history["rs_pos_var"],
                              history["rs_neg_var"])


def make_harx_fp(cols):
    def fp(train, history):
        model = HARX().fit_x(train["rv"], train[cols], train["target"])
        return model.predict_x(history["rv"], history[cols])
    return fp


def make_lasso_fp(cols, val_tail: int = 126):
    """HAR + all features, LASSO-selected, penalty chosen on a validation tail.

    The penalty is chosen by QLIKE on the last `val_tail` rows of the TRAINING
    window and the model is then refit on the whole training window, so nothing
    about the test period touches either the selection or the fit. Features are
    standardised with training-window statistics for the same reason; LASSO is
    not scale-invariant, so unstandardised features would let the penalty pick
    on units.
    """
    from sklearn.linear_model import Lasso

    har_cols = ["rv", "rv_w", "rv_m"]
    alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    def fp(train, history):
        use = har_cols + list(cols)
        fit_part, val_part = train.iloc[:-val_tail], train.iloc[-val_tail:]
        if len(fit_part) < MIN_TRAIN // 2:
            fit_part, val_part = train, train.iloc[-val_tail:]
        mu, sd = fit_part[use].mean(), fit_part[use].std().replace(0, np.nan)
        scale = lambda df: ((df[use] - mu) / sd).fillna(0.0)
        best_alpha, best_loss = alphas[0], np.inf
        for alpha in alphas:
            m = Lasso(alpha=alpha, max_iter=20000).fit(scale(fit_part), fit_part["target"])
            pred = pd.Series(m.predict(scale(val_part)), index=val_part.index).clip(lower=FLOOR)
            loss = qlike_series(pred, val_part["target"]).mean()
            if np.isfinite(loss) and loss < best_loss:
                best_alpha, best_loss = alpha, loss
        mu, sd = train[use].mean(), train[use].std().replace(0, np.nan)
        model = Lasso(alpha=best_alpha, max_iter=20000).fit(scale(train), train["target"])
        return pd.Series(model.predict(scale(history)), index=history.index)
    return fp


def make_lstm_fp(cols, seq_len: int = 10, val_tail: int = 126, seed: int = 0):
    """The notebook LSTM, refit on the same schedule as everything else.

    Refitting a network every 21 days is expensive but it is the protocol every
    other model here runs under, and giving the network a single fit over the
    whole training period while HAR refits monthly would compare two different
    experiments.
    """
    from lstm_forecasting import LSTMForecaster

    har_cols = ["rv", "rv_w", "rv_m"]

    def fp(train, history):
        use = har_cols + list(cols)
        model = LSTMForecaster(seq_len=seq_len, val_tail=val_tail, seed=seed)
        model.fit(train, use)
        return model.predict(history)
    return fp


def make_hgb_fp(cols, val_tail: int = 126):
    """Gradient-boosted trees on the HAR terms plus every feature.

    LightGBM is not used; sklearn's HistGradientBoosting is the same histogram
    algorithm and keeps the dependency list to one library. Depth and iteration
    count are small on purpose: roughly 1,100 training rows with 18 features is
    not a regime in which a deep ensemble can be fit honestly.

    Early stopping is done by hand rather than with sklearn's `early_stopping`
    flag. That flag holds out a RANDOM fraction of the training rows, which on a
    time series puts later rows into the validation set and earlier ones into
    training. Here the number of boosting rounds is chosen by QLIKE on the last
    `val_tail` rows of the training window, in time order, and the model is then
    refit on the whole training window with that count, which is the same
    protocol the LASSO penalty is chosen under.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    har_cols = ["rv", "rv_w", "rv_m"]
    iteration_grid = (50, 100, 200, 400)

    def _make(max_iter):
        return HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=0.05, max_depth=3,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
            random_state=0)

    def fp(train, history):
        use = har_cols + list(cols)
        fit_part, val_part = train.iloc[:-val_tail], train.iloc[-val_tail:]
        best_iter, best_loss = iteration_grid[0], np.inf
        if len(fit_part) >= MIN_TRAIN // 2:
            for n_iter in iteration_grid:
                model = _make(n_iter).fit(fit_part[use], fit_part["target"])
                pred = pd.Series(model.predict(val_part[use]),
                                 index=val_part.index).clip(lower=FLOOR)
                loss = qlike_series(pred, val_part["target"]).mean()
                if np.isfinite(loss) and loss < best_loss:
                    best_iter, best_loss = n_iter, loss
        model = _make(best_iter).fit(train[use], train["target"])
        return pd.Series(model.predict(history[use].fillna(0.0)), index=history.index)
    return fp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_frame(intraday_path: Path, horizon: int, data_dir: Path,
                extra_lag: int, features: Optional[list] = None
                ) -> tuple[pd.DataFrame, list[str], str]:
    intra = pd.read_csv(intraday_path, parse_dates=["date"]).set_index("date").sort_index()
    for col in ("rs_pos_var", "rs_neg_var"):
        if col not in intra.columns:
            raise SystemExit(
                f"{intraday_path} has no `{col}`; rerun build_intraday_rv.py")

    rv = intra["rv5m"]
    wanted = list(features) if features else list(ALL_FEATURES)
    panel, report = build_feature_panel(rv.index, data_dir=data_dir,
                                        extra_lag=extra_lag, features=wanted)
    base = pd.DataFrame({
        "rv": rv,
        "rv_w": rv.rolling(5).mean(),
        "rv_m": rv.rolling(22).mean(),
        "rs_pos_var": intra["rs_pos_var"],
        "rs_neg_var": intra["rs_neg_var"],
        "rs_pos_vol": semivol(intra["rs_pos_var"]),
        "rs_neg_vol": semivol(intra["rs_neg_var"]),
        "target": forward_vol_from_variance(intra["rv5m_var"], horizon),
    })
    frame = pd.concat([base, panel[wanted]], axis=1).dropna()
    return frame, wanted, report.to_string()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def score(forecasts: dict[str, pd.Series], test: pd.DataFrame, horizon: int,
          benchmark: str = "har") -> tuple[pd.DataFrame, pd.DataFrame]:
    losses = {k: qlike_series(v.loc[test.index], test["target"])
              for k, v in forecasts.items()}
    dm_lag = max(horizon - 1, 1)
    rows = []
    for name, loss in losses.items():
        pred = forecasts[name].loc[test.index]
        stat, p = ((np.nan, np.nan) if name == benchmark
                   else diebold_mariano(loss, losses[benchmark], lag=dm_lag))
        rows.append({
            "model": name,
            "qlike_mean": float(loss.mean()),
            "qlike_median": float(loss.median()),
            "mse": float(((pred - test["target"]) ** 2).mean()),
            "collapsed": int((pred <= 1.01 * FLOOR).sum()),
            "dm_vs_har": stat,
            "p_vs_har": p,
            "n_obs": int(loss.notna().sum()),
        })
    return pd.DataFrame(rows), pd.DataFrame(losses)


def attach_mcs(table: pd.DataFrame, losses: pd.DataFrame, alpha: float,
               n_boot: int, seed: int) -> pd.DataFrame:
    mcs = model_confidence_set(losses.dropna(), alpha=alpha, n_boot=n_boot, seed=seed)
    return table.merge(mcs[["mcs_pvalue", "in_mcs"]].rename_axis("model").reset_index(),
                       on="model", how="left")


def run_horizon(args, horizon: int) -> dict[str, pd.DataFrame]:
    frame, features, coverage = build_frame(args.intraday, horizon, args.data_dir,
                                            args.extra_lag, args.features)
    test_start = int(len(frame) * (1.0 - args.test_frac))
    test = frame.iloc[test_start:]
    purge = horizon
    print(f"\n{'=' * 78}\nhorizon {horizon}d: {len(frame):,} aligned days "
          f"{frame.index.min().date()} -> {frame.index.max().date()}; "
          f"OOS {len(test):,} from {test.index[0].date()}")

    structural = {
        "persistence": frame["rv"],
        "har": walk_forward(frame, test_start, args.refit, har_fp, purge=purge),
        "shar": walk_forward(frame, test_start, args.refit, shar_fp, purge=purge),
        "har_rv_iv": walk_forward(frame, test_start, args.refit,
                                  make_harx_fp(["atm_ivar_30"]), purge=purge),
        "har_x_lasso": walk_forward(frame, test_start, args.refit,
                                    make_lasso_fp(features), purge=purge),
        "hgb": walk_forward(frame, test_start, args.refit, make_hgb_fp(features),
                            purge=purge),
        "lstm": walk_forward(frame, test_start, args.refit,
                             make_lstm_fp([], seed=args.seed), purge=purge),
        "lstm_x": walk_forward(frame, test_start, args.refit,
                               make_lstm_fp(features, seed=args.seed), purge=purge),
    }
    structural["combination"] = mean_combination(
        [structural[m].loc[test.index] for m in COMBINATION_MEMBERS])

    table_a, losses_a = score(structural, test, horizon)
    table_a = attach_mcs(table_a, losses_a, args.alpha, args.n_boot, args.seed)
    table_a.insert(0, "horizon", horizon)

    marginal = {"har": structural["har"]}
    for feat in features:
        marginal[f"har_x__{feat}"] = walk_forward(
            frame, test_start, args.refit, make_harx_fp([feat]), purge=purge)
    table_b, losses_b = score(marginal, test, horizon)
    table_b = attach_mcs(table_b, losses_b, args.alpha, args.n_boot, args.seed)
    har_mean = float(table_b.loc[table_b["model"] == "har", "qlike_mean"].iloc[0])
    table_b["qlike_delta_vs_har"] = table_b["qlike_mean"] - har_mean
    table_b["block"] = [FEATURE_BLOCK.get(m.replace("har_x__", ""), "baseline")
                        for m in table_b["model"]]
    table_b.insert(0, "horizon", horizon)

    rich_base = list(RICH_BASE)
    rich = {"har_rs_iv": walk_forward(frame, test_start, args.refit,
                                      make_harx_fp(rich_base), purge=purge)}
    for feat in features:
        if feat in rich_base:
            continue
        rich[f"rich_x__{feat}"] = walk_forward(
            frame, test_start, args.refit, make_harx_fp(rich_base + [feat]),
            purge=purge)
    table_c, losses_c = score(rich, test, horizon, benchmark="har_rs_iv")
    table_c = attach_mcs(table_c, losses_c, args.alpha, args.n_boot, args.seed)
    rich_mean = float(table_c.loc[table_c["model"] == "har_rs_iv", "qlike_mean"].iloc[0])
    table_c["qlike_delta_vs_base"] = table_c["qlike_mean"] - rich_mean
    table_c["block"] = [FEATURE_BLOCK.get(m.replace("rich_x__", ""), "baseline")
                        for m in table_c["model"]]
    table_c = table_c.rename(columns={"dm_vs_har": "dm_vs_base",
                                      "p_vs_har": "p_vs_base"})
    table_c.insert(0, "horizon", horizon)

    # Two window lengths on purpose. Two years is the regime length the question
    # is really about, but with this out-of-sample span it only yields about a
    # dozen windows, all of them at the end of the sample. One year yields
    # enough windows to see membership move, at the cost of a noisier MCS in
    # each. Reporting one without the other would be choosing the answer.
    rolling_parts = []
    for window in args.mcs_window:
        block = rolling_model_confidence_set(
            losses_a, window=window, step=args.refit, alpha=args.alpha,
            n_boot=args.rolling_boot, seed=args.seed)
        if block.empty:
            continue
        block = block.reset_index()
        block.insert(0, "window", window)
        block.insert(0, "horizon", horizon)
        rolling_parts.append(block)
    rolling = (pd.concat(rolling_parts, ignore_index=True) if rolling_parts
               else pd.DataFrame())

    # Descriptive, in-sample, and labelled as such: the Patton-Sheppard
    # asymmetry is a claim about COEFFICIENTS, and it can hold clearly while the
    # model still fails to forecast better out of sample. Reporting only the
    # QLIKE column would hide half of what happened.
    shar_full = SHARRV().fit_shar(frame["rv"], frame["rs_pos_var"],
                                  frame["rs_neg_var"], frame["target"])
    har_full = HARRV().fit(frame["rv"], frame["target"])
    b0, b_pos, b_neg, b_w, b_m = shar_full.coef_
    diagnostics = pd.DataFrame([{
        "horizon": horizon,
        "shar_intercept": b0, "shar_b_pos": b_pos, "shar_b_neg": b_neg,
        "shar_b_weekly": b_w, "shar_b_monthly": b_m,
        "har_b_daily": har_full.coef_[1], "har_b_weekly": har_full.coef_[2],
        "har_b_monthly": har_full.coef_[3],
        "corr_semivols": float(frame["rs_pos_vol"].corr(frame["rs_neg_vol"])),
        "corr_pos_rv": float(frame["rs_pos_vol"].corr(frame["rv"])),
        "corr_neg_rv": float(frame["rs_neg_vol"].corr(frame["rv"])),
        "n_obs": len(frame),
    }])

    forecasts = pd.DataFrame(structural).loc[test.index]
    forecasts.insert(0, "target", test["target"])
    return {"table_a": table_a, "table_b": table_b, "table_c": table_c,
            "rolling": rolling, "forecasts": forecasts, "coverage": coverage,
            "diagnostics": diagnostics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intraday", type=Path, default=Path("data/SPY_intraday_rv.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21])
    parser.add_argument("--refit", type=int, default=21)
    parser.add_argument("--test-frac", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--rolling-boot", type=int, default=500)
    parser.add_argument("--mcs-window", type=int, nargs="+", default=[252, 504],
                        help="rolling MCS window lengths, in observations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", nargs="+", default=None,
                        help="restrict the panel to these features "
                             "(default: every feature in alt_data.ALL_FEATURES)")
    parser.add_argument("--extra-lag", type=int, default=0,
                        help="extra trading days of publication lag on every feature")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("" if not args.extra_lag else f"_lag{args.extra_lag}")
    a_parts, b_parts, c_parts, d_parts, r_parts = [], [], [], [], []
    for horizon in args.horizons:
        res = run_horizon(args, horizon)
        if horizon == args.horizons[0]:
            print(res["coverage"])
        show = ["model", "qlike_mean", "qlike_median", "collapsed", "dm_vs_har",
                "p_vs_har", "mcs_pvalue", "in_mcs"]
        print("\nA. structural models")
        print(res["table_a"].sort_values("qlike_mean")[show]
              .to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        members = res["table_a"].loc[res["table_a"]["in_mcs"], "model"].tolist()
        print(f"  MCS_{1 - args.alpha:.0%} = {{{', '.join(members)}}}")
        print("\nB. HAR plus one feature (marginal value)")
        print(res["table_b"].sort_values("qlike_mean")[
            ["model", "block", "qlike_mean", "qlike_delta_vs_har", "dm_vs_har",
             "p_vs_har", "in_mcs"]]
            .to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        print("\nC. HAR + semivariance + implied variance, plus one feature")
        print(res["table_c"].sort_values("qlike_mean")[
            ["model", "block", "qlike_mean", "qlike_delta_vs_base", "dm_vs_base",
             "p_vs_base", "in_mcs"]]
            .to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        print("\nSHAR coefficients, in sample, descriptive only "
              "(Patton-Sheppard predict b_neg well above b_pos)")
        print(res["diagnostics"][["shar_b_pos", "shar_b_neg", "har_b_daily",
                                  "corr_semivols"]]
              .to_string(index=False, float_format=lambda v: f"{v:0.4f}"))
        a_parts.append(res["table_a"])
        b_parts.append(res["table_b"])
        c_parts.append(res["table_c"])
        d_parts.append(res["diagnostics"])
        if not res["rolling"].empty:
            r_parts.append(res["rolling"])
        res["forecasts"].to_csv(args.out_dir / f"altdata_forecasts_h{horizon}{tag}.csv")

    pd.concat(a_parts, ignore_index=True).to_csv(
        args.out_dir / f"altdata_models{tag}.csv", index=False)
    pd.concat(b_parts, ignore_index=True).to_csv(
        args.out_dir / f"altdata_marginal{tag}.csv", index=False)
    pd.concat(c_parts, ignore_index=True).to_csv(
        args.out_dir / f"altdata_marginal_rich{tag}.csv", index=False)
    pd.concat(d_parts, ignore_index=True).to_csv(
        args.out_dir / f"altdata_shar_coefficients{tag}.csv", index=False)
    if r_parts:
        pd.concat(r_parts, ignore_index=True).to_csv(
            args.out_dir / f"altdata_rolling_mcs{tag}.csv", index=False)
    print(f"\nsaved -> {args.out_dir}/altdata_*{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
