"""Alternative-data feature panel: alignment, publication lags, transformations.

Every feature in this module is a daily series stamped on a SPY trading date and
built so that the value at date t uses only information available by the close of
t. The target it is paired with is realized variance over (t, t+h], so nothing
here can see its own target.

THE ONE CONVENTION THAT MATTERS. "Available by the close of t" is the information
set, not "published strictly before the close". Realized variance RV_t is itself
computed from date t's own session, and the HAR literature forecasts RV over
(t, t+h] from RV_t, so an option-market quote struck at the same close is on
exactly the same footing. Features whose daily aggregation window runs past the
New York close are a different case and are lagged one trading day:

    option market   lag 0   OptionMetrics and CBOE closing values for date t
    news (GDELT)    lag 1   GDELT days are UTC days, which run to 20:00 New York
    attention       lag 1   Wikimedia pageview days are UTC days
    uncertainty     lag 1   the EPU daily index counts a whole newspaper day
    EMV             month   monthly series, applied from the month after the one
                            it measures, which is the earliest it can be known
    calendar        lag 0   release dates are published years in advance

`build_feature_panel(..., extra_lag=1)` shifts every one of these by a further
day. If a result only survives at extra_lag=0 it is an artifact of the
convention, and the benchmark reports both.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

# Feature blocks, their source file, and the publication lag in trading days.
# `None` for lag means the block has its own alignment rule (EMV, calendar).
OPTION_FEATURES = ["atm_ivar_30", "atm_iv_30", "skew_25d_30", "term_slope_30_91",
                   "vix_vol", "spy_put_call"]
NEWS_FEATURES = ["gdelt_tone_mkt", "gdelt_share_mkt", "gdelt_tone_econ",
                 "gdelt_share_econ"]
ATTENTION_FEATURES = ["wiki_attention"]
UNCERTAINTY_FEATURES = ["epu_log", "emv_overall"]
CALENDAR_FEATURES = ["is_fomc", "is_cpi", "is_payrolls"]

ALL_FEATURES = (OPTION_FEATURES + NEWS_FEATURES + ATTENTION_FEATURES
                + UNCERTAINTY_FEATURES + CALENDAR_FEATURES)

FEATURE_BLOCK = (
    {f: "option" for f in OPTION_FEATURES}
    | {f: "news" for f in NEWS_FEATURES}
    | {f: "attention" for f in ATTENTION_FEATURES}
    | {f: "uncertainty" for f in UNCERTAINTY_FEATURES}
    | {f: "calendar" for f in CALENDAR_FEATURES}
)


def align_to_sessions(frame: pd.DataFrame, sessions: pd.DatetimeIndex,
                      lag: int = 1, ffill_limit: Optional[int] = 5) -> pd.DataFrame:
    """Put a daily series on the trading calendar and apply a publication lag.

    Three steps, in this order, because the order changes the answer:

    1. Reindex onto the union of the source dates and the trading sessions, then
       forward-fill at most `ffill_limit` days. Filling first means a Friday
       value carries into a Monday session, which is right for a series that is
       published on non-trading days (GDELT, Wikipedia and EPU all are).
    2. Restrict to trading sessions.
    3. Shift by `lag` SESSIONS, not calendar days. Shifting after the restriction
       is what makes the lag mean "one trading day of information", which is the
       unit the forecasts are made in.

    A `ffill_limit` of None fills without limit; use it only for series that are
    genuinely step functions (the monthly EMV tracker).
    """
    if lag < 0:
        raise ValueError("lag must be non-negative")
    idx = frame.index.union(sessions)
    out = frame.reindex(idx).ffill(limit=ffill_limit).reindex(sessions)
    return out.shift(lag) if lag else out


def causal_zscore(series: pd.Series, window: int = 250, min_periods: int = 60) -> pd.Series:
    """Standardise against a TRAILING window that excludes the current value.

    Scaling by full-sample statistics is the classic silent leak in an
    attention index: the standard deviation of 2020 would set the scale of 2018.
    The shift(1) before the rolling window is what keeps day t out of its own
    mean and standard deviation.
    """
    s = pd.to_numeric(series, errors="coerce")
    past = s.shift(1)
    mu = past.rolling(window, min_periods=min_periods).mean()
    sd = past.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def abnormal_attention(views: pd.Series, window: int = 60) -> pd.Series:
    """Abnormal attention: log views minus the median of the trailing `window`.

    The median-of-past-log-views baseline is the abnormal-search-volume
    definition of Da, Engelberg and Gao (2011), used here on Wikipedia
    pageviews because Google Trends is not reproducible across pulls (its
    samples differ between requests) and was dropped from this study by
    decision, not by accident.
    """
    v = pd.to_numeric(views, errors="coerce")
    logv = np.log(v.where(v > 0))
    baseline = logv.shift(1).rolling(window, min_periods=window // 2).median()
    return logv - baseline


# ---------------------------------------------------------------------------
# Block loaders. Each returns a frame indexed by its own source dates; the
# alignment and lag are applied once, in build_feature_panel.
# ---------------------------------------------------------------------------


def load_option_block(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / "features_option_market.csv",
                      parse_dates=["date"]).set_index("date").sort_index()
    out = pd.DataFrame(index=raw.index)
    out["atm_iv_30"] = raw["atm_iv_30"]
    out["atm_ivar_30"] = raw["atm_ivar_30"]
    out["skew_25d_30"] = raw["skew_25d_30"]
    out["term_slope_30_91"] = raw["term_slope_30_91"]
    # VIX is quoted in percentage points; the rest of this repo works in
    # decimal annualised volatility, and mixing the two is how a regression
    # coefficient ends up off by a factor of 100
    out["vix_vol"] = raw["vix"] / 100.0
    out["spy_put_call"] = raw["spy_put_call"]
    return out


def load_news_block(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / "features_gdelt.csv",
                      parse_dates=["date"]).set_index("date").sort_index()
    out = pd.DataFrame(index=raw.index)
    out["gdelt_tone_mkt"] = raw["gdelt_tone_mkt"]
    out["gdelt_tone_econ"] = raw["gdelt_tone_econ"]
    # share of all monitored coverage, logged: the raw article count tracks the
    # tenfold growth of GDELT's crawl over the sample, not the news
    for label in ("mkt", "econ"):
        share = pd.to_numeric(raw[f"gdelt_share_{label}"], errors="coerce")
        out[f"gdelt_share_{label}"] = np.log(share.where(share > 0))
    return out


def load_attention_block(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / "features_wikipedia.csv",
                      parse_dates=["date"]).set_index("date").sort_index()
    articles = [c for c in raw.columns if c.startswith("wiki_")]
    abnormal = pd.DataFrame({c: abnormal_attention(raw[c]) for c in articles})
    out = abnormal.rename(columns=lambda c: c.replace("wiki_", "wiki_abn_"))
    # FEARS-style aggregation: standardise each article's abnormal attention on
    # its own trailing window, then equal-weight. Equal weights are used rather
    # than fitted ones because fitted weights on four series over one sample is
    # a selection exercise dressed as an index.
    z = pd.DataFrame({c: causal_zscore(abnormal[c]) for c in articles})
    out["wiki_attention"] = z.mean(axis=1, skipna=True)
    return out


def load_uncertainty_block(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (daily block, monthly EMV block); they need different alignment."""
    raw = pd.read_csv(data_dir / "features_uncertainty.csv",
                      parse_dates=["date"]).set_index("date").sort_index()
    epu = pd.to_numeric(raw["epu_daily"], errors="coerce")
    daily = pd.DataFrame({"epu_log": np.log(epu.where(epu > 0))}).dropna()
    monthly = raw[["emv_overall"]].dropna()
    return daily, monthly


def load_calendar_block(sessions: pd.DatetimeIndex,
                        data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Announcement-day dummies, aligned to the next session if a date is a holiday.

    A release that lands on a non-trading day (the 15 March 2020 emergency FOMC
    cut was a Sunday) still hits the market, at the next open, so the dummy is
    moved forward rather than dropped.
    """
    events = pd.read_csv(data_dir / "calendar_events.csv", parse_dates=["date"])
    out = pd.DataFrame(0.0, index=sessions,
                       columns=["is_fomc", "is_cpi", "is_payrolls"])
    positions = sessions.searchsorted(events["date"].to_numpy(), side="left")
    for pos, event in zip(positions, events["event"]):
        if pos >= len(sessions):
            continue
        out.iloc[pos, out.columns.get_loc(f"is_{event}")] = 1.0
    return out


@dataclass
class PanelReport:
    """What the panel builder saw, so coverage is reported rather than assumed."""

    n_sessions: int
    coverage: pd.Series
    first_complete: Optional[pd.Timestamp]

    def to_string(self) -> str:
        lines = [f"sessions: {self.n_sessions:,}",
                 f"first date with every feature present: "
                 f"{self.first_complete.date() if self.first_complete is not None else 'never'}",
                 "coverage:"]
        lines += [f"  {k:<20s} {v:6.3f}" for k, v in self.coverage.items()]
        return "\n".join(lines)


def build_feature_panel(sessions: pd.DatetimeIndex, data_dir: Path = DATA_DIR,
                        extra_lag: int = 0,
                        features: Optional[Iterable[str]] = None
                        ) -> tuple[pd.DataFrame, PanelReport]:
    """Assemble every alternative-data feature on the SPY trading calendar.

    `extra_lag` adds the same number of trading days to every block's lag. It
    exists so the headline result can be re-run one day more conservatively
    without editing anything.
    """
    sessions = pd.DatetimeIndex(sessions).sort_values()
    option = align_to_sessions(load_option_block(data_dir), sessions, lag=0 + extra_lag)
    news = align_to_sessions(load_news_block(data_dir), sessions, lag=1 + extra_lag)
    attention = align_to_sessions(load_attention_block(data_dir), sessions,
                                  lag=1 + extra_lag)
    unc_daily, unc_monthly = load_uncertainty_block(data_dir)
    epu = align_to_sessions(unc_daily, sessions, lag=1 + extra_lag)
    # EMV measures a whole calendar month, so the earliest date it can be known
    # is the first session of the following month. Stamping it on the first of
    # the month it measures and lagging by one month does exactly that; the
    # forward fill is unlimited because a monthly series IS a step function.
    emv = unc_monthly.copy()
    emv.index = emv.index + pd.offsets.MonthBegin(1)
    emv = align_to_sessions(emv, sessions, lag=0 + extra_lag, ffill_limit=None)
    calendar = load_calendar_block(sessions, data_dir)
    if extra_lag:
        calendar = calendar.shift(extra_lag)

    panel = pd.concat([option, news, attention, epu, emv, calendar], axis=1)
    wanted = list(features) if features is not None else ALL_FEATURES
    missing = [c for c in wanted if c not in panel.columns]
    if missing:
        raise KeyError(f"features not built: {missing}")
    extras = [c for c in panel.columns if c.startswith("wiki_abn_")]
    panel = panel[wanted + [c for c in extras if c not in wanted]]

    coverage = panel[wanted].notna().mean()
    complete = panel[wanted].dropna()
    report = PanelReport(n_sessions=len(sessions), coverage=coverage,
                         first_complete=complete.index.min() if len(complete) else None)
    return panel, report
