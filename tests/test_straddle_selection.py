"""Regression tests for the 2026-10 straddle and data-licensing corrections.

1. Contract selection: the chosen expiry must outlive the 21-session hold.
   Picking the expiry nearest 30 calendar days chose one that often expired
   before the exit session, and those trades were rejected as "incomplete".
2. Admission accounting: signals too close to the end of the data are counted
   on their own, not as quote failures.
3. The always-long control is one book, not two identical ones.
4. Licensed option-market series and Reddit text are not committed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import alt_data  # noqa: E402
from option_strategies import (  # noqa: E402
    StraddleConfig,
    choose_straddle_contracts,
    straddle_backtest,
    straddle_candidates_from_chain,
)
from test_option_strategies import synthetic_chain  # noqa: E402

SESSIONS = pd.bdate_range("2024-01-01", periods=60)


def _candidates(rows):
    return pd.DataFrame(rows, columns=["date", "exdate", "strike", "spot"])


# ---------------------------------------------------------------------------
# contract selection
# ---------------------------------------------------------------------------


def test_chosen_expiry_outlives_the_hold():
    d = SESSIONS[0]
    exit_session = SESSIONS[21]
    early = d + pd.Timedelta(days=28)          # nearer 30 days, but before the exit
    assert early < exit_session
    late = exit_session + pd.Timedelta(days=4)  # 33 days out
    picks = choose_straddle_contracts(
        _candidates([(d, early, 100.0, 100.0), (d, late, 100.0, 100.0)]),
        SESSIONS, hold_days=21, target_days=30)
    assert picks["exdate"].tolist() == [late]
    assert (picks["exdate"] >= exit_session).all()


def test_nearest_target_expiry_wins_then_nearest_strike():
    d = SESSIONS[0]
    exit_session = SESSIONS[21]
    near = exit_session + pd.Timedelta(days=1)
    far = exit_session + pd.Timedelta(days=20)
    picks = choose_straddle_contracts(
        _candidates([(d, far, 100.0, 100.2), (d, near, 99.0, 100.2),
                     (d, near, 100.0, 100.2), (d, near, 101.0, 100.2)]),
        SESSIONS, hold_days=21, target_days=30)
    assert picks.iloc[0]["exdate"] == near
    assert picks.iloc[0]["strike"] == 100.0


def test_moneyness_tolerance_refuses_a_far_strike():
    d = SESSIONS[0]
    ex = SESSIONS[21] + pd.Timedelta(days=2)
    rows = [(d, ex, 104.0, 100.0)]
    assert len(choose_straddle_contracts(_candidates(rows), SESSIONS)) == 1
    assert choose_straddle_contracts(_candidates(rows), SESSIONS,
                                     max_moneyness=0.01).empty


def test_a_date_without_a_full_hold_gets_no_contract():
    d = SESSIONS[-5]
    ex = d + pd.Timedelta(days=40)
    assert choose_straddle_contracts(_candidates([(d, ex, 100.0, 100.0)]),
                                     SESSIONS, hold_days=21).empty


def test_cached_candidates_need_both_legs_with_a_live_bid():
    dates = SESSIONS[:2]
    chain = synthetic_chain([5.0, 5.0], [5.0, 5.0], spread=0.2, dates=dates)
    chain.loc[(chain.date == dates[1]) & (chain.cp_flag == "P"), "best_bid"] = 0.0
    out = straddle_candidates_from_chain(chain, pd.Series(100.0, index=dates))
    assert out["date"].tolist() == [dates[0]]
    assert out["spot"].tolist() == [100.0]


# ---------------------------------------------------------------------------
# admission accounting and the control books
# ---------------------------------------------------------------------------


def test_sample_end_signals_are_counted_separately_from_quote_failures():
    n = 30
    dates = pd.bdate_range("2020-01-01", periods=n)
    chain = synthetic_chain(np.full(n, 5.0), np.full(n, 5.0), dates=dates)
    entries = pd.DataFrame({"date": dates, "exdate": pd.Timestamp("2020-03-20"),
                            "strike": 100.0})
    signal = pd.Series(1, index=dates)
    config = StraddleConfig(hold_days=10, spread_fraction=0.0, delta_hedge=False)
    _, trades = straddle_backtest(chain, entries, pd.Series(100.0, index=dates),
                                  signal, config)
    assert trades.attrs["after_last_full_hold"] == 10
    assert trades.attrs["rejected_path"] == 0
    assert len(trades) == n - 10
    assert trades.attrs["attempted_signals"] == len(trades) + 10


def test_always_long_is_reported_once(monkeypatch):
    import run_option_pnl

    dates = pd.bdate_range("2020-01-01", periods=100)
    forecast = pd.Series(np.linspace(0.1, 0.3, 100), index=dates)
    implied = pd.Series(0.2, index=dates)

    def fake_backtest(chain, entries, underlying, side, config, long_only=False):
        book = pd.Series(0.001, index=underlying.index[50:])
        trades = pd.DataFrame({"trade_return": [0.01], "direction": [1]})
        return book, trades

    monkeypatch.setattr(run_option_pnl, "straddle_backtest", fake_backtest)
    under = pd.Series(100.0, index=pd.bdate_range(dates[0], periods=105))
    _, books, _ = run_option_pnl.run_straddles(
        pd.DataFrame({"har": forecast, "target": forecast}), implied,
        pd.DataFrame(), pd.DataFrame(), under, StraddleConfig(), 0.5)
    assert sorted(books) == ["always_long__both", "always_short__both",
                             "har__both", "har__long_only"]


def test_entries_are_reselected_from_the_cache_to_outlive_the_hold():
    import run_option_pnl

    sessions = pd.bdate_range("2024-01-01", periods=40)
    d = sessions[0]
    early = d + pd.Timedelta(days=28)
    late = sessions[21] + pd.Timedelta(days=4)
    assert early < sessions[21]
    rows = []
    for ex in (early, late):
        for cp in ("C", "P"):
            rows.append(dict(date=d, exdate=ex, strike=100.0, cp_flag=cp,
                             best_bid=1.0, best_offer=1.1))
    entries = pd.DataFrame({"date": [d], "exdate": [early], "strike": [100.0],
                            "spot": [100.0]})
    picked = run_option_pnl.select_straddle_entries(
        pd.DataFrame(rows), entries, sessions, hold_days=21, target_days=30,
        max_moneyness=0.01)
    assert picked["exdate"].tolist() == [late]


def test_option_chain_pull_selects_hold_safe_expiries():
    pytest.importorskip("psycopg2")
    import fetch_option_chain

    sessions = pd.bdate_range("2024-01-01", periods=40)
    d = sessions[0]
    early = (d + pd.Timedelta(days=28)).date()
    late = (sessions[21] + pd.Timedelta(days=4)).date()
    candidates = pd.DataFrame({
        "date": [s.date() for s in sessions] + [d.date()],
        "exdate": [early] * len(sessions) + [late],
        "strike_price": [100000] * (len(sessions) + 1),
        "close": [100.0] * (len(sessions) + 1),
    })
    picks = fetch_option_chain.select_entries(candidates, hold_days=21,
                                              target_days=30, lo=15, hi=60)
    first = picks[picks["date"] == d].iloc[0]
    assert first["exdate"] == pd.Timestamp(late)
    assert first["strike"] == 100.0


# ---------------------------------------------------------------------------
# licensed and third-party data
# ---------------------------------------------------------------------------


def test_committed_option_file_carries_no_vendor_series():
    header = pd.read_csv(ROOT / "data" / "features_option_market.csv", nrows=0).columns
    assert not set(alt_data.OPTION_VENDOR_COLUMNS) & set(header)


def test_vendor_series_are_merged_from_a_local_file(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=3)
    pd.DataFrame({"date": dates, "term_slope_30_91": [-0.01, -0.02, -0.03],
                  "skew_25d_30": [0.04, 0.05, 0.06],
                  "spy_put_call": [1.1, 1.2, 1.3]}).to_csv(
        tmp_path / "features_option_market.csv", index=False)
    with pytest.raises(FileNotFoundError):
        alt_data.load_option_block(tmp_path)
    vendor = tmp_path / "elsewhere.csv"
    pd.DataFrame({"date": dates, "atm_iv_30": [0.1, 0.2, 0.3],
                  "atm_iv_91": [0.12, 0.22, 0.32],
                  "atm_ivar_30": [0.01, 0.04, 0.09], "vix": [10.0, 20.0, 30.0],
                  "vxo": [np.nan] * 3}).to_csv(vendor, index=False)
    block = alt_data.load_option_block(tmp_path, vendor_csv=vendor)
    assert block["vix_vol"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert block["skew_25d_30"].tolist() == pytest.approx([0.04, 0.05, 0.06])


def test_wrds_pull_writes_vendor_series_to_the_local_file_only():
    pytest.importorskip("psycopg2")
    from fetch_wrds_features import split_option_market

    panel = pd.DataFrame(1.0, index=pd.bdate_range("2020-01-01", periods=2),
                         columns=["atm_iv_30", "atm_iv_91", "atm_ivar_30",
                                  "term_slope_30_91", "skew_25d_30",
                                  "spy_put_call", "vix", "vxo"])
    derived, vendor = split_option_market(panel)
    assert list(derived.columns) == ["term_slope_30_91", "skew_25d_30", "spy_put_call"]
    assert set(vendor.columns) == set(alt_data.OPTION_VENDOR_COLUMNS)


def test_reddit_crawl_keeps_scores_but_no_text_or_ids():
    data = json.loads((ROOT / "data" / "sentiment_prev_full_month.json").read_text())
    posts = [p for day in data["days"].values() for p in day["posts"]]
    assert posts
    banned = {"title", "title_key", "body", "reddit_id", "created_utc"}
    assert not banned & {k for p in posts for k in p}
    assert all("average_combined_sentiment" in day for day in data["days"].values())
