"""Tests for the alternative-data feature panel.

Two properties matter more than any other and are tested directly rather than
inspected: a feature stamped on date t must not move when FUTURE source values
change, and the publication lag must be counted in trading sessions rather than
calendar days. Everything else in this file supports those two.

No test here touches the network or a credential; the panel is built from
synthetic CSVs written into a temporary directory with the same schemas the
fetch scripts produce.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alt_data import (  # noqa: E402
    ALL_FEATURES,
    abnormal_attention,
    align_to_sessions,
    build_feature_panel,
    causal_zscore,
    load_calendar_block,
)
from fetch_alt_data import _xlsx_rows  # noqa: E402


SESSIONS = pd.bdate_range("2020-01-01", periods=260)


def _write_panel_inputs(tmp_path: Path, sessions: pd.DatetimeIndex,
                        seed: int = 0) -> Path:
    """Synthetic versions of every committed feature file, on the real schemas."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(sessions[0]) - pd.DateOffset(days=400)
    calendar_days = pd.date_range(start, pd.Timestamp(sessions[-1]))
    n_cal = len(calendar_days)

    pd.DataFrame({
        "date": sessions,
        "atm_iv_30": 0.15 + 0.01 * rng.normal(size=len(sessions)),
        "atm_iv_91": 0.16 + 0.01 * rng.normal(size=len(sessions)),
        "atm_ivar_30": 0.15 ** 2 + 0.001 * rng.normal(size=len(sessions)),
        "term_slope_30_91": -0.01 + 0.002 * rng.normal(size=len(sessions)),
        "skew_25d_30": 0.03 + 0.005 * rng.normal(size=len(sessions)),
        "spy_put_call": 1.2 + 0.1 * rng.normal(size=len(sessions)),
        "vix": 15.0 + rng.normal(size=len(sessions)),
        "vxo": np.nan,
    }).to_csv(tmp_path / "features_option_market.csv", index=False)

    pd.DataFrame({
        "date": calendar_days,
        "gdelt_tone_mkt": rng.normal(size=n_cal),
        "gdelt_share_mkt": 0.02 + 0.002 * rng.random(size=n_cal),
        "gdelt_count_mkt": rng.integers(500, 5000, size=n_cal),
        "gdelt_tone_econ": rng.normal(size=n_cal),
        "gdelt_share_econ": 0.03 + 0.002 * rng.random(size=n_cal),
        "gdelt_count_econ": rng.integers(500, 5000, size=n_cal),
    }).to_csv(tmp_path / "features_gdelt.csv", index=False)

    pd.DataFrame({
        "date": calendar_days,
        "wiki_sp500": rng.integers(2000, 6000, size=n_cal),
        "wiki_crash": rng.integers(200, 900, size=n_cal),
        "wiki_recession": rng.integers(500, 2000, size=n_cal),
        "wiki_vix": rng.integers(300, 1200, size=n_cal),
    }).to_csv(tmp_path / "features_wikipedia.csv", index=False)

    months = pd.date_range(calendar_days[0].replace(day=1), calendar_days[-1], freq="MS")
    unc = pd.DataFrame({"date": calendar_days,
                        "epu_daily": 100 + 20 * rng.random(size=n_cal),
                        "emv_overall": np.nan})
    monthly = pd.DataFrame({"date": months,
                            "epu_daily": np.nan,
                            "emv_overall": 20 + rng.random(size=len(months))})
    pd.concat([unc, monthly]).sort_values("date").to_csv(
        tmp_path / "features_uncertainty.csv", index=False)

    events = []
    for i, d in enumerate(sessions[::40]):
        events.append({"date": d, "event": ["fomc", "cpi", "payrolls"][i % 3]})
    pd.DataFrame(events).to_csv(tmp_path / "calendar_events.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# alignment primitives
# ---------------------------------------------------------------------------


def test_lag_is_counted_in_sessions_not_calendar_days():
    # a Friday value with lag 1 must appear on the following MONDAY session,
    # which is three calendar days later
    friday = pd.Timestamp("2020-01-03")
    monday = pd.Timestamp("2020-01-06")
    source = pd.DataFrame({"x": [1.0, 2.0]}, index=[friday, monday])
    out = align_to_sessions(source, SESSIONS, lag=1)
    assert out.loc[monday, "x"] == 1.0
    assert np.isnan(out.loc[friday, "x"])


def test_zero_lag_leaves_the_stamp_where_it_was():
    friday = pd.Timestamp("2020-01-03")
    source = pd.DataFrame({"x": [7.0]}, index=[friday])
    out = align_to_sessions(source, SESSIONS, lag=0)
    assert out.loc[friday, "x"] == 7.0


def test_weekend_values_are_carried_into_the_next_session():
    saturday = pd.Timestamp("2020-01-04")
    monday = pd.Timestamp("2020-01-06")
    source = pd.DataFrame({"x": [5.0]}, index=[saturday])
    out = align_to_sessions(source, SESSIONS, lag=0)
    assert out.loc[monday, "x"] == 5.0


def test_forward_fill_limit_stops_a_stale_value_running_forever():
    source = pd.DataFrame({"x": [1.0]}, index=[pd.Timestamp("2020-01-02")])
    out = align_to_sessions(source, SESSIONS, lag=0, ffill_limit=2)
    assert out["x"].notna().sum() == 3        # the stamp plus two filled sessions


def test_causal_zscore_excludes_the_current_observation():
    s = pd.Series(np.arange(1.0, 401.0), index=pd.bdate_range("2019-01-01", periods=400))
    z = causal_zscore(s, window=100, min_periods=50)
    # changing only the LAST value cannot change any earlier z-score
    bumped = s.copy()
    bumped.iloc[-1] = 1e6
    z2 = causal_zscore(bumped, window=100, min_periods=50)
    pd.testing.assert_series_equal(z.iloc[:-1], z2.iloc[:-1])
    # and the last z-score must be enormous, since its own value is not in the scale
    assert z2.iloc[-1] > z.iloc[-1]


def test_abnormal_attention_uses_only_past_views():
    views = pd.Series(1000.0, index=pd.bdate_range("2020-01-01", periods=200))
    base = abnormal_attention(views, window=60)
    spiked = views.copy()
    spiked.iloc[150] = 10_000.0
    bumped = abnormal_attention(spiked, window=60)
    pd.testing.assert_series_equal(base.iloc[:150], bumped.iloc[:150])
    assert bumped.iloc[150] > 2.0             # log(10) on a flat baseline


def test_calendar_dummy_moves_to_the_next_session_when_the_date_is_a_holiday(tmp_path):
    sunday = pd.Timestamp("2020-01-05")
    monday = pd.Timestamp("2020-01-06")
    pd.DataFrame({"date": [sunday], "event": ["fomc"]}).to_csv(
        tmp_path / "calendar_events.csv", index=False)
    out = load_calendar_block(SESSIONS, tmp_path)
    assert out.loc[monday, "is_fomc"] == 1.0
    assert out["is_fomc"].sum() == 1.0


# ---------------------------------------------------------------------------
# the panel as a whole
# ---------------------------------------------------------------------------


def test_panel_builds_every_declared_feature(tmp_path):
    _write_panel_inputs(tmp_path, SESSIONS)
    panel, report = build_feature_panel(SESSIONS, data_dir=tmp_path)
    assert set(ALL_FEATURES).issubset(panel.columns)
    assert report.n_sessions == len(SESSIONS)
    assert report.first_complete is not None


@pytest.mark.parametrize("source_file,column", [
    ("features_option_market.csv", "atm_iv_30"),
    ("features_gdelt.csv", "gdelt_tone_mkt"),
    ("features_wikipedia.csv", "wiki_sp500"),
    ("features_uncertainty.csv", "epu_daily"),
])
def test_no_feature_moves_when_the_future_of_its_source_changes(tmp_path, source_file,
                                                                column):
    """The lookahead test, run once per source file.

    Multiply every source observation after a cut date by 10 and rebuild. If any
    feature value on or before the cut changes, that feature is reading its own
    future. This catches the mistakes that a coverage check never would:
    a backward fill, a centred rolling window, a full-sample standardisation.
    """
    _write_panel_inputs(tmp_path, SESSIONS)
    cut = SESSIONS[180]
    before, _ = build_feature_panel(SESSIONS, data_dir=tmp_path)

    path = tmp_path / source_file
    raw = pd.read_csv(path, parse_dates=["date"])
    future = raw["date"] > cut
    raw.loc[future, column] = raw.loc[future, column] * 10.0
    raw.to_csv(path, index=False)

    after, _ = build_feature_panel(SESSIONS, data_dir=tmp_path)
    head_before = before.loc[before.index <= cut, ALL_FEATURES]
    head_after = after.loc[after.index <= cut, ALL_FEATURES]
    pd.testing.assert_frame_equal(head_before, head_after)


def test_extra_lag_shifts_every_block_by_one_more_session(tmp_path):
    _write_panel_inputs(tmp_path, SESSIONS)
    base, _ = build_feature_panel(SESSIONS, data_dir=tmp_path)
    lagged, _ = build_feature_panel(SESSIONS, data_dir=tmp_path, extra_lag=1)
    for col in ALL_FEATURES:
        shifted = base[col].shift(1)
        both = pd.concat([shifted.rename("a"), lagged[col].rename("b")], axis=1).dropna()
        assert len(both) > 100, col
        assert np.allclose(both["a"], both["b"]), col


def test_unknown_feature_name_is_an_error_not_a_silent_drop(tmp_path):
    _write_panel_inputs(tmp_path, SESSIONS)
    with pytest.raises(KeyError):
        build_feature_panel(SESSIONS, data_dir=tmp_path, features=["not_a_feature"])


# ---------------------------------------------------------------------------
# the standard-library xlsx reader used for the EMV workbook
# ---------------------------------------------------------------------------


def _minimal_xlsx(rows) -> bytes:
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    shared = sorted({v for row in rows for v in row if isinstance(v, str)})
    body = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row):
            ref = f"{chr(ord('A') + c)}{r}"
            if isinstance(value, str):
                cells.append(f'<c r="{ref}" t="s"><v>{shared.index(value)}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
        body.append(f'<row r="{r}">{"".join(cells)}</row>')
    sheet = f'<worksheet {ns}><sheetData>{"".join(body)}</sheetData></worksheet>'
    sst = "".join(f"<si><t>{v}</t></si>" for v in shared)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml",
                   f'<sst {ns} count="{len(shared)}">{sst}</sst>')
    return buf.getvalue()


def test_xlsx_reader_returns_strings_and_numbers_in_place():
    rows = [["Year", "Month", "Overall EMV Tracker"], [1985, 1, 11.3], [1985, 2, 9.46]]
    parsed = _xlsx_rows(_minimal_xlsx(rows))
    assert parsed[0] == ["Year", "Month", "Overall EMV Tracker"]
    assert [float(v) for v in parsed[1]] == [1985.0, 1.0, 11.3]
    assert len(parsed) == 3
