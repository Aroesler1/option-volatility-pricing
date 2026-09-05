"""Tests for the public underlying series.

The option P&L takes two things from this file: the price the delta hedge trades
at, and a total return. Getting either subtly wrong is invisible in the P&L and
fatal to it, so both are pinned here, along with the column extraction, because
yfinance changes the shape of what it returns between releases.

No test reaches the network; a stub module stands in for yfinance.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch_spy_daily import _column, fetch  # noqa: E402

DATES = pd.date_range("2020-01-02", periods=4, freq="B")
CLOSE = [100.0, 101.0, 99.0, 102.0]
# a 1.00 dividend goes ex on the third day, so the adjusted series is not a
# constant multiple of the close
ADJUSTED = [99.0, 99.99, 99.0, 102.0]


def _flat_frame():
    return pd.DataFrame({"Adj Close": ADJUSTED, "Close": CLOSE,
                         "Open": CLOSE, "High": CLOSE, "Low": CLOSE,
                         "Volume": [1] * 4}, index=DATES)


def _multiindex_frame():
    frame = _flat_frame()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]],
                                               names=["Price", "Ticker"])
    return frame


@pytest.fixture
def stub_yfinance(monkeypatch):
    def install(frame):
        module = types.ModuleType("yfinance")
        module.download = lambda *args, **kwargs: frame
        monkeypatch.setitem(sys.modules, "yfinance", module)
    return install


@pytest.mark.parametrize("builder", [_flat_frame, _multiindex_frame])
def test_columns_are_found_in_either_shape_yfinance_returns(builder):
    frame = builder()
    assert list(_column(frame, "Close", "SPY")) == CLOSE
    assert list(_column(frame, "Adj Close", "SPY")) == ADJUSTED


@pytest.mark.parametrize("builder", [_flat_frame, _multiindex_frame])
def test_close_is_the_unadjusted_price_the_hedge_would_trade_at(stub_yfinance, builder):
    stub_yfinance(builder())
    out = fetch("SPY", "2020-01-01")
    assert list(out["close"]) == CLOSE


@pytest.mark.parametrize("builder", [_flat_frame, _multiindex_frame])
def test_return_is_a_total_return_not_a_price_return(stub_yfinance, builder):
    """On the ex-dividend day the price falls and the total return does not."""
    stub_yfinance(builder())
    out = fetch("SPY", "2020-01-01")
    price_return = pd.Series(CLOSE).pct_change()
    assert out["ret"].iloc[2] == pytest.approx(99.0 / 99.99 - 1.0)
    assert out["ret"].iloc[2] > price_return.iloc[2]
    assert np.isnan(out["ret"].iloc[0])


def test_the_index_is_tz_naive_so_it_joins_the_rest_of_the_panel(stub_yfinance):
    frame = _flat_frame()
    frame.index = frame.index.tz_localize("America/New_York")
    stub_yfinance(frame)
    out = fetch("SPY", "2020-01-01")
    assert out.index.tz is None
    assert out.index.name == "date"


def test_an_empty_download_fails_loudly_rather_than_writing_an_empty_file(stub_yfinance):
    stub_yfinance(pd.DataFrame())
    with pytest.raises(SystemExit, match="no rows"):
        fetch("SPY", "2020-01-01")


def test_rows_without_a_close_are_dropped(stub_yfinance):
    frame = _flat_frame()
    frame.loc[DATES[1], "Close"] = np.nan
    stub_yfinance(frame)
    out = fetch("SPY", "2020-01-01")
    assert len(out) == 3
    assert DATES[1] not in out.index
