#!/usr/bin/env python3
"""Pull the free alternative-data blocks of the panel and write derived daily CSVs.

Three sources, none of which needs a credential:

news tone (GDELT 2.0 DOC API)
    `mode=timelinetone` returns the average GKG tone of matching coverage per
    day; `mode=timelinevolraw` returns both the matching article count and the
    total number of articles GDELT monitored that day. The count is only
    meaningful as a SHARE of total coverage: the size of GDELT's crawl moved by
    a factor of about three across this sample (downward, from roughly 520,000
    monitored articles a day in early 2017 to 163,000 in late 2025), so the raw
    count encodes the date more than it encodes the news.

    Two theme filters, both GKG themes, stated exactly so the query is
    reproducible:
        market  : theme:ECON_STOCKMARKET sourcelang:eng
        economy : theme:EPU_ECONOMY      sourcelang:eng
    ECON_STOCKMARKET is the GKG stock-market theme. EPU_ECONOMY is the economy
    theme from the Baker-Bloom-Davis policy-uncertainty theme set that GDELT
    ships in the GKG taxonomy, which is the closest free analogue to the
    newspaper-based economy filter that literature uses.

    GDELT's stated limit is one request per 5 seconds, but in practice it
    answers an UNCACHED timeline query about once a minute and returns HTTP 429
    with the rate-limit text for everything in between, so the pause between
    requests is 80 seconds and every chunk is cached to disk. A full pull is
    about 40 requests and takes roughly an hour; rerunning it costs nothing
    because the cache is checked first. Ranges much longer than a year are
    refused however long you wait, which is why the pull is chunked by year.

attention (Wikimedia pageviews REST API)
    Daily views for four articles, `agent=user` so bots and spiders are
    excluded. Aggregated FEARS-style in `alt_data.py`, not here: this script
    only stores the raw daily counts.

uncertainty (policyuncertainty.com)
    The daily news-based Economic Policy Uncertainty index (Baker, Bloom and
    Davis), and the Equity Market Volatility tracker of Baker, Bloom, Davis and
    Kost (2019). EMV is published MONTHLY, not daily; it is stored at monthly
    frequency here and lagged into daily alignment in `alt_data.py`, where the
    lag is visible rather than buried in a fetch script.

    The EMV workbook is .xlsx. It is parsed with the standard library rather
    than by adding an Excel dependency: an .xlsx is a zip of XML, and the sheet
    is a plain grid of numbers.

Not available, recorded rather than quietly dropped:
    AAII weekly bull-bear spread. aaii.com returns HTTP 403 to programmatic
    requests, so the series cannot be fetched reproducibly and is excluded.
    CBOE's daily put/call ratio file is likewise 403; `fetch_wrds_features.py`
    builds a SPY put/call VOLUME ratio from OptionMetrics instead, which is a
    narrower measure and is labelled as such.

Usage:
    python fetch_alt_data.py                  # all three
    python fetch_alt_data.py --source gdelt
"""
from __future__ import annotations

import argparse
import io
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

USER_AGENT = "option-volatility-pricing/0.1 (academic research; volatility forecasting)"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_SLEEP = 80.0         # the real limit, not the stated one; see the docstring
GDELT_CACHE = Path("data/.gdelt_cache")
WIKI_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
EMV_URL = "https://www.policyuncertainty.com/media/EMV_Data.xlsx"

GDELT_THEMES = {
    "mkt": "theme:ECON_STOCKMARKET sourcelang:eng",
    "econ": "theme:EPU_ECONOMY sourcelang:eng",
}
WIKI_ARTICLES = {
    "sp500": "S&P_500",
    "crash": "Stock_market_crash",
    "recession": "Recession",
    "vix": "VIX",
}


def _get(url: str, retries: int = 14, retry_wait: float = 150.0,
         **kwargs) -> requests.Response:
    """GET that keeps trying through a throttle or a dropped connection.

    Two failure modes are retried, because over an hours-long pull both happen.
    GDELT answers an uncached timeline query at best once a minute and returns
    HTTP 429 with a plain-text scolding otherwise; and a laptop that sleeps or
    changes network drops DNS, which arrives as a ConnectionError and used to
    kill the whole run several chunks in. Neither is a reason to stop, and
    everything already fetched is on disk, so the retry is a flat wait rather
    than an exponential one: the throttle is a fixed cooldown, and doubling just
    waits hours for a limit that cleared long ago.
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120,
                             **kwargs)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(retry_wait)
            continue
        if r.status_code != 429:
            r.raise_for_status()
            return r
        if attempt == retries - 1:
            break
        time.sleep(retry_wait)
    raise RuntimeError(f"rate limited after {retries} attempts: {url}")


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------


def _gdelt_timeline(query: str, mode: str, start: str, end: str,
                    cache_dir: Path = GDELT_CACHE) -> tuple[pd.DataFrame, bool]:
    """One timeline chunk, served from disk when it has been fetched before.

    Returns (frame, fetched). A cache hit must not trigger the 80-second pause,
    so the caller needs to know which it got.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in query)[:60]
    path = cache_dir / f"{mode}_{slug}_{start}_{end}.csv"
    if path.exists():
        return pd.read_csv(path), False
    params = {"query": query, "mode": mode, "format": "csv",
              "startdatetime": f"{start}000000", "enddatetime": f"{end}000000"}
    # pause BEFORE the request, not after a successful one: the cooldown has to
    # have elapsed by the time the request is made, and the first request of a
    # run is the one most likely to arrive too soon after whatever came before
    time.sleep(GDELT_SLEEP)
    text = _get(GDELT_URL, params=params).text
    if not text.lstrip("\ufeff").startswith("Date"):
        raise RuntimeError(f"GDELT returned a non-CSV body for {mode}: {text[:200]}")
    frame = pd.read_csv(io.StringIO(text))
    frame.to_csv(path, index=False)
    return frame, True


def fetch_gdelt(start: str, end: str) -> pd.DataFrame:
    """Daily tone and coverage share per theme, pulled one calendar year at a time."""
    years = range(int(start[:4]), int(end[:4]) + 1)
    out: dict[str, pd.Series] = {}
    for label, query in GDELT_THEMES.items():
        tone_parts, vol_parts = [], []
        for year in years:
            lo = max(f"{year}0101", start.replace("-", ""))
            hi = min(f"{year + 1}0101", end.replace("-", ""))
            if lo >= hi:
                continue
            for mode, parts in (("timelinetone", tone_parts),
                                ("timelinevolraw", vol_parts)):
                chunk, fetched = _gdelt_timeline(query, mode, lo, hi)
                parts.append(chunk)
                print(f"  {label} {mode} {lo[:4]}: {len(chunk)} rows"
                      f"{'' if fetched else ' (cached)'}", flush=True)
        tone = pd.concat(tone_parts, ignore_index=True)
        vol = pd.concat(vol_parts, ignore_index=True)
        tone_s = (tone.set_index(pd.to_datetime(tone["Date"]))["Value"]
                  .groupby(level=0).mean())
        wide = vol.pivot_table(index=pd.to_datetime(vol["Date"]), columns="Series",
                               values="Value", aggfunc="sum")
        share = wide["Article Count"] / wide["Total Monitored Articles"].replace(0, np.nan)
        out[f"gdelt_tone_{label}"] = tone_s
        out[f"gdelt_share_{label}"] = share
        out[f"gdelt_count_{label}"] = wide["Article Count"]
    frame = pd.DataFrame(out).sort_index()
    frame.index.name = "date"
    return frame


def fetch_wikipedia(start: str, end: str) -> pd.DataFrame:
    """Daily human pageviews for the four attention articles."""
    lo, hi = start.replace("-", ""), end.replace("-", "")
    cols = {}
    for label, article in WIKI_ARTICLES.items():
        title = requests.utils.quote(article, safe="")
        payload = _get(WIKI_URL.format(article=title, start=lo, end=hi)).json()
        items = payload["items"]
        idx = pd.to_datetime([it["timestamp"][:8] for it in items])
        cols[f"wiki_{label}"] = pd.Series([it["views"] for it in items], index=idx)
    frame = pd.DataFrame(cols).sort_index()
    frame.index.name = "date"
    return frame


# ---------------------------------------------------------------------------
# Policy uncertainty
# ---------------------------------------------------------------------------


def _xlsx_rows(content: bytes) -> list[list[str | None]]:
    """Read the first worksheet of an .xlsx as a list of row value lists.

    An .xlsx is a zip of XML parts. Cell values live in `xl/worksheets/sheet1.xml`
    as <c><v>..</v></c>, with string cells (t="s") holding an index into
    `xl/sharedStrings.xml`. That is the whole format needed here, so the standard
    library is enough and no Excel dependency is added for one file.
    """
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(ns + "t"))
                      for si in root.findall(ns + "si")]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(ns + "row"):
        values: list[str | None] = []
        for cell in row.findall(ns + "c"):
            v = cell.find(ns + "v")
            if v is None:
                values.append(None)
            elif cell.get("t") == "s":
                values.append(shared[int(v.text)])
            else:
                values.append(v.text)
        rows.append(values)
    return rows


def fetch_epu_daily() -> pd.Series:
    raw = pd.read_csv(io.StringIO(_get(EPU_URL).text))
    raw = raw.dropna(subset=["year", "month", "day"])
    idx = pd.to_datetime(dict(year=raw["year"].astype(int),
                              month=raw["month"].astype(int),
                              day=raw["day"].astype(int)), errors="coerce")
    s = pd.Series(pd.to_numeric(raw["daily_policy_index"], errors="coerce").to_numpy(),
                  index=idx, name="epu_daily").dropna()
    return s[~s.index.isna()].sort_index()


def fetch_emv_monthly() -> pd.DataFrame:
    rows = _xlsx_rows(_get(EMV_URL).content)
    header = rows[0]
    want = {"Overall EMV Tracker": "emv_overall",
            "Policy-Related EMV Tracker": "emv_policy"}
    keep = {header.index(k): v for k, v in want.items() if k in header}
    if len(keep) != len(want):
        raise RuntimeError(f"EMV workbook columns changed: {header[:6]}")
    recs = []
    for row in rows[1:]:
        if len(row) < 3 or row[0] is None or not str(row[0]).strip().isdigit():
            continue        # the file ends with a free-text attribution line
        rec = {"date": pd.Timestamp(int(row[0]), int(float(row[1])), 1)}
        for i, name in keep.items():
            rec[name] = float(row[i]) if row[i] is not None else np.nan
        recs.append(rec)
    return pd.DataFrame(recs).set_index("date").sort_index()


def fetch_uncertainty() -> pd.DataFrame:
    epu = fetch_epu_daily()
    emv = fetch_emv_monthly()
    frame = pd.DataFrame({"epu_daily": epu}).join(emv, how="outer").sort_index()
    frame.index.name = "date"
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["gdelt", "wikipedia", "uncertainty", "all"],
                        default="all")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    jobs = {
        "gdelt": (lambda: fetch_gdelt(args.start, args.end), "features_gdelt.csv"),
        "wikipedia": (lambda: fetch_wikipedia(args.start, args.end), "features_wikipedia.csv"),
        "uncertainty": (fetch_uncertainty, "features_uncertainty.csv"),
    }
    todo = list(jobs) if args.source == "all" else [args.source]
    for name in todo:
        build, filename = jobs[name]
        frame = build()
        path = args.data_dir / filename
        frame.to_csv(path, float_format="%.10g")
        print(f"{name}: {len(frame):,} rows, "
              f"{frame.index.min().date()} -> {frame.index.max().date()} -> {path}")
        print(frame.notna().mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
