# Data provenance

Every number in this repository comes from one of the sources below. The rule
that governs what is published is simple: **derived daily series are committed,
licensed rows are not.** Where a source could not be obtained reproducibly it is
recorded here as unavailable rather than quietly dropped.

## Price and volatility

| file | source | licence | what it is |
|---|---|---|---|
| `data/SPY_intraday_rv.csv` | Databento `XNAS.ITCH`, SPY 1-minute bars | programme licence, derived aggregate published | 2,094 daily rows, 2018-05 to 2026-08: realized variance, realized quarticity, bucket count, and the positive and negative realized semivariances |
| `data/SPY_optionmetrics_close.csv` | OptionMetrics `secprd` via WRDS | licensed, derived daily series published | SPY closing price and total return, 2018 to 2025-08 |
| `data/databento/` | Databento raw `.dbn.zst` | licensed, **gitignored** | the 1-minute extract the RV series is built from |

`build_intraday_rv.py` rebuilds `SPY_intraday_rv.csv` from the raw extract. It
reproduces the previously committed series to floating-point precision (maximum
relative difference 5e-12 on realized variance and quarticity) and adds the two
semivariance columns the Patton-Sheppard model needs. Five-minute returns are
sampled inside the regular session only, on the grid 09:35 to 16:00 New York,
which gives 77 within-session returns on a full day.

## Option market

| series | source | notes |
|---|---|---|
| `atm_iv_30`, `atm_iv_91` | `optionm.stdopdYYYY`, days = 30 and 91 | standardised at-the-money implied volatility, call and put averaged |
| `atm_ivar_30` | derived | the square of `atm_iv_30`, the implied VARIANCE regressor of Busch, Christensen and Nielsen (2011) |
| `term_slope_30_91` | derived | 30-day minus 91-day ATM implied volatility |
| `skew_25d_30` | `optionm.vsurfdYYYY`, days = 30, delta -25 and +25 | 25-delta put IV minus 25-delta call IV |
| `spy_put_call` | `optionm.opprcdYYYY` | total SPY put contract volume over total call contract volume |
| `vix`, `vxo` | `cboe.cboe` | VIX runs to 2026-08; VXO was discontinued and is present on 41% of dates, so it is fetched, reported and then excluded from the model panel |

Written by `fetch_wrds_features.py` to `data/features_option_market.csv`
(2,460 rows, 2017-01 to 2026-08). OptionMetrics coverage in WRDS ends
**2025-08-29**, which is what sets the right-hand end of the alternative-data
study.

`fetch_option_chain.py` pulls the individual straddle legs for the option P&L
backtest into `data/option_chain_spy.parquet`. Those are option-level licensed
rows and are **gitignored**; only the daily P&L series in `results/` is
published. OptionMetrics' own `forward_price` column is NULL for SPY across this
sample, so at-the-money strikes are chosen against the `secprd` closing spot.

## News tone

`data/features_gdelt.csv`, from the GDELT 2.0 DOC API (free, no credential).
Two theme filters, stated exactly so the pull is reproducible:

```
market  : theme:ECON_STOCKMARKET sourcelang:eng
economy : theme:EPU_ECONOMY      sourcelang:eng
```

`mode=timelinetone` gives the average GKG tone of matching coverage per day.
`mode=timelinevolraw` gives both the matching article count and the total number
of articles GDELT monitored that day; only the SHARE is used. The size of
GDELT's crawl moved by a factor of about three over this sample, falling from
roughly 520,000 monitored articles a day in early 2017 to 163,000 in late 2025,
so the raw count encodes the date more than it encodes the news.

Sanity check on the completed stock-market series: daily tone correlates -0.63
with VIX and log coverage share +0.30, so tone falls and coverage rises when
volatility rises. Both signs are the expected ones.

`data/features_gdelt.csv` holds 2,782 daily rows, 2018-01-01 to 2025-09-01,
with no gaps in either theme.

The API answers roughly one uncached timeline query every 10 to 15 minutes and
returns HTTP 429 otherwise, so `fetch_alt_data.py` pauses between requests and
caches every chunk under `data/.gdelt_cache/` (gitignored). A cold pull is 32
chunks and takes several hours; a rerun is free. Ranges much longer than a year
are refused however long you wait, which is why the pull is chunked by year.

## Attention

`data/features_wikipedia.csv`, from the Wikimedia pageviews REST API,
`agent=user` so bots are excluded. Four articles: `S&P_500`,
`Stock_market_crash`, `Recession`, `VIX`. 3,531 daily rows, 2017-01 to 2026-09.

Aggregation into a single attention index happens in `alt_data.py`, not in the
fetch: abnormal attention is log views minus the median of the previous 60 days
(the abnormal-search-volume definition of Da, Engelberg and Gao 2011), each
article is standardised on its own trailing 250-day window, and the four are
equal-weighted.

**Google Trends is deliberately absent.** Its endpoint is unofficial and its
samples differ between pulls, so a result built on it cannot be reproduced. That
is a decision, not an omission.

## Uncertainty

`data/features_uncertainty.csv`, from policyuncertainty.com, free to use with
attribution to the authors and the site.

- `epu_daily`: the daily news-based Economic Policy Uncertainty index of Baker,
  Bloom and Davis. Daily, 1985 to 2026-09.
- `emv_overall`: the Equity Market Volatility tracker of Baker, Bloom, Davis and
  Kost (2019). **Monthly**, not daily. It is stamped on the month it measures
  and applied from the following month, which is the earliest date it could be
  known.

The EMV workbook is `.xlsx`. It is parsed with the standard library (an `.xlsx`
is a zip of XML) rather than by adding an Excel dependency for one file.

## Calendar

`data/calendar_events.csv`: 278 announcement dates, 2018 to 2026.

- FOMC policy statement days (70), the final day of each scheduled meeting from
  federalreserve.gov, plus the two March 2020 emergency cuts dated by their
  announcement (3 March and 15 March 2020). The 15 March announcement was a
  Sunday and the dummy therefore lands on 16 March.
- CPI release days (104) and Employment Situation release days (104), from the
  BLS yearly release schedules at bls.gov/schedule. 2025 carries fewer than
  twelve of each because the government shutdown displaced the October and
  November releases.

These are published schedules, known years in advance, so the dummies carry no
publication lag. bls.gov refuses programmatic requests, so the dates are
committed as a reference file rather than fetched at run time; the source pages
are named above.

## Not available

- **RavenPack.** The `rpna` tables deny access under this WRDS entitlement and
  the trial table holds a single day. No result here depends on it.
- **AAII weekly bull-bear spread.** aaii.com returns HTTP 403 to programmatic
  requests, so the series cannot be pulled reproducibly and is excluded.
- **CBOE market-wide put/call ratio.** The daily statistics file is likewise
  403. The SPY-only put/call volume ratio from OptionMetrics is used instead and
  is named `spy_put_call` so the substitution is visible wherever it appears.
- **`data/sentiment_prev_full_month.json`.** A one-month Reddit crawl from 2025.
  It is not reproducible and is retained as an artifact of the original
  coursework only. No result depends on it.

## Publication lags

A feature stamped on date t must be knowable by the close of t. The information
set is "observable at the close of t", the same set that contains RV_t itself,
and the target always starts at t+1.

| block | lag | why |
|---|---|---|
| option market | 0 | OptionMetrics and CBOE closing values for date t, contemporaneous with RV_t |
| news (GDELT) | 1 session | GDELT days are UTC days, which run past the New York close |
| attention | 1 session | Wikimedia pageview days are UTC days |
| uncertainty (EPU) | 1 session | a whole newspaper day |
| EMV | 1 month | monthly series, applied from the month after the one it measures |
| calendar | 0 | release dates are published years in advance |

`run_altdata_benchmark.py --extra-lag 1` shifts every block one further session.
If a result only survives at `--extra-lag 0` it is an artifact of the
convention, and the README reports both.

## Reproducing

```bash
python build_intraday_rv.py                     # needs the Databento extract
WRDS_USERNAME=yourlogin python fetch_wrds_features.py
WRDS_USERNAME=yourlogin python fetch_option_chain.py
python fetch_alt_data.py                        # free sources, slow (GDELT)
python run_altdata_benchmark.py
python run_option_pnl.py
```

Credentials come from `~/.pgpass` for WRDS and from the `DATABENTO_RAW_DIR`
environment variable for the intraday extract. No credential is stored in this
repository, and no test reads one.
