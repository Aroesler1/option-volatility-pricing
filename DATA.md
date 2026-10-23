# Data provenance

Every number in this repository comes from one of the sources below. The rule
that governs what is published is simple: **derived daily series are committed,
licensed rows are not.** Where a source could not be obtained reproducibly it is
recorded here as unavailable rather than quietly dropped.

## Price and volatility

| file | source | licence | what it is |
|---|---|---|---|
| `data/SPY_intraday_rv.csv` | Databento `XNAS.ITCH`, SPY 1-minute bars | programme licence, derived aggregate published | 2,094 daily rows, 2018-05 to 2026-08: realized variance, realized quarticity, bucket count, and the positive and negative realized semivariances |
| `data/SPY_daily.csv` | yfinance (public) | public | SPY unadjusted close and dividend-inclusive total return, 2018-01 to 2025-08, written by `fetch_spy_daily.py` |
| `data/databento/` | Databento raw `.dbn.zst` | licensed, **gitignored** | the 1-minute extract the RV series is built from |

`build_intraday_rv.py` rebuilds `SPY_intraday_rv.csv` from the raw extract. It
reproduces the previously committed series to floating-point precision (maximum
relative difference 5e-12 on realized variance and quarticity) and adds the two
semivariance columns the Patton-Sheppard model needs. Five-minute returns are
sampled inside the regular session only, on the grid 09:35 to 16:00 New York,
which gives 77 within-session returns on a full day.

## Option market

| series | source | committed? | notes |
|---|---|---|---|
| `atm_iv_30`, `atm_iv_91` | `optionm.stdopdYYYY`, days = 30 and 91 | no | standardised at-the-money implied volatility, call and put averaged |
| `atm_ivar_30` | derived | no | the square of `atm_iv_30`, the implied VARIANCE regressor of Busch, Christensen and Nielsen (2011) |
| `term_slope_30_91` | derived | yes | 30-day minus 91-day ATM implied volatility |
| `skew_25d_30` | `optionm.vsurfdYYYY`, days = 30, delta -25 and +25 | yes | 25-delta put IV minus 25-delta call IV |
| `spy_put_call` | `optionm.opprcdYYYY` | yes | total SPY put contract volume over total call contract volume |
| `vix`, `vxo` | `cboe.cboe` | no | VIX runs to 2026-08; VXO was discontinued and is present on 41% of dates, so it is fetched, reported and then excluded from the model panel |

`fetch_wrds_features.py` writes two files. `data/features_option_market.csv`
(committed, 2,177 rows, 2017-01-03 to 2025-08-29) holds the three derived
spreads and ratios. `data/option_market_vendor.csv` (gitignored) holds the
series that are vendor values or a one-to-one transform of one: the ATM
implied volatilities, implied variance, VIX and VXO. `alt_data.load_option_market`
joins them; `run_altdata_benchmark.py`, `run_option_pnl.py` and
`run_inference_audit.py` take `--option-vendor-csv` to point at a local copy
kept elsewhere. OptionMetrics coverage in WRDS ends **2025-08-29**, which is
what sets the right-hand end of the alternative-data study.

**What is committed and what a reader without the licence does.** Until
2026-10 the committed file also carried the ATM implied volatilities, implied
variance, VIX and VXO; those are now kept local (see
[docs/audit_history.md](docs/audit_history.md)). The remaining columns are
daily spreads and ratios derived from OptionMetrics under the Berkeley MFE
programme licence: one number per day per series, no option-level rows, no
quotes, no contracts. A reader without the licence rebuilds both files by
running `fetch_wrds_features.py` against their own WRDS entitlement; the script
is the definition of what the columns are. The underlying price and return
series is public (yfinance), and the option-level quotes the straddle backtest
needs are pulled to a gitignored parquet and never committed.
`fetch_wrds_features.py --surface-out data/SPY_ivsurface.parquet` also saves
the volatility-surface extract `run_iv_benchmark.py` reads (gitignored).

The underlying series was OptionMetrics `secprd` until 2026-09 and is now
yfinance. On the 1,926 overlapping days the two agree to $0.07 on the close at
worst and 4e-05 on average, and the change moved no straddle Sharpe by more than
3e-06 and no volatility-managed Sharpe by more than 0.008.

`fetch_option_chain.py` pulls the individual straddle legs for the option P&L
backtest into `data/option_chain_spy.parquet`. Those are option-level licensed
rows and are **gitignored**; only the daily P&L series in `results/` is
published. For each entry date the pull keeps the expiry nearest 30 calendar
days among those on or after the exit session 21 sessions later, so a hold
never runs past expiry. OptionMetrics' own `forward_price` column is NULL for
SPY across this sample, so at-the-money strikes are chosen against the `secprd`
closing spot inside the query. The local cache used for the committed results
was pulled before that expiry rule existed, so `run_option_pnl.py` re-selects
from the contracts it holds, accepting strikes within 1% of spot
(`--atm-tolerance`).

Both WRDS scripts refuse to open a connection unless `WRDS_DUO_READY=1` is set
for that run, attempt exactly one connection per invocation, and never retry a
refusal. WRDS is behind Duo, and a retry loop against Duo is how an account gets
locked.

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

The API answers few uncached timeline queries and returns HTTP 429 to the rest:
in the pull that built this file, successful requests came 1.5 to about 60
minutes apart (median about 12). `fetch_alt_data.py` pauses between requests
and caches every chunk under `data/.gdelt_cache/` (gitignored). This range is
32 chunks (two themes, two modes, eight yearly chunks; run with
`--end 2025-09-01`) and a cold pull takes several hours; a rerun is free. Ranges much longer than a year
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
  coursework only. No result depends on it. Only the per-post and daily
  sentiment scores are kept; post titles, bodies, Reddit IDs and timestamps
  were removed in 2026-10. `data/TSM_data.csv` and
  `data/weekly_stock_data.csv` are the course project's news-sentiment and
  price inputs for `Main.ipynb`, kept on the same basis.

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
convention, and [docs/full_results.md](docs/full_results.md) reports both.

## Reproducing

```bash
python build_intraday_rv.py --raw <extract>     # needs the Databento extract
python fetch_spy_daily.py                       # public, no credential
python fetch_alt_data.py                        # free sources, slow (GDELT)

# WRDS, behind Duo: set the acknowledgement only when ready to approve a push
WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_wrds_features.py
WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_option_chain.py

python run_altdata_benchmark.py --tag _purged   # current results; no tag rewrites the historical files
python run_option_pnl.py --tag _purged
```

Credentials come from `~/.pgpass` for WRDS, and the path to the intraday extract
is passed to `build_intraday_rv.py` with `--raw`. No credential is stored in this
repository, and no test reads one.


## Derived inference audit, 2026-09-06

No new source or licence is introduced. `results/audit_*.csv` records reproduced
QLIKE means, raw and Holm feature tests, frozen trading-rule parameters,
aggregate trade statistics, common-calendar daily portfolio P&L, and old/new
performance tables. The only licensed inputs used for the corrected trading
calculation were the existing ignored option-chain and entry caches. Contract
identifiers, strikes, quotes and contract-level deltas are not redistributed.

`python run_inference_audit.py --check` regenerates the audit summaries from
committed derived data. Rebuilding the daily P&L itself uses the cached chain
via `--rebuild-straddles --chain-dir <local-cache>` and makes no network call.
The stored adaptive-model forecasts predate the inner-validation repair and
are preserved; the later `_purged.csv` files contain the completed fixed-protocol refit. This audit does not create
a fresh holdout or alter historical forecast tables.

## Derived execution repair, 2026-09-08

No new source or licence is introduced. `results/*_purged.csv` preserves the
fixed-specification forecast refit and repaired strategy diagnostics separately
from historical outputs. The straddle daily file contains aggregate portfolio
P&L only. `results/integrity_*.csv` reconciles forecast loss, family inference,
trade admission counts and every premium-budget P&L row. Quote paths, strikes,
contract identifiers and raw deltas remain in the existing ignored cache.
Run `python run_integrity_report.py --check` without credentials or network.

## Straddle contract correction, 2026-10-23

No new source or pull. The straddle files and `results/integrity_pnl_comparison.csv`
were regenerated from the same local option-chain cache with expiries that
outlive the hold; old and new numbers are in
[docs/audit_history.md](docs/audit_history.md).
