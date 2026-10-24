# SPY volatility forecasting with alternative data

I test whether option-market, news, attention and macro-calendar data improve
forecasts of SPY realized volatility over the HAR benchmark, and whether a
better forecast makes money in volatility-targeting and delta-hedged straddle
strategies.

Volatility forecasts drive option pricing, risk targeting and position sizing.
HAR, a regression on daily, weekly and monthly realized volatility, is the
standard benchmark and is hard to beat. Many papers report that sentiment or
attention data helps, but the gains are usually small, are often measured
against bare HAR, and rarely come with a trading test. I test the features
against HAR and a stronger HAR, correct for the number of tests, and finish
with trading tests, all on one out-of-sample protocol.

## Results

Sample: SPY, 720 out-of-sample forecast dates from 2022-10-07 to 2025-08-29,
forecasting realized volatility over the next 1, 5 and 21 trading days. Loss is
QLIKE, the standard loss for variance forecasts (lower is better).

- 1-day forecasts beat HAR. LSTM-X (an LSTM given the 16 features) has QLIKE
  0.1828 against 0.2095 for HAR, 12.7% lower, with Diebold-Mariano p = 0.038.
  LASSO over all features (0.1841, p = 0.002), HAR plus implied variance
  (0.1953, p = 0.001) and an equal-weight combination (0.1867, p < 0.001) also
  beat HAR at 5%. Source:
  [`results/integrity_dm_mcs.csv`](results/integrity_dm_mcs.csv).
- The gain fades with the horizon. At 5 days only LASSO and the combination
  beat HAR at 5%. At 21 days no model does, and LSTM-X is significantly worse
  than HAR (p = 0.044). Same source.
- No single feature survives a multiple-testing correction. Across 93
  one-feature-at-a-time tests (48 against HAR, 45 against HAR plus
  semivariance and implied variance), no improvement survives Holm. The best,
  1-day implied variance, goes from raw p = 0.0012 to Holm p = 0.051 within
  its 48 tests and 0.10 across all 93. Four features make 1-day forecasts
  significantly worse even after correction.
  Source: [`results/integrity_marginal_holm.csv`](results/integrity_marginal_holm.csv).
- Better forecasts did not improve volatility targeting. With a 15% volatility
  target and 5 bps costs, no model beats buy-and-hold by a significant margin
  (1-day: HAR Sharpe 1.39 against 1.31, paired block bootstrap p = 0.67). At 1
  day, models with lower QLIKE had lower Sharpe (Spearman +0.78). Source:
  [`results/option_pnl_volmanaged_purged.csv`](results/option_pnl_volmanaged_purged.csv).
- Straddles do not make a convincing case either. Over 365 sessions from
  2024-03-18, 6 of 20 delta-hedged straddle books have a positive sample
  Sharpe, all of them long-only, and every two-sided book loses. The best is
  long-only HAR plus implied variance at 0.20; the always-long control is
  -0.20 and always-short -1.23. Source:
  [`results/integrity_pnl_comparison.csv`](results/integrity_pnl_comparison.csv).

## How it works

1. Realized volatility: daily realized variance and up/down semivariances from
   SPY 1-minute bars, sampled every 5 minutes inside the regular session
   ([`build_intraday_rv.py`](build_intraday_rv.py)).
2. Features: 16 daily series in five blocks. Option market (OptionMetrics
   implied volatility and variance, skew, term slope, SPY put/call volume,
   VIX), GDELT news tone and coverage, Wikipedia attention, policy uncertainty,
   and FOMC, CPI and payrolls dates. Each one is lagged so it is known at the
   close of the forecast date ([`alt_data.py`](alt_data.py), [`DATA.md`](DATA.md)).
3. Models: persistence, HAR, semivariance HAR, HAR plus implied variance, HAR
   plus each feature alone, LASSO over all features, gradient-boosted trees, an
   LSTM with and without the features, and an equal-weight combination
   ([`vol_forecasting.py`](vol_forecasting.py),
   [`lstm_forecasting.py`](lstm_forecasting.py)).
4. Protocol: expanding window refit every 21 days, with the last 40% of 1,800
   aligned days held out. Training rows whose targets overlap the test block
   are dropped, and so are the overlapping rows in the inner validation splits
   that tune LASSO, boosting and the LSTM. The LSTM's scaler sees only inner
   training rows ([`run_altdata_benchmark.py`](run_altdata_benchmark.py)).
5. Statistics: Diebold-Mariano tests against HAR with a Newey-West lag of h-1,
   a 90% Model Confidence Set, and Holm corrections over the 93 single-feature
   tests ([`run_integrity_report.py`](run_integrity_report.py)).
6. Trading tests: volatility-targeted SPY, and delta-hedged at-the-money
   straddles that go long when the 21-day forecast (plus a premium estimated on
   the first half of the test window) is above 30-day implied variance and short
   when it is below. Each straddle uses the expiry nearest 30 days among those
   that outlive the 21-session hold, at the strike nearest spot. A decision at
   close t trades at close t+1. Option spreads and 5 bps hedge costs are charged
   ([`option_strategies.py`](option_strategies.py),
   [`run_option_pnl.py`](run_option_pnl.py)).
7. Checks: CI runs the tests and recomputes the committed headline tables from
   the derived CSVs ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

## Limitations

- One asset and one test window. The current results come from refitting
  fixed model specifications after an audit of a sample I had already studied,
  so they are not a fresh holdout.
- The 1-day gains rely on option quotes from the same close as the forecast.
  In the pre-repair run, one extra day of feature lag moved LASSO's p-value
  from 0.002 to 0.28 ([`results/altdata_models_lag1.csv`](results/altdata_models_lag1.csv)).
- The straddle rerun picks contracts from a local quote cache rather than a
  fresh pull, accepting strikes within 1% of spot, and the number of positive
  books depends on that tolerance ([audit history](docs/audit_history.md)). It
  ignores margin and financing, and returns are measured on a fixed
  option-premium budget.
- The model-level gains do not trace back to any single feature that survives
  the multiple-testing correction.

## Repository layout

| path | contents |
|---|---|
| `vol_forecasting.py` | HAR family, QLIKE, Diebold-Mariano, Model Confidence Set |
| `lstm_forecasting.py` | PyTorch LSTM forecaster |
| `alt_data.py` | feature panel with publication lags |
| `option_strategies.py` | volatility targeting and delta-hedged straddles |
| `run_altdata_benchmark.py`, `run_option_pnl.py` | main forecasting study and trading tests |
| `run_integrity_report.py`, `run_inference_audit.py` | offline verifiers for the committed tables |
| `build_intraday_rv.py`, `fetch_*.py` | data builders (some need licensed data) |
| `run_vol_benchmark.py`, `run_intraday_benchmark.py`, `run_iv_benchmark.py` | earlier baseline study of estimators and implied volatility |
| `report_tables.py` | regenerates the tables in `docs/full_results.md` |
| `data/` | derived daily inputs; licensed raw data is not committed |
| `results/` | every result CSV; files ending `_purged` and `integrity_*` are current |
| `tests/` | offline pytest suite |
| `Main.ipynb`, `report.pdf` | the course project this repository grew out of |

## Running it

These steps need no credentials, licensed data or network access. The tests use
synthetic fixtures, and the two verifiers recompute the committed tables from
the derived CSVs in `results/`. There is no end-to-end synthetic demo.

```bash
pip install -r requirements.txt
python -m pytest tests -q
python run_integrity_report.py --check
python run_inference_audit.py --check
python report_tables.py --which integrity_dm_mcs
```

A full rebuild needs licensed data:

- Databento SPY 1-minute bars: `python build_intraday_rv.py --raw path/to/extract.dbn.zst`
- WRDS access to OptionMetrics and CBOE (behind Duo):
  `WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_wrds_features.py`,
  then the same for `fetch_option_chain.py`. Only derived option-market
  series are committed; implied volatility, VIX and option-level quotes go to
  gitignored local files (`--option-vendor-csv` and `--chain-dir` point the
  scripts at them).
- Free sources: `python fetch_alt_data.py` (the GDELT pull is slow and cached)
  and `python fetch_spy_daily.py`.

Then run `python run_altdata_benchmark.py --tag _purged`,
`python run_option_pnl.py --tag _purged` and `python run_integrity_report.py`.
Without `--tag _purged` the scripts overwrite the historical result files that
the audit compares against. `run_iv_benchmark.py` also needs the local
volatility-surface extract that `fetch_wrds_features.py --surface-out
data/SPY_ivsurface.parquet` writes.

## More detail

- [Full results](docs/full_results.md): every table, including superseded
  stages, the baseline estimator study, regime splits and the extra-lag
  robustness check.
- [Audit history](docs/audit_history.md): dated corrections with old and new
  numbers (latest: 2026-10-23 straddle contract selection).
- [Forecast and execution repair, 2026-09-08](docs/execution_integrity_2026.md)
- [Inference and trading-clock audit, 2026-09-06](docs/inference_audit_2026.md)
- [Literature review](docs/literature_alt_data.md)
- [Data sources, licences and publication lags](DATA.md)
- [Original course report](report.pdf) and [notebook](Main.ipynb): a
  three-person course project (PIC 16B) on sentiment-based Black-Scholes
  pricing. It predates this study and its methods and results are superseded.
