# Audit history

Dated corrections to published numbers: what changed, old and new values, why,
and how the change was checked. Earlier stages are documented in
[the inference and trading-clock audit](inference_audit_2026.md) (2026-09-06),
[the forecast and execution repair](execution_integrity_2026.md) (2026-09-08)
and the historical tables in [full_results.md](full_results.md).

## 2026-10-23: straddle contract selection, duplicate control, licensed data

### 1. The straddle expiry now outlives the 21-session hold

Problem. Each entry date used the expiry closest to 30 calendar days. In the
local quote cache those expiries were 27 to 32 days out, while a 21-session
hold ends 29 or more calendar days later, so many contracts expired before the
exit session. The backtest rejected those trades and the write-up described
them as incomplete quote paths. For the always-long control, 131 of its 152
rejections were expiry-before-exit, 21 were signals in the last 21 sessions of
data, and none were missing quotes.

Fix. `option_strategies.choose_straddle_contracts` keeps only expiries on or
after the exit session, then takes the one closest to 30 days and the strike
closest to spot. `fetch_option_chain.py` uses it for new pulls. The rerun did
not pull new data: `run_option_pnl.py` re-selects from the existing local quote
cache and accepts strikes within 1% of spot, the worst moneyness among the
original at-the-money picks in that cache (0.96%). Signals in the final 21
sessions are now counted as `after_last_full_hold` instead of as rejected
paths. Regression tests: `tests/test_straddle_selection.py`.

Rerun. `python run_option_pnl.py --tag _purged --chain-dir <local cache>
--option-vendor-csv <local file>`, then `python run_integrity_report.py`. The
volatility-targeting and VIX-proxy outputs reproduced byte for byte; only
`option_pnl_straddles_purged.csv`, `option_pnl_straddle_daily_purged.csv` and
`integrity_pnl_comparison.csv` changed.

Headline changes (`results/integrity_pnl_comparison.csv`):

| quantity | old | new |
|---|---|---|
| distinct books | 21 | 20 |
| books with positive sample Sharpe | 9 of 21 | 6 of 20, all long-only |
| best book | long-only HAR + implied variance, 0.413 | same book, 0.201 |
| two-sided books with positive Sharpe | 3 | 0 |
| always-long Sharpe | -0.224 | -0.198 |
| always-short Sharpe | -1.239 | -1.231 |
| attempts rejected for incomplete paths | 42% to 57% per book | 0 |
| attempts with no cached contract within 1% of spot | 0 | 76 of 3,361 (0% to 5% per book) |
| attempts in the last 21 sessions | counted as rejected | 228 of 3,361, counted separately |

Sensitivity of the new result to the strike tolerance: at 0.5% six long-only
books are positive, with no tolerance four are; every two-sided book is
negative in all three runs. These runs are not committed; repeat them with
`python run_option_pnl.py --tag _purged --horizons 21 --atm-tolerance 0.005`
(or `1.0`) and a
scratch `--results-dir` holding a copy of `altdata_forecasts_h21_purged.csv`.

All books:

| book | old Sharpe | new Sharpe | old trades / attempts | new trades / attempts |
|---|---:|---:|---:|---:|
| har, both | 0.041 | -0.114 | 65 / 128 | 121 / 128 |
| har_rv_iv, both | 0.025 | -0.069 | 59 / 122 | 114 / 122 |
| shar, both | 0.011 | -0.187 | 65 / 130 | 123 / 130 |
| persistence, both | -0.029 | -0.068 | 84 / 148 | 142 / 148 |
| hgb, both | -0.105 | -0.196 | 106 / 195 | 179 / 195 |
| combination, both | -0.211 | -0.167 | 79 / 167 | 146 / 167 |
| always_long, both | -0.224 | -0.198 | 213 / 365 | 335 / 365 |
| har_x_lasso, both | -0.522 | -0.392 | 86 / 176 | 155 / 176 |
| lstm, both | -0.886 | -0.718 | 99 / 180 | 162 / 180 |
| lstm_x, both | -0.991 | -0.867 | 119 / 228 | 203 / 228 |
| always_short, both | -1.239 | -1.231 | 213 / 365 | 335 / 365 |
| har_rv_iv, long_only | 0.413 | 0.201 | 40 / 93 | 87 / 93 |
| har, long_only | 0.398 | 0.160 | 46 / 99 | 94 / 99 |
| shar, long_only | 0.378 | 0.088 | 46 / 101 | 96 / 101 |
| persistence, long_only | 0.287 | 0.096 | 53 / 109 | 103 / 109 |
| combination, long_only | 0.261 | 0.164 | 57 / 133 | 114 / 133 |
| har_x_lasso, long_only | 0.058 | 0.016 | 61 / 139 | 120 / 139 |
| hgb, long_only | -0.033 | -0.086 | 85 / 165 | 151 / 165 |
| lstm, long_only | -0.204 | -0.157 | 68 / 137 | 119 / 137 |
| always_long, long_only | -0.224 | removed (duplicate) | 213 / 365 |  |
| lstm_x, long_only | -0.483 | -0.439 | 88 / 181 | 158 / 181 |

### 2. The duplicate always-long book is gone

The long-only filter leaves the always-long control unchanged, so its
long-only book was a copy of its two-sided book and was counted twice. The
study now has 20 distinct straddle books: 9 models in two-sided and long-only
form plus the always-long and always-short controls.

### 3. Licensed and third-party data removed from the tree (no result changes)

- `data/features_option_market.csv` no longer carries `atm_iv_30`, `atm_iv_91`,
  `atm_ivar_30`, `vix` or `vxo`. It keeps the derived `skew_25d_30`,
  `term_slope_30_91` and `spy_put_call` (2,460 rows to 2,177; dates with no
  derived value were dropped). The vendor series now go to the gitignored
  `data/option_market_vendor.csv`, and the scripts accept
  `--option-vendor-csv`. Check: the rebuilt model frames are identical to the
  old ones at all three horizons (1,800 rows each), so no forecast changes.
- `data/sentiment_prev_full_month.json` no longer carries Reddit post titles,
  bodies, IDs or timestamps. The per-post and daily scores the notebook reads
  are unchanged.
- `Main.ipynb` no longer contains a NewsAPI key; it reads `NEWSAPI_KEY` from the
  environment. Git history still contains the old key.

### 4. Other checks and corrections

- `run_iv_benchmark.py` reproduces `results/iv_vs_timeseries.csv` exactly from
  the local volatility-surface extract. Its out-of-sample window is 657 days
  ending 2025-01-02, where that extract ends, not the 821-day window of the
  intraday comparison. `fetch_wrds_features.py --surface-out` now writes the
  extract.
- The GDELT pacing statements were reconciled with the cache timestamps of the
  committed pull: successful requests came 1.5 to about 60 minutes apart
  (median about 12), and the 32-chunk pull took several hours.
