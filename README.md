# Volatility Forecasting: HAR, HARQ, and Path-Dependent Models

A horse race between the standard econometric volatility forecasters and the
path-dependent volatility model of Guyon & Lekeufack, evaluated out-of-sample on
21 years of SPY under QLIKE with Diebold-Mariano tests.

**One specification significantly beats HAR-RV: HAR combined with option-implied
volatility** (p = 0.0188). Nothing built from returns alone does, and that null
is reported just as plainly.

## Out-of-sample results

SPY, 2005-2026. Target is realized volatility over the *next* 21 trading days,
built from strictly future returns. Test period 2018-01 to 2026-07 (2,155
observations), expanding-window refits every 21 days. QLIKE is the standard
robust loss for variance forecasts; lower is better.

| model | QLIKE | MSE |
|---|---|---|
| persistence (RV random walk) | 0.7922 | 0.0120 |
| **HAR-RV** (Corsi 2009) | 0.5425 | 0.0090 |
| HARQ (Bollerslev-Patton-Quaedvlieg) | 0.5438 | 0.0091 |
| PDV (Guyon-Lekeufack) | 0.8022 | 0.0115 |
| **HAR + PDV** (nested) | **0.5381** | 0.0090 |
| ridge on HAR features + extras | 0.5426 | 0.0084 |

Diebold-Mariano, Newey-West lag 20 (negative favours the first model):

| comparison | DM stat | p |
|---|---|---|
| HAR-RV vs persistence | **-3.43** | **0.0006** |
| HARQ vs HAR-RV | +0.28 | 0.78 |
| PDV vs HAR-RV | +1.42 | 0.16 |
| HAR+PDV vs HAR-RV | -0.77 | 0.44 |
| ridge vs HAR-RV | +0.00 | 1.00 |

### Reading these honestly

- **HAR-RV decisively beats persistence** (p = 0.0006). That is the one
  unambiguous result.
- **PDV alone underperforms HAR** at this horizon, scoring about the same as
  persistence. This is *not* a refutation of Guyon-Lekeufack: their claim is
  that path-dependence explains the *contemporaneous level* of volatility and
  the implied-vol surface, not that it forecasts 21-day-ahead realized vol
  better than HAR. This result scopes the claim rather than contradicting it.
- **Nesting PDV inside HAR gives the best QLIKE**, but the improvement over
  plain HAR is 0.8% and does not clear significance (p = 0.44). The honest
  reading is that path-dependence adds little that HAR's horizon averages do
  not already carry at this horizon.
- **HARQ does not improve on HAR here.** The realized-quarticity proxy is built
  from daily returns; HARQ is designed around intraday RQ, and the coarser proxy
  plausibly removes the measurement-error signal it exploits.
- **Ridge is statistically indistinguishable from HAR** (p = 1.00), matching the
  2026 finding that across nine zero-shot time-series foundation models and 50
  assets, econometric benchmarks remain competitive and only one small model
  beat Log-HAR at every horizon ([arXiv 2607.05291](https://arxiv.org/abs/2607.05291)).

## Does the option market beat the time series? Only once you strip the premium.

The models above forecast realized volatility from its own history. The option
market publishes a forward-looking estimate of the same quantity every day.
`run_iv_benchmark.py` tests whether it helps, using 30-day at-the-money implied
volatility from the OptionMetrics standardised surface (|delta| = 50), lagged one
day so no same-session information leaks in.

First, the bias is measured rather than assumed:

    mean(IV - subsequent realized vol) = +0.0422

Implied volatility sits **4.2 volatility points above** what actually gets
realized. That is the variance risk premium, and it means raw IV cannot be used
as a point forecast without a systematic error.

| model | QLIKE mean | QLIKE median | DM vs HAR-RV | p |
|---|---|---|---|---|
| persistence | 0.3835 | 0.1022 | +3.22 | 0.0013 |
| HAR-RV | 0.1274 | 0.0827 | — | — |
| raw ATM implied vol | 0.2434 | 0.2128 | +4.49 | 0.0000 |
| **HAR + implied vol** | **0.1182** | **0.0754** | **−2.35** | **0.0188** |

Two findings, and the second is the headline.

**Raw implied volatility is a worse forecast than HAR-RV** (p < 0.0001). Used
directly it is beaten decisively by a model that sees only past returns, because
the risk premium biases every forecast upward.

**But combined with HAR it beats HAR significantly** (p = 0.0188). Once a
regression is allowed to scale the premium away, implied volatility carries
genuine incremental information that the return history does not. This is the
only specification in this repository that significantly improves on HAR-RV, and
it is consistent with the literature: option-implied volatility is informative
but biased, and the two facts have to be handled separately.

Reproduce:

```bash
python run_iv_benchmark.py
```

## Does the realized-variance estimator change the answer? Yes.

The results above estimate realized volatility as a rolling standard deviation
of **daily** returns. Corsi's HAR-RV is defined on **intraday** realized
variance — the sum of squared intraday returns within a day. These are different
estimators, and the daily proxy uses one observation per day where the intraday
estimator uses 78.

`run_intraday_benchmark.py` holds everything else fixed — same 2,052-day sample
(2018-05 to 2026-07), same models, same target, same out-of-sample protocol —
and varies only the estimator. Intraday data is SPY 1-minute bars from Databento
`XNAS.ITCH`, aggregated to 5-minute returns. Both arms predict the *same*
intraday-based target, so only the regressors' estimator differs.

| estimator | model | QLIKE mean | QLIKE median | collapsed forecasts |
|---|---|---|---|---|
| intraday 5-min RV | persistence | 0.6422 | 0.1438 | 0 |
| intraday 5-min RV | **HAR-RV** | 0.2307 | 0.1064 | 0 |
| intraday 5-min RV | **HARQ** | 0.2315 | **0.1000** | 0 |
| intraday 5-min RV | HAR+PDV | 0.2378 | 0.1300 | 0 |
| daily-return proxy | HAR-RV | **2300.3** | 0.1378 | **2** |
| daily-return proxy | HARQ | 35.6 | 0.1251 | 0 |
| daily-return proxy | HAR+PDV | **23,000,658** | 0.1620 | **2** |

Three things follow.

**Intraday RV wins on both mean and median, for every model.** HAR-RV's median
QLIKE improves from 0.1378 to 0.1064 purely by changing the estimator.

**The daily proxy occasionally produces catastrophic forecasts.** Exactly 2 of
821 HAR forecasts collapse to the clipping floor, and because QLIKE punishes
under-forecasting without bound, those two days each contribute roughly 944,000
and destroy the mean. The intraday estimator produces zero such collapses. Mean
and median are both reported precisely so this tail behaviour is visible rather
than hidden behind whichever summary flatters the story — a volatility model
that occasionally forecasts approximately zero is dangerous in a way an average
does not convey.

**HARQ only works with the data it was designed for.** On the daily proxy, HARQ
was statistically indistinguishable from HAR (p = 0.78 in the table above). With
genuine intraday realized quarticity it posts the best median QLIKE of any model
tested, 0.1000 against HAR's 0.1064. That is exactly what the theory predicts:
HARQ corrects for time-varying measurement error in realized variance, and a
daily-return proxy for quarticity is too coarse to carry that signal. The
improvement does not clear significance on the mean-based DM test (+0.61,
p = 0.54), so it is reported as suggestive rather than established.

Reproduce:

```bash
python run_intraday_benchmark.py
```

Two methodology points that materially affect these numbers:

1. **Kernel selection is on QLIKE, not MSE.** MSE on a right-skewed volatility
   target rewards over-smoothed forecasts that cannot rise during stress. The
   MSE-selected PDV kernel scored a QLIKE of 1.7e6, because its forecasts
   collapse toward zero and QLIKE punishes under-forecasting without bound. The
   QLIKE-selected kernel scores 0.61 in sample.
2. **The DM tests use a Newey-West lag of h-1 = 20.** These are overlapping
   21-day forecasts, so loss differentials are serially correlated out to ~20
   lags; the generic n^(1/3) default (~13) understates the long-run variance and
   overstates significance.

Reproduce:

```bash
python run_vol_benchmark.py --ticker SPY --start 2005-01-01
```

## Scope note on the notebook

`Main.ipynb` is earlier coursework retained as a historical record. It collects
Google News and Reddit text and scores it with TextBlob, trains LSTM models, and
compares Black-Scholes prices with observed option quotes. Its volatility
methodology has two defects, both documented and fixed in `vol_forecasting.py`
(see below); the sentiment component is not part of the results above and no
claim in this README depends on it.

## Repository layout

- `Main.ipynb`: end-to-end notebook for data collection, feature engineering, modeling, and option-pricing experiments
- `vol_forecasting.py`: corrected, tested forecasting methodology (forward realized-vol target, train-only scaling, HAR-RV baseline, QLIKE, Diebold-Mariano)
- `tests/`: unit tests for the corrected methodology (no network or credentials needed)
- `data/weekly_stock_data.csv`: weekly multi-ticker research dataset with prices, sentiment, and forward-looking targets
- `data/TSM_data.csv`: daily volatility-oriented dataset used in the notebook experiments
- `data/sentiment_prev_full_month.json`: saved Reddit sentiment crawl used as a reproducible input artifact
- `report.pdf`: written project report summarizing the analysis and results

## Quickstart

Create an environment and install the libraries used in the notebook:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jupyter pandas numpy matplotlib scipy yfinance feedparser praw textblob torch pandas_market_calendars streamlit
```

Launch the notebook:

```bash
jupyter notebook Main.ipynb
```

If you want to use the Reddit ingestion cells, export credentials before opening the notebook:

```bash
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret
export REDDIT_USER_AGENT=option-volatility-pricing/0.1
```

## Methodology

The workflow combines three strands of analysis. First, it builds sentiment signals from Google News RSS and Reddit text using TextBlob polarity and subjectivity scores. Second, it downloads market and option-chain data with `yfinance`, computes rolling volatility features, and organizes the resulting data into weekly and daily research tables. Third, it trains LSTM models in PyTorch to forecast volatility-related targets and uses Black-Scholes plus `brentq` inversion to compare model-based prices and implied volatility with observed option quotes.

## Output

Running the notebook produces:

- cleaned sentiment datasets saved in `data/`
- volatility features and weekly modeling tables
- training curves and diagnostic plots for the LSTM experiments
- Black-Scholes prices and implied-volatility comparisons for selected option chains

## Methodology corrections (2026-08 revision)

The original notebook pipeline has two defects that `vol_forecasting.py` fixes (each covered by a unit test):

1. **Target definition.** The LSTM predicted the *current* 30-day rolling volatility, whose estimation window overlaps ~29/30 days with the input features; such a model mostly restates persistence. The corrected target is realized volatility over the *next* h days, computed from strictly future returns.
2. **Scaler leakage.** `MinMaxScaler` was fit on the full sample before the chronological split; the corrected utility scales with train-window statistics only.

The module also adds what any volatility-forecasting claim must be measured against: the HAR-RV baseline (Corsi 2009), the QLIKE loss (the standard robust loss for variance forecasts), and a Diebold-Mariano comparison test. Notebook outputs are retained as the historical record of the original experiments.

### Benchmark result (SPY, 21-day forward vol, OOS 2017-12 to 2026-07)

`run_vol_benchmark.py` runs the corrected pipeline end to end on ~21 years of SPY daily data (2,153 OOS observations, expanding-window refits every 21 days, checked-in results in `results/`):

| model | QLIKE | MSE | DM vs persistence |
|---|---|---|---|
| persistence (RV random walk) | 0.793 | 0.0120 | - |
| HAR-RV | **0.543** | 0.0090 | -3.83 (p = 0.0001) |
| ridge (HAR features + return extras) | 0.543 | 0.0084 | -3.77 (p = 0.0002) |

HAR-RV beats persistence decisively, and the regularized ML model is statistically indistinguishable from HAR-RV (DM -0.01, p = 0.99), replicating the standard finding that ML gains over HAR-RV on index volatility are marginal. Any LSTM re-run of the original notebook experiments should be held to this bar.

## Known limits

- Sentiment is driven by TextBlob rather than a finance-specific language model (FinBERT or LLM scoring is the natural upgrade; raw text is already stored)
- The workflow is notebook-centric and depends on interactive execution order
- Market and option-chain data from public sources can be sparse or inconsistent; yfinance chains are current-snapshot only, so implied-vol comparisons should use mids and drop zero-bid strikes
- Black-Scholes is used as a baseline pricing model and does not capture the full surface dynamics of listed options
