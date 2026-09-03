# Volatility Forecasting: HAR, HARQ, PDV, and Implied Volatility

A controlled horse race across the volatility-forecasting literature, evaluated
out-of-sample under QLIKE with Diebold-Mariano tests on 21 years of SPY.

## What it does

- **Nine forecasters compared like-for-like** in the intraday arm: persistence,
  HAR-RV (Corsi 2009), HARQ (Bollerslev-Patton-Quaedvlieg), path-dependent
  volatility (Guyon-Lekeufack), a nested HAR+PDV, log-HAR, WLS-HAR under two
  weightings, and an equal-weight combination — with option-implied volatility
  and ridge compared in the other two arms
- **Three data regimes**, so the effect of the input is separable from the
  effect of the model: a daily-return proxy, true 5-minute realized variance
  from Databento intraday bars, and the OptionMetrics implied-volatility surface
- **Expanding-window refits** with a forward target built from strictly future
  returns, train-only scaling, and QLIKE as the loss
- **A Model Confidence Set** (Hansen, Lunde & Nason 2011) over all nine models,
  because pairwise Diebold-Mariano is the wrong tool for ranking many
  forecasters; DM with a Newey-West lag of h−1 is retained alongside it
- **The Clements-Preve remedies** — log-RV transformation and WLS estimation —
  which beat both HAR and HARQ here

## The three results

1. **Option-implied volatility significantly beats HAR-RV** (p = 0.0188) — but
   only once the variance risk premium is scaled out. Raw implied vol is
   decisively *worse* than a model that sees only past returns.
2. **The realized-variance estimator matters as much as the model.** Switching
   from a daily-return proxy to true intraday RV improves HAR-RV's median QLIKE
   from 0.1378 to 0.1064, and eliminates catastrophic forecast collapses.
3. **The winning change is to the estimator, not the model.** On true intraday
   RV, fitting HAR on log RV or by weighted least squares beats both HAR and
   HARQ, while PDV, HAR+PDV and a forecast combination are rejected from the
   90% Model Confidence Set. Those nulls are reported as plainly as the
   positive result.

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

Nine forecasters are compared, and the comparison is made with the **Model
Confidence Set** of [Hansen, Lunde and Nason](https://www.jstor.org/stable/41057463)
(*Econometrica* 79(2), 2011) rather than a fan of pairwise Diebold-Mariano
tests. With nine models there are 36 pairwise comparisons, each run at nominal
size, and the answer depends on which model gets nominated as the benchmark.
The MCS needs no benchmark: it returns the set of models that cannot be
distinguished from the best one, at a stated confidence level. The
implementation uses the range statistic of Hansen, Lunde and Nason (2003) with a
stationary bootstrap (Politis-Romano), which is what the overlapping 21-day
forecasts require - their loss differentials are serially correlated out to
about 20 lags.

Four of the nine come from Clements and Preve,
["A Practical Guide to Harnessing the HAR Volatility Model"](https://www.garp.org/hubfs/Whitepapers/a2r1W000000iDb0QAE_RiskIntell.6.20.19.Whitepaper.Volatility.pdf)
(*Journal of Banking and Finance*, 2021; [SSRN 3369484](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3369484)).
Their argument is that standard HAR pairs a dependent variable with sample
skewness above 10 with an estimator that is only efficient under homoskedastic
Gaussian errors, and that fixing the pairing beats improving the model. On SPX,
DJI and DAX they find the WLS and transformed-RV schemes are *always* in the 90%
MCS, while HARQ is not.

### Intraday 5-minute RV

| model | QLIKE mean | QLIKE median | collapsed | MCS p | in 90% MCS |
|---|---|---|---|---|---|
| **WLS-HAR** (w = 1/sqrt(RQ)) | **0.2132** | 0.0993 | 0 | 1.000 | **yes** |
| **log-HAR** | 0.2156 | **0.0946** | 0 | 0.696 | **yes** |
| **WLS-HAR** (w = 1/RV) | 0.2166 | 0.0960 | 0 | 0.696 | **yes** |
| mean combination | 0.2222 | 0.0994 | 0 | 0.089 | no |
| HAR-RV | 0.2307 | 0.1064 | 0 | 0.191 | yes |
| HARQ | 0.2315 | 0.1000 | 0 | 0.191 | yes |
| HAR + PDV | 0.2378 | 0.1300 | 0 | 0.014 | no |
| PDV | 0.3388 | 0.2303 | 0 | 0.001 | no |
| persistence | 0.6422 | 0.1438 | 0 | 0.001 | no |

**MCS(90%) = {WLS-HAR(RQ), log-HAR, WLS-HAR(RV), HAR-RV, HARQ}.** Membership is
identical across eight bootstrap seeds, and the excluded models' p-values never
approach 0.10.

**The Clements-Preve remedies beat HARQ.** All three post a lower mean QLIKE
than HARQ, and log-HAR and WLS-HAR(1/RV) also beat it on the median. The
previous headline here - that HARQ posts the best median QLIKE once it has real
intraday quarticity - no longer holds: log-HAR's 0.0946 beats HARQ's 0.1000, and
it gets there by changing the estimator rather than the model.

**HARQ is not rejected.** It sits inside the 90% MCS, as does plain HAR-RV. The
honest statement is two-sided: the remedies win on point estimate, and the MCS
cannot separate any of the five from each other on 821 overlapping
observations. What the MCS *does* settle is the bottom of the table - PDV,
HAR+PDV and persistence are excluded decisively.

**The forecast combination is excluded, and its constituents are not.** That
looks contradictory until you look at the elimination rule, which is
studentized. The combination is the average of HAR, HARQ, log-HAR and
WLS-HAR(1/RV), so it shares almost all of its variance with them; being
slightly worse than the best constituent is therefore measured very precisely,
and it is eliminated early despite a mean loss below HAR's and HARQ's. Equal
weighting is usually hard to beat, but not when one constituent dominates the
pool. (Forecast combination is not a Clements-Preve remedy - their paper does
not consider it. It is included so the MCS has one to reject.)

### Daily-return proxy, same target

| model | QLIKE mean | QLIKE median | collapsed | in 90% MCS |
|---|---|---|---|---|
| WLS-HAR (w = 1/RV) | 0.2759 | **0.1181** | 0 | yes |
| WLS-HAR (w = 1/sqrt(RQ)) | 0.2452 | 0.1205 | 0 | yes |
| log-HAR | 0.2480 | 0.1224 | 0 | yes |
| mean combination | 0.2537 | 0.1235 | 0 | yes |
| HARQ | 35.6 | 0.1251 | 0 | yes |
| HAR-RV | **2300.3** | 0.1378 | **2** | yes |
| HAR + PDV | **23,000,658** | 0.1620 | **2** | yes |
| persistence | 0.3232 | 0.1925 | 0 | yes |
| PDV | 0.3388 | 0.2303 | 0 | no |

Three things follow.

**Intraday RV wins on both mean and median, for every model.** HAR-RV's median
QLIKE improves from 0.1378 to 0.1064 purely by changing the estimator.

**The daily proxy occasionally produces catastrophic forecasts.** Exactly 2 of
821 HAR forecasts collapse to the clipping floor, and because QLIKE punishes
under-forecasting without bound, those two days each contribute roughly 944,000
and destroy the mean. The intraday estimator produces zero such collapses. Mean
and median are both reported precisely so this tail behaviour is visible rather
than hidden behind whichever summary flatters the story.

HARQ shows why the collapse count is a floor indicator and not a health check: it
posts a mean of 35.6 with **zero** collapses, because the counter only catches
forecasts pinned to the clipping floor (`pred <= 1.01e-4`) and HARQ's worst
forecasts land just above it — small enough that QLIKE's unbounded
under-forecast penalty still dominates the mean, large enough never to be
counted. A collapse count of 0 therefore means "nothing hit the floor", not
"nothing went wrong"; the mean-median gap is the statistic that catches it.

**Those collapses also destroy the MCS's power.** On the daily proxy eight of
nine models are in the 90% MCS, including persistence. That is not evidence that
persistence is competitive; it is the bootstrap variance of the loss
differentials being inflated by two observations of order 1e6 until nothing can
be distinguished from anything. It is a good illustration of why the collapse
count belongs in the table: a single pathological forecast does not just move a
mean, it disables the inference. The remedies help here too - **log-HAR and
WLS-HAR produce zero collapses on the daily proxy as well**, log-HAR because
exp() cannot return a non-positive forecast, so there is nothing to clip.

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
   overstates significance. The MCS handles the same dependence through the
   stationary bootstrap's block length rather than a kernel lag.
3. **The Clements-Preve remedies are adapted from variance to volatility
   units.** Their HAR models realized variance; this repository models
   annualized realized volatility throughout and squares the forecast inside
   QLIKE. Because log(RV) = 2·log(vol), the log-HAR slope coefficients are
   identical either way and only the intercept and the retransformation
   constant differ, so the +σ²/2 correction is applied on the scale the model
   is estimated on. The WLS weights follow their section 2.3.3 verbatim:
   w = 1/RV and w = 1/√RQ. Their other two schemes (GARCH-fitted weights, and
   weights from a fitted OLS HAR) are not implemented.

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
- `vol_forecasting.py`: corrected, tested forecasting methodology (forward realized-vol target, train-only scaling, HAR-RV / HARQ / PDV / log-HAR / WLS-HAR, QLIKE, Diebold-Mariano, and the Model Confidence Set)
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
- The MCS is run on 821 overlapping 21-day forecasts from a single asset. It separates the bottom of the table decisively but cannot separate the top five, and a longer sample or a cross-section of assets is what would settle that
- Clements and Preve also propose LAD (robust regression), a quartic-root transformation, and two further WLS weighting schemes. Only log-RV and the two nonparametric WLS schemes are implemented here
