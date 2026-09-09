# Can alternative data improve HAR forecasts of realized variance, and does it pay?

The hypothesis this repository tests: **sentiment, attention, news tone and the
option market carry information about future S&P 500 realized variance that the
volatility path itself does not, the gain is concentrated at particular
horizons, and if it is real it should show up as money in a strategy that bets
on volatility.**

Three horizons (1, 5 and 21 trading days), one out-of-sample protocol, one loss
function, and an economic test at the end. Everything that failed is reported
next to everything that worked.

**Sample:** SPY only, 720 forecast dates from 2022-10-07 to 2025-08-29.
The latest option diagnostic uses 365 common sessions from 2024-03-18 to
2025-08-29, after calibration and next-session execution. It admits only complete
quote paths, a retrospective coverage condition. Results apply to SPY and these
dates, not to a population or regime.

## The answer after the accounting and validation repairs

1. **The fixed-protocol forecast refit is complete.** Inner fitting and validation
   outcomes are separated, including the LSTM scaler. No specification was
   retuned. Historical and repaired QLIKE are side by side in
   `results/integrity_forecasts.csv`. At one day, LSTM-X changes from 0.18727
   to 0.18280; HAR remains 0.20947. This known-sample repair is not a new holdout.
2. **No individual feature improvement survives the family-wide test.** Holm
   counts 48 feature/horizon tests against HAR and 45 against rich HAR; the
   joint 93-test sensitivity agrees. The adjusted results remain failures of
   the incremental-feature hypothesis, despite selected-model forecast gains.
3. **The semivariance adaptation does not beat HAR.** It uses square-root
   semivariances and is not a direct replication of the cited variance-level model.
4. **The latest option accounting reverses the claim that every book loses.**
   Nine of 21 premium-budget books have positive sample Sharpe. These are
   conditional diagnostics, not established profits: incomplete future quote
   paths are excluded, margin and financing are absent, and multiple variants
   were inspected. The table below reports failures as well as improvements.

| model and rule | original Sharpe | latest Sharpe | admitted / attempted trades |
|---|---:|---:|---:|
| har, both | -0.672 | 0.041 | 65 / 128 |
| always_long, both | -1.174 | -0.224 | 213 / 365 |
| always_short, both | -0.720 | -1.239 | 213 / 365 |
| har_rv_iv, long_only | -0.631 | 0.413 | 40 / 93 |

Source: `results/integrity_pnl_comparison.csv`. Fees, hedge cash flows, signal
admissions and each daily book are independently reconciled by
`python run_integrity_report.py --check`. Missing quote paths are not silently
shortened or described as trades that could have been selected in advance.

The latest protocol is in [docs/execution_integrity_2026.md](docs/execution_integrity_2026.md).
The earlier [inference audit](docs/inference_audit_2026.md) and its tables are
preserved below as historical stages. Their all-books-lose conclusion and
unpurged model inputs are superseded by the latest tables, not overwritten.

## What the literature says to expect

The review is in [`docs/literature_alt_data.md`](docs/literature_alt_data.md),
which records for each paper the data, the specification, the horizon at which
gains were found and whether any P&L test was run. The predictions this study
was set up to check:

| expectation | source | horizon |
|---|---|---|
| Signed semivariance contains information beyond total variance | Patton and Sheppard (REStat 2015): Table 2 reports in-sample variance-level fits; Section VI separately tests forecasts. The model here is a square-root adaptation | 1, 5, 22 and 66 sessions |
| Option-implied information is the strongest single alternative-data class | Busch, Christensen and Nielsen (JoE 2011) | 5 to 21 days |
| Attention and sentiment act contemporaneously and decay fast | Da, Engelberg and Gao (RFS 2015): FEARS loads on realized variance and VIX only contemporaneously, and reverses within two days | 1 day, if anywhere |
| Search-based attention is the exception, with gains that GROW with the horizon | Dimpfl and Jank (EFM 2016): Mincer-Zarnowitz R2 up more than 3 points from 1 day to 2 weeks | 1 day to 2 weeks |
| Social-media sentiment shows in-sample significance and no out-of-sample gain | Behrendt and Schmidt (JBF 2018), an explicit negative result | none |
| Macro-announcement effects live intraday to daily and average out by a month | Andersen, Bollerslev, Diebold and Vega (2003, 2007) | 1 day |
| Statistical significance without economic significance is the norm for sentiment | Audrino, Sigrist and Ballinari (IJF 2020) call their own gain "small from an economic point of view" | all |
| A credible P&L story comes from an options overlay, not from the forecast alone | Goyal and Saretto (JFE 2009), Coval and Shumway (JF 2001) | monthly |
| Any "incremental" claim should be tested on top of a benchmark that ALREADY has signed jumps and implied volatility, not on top of bare HAR | the review's own methodological point | all |

Some of these survived contact with the data and some did not, and the last one
mattered enough that the marginal-value exercise is run twice.

## The data

Full provenance, licences and publication lags are in [`DATA.md`](DATA.md).
Sixteen features in five blocks:

| block | features | source |
|---|---|---|
| option market | 30-day ATM implied volatility and implied variance, 25-delta skew, 30 minus 91 day term slope, SPY put/call volume, VIX | OptionMetrics and CBOE via WRDS |
| news | GDELT average tone and log coverage share, for the stock-market and economy GKG themes | GDELT 2.0 DOC API, free |
| attention | Wikipedia pageviews for "S&P 500", "Stock market crash", "Recession" and "VIX", aggregated FEARS-style | Wikimedia REST API, free |
| uncertainty | daily news-based Economic Policy Uncertainty, monthly Equity Market Volatility tracker | policyuncertainty.com, free |
| calendar | FOMC statement, CPI release and payrolls release dummies | federalreserve.gov, bls.gov |

Realized variance, realized quarticity and the two realized semivariances come
from SPY 1-minute Databento bars aggregated to 5-minute returns, rebuilt from
the raw extract by [`build_intraday_rv.py`](build_intraday_rv.py), which
reproduces the previously committed series to 5e-12 relative and adds the
semivariance columns.

**Three sources are missing and are named rather than hidden.** RavenPack denies
access under this WRDS entitlement. The AAII bull-bear survey and CBOE's
market-wide put/call file both return HTTP 403 to programmatic requests; the
SPY-only put/call volume ratio stands in for the second and is named
`spy_put_call` so the substitution is visible. Google Trends was dropped **by
decision**, not by accident: its endpoint is unofficial and its samples differ
between pulls, so a result built on it cannot be reproduced.

The aligned sample runs **2018-05-31 to 2025-08-29**, starting where the
Databento intraday extract does and ending where OptionMetrics coverage in WRDS
ends. That is 1,800 days with every feature present. The last 40% is held out,
which gives **720 out-of-sample forecasts from 2022-10-07**, and every model is
scored on the identical set of dates, which the Model Confidence Set requires in
any case.

### The no-lookahead convention, stated once

A feature stamped on date t must be observable **at the close of t**. That is
the same information set that contains RV_t, which HAR already uses to forecast
realized variance over (t, t+h], so an option quote struck at the same close is
on the same footing. Series whose daily window runs past the New York close are
lagged one trading session: GDELT and Wikimedia days are UTC days, and the EPU
index counts a whole newspaper day. The monthly EMV tracker is applied from the
month after the one it measures. Announcement dummies carry no lag because
release schedules are published years in advance.

Every feature builder is tested for this directly: multiply a source file's
values after a cut date by ten, rebuild the panel, and assert that nothing on or
before the cut moved. That catches the mistakes a coverage check never does, a
backward fill, a centred window, a full-sample standardisation.

The convention is not neutral, and the robustness section below reports what
happens when it is tightened.

## The models

| model | what it adds |
|---|---|
| persistence | today's realized volatility, carried forward |
| HAR-RV | Corsi (2009), the benchmark everything is measured against |
| SHAR | Patton and Sheppard (2015): the daily term split into positive and negative semivariance |
| HAR-RV-IV | HAR plus 30-day at-the-money implied variance |
| HAR-X, one feature at a time | sixteen models, so each feature's marginal value is attributable |
| HAR-X with LASSO | every feature, with the selection refit inside the walk-forward |
| HistGradientBoosting | sklearn's histogram boosting on the HAR terms plus every feature |
| LSTM and LSTM-X | the notebook's PyTorch network on the corrected protocol, without and with the features |
| combination | equal-weight mean of everything above except persistence |

Protocol: expanding window, refit every 21 days, QLIKE loss, Diebold-Mariano
against HAR with a Newey-West lag of h-1, and a Model Confidence Set (Hansen,
Lunde and Nason 2011) at 90% over the structural models. A rolling MCS on
one-year and two-year windows shows which models survive in which regime.

### Two protocol details that moved the numbers

**The training window is purged.** The target at row t covers (t, t+h], so the
last h rows of a training window that ends where the test block begins have
targets reaching into that block. Those rows are dropped, at a cost of about 2%
of the training rows.

**Early stopping is done by hand.** sklearn's `early_stopping` flag holds out a
RANDOM fraction of training rows, which on a time series puts later rows into
validation and earlier ones into training. The boosting-round count is instead
chosen by QLIKE on the last 126 rows of the training window in time order.

Together those two changes cut the boosted model's apparent QLIKE advantage over
HAR roughly in half. Both leaks were small, and both flattered the
machine-learning model specifically. The baseline scripts in this repository do
not purge; the effect there is small and identical across their models, but it
is a genuine difference in protocol and is stated here rather than left for a
reader to find.

The audit also found an overlap at the INNER validation boundary. LASSO,
boosting and LSTM now purge that boundary by the forecast horizon; the LSTM
scaler now uses only inner fitting rows. The historical adaptive forecasts below
predate these repairs; the `_purged.csv` results above include them. They are retained for audit and must not be presented as
results regenerated by the repaired code. No specification was retuned here.

## Statistical results, stored before the inner-validation repair

Every table below is generated from `results/`. Regenerate them with
`python report_tables.py --inject README.md`.

Table A is the structural horse race: the models that differ in what they
know, scored on identical dates, with the Model Confidence Set run over
them. The one-feature-at-a-time models are kept out of it, because sixteen
near-identical HAR-X models would destroy the MCS's power over the models
that actually differ.

<!-- RESULTS:MODELS -->

#### Horizon 1 day

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| har_x_lasso | 0.1841 | 0.0815 | -3.0555 | 0.0022 | 1.0000 | yes |
| combination | 0.1871 | 0.0809 | -4.0950 | 0.0000 | 0.8210 | yes |
| lstm_x | 0.1873 | 0.0770 | -1.7534 | 0.0795 | 0.8210 | yes |
| hgb | 0.1945 | 0.0873 | -1.3813 | 0.1672 | 0.1640 | yes |
| har_rv_iv | 0.1953 | 0.0797 | -3.2499 | 0.0012 | 0.1365 | yes |
| har | 0.2095 | 0.0836 |  |  | 0.0055 | no |
| shar | 0.2102 | 0.0854 | 0.3673 | 0.7134 | 0.0055 | no |
| lstm | 0.2286 | 0.0795 | 2.3291 | 0.0199 | 0.0335 | no |
| persistence | 0.2783 | 0.1049 | 5.4691 | 0.0000 | 0.0000 | no |

MCS(90%) = {har_x_lasso, combination, lstm_x, hgb, har_rv_iv}

#### Horizon 5 days

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| hgb | 0.1847 | 0.0636 | -1.4680 | 0.1421 | 1.0000 | yes |
| har_x_lasso | 0.1866 | 0.0523 | -2.5245 | 0.0116 | 0.8600 | yes |
| combination | 0.1948 | 0.0567 | -2.2502 | 0.0244 | 0.3780 | yes |
| har_rv_iv | 0.2033 | 0.0569 | -0.6192 | 0.5358 | 0.3780 | yes |
| har | 0.2090 | 0.0595 |  |  | 0.1020 | yes |
| shar | 0.2099 | 0.0629 | 1.3060 | 0.1916 | 0.1020 | yes |
| lstm | 0.2430 | 0.0727 | 2.1400 | 0.0324 | 0.0460 | no |
| lstm_x | 0.2494 | 0.0578 | 1.4678 | 0.1422 | 0.1145 | yes |
| persistence | 0.3530 | 0.0953 | 5.1275 | 0.0000 | 0.0000 | no |

MCS(90%) = {hgb, har_x_lasso, combination, har_rv_iv, har, shar, lstm_x}

#### Horizon 21 days

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| hgb | 0.2278 | 0.0547 | -0.8588 | 0.3905 | 1.0000 | yes |
| combination | 0.2332 | 0.0750 | -1.1526 | 0.2491 | 0.7305 | yes |
| har | 0.2452 | 0.1023 |  |  | 0.5820 | yes |
| shar | 0.2453 | 0.1042 | 0.3823 | 0.7023 | 0.5820 | yes |
| har_rv_iv | 0.2485 | 0.1041 | 0.7209 | 0.4710 | 0.3440 | yes |
| har_x_lasso | 0.2530 | 0.0610 | 0.3025 | 0.7622 | 0.5820 | yes |
| lstm | 0.2641 | 0.0999 | 1.1107 | 0.2667 | 0.0565 | no |
| lstm_x | 0.3142 | 0.0562 | 1.9318 | 0.0534 | 0.0250 | no |
| persistence | 0.5811 | 0.1220 | 3.7072 | 0.0002 | 0.0005 | no |

MCS(90%) = {hgb, combination, har, shar, har_rv_iv, har_x_lasso}

<!-- END:MODELS -->

### The rolling Model Confidence Set

A single MCS over the whole sample averages a hiking cycle, a regional-banking
scare and two quiet years into one verdict. The rolling version shows whether a
model's membership is stable or carried by one regime.

<!-- RESULTS:ROLLING -->

#### Horizon 1, 252-observation windows (23 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| lstm_x | 96% |
| combination | 96% |
| har_rv_iv | 91% |
| hgb | 87% |
| lstm | 43% |
| har | 39% |
| shar | 39% |
| persistence | 22% |

#### Horizon 1, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| lstm_x | 100% |
| combination | 100% |
| har_rv_iv | 91% |
| hgb | 91% |
| persistence | 0% |
| har | 0% |
| shar | 0% |
| lstm | 0% |

#### Horizon 5, 252-observation windows (23 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| lstm_x | 100% |
| combination | 100% |
| har_rv_iv | 87% |
| hgb | 87% |
| lstm | 65% |
| har | 43% |
| shar | 43% |
| persistence | 22% |

#### Horizon 5, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_rv_iv | 100% |
| har_x_lasso | 100% |
| hgb | 100% |
| combination | 100% |
| lstm_x | 91% |
| har | 64% |
| shar | 64% |
| lstm | 45% |
| persistence | 0% |

#### Horizon 21, 252-observation windows (23 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| hgb | 100% |
| combination | 96% |
| lstm_x | 83% |
| lstm | 65% |
| har_rv_iv | 57% |
| har | 43% |
| shar | 43% |
| persistence | 0% |

#### Horizon 21, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| hgb | 100% |
| combination | 100% |
| lstm_x | 82% |
| har | 55% |
| shar | 55% |
| lstm | 45% |
| har_rv_iv | 36% |
| persistence | 0% |

<!-- END:ROLLING -->

### What each feature is worth on its own

Table B adds one feature at a time to HAR. Table C does the same on top of HAR
plus semivariance plus implied variance, which is the comparison the literature
review argued for: testing a sentiment series against bare HAR overstates what
it adds once the two cheap improvements are already in the model.

**Family-wide audit:** all horizons belong to the feature family. The original
raw tests are not individual discovery claims. Holm controls the chance of any
false rejection under valid input p-values; it does not repair other uncounted
research choices or historical adaptive-model validation.

<!-- RESULTS:AUDIT_HOLM -->

| benchmark | tests | raw better | raw worse | Holm better | Holm worse | joint better | joint worse |
|---|---|---|---|---|---|---|---|
| har | 48 | 2 | 9 | 0 | 4 | 0 | 3 |
| rich_har | 45 | 0 | 7 | 0 | 1 | 0 | 1 |

<!-- END:AUDIT_HOLM -->

<!-- RESULTS:MARGINAL -->

#### Horizon 1 day

| model | block | QLIKE mean | delta vs base | DM | raw p | Holm p | joint Holm p |
|---|---|---|---|---|---|---|---|
| atm_ivar_30 | option | 0.1953 | -0.0142 | -3.2499 | 0.0012 | 0.0508 | 0.1016 |
| term_slope_30_91 | option | 0.1988 | -0.0107 | -1.8357 | 0.0664 | 1.0000 | 1.0000 |
| atm_iv_30 | option | 0.2044 | -0.0051 | -0.7214 | 0.4707 | 1.0000 | 1.0000 |
| epu_log | uncertainty | 0.2062 | -0.0033 | -1.5447 | 0.1224 | 1.0000 | 1.0000 |
| is_payrolls | calendar | 0.2076 | -0.0019 | -1.1312 | 0.2580 | 1.0000 | 1.0000 |
| is_fomc | calendar | 0.2090 | -0.0005 | -0.6388 | 0.5230 | 1.0000 | 1.0000 |
| har | baseline | 0.2095 | 0.0000 |  |  |  |  |
| vix_vol | option | 0.2095 | 0.0000 | 0.0047 | 0.9962 | 1.0000 | 1.0000 |
| emv_overall | uncertainty | 0.2097 | 0.0002 | 0.3854 | 0.7000 | 1.0000 | 1.0000 |
| is_cpi | calendar | 0.2101 | 0.0007 | 0.5705 | 0.5683 | 1.0000 | 1.0000 |
| gdelt_share_econ | news | 0.2103 | 0.0008 | 2.1727 | 0.0298 | 1.0000 | 1.0000 |
| wiki_attention | attention | 0.2107 | 0.0012 | 0.4934 | 0.6218 | 1.0000 | 1.0000 |
| gdelt_share_mkt | news | 0.2113 | 0.0018 | 1.3562 | 0.1750 | 1.0000 | 1.0000 |
| spy_put_call | option | 0.2153 | 0.0059 | 3.4618 | 0.0005 | 0.0247 | 0.0483 |
| skew_25d_30 | option | 0.2206 | 0.0111 | 3.3716 | 0.0007 | 0.0336 | 0.0665 |
| gdelt_tone_econ | news | 0.2216 | 0.0121 | 3.9954 | 0.0001 | 0.0030 | 0.0059 |
| gdelt_tone_mkt | news | 0.2372 | 0.0277 | 4.5138 | 0.0000 | 0.0003 | 0.0006 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | raw p | Holm p | joint Holm p |
|---|---|---|---|---|---|---|---|
| atm_ivar_30 | option | 0.2033 | -0.0057 | -0.6192 | 0.5358 | 1.0000 | 1.0000 |
| term_slope_30_91 | option | 0.2038 | -0.0052 | -0.5145 | 0.6069 | 1.0000 | 1.0000 |
| epu_log | uncertainty | 0.2069 | -0.0022 | -0.7929 | 0.4279 | 1.0000 | 1.0000 |
| is_fomc | calendar | 0.2073 | -0.0017 | -1.6099 | 0.1074 | 1.0000 | 1.0000 |
| vix_vol | option | 0.2075 | -0.0015 | -0.1847 | 0.8535 | 1.0000 | 1.0000 |
| is_payrolls | calendar | 0.2084 | -0.0006 | -2.4578 | 0.0140 | 0.5732 | 1.0000 |
| is_cpi | calendar | 0.2087 | -0.0004 | -0.5931 | 0.5531 | 1.0000 | 1.0000 |
| atm_iv_30 | option | 0.2089 | -0.0001 | -0.0138 | 0.9890 | 1.0000 | 1.0000 |
| har | baseline | 0.2090 | 0.0000 |  |  |  |  |
| wiki_attention | attention | 0.2095 | 0.0005 | 0.4956 | 0.6202 | 1.0000 | 1.0000 |
| emv_overall | uncertainty | 0.2101 | 0.0011 | 0.6336 | 0.5263 | 1.0000 | 1.0000 |
| gdelt_share_mkt | news | 0.2106 | 0.0015 | 1.6386 | 0.1013 | 1.0000 | 1.0000 |
| spy_put_call | option | 0.2109 | 0.0018 | 1.4772 | 0.1396 | 1.0000 | 1.0000 |
| gdelt_share_econ | news | 0.2133 | 0.0042 | 1.5757 | 0.1151 | 1.0000 | 1.0000 |
| skew_25d_30 | option | 0.2230 | 0.0140 | 2.1559 | 0.0311 | 1.0000 | 1.0000 |
| gdelt_tone_econ | news | 0.2325 | 0.0234 | 2.4148 | 0.0157 | 0.6297 | 1.0000 |
| gdelt_tone_mkt | news | 0.2418 | 0.0328 | 2.7300 | 0.0063 | 0.2724 | 0.5320 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | raw p | Holm p | joint Holm p |
|---|---|---|---|---|---|---|---|
| emv_overall | uncertainty | 0.2438 | -0.0013 | -0.3013 | 0.7632 | 1.0000 | 1.0000 |
| wiki_attention | attention | 0.2447 | -0.0005 | -0.5925 | 0.5535 | 1.0000 | 1.0000 |
| is_payrolls | calendar | 0.2448 | -0.0003 | -1.2738 | 0.2027 | 1.0000 | 1.0000 |
| is_cpi | calendar | 0.2452 | -0.0000 | -0.0904 | 0.9280 | 1.0000 | 1.0000 |
| har | baseline | 0.2452 | 0.0000 |  |  |  |  |
| is_fomc | calendar | 0.2454 | 0.0002 | 0.0896 | 0.9286 | 1.0000 | 1.0000 |
| atm_ivar_30 | option | 0.2485 | 0.0033 | 0.7209 | 0.4710 | 1.0000 | 1.0000 |
| vix_vol | option | 0.2493 | 0.0041 | 0.2574 | 0.7969 | 1.0000 | 1.0000 |
| atm_iv_30 | option | 0.2501 | 0.0049 | 0.2967 | 0.7667 | 1.0000 | 1.0000 |
| spy_put_call | option | 0.2513 | 0.0062 | 0.9939 | 0.3203 | 1.0000 | 1.0000 |
| gdelt_share_mkt | news | 0.2520 | 0.0068 | 1.6908 | 0.0909 | 1.0000 | 1.0000 |
| term_slope_30_91 | option | 0.2541 | 0.0090 | 0.8787 | 0.3796 | 1.0000 | 1.0000 |
| skew_25d_30 | option | 0.2549 | 0.0097 | 0.6513 | 0.5149 | 1.0000 | 1.0000 |
| gdelt_share_econ | news | 0.2553 | 0.0101 | 0.8203 | 0.4120 | 1.0000 | 1.0000 |
| epu_log | uncertainty | 0.2589 | 0.0137 | 2.5369 | 0.0112 | 0.4697 | 0.9283 |
| gdelt_tone_econ | news | 0.2649 | 0.0197 | 0.7533 | 0.4513 | 1.0000 | 1.0000 |
| gdelt_tone_mkt | news | 0.3136 | 0.0684 | 1.3323 | 0.1827 | 1.0000 | 1.0000 |

<!-- END:MARGINAL -->

<!-- RESULTS:MARGINAL_RICH -->

#### Horizon 1 day

| model | block | QLIKE mean | delta vs base | DM | raw p | Holm p | joint Holm p |
|---|---|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.1904 | -0.0043 | -1.1699 | 0.2420 | 1.0000 | 1.0000 |
| gdelt_share_econ | news | 0.1938 | -0.0009 | -1.3156 | 0.1883 | 1.0000 | 1.0000 |
| is_payrolls | calendar | 0.1939 | -0.0008 | -0.7193 | 0.4720 | 1.0000 | 1.0000 |
| is_fomc | calendar | 0.1945 | -0.0002 | -0.2778 | 0.7811 | 1.0000 | 1.0000 |
| emv_overall | uncertainty | 0.1946 | -0.0001 | -0.1158 | 0.9078 | 1.0000 | 1.0000 |
| skew_25d_30 | option | 0.1946 | -0.0001 | -0.2560 | 0.7979 | 1.0000 | 1.0000 |
| har_rs_iv | baseline | 0.1947 | 0.0000 |  |  |  |  |
| is_cpi | calendar | 0.1954 | 0.0007 | 0.5633 | 0.5732 | 1.0000 | 1.0000 |
| wiki_attention | attention | 0.1956 | 0.0009 | 0.2906 | 0.7714 | 1.0000 | 1.0000 |
| epu_log | uncertainty | 0.1958 | 0.0011 | 1.8052 | 0.0710 | 1.0000 | 1.0000 |
| gdelt_share_mkt | news | 0.1979 | 0.0032 | 2.1110 | 0.0348 | 1.0000 | 1.0000 |
| vix_vol | option | 0.1991 | 0.0044 | 0.9952 | 0.3196 | 1.0000 | 1.0000 |
| spy_put_call | option | 0.1992 | 0.0045 | 3.1481 | 0.0016 | 0.0723 | 0.1430 |
| gdelt_tone_econ | news | 0.1993 | 0.0046 | 2.7780 | 0.0055 | 0.2297 | 0.4649 |
| atm_iv_30 | option | 0.1998 | 0.0051 | 1.0400 | 0.2983 | 1.0000 | 1.0000 |
| gdelt_tone_mkt | news | 0.2120 | 0.0173 | 3.6601 | 0.0003 | 0.0113 | 0.0229 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | raw p | Holm p | joint Holm p |
|---|---|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.2021 | -0.0020 | -0.2563 | 0.7977 | 1.0000 | 1.0000 |
| is_fomc | calendar | 0.2029 | -0.0012 | -1.2684 | 0.2046 | 1.0000 | 1.0000 |
| vix_vol | option | 0.2032 | -0.0009 | -0.1454 | 0.8844 | 1.0000 | 1.0000 |
| is_cpi | calendar | 0.2037 | -0.0004 | -1.2265 | 0.2200 | 1.0000 | 1.0000 |
| har_rs_iv | baseline | 0.2041 | 0.0000 |  |  |  |  |
| is_payrolls | calendar | 0.2042 | 0.0001 | 0.3373 | 0.7359 | 1.0000 | 1.0000 |
| wiki_attention | attention | 0.2044 | 0.0003 | 0.2534 | 0.7999 | 1.0000 | 1.0000 |
| spy_put_call | option | 0.2054 | 0.0013 | 1.2091 | 0.2266 | 1.0000 | 1.0000 |
| gdelt_share_econ | news | 0.2058 | 0.0017 | 1.7060 | 0.0880 | 1.0000 | 1.0000 |
| gdelt_share_mkt | news | 0.2058 | 0.0017 | 1.5350 | 0.1248 | 1.0000 | 1.0000 |
| atm_iv_30 | option | 0.2060 | 0.0019 | 0.2523 | 0.8008 | 1.0000 | 1.0000 |
| emv_overall | uncertainty | 0.2062 | 0.0021 | 1.5928 | 0.1112 | 1.0000 | 1.0000 |
| epu_log | uncertainty | 0.2070 | 0.0029 | 2.9618 | 0.0031 | 0.1315 | 0.2631 |
| skew_25d_30 | option | 0.2071 | 0.0030 | 1.5824 | 0.1136 | 1.0000 | 1.0000 |
| gdelt_tone_econ | news | 0.2150 | 0.0109 | 1.8530 | 0.0639 | 1.0000 | 1.0000 |
| gdelt_tone_mkt | news | 0.2278 | 0.0237 | 2.1622 | 0.0306 | 1.0000 | 1.0000 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | raw p | Holm p | joint Holm p |
|---|---|---|---|---|---|---|---|
| wiki_attention | attention | 0.2478 | -0.0008 | -0.8408 | 0.4004 | 1.0000 | 1.0000 |
| is_payrolls | calendar | 0.2482 | -0.0003 | -1.1955 | 0.2319 | 1.0000 | 1.0000 |
| emv_overall | uncertainty | 0.2482 | -0.0003 | -0.0657 | 0.9476 | 1.0000 | 1.0000 |
| har_rs_iv | baseline | 0.2485 | 0.0000 |  |  |  |  |
| is_cpi | calendar | 0.2485 | 0.0000 | 0.1053 | 0.9161 | 1.0000 | 1.0000 |
| is_fomc | calendar | 0.2487 | 0.0002 | 0.0945 | 0.9247 | 1.0000 | 1.0000 |
| vix_vol | option | 0.2492 | 0.0007 | 0.0533 | 0.9575 | 1.0000 | 1.0000 |
| atm_iv_30 | option | 0.2499 | 0.0013 | 0.1050 | 0.9163 | 1.0000 | 1.0000 |
| spy_put_call | option | 0.2548 | 0.0062 | 1.0128 | 0.3111 | 1.0000 | 1.0000 |
| gdelt_share_mkt | news | 0.2555 | 0.0070 | 1.6241 | 0.1044 | 1.0000 | 1.0000 |
| skew_25d_30 | option | 0.2573 | 0.0088 | 0.7581 | 0.4484 | 1.0000 | 1.0000 |
| gdelt_share_econ | news | 0.2578 | 0.0093 | 0.8550 | 0.3925 | 1.0000 | 1.0000 |
| term_slope_30_91 | option | 0.2589 | 0.0103 | 1.0286 | 0.3037 | 1.0000 | 1.0000 |
| epu_log | uncertainty | 0.2655 | 0.0170 | 2.1909 | 0.0285 | 1.0000 | 1.0000 |
| gdelt_tone_econ | news | 0.2664 | 0.0179 | 0.7227 | 0.4699 | 1.0000 | 1.0000 |
| gdelt_tone_mkt | news | 0.3157 | 0.0672 | 1.3195 | 0.1870 | 1.0000 | 1.0000 |

<!-- END:MARGINAL_RICH -->

Against bare HAR, one-day implied variance lowers the stored QLIKE by 0.014.
Its raw p-value 0.001155 becomes 0.050801 across the 48-test family and
0.101601 across the joint family. No improvement survives either family-wide
threshold. Against rich HAR, none improves even before adjustment. The
incremental-feature hypothesis fails this audit; selected-model confidence
sets answer a different question and do not reverse that result.

### Calm days against stressed days

The news and embedding literature is consistent that text signals earn their
keep on high-volatility days and contribute nothing the rest of the time, so a
pooled average can hide the whole effect. Days are split by whether realized
volatility exceeds the 80th percentile of every realized volatility observed
strictly before them, which is knowable in real time.

<!-- RESULTS:REGIME -->

#### Horizon 1 day (70 stressed of 720 days)

| model | QLIKE calm | QLIKE stressed | calm vs HAR | stressed vs HAR |
|---|---|---|---|---|
| lstm_x | 0.1802 | 0.2529 | -0.0117 | -0.1198 |
| har_x_lasso | 0.1732 | 0.2853 | -0.0187 | -0.0875 |
| hgb | 0.1834 | 0.2976 | -0.0085 | -0.0752 |
| combination | 0.1727 | 0.3209 | -0.0192 | -0.0519 |
| persistence | 0.2716 | 0.3405 | 0.0797 | -0.0323 |
| har_rv_iv | 0.1794 | 0.3429 | -0.0125 | -0.0299 |
| shar | 0.1932 | 0.3678 | 0.0013 | -0.0049 |
| har | 0.1919 | 0.3727 | 0.0000 | 0.0000 |
| lstm | 0.1972 | 0.5201 | 0.0053 | 0.1474 |

#### Horizon 5 days (70 stressed of 720 days)

| model | QLIKE calm | QLIKE stressed | calm vs HAR | stressed vs HAR |
|---|---|---|---|---|
| hgb | 0.1449 | 0.5537 | -0.0132 | -0.1279 |
| har_x_lasso | 0.1397 | 0.6223 | -0.0185 | -0.0594 |
| persistence | 0.3226 | 0.6351 | 0.1644 | -0.0465 |
| combination | 0.1433 | 0.6727 | -0.0148 | -0.0089 |
| har | 0.1581 | 0.6817 | 0.0000 | 0.0000 |
| shar | 0.1588 | 0.6843 | 0.0007 | 0.0026 |
| har_rv_iv | 0.1501 | 0.6972 | -0.0080 | 0.0155 |
| lstm_x | 0.1798 | 0.8951 | 0.0217 | 0.2134 |
| lstm | 0.1705 | 0.9154 | 0.0124 | 0.2337 |

#### Horizon 21 days (70 stressed of 720 days)

| model | QLIKE calm | QLIKE stressed | calm vs HAR | stressed vs HAR |
|---|---|---|---|---|
| hgb | 0.2152 | 0.3445 | -0.0170 | -0.0210 |
| har | 0.2322 | 0.3655 | 0.0000 | 0.0000 |
| shar | 0.2323 | 0.3663 | 0.0001 | 0.0008 |
| har_rv_iv | 0.2357 | 0.3673 | 0.0035 | 0.0018 |
| combination | 0.2181 | 0.3733 | -0.0141 | 0.0078 |
| har_x_lasso | 0.2393 | 0.3803 | 0.0070 | 0.0148 |
| persistence | 0.5942 | 0.4592 | 0.3620 | 0.0937 |
| lstm | 0.2347 | 0.5367 | 0.0025 | 0.1712 |
| lstm_x | 0.2882 | 0.5556 | 0.0559 | 0.1901 |

<!-- END:REGIME -->

The split cuts in opposite directions at the two ends of the horizon range, and
that is the most interesting thing in this section. At 1 day the feature-driven
models earn their advantage disproportionately in stress: the feature-augmented
LSTM is 0.120 QLIKE better than HAR on the 70 stressed days against 0.012 better
on the 650 calm ones, and the LASSO 0.088 against 0.019. That is exactly the
pattern the news and embedding literature reports. At 21 days the same two
models reverse: both LSTMs are far worse than HAR in stress and close to it
otherwise. A network that helps most when volatility spikes at a one-day horizon
and hurts most when it spikes at a one-month horizon is not a model anyone should
deploy on the strength of a pooled average.

### The semivariance result, in full

<!-- RESULTS:SHAR -->

| horizon | b on RS+ | b on RS- | HAR b on RV | corr(RS+, RS-) |
|---|---|---|---|---|
| 1 | 0.1686 | 0.5699 | 0.5254 | 0.9186 |
| 5 | 0.2228 | 0.4377 | 0.4625 | 0.9186 |
| 21 | 0.1481 | 0.2678 | 0.2904 | 0.9186 |

<!-- END:SHAR -->

The down-move term has the larger fitted coefficient at each horizon, while
this particular square-root semivariance variant fails to beat HAR. Correlated
regressors are a possible explanation, not a demonstrated cause. These point
estimates neither establish a significant coefficient contrast nor reject
Patton and Sheppard's variance-level forecasting result.

## Robustness: one more day of lag

`run_altdata_benchmark.py --extra-lag 1` shifts every feature one further
trading session back. It tests whether the 1-day result depends on using an
option quote struck at the same close it forecasts from.

It does.

<!-- RESULTS:LAGCOMPARE -->

#### Horizon 1 day

| model | p, lag as stated | p, one more day | in MCS | in MCS, one more day |
|---|---|---|---|---|
| har_x_lasso | 0.0022 | 0.2824 | yes | yes |
| combination | 0.0000 | 0.0594 | yes | yes |
| lstm_x | 0.0795 | 0.4071 | yes | yes |
| hgb | 0.1672 | 0.7356 | yes | no |
| har_rv_iv | 0.0012 | 0.1750 | yes | yes |
| har |  |  | no | yes |
| shar | 0.7134 | 0.7443 | no | yes |
| lstm | 0.0199 | 0.0223 | no | no |
| persistence | 0.0000 | 0.0000 | no | no |

#### Horizon 5 days

| model | p, lag as stated | p, one more day | in MCS | in MCS, one more day |
|---|---|---|---|---|
| hgb | 0.1421 | 0.7137 | yes | yes |
| har_x_lasso | 0.0116 | 0.5758 | yes | yes |
| combination | 0.0244 | 0.9772 | yes | yes |
| har_rv_iv | 0.5358 | 0.9298 | yes | yes |
| har |  |  | yes | yes |
| shar | 0.1916 | 0.1945 | yes | yes |
| lstm | 0.0324 | 0.0368 | no | yes |
| lstm_x | 0.1422 | 0.0307 | yes | yes |
| persistence | 0.0000 | 0.0000 | no | no |

#### Horizon 21 days

| model | p, lag as stated | p, one more day | in MCS | in MCS, one more day |
|---|---|---|---|---|
| hgb | 0.3905 | 0.7175 | yes | yes |
| combination | 0.2491 | 0.2998 | yes | yes |
| har |  |  | yes | yes |
| shar | 0.7023 | 0.6933 | yes | yes |
| har_rv_iv | 0.4710 | 0.4593 | yes | yes |
| har_x_lasso | 0.7622 | 0.6586 | yes | yes |
| lstm | 0.2667 | 0.2323 | no | no |
| lstm_x | 0.0534 | 0.0422 | no | no |
| persistence | 0.0002 | 0.0002 | no | no |

<!-- END:LAGCOMPARE -->

The stated convention is defensible: an OptionMetrics closing quote for date t
is known at the close of t, which is the same moment RV_t is known, and the
target starts at t+1. But the relationship is tight enough that one day of extra
caution removes most of it, and a reader is entitled to both numbers rather than
the flattering one.

The same thing happens at 5 days, where the LASSO goes from p = 0.012 to
p = 0.576 and the combination from 0.024 to 0.977. At 21 days nothing moves,
because nothing was significant there to begin with. These are historical, unadjusted model comparisons. They show sensitivity
to publication timing; they are not family-wide incremental-feature discoveries
and do not independently validate the later repaired forecasts.

## Did the statistical gains become money?

Three tests, in increasing order of what they cost to implement.

### Volatility-managed SPY

Weight is a 15% annualized volatility target divided by each model's forecast,
capped at 2, rebalanced daily, 5 bps per unit of turnover, with the weight set
at t earned on t+1. The cap never binds: HAR's forecast volatility never fell
below 8.0%, and the cap starts biting at 7.5%.

<!-- RESULTS:VOLMANAGED -->

#### Horizon 1

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| buy_and_hold |  | 1.2648 | 0.2110 | 0.1669 | -0.2023 |  |
| lstm | 0.2286 | 1.2648 | 0.2551 | 0.2017 | -0.2491 | 26.8163 |
| lstm_x | 0.1873 | 1.2540 | 0.2521 | 0.2011 | -0.2312 | 28.5267 |
| combination | 0.1871 | 1.2395 | 0.2477 | 0.1998 | -0.2352 | 32.4567 |
| har_x_lasso | 0.1841 | 1.2366 | 0.2469 | 0.1997 | -0.2423 | 30.3772 |
| har_rv_iv | 0.1953 | 1.2364 | 0.2521 | 0.2039 | -0.2461 | 36.2102 |
| har | 0.2095 | 1.2289 | 0.2479 | 0.2017 | -0.2392 | 39.8249 |
| shar | 0.2102 | 1.2219 | 0.2458 | 0.2012 | -0.2376 | 42.2129 |
| hgb | 0.1945 | 1.2075 | 0.2391 | 0.1980 | -0.2337 | 35.6330 |
| persistence | 0.2783 | 1.1694 | 0.2400 | 0.2053 | -0.2412 | 61.7334 |

QLIKE winner har_x_lasso, Sharpe winner lstm; Spearman(QLIKE, Sharpe) = -0.367 (p = 0.332)

#### Horizon 5

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| har_x_lasso | 0.1866 | 1.2870 | 0.2517 | 0.1956 | -0.2295 | 27.5989 |
| buy_and_hold |  | 1.2648 | 0.2110 | 0.1669 | -0.2023 |  |
| lstm_x | 0.2494 | 1.2577 | 0.2535 | 0.2016 | -0.2607 | 17.5289 |
| combination | 0.1948 | 1.2436 | 0.2401 | 0.1931 | -0.2362 | 26.7172 |
| har_rv_iv | 0.2033 | 1.2378 | 0.2407 | 0.1945 | -0.2400 | 32.5032 |
| lstm | 0.2430 | 1.2242 | 0.2383 | 0.1947 | -0.2516 | 15.2106 |
| har | 0.2090 | 1.2158 | 0.2349 | 0.1932 | -0.2355 | 37.7296 |
| hgb | 0.1847 | 1.2135 | 0.2329 | 0.1919 | -0.2246 | 31.9187 |
| shar | 0.2099 | 1.2108 | 0.2337 | 0.1930 | -0.2350 | 38.5769 |
| persistence | 0.3530 | 1.1694 | 0.2400 | 0.2053 | -0.2412 | 61.7334 |

QLIKE winner hgb, Sharpe winner har_x_lasso; Spearman(QLIKE, Sharpe) = -0.317 (p = 0.406)

#### Horizon 21

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| lstm_x | 0.3142 | 1.3639 | 0.2765 | 0.2027 | -0.2546 | 12.3101 |
| combination | 0.2332 | 1.3174 | 0.2437 | 0.1850 | -0.2232 | 16.9361 |
| hgb | 0.2278 | 1.3147 | 0.2513 | 0.1911 | -0.2261 | 20.4999 |
| har_x_lasso | 0.2530 | 1.2985 | 0.2537 | 0.1954 | -0.2276 | 27.5481 |
| har_rv_iv | 0.2485 | 1.2954 | 0.2324 | 0.1794 | -0.2207 | 23.2805 |
| har | 0.2452 | 1.2895 | 0.2309 | 0.1791 | -0.2211 | 24.6104 |
| shar | 0.2453 | 1.2849 | 0.2303 | 0.1793 | -0.2211 | 24.3932 |
| lstm | 0.2641 | 1.2751 | 0.2417 | 0.1896 | -0.2423 | 9.0299 |
| buy_and_hold |  | 1.2648 | 0.2110 | 0.1669 | -0.2023 |  |
| persistence | 0.5811 | 1.1694 | 0.2400 | 0.2053 | -0.2412 | 61.7334 |

QLIKE winner hgb, Sharpe winner lstm_x; Spearman(QLIKE, Sharpe) = -0.367 (p = 0.332)

<!-- END:VOLMANAGED -->

The reported paired comparison does not establish an improvement over
buy-and-hold. At five days, LASSO's stored Sharpe 1.287 exceeds buy-and-hold's
1.265, so the earlier statement that buy-and-hold led at both short horizons
was incorrect. These historical inputs are superseded by the separate purged rerun.

The rank correlation says the same thing more precisely. Spearman between QLIKE
and Sharpe across models is -0.37, -0.32 and -0.37 at 1, 5 and 21 days, all with
p above 0.3. **A better volatility forecast tilted the P&L ranking in the right
direction and not by enough to distinguish from noise.**

### Historical intermediate straddle audit, superseded above

At each forecast date the 21-day forecast, with the variance risk premium added
back, is compared to 30-day at-the-money implied variance: long a straddle when
the forecast is higher by a frozen margin, short when lower, flat otherwise.
Both the premium adjustment and the margin are estimated on the first half of
the out-of-sample window and then frozen. Entry is at the touch and exit is at
the touch, so a round trip pays the full quoted spread, and the SPY hedge is
charged 5 bps on the shares it trades.

The old implementation traded its calibration half. The corrected calculation
starts every conditional and unconditional rule after 2024-03-14, uses one
calendar including flat days, and refuses a hedge whose entry delta is missing.
Later missing deltas may carry the previous observed delta, never a future one.
The table below preserves the historical results beside the trading-clock and
hedge correction. Adaptive forecast inputs are still the stored legacy inputs;
this does not validate their model selection.

<!-- RESULTS:AUDIT_STRADDLES -->

#### both

| model | old Sharpe | corrected Sharpe | old mean p.a. | corrected mean p.a. | old trades | corrected trades |
|---|---|---|---|---|---|---|
| always_long | -1.1742 | -0.2996 | -1.0433 | -0.3160 | 719 | 359 |
| hgb | -0.4820 | -0.4079 | -0.2675 | -0.2601 | 380 | 200 |
| combination | -0.5621 | -0.4925 | -0.3080 | -0.3064 | 347 | 167 |
| har_rv_iv | -0.6359 | -0.5151 | -0.2908 | -0.2419 | 302 | 122 |
| har | -0.6720 | -0.5889 | -0.3036 | -0.2686 | 308 | 128 |
| har_x_lasso | -0.6190 | -0.5913 | -0.3482 | -0.3897 | 353 | 173 |
| persistence | -0.6311 | -0.6027 | -0.2206 | -0.2200 | 328 | 148 |
| shar | -0.7435 | -0.6974 | -0.3337 | -0.3136 | 310 | 130 |
| lstm_x | -0.9694 | -1.0676 | -0.5709 | -0.7814 | 410 | 230 |
| lstm | -0.9566 | -1.1452 | -0.4711 | -0.6110 | 349 | 169 |
| always_short | -0.7202 | -1.2061 | -0.6626 | -1.3056 | 719 | 359 |

#### long_only

| model | old Sharpe | corrected Sharpe | old mean p.a. | corrected mean p.a. | old trades | corrected trades |
|---|---|---|---|---|---|---|
| combination | -0.4926 | -0.1399 | -0.2373 | -0.0825 | 249 | 136 |
| har_rv_iv | -0.6315 | -0.1800 | -0.2453 | -0.0803 | 204 | 93 |
| har_x_lasso | -0.5072 | -0.1820 | -0.2441 | -0.1104 | 243 | 136 |
| har | -0.6689 | -0.2467 | -0.2560 | -0.1070 | 210 | 99 |
| hgb | -0.5407 | -0.2804 | -0.2769 | -0.1790 | 287 | 171 |
| always_long | -1.1742 | -0.2996 | -1.0433 | -0.3160 | 719 | 359 |
| persistence | -0.7888 | -0.3389 | -0.2416 | -0.1286 | 198 | 109 |
| shar | -0.7521 | -0.3559 | -0.2853 | -0.1520 | 212 | 101 |
| lstm | -0.9292 | -0.6697 | -0.3620 | -0.3046 | 233 | 124 |
| lstm_x | -0.8833 | -0.6823 | -0.4536 | -0.4583 | 289 | 181 |

<!-- END:AUDIT_STRADDLES -->

The full corrected metrics are in `results/audit_straddle_summary.csv`, with
the common daily P&L in `results/audit_straddle_daily.csv`. There are 360 decision
dates and six intervening flat sessions. In this earlier intermediate calculation, all 21 variants lose. The
always-long control beats every two-sided model on Sharpe, while several
long-only filters lose less than the control. Neither finding establishes
profitable trading or significance of a selected ranking.

#### Historical table, including calibration-period trades

<!-- RESULTS:STRADDLES -->

#### both

| model | Sharpe | mean p.a. | max drawdown | worst month | trade hit rate | turnover p.a. | trades |
|---|---|---|---|---|---|---|---|
| hgb | -0.4820 | -0.2675 | -1.0346 | -0.2582 | 0.4053 | 6.2810 | 380 |
| combination | -0.5621 | -0.3080 | -0.9934 | -0.3911 | 0.3948 | 5.7355 | 347 |
| har_x_lasso | -0.6190 | -0.3482 | -1.1254 | -0.4875 | 0.3796 | 5.8347 | 353 |
| persistence | -0.6311 | -0.2206 | -0.8022 | -0.1515 | 0.4482 | 5.4215 | 328 |
| har_rv_iv | -0.6359 | -0.2908 | -1.0570 | -0.3053 | 0.4040 | 5.0333 | 302 |
| har | -0.6720 | -0.3036 | -1.0407 | -0.3049 | 0.3994 | 5.1333 | 308 |
| always_short | -0.7202 | -0.6626 | -3.3234 | -1.2436 | 0.5285 | 11.8843 | 719 |
| shar | -0.7435 | -0.3337 | -1.0904 | -0.3049 | 0.3871 | 5.1667 | 310 |
| lstm | -0.9566 | -0.4711 | -1.4652 | -0.5639 | 0.3926 | 5.7686 | 349 |
| lstm_x | -0.9694 | -0.5709 | -1.7523 | -0.5474 | 0.3634 | 6.7769 | 410 |
| always_long | -1.1742 | -1.0433 | -3.7604 | -0.3153 | 0.2378 | 11.8843 | 719 |

#### long_only

| model | Sharpe | mean p.a. | max drawdown | worst month | trade hit rate | turnover p.a. | trades |
|---|---|---|---|---|---|---|---|
| combination | -0.4926 | -0.2373 | -0.9907 | -0.2445 | 0.3173 | 4.1157 | 249 |
| har_x_lasso | -0.5072 | -0.2441 | -0.9318 | -0.2321 | 0.3004 | 4.0165 | 243 |
| hgb | -0.5407 | -0.2769 | -1.0550 | -0.2582 | 0.3449 | 4.7438 | 287 |
| har_rv_iv | -0.6315 | -0.2453 | -1.0615 | -0.2519 | 0.3137 | 3.4000 | 204 |
| har | -0.6689 | -0.2560 | -1.0391 | -0.2432 | 0.3143 | 3.5000 | 210 |
| shar | -0.7521 | -0.2853 | -1.0774 | -0.2519 | 0.3019 | 3.5333 | 212 |
| persistence | -0.7888 | -0.2416 | -0.9131 | -0.1515 | 0.3030 | 3.2727 | 198 |
| lstm_x | -0.8833 | -0.4536 | -1.3388 | -0.2621 | 0.2699 | 4.7769 | 289 |
| lstm | -0.9292 | -0.3620 | -1.1062 | -0.2583 | 0.2876 | 3.8512 | 233 |
| always_long | -1.1742 | -1.0433 | -3.7604 | -0.3153 | 0.2378 | 11.8843 | 719 |

<!-- END:STRADDLES -->

The historical table above includes calibration-period trades and unequal
calendars. Its model-versus-control ranking is superseded by the corrected
table. The previously quoted cross-model rank-correlation tests also belong
to that historical calculation.

### Descriptive VIX versus SPY variance proxy

<!-- RESULTS:SWAP -->

n_overlapping                                  720
mean_variance_points                     -0.018824
mean_vol_points_equivalent               -0.137201
nw_tstat_overlapping                     -6.969029
n_non_overlapping                               35
mean_non_overlapping                     -0.018228
tstat_non_overlapping                     -4.66586
sharpe_annualised                        -2.732048
share_positive                            0.051389
worst_observation                        -0.208197
model                       long_variance_swap_vix

<!-- END:SWAP -->

The stored long-variance proxy averages -0.0188 annualized variance units;
its single non-overlapping offset has Sharpe -2.73 over 35 observations.
These are descriptive statistics of SPY regular-session realized variance
minus VIX squared. VIX reflects SPX options over 30 calendar days. The target
uses SPY within-session returns over 21 trading days, excluding overnight
variation. Underlying, horizon and payoff coverage differ even at zero cost.

The old statement that a +2.7 versus -0.72 Sharpe gap was entirely implementation
is withdrawn. This comparison does not identify a tradeable swap return, a
matched variance risk premium, or the amount of option loss attributable to
costs. The existing proxy table remains an audit record.

### One sentence per horizon

- **1 day:** stored QLIKE gains are largest, but no individual feature clears
  the family-wide audit; the volatility overlay shows no established economic gain.
- **5 days:** stored model rankings do not establish incremental information
  after the audit; no five-day option-holding strategy was evaluated.
- **21 days:** nine latest complete-path books have positive sample Sharpe,
  but missing-path selection and incomplete capital accounting prevent a tradable-profit claim.

## What did not work

- The incremental-feature hypothesis fails the 48/45-test Holm audit and its
  joint 93-test sensitivity. Significant worsening survives for some features.
- The earlier all-straddles-lose conclusion is superseded. The latest positive
  books still fail to establish executable, selection-adjusted profitability.
- The square-root semivariance adaptation does not beat HAR; it is not a
  direct replication of the variance-level model in the cited paper.
- Adaptive models' inner validation and the LSTM scaler required repair.
  The fixed-protocol rerun is now in the separate `_purged.csv` files.
- The VIX proxy cannot isolate implementation costs or establish executable
  variance-swap performance.

## Baseline: the horse race this study starts from

Three results established before any alternative data entered the repository.
They are unchanged, they still reproduce exactly, and they set the bar. Sample
SPY 2005-2026 for the daily arm and 2018-05 to 2026-07 for the intraday arm,
821 overlapping 21-day forecasts, expanding-window refits every 21 days.

**1. Option-implied volatility beats HAR-RV, but only after the risk premium is
scaled out.** Implied volatility sits 4.2 volatility points above what gets
realized, so raw implied vol used as a point forecast is decisively worse than a
model that sees only past returns. Allowed to scale the premium away, it wins.

| model | QLIKE mean | QLIKE median | DM vs HAR-RV | p |
|---|---|---|---|---|
| persistence | 0.3835 | 0.1022 | +3.22 | 0.0013 |
| HAR-RV | 0.1274 | 0.0827 | | |
| raw ATM implied vol | 0.2434 | 0.2128 | +4.49 | 0.0000 |
| **HAR + implied vol** | **0.1182** | **0.0754** | **-2.35** | **0.0188** |

**2. The realized-variance estimator matters as much as the model.** Switching
from a daily-return proxy to true intraday 5-minute RV improves HAR-RV's median
QLIKE from 0.1378 to 0.1064 and eliminates catastrophic forecast collapses. On
the daily proxy exactly 2 of 821 HAR forecasts collapse to the clipping floor,
and because QLIKE punishes under-forecasting without bound those two days each
contribute roughly 944,000 and destroy the mean.

**3. The winning change is to the estimator, not the model.** On true intraday
RV, fitting HAR on log RV or by weighted least squares beats both HAR and HARQ.

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

**The honest MCS statement, kept verbatim.** The Clements-Preve remedies beat
HARQ on point estimate, and HARQ is **not rejected**: it sits inside the 90%
MCS, as does plain HAR-RV, and the MCS cannot separate any of the top five from
each other on 821 overlapping observations. What the MCS does settle is the
bottom of the table, where PDV, HAR+PDV and persistence are excluded decisively.
Membership is identical across eight bootstrap seeds.

The forecast combination is excluded while its own constituents are not. That
looks contradictory until you look at the elimination rule, which is
studentized: the combination shares almost all of its variance with its
constituents, so being slightly worse than the best of them is measured very
precisely and it goes early.

Reproduce the baseline:

```bash
python run_vol_benchmark.py --ticker SPY --start 2005-01-01
python run_intraday_benchmark.py
python run_iv_benchmark.py
```

## Reproducing the alternative-data study

```bash
pip install -r requirements.txt

python build_intraday_rv.py
python fetch_spy_daily.py
python fetch_alt_data.py

# WRDS is behind Duo. The fetchers refuse to connect unless this is set for the
# run, attempt one connection each, and never retry a refusal.
WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_wrds_features.py
WRDS_DUO_READY=1 WRDS_USERNAME=yourlogin python fetch_option_chain.py

python run_altdata_benchmark.py
python run_option_pnl.py
python run_altdata_benchmark.py --extra-lag 1
python report_tables.py
```

WRDS credentials come from `~/.pgpass` and the intraday extract from
`DATABENTO_RAW_DIR`. No credential is stored in this repository and no test
reads one. The only OptionMetrics material committed here is
`data/features_option_market.csv`, which holds daily summaries derived under the
programme licence; a reader without that licence regenerates it with
`fetch_wrds_features.py`. The underlying price and return series is public. GDELT's API answers roughly one uncached timeline query every several
minutes and returns HTTP 429 in between, so a cold news pull takes hours; every
chunk is cached under `data/.gdelt_cache/`, so a rerun is free.

## Repository layout

| file | what it is |
|---|---|
| `vol_forecasting.py` | the model library: HAR, HARQ, PDV, log-HAR, WLS-HAR, SHAR, HAR-X, QLIKE, Diebold-Mariano, the Model Confidence Set and its rolling version |
| `lstm_forecasting.py` | the notebook's LSTM as a tested module on the corrected protocol |
| `alt_data.py` | the feature panel: alignment, publication lags, causal standardisation |
| `option_strategies.py` | volatility-managed exposure, delta-hedged straddles, the synthetic variance swap |
| `build_intraday_rv.py` | rebuilds the realized-variance series from the raw Databento extract |
| `fetch_wrds_features.py`, `fetch_option_chain.py` | the WRDS pulls, behind a Duo guard that allows one connection attempt per run |
| `fetch_alt_data.py`, `fetch_spy_daily.py` | the public pulls: GDELT, Wikipedia, policy uncertainty, and SPY daily prices |
| `run_altdata_benchmark.py` | the horse race at three horizons |
| `run_option_pnl.py` | the economic evaluation |
| `report_tables.py` | prints this README's tables from `results/` |
| `run_vol_benchmark.py`, `run_intraday_benchmark.py`, `run_iv_benchmark.py` | the baseline chapter |
| `docs/literature_alt_data.md` | the literature review, with the horizon of each paper's result |
| `tests/` | the suite, offline, no credentials |
| `Main.ipynb`, `report.pdf` | the original coursework, retained as a historical record |

## Known limits

- **One asset and one evaluation window.** No cross-asset or regime conclusion
  follows from this sample.
- **The 1-day result depends on the timing convention.** One extra day of
  publication lag removes most of it. Both numbers are reported above.
- **The MCS has limited power here.** At 21 days it retains six of
  nine models in the stored table. A cross-section of assets is what would settle that, and it is
  the natural next step: pooled HAR across assets (Bollerslev, Hood, Huss and
  Pedersen, RFS 2018) estimated on TAQ.
- **The straddle book holds one at-the-money straddle per signal.** Its payoff
  differs from the variance proxy, and the difference is not identified as costs.
- **The VIX/SPY proxy is not a matched swap payoff.** Its underlying, horizon
  and overnight coverage differ, as well as its omitted costs.
- **`data/sentiment_prev_full_month.json`** is a one-month Reddit crawl from
  2025, kept as an artifact of the original coursework. It is not reproducible
  and no result here depends on it.
- Clements and Preve also propose LAD regression, a quartic-root transformation
  and two further WLS weighting schemes. Only log-RV and the two nonparametric
  WLS schemes are implemented in the baseline chapter.
