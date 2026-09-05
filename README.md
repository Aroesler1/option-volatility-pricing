# Can alternative data improve HAR forecasts of realized variance, and does it pay?

The hypothesis this repository tests: **sentiment, attention, news tone and the
option market carry information about future S&P 500 realized variance that the
volatility path itself does not, the gain is concentrated at particular
horizons, and if it is real it should show up as money in a strategy that bets
on volatility.**

Three horizons (1, 5 and 21 trading days), one out-of-sample protocol, one loss
function, and an economic test at the end. Everything that failed is reported
next to everything that worked.

## The answer, in four lines

1. **Alternative data helps, and only at the short horizon.** At 1 day the 90%
   Model Confidence Set holds five models and plain HAR is not one of them. At
   21 days nothing beats HAR, HAR itself is back in the set, and the MCS cannot
   separate six of the nine models from each other.
2. **Almost all of the gain is the option market, not sentiment.** Implied
   variance on its own captures more than half of the 1-day improvement that
   the full sixteen-feature LASSO achieves. Added to HAR one at a time, both
   GDELT tone series are significantly WORSE than HAR at 1 day, and Wikipedia
   attention and policy uncertainty are indistinguishable from it. Measured
   against a benchmark that already contains semivariance and implied variance,
   **not one of the sixteen features is significantly better and three are
   significantly worse.**
3. **The improvement the literature was most confident about does not
   replicate.** Semivariance HAR fails to beat HAR at every horizon, even though
   the sign asymmetry it is built on is unmistakably present in its
   coefficients.
4. **None of it became money.** The volatility-managed strategies edge past
   buy-and-hold at 21 days by margins nowhere near significance, and every
   straddle book loses, including an unconditional short-variance book that
   should have harvested a risk premium worth a Sharpe of 2.7 in swap space.

## What the literature says to expect

The review is in [`docs/literature_alt_data.md`](docs/literature_alt_data.md),
which records for each paper the data, the specification, the horizon at which
gains were found and whether any P&L test was run. The predictions this study
was set up to check:

| expectation | source | horizon |
|---|---|---|
| Splitting realized variance by the SIGN of the intraday return is the most reliable improvement, and needs no external data at all | Patton and Sheppard (REStat 2015): SPY R2 rises 0.532 to 0.611 at h=1 and 0.282 to 0.313 at h=66 | all |
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

## Statistical results

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

<!-- RESULTS:MARGINAL -->

#### Horizon 1 day

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| atm_ivar_30 | option | 0.1953 | -0.0142 | -3.2499 | 0.0012 |
| term_slope_30_91 | option | 0.1988 | -0.0107 | -1.8357 | 0.0664 |
| atm_iv_30 | option | 0.2044 | -0.0051 | -0.7214 | 0.4707 |
| epu_log | uncertainty | 0.2062 | -0.0033 | -1.5447 | 0.1224 |
| is_payrolls | calendar | 0.2076 | -0.0019 | -1.1312 | 0.2580 |
| is_fomc | calendar | 0.2090 | -0.0005 | -0.6388 | 0.5230 |
| har | baseline | 0.2095 | 0.0000 |  |  |
| vix_vol | option | 0.2095 | 0.0000 | 0.0047 | 0.9962 |
| emv_overall | uncertainty | 0.2097 | 0.0002 | 0.3854 | 0.7000 |
| is_cpi | calendar | 0.2101 | 0.0007 | 0.5705 | 0.5683 |
| gdelt_share_econ | news | 0.2103 | 0.0008 | 2.1727 | 0.0298 |
| wiki_attention | attention | 0.2107 | 0.0012 | 0.4934 | 0.6218 |
| gdelt_share_mkt | news | 0.2113 | 0.0018 | 1.3562 | 0.1750 |
| spy_put_call | option | 0.2153 | 0.0059 | 3.4618 | 0.0005 |
| skew_25d_30 | option | 0.2206 | 0.0111 | 3.3716 | 0.0007 |
| gdelt_tone_econ | news | 0.2216 | 0.0121 | 3.9954 | 0.0001 |
| gdelt_tone_mkt | news | 0.2372 | 0.0277 | 4.5138 | 0.0000 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| atm_ivar_30 | option | 0.2033 | -0.0057 | -0.6192 | 0.5358 |
| term_slope_30_91 | option | 0.2038 | -0.0052 | -0.5145 | 0.6069 |
| epu_log | uncertainty | 0.2069 | -0.0022 | -0.7929 | 0.4279 |
| is_fomc | calendar | 0.2073 | -0.0017 | -1.6099 | 0.1074 |
| vix_vol | option | 0.2075 | -0.0015 | -0.1847 | 0.8535 |
| is_payrolls | calendar | 0.2084 | -0.0006 | -2.4578 | 0.0140 |
| is_cpi | calendar | 0.2087 | -0.0004 | -0.5931 | 0.5531 |
| atm_iv_30 | option | 0.2089 | -0.0001 | -0.0138 | 0.9890 |
| har | baseline | 0.2090 | 0.0000 |  |  |
| wiki_attention | attention | 0.2095 | 0.0005 | 0.4956 | 0.6202 |
| emv_overall | uncertainty | 0.2101 | 0.0011 | 0.6336 | 0.5263 |
| gdelt_share_mkt | news | 0.2106 | 0.0015 | 1.6386 | 0.1013 |
| spy_put_call | option | 0.2109 | 0.0018 | 1.4772 | 0.1396 |
| gdelt_share_econ | news | 0.2133 | 0.0042 | 1.5757 | 0.1151 |
| skew_25d_30 | option | 0.2230 | 0.0140 | 2.1559 | 0.0311 |
| gdelt_tone_econ | news | 0.2325 | 0.0234 | 2.4148 | 0.0157 |
| gdelt_tone_mkt | news | 0.2418 | 0.0328 | 2.7300 | 0.0063 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| emv_overall | uncertainty | 0.2438 | -0.0013 | -0.3013 | 0.7632 |
| wiki_attention | attention | 0.2447 | -0.0005 | -0.5925 | 0.5535 |
| is_payrolls | calendar | 0.2448 | -0.0003 | -1.2738 | 0.2027 |
| is_cpi | calendar | 0.2452 | -0.0000 | -0.0904 | 0.9280 |
| har | baseline | 0.2452 | 0.0000 |  |  |
| is_fomc | calendar | 0.2454 | 0.0002 | 0.0896 | 0.9286 |
| atm_ivar_30 | option | 0.2485 | 0.0033 | 0.7209 | 0.4710 |
| vix_vol | option | 0.2493 | 0.0041 | 0.2574 | 0.7969 |
| atm_iv_30 | option | 0.2501 | 0.0049 | 0.2967 | 0.7667 |
| spy_put_call | option | 0.2513 | 0.0062 | 0.9939 | 0.3203 |
| gdelt_share_mkt | news | 0.2520 | 0.0068 | 1.6908 | 0.0909 |
| term_slope_30_91 | option | 0.2541 | 0.0090 | 0.8787 | 0.3796 |
| skew_25d_30 | option | 0.2549 | 0.0097 | 0.6513 | 0.5149 |
| gdelt_share_econ | news | 0.2553 | 0.0101 | 0.8203 | 0.4120 |
| epu_log | uncertainty | 0.2589 | 0.0137 | 2.5369 | 0.0112 |
| gdelt_tone_econ | news | 0.2649 | 0.0197 | 0.7533 | 0.4513 |
| gdelt_tone_mkt | news | 0.3136 | 0.0684 | 1.3323 | 0.1827 |

<!-- END:MARGINAL -->

<!-- RESULTS:MARGINAL_RICH -->

#### Horizon 1 day

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.1904 | -0.0043 | -1.1699 | 0.2420 |
| gdelt_share_econ | news | 0.1938 | -0.0009 | -1.3156 | 0.1883 |
| is_payrolls | calendar | 0.1939 | -0.0008 | -0.7193 | 0.4720 |
| is_fomc | calendar | 0.1945 | -0.0002 | -0.2778 | 0.7811 |
| emv_overall | uncertainty | 0.1946 | -0.0001 | -0.1158 | 0.9078 |
| skew_25d_30 | option | 0.1946 | -0.0001 | -0.2560 | 0.7979 |
| har_rs_iv | baseline | 0.1947 | 0.0000 |  |  |
| is_cpi | calendar | 0.1954 | 0.0007 | 0.5633 | 0.5732 |
| wiki_attention | attention | 0.1956 | 0.0009 | 0.2906 | 0.7714 |
| epu_log | uncertainty | 0.1958 | 0.0011 | 1.8052 | 0.0710 |
| gdelt_share_mkt | news | 0.1979 | 0.0032 | 2.1110 | 0.0348 |
| vix_vol | option | 0.1991 | 0.0044 | 0.9952 | 0.3196 |
| spy_put_call | option | 0.1992 | 0.0045 | 3.1481 | 0.0016 |
| gdelt_tone_econ | news | 0.1993 | 0.0046 | 2.7780 | 0.0055 |
| atm_iv_30 | option | 0.1998 | 0.0051 | 1.0400 | 0.2983 |
| gdelt_tone_mkt | news | 0.2120 | 0.0173 | 3.6601 | 0.0003 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.2021 | -0.0020 | -0.2563 | 0.7977 |
| is_fomc | calendar | 0.2029 | -0.0012 | -1.2684 | 0.2046 |
| vix_vol | option | 0.2032 | -0.0009 | -0.1454 | 0.8844 |
| is_cpi | calendar | 0.2037 | -0.0004 | -1.2265 | 0.2200 |
| har_rs_iv | baseline | 0.2041 | 0.0000 |  |  |
| is_payrolls | calendar | 0.2042 | 0.0001 | 0.3373 | 0.7359 |
| wiki_attention | attention | 0.2044 | 0.0003 | 0.2534 | 0.7999 |
| spy_put_call | option | 0.2054 | 0.0013 | 1.2091 | 0.2266 |
| gdelt_share_econ | news | 0.2058 | 0.0017 | 1.7060 | 0.0880 |
| gdelt_share_mkt | news | 0.2058 | 0.0017 | 1.5350 | 0.1248 |
| atm_iv_30 | option | 0.2060 | 0.0019 | 0.2523 | 0.8008 |
| emv_overall | uncertainty | 0.2062 | 0.0021 | 1.5928 | 0.1112 |
| epu_log | uncertainty | 0.2070 | 0.0029 | 2.9618 | 0.0031 |
| skew_25d_30 | option | 0.2071 | 0.0030 | 1.5824 | 0.1136 |
| gdelt_tone_econ | news | 0.2150 | 0.0109 | 1.8530 | 0.0639 |
| gdelt_tone_mkt | news | 0.2278 | 0.0237 | 2.1622 | 0.0306 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| wiki_attention | attention | 0.2478 | -0.0008 | -0.8408 | 0.4004 |
| is_payrolls | calendar | 0.2482 | -0.0003 | -1.1955 | 0.2319 |
| emv_overall | uncertainty | 0.2482 | -0.0003 | -0.0657 | 0.9476 |
| har_rs_iv | baseline | 0.2485 | 0.0000 |  |  |
| is_cpi | calendar | 0.2485 | 0.0000 | 0.1053 | 0.9161 |
| is_fomc | calendar | 0.2487 | 0.0002 | 0.0945 | 0.9247 |
| vix_vol | option | 0.2492 | 0.0007 | 0.0533 | 0.9575 |
| atm_iv_30 | option | 0.2499 | 0.0013 | 0.1050 | 0.9163 |
| spy_put_call | option | 0.2548 | 0.0062 | 1.0128 | 0.3111 |
| gdelt_share_mkt | news | 0.2555 | 0.0070 | 1.6241 | 0.1044 |
| skew_25d_30 | option | 0.2573 | 0.0088 | 0.7581 | 0.4484 |
| gdelt_share_econ | news | 0.2578 | 0.0093 | 0.8550 | 0.3925 |
| term_slope_30_91 | option | 0.2589 | 0.0103 | 1.0286 | 0.3037 |
| epu_log | uncertainty | 0.2655 | 0.0170 | 2.1909 | 0.0285 |
| gdelt_tone_econ | news | 0.2664 | 0.0179 | 0.7227 | 0.4699 |
| gdelt_tone_mkt | news | 0.3157 | 0.0672 | 1.3195 | 0.1870 |

<!-- END:MARGINAL_RICH -->

The literature review's methodological point was right, and it is decisive.
Against bare HAR at 1 day, implied variance is worth 0.014 of QLIKE and the term
slope another 0.011. Against a base that already contains semivariance and
implied variance, **not one of the sixteen features is significantly better**,
and three are significantly worse. Every claim in this study that alternative
data adds something is a claim about adding it to HAR, not about adding it to a
model that already knows what the option market is pricing.

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

The Patton-Sheppard asymmetry is a claim about coefficients, and on this sample
it holds cleanly: volatility arriving on down moves loads far more heavily on
future volatility than volatility arriving on up moves, at every horizon. The
model built on it still fails to beat HAR out of sample, at every horizon.

Those two facts are consistent. The two semivolatilities are 0.92 correlated
with each other and 0.98 with realized volatility, so splitting the daily term
buys a second parameter with almost no independent variation behind it, and the
estimation noise costs more than the asymmetry pays. This is the clearest
example in the study of an effect that is real, well documented, visible in the
coefficients, and worthless as a forecast.

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
because nothing was significant there to begin with. Read together: **every
statistically significant result in this study depends on using option data from
the close it is forecasting from.** That is a legitimate information set and it
is also the whole result, so it is stated here rather than in a footnote.

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
| buy_and_hold |  | 1.2588 | 0.2101 | 0.1669 | -0.2023 |  |
| lstm | 0.2286 | 1.2568 | 0.2536 | 0.2017 | -0.2492 | 26.8163 |
| lstm_x | 0.1873 | 1.2492 | 0.2512 | 0.2011 | -0.2312 | 28.5267 |
| combination | 0.1871 | 1.2328 | 0.2464 | 0.1999 | -0.2352 | 32.4567 |
| har_x_lasso | 0.1841 | 1.2299 | 0.2456 | 0.1997 | -0.2423 | 30.3772 |
| har_rv_iv | 0.1953 | 1.2293 | 0.2507 | 0.2039 | -0.2462 | 36.2102 |
| har | 0.2095 | 1.2215 | 0.2465 | 0.2018 | -0.2392 | 39.8249 |
| shar | 0.2102 | 1.2146 | 0.2444 | 0.2012 | -0.2377 | 42.2129 |
| hgb | 0.1945 | 1.2005 | 0.2377 | 0.1980 | -0.2337 | 35.6330 |
| persistence | 0.2783 | 1.1627 | 0.2387 | 0.2053 | -0.2412 | 61.7334 |

QLIKE winner har_x_lasso, Sharpe winner lstm; Spearman(QLIKE, Sharpe) = -0.367 (p = 0.332)

#### Horizon 5

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| har_x_lasso | 0.1866 | 1.2805 | 0.2505 | 0.1956 | -0.2296 | 27.5989 |
| buy_and_hold |  | 1.2588 | 0.2101 | 0.1669 | -0.2023 |  |
| lstm_x | 0.2494 | 1.2508 | 0.2522 | 0.2016 | -0.2608 | 17.5289 |
| combination | 0.1948 | 1.2365 | 0.2388 | 0.1931 | -0.2362 | 26.7172 |
| har_rv_iv | 0.2033 | 1.2307 | 0.2394 | 0.1945 | -0.2401 | 32.5032 |
| lstm | 0.2430 | 1.2172 | 0.2370 | 0.1947 | -0.2517 | 15.2106 |
| har | 0.2090 | 1.2085 | 0.2335 | 0.1932 | -0.2356 | 37.7296 |
| hgb | 0.1847 | 1.2065 | 0.2316 | 0.1919 | -0.2247 | 31.9187 |
| shar | 0.2099 | 1.2036 | 0.2324 | 0.1931 | -0.2351 | 38.5769 |
| persistence | 0.3530 | 1.1627 | 0.2387 | 0.2053 | -0.2412 | 61.7334 |

QLIKE winner hgb, Sharpe winner har_x_lasso; Spearman(QLIKE, Sharpe) = -0.317 (p = 0.406)

#### Horizon 21

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| lstm_x | 0.3142 | 1.3586 | 0.2754 | 0.2027 | -0.2547 | 12.3101 |
| combination | 0.2332 | 1.3106 | 0.2424 | 0.1850 | -0.2233 | 16.9361 |
| hgb | 0.2278 | 1.3071 | 0.2499 | 0.1912 | -0.2261 | 20.4999 |
| har_x_lasso | 0.2530 | 1.2925 | 0.2526 | 0.1954 | -0.2277 | 27.5481 |
| har_rv_iv | 0.2485 | 1.2882 | 0.2312 | 0.1794 | -0.2207 | 23.2805 |
| har | 0.2452 | 1.2821 | 0.2297 | 0.1791 | -0.2211 | 24.6104 |
| shar | 0.2453 | 1.2777 | 0.2291 | 0.1793 | -0.2211 | 24.3932 |
| lstm | 0.2641 | 1.2686 | 0.2405 | 0.1896 | -0.2424 | 9.0299 |
| buy_and_hold |  | 1.2588 | 0.2101 | 0.1669 | -0.2023 |  |
| persistence | 0.5811 | 1.1627 | 0.2387 | 0.2053 | -0.2412 | 61.7334 |

QLIKE winner hgb, Sharpe winner lstm_x; Spearman(QLIKE, Sharpe) = -0.367 (p = 0.332)

<!-- END:VOLMANAGED -->

**Nothing here is significant.** At 21 days eight of the nine models edge past
buy-and-hold, the best by 0.10 of a Sharpe point, and the paired block bootstrap
puts that at p = 0.83. At 1 and 5 days buy-and-hold is ahead, also
insignificantly. Volatility timing neither paid nor cost anything measurable in
this sample, which was a strong bull market with two brief volatility spikes,
and that is the honest scope of the result rather than a general claim.

The rank correlation says the same thing more precisely. Spearman between QLIKE
and Sharpe across models is -0.37, -0.32 and -0.37 at 1, 5 and 21 days, all with
p above 0.3. **A better volatility forecast tilted the P&L ranking in the right
direction and not by enough to distinguish from noise.**

### Delta-hedged straddles

At each forecast date the 21-day forecast, with the variance risk premium added
back, is compared to 30-day at-the-money implied variance: long a straddle when
the forecast is higher by a frozen margin, short when lower, flat otherwise.
Both the premium adjustment and the margin are estimated on the first half of
the out-of-sample window and then frozen. Entry is at the touch and exit is at
the touch, so a round trip pays the full quoted spread, and the SPY hedge is
charged 5 bps on the shares it trades.

Two unconditional books are included that use no forecast at all, because they
are what decides whether any of the others had a chance.

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
| har | -0.6690 | -0.2560 | -1.0391 | -0.2432 | 0.3143 | 3.5000 | 210 |
| shar | -0.7521 | -0.2853 | -1.0774 | -0.2519 | 0.3019 | 3.5333 | 212 |
| persistence | -0.7888 | -0.2416 | -0.9131 | -0.1515 | 0.3030 | 3.2727 | 198 |
| lstm_x | -0.8833 | -0.4536 | -1.3388 | -0.2621 | 0.2699 | 4.7769 | 289 |
| lstm | -0.9292 | -0.3620 | -1.1062 | -0.2583 | 0.2876 | 3.8512 | 233 |
| always_long | -1.1742 | -1.0433 | -3.7604 | -0.3153 | 0.2378 | 11.8843 | 719 |

<!-- END:STRADDLES -->

**Every book loses.** That the forecast books beat both unconditional books is
the result. The best model book loses 26.8% a year against the always-short
book's 66.3% and the always-long book's 104.3%, a reduction of about 60%, and
part of how it gets there is by trading less: 5 to 6 times the premium budget a
year against 11.9 for a book that opens a position every day. The forecasts are
worth something. They are worth less than the cost of expressing them.

The rank correlation is stronger here than in volatility timing, and this is the
strategy whose 21-day holding period matches the forecast horizon: Spearman
between QLIKE and Sharpe is -0.55 (p = 0.125) across the two-sided books and
-0.68 (p = 0.042) across the long-only ones.

### The model-free check

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

A synthetic variance swap struck at VIX squared, long realized variance, loses
0.0188 variance points per 21 days, with a Newey-West t of -7.0 on the
overlapping series and -4.7 on 35 non-overlapping observations. Realized
variance came in under VIX squared on 95% of days. The short side of that trade
has an annualized Sharpe of 2.73 and needs no forecast at all.

Put the two together and the shape of the answer is clear. **The variance risk
premium in this sample was large, real, and completely consumed by the cost of
expressing it in listed options.** A variance swap harvests realized variance
uniformly and, as reported here, costlessly; a 21-day delta-hedged straddle pays
the full quoted spread twice, hedges discretely at 5 bps, and has its gamma
concentrated near one strike. The gap between a Sharpe of +2.7 and a Sharpe of
-0.72 is entirely implementation.

### One sentence per horizon

- **1 day.** The statistical gain is the largest and the only one that pushes
  HAR out of the Model Confidence Set, and no strategy in this study trades at
  that horizon, so it converted into nothing at all.
- **5 days.** A real but smaller statistical gain, no strategy at that horizon
  either, and the weakest rank transfer of the three.
- **21 days.** The smallest statistical gain, the only horizon with strategies
  attached, and the clearest rank transfer into P&L, which still leaves every
  strategy losing money or indistinguishable from doing nothing.

## What did not work

Reported at the same volume as what did.

- **Semivariance HAR.** Fails at every horizon despite its premise holding in
  the coefficients. The single most confident prediction in the literature
  review.
- **Sentiment, attention and uncertainty, one at a time.** Wikipedia attention,
  GDELT tone and coverage share, EPU and EMV are all within a hair of HAR on
  their own at every horizon, and several are worse. Nothing in the free
  alternative-data blocks earns its place next to the option market.
- **The LSTM.** Worse than HAR at every horizon, significantly so at 1 and 5
  days. Adding the features rescues it at 1 day, where it enters the MCS, and
  makes it the worst model in the study at 21.
- **Both neural models under stress.** On the 70 highest-volatility days out of
  720, the LSTM is 0.171 QLIKE worse than HAR and the feature-augmented LSTM
  0.190 worse, while both sit close to HAR on calm days. Whatever the networks
  learned, they lose it exactly when volatility forecasting matters.
- **Volatility timing.** Not distinguishable from buy-and-hold in either
  direction at any horizon.
- **Every straddle strategy**, including the unconditional ones.
- **GDELT news tone.** The worst single feature in the study. Added to HAR on
  its own at 1 day, stock-market tone is 0.028 QLIKE worse than HAR (p < 0.0001)
  and economy tone 0.012 worse (p = 0.0001). The free news block is not merely
  uninformative here; on its own it actively degrades the forecast.
- **25-delta skew**, significantly worse than HAR at 1 day (p = 0.0007).

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
WRDS_USERNAME=yourlogin python fetch_wrds_features.py
WRDS_USERNAME=yourlogin python fetch_option_chain.py
python fetch_alt_data.py

python run_altdata_benchmark.py
python run_option_pnl.py
python run_altdata_benchmark.py --extra-lag 1
python report_tables.py
```

WRDS credentials come from `~/.pgpass` and the intraday extract from
`DATABENTO_RAW_DIR`. No credential is stored in this repository and no test
reads one. GDELT's API answers roughly one uncached timeline query every several
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
| `fetch_wrds_features.py`, `fetch_option_chain.py`, `fetch_alt_data.py` | the data pulls |
| `run_altdata_benchmark.py` | the horse race at three horizons |
| `run_option_pnl.py` | the economic evaluation |
| `report_tables.py` | prints this README's tables from `results/` |
| `run_vol_benchmark.py`, `run_intraday_benchmark.py`, `run_iv_benchmark.py` | the baseline chapter |
| `docs/literature_alt_data.md` | the literature review, with the horizon of each paper's result |
| `tests/` | the suite, offline, no credentials |
| `Main.ipynb`, `report.pdf` | the original coursework, retained as a historical record |

## Known limits

- **One asset, one sample, one regime.** The out-of-sample window contained a
  strong bull market and two brief volatility spikes. The volatility-timing
  result in particular would look different in a sample containing a crash.
- **The 1-day result depends on the timing convention.** One extra day of
  publication lag removes most of it. Both numbers are reported above.
- **The MCS has limited power here.** At 21 days it cannot separate eight of
  nine models. A cross-section of assets is what would settle that, and it is
  the natural next step: pooled HAR across assets (Bollerslev, Hood, Huss and
  Pedersen, RFS 2018) estimated on TAQ.
- **The straddle book holds one at-the-money straddle per signal.** It is not a
  variance swap and does not try to be; the gap between the two is measured
  above rather than assumed away.
- **The variance swap is reported with no transaction cost at all**, which is
  why it is labelled the model-free check rather than a tradeable strategy.
- **`data/sentiment_prev_full_month.json`** is a one-month Reddit crawl from
  2025, kept as an artifact of the original coursework. It is not reproducible
  and no result here depends on it.
- Clements and Preve also propose LAD regression, a quartic-root transformation
  and two further WLS weighting schemes. Only log-RV and the two nonparametric
  WLS schemes are implemented in the baseline chapter.
