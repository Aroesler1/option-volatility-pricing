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
   Model Confidence Set contains four models and plain HAR is not one of them.
   At 21 days nothing beats HAR and eight of nine models survive the MCS.
2. **Almost all of the gain is the option market, not sentiment.** Implied
   variance alone accounts for most of the 1-day improvement. News tone,
   Wikipedia attention and policy uncertainty contribute essentially nothing on
   their own at any horizon.
3. **The improvement the literature was most confident about does not
   replicate.** Semivariance HAR fails to beat HAR at every horizon, even though
   the sign asymmetry it is built on is unmistakably present in its
   coefficients.
4. **None of it became money.** No volatility-managed strategy beat buying and
   holding SPY, and every straddle book lost, including an unconditional
   short-variance book that should have harvested a risk premium worth a Sharpe
   of 2.4 in swap space.

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
Fifteen features in five blocks:

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
| HAR-X, one feature at a time | fifteen models, so each feature's marginal value is attributable |
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

Regenerate every table below with `python report_tables.py`.

<!-- RESULTS:MODELS -->

#### Horizon 1 day

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| combination | 0.1878 | 0.0801 | -4.6258 | 0.0000 | 1.0000 | yes |
| har_x_lasso | 0.1884 | 0.0860 | -2.9155 | 0.0036 | 0.8980 | yes |
| lstm_x | 0.1923 | 0.0758 | -1.6187 | 0.1055 | 0.8575 | yes |
| hgb | 0.1928 | 0.0902 | -1.6395 | 0.1011 | 0.8575 | yes |
| har_rv_iv | 0.1965 | 0.0803 | -3.1151 | 0.0018 | 0.1010 | yes |
| har | 0.2102 | 0.0835 |  |  | 0.0015 | no |
| shar | 0.2110 | 0.0862 | 0.3885 | 0.6976 | 0.0000 | no |
| lstm | 0.2194 | 0.0802 | 1.4548 | 0.1457 | 0.0055 | no |
| persistence | 0.2785 | 0.1043 | 5.4617 | 0.0000 | 0.0000 | no |

MCS(90%) = {combination, har_x_lasso, lstm_x, hgb, har_rv_iv}

#### Horizon 5 days

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| har_x_lasso | 0.1872 | 0.0564 | -2.3671 | 0.0179 | 1.0000 | yes |
| hgb | 0.1888 | 0.0603 | -1.1613 | 0.2455 | 0.8815 | yes |
| combination | 0.1949 | 0.0596 | -1.9326 | 0.0533 | 0.5090 | yes |
| har_rv_iv | 0.2032 | 0.0577 | -0.5102 | 0.6099 | 0.4010 | yes |
| har | 0.2080 | 0.0596 |  |  | 0.1295 | yes |
| shar | 0.2089 | 0.0624 | 1.3825 | 0.1668 | 0.1295 | yes |
| lstm | 0.2401 | 0.0711 | 1.8879 | 0.0590 | 0.0850 | no |
| lstm_x | 0.2488 | 0.0609 | 1.2243 | 0.2208 | 0.2540 | yes |
| persistence | 0.3517 | 0.0954 | 5.1279 | 0.0000 | 0.0000 | no |

MCS(90%) = {har_x_lasso, hgb, combination, har_rv_iv, har, shar, lstm_x}

#### Horizon 21 days

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| hgb | 0.2193 | 0.0571 | -1.0591 | 0.2895 | 1.0000 | yes |
| combination | 0.2313 | 0.0857 | -1.5179 | 0.1290 | 0.4570 | yes |
| har | 0.2453 | 0.1047 |  |  | 0.2850 | yes |
| shar | 0.2454 | 0.1050 | 0.3468 | 0.7288 | 0.2850 | yes |
| har_x_lasso | 0.2464 | 0.0950 | 0.0795 | 0.9367 | 0.2850 | yes |
| har_rv_iv | 0.2490 | 0.1082 | 0.7917 | 0.4285 | 0.2850 | yes |
| lstm | 0.2511 | 0.0885 | 0.3789 | 0.7048 | 0.2850 | yes |
| lstm_x | 0.2783 | 0.0717 | 1.0503 | 0.2936 | 0.2715 | yes |
| persistence | 0.5800 | 0.1216 | 3.6988 | 0.0002 | 0.0020 | no |

MCS(90%) = {hgb, combination, har, shar, har_x_lasso, har_rv_iv, lstm, lstm_x}

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
| hgb | 100% |
| combination | 100% |
| lstm_x | 96% |
| har_rv_iv | 87% |
| har | 39% |
| shar | 39% |
| lstm | 39% |
| persistence | 22% |

#### Horizon 1, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| lstm_x | 100% |
| combination | 100% |
| hgb | 91% |
| har_rv_iv | 73% |
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
| lstm | 91% |
| har_rv_iv | 87% |
| hgb | 83% |
| shar | 48% |
| har | 43% |
| persistence | 22% |

#### Horizon 5, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_rv_iv | 100% |
| har_x_lasso | 100% |
| lstm_x | 100% |
| combination | 100% |
| hgb | 91% |
| lstm | 91% |
| har | 64% |
| shar | 64% |
| persistence | 0% |

#### Horizon 21, 252-observation windows (23 windows)

| model | share of windows in the 90% MCS |
|---|---|
| hgb | 100% |
| lstm_x | 83% |
| har_rv_iv | 61% |
| har_x_lasso | 61% |
| lstm | 61% |
| combination | 61% |
| har | 43% |
| shar | 43% |
| persistence | 0% |

#### Horizon 21, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| hgb | 100% |
| lstm | 100% |
| lstm_x | 100% |
| combination | 100% |
| har_x_lasso | 45% |
| har | 36% |
| shar | 36% |
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
| atm_ivar_30 | option | 0.1965 | -0.0137 | -3.1151 | 0.0018 |
| term_slope_30_91 | option | 0.2000 | -0.0102 | -1.7648 | 0.0776 |
| atm_iv_30 | option | 0.2058 | -0.0044 | -0.6156 | 0.5382 |
| epu_log | uncertainty | 0.2072 | -0.0030 | -1.4490 | 0.1473 |
| is_payrolls | calendar | 0.2084 | -0.0018 | -1.0870 | 0.2770 |
| is_fomc | calendar | 0.2097 | -0.0005 | -0.6324 | 0.5271 |
| har | baseline | 0.2102 | 0.0000 |  |  |
| emv_overall | uncertainty | 0.2104 | 0.0002 | 0.6677 | 0.5043 |
| vix_vol | option | 0.2107 | 0.0005 | 0.0689 | 0.9451 |
| is_cpi | calendar | 0.2108 | 0.0006 | 0.5202 | 0.6029 |
| wiki_attention | attention | 0.2115 | 0.0013 | 0.5324 | 0.5944 |
| spy_put_call | option | 0.2159 | 0.0057 | 3.3762 | 0.0007 |
| skew_25d_30 | option | 0.2213 | 0.0111 | 3.3683 | 0.0008 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| atm_ivar_30 | option | 0.2032 | -0.0047 | -0.5102 | 0.6099 |
| term_slope_30_91 | option | 0.2038 | -0.0042 | -0.4160 | 0.6774 |
| epu_log | uncertainty | 0.2063 | -0.0016 | -0.6796 | 0.4968 |
| is_fomc | calendar | 0.2064 | -0.0016 | -1.5365 | 0.1244 |
| is_payrolls | calendar | 0.2073 | -0.0007 | -2.5466 | 0.0109 |
| vix_vol | option | 0.2076 | -0.0003 | -0.0415 | 0.9669 |
| is_cpi | calendar | 0.2077 | -0.0003 | -0.4445 | 0.6567 |
| har | baseline | 0.2080 | 0.0000 |  |  |
| wiki_attention | attention | 0.2085 | 0.0006 | 0.5752 | 0.5651 |
| emv_overall | uncertainty | 0.2086 | 0.0006 | 0.4822 | 0.6296 |
| spy_put_call | option | 0.2096 | 0.0017 | 1.3093 | 0.1904 |
| atm_iv_30 | option | 0.2096 | 0.0017 | 0.1556 | 0.8764 |
| skew_25d_30 | option | 0.2212 | 0.0132 | 2.0543 | 0.0399 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| emv_overall | uncertainty | 0.2435 | -0.0019 | -0.4086 | 0.6828 |
| wiki_attention | attention | 0.2446 | -0.0007 | -0.8399 | 0.4010 |
| is_payrolls | calendar | 0.2450 | -0.0003 | -1.2122 | 0.2254 |
| har | baseline | 0.2453 | 0.0000 |  |  |
| is_cpi | calendar | 0.2454 | 0.0001 | 1.2034 | 0.2288 |
| is_fomc | calendar | 0.2456 | 0.0002 | 0.1233 | 0.9019 |
| atm_ivar_30 | option | 0.2490 | 0.0036 | 0.7917 | 0.4285 |
| vix_vol | option | 0.2501 | 0.0048 | 0.3038 | 0.7613 |
| spy_put_call | option | 0.2511 | 0.0058 | 0.9640 | 0.3350 |
| atm_iv_30 | option | 0.2513 | 0.0059 | 0.3614 | 0.7178 |
| skew_25d_30 | option | 0.2545 | 0.0091 | 0.6166 | 0.5375 |
| term_slope_30_91 | option | 0.2550 | 0.0096 | 0.9410 | 0.3467 |
| epu_log | uncertainty | 0.2587 | 0.0133 | 2.6600 | 0.0078 |

<!-- END:MARGINAL -->

<!-- RESULTS:MARGINAL_RICH -->

#### Horizon 1 day

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.1920 | -0.0039 | -1.0809 | 0.2797 |
| is_payrolls | calendar | 0.1951 | -0.0008 | -0.7295 | 0.4657 |
| is_fomc | calendar | 0.1957 | -0.0002 | -0.2783 | 0.7808 |
| skew_25d_30 | option | 0.1958 | -0.0000 | -0.1717 | 0.8637 |
| har_rs_iv | baseline | 0.1959 | 0.0000 |  |  |
| emv_overall | uncertainty | 0.1959 | 0.0000 | 0.0477 | 0.9619 |
| is_cpi | calendar | 0.1964 | 0.0006 | 0.5184 | 0.6042 |
| wiki_attention | attention | 0.1966 | 0.0007 | 0.2504 | 0.8023 |
| epu_log | uncertainty | 0.1973 | 0.0014 | 2.2507 | 0.0244 |
| spy_put_call | option | 0.2002 | 0.0044 | 3.0325 | 0.0024 |
| vix_vol | option | 0.2005 | 0.0046 | 1.0410 | 0.2979 |
| atm_iv_30 | option | 0.2013 | 0.0054 | 1.1149 | 0.2649 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.2027 | -0.0014 | -0.1818 | 0.8557 |
| is_fomc | calendar | 0.2029 | -0.0011 | -1.2100 | 0.2263 |
| vix_vol | option | 0.2037 | -0.0004 | -0.0649 | 0.9483 |
| is_cpi | calendar | 0.2037 | -0.0004 | -1.0411 | 0.2979 |
| har_rs_iv | baseline | 0.2041 | 0.0000 |  |  |
| is_payrolls | calendar | 0.2041 | 0.0000 | 0.2725 | 0.7853 |
| wiki_attention | attention | 0.2044 | 0.0003 | 0.2499 | 0.8026 |
| spy_put_call | option | 0.2052 | 0.0011 | 1.0433 | 0.2968 |
| emv_overall | uncertainty | 0.2060 | 0.0019 | 1.6545 | 0.0980 |
| skew_25d_30 | option | 0.2068 | 0.0027 | 1.4396 | 0.1500 |
| atm_iv_30 | option | 0.2068 | 0.0028 | 0.3774 | 0.7059 |
| epu_log | uncertainty | 0.2074 | 0.0033 | 3.0013 | 0.0027 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| wiki_attention | attention | 0.2480 | -0.0010 | -1.0750 | 0.2824 |
| emv_overall | uncertainty | 0.2481 | -0.0009 | -0.1794 | 0.8576 |
| is_payrolls | calendar | 0.2487 | -0.0003 | -1.1643 | 0.2443 |
| har_rs_iv | baseline | 0.2490 | 0.0000 |  |  |
| is_cpi | calendar | 0.2491 | 0.0001 | 1.9136 | 0.0557 |
| is_fomc | calendar | 0.2492 | 0.0002 | 0.1134 | 0.9097 |
| vix_vol | option | 0.2500 | 0.0010 | 0.0855 | 0.9318 |
| atm_iv_30 | option | 0.2510 | 0.0020 | 0.1598 | 0.8730 |
| spy_put_call | option | 0.2549 | 0.0059 | 0.9839 | 0.3252 |
| skew_25d_30 | option | 0.2568 | 0.0078 | 0.6793 | 0.4969 |
| term_slope_30_91 | option | 0.2598 | 0.0108 | 1.0724 | 0.2835 |
| epu_log | uncertainty | 0.2655 | 0.0165 | 2.2842 | 0.0224 |

<!-- END:MARGINAL_RICH -->

### Calm days against stressed days

The news and embedding literature is consistent that text signals earn their
keep on high-volatility days and contribute nothing the rest of the time, so a
pooled average can hide the whole effect. Days are split by whether realized
volatility exceeds the 80th percentile of every realized volatility observed
strictly before them, which is knowable in real time.

<!-- RESULTS:REGIME -->

#### Horizon 1 day (68 stressed of 723 days)

| model | QLIKE calm | QLIKE stressed | calm vs HAR | stressed vs HAR |
|---|---|---|---|---|
| lstm_x | 0.1852 | 0.2615 | -0.0070 | -0.1228 |
| hgb | 0.1825 | 0.2913 | -0.0096 | -0.0929 |
| combination | 0.1739 | 0.3222 | -0.0182 | -0.0621 |
| har_x_lasso | 0.1733 | 0.3342 | -0.0188 | -0.0501 |
| persistence | 0.2711 | 0.3501 | 0.0789 | -0.0342 |
| har_rv_iv | 0.1803 | 0.3530 | -0.0118 | -0.0312 |
| shar | 0.1935 | 0.3790 | 0.0014 | -0.0053 |
| har | 0.1921 | 0.3843 | 0.0000 | 0.0000 |
| lstm | 0.1955 | 0.4498 | 0.0033 | 0.0655 |

#### Horizon 5 days (68 stressed of 723 days)

| model | QLIKE calm | QLIKE stressed | calm vs HAR | stressed vs HAR |
|---|---|---|---|---|
| hgb | 0.1482 | 0.5801 | -0.0088 | -0.1189 |
| har_x_lasso | 0.1398 | 0.6429 | -0.0171 | -0.0561 |
| persistence | 0.3205 | 0.6515 | 0.1635 | -0.0475 |
| har | 0.1570 | 0.6990 | 0.0000 | 0.0000 |
| shar | 0.1577 | 0.7018 | 0.0007 | 0.0027 |
| combination | 0.1416 | 0.7083 | -0.0154 | 0.0093 |
| har_rv_iv | 0.1500 | 0.7160 | -0.0070 | 0.0169 |
| lstm | 0.1647 | 0.9669 | 0.0077 | 0.2678 |
| lstm_x | 0.1618 | 1.0869 | 0.0048 | 0.3879 |

#### Horizon 21 days (68 stressed of 723 days)

| model | QLIKE calm | QLIKE stressed | calm vs HAR | stressed vs HAR |
|---|---|---|---|---|
| hgb | 0.2080 | 0.3274 | -0.0240 | -0.0463 |
| combination | 0.2172 | 0.3667 | -0.0148 | -0.0069 |
| har | 0.2320 | 0.3736 | 0.0000 | 0.0000 |
| shar | 0.2321 | 0.3743 | 0.0001 | 0.0007 |
| har_rv_iv | 0.2358 | 0.3754 | 0.0038 | 0.0017 |
| har_x_lasso | 0.2321 | 0.3840 | 0.0001 | 0.0103 |
| persistence | 0.5912 | 0.4725 | 0.3592 | 0.0989 |
| lstm | 0.2268 | 0.4853 | -0.0052 | 0.1117 |
| lstm_x | 0.2462 | 0.5877 | 0.0142 | 0.2141 |

<!-- END:REGIME -->

### The semivariance result, in full

<!-- RESULTS:SHAR -->

| horizon | b on RS+ | b on RS- | HAR b on RV | corr(RS+, RS-) |
|---|---|---|---|---|
| 1 | 0.1684 | 0.5703 | 0.5256 | 0.9187 |
| 5 | 0.2222 | 0.4380 | 0.4624 | 0.9187 |
| 21 | 0.1473 | 0.2687 | 0.2905 | 0.9187 |

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

| 1-day horizon | lag as stated | one more day |
|---|---|---|
| HAR-RV-IV vs HAR | p = **0.0021** | p = 0.216 |
| HAR-X LASSO vs HAR | p = **0.0053** | p = 0.376 |
| combination vs HAR | p = **0.0000** | p = 0.049 |
| HAR in the 90% MCS | **excluded** | included |

The stated convention is defensible: an OptionMetrics closing quote for date t
is known at the close of t, which is the same moment RV_t is known, and the
target starts at t+1. But the relationship is tight enough that one day of extra
caution removes most of it, and a reader is entitled to both numbers rather than
the flattering one. At 5 and 21 days the extra lag changes little, because
little was there to lose.

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
| buy_and_hold |  | 1.3981 | 0.2316 | 0.1656 | -0.2023 |  |
| lstm | 0.2194 | 1.3863 | 0.2795 | 0.2016 | -0.2401 | 25.6119 |
| combination | 0.1878 | 1.3407 | 0.2691 | 0.2007 | -0.2376 | 32.5050 |
| hgb | 0.1928 | 1.3396 | 0.2670 | 0.1993 | -0.2366 | 34.4612 |
| lstm_x | 0.1923 | 1.3372 | 0.2746 | 0.2054 | -0.2455 | 28.8918 |
| har_rv_iv | 0.1965 | 1.3244 | 0.2699 | 0.2038 | -0.2456 | 36.2135 |
| har | 0.2102 | 1.3193 | 0.2660 | 0.2016 | -0.2392 | 39.5698 |
| shar | 0.2110 | 1.3122 | 0.2638 | 0.2011 | -0.2376 | 41.9423 |
| har_x_lasso | 0.1884 | 1.2887 | 0.2568 | 0.1993 | -0.2404 | 32.6669 |
| persistence | 0.2785 | 1.2591 | 0.2583 | 0.2051 | -0.2412 | 61.4027 |

QLIKE winner combination, Sharpe winner lstm; Spearman(QLIKE, Sharpe) = -0.267 (p = 0.488)

#### Horizon 5

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| buy_and_hold |  | 1.3981 | 0.2316 | 0.1656 | -0.2023 |  |
| lstm_x | 0.2488 | 1.3957 | 0.2824 | 0.2023 | -0.2440 | 16.6274 |
| har_x_lasso | 0.1872 | 1.3453 | 0.2579 | 0.1917 | -0.2296 | 29.6729 |
| combination | 0.1949 | 1.3291 | 0.2561 | 0.1927 | -0.2359 | 26.8885 |
| har_rv_iv | 0.2032 | 1.3250 | 0.2575 | 0.1943 | -0.2397 | 32.4668 |
| lstm | 0.2401 | 1.3144 | 0.2566 | 0.1952 | -0.2534 | 15.0720 |
| har | 0.2080 | 1.3103 | 0.2529 | 0.1930 | -0.2357 | 37.6120 |
| shar | 0.2089 | 1.3041 | 0.2514 | 0.1928 | -0.2352 | 38.4433 |
| hgb | 0.1888 | 1.2755 | 0.2446 | 0.1918 | -0.2331 | 31.0206 |
| persistence | 0.3517 | 1.2591 | 0.2583 | 0.2051 | -0.2412 | 61.4027 |

QLIKE winner har_x_lasso, Sharpe winner lstm_x; Spearman(QLIKE, Sharpe) = -0.233 (p = 0.546)

#### Horizon 21

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| hgb | 0.2193 | 1.4505 | 0.2731 | 0.1883 | -0.2088 | 21.8098 |
| har_x_lasso | 0.2464 | 1.4053 | 0.2554 | 0.1818 | -0.2169 | 24.0653 |
| combination | 0.2313 | 1.4014 | 0.2540 | 0.1812 | -0.2185 | 16.6249 |
| buy_and_hold |  | 1.3981 | 0.2316 | 0.1656 | -0.2023 |  |
| lstm | 0.2511 | 1.3820 | 0.2565 | 0.1856 | -0.2347 | 8.9052 |
| har_rv_iv | 0.2490 | 1.3810 | 0.2474 | 0.1792 | -0.2208 | 23.2798 |
| har | 0.2453 | 1.3774 | 0.2464 | 0.1789 | -0.2212 | 24.6481 |
| shar | 0.2454 | 1.3731 | 0.2459 | 0.1791 | -0.2212 | 24.4055 |
| lstm_x | 0.2783 | 1.3396 | 0.2631 | 0.1964 | -0.2442 | 10.5354 |
| persistence | 0.5800 | 1.2591 | 0.2583 | 0.2051 | -0.2412 | 61.4027 |

QLIKE winner hgb, Sharpe winner hgb; Spearman(QLIKE, Sharpe) = -0.683 (p = 0.042)

<!-- END:VOLMANAGED -->

**Buying and holding SPY beat every model at every horizon**, and none of the
differences comes close to significance under a paired block bootstrap.
Volatility timing did not pay in this sample, which was a strong bull market
with two brief volatility spikes. That is the honest scope of the result rather
than a general claim.

The rank correlation is the more interesting number. Spearman between QLIKE and
Sharpe across models is about -0.6 at 21 days and about -0.2 at 1 and 5 days.
**The forecast ranking transfers into the P&L ranking when the forecast horizon
matches the rebalancing horizon, and barely transfers otherwise.**

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
| hgb | -0.2520 | -0.1323 | -0.9353 | -0.2582 | 0.4392 | 5.5934 | 337 |
| combination | -0.5538 | -0.2895 | -1.0532 | -0.4031 | 0.4156 | 5.3112 | 320 |
| har_rv_iv | -0.5979 | -0.2584 | -1.0064 | -0.3065 | 0.4116 | 4.8797 | 294 |
| persistence | -0.6163 | -0.2184 | -0.7887 | -0.1515 | 0.4451 | 5.4440 | 328 |
| shar | -0.6175 | -0.2642 | -0.9912 | -0.3020 | 0.4128 | 4.9461 | 298 |
| har | -0.6409 | -0.2736 | -1.0219 | -0.3049 | 0.4086 | 4.9959 | 301 |
| always_short | -0.6923 | -0.6390 | -3.3234 | -1.2436 | 0.5305 | 11.9834 | 722 |
| lstm | -0.7462 | -0.3822 | -1.2006 | -0.5653 | 0.4286 | 5.8091 | 350 |
| har_x_lasso | -0.8268 | -0.4697 | -1.4759 | -0.4875 | 0.3707 | 5.7759 | 348 |
| lstm_x | -1.1725 | -0.6689 | -2.0119 | -0.6031 | 0.3780 | 6.9378 | 418 |
| always_long | -1.2112 | -1.0796 | -3.7604 | -0.4191 | 0.2368 | 11.9834 | 722 |

#### long_only

| model | Sharpe | mean p.a. | max drawdown | worst month | trade hit rate | turnover p.a. | trades |
|---|---|---|---|---|---|---|---|
| hgb | -0.3559 | -0.1696 | -0.9559 | -0.2582 | 0.3765 | 4.0996 | 247 |
| combination | -0.4723 | -0.2116 | -1.0540 | -0.2523 | 0.3423 | 3.6846 | 222 |
| har_rv_iv | -0.6329 | -0.2216 | -0.9853 | -0.2354 | 0.3155 | 3.1037 | 187 |
| lstm | -0.6608 | -0.2794 | -1.0758 | -0.2803 | 0.3304 | 3.8174 | 230 |
| shar | -0.6684 | -0.2308 | -0.9668 | -0.2354 | 0.3158 | 3.1535 | 190 |
| har | -0.6904 | -0.2400 | -0.9746 | -0.2354 | 0.3093 | 3.2199 | 194 |
| har_x_lasso | -0.7062 | -0.3394 | -1.3183 | -0.2859 | 0.3020 | 4.0664 | 245 |
| persistence | -0.7731 | -0.2366 | -0.8959 | -0.1515 | 0.3061 | 3.2531 | 196 |
| lstm_x | -1.1125 | -0.5285 | -1.5541 | -0.4191 | 0.2702 | 4.7303 | 285 |
| always_long | -1.2112 | -1.0796 | -3.7604 | -0.4191 | 0.2368 | 11.9834 | 722 |

<!-- END:STRADDLES -->

**Every book loses.** That the forecast books beat both unconditional books by a
wide margin is the result. The forecasts are worth something: they cut the loss
by roughly two thirds against always being short, partly by trading less, about
5 times the premium budget a year against 11.9. They are worth less than the
cost of expressing them.

### The model-free check

<!-- RESULTS:SWAP -->

n_overlapping                                  723
mean_variance_points                     -0.018676
mean_vol_points_equivalent               -0.136662
nw_tstat_overlapping                     -7.048364
n_non_overlapping                               35
mean_non_overlapping                     -0.018232
tstat_non_overlapping                    -4.272333
sharpe_annualised                        -2.501622
share_positive                            0.051176
worst_observation                        -0.208197
model                       long_variance_swap_vix

<!-- END:SWAP -->

A synthetic variance swap struck at VIX squared, long realized variance, loses
0.0186 variance points per 21 days, with a Newey-West t of -7.1 on the
overlapping series and -4.0 on 35 non-overlapping observations. The short side
of that trade has an annualized Sharpe of 2.37 and needs no forecast at all.

Put the two together and the shape of the answer is clear. **The variance risk
premium in this sample was large, real, and completely consumed by the cost of
expressing it in listed options.** A variance swap harvests realized variance
uniformly and, as reported here, costlessly; a 21-day delta-hedged straddle pays
the full quoted spread twice, hedges discretely at 5 bps, and has its gamma
concentrated near one strike. The gap between a Sharpe of +2.4 and a Sharpe of
-0.69 is entirely implementation.

### One sentence per horizon

- **1 day.** The statistical gain is the largest and most significant of the
  three, and no strategy in this study trades at that horizon, so it converted
  into nothing at all.
- **5 days.** A real but smaller statistical gain, and the weakest transfer into
  P&L of the three horizons.
- **21 days.** The smallest statistical gain, and the only horizon where the
  forecast ranking clearly survives into the P&L ranking, which still is not
  enough to make any of it profitable.

## What did not work

Reported at the same volume as what did.

- **Semivariance HAR.** Fails at every horizon despite its premise holding in
  the coefficients. The single most confident prediction in the literature
  review.
- **Sentiment, attention and uncertainty, one at a time.** Wikipedia attention,
  GDELT tone and coverage share, EPU and EMV are all within a hair of HAR on
  their own at every horizon, and several are worse. Nothing in the free
  alternative-data blocks earns its place next to the option market.
- **The LSTM.** Worse than HAR at every horizon, and significantly worse at 5
  days. Adding the features helps at 1 day and hurts at 21.
- **Both neural models under stress.** On the 68 highest-volatility days out of
  723, the LSTM is 0.112 QLIKE worse than HAR and the feature-augmented LSTM
  0.214 worse, while both sit close to HAR on calm days. Whatever the networks
  learned, they lose it exactly when volatility forecasting matters.
- **Volatility timing.** No model beat buy-and-hold at any horizon.
- **Every straddle strategy**, including the unconditional ones.
- **25-delta skew.** The worst single feature in the study, significantly worse
  than HAR at both 1 and 21 days.

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
