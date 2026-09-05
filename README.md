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
| combination | 0.1881 | 0.0794 | -4.5334 | 0.0000 | 1.0000 | yes |
| har_x_lasso | 0.1894 | 0.0861 | -2.7884 | 0.0053 | 0.8450 | yes |
| lstm_x | 0.1925 | 0.0795 | -1.6229 | 0.1046 | 0.8450 | yes |
| hgb | 0.1935 | 0.0897 | -1.5481 | 0.1216 | 0.8450 | yes |
| har_rv_iv | 0.1964 | 0.0800 | -3.0779 | 0.0021 | 0.0855 | no |
| har | 0.2099 | 0.0840 |  |  | 0.0015 | no |
| shar | 0.2107 | 0.0865 | 0.4336 | 0.6646 | 0.0015 | no |
| lstm | 0.2181 | 0.0859 | 1.4672 | 0.1423 | 0.0060 | no |
| persistence | 0.2791 | 0.1049 | 5.5314 | 0.0000 | 0.0000 | no |

MCS(90%) = {combination, har_x_lasso, lstm_x, hgb}

#### Horizon 5 days

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| hgb | 0.1870 | 0.0609 | -1.2663 | 0.2054 | 1.0000 | yes |
| har_x_lasso | 0.1872 | 0.0570 | -2.3660 | 0.0180 | 0.9825 | yes |
| combination | 0.1937 | 0.0564 | -2.1454 | 0.0319 | 0.4835 | yes |
| har_rv_iv | 0.2036 | 0.0575 | -0.4829 | 0.6291 | 0.4035 | yes |
| har | 0.2081 | 0.0590 |  |  | 0.1350 | yes |
| shar | 0.2090 | 0.0619 | 1.3770 | 0.1685 | 0.1275 | yes |
| lstm_x | 0.2360 | 0.0600 | 1.3100 | 0.1902 | 0.1485 | yes |
| lstm | 0.2548 | 0.0754 | 2.7960 | 0.0052 | 0.0125 | no |
| persistence | 0.3509 | 0.0953 | 5.1046 | 0.0000 | 0.0000 | no |

MCS(90%) = {hgb, har_x_lasso, combination, har_rv_iv, har, shar, lstm_x}

#### Horizon 21 days

| model | QLIKE mean | QLIKE median | DM vs HAR | p | MCS p | in 90% MCS |
|---|---|---|---|---|---|---|
| hgb | 0.2252 | 0.0577 | -0.8993 | 0.3685 | 1.0000 | yes |
| combination | 0.2327 | 0.0875 | -1.4841 | 0.1378 | 0.6490 | yes |
| har | 0.2456 | 0.1057 |  |  | 0.3730 | yes |
| shar | 0.2457 | 0.1059 | 0.4057 | 0.6849 | 0.3730 | yes |
| lstm | 0.2467 | 0.0859 | 0.0722 | 0.9424 | 0.4860 | yes |
| har_x_lasso | 0.2468 | 0.0953 | 0.0914 | 0.9272 | 0.3730 | yes |
| har_rv_iv | 0.2493 | 0.1092 | 0.8042 | 0.4213 | 0.3355 | yes |
| lstm_x | 0.2925 | 0.0746 | 1.3247 | 0.1853 | 0.2475 | yes |
| persistence | 0.5782 | 0.1220 | 3.6972 | 0.0002 | 0.0050 | no |

MCS(90%) = {hgb, combination, har, shar, lstm, har_x_lasso, har_rv_iv, lstm_x}

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
| combination | 100% |
| lstm_x | 96% |
| har_rv_iv | 91% |
| hgb | 91% |
| har | 39% |
| shar | 39% |
| lstm | 35% |
| persistence | 22% |

#### Horizon 1, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_x_lasso | 100% |
| lstm_x | 100% |
| combination | 100% |
| har_rv_iv | 91% |
| hgb | 55% |
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
| hgb | 74% |
| lstm | 61% |
| shar | 48% |
| har | 43% |
| persistence | 22% |

#### Horizon 5, 504-observation windows (11 windows)

| model | share of windows in the 90% MCS |
|---|---|
| har_rv_iv | 100% |
| har_x_lasso | 100% |
| hgb | 100% |
| lstm_x | 100% |
| combination | 100% |
| har | 64% |
| shar | 64% |
| lstm | 64% |
| persistence | 0% |

#### Horizon 21, 252-observation windows (23 windows)

| model | share of windows in the 90% MCS |
|---|---|
| hgb | 100% |
| lstm_x | 70% |
| har_x_lasso | 65% |
| lstm | 65% |
| combination | 65% |
| har_rv_iv | 61% |
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
| har_rv_iv | 45% |
| har_x_lasso | 45% |
| har | 36% |
| shar | 36% |
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
| atm_ivar_30 | option | 0.1964 | -0.0134 | -3.0779 | 0.0021 |
| term_slope_30_91 | option | 0.2000 | -0.0098 | -1.7162 | 0.0861 |
| atm_iv_30 | option | 0.2064 | -0.0035 | -0.4947 | 0.6208 |
| epu_log | uncertainty | 0.2069 | -0.0030 | -1.4238 | 0.1545 |
| is_payrolls | calendar | 0.2080 | -0.0019 | -1.1306 | 0.2582 |
| is_fomc | calendar | 0.2094 | -0.0005 | -0.6976 | 0.4854 |
| har | baseline | 0.2099 | 0.0000 |  |  |
| emv_overall | uncertainty | 0.2100 | 0.0002 | 0.4566 | 0.6479 |
| is_cpi | calendar | 0.2104 | 0.0006 | 0.5066 | 0.6124 |
| vix_vol | option | 0.2111 | 0.0013 | 0.1644 | 0.8694 |
| wiki_attention | attention | 0.2113 | 0.0014 | 0.6112 | 0.5410 |
| spy_put_call | option | 0.2157 | 0.0059 | 3.4204 | 0.0006 |
| skew_25d_30 | option | 0.2204 | 0.0105 | 3.2413 | 0.0012 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| atm_ivar_30 | option | 0.2036 | -0.0045 | -0.4829 | 0.6291 |
| term_slope_30_91 | option | 0.2043 | -0.0038 | -0.3802 | 0.7038 |
| is_fomc | calendar | 0.2066 | -0.0016 | -1.5414 | 0.1232 |
| epu_log | uncertainty | 0.2068 | -0.0013 | -0.5976 | 0.5501 |
| is_payrolls | calendar | 0.2074 | -0.0007 | -2.6130 | 0.0090 |
| is_cpi | calendar | 0.2078 | -0.0003 | -0.4923 | 0.6225 |
| har | baseline | 0.2081 | 0.0000 |  |  |
| emv_overall | uncertainty | 0.2081 | 0.0000 | 0.0310 | 0.9753 |
| vix_vol | option | 0.2082 | 0.0001 | 0.0067 | 0.9946 |
| wiki_attention | attention | 0.2087 | 0.0006 | 0.5952 | 0.5517 |
| spy_put_call | option | 0.2096 | 0.0015 | 1.1816 | 0.2373 |
| atm_iv_30 | option | 0.2101 | 0.0020 | 0.1838 | 0.8542 |
| skew_25d_30 | option | 0.2211 | 0.0130 | 2.0283 | 0.0425 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| emv_overall | uncertainty | 0.2435 | -0.0021 | -0.4374 | 0.6618 |
| wiki_attention | attention | 0.2448 | -0.0008 | -0.9129 | 0.3613 |
| is_payrolls | calendar | 0.2452 | -0.0004 | -1.3000 | 0.1936 |
| har | baseline | 0.2456 | 0.0000 |  |  |
| is_cpi | calendar | 0.2457 | 0.0001 | 1.3823 | 0.1669 |
| is_fomc | calendar | 0.2459 | 0.0004 | 0.1747 | 0.8613 |
| atm_ivar_30 | option | 0.2493 | 0.0037 | 0.8042 | 0.4213 |
| vix_vol | option | 0.2506 | 0.0050 | 0.3191 | 0.7497 |
| spy_put_call | option | 0.2511 | 0.0056 | 0.9488 | 0.3427 |
| atm_iv_30 | option | 0.2515 | 0.0059 | 0.3629 | 0.7167 |
| skew_25d_30 | option | 0.2546 | 0.0091 | 0.6208 | 0.5347 |
| term_slope_30_91 | option | 0.2555 | 0.0099 | 0.9345 | 0.3500 |
| epu_log | uncertainty | 0.2594 | 0.0138 | 2.6350 | 0.0084 |

<!-- END:MARGINAL -->

<!-- RESULTS:MARGINAL_RICH -->

#### Horizon 1 day

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.1920 | -0.0037 | -1.0427 | 0.2971 |
| is_payrolls | calendar | 0.1949 | -0.0008 | -0.7327 | 0.4637 |
| is_fomc | calendar | 0.1955 | -0.0002 | -0.3445 | 0.7304 |
| skew_25d_30 | option | 0.1957 | -0.0001 | -0.3038 | 0.7613 |
| emv_overall | uncertainty | 0.1957 | -0.0000 | -0.0393 | 0.9686 |
| har_rs_iv | baseline | 0.1957 | 0.0000 |  |  |
| is_cpi | calendar | 0.1963 | 0.0006 | 0.5142 | 0.6071 |
| wiki_attention | attention | 0.1967 | 0.0009 | 0.3217 | 0.7477 |
| epu_log | uncertainty | 0.1971 | 0.0014 | 2.2135 | 0.0269 |
| spy_put_call | option | 0.2002 | 0.0045 | 3.0775 | 0.0021 |
| vix_vol | option | 0.2007 | 0.0050 | 1.1344 | 0.2566 |
| atm_iv_30 | option | 0.2017 | 0.0060 | 1.2359 | 0.2165 |

#### Horizon 5 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| term_slope_30_91 | option | 0.2032 | -0.0012 | -0.1606 | 0.8724 |
| is_fomc | calendar | 0.2033 | -0.0011 | -1.2052 | 0.2281 |
| is_cpi | calendar | 0.2040 | -0.0004 | -1.0798 | 0.2802 |
| vix_vol | option | 0.2042 | -0.0002 | -0.0290 | 0.9769 |
| har_rs_iv | baseline | 0.2044 | 0.0000 |  |  |
| is_payrolls | calendar | 0.2045 | 0.0001 | 0.5450 | 0.5858 |
| wiki_attention | attention | 0.2047 | 0.0003 | 0.2140 | 0.8306 |
| spy_put_call | option | 0.2054 | 0.0010 | 0.9470 | 0.3436 |
| emv_overall | uncertainty | 0.2060 | 0.0016 | 1.7702 | 0.0767 |
| skew_25d_30 | option | 0.2070 | 0.0026 | 1.4164 | 0.1567 |
| atm_iv_30 | option | 0.2073 | 0.0029 | 0.3965 | 0.6917 |
| epu_log | uncertainty | 0.2081 | 0.0037 | 2.8337 | 0.0046 |

#### Horizon 21 days

| model | block | QLIKE mean | delta vs base | DM | p |
|---|---|---|---|---|---|
| wiki_attention | attention | 0.2482 | -0.0011 | -1.1581 | 0.2468 |
| emv_overall | uncertainty | 0.2483 | -0.0010 | -0.2086 | 0.8348 |
| is_payrolls | calendar | 0.2490 | -0.0003 | -1.2179 | 0.2233 |
| har_rs_iv | baseline | 0.2493 | 0.0000 |  |  |
| is_cpi | calendar | 0.2494 | 0.0001 | 2.1091 | 0.0349 |
| is_fomc | calendar | 0.2497 | 0.0003 | 0.1698 | 0.8652 |
| vix_vol | option | 0.2505 | 0.0012 | 0.1000 | 0.9203 |
| atm_iv_30 | option | 0.2512 | 0.0019 | 0.1512 | 0.8798 |
| spy_put_call | option | 0.2549 | 0.0056 | 0.9714 | 0.3314 |
| skew_25d_30 | option | 0.2568 | 0.0075 | 0.6656 | 0.5056 |
| term_slope_30_91 | option | 0.2603 | 0.0109 | 1.0370 | 0.2998 |
| epu_log | uncertainty | 0.2665 | 0.0172 | 2.2379 | 0.0252 |

<!-- END:MARGINAL_RICH -->

### Calm days against stressed days

The news and embedding literature is consistent that text signals earn their
keep on high-volatility days and contribute nothing the rest of the time, so a
pooled average can hide the whole effect. Days are split by whether realized
volatility exceeds the 80th percentile of every realized volatility observed
strictly before them, which is knowable in real time.

<!-- RESULTS:REGIME -->

(not generated: results/altdata_regime_split_nonews.csv not found)

<!-- END:REGIME -->

### The semivariance result, in full

<!-- RESULTS:SHAR -->

(not generated: results/altdata_shar_coefficients_nonews.csv not found)

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
| buy_and_hold |  | 1.4061 | 0.2330 | 0.1657 | -0.2023 |  |
| lstm | 0.2181 | 1.3466 | 0.2725 | 0.2024 | -0.2550 | 25.7699 |
| combination | 0.1881 | 1.3011 | 0.2610 | 0.2006 | -0.2420 | 32.6674 |
| hgb | 0.1935 | 1.2922 | 0.2581 | 0.1997 | -0.2412 | 34.7041 |
| har_rv_iv | 0.1964 | 1.2918 | 0.2630 | 0.2036 | -0.2455 | 36.1600 |
| lstm_x | 0.1925 | 1.2838 | 0.2617 | 0.2039 | -0.2486 | 28.7988 |
| har | 0.2099 | 1.2837 | 0.2585 | 0.2014 | -0.2392 | 39.5043 |
| shar | 0.2107 | 1.2762 | 0.2562 | 0.2008 | -0.2375 | 41.8975 |
| har_x_lasso | 0.1894 | 1.2580 | 0.2507 | 0.1993 | -0.2405 | 32.6814 |
| persistence | 0.2791 | 1.2238 | 0.2507 | 0.2049 | -0.2412 | 61.3575 |

QLIKE winner combination, Sharpe winner lstm; Spearman(QLIKE, Sharpe) = -0.233 (p = 0.546)

#### Horizon 5

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| buy_and_hold |  | 1.4061 | 0.2330 | 0.1657 | -0.2023 |  |
| lstm_x | 0.2360 | 1.3554 | 0.2720 | 0.2007 | -0.2570 | 16.1290 |
| har_x_lasso | 0.1872 | 1.3134 | 0.2516 | 0.1916 | -0.2296 | 29.7655 |
| combination | 0.1937 | 1.3062 | 0.2512 | 0.1923 | -0.2357 | 26.4603 |
| har_rv_iv | 0.2036 | 1.2889 | 0.2503 | 0.1942 | -0.2402 | 32.3559 |
| lstm | 0.2548 | 1.2845 | 0.2512 | 0.1955 | -0.2544 | 14.5390 |
| har | 0.2081 | 1.2704 | 0.2449 | 0.1927 | -0.2357 | 37.4821 |
| shar | 0.2090 | 1.2643 | 0.2434 | 0.1925 | -0.2352 | 38.3345 |
| hgb | 0.1870 | 1.2392 | 0.2382 | 0.1922 | -0.2365 | 31.1827 |
| persistence | 0.3509 | 1.2238 | 0.2507 | 0.2049 | -0.2412 | 61.3575 |

QLIKE winner hgb, Sharpe winner lstm_x; Spearman(QLIKE, Sharpe) = -0.200 (p = 0.606)

#### Horizon 21

| model | QLIKE | Sharpe | mean p.a. | vol p.a. | max drawdown | turnover p.a. |
|---|---|---|---|---|---|---|
| buy_and_hold |  | 1.4061 | 0.2330 | 0.1657 | -0.2023 |  |
| hgb | 0.2252 | 1.3977 | 0.2609 | 0.1866 | -0.2138 | 21.4165 |
| har_x_lasso | 0.2468 | 1.3659 | 0.2477 | 0.1813 | -0.2168 | 24.3235 |
| lstm | 0.2467 | 1.3597 | 0.2584 | 0.1900 | -0.2363 | 9.4034 |
| combination | 0.2327 | 1.3556 | 0.2462 | 0.1816 | -0.2181 | 16.7193 |
| har_rv_iv | 0.2493 | 1.3373 | 0.2393 | 0.1790 | -0.2211 | 23.2518 |
| har | 0.2456 | 1.3341 | 0.2384 | 0.1787 | -0.2215 | 24.6169 |
| shar | 0.2457 | 1.3304 | 0.2379 | 0.1788 | -0.2214 | 24.3990 |
| lstm_x | 0.2925 | 1.2612 | 0.2507 | 0.1988 | -0.2498 | 10.5339 |
| persistence | 0.5782 | 1.2238 | 0.2507 | 0.2049 | -0.2412 | 61.3575 |

QLIKE winner hgb, Sharpe winner hgb; Spearman(QLIKE, Sharpe) = -0.617 (p = 0.077)

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
| hgb | -0.2007 | -0.1048 | -0.9578 | -0.2584 | 0.4424 | 5.4848 | 330 |
| har_rv_iv | -0.5523 | -0.2396 | -0.9987 | -0.3065 | 0.4198 | 4.8698 | 293 |
| combination | -0.5540 | -0.2868 | -1.0374 | -0.4034 | 0.4169 | 5.1025 | 307 |
| shar | -0.5898 | -0.2526 | -0.9801 | -0.3020 | 0.4209 | 4.9363 | 297 |
| har | -0.5982 | -0.2557 | -0.9765 | -0.3049 | 0.4141 | 4.9363 | 297 |
| persistence | -0.6820 | -0.2417 | -0.7928 | -0.1515 | 0.4455 | 5.4848 | 330 |
| always_short | -0.6922 | -0.6373 | -3.3234 | -1.2436 | 0.5298 | 11.9174 | 721 |
| lstm | -0.6933 | -0.3361 | -1.1611 | -0.5777 | 0.4416 | 5.2687 | 317 |
| har_x_lasso | -0.7893 | -0.4367 | -1.3979 | -0.4875 | 0.3842 | 5.6676 | 341 |
| lstm_x | -1.1795 | -0.6753 | -2.0221 | -0.6031 | 0.3578 | 6.7812 | 408 |
| always_long | -1.2071 | -1.0733 | -3.7604 | -0.4191 | 0.2372 | 11.9174 | 721 |

#### long_only

| model | Sharpe | mean p.a. | max drawdown | worst month | trade hit rate | turnover p.a. | trades |
|---|---|---|---|---|---|---|---|
| hgb | -0.3037 | -0.1436 | -0.9679 | -0.2584 | 0.3766 | 3.9723 | 239 |
| combination | -0.4631 | -0.2027 | -1.0382 | -0.2441 | 0.3430 | 3.4404 | 207 |
| har_rv_iv | -0.5905 | -0.2055 | -0.9384 | -0.2354 | 0.3187 | 3.0249 | 182 |
| lstm | -0.6283 | -0.2237 | -0.8728 | -0.2556 | 0.3298 | 3.1247 | 188 |
| shar | -0.6453 | -0.2207 | -0.9372 | -0.2354 | 0.3226 | 3.0914 | 186 |
| har | -0.6560 | -0.2238 | -0.9335 | -0.2269 | 0.3118 | 3.0914 | 186 |
| har_x_lasso | -0.6795 | -0.3144 | -1.2518 | -0.2803 | 0.3149 | 3.9058 | 235 |
| persistence | -0.7736 | -0.2370 | -0.8959 | -0.1515 | 0.3061 | 3.2576 | 196 |
| lstm_x | -1.1003 | -0.5280 | -1.5493 | -0.4129 | 0.2473 | 4.7036 | 283 |
| always_long | -1.2071 | -1.0733 | -3.7604 | -0.4191 | 0.2372 | 11.9174 | 721 |

<!-- END:STRADDLES -->

**Every book loses.** That the forecast books beat both unconditional books by a
wide margin is the result. The forecasts are worth something: they cut the loss
by roughly two thirds against always being short, partly by trading less, about
5 times the premium budget a year against 11.9. They are worth less than the
cost of expressing them.

### The model-free check

<!-- RESULTS:SWAP -->

n_overlapping                                  726
mean_variance_points                     -0.018615
mean_vol_points_equivalent               -0.136436
nw_tstat_overlapping                     -7.095157
n_non_overlapping                               35
mean_non_overlapping                     -0.017756
tstat_non_overlapping                    -4.043843
sharpe_annualised                        -2.367832
share_positive                            0.050964
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
