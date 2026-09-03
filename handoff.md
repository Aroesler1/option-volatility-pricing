# Handoff

## Current goal - DONE

Two gaps closed in the volatility horse race: pairwise Diebold-Mariano replaced
(not removed) by the Model Confidence Set, and the Clements-Preve remedies added
so the HARQ headline is tested against the alternatives the literature says beat
it.

## Headline finding

The old headline - "HARQ posts the best median QLIKE once it has real intraday
quarticity" - no longer holds. On true 5-minute RV, 821 overlapping forecasts:

| model | QLIKE mean | median | collapsed | MCS p | in 90% MCS |
|---|---|---|---|---|---|
| WLS-HAR (1/sqrt(RQ)) | **0.2132** | 0.0993 | 0 | 1.000 | yes |
| log-HAR | 0.2156 | **0.0946** | 0 | 0.696 | yes |
| WLS-HAR (1/RV) | 0.2166 | 0.0960 | 0 | 0.696 | yes |
| mean combination | 0.2222 | 0.0994 | 0 | 0.089 | **no** |
| HAR-RV | 0.2307 | 0.1064 | 0 | 0.191 | yes |
| HARQ | 0.2315 | 0.1000 | 0 | 0.191 | yes |
| HAR+PDV | 0.2378 | 0.1300 | 0 | 0.014 | no |
| PDV | 0.3388 | 0.2303 | 0 | 0.001 | no |
| persistence | 0.6422 | 0.1438 | 0 | 0.001 | no |

MCS(90%) = {WLS-HAR(RQ), log-HAR, WLS-HAR(RV), HAR-RV, HARQ}.

Two-sided and both halves belong in the README: the remedies **beat HARQ on
point estimate**, but HARQ is **not rejected** - the MCS cannot separate the top
five. The MCS does decisively reject PDV, HAR+PDV and persistence.

The forecast combination is excluded while its own constituents are not. Not a
bug: the elimination rule is studentized, the combination shares nearly all its
variance with its constituents, so being slightly worse than the best of them is
measured very precisely and it goes early.

## Verified state

- `.venv/bin/python -m pytest tests -q` -> **16 passed** (was 4).
- MCS membership verified **stable across 8 bootstrap seeds**: the same 5 models
  in every time, the same 4 out every time. Combination p ranges 0.064-0.089,
  never reaching 0.10.
- `results/estimator_comparison.csv` regenerated with the 9 models.

## Paper facts, read from the source, not assumed

Fetched the public GARP whitepaper PDF (SSRN itself is Cloudflare-blocked):
`https://www.garp.org/hubfs/Whitepapers/a2r1W000000iDb0QAE_RiskIntell.6.20.19.Whitepaper.Volatility.pdf`

- **WLS weights, section 2.3.3, verbatim**: four schemes. WLS_G uses
  w = 1/h_hat from a GARCH(1,1) on OLS residuals; WLS_RVhat uses w = 1/RV_hat
  from a fitted OLS HAR; WLS_RQ uses **w = 1/sqrt(RQ)**; WLS_RV uses
  **w = 1/RV**. The last two are implemented; the paper reports both are
  *always* in the 90% MCS.
- **log-HAR retransformation, equation (8)**, citing Proietti and Lutkepohl
  (2013): `F_t = exp(b0 + b1 log RV_d + b2 log RV_w + b3 log RV_m + sigma_u^2/2)`.
  The bias correction is +sigma_u^2/2 inside the exponent. Not guessed.
- **MCS**: they use the range statistic of Hansen, Lunde and Nason (2003), 90%
  level, QLIKE loss. Implemented to match.

**Two corrections to the original brief.** (1) The paper's sample is **SPX, DJI
and DAX** - three major stock indices over 16 years - not "S&P 500 + 26 NYSE
stocks". (2) **Forecast combination is not in the paper**; "combin" appears only
in the phrase "this combination should be far from ideal" about RV+OLS. The mean
combination was still implemented as asked, and is attributed to the
combination-puzzle literature rather than to Clements-Preve.

## Unit convention (easy to get wrong)

Clements-Preve model realized VARIANCE; this repo models annualized realized
VOLATILITY throughout and squares the forecast inside QLIKE. Since
log(RV) = 2*log(vol), log-HAR slopes are identical either way and only the
intercept and retransformation constant differ, so the +sigma^2/2 correction is
applied on the scale the model is estimated on. Documented in
`vol_forecasting.py` and in the README methodology notes.

## Files

New in `vol_forecasting.py`: `model_confidence_set`,
`stationary_bootstrap_indices`, `LogHARRV`, `WLSHARRV`,
`clements_preve_weights`, `mean_combination`.
New in `run_vol_benchmark.py`: `log_har_forecasts`, `wls_har_forecasts`.
`run_intraday_benchmark.py` now evaluates 9 models and reports MCS membership;
new flags `--alpha`, `--n-boot`, `--seed`.
`chronological_split` was **kept, not removed** - its own test uses it to build
a split. Its docstring now says the benchmark scripts use their own
expanding-window refit loop instead.

## Next actions

1. The remaining Clements-Preve remedies: LAD (robust regression), the
   quartic-root transformation (their equation 9 has its own retransformation),
   and the two parametric WLS schemes. LAD is the one they report most strongly
   alongside WLS.
2. The MCS cannot separate the top five on one asset over 821 overlapping
   forecasts. A cross-section of assets is what would give it power.
3. `run_vol_benchmark.py` and `run_iv_benchmark.py` still report pairwise DM
   only; the MCS is wired into the intraday benchmark alone.
4. Consider the insanity filter of Swanson and White (1995), which
   Clements-Preve apply to all forecasts; this repo clips at 1e-4 instead and
   counts the clips.
