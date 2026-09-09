# Inference and trading-clock audit

Protocol fixed on 2026-09-06 before computing the corrected straddle results.
This is a retrospective repair of an existing study, not a fresh holdout or a
newly preregistered trading discovery. The earlier headline tables remain as
historical evidence.

## Ranked changes and failing hypotheses

1. Separate rule calibration from trading. The existing first-half premium and
   margin calibration is preserved. Only dates after the calibration endpoint
   may open trades, including the unconditional controls, and all books are
   scored on the same calendar. Hypothesis: the old conclusion that every book
   loses survives this correction. No parameter will change if it fails.
2. Correct feature selection inference. Report two-sided raw and Holm-adjusted
   p-values across all feature/horizon pairs: 48 against HAR and 45 against the
   richer HAR benchmark. Hypothesis: at least one incremental feature improves
   its benchmark at familywise 5%. This audit has already inspected raw results,
   so adjustment provides a robustness check, not independent confirmation.
3. Relabel the VIX comparison. It is a descriptive implied-minus-realized proxy
   using different assets, clocks and realized-variance coverage. It does not
   identify the portion of a straddle's loss caused by trading costs.

The corrected trading calculation reuses the frozen forecast files. Their
LASSO, boosting and LSTM hyperparameter validation tails were chronological but
not purged for overlapping targets. They remain historical inputs, and economic
statistics from them do not establish clean model selection. Fixing that inner
split requires fresh forecast generation under an unchanged protocol, not
selection of new specifications against this known evaluation sample.

## Reproduction and scope

SPY only. Historical forecast dates are 2022-10-07 to 2025-08-29. Historical
option rules use the first 360 aligned forecasts for calibration. Corrected
trading starts strictly after 2024-03-14. No new data is fetched. Public derived
daily P&L is sufficient to regenerate corrected performance tables; rebuilding
daily P&L itself requires the existing licensed, ignored option-chain cache.

## Primary-source checks

- [Holm (1979), A Simple Sequentially Rejective Multiple Test Procedure](https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf):
  mathematical multiple-testing result, no financial sample or horizon. The
  sequential Bonferroni procedure controls familywise error without requiring
  independent tests. It does not repair an invalid input p-value or selection
  among uncounted research choices. Here horizons belong to each feature family;
  the two benchmark families and historical robustness arms are disclosed.
- [Cboe VIX FAQ](https://www.cboe.com/tradable_products/vix/faqs):
  methodology, no empirical sample. VIX uses SPX option quotations to measure
  constant 30-day expected volatility. SPY regular-session realized variance
  over 21 trading days is not the same payoff. Missing overnight variation,
  horizon and underlying differences survive even at zero transaction costs.
- [Patton and Sheppard (2015)](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf):
  SPDR plus 105 stocks selected from S&P 100 membership, June 1997 to July 2008;
  regular-session transaction data, event-time sampling and subsampling.
  Horizons 1, 5, 22 and 66 sessions. Table 2's SPDR R2 values 0.532 and 0.611
  are in-sample variance-level regressions. Section VI separately evaluates
  pseudo-out-of-sample forecasts. This repository's square-root semivariance
  model is an adaptation, so its failure cannot reject the paper's result.
- [O Nuallain (2025), GNAR-HARX](https://arxiv.org/abs/2510.24443v1):
  ten international indices, roughly 2005-2020 out of sample, one-day forecasts.
  Its QLIKE winner excludes exogenous variables; implied volatility improves
  some other specifications. A working paper, not evidence that all additional
  information improves a strong benchmark. No numerical trading result is
  claimed here. Cross-market replication is more informative than another SPY
  feature search, but needs data outside the present committed panel.
- [Fan, Wang and Ye (2026), options-driven RV forecasting](https://arxiv.org/abs/2604.02743):
  SPX five-minute returns and OptionMetrics option panels, reported overall
  sample January 2011 to June 2021. The inspected
  [HTML text](https://arxiv.org/html/2604.02743v1) gives conflicting OOS dates:
  Section 2 says June 2021 to June 2021, Section 4.4 says January to June 2020.
  It reports one-day QLIKE 0.0428 for HAR and 0.0403 for rough-Heston HAR, with
  gains across approximately one month; its multiday definition also includes
  the current day. No executable P&L test is reported. This is a possible future
  option-feature direction, not a directly comparable validated improvement.

The search was checked through 2026-09-06. It is a targeted update addressing
the selected changes, not a claim to have verified every existing citation or
reviewed every paper published in 2026.


## Measured result and validation

The original 360-date calibration ends 2024-03-14. Trading-clock-corrected
performance uses 366 common sessions, 2024-03-15 to 2025-08-29. Every one of
21 reported variants has a negative Sharpe. Always-long improves from -1.1742
to -0.2996, ahead of every two-sided forecast rule (best HGB -0.4079).
Always-short changes from -0.7202 to -1.2061. These rankings are descriptive,
not new tests of selected economic superiority.

No feature improvement survives Holm within either benchmark family or across
the joint family. One-day implied variance's raw p=.0011546 becomes .0508005
in its 48-test family and .101601 across all 93 tests. This answers the stated
incremental-feature hypothesis negatively. The trading-clock repair preserves
the original loss conclusion but overturns the two-sided-control ranking.

Validation before final documentation: 127 baseline tests and 143 tests after
the implementation passed. The derived-table verifier passed. A full forecast
refit was not run: the stored LASSO, HGB, LSTM and combination forecasts remain
legacy inputs. Their corrected trading-clock statistics do not remove their
model-selection limitation.
