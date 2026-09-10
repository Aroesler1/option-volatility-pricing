# Forecast and execution repair, 2026-09-08

Scope: the existing SPY sample, with no new search or data pull. The frozen
forecast specifications were refitted after purging inner validation outcomes
and fitting the LSTM scaler only on inner training rows. This is a retrospective
repair, not an untouched confirmation. Historical files remain unchanged.

The strategy clock now makes a close-t decision executable at close t+1.
Volatility-managed exposure earns the following close-to-close return. Missing
forecasts retain existing decisions on the market calendar. Rebalancing pays
for drift in actual exposure, including prior fees, rather than just changes
in target weights. Nonpositive equity raises an error.

For straddles, spread is charged on entry day; the exit hedge unwinds existing
shares without an unnecessary exit rebalance. Every admitted path contains the
full holding interval and both legs. Missing later delta can use an earlier
delta only. Daily P&L separately reconciles option marks, hedge price changes,
option spreads and hedge costs. Initial capital enters the drawdown peak.

**The full-path admission rule is retrospective.** A trader cannot know future
quote availability when placing the trade. All attempts, missing entries and
rejected paths are counted in the result table. Positive results therefore do
not establish an executable strategy. A future experiment needs a rule for
missing marks and forced liquidation fixed before entry, verified quote
coverage, financing, margin, and selection-adjusted inference. Do not replace
missing quotes with favorable synthetic prices.

The portfolio units are a fixed option-premium budget divided across overlapping
cohorts. They are not returns on fully collateralized capital. No transaction
cost or margin assumption was optimized to improve the result. The same known
sample and inspected variants preclude interpreting a new positive Sharpe as
independent evidence.

`run_integrity_report.py --check` recomputes forecast QLIKE, full-family Holm
values, daily P&L metrics, aggregate component sums and admission counts from
included derived files. It cannot reconstruct licensed option quotes from those
aggregates. The original and repaired results appear side by side in
`results/integrity_*.csv`; earlier `audit_*.csv` outputs are intermediate history.

Primary-source literature and its samples and horizons remain in
[the inference audit](inference_audit_2026.md) and
[the alternative-data review](literature_alt_data.md). The accounting repairs
do not strengthen the cited papers into evidence for this implementation.
What failed: feature improvements still do not survive family-wide inference,
and the option diagnostic still cannot establish tradable profitability.
