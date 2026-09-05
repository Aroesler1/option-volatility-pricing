# Alternative Data and HAR-Family Realized Variance Forecasting: A Literature Review

## Executive Summary

1. Alternative data (search, sentiment, Twitter/StockTwits, news text and embeddings) is statistically significant in essentially every paper reviewed, but the size of the improvement is usually small and concentrated at specific horizons.
2. Attention/sentiment proxies (Google search, FEARS, Twitter) tend to have their strongest, cleanest effect either contemporaneously (same day) or at the 1-day-ahead horizon; several papers (Da, Engelberg, Gao 2015; Behrendt and Schmidt 2018) find the effect on volatility is transitory and does not survive as a multi-day forecast improvement once persistence is controlled for.
3. Search-query data (Dimpfl and Jank 2016) is the clearest counter-example: gains there grow with the forecast horizon (1 day to 2 weeks) and are largest in high-volatility regimes.
4. Sign-based decompositions of realized variance (Patton and Sheppard 2015, realized semivariance) deliver the most robust and horizon-stable statistical gains of any predictor class reviewed here, from 1 day out to 3 months, without needing any external data.
5. Option-implied information (Busch, Christensen, Nielsen 2011; Bollerslev, Tauchen, Zhou 2009) is the strongest single alternative-data class, but the BTZ variance risk premium forecasts returns, not variance, and its explanatory power peaks at the quarterly (about 3-month) horizon, not at 1 day.
6. Macro-announcement volatility work (Andersen, Bollerslev, Diebold, Vega 2003, 2007) documents effects that live at the intraday-to-daily horizon; by the 21-day horizon these effects are expected to average out.
7. Economic/P&L evaluation is rare in the sentiment and attention literature; where it exists (fund flows, Sharpe-style realized utility) it is a secondary check, not a backtested trading strategy.
8. Economic/P&L evaluation is central and large in the option-return literature (Coval and Shumway 2001; Goyal and Saretto 2009), where realized-minus-implied volatility spreads and negative variance risk premia translate directly into tradeable straddle strategies.
9. Recent (2023-2026) LLM- and embedding-based work (FinText, M2VN, RiskLabs, Bodilsen and Lunde) shows the same pattern as older sentiment work: gains concentrate on high-volatility/jump days, are often only realized when the text signal is combined with a strong HAR-type history benchmark rather than used alone, and rarely include a P&L test.
10. Overall takeaway for a HAR-family SPY design: expect alternative data to help modestly and unevenly across our three target horizons (1-day, 5-day, 21-day), with attention/sentiment strongest near 1 day, implied volatility and jump-sign information strongest near 5-21 days, and any credible P&L story more likely to come from an options overlay than from the volatility forecast alone.

## Summary Table

| Paper | Data | Horizon(s) | Reported gain | Economic test? |
|---|---|---|---|---|
| Audrino, Sigrist, Ballinari (2020) | Twitter, StockTwits, RavenPack news, Google Trends, Wikipedia + macro/financial controls | Daily, weekly, monthly (HAR cascade) | Statistically significant improvement; magnitude described by authors as "small from an economic point of view" (exact R2/QLIKE not verified) | Not verified (no clear backtested strategy found) |
| Dimpfl and Jank (2016) | Google Trends SVI for DJIA, FTSE, CAC, DAX names | 1 day, 1 week, 2 weeks (out-of-sample) | Mincer-Zarnowitz R2 higher by more than 3 percentage points (FTSE: +3.6pp) in VHAR(3) vs HAR(3); gains grow with horizon; log-SQ explains 9%-23% of log-RV variance | No |
| Vlastakis and Markellos (2012) | Weekly Google Trends search volume, 30 NYSE/NASDAQ stocks | Weekly (relationship study, not a multi-horizon forecast comparison) | Not verified (no R2/QLIKE reported in sources reviewed) | Not verified |
| Da, Engelberg, Gao (2015), FEARS | Daily Google SVI (149 household-concern terms), SPY RV, VIX, VIX futures, CRSP portfolios, mutual fund flows, 2004-2011 | Contemporaneous (day t) for RV/VIX levels; 2-day reversal for returns and VIX futures | RV/VIX load on FEARS only contemporaneously, not at lags (transitory effect); VIX futures +43bp contemporaneous, -31bp reversal over 2 days; equity outflows about 55% of average daily flow | Partial (fund-flow and portfolio-spread evidence, not a backtested trading strategy) |
| Baker, Bloom, Davis, Kost (2019/2025), EMV | US newspaper text volume, 1985-2023+, 40 category trackers, 10-K text | Monthly/quarterly correlation with VIX and RV; not a multi-horizon forecast test | EMV tracker correlates about 0.8 (monthly) / 0.85 (quarterly) with VIX and realized S&P500 volatility | No |
| Behrendt and Schmidt (2018) | Twitter/StockTwits sentiment and activity, DJIA constituents, 5-minute intraday | Intraday / daily out-of-sample | Statistically significant co-movement but "negligible magnitude"; no out-of-sample forecast improvement | No (explicit negative result on usefulness) |
| Rahimikia, Zohren, Poon (2021/2023, updated) FinText | Dow Jones Newswires text, FinText embeddings, 23 NASDAQ stocks, 2007-2016 | 1-day ahead, emphasis on jump/high-volatility days | News-only does not beat a strong history benchmark alone; combined with history it lowers forecast loss and raises realized utility; best result is FinText + LOB ensemble | Yes (realized utility metric) |
| Busch, Christensen, Nielsen (2011) | Option-implied volatility, FX/stock/bond markets, HAR-CJ + VecHAR | Not fully verified (HAR-cascade daily horizons implied) | IV has incremental predictive power for future RV in all 3 markets, unbiased forecast in FX and stock; jump component also partly predictable from IV | No |
| Patton and Sheppard (2015) | High-frequency S&P500 SPDR + 105 individual stocks, 1997-2008 | 1, 5, 22, 66 trading days | SPDR R2: h=1 0.532 to 0.611; h=5 0.563 to 0.620; h=22 0.468 to 0.508; h=66 0.282 to 0.313 (semivariance model explains 10%-20% more variation than RV-only model) | No |
| Bollerslev, Tauchen, Zhou (2009) | S&P500 model-free implied and realized variance, 1990-2007 | 1 to 24 months (returns, not RV); peaks at quarterly (h=3) | Adjusted R2: h=1 1.07%, h=3 6.82% (max, t=2.86), h=6 5.42%, h=9 2.30%, h=12 1.23%, h=24 -0.50% | Not verified (predictive regression only in material reviewed) |
| Goyal and Saretto (2009) | US equity options, historical RV vs ATM IV spread, decile straddle/delta-hedged portfolios | Monthly (option roll) | Long-short straddle decile average monthly return about 22.7%, Sharpe about 0.71 (corroborated via secondary citation, not primary table) | Yes (this is the P&L paper) |
| Coval and Shumway (2001) | S&P 500 index (OEX) options | Weekly | Zero-beta ATM straddles lose about 3% per week on average | Yes (direct P&L test) |
| Andersen, Bollerslev, Diebold, Vega (2003, 2007) | High-frequency FX, stock, bond futures around scheduled US macro announcements | Intraday (minutes) to daily | Announcement surprises produce conditional mean jumps and volatility responses; bond > FX > equity sensitivity, equity state-dependent | No |
| Bodilsen and Lunde (2025) | Macro and firm-specific news sentiment, S&P500 index and individual stocks | Long-horizon (macro sentiment) and 1-period-ahead (firm news count, overnight) | "Substantial" and "significant" long-horizon gains from macro sentiment (no numeric R2/QLIKE verified) | Not verified |
| Halousková and Lyocsa (2025) | Social media, news, search/attention to 10 scheduled macro announcements, 404 US stocks | Daily, emphasis on extreme-variation days | Average improvement up to 14.99% vs benchmark on high-volatility days | Not verified |
| Kong, Hwang, Kaiser, Vryonides, Oomen, Zohren (2025), M2VN | FNSPID news + TimeMachineGPT point-in-time embeddings, 7 large-cap US equities, 2013-2023 | 1 day ahead | QLIKE down 11.7% vs TimesNet baseline; MAPE down 3.6% | No |
| Cao et al. (2024), RiskLabs | Earnings call text/audio, daily news, time series; single-stock (TWTR) case study | 3, 7, 15, 30 days (explicitly multi-horizon) | Estimation bias (AEP) smaller at longer horizons in rolling simulation; no HAR baseline comparison shown | No |

## Paper-by-Paper Detail

### 1. Audrino, Sigrist, Ballinari (2020)

**Citation:** Francesco Audrino, Fabio Sigrist, Daniele Ballinari, "The impact of sentiment and attention measures on stock market volatility," *International Journal of Forecasting*, 36(2), 2020, 334-357.

**Data:** A large panel combining social media (Twitter, StockTwits), news articles (RavenPack News Analytics), and information-consumption/search data (Google Trends, Wikipedia page views), alongside standard economic and financial control variables.

**Model/spec:** HAR-RV as the baseline volatility model, extended with a large set of candidate sentiment/attention regressors selected via a penalized (LASSO-type) regression framework; evaluated with QLIKE, MSE, and (per one source) a realized-utility style metric.

**Horizon(s) of gains:** The HAR cascade structure implies daily, weekly, and monthly horizons are all examined. Google search volume on financial keywords and StockTwits message volume were identified as the most useful predictors across the panel.

**Magnitude:** Described qualitatively in the sources reviewed as a statistically significant but "relatively small" improvement "from an economic point of view." Specific numeric R2/QLIKE improvement figures could not be independently verified from the abstract and secondary sources available; marked **not verified**.

**Economic evaluation:** Not verified whether a formal backtested trading strategy was reported.

### 2. Dimpfl and Jank (2016)

**Citation:** Thomas Dimpfl, Stephan Jank, "Can Internet search queries help to predict stock market volatility?" *European Financial Management*, 22(2), 2016, 171-192 (working paper version: University of Tuebingen Working Papers in Economics and Finance No. 18).

**Data:** Realized volatility (10-minute intraday sampling) for DJIA, FTSE, CAC, DAX, July 2006-June 2011; Google Trends search volume index (SVI) for the index names ("Dow," "FTSE," "CAC," "DAX") in the respective home country.

**Model/spec:** AR(1), AR(3), and Corsi's HAR(3) model, each estimated with and without one lag of log search volume (log-SQ); also VAR(1)/VAR(3) and VHAR(3) bivariate extensions modeling RV and search jointly. Loss functions: MSE, QLIKE, and Mincer-Zarnowitz R2.

**Horizon(s) of gains:** 1-day-ahead in-sample; 1-day, 1-week, and 2-week out-of-sample forecasts. The paper explicitly states: "the longer the forecast horizon, the more efficiency gains are apparent."

**Magnitude:** Multivariate (search-augmented) models outperform univariate models on MSE, QLIKE, and R2 for all four indices at all horizons tested. For FTSE, the Mincer-Zarnowitz R2 of VHAR(3) is higher than HAR(3) by 3.6 percentage points; other indices show gains "more than 3 percentage points." A long-run variance decomposition finds log search queries account for 9% (FTSE) to 23% (CAC) of the variance of log realized volatility. Gains are concentrated in high-volatility periods (the October 2008 crisis is highlighted).

**Economic evaluation:** None. This is a pure statistical forecast-evaluation paper (MSE, QLIKE, R2); no trading strategy or P&L test.

### 3. Vlastakis and Markellos (2012)

**Citation:** Nikolaos Vlastakis, Raphael N. Markellos, "Information demand and stock market volatility," *Journal of Banking & Finance*, 36(6), 2012, 1808-1821.

**Data:** 30 of the largest NYSE- and NASDAQ-listed stocks; information demand proxied by weekly Google Trends search volume; compared against historical and implied volatility and trading volume.

**Model/spec:** Regression/VAR-style analysis of the dynamic relationship between information demand, information supply, returns, and volatility; not framed as a HAR-style multi-horizon out-of-sample forecast comparison in the sources reviewed.

**Horizon(s) of gains:** Data are weekly; the paper is a relationship/demand study rather than an explicit multi-horizon forecast-improvement exercise, so no distinct horizon-by-horizon gain is reported in the sources reviewed. **Not verified.**

**Magnitude:** Information demand is "significantly positively related" to historical and implied volatility and to trading volume, even controlling for returns and information supply, and increases with higher-return periods. No R2, QLIKE, or MSE figures were located. **Not verified.**

**Economic evaluation:** Not verified from sources reviewed; no explicit trading test found.

### 4. Da, Engelberg, Gao (2015), "The Sum of All FEARS"

**Citation:** Zhi Da, Joseph Engelberg, Pengjie Gao, "The Sum of All FEARS: Investor Sentiment and Asset Prices," *Review of Financial Studies*, 28(1), 2015, 1-32.

**Data:** Daily Google search volume index (SVI) for 30 household-concern terms (e.g., "recession," "bankruptcy," "unemployment") selected via the Harvard IV-4/Lasswell dictionaries, aggregated into the FEARS index, January 2004-December 2011. Volatility proxies: SPY realized volatility (15-minute intraday sampling, following Andersen, Bollerslev, Diebold, Ebens 2001) and the CBOE VIX; also CBOE VIX futures returns, CRSP beta/downside-risk-sorted decile portfolios, and TrimTabs daily mutual fund flow data (equity vs. intermediate Treasury bond funds).

**Model/spec:** Predictive regressions of returns, RV/VIX levels, VIX futures returns, and fund flows on FEARS at lags k=0,1,2; ARFIMA(1,d,1) models for realized volatility and VIX (to control persistence) with FEARS and macro controls (EPU, ADS index) as regressors.

**Horizon(s) of gains:** FEARS predicts a return reversal over the following two trading days (k=1, k=2). For realized volatility and VIX levels, the FEARS coefficient is positive and significant **only contemporaneously** (day t); once the persistent ARFIMA component is controlled for, neither RV nor VIX loads significantly on **lagged** FEARS. This means the volatility effect is transitory/same-day, not a forward-looking multi-day forecast improvement. VIX futures returns show a contemporaneous +43bp move with a -31bp reversal over the following two days.

**Magnitude:** A one-standard-deviation increase in FEARS is associated with a 23bp contemporaneous decrease in the high-beta-minus-low-beta portfolio return spread (reversed by k=2); a 39bp decrease in the high-minus-low downside-beta spread; equity mutual fund outflows equal to about 55% of the average daily equity fund flow (-2.79e-5 vs. average -5.06e-5); bond fund inflows of 8.2e-5 (larger than the average daily bond flow of 7.49e-5).

**Economic evaluation:** Partial. The fund-flow and portfolio-return-spread results are economically interpretable (a "flight to safety" pattern) but the paper does not backtest a trading strategy or report transaction-cost-adjusted P&L.

### 5. Baker, Bloom, Davis, Kost (2019/2025), Equity Market Volatility (EMV) Tracker

**Citation:** Scott R. Baker, Nicholas Bloom, Steven J. Davis, Kyle J. Kost, "Policy News and Stock Market Volatility," NBER Working Paper 25720, March 2019 (revised May 2025); forthcoming, *Journal of Financial Economics*.

**Data:** Text volume from major US newspapers, 1985 to present, used to construct a headline daily/monthly Equity Market Volatility (EMV) tracker and 40 category-specific EMV trackers (commodity markets, interest rates, real estate, aggregate activity, inflation, tax/monetary/regulatory policy, etc.); combined with textual analysis of 10-K filings for firm-level risk exposure measures.

**Model/spec:** The EMV tracker is validated by its correlation with the VIX and realized S&P500 return volatility, in and out of sample. Firm-level panel regressions relate realized volatility to category-specific EMV exposure measures, with firm and time fixed effects.

**Horizon(s) of gains:** The paper documents contemporaneous/near-term (monthly and quarterly) co-movement between EMV and VIX/realized volatility. It is not framed as a multi-horizon out-of-sample forecast-improvement exercise in the style of a HAR study, so no horizon-specific forecast gain (R2, QLIKE) is reported in the sources reviewed.

**Magnitude:** The EMV tracker correlates with the VIX and realized daily-return volatility of the S&P500 at about 0.8 (monthly) and 0.85 (quarterly). Roughly 30% of EMV articles discuss tax policy, 30% monetary policy, and 25% some form of regulation; commodity-market news appears in over 40% of EMV articles.

**Economic evaluation:** None. This is a descriptive/decomposition paper, not a trading or forecast-improvement study.

### 6. Behrendt and Schmidt (2018), "The Twitter Myth Revisited"

**Citation:** Simon Behrendt, Alexander Schmidt, "The Twitter myth revisited: Intraday investor sentiment, Twitter activity and individual-level stock return volatility," *Journal of Banking & Finance*, 96, 2018, 355-367.

**Data:** Dow Jones Industrial Average constituent stocks; intraday absolute 5-minute returns as the volatility measure; Twitter sentiment and activity (tweet volume) measures.

**Model/spec:** Time series models controlling for intraday periodicity in absolute returns, augmented with Twitter sentiment/activity as predictors; out-of-sample forecast comparison against a benchmark without Twitter variables.

**Horizon(s) of gains:** Intraday (5-minute) and daily.

**Magnitude:** Statistically significant co-movement between Twitter activity/sentiment and volatility, but the authors characterize the effect as of "negligible magnitude." Critically, **out-of-sample forecasting performance showed no improvement** when Twitter variables were added as predictors.

**Economic evaluation:** None; the paper's explicit conclusion is that "high-frequency Twitter information is not particularly useful for highly active investors" forecasting intraday volatility, i.e., a negative result on practical usefulness.

### 7. Rahimikia, Zohren, Poon (2021/2023, updated through 2025/2026), FinText

**Citation:** Eghbal Rahimikia, Stefan Zohren, Ser-Huang Poon, "Realised Volatility Forecasting: Machine Learning via Financial Word Embedding," first posted 2021 (SSRN/arXiv:2108.00480), multiple revisions through 2025/2026.

**Data:** Dow Jones Newswires Text News Feed Database, 23 NASDAQ-listed stocks, 27 July 2007-18 November 2016; FinText, a specialized financial word embedding trained on this corpus, compared against general-purpose embeddings.

**Model/spec:** Machine learning models (the exact architecture was not fully verified from the abstract alone) forecasting realized volatility, using FinText/general embeddings as news-based features, benchmarked against and combined with a strong volatility-history model and a limit-order-book (LOB) based ML model; SHAP/Integrated Gradients used for interpretability.

**Horizon(s) of gains:** One-day-ahead, with particular emphasis on days with volatility jumps / high-volatility regimes.

**Magnitude:** "News-only forecasts contain useful predictive information but generally do not outperform strong volatility-history benchmarks." Combining stock-related news forecasts with the history benchmark "lowers forecast losses for several specifications and increases realized utility," i.e., evidence of complementarity rather than substitution. A simple ensemble of the FinText model and a LOB-based ML model gives the best performance for both normal and jump-volatility days. Exact numeric loss reductions were not verified from the abstract.

**Economic evaluation:** Yes, via a "realized utility" metric (an economic/loss-based evaluation, distinct from a full backtested trading strategy).

### 8. Busch, Christensen, Nielsen (2011)

**Citation:** Thomas Busch, Bent Jesper Christensen, Morten Orregaard Nielsen, "The role of implied volatility in forecasting future realized volatility and jumps in foreign exchange, stock, and bond markets," *Journal of Econometrics*, 160(1), 2011, 48-57.

**Data:** Option-implied volatility and high-frequency realized volatility (decomposed into continuous and jump components) across foreign exchange, stock, and bond markets.

**Model/spec:** HAR-type models augmented with implied volatility (HAR-RV-CJ-IV style) and a vector-HAR (VecHAR) specification to address the endogeneity of implied volatility as a regressor.

**Horizon(s) of gains:** The HAR-cascade structure implies daily-horizon forecasts (with weekly/monthly aggregation components), but the specific horizon(s) at which gains were reported could not be fully verified from the sources reviewed. **Not verified.**

**Magnitude:** Implied volatility is found to contain incremental information about future (continuous) realized volatility in all three markets and is an unbiased forecast of realized volatility in the foreign exchange and stock markets. The jump component of volatility is also found to be, to some extent, predictable from implied volatility, suggesting options are calibrated to incorporate information about future jumps. No specific R2/QLIKE/MSE figures were located. **Magnitude not verified.**

**Economic evaluation:** None reported.

### 9. Patton and Sheppard (2015), "Good Volatility, Bad Volatility"

**Citation:** Andrew J. Patton, Kevin Sheppard, "Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility," *Review of Economics and Statistics*, 97(3), July 2015, 683-697.

**Data:** High-frequency data for the S&P 500 SPDR ETF (SPY) and 105 individual S&P 500 constituent stocks, 1997-2008.

**Model/spec:** HAR-RV extended by decomposing realized variance into realized semivariance from positive and negative high-frequency returns (RS+, RS-), following the estimator of Barndorff-Nielsen, Kinnebrock, Shephard (2010); also a signed-jump-variation extension. Forecasting horizons h = 1, 5, 22, 66 trading days (roughly 1 day, 1 week, 1 month, 1 quarter), evaluated both in-sample and pseudo-out-of-sample via Mincer-Zarnowitz R2.

**Horizon(s) of gains:** All four horizons tested, h=1 through h=66 (1 day to about 3 months); the paper states the semivariance decomposition "significantly improves forecasts of future volatility... true for horizons ranging from one day to three months, both in-sample and (pseudo-)out-of-sample."

**Magnitude:** For the SPDR (S&P 500 ETF), Mincer-Zarnowitz R2 rises from the RV-only HAR benchmark to the semivariance-augmented model as follows: h=1: 0.532 to 0.611; h=5: 0.563 to 0.620; h=22: 0.468 to 0.508; h=66: 0.282 to 0.313. The text states the semivariance model "explains 10% to 20% more of the variation in future volatility than the model that contains only realized variance." Negative semivariance is the dominant driver at all horizons; positive semivariance is economically small beyond about h=15. Similar (slightly larger baseline R2, similar gain pattern) results hold for the panel of 105 individual stocks.

**Economic evaluation:** None. Purely a statistical forecast-improvement paper (R2, implicitly MSE via the Mincer-Zarnowitz regression); no trading strategy or P&L test.

### 10. Bollerslev, Tauchen, Zhou (2009)

**Citation:** Tim Bollerslev, George Tauchen, Hao Zhou, "Expected Stock Returns and Variance Risk Premia," *Review of Financial Studies*, 22(11), November 2009, 4463-4492.

**Data:** S&P 500 monthly excess returns, "model-free" implied variance from OTM S&P 500 index options, and model-free realized variance from high-frequency (5-minute) returns, January 1990-December 2007; also standard predictors (P/E ratio, dividend yield, default spread DFSP, term spread TMSP, relative risk-free rate RREL, consumption-wealth ratio CAY).

**Model/spec:** Predictive regressions of h-month cumulative excess S&P 500 returns on the variance risk premium (IV minus RV), for horizons h = 1, 3, 6, 9, 12, 15, 18, 24 months, both alone and combined with the traditional predictors, using Hodrick (1992) standard errors to account for overlapping observations.

**Horizon(s) of gains:** This paper forecasts **returns**, not realized variance itself. The degree of predictability peaks at the **quarterly (h=3 month) horizon**: "the magnitude of the predictability is particularly strong at the intermediate quarterly return horizon, where it dominates that afforded by other popular predictor variables."

**Magnitude:** Adjusted R2 by horizon (Table 2): h=1: 1.07%, h=3: 6.82% (maximum; t-statistic = 2.86), h=6: 5.42%, h=9: 2.30%, h=12: 1.23%, h=15: 1.00%, h=18: 0.05%, h=24: -0.50%. Combining the variance premium with the P/E ratio in a monthly multiple regression raises the adjusted R2 to 3.77% with both coefficients significant, exceeding the sum of the individual-regressor R2s.

**Economic evaluation:** Not verified from the material reviewed whether a transaction-cost-adjusted, backtested trading strategy is reported elsewhere in the paper; the core empirical contribution is the predictive regression, not a P&L exercise.

### 11. Goyal and Saretto (2009)

**Citation:** Amit Goyal, Alessio Saretto, "Cross-section of option returns and volatility," *Journal of Financial Economics*, 94(2), 2009, 310-326.

**Data:** US individual-equity options; historical (backward-looking) realized volatility and at-the-money implied volatility for each underlying stock.

**Model/spec:** Stocks/options are sorted into deciles by the spread between historical realized volatility and ATM implied volatility. Decile portfolios of straddles and of delta-hedged calls and puts are formed; a zero-cost long-short strategy goes long the decile with the largest positive RV-IV spread and short the decile with the largest negative spread, rebalanced monthly.

**Horizon(s) of gains:** Monthly (matching the option roll/rebalancing frequency).

**Magnitude:** The long-short straddle decile portfolio produces an average monthly return of approximately 22.7% with a Sharpe ratio of approximately 0.71 (this figure is corroborated by a secondary source citing Goyal and Saretto's results rather than directly confirmed from the primary paper's tables in the sources reviewed; treat as **likely correct but not independently verified from the primary table**). Delta-hedged calls and puts show statistically and economically significant positive returns in high-spread deciles and negative returns in low-spread deciles. Results are robust across market conditions, stock characteristics, industry groupings, option liquidity, and standard risk-factor models.

**Economic evaluation:** Yes. This is the primary economic/P&L evaluation paper in the set: an explicit, factor-model-adjusted, long-short options trading strategy with reported returns and Sharpe ratios.

### 12. Coval and Shumway (2001)

**Citation:** Joshua D. Coval, Tyler Shumway, "Expected Option Returns," *Journal of Finance*, 56(3), June 2001, 983-1009.

**Data:** S&P 100 index (OEX) options.

**Model/spec:** A theoretical option-pricing/beta framework predicting that expected call returns exceed the underlying's and rise with strike price (and that put returns fall below the risk-free rate and rise, in absolute value, with strike price), tested against realized returns on zero-beta, at-the-money straddle positions.

**Horizon(s) of gains:** Weekly rebalanced positions.

**Magnitude:** Zero-beta, at-the-money straddle positions produce **average losses of approximately 3% per week**, a large and statistically robust anomaly relative to the null of zero expected return implied by simple beta-based option pricing.

**Economic evaluation:** Yes. This is a direct, realized P&L test of a market-neutral options strategy; the authors interpret the systematic losses as evidence that stochastic volatility carries a priced (negative) risk premium not captured by standard option betas.

### 13. Andersen, Bollerslev, Diebold, Vega (2003, 2007), Macro Announcements

**Citations:**
- Torben G. Andersen, Tim Bollerslev, Francis X. Diebold, Clara Vega, "Micro Effects of Macro Announcements: Real-Time Price Discovery in Foreign Exchange," *American Economic Review*, 93(1), 2003, 38-62.
- Torben G. Andersen, Tim Bollerslev, Francis X. Diebold, Clara Vega, "Real-Time Price Discovery in Global Stock, Bond and Foreign Exchange Markets," *Journal of International Economics*, 73(2), 2007, 251-277 (NBER Working Paper 11312, 2005).

**Data:** (2003) Six years of real-time, 5-minute USD exchange-rate quotations around scheduled US macro announcements. (2007) High-frequency futures returns for US, German, and British stock, bond, and foreign exchange markets around real-time US macro news releases.

**Model/spec:** Jump-diffusion / conditional-mean and conditional-variance models estimating the response of high-frequency returns and volatility to macro announcement surprises.

**Horizon(s) of gains:** Intraday (minutes) response window; volatility effects are documented at the intraday-to-daily horizon, not multi-day or multi-week.

**Magnitude:** (2003) Announcement surprises produce conditional mean "jumps" in FX rates linked directly to the size of the surprise. (2007) Bond markets react most strongly to macro news, followed by foreign exchange, then equity markets; equity market reactions are state-dependent (bad news has a positive impact during economic expansions and the traditionally expected negative impact during recessions), reflecting offsetting cash-flow and discount-rate channels. Significant cross-market volatility linkages beyond direct announcement effects are also documented via pronounced heteroskedasticity in the high-frequency data.

**Economic evaluation:** None; these are market-microstructure/price-discovery papers, not trading studies.

## Recent (2023-2026) LLM- and Embedding-Based Volatility Forecasting

### A. Bodilsen and Lunde (2025)

**Citation:** Simon Tranberg Bodilsen, Asger Lunde, "Exploiting News Analytics for Volatility Forecasting," *Journal of Applied Econometrics*, 40(1), 2025, 18-36.

**Data:** Sentiment scores for macroeconomic and firm-specific news (news source not fully verified from the abstract), applied to the S&P 500 index and individual stocks.

**Model/spec:** Traditional time-series (HAR-type) models of realized volatility augmented with macro and firm-specific news sentiment, plus a firm-specific overnight news-count variable.

**Horizon(s) of gains:** Macro news sentiment shows "substantial enhancements" specifically at **long horizons**; firm-specific news sentiment alone has only modest predictive power, but the count of overnight firm-specific news significantly improves **one-period-ahead (short horizon)** forecasts.

**Magnitude:** Not verified; the abstract available describes results qualitatively ("significantly," "substantial") without numeric R2/QLIKE/MSE figures.

**Economic evaluation:** Not verified from sources reviewed.

### B. Halousková and Lyocsa (2025)

**Citation:** Martina Halousková, Stefan Lyocsa, "Forecasting U.S. equity market volatility with attention and sentiment to the economy," arXiv:2503.19767 [q-fin.GN], 2025.

**Data:** Social media, news articles, information-consumption data, and search-engine data measuring public attention and sentiment toward 10 scheduled US macroeconomic announcements; applied to 404 major US stocks.

**Model/spec:** Standard and machine-learning methods estimating attention/sentiment measures, incorporated into volatility forecasting models.

**Horizon(s) of gains:** Daily, with the largest effect on days of extreme price variation.

**Magnitude:** Models incorporating attention/sentiment "consistently improve volatility forecasts across all economic sectors," with the greatest improvement averaging **14.99%** against the benchmark on high-volatility days.

**Economic evaluation:** Not verified; the abstract focuses on forecast-accuracy metrics.

### C. Kong, Hwang, Kaiser, Vryonides, Oomen, Zohren (2025), M2VN

**Citation:** Yaxuan Kong, Yoontae Hwang, Marcus Kaiser, Chris Vryonides, Roel Oomen, Stefan Zohren, "Fusing Narrative Semantics for Financial Volatility Forecasting," arXiv:2510.20699 [q-fin.CP], October 2025.

**Data:** Seven large-cap US equities (KO, CMCSA, COP, GILD, MRK, NKE, ORCL); training 2013-2017, validation 2018-2020, test 2021-2023; news from the FNSPID dataset, embedded using TimeMachineGPT, a point-in-time LLM chosen specifically to avoid look-ahead bias; daily OHLCV price data.

**Model/spec:** M2VN, a multi-modal deep learning network combining a price/time-series encoder, LLM news embeddings, and temporal markers, with a dual objective (MSE prediction loss plus a contrastive alignment loss). Target: realized volatility via Parkinson, Garman-Klass, and Rogers-Satchell estimators.

**Horizon(s) of gains:** One-day-ahead forecasting (12-day lookback window).

**Magnitude:** QLIKE reduced by 11.7% versus a TimesNet baseline (best or tied-best in 10 of 14 stock-metric settings); MAPE reduced by 3.6% (top-2 in 12 of 14 settings). Ablating news degrades performance substantially (example: COP QLIKE worsens from 0.0552 to 0.0599 without news); ablating volume degrades 27 of 28 stock-metric pairs.

**Economic evaluation:** None. No trading simulation, transaction-cost analysis, or P&L backtest; evaluation is limited to QLIKE and MAPE forecast-accuracy metrics.

### D. Cao, Chen, Kumar, Pei, Yu, Li, Dimino, Ausiello, Subbalakshmi, Ndiaye (2024), RiskLabs

**Citation:** Yupeng Cao, Zhi Chen, Prashant Kumar, Qingyun Pei, Yangyang Yu, Haohang Li, Fabrizio Dimino, Lorenzo Ausiello, K.P. Subbalakshmi, Papa Momar Ndiaye, "RiskLabs: Predicting Financial Risk Using Large Language Model based on Multimodal and Multi-Sources Data," MFFM Workshop @ ACM ICAIF '24, November 2024 (arXiv:2404.07452).

**Data:** Earnings conference call transcripts and audio, daily news, and time-series market data; empirical illustration is a **single-stock case study (ticker TWTR)** with a 250-trading-day training window (2016-02-22 to 2017-02-15).

**Model/spec:** LLM-based encoders for earnings-call text/audio and news, fused with a time-series encoder, feeding a Bayesian VAR (estimated via MCMC/Gibbs sampling) that links volatility across horizons, plus a multi-task head for Value-at-Risk. This is explicitly positioned as a position/workshop paper rather than a fully benchmarked study.

**Horizon(s) of gains:** Explicitly multi-horizon: 3-day, 7-day, 15-day, and 30-day realized volatility.

**Magnitude:** In a rolling-window simulation (250-day training window, 100 iterations), the average estimation bias (Absolute Error Percentage, AEP) is reported to become smaller as the volatility horizon lengthens, i.e., longer-horizon forecasts were more accurate than shorter-horizon ones in this specific case study. Exact per-horizon AEP percentages appear in the paper's Table 5 but were **not independently verified to a precise figure** in the material reviewed. No comparison against a HAR or other standard baseline was found in the material reviewed.

**Economic evaluation:** None. Also note the important caveat that this is a single-stock case study without a rigorous out-of-sample benchmark comparison, so its evidentiary weight is lower than the other papers in this section.

## What This Implies for Our Design

For a HAR-family study of SPY realized variance at 1-day, 5-day, and 21-day horizons, the literature above supports the following concrete, testable expectations:

1. **Realized semivariance (signed jumps) should be tried first and will likely be the single most reliable improvement at all three horizons.** Patton and Sheppard (2015) show gains from 1 day to 66 days using only price data, no external alternative data required; this is the cheapest, most robust upgrade to a baseline HAR-RV model and should be the benchmark against which alternative data is judged incremental.

2. **Attention/sentiment variables (Google Trends, StockTwits, Twitter, FEARS-style indices) are expected to show their largest, most reliably significant effect at or near the 1-day horizon, and to weaken or vanish by 21 days**, based on Da, Engelberg, Gao (2015) finding a purely contemporaneous/2-day-reversal pattern and Behrendt and Schmidt (2018) finding no out-of-sample gain at all from Twitter data.

3. **Google-search-based attention is the one sentiment-adjacent signal in this review with evidence of horizon-increasing gains** (Dimpfl and Jank 2016, gains growing from 1 day to 2 weeks). We should explicitly test whether this pattern replicates for SPY-level (rather than single-index) search terms and holds out to 21 days, rather than assuming attention data is inherently short-horizon.

4. **Option-implied volatility should be tested primarily as a 5-day and 21-day horizon variable, not a 1-day variable**, based on Busch, Christensen, Nielsen (2011) (IV forecasts the continuous RV component and is calibrated to future jumps) and Bollerslev, Tauchen, Zhou (2009) (variance risk premium predictability of returns peaks at the quarterly horizon). We should include the variance risk premium (IV minus RV) as a control before crediting any other alternative-data variable with an "incremental" gain.

5. **Expect statistical significance without economic significance to be the norm, not the exception, for sentiment/attention variables**, per Audrino, Sigrist, Ballinari (2020) and Behrendt and Schmidt (2018). We should report QLIKE/R2 gains alongside a realized-utility or economic-loss metric (as in Rahimikia, Zohren, Poon) rather than R2 alone, and pre-specify a minimum economically meaningful improvement threshold before claiming a variable "works."

6. **News/LLM-embedding signals are most likely to add value on high-volatility or jump days specifically, and mainly when combined with (not substituted for) the HAR history benchmark**, per Rahimikia, Zohren, Poon (2021/2023) and Kong et al. (2025). Our evaluation should include a volatility-regime split (e.g., top-quintile RV days vs. rest) rather than reporting only a pooled-sample average.

7. **A credible P&L story is far more likely to come from an options overlay (straddle/delta-hedged strategy conditioned on our RV forecast vs. market IV) than from the RV forecast alone**, per Goyal and Saretto (2009) and Coval and Shumway (2001), both of which show large, robust, direct P&L from RV-IV spread and variance-risk-premium strategies. If economic evaluation is a project goal, we should design an explicit options-based backtest, not rely on forecast-accuracy improvement as a proxy for tradeable value.

8. **Macro-announcement effects (Andersen, Bollerslev, Diebold, Vega 2003, 2007) should be controlled for at the 1-day horizon specifically** (e.g., FOMC/CPI/NFP announcement-day dummies), since their documented effects live at the intraday-to-daily horizon and would otherwise contaminate estimates of what alternative data contributes.

9. **Expect diminishing or negative marginal contribution from adding multiple alternative-data sources at once**, since most reviewed papers test one data type against a bare HAR benchmark rather than against a benchmark that already includes IV and signed jumps; our design should test alternative data's marginal R2/QLIKE contribution on top of a HAR-RS-IV benchmark, not on top of bare HAR-RV, to avoid overstating the value of sentiment/attention/news data.

## Sources

- Audrino, Sigrist, Ballinari (2020): https://www.sciencedirect.com/science/article/pii/S0169207019301645 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3188941 ; https://econpapers.repec.org/RePEc:eee:intfor:v:36:y:2020:i:2:p:334-357
- Dimpfl and Jank (2016): https://onlinelibrary.wiley.com/doi/abs/10.1111/eufm.12058 ; working paper: https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/47872/pdf/WPEF18Dimpfl_Jank.pdf
- Vlastakis and Markellos (2012): https://www.sciencedirect.com/science/article/abs/pii/S0378426612000507 ; https://econpapers.repec.org/RePEc:eee:jbfina:v:36:y:2012:i:6:p:1808-1821
- Da, Engelberg, Gao (2015): https://rady.ucsd.edu/faculty/directory/engelberg/pub/portfolios/FEARS.pdf ; https://academic.oup.com/rfs/article-abstract/28/1/1/1682440
- Baker, Bloom, Davis, Kost (2019/2025): https://www.nber.org/papers/w25720 ; https://www.nber.org/system/files/working_papers/w25720/w25720.pdf ; https://www.sciencedirect.com/science/article/abs/pii/S0304405X25001953
- Behrendt and Schmidt (2018): https://www.sciencedirect.com/science/article/abs/pii/S0378426618302115 ; https://ideas.repec.org/a/eee/jbfina/v96y2018icp355-367.html
- Rahimikia, Zohren, Poon (2021/2023): https://arxiv.org/abs/2108.00480 ; https://doi.org/10.2139/ssrn.3895272
- Busch, Christensen, Nielsen (2011): https://econpapers.repec.org/article/eeeeconom/v_3a160_3ay_3a2011_3ai_3a1_3ap_3a48-57.htm ; https://www.researchgate.net/publication/314836148
- Patton and Sheppard (2015): https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf ; https://www.jstor.org/stable/43555003
- Bollerslev, Tauchen, Zhou (2009): https://public.econ.duke.edu/~boller/Published_Papers/rfs_09.pdf ; https://academic.oup.com/rfs/article-abstract/22/11/4463/1565787
- Goyal and Saretto (2009): https://docs.lib.purdue.edu/ciberwp/55 ; https://www.sciencedirect.com/science/article/abs/pii/S0304405X09001251 ; https://www.semanticscholar.org/paper/Cross-Section-of-Option-Returns-and-Volatility-Goyal-Saretto/2584305dfcd7b652d7eeb701ed3ece03641f2fc4
- Coval and Shumway (2001): https://ideas.repec.org/a/bla/jfinan/v56y2001i3p983-1009.html ; https://www.jstor.org/stable/222539
- Andersen, Bollerslev, Diebold, Vega (2003): https://ideas.repec.org/a/aea/aecrev/v93y2003i1p38-62.html
- Andersen, Bollerslev, Diebold, Vega (2007): https://www.nber.org/papers/w11312 ; https://www.nber.org/system/files/working_papers/w11312/w11312.pdf ; https://www.sciencedirect.com/science/article/abs/pii/S0022199607000608
- Bodilsen and Lunde (2025): https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3095 ; https://pure.au.dk/portal/en/publications/exploiting-news-analytics-for-volatility-forecasting/ ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4401032
- Halousková and Lyocsa (2025): https://arxiv.org/pdf/2503.19767 ; https://arxiv.org/abs/2503.19767
- Kong, Hwang, Kaiser, Vryonides, Oomen, Zohren (2025): https://arxiv.org/abs/2510.20699 ; https://arxiv.org/html/2510.20699
- Cao et al. (2024), RiskLabs: https://arxiv.org/pdf/2404.07452 ; https://arxiv.org/abs/2404.07452
- Parvini and Assa (2025), noted but not fully detailed above (data/horizon/magnitude not verified from sources reviewed): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5136391
