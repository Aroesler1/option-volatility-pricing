# Sentiment, Volatility Forecasting, and Options Pricing

This repository explores how text sentiment, realized volatility features, and option-market data can be combined in a single research workflow. The main notebook collects Google News and Reddit text, engineers weekly and daily features, trains PyTorch LSTM models for volatility forecasting, and compares Black-Scholes prices with observed option-chain quotes.

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
