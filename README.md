# PIC 16B Final Project — Sentiment, Volatility Forecasting, and Options Pricing

This repo centers on **`Main.ipynb`**, an exploratory workflow that combines:
- **Market data** (prices + option chains via `yfinance`)
- **Sentiment signals** from **Google News RSS** and **Reddit** (via `feedparser`/`requests` and `praw`)
- **Volatility forecasting** with **PyTorch LSTMs**
- **Black–Scholes option pricing** and **implied volatility** inversion

It also contains a **Streamlit dashboard prototype** (included in the notebook) for interactive option-pricing exploration.

## What’s in this repo

- **`Main.ipynb`**: the main notebook (data collection, feature engineering, modeling, plots, and prototype app code)
- **`data/weekly_stock_data.csv`**: weekly dataset for multiple tickers (sentiment, prices, forward returns, vol, article volume)
- **`data/TSM_data.csv`**: daily dataset (volatility, average sentiment, volume) used in the notebook
- **`data/sentiment_prev_full_month.json`**: Reddit crawl output (posts + comments) with TextBlob-based sentiment fields
- **`weekly_stock_data.csv`**: a second copy/export of the weekly dataset

## Project overview (as implemented)

### 1) Sentiment data
- **Google News RSS sentiment**: pulls RSS entries across multiple regions and query variants, then computes **TextBlob polarity** on cleaned article text.
- **Reddit sentiment**: searches Reddit via `praw`, pulls post text + comments, then computes **TextBlob polarity/subjectivity** and aggregates a combined sentiment score.

### 2) Market + volatility data
- **Prices / returns**: downloads historical closes via `yfinance` and constructs weekly forward returns.
- **Volatility features**: computes rolling realized volatility and stores weekly volatility features in the saved CSVs.

### 3) Modeling (PyTorch)
- Trains **LSTM models** to predict volatility (and related targets) using sentiment and market features.
- Includes a simple **grid search** helper for LSTM hyperparameters in the notebook.

### 4) Options pricing
- **Black–Scholes** call/put pricing for given \(S, K, T, r, \sigma\).
- **Implied volatility** via numerical root-finding (`brentq`) to invert Black–Scholes against a market option price.
- Fetches option chains via `yfinance` and compares model outputs to nearby quoted strikes.

### 5) Streamlit app (prototype)
There is a **Streamlit UI prototype** embedded in `Main.ipynb` that exposes a Black–Scholes calculator with live metrics and option chain lookup.

Important note: the notebook’s *core sentiment pipelines* use **TextBlob** (Google News RSS + Reddit). The Streamlit prototype includes **placeholder sentiment code** intended to be swapped/extended.

## Known limitations

- **Sentiment quality**: TextBlob is general-purpose and can miss finance-specific nuance.
- **Coverage bias**: news/social sources can be sparse or concentrated in certain time windows, which affects aggregates.
- **Data availability**: option chains and some price histories can have gaps.
- **Model scope**: Black–Scholes assumes constant volatility and lognormal dynamics; dividends/borrowing costs are not modeled here.

## Roadmap ideas

- Improve/expand data sources and backfilling
- Finance-tuned sentiment models (e.g., FinBERT) and better aggregation
- Stronger baselines and calibration for sentiment→volatility links
