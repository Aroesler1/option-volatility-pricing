# Sentiment, Volatility Forecasting, and Options Pricing

This repository explores how text sentiment, realized volatility features, and option-market data can be combined in a single research workflow. The main notebook collects Google News and Reddit text, engineers weekly and daily features, trains PyTorch LSTM models for volatility forecasting, and compares Black-Scholes prices with observed option-chain quotes.

## Repository layout

- `Main.ipynb`: end-to-end notebook for data collection, feature engineering, modeling, and option-pricing experiments
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

## Known limits

- Sentiment is driven by TextBlob rather than a finance-specific language model
- The workflow is notebook-centric and depends on interactive execution order
- Market and option-chain data from public sources can be sparse or inconsistent
- Black-Scholes is used as a baseline pricing model and does not capture the full surface dynamics of listed options
