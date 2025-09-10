# Black-Scholes Options & Sentiment Notebook

**Main.ipynb** is an exploratory project that unifies option pricing, implied & historical volatility, and news/social sentiment into a single, inspectable workflow. It also includes a Streamlit UI sketch that turns the analysis into an interactive dashboard.

## Project overview

- **Market data layer:** pulls spot and historical prices for equities and exposes option chains (calls/puts by expiry and strike).
- **Pricing & volatility layer:** implements Black–Scholes pricing, computes **implied volatility** from market quotes, and **historical volatility** from returns.
- **Sentiment layer:** fetches recent headlines and Reddit-style sources, scores them with VADER, and maps the aggregate sentiment to a volatility proxy.
- **App layer (prototype):** a Streamlit interface with inputs (ticker, strike, expiry, risk-free rate) and outputs (spot, vols, option prices, nearest market quotes, IV).

## Major components

### 1) Data ingestion
- **Equity prices:** minute/day bars for the underlying, used for spot and realized-vol calculations.
- **Options data:** option chain by **expiry**; used to find the nearest strike to the user’s selection and to compare model prices to quotes.
- **News & social signals:** recent headlines (Google/NewsAPI) and Reddit-derived items for sentiment estimation.

### 2) Pricing & analytics
- **Black–Scholes model:** closed-form Call/Put prices for given \(S, K, T, r, \sigma\).
- **Implied volatility (IV):** numerical root-finding (brentq) to invert Black–Scholes so the model price matches a quoted option price.
- **Historical volatility:** daily log-return standard deviation, annualized with \(\sqrt{252}\).
- **Sentiment-based volatility:** a simple, bounded mapping from average headline sentiment (VADER compound) to a volatility proxy, used as an illustrative signal alongside historical vol.

### 3) Streamlit UI (prototype)
- **Sidebar inputs:** ticker, strike, risk-free rate, expiry date.
- **Live metrics:** spot price; **historical volatility** (%); **sentiment-based volatility** (%).
- **Pricing outputs:** model Call/Put prices from Black–Scholes; nearest available option quotes; computed **implied volatility** for the closest quoted contract (when available).
- **UX details:** expiry selection via valid exchange dates; nearest-strike matching to the entered strike; defensive messaging when data isn’t available.

## Methods and assumptions

- **No dividends:** current formulation prices non-dividend underlyings (or assumes negligible dividends).
- **Volatility inputs:** 
  - *Historical vol* from realized returns (simple estimator).
  - *Sentiment vol* from VADER scores; clipped to a sensible range to avoid extremes.
- **Implied vol bracketing:** search bounds wide enough for typical equities; failures handled with warnings/NA.
- **Nearest-strike mapping:** market quotes are compared at the closest available strike to the desired strike.

## Key functions (high level)

- `fetch_stock_data(ticker)` — retrieves the latest close/spot for the underlying.
- `get_valid_expiry_dates(ticker)` — returns listed option expiries for the symbol.
- `fetch_option_prices(ticker, expiry_date)` — fetches the calls/puts chain for a given expiry.
- `implied_volatility(option_price, S, K, T, r, option_type)` — numerically inverts Black–Scholes to compute IV.
- `fetch_sentiment_data(ticker)` — gathers recent headlines for the ticker.
- `analyze_sentiment(texts)` — computes VADER compound scores for a set of headlines.
- `calculate_volatility_from_sentiment(ticker)` — converts average sentiment to a bounded volatility proxy.
- `calculate_historical_volatility(ticker, months)` — computes annualized realized volatility from historical returns.
- `main()` — Streamlit entry point wiring inputs, data retrieval, analytics, and UI outputs.

## What’s visible in the notebook

- **Walkthrough cells** computing:
  - spot and realized volatility
  - Black–Scholes prices for chosen inputs
  - option chain fetch + nearest strike comparison
  - implied volatility from quotes
  - sentiment fetch → scoring → volatility proxy
- **UI sketch** surfacing the analytics as a real-time dashboard (metrics, selectors, computed outputs).

## Known limitations & guardrails

- **Coverage bias:** headline/social data is heavier toward the most recent dates, which can skew sentiment aggregates.
- **Domain sentiment:** VADER is general-purpose; finance nuance (earnings language, guidance, macro) is imperfectly captured.
- **Data availability:** option chains and intraday bars can have gaps; placeholders are shown when endpoints return no data.
- **Model scope:** Black–Scholes assumes constant volatility and lognormal dynamics; dividends/borrowing costs are not modeled.

## Roadmap / planned enhancements

I plan to return to this project to **enhance data collection**, as there is clearly **more data toward the end** of the date ranges for both **Google RSS feeds** and **Reddit** data, and to **expand to other data sources**. I also plan to **add better sentiment analysis tools** and **enhance model accuracy**.

More concretely:
- **Data collection**
  - Normalize sampling across the full timeline; backfill earlier periods; add caching/retry and de-duplication.
- **Additional sources**
  - Company filings and transcripts, alternative newsfeeds, more granular social data, macro calendars.
- **Sentiment upgrades**
  - Finance-tuned transformers (e.g., FinBERT), entity/sector tagging, horizon-aware aggregation, sarcasm/negation handling.
- **Model accuracy**
  - Calibrate sentiment-to-volatility mapping against realized vol; add hyperparameter tuning for predictive baselines; explore regime detection and richer feature engineering.
