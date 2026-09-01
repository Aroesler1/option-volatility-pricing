# Data provenance

**Two sources.**

1. **yfinance** (public) — daily SPY history, 2005-2026, for the long-sample forecasting benchmark.
2. **Databento `XNAS.ITCH`** (Berkeley MFE programme account) — SPY 1-minute bars, 2018-2026, aggregated to 5-minute returns for true intraday realized variance and realized quarticity.

## What is committed

- Source code, tests, and the notebook
- `data/SPY_intraday_rv.csv`: 2,094 days of derived daily realized variance and quarticity. Aggregated, small, and sufficient to reproduce the estimator comparison.
- Derived results under `results/`

## What is not committed

- `data/databento/` (gitignored): raw 1-minute bar extracts (~23 MB compressed)

## Reproducing

```bash
python run_vol_benchmark.py --ticker SPY --start 2005-01-01   # public data only
python run_intraday_benchmark.py                              # uses the committed derived series
```

The first needs no credentials. The second runs from the committed derived file.

## Licence and retention

Databento data is accessed under a programme licence; only a derived daily aggregate is published. yfinance data is public.
