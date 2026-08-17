# Data Directory

This directory stores historical market data used for backtesting.

## Data Sources

Market data is sourced from **Databento** using the `/databento` API.

## File Structure

```
data/
├── {INSTRUMENT}_1m.parquet    # 1-minute OHLCV data
├── {INSTRUMENT}_1d.parquet    # Daily OHLCV data
└── cache/                      # Temporary cache files
```

## Supported Instruments

- **NQ** - Nasdaq-100 E-mini Futures
- **ES** - S&P 500 E-mini Futures
- **RTY** - Russell 2000 E-mini Futures
- **GC** - Gold Futures
- **CL** - Crude Oil Futures
- **BTC** - Bitcoin
- **ETH** - Ethereum

## Regenerating Data

Data files are **not committed to git** (excluded via `.gitignore`).

To regenerate historical data:

```python
from src.data_layer import ensure_data

# Downloads and caches data automatically
df = ensure_data("NQ")
```

Data is cached locally after first download and automatically updated when stale.

## Note

Data files can be large (hundreds of MB). They are excluded from version control to keep the repository lightweight.
