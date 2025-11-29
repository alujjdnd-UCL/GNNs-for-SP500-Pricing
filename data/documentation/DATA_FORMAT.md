# Data Format Specification

This document describes the required data format for replicating the experiments. **Raw data is not included** in this package due to licensing restrictions.

## Overview

The experiments require daily stock price data for S&P 500 companies covering the period from 2013 to 2024, organized into non-overlapping 200-day market phases.

## Directory Structure

```
data/
├── SP500/
│   ├── filtered_symbols.csv          # Stock symbols and industry mapping
│   ├── symbol_industry.csv           # Alternative industry mapping file
│   └── raw/
│       ├── adj_500_sectors_only.npy  # Pre-computed adjacency matrix (sectors only)
│       ├── aggregated_cleaned.csv    # Full historical data with indicators
│       ├── prices_500_truncated.csv  # OHLCV price data
│       └── phases/                    # Market phase files
│           ├── P0#23-08-2013#10-06-2014.csv
│           ├── P1#11-06-2014#26-03-2015.csv
│           ├── ...
│           └── P14#09-10-2024#06-12-2024.csv
```

## Required Files

### 1. `filtered_symbols.csv`

Maps stock ticker symbols to their industry sectors.

**Format**: CSV with header
**Columns**:
- `Symbol` (string): Stock ticker symbol (e.g., "AAPL", "MSFT")
- `Industry` (string): Industry classification (e.g., "Technology", "Healthcare")

**Example**:
```csv
Symbol,Industry
AAPL,Technology
MSFT,Technology
JPM,Financials
JNJ,Healthcare
```

**Notes**:
- Should contain ~500 S&P 500 stocks
- Industry classifications used for sector-based graph edges
- Symbols must match those in price data files

---

### 2. `symbol_industry.csv`

Alternative industry mapping file (same format as `filtered_symbols.csv`).

This file may be used as a fallback or alternative industry classification scheme.

---

### 3. `raw/aggregated_cleaned.csv`

Complete historical stock data with technical indicators.

**Format**: CSV with header (long format - one row per stock per date)

**Columns**:
- `Date` (string): Trading date in format "YYYY-MM-DD" (e.g., "2013-08-23")
- `Symbol` (string): Stock ticker symbol
- `Open` (float): Opening price
- `High` (float): Highest price
- `Low` (float): Lowest price
- `Close` (float): Closing price (adjusted for splits/dividends)
- `Volume` (int): Trading volume
- `NormClose` (float): Normalized closing price
- `ALR1W` (float): Average Log Return over 1 week
- `ALR2M` (float): Average Log Return over 2 months
- `RSI` (float): Relative Strength Index (14-day)

**Example**:
```csv
Date,Symbol,Open,High,Low,Close,Volume,NormClose,ALR1W,ALR2M,RSI
2013-08-23,AAPL,505.00,506.50,502.00,504.50,45123000,1.000,0.0015,-0.0023,55.2
2013-08-23,MSFT,32.50,32.75,32.40,32.60,28456000,1.000,0.0008,0.0012,58.7
2013-08-26,AAPL,504.00,508.00,503.50,507.25,42367000,1.005,0.0018,-0.0020,56.1
```

**Notes**:
- Long format: Each row represents one stock on one date
- Dates must be consecutive trading days (excluding weekends/holidays)
- All ~500 stocks should have entries for each date
- Missing data should be handled via forward-fill or removal
- Technical indicators calculated using the `ta` library (see preprocessing scripts)

---

### 4. `raw/prices_500_truncated.csv`

OHLCV price data without technical indicators (subset of `aggregated_cleaned.csv`).

**Format**: CSV with header (long format)

**Columns**:
- `Date` (string): Trading date "YYYY-MM-DD"
- `Symbol` (string): Stock ticker
- `Open`, `High`, `Low`, `Close` (float): OHLC prices
- `Volume` (int): Trading volume

This file is used as an intermediate data product before indicator calculation.

---

### 5. `raw/phases/P*#DD-MM-YYYY#DD-MM-YYYY.csv`

Market phase files containing 200-day non-overlapping intervals.

**Naming Convention**: `P{phase_number}#{start_date}#{end_date}.csv`
- Phase number: Sequential integer (P0, P1, P2, ...)
- Dates: Format DD-MM-YYYY (e.g., 23-08-2013)

**Format**: Same as `aggregated_cleaned.csv` (long format with indicators)

**Columns**: Same as `aggregated_cleaned.csv`:
- Date, Symbol, Open, High, Low, Close, Volume
- NormClose, ALR1W, ALR2M, RSI

**Example Filenames**:
```
P0#23-08-2013#10-06-2014.csv    # 200 trading days
P1#11-06-2014#26-03-2015.csv    # 200 trading days
P2#27-03-2015#11-01-2016.csv    # 200 trading days
...
P14#09-10-2024#06-12-2024.csv   # Final phase
```

**Notes**:
- Each phase contains exactly 200 trading days (non-overlapping)
- Phases are chronologically sequential
- Used for train/test splitting and temporal evaluation
- Models are trained on first 75% (150 days) and tested on last 25% (50 days)

---

### 6. `raw/adj_500_sectors_only.npy`

Pre-computed adjacency matrix for sector-based graph.

**Format**: NumPy binary file (.npy)

**Contents**: 2D array of shape (N, N) where N ≈ 500 stocks
- Binary adjacency matrix: 1 if stocks in same sector, 0 otherwise
- Symmetric matrix (undirected graph)
- Diagonal can be 0 (no self-loops) or 1 (with self-loops)

**Loading**:
```python
import numpy as np
adj_matrix = np.load('adj_500_sectors_only.npy')
```

**Notes**:
- Used as a baseline graph structure
- Can be regenerated from `filtered_symbols.csv` using graph construction scripts
- Optional: Can be recomputed at runtime if not provided

---

## Technical Indicators

### NormClose (Normalized Close)
Closing price normalized by the first closing price in the window:
```
NormClose = Close / Close[0]
```

### ALR1W (Average Log Return - 1 Week)
Mean of daily log returns over the past 5 trading days:
```
ALR1W = mean(log(Close[t] / Close[t-1])) for t in [-5, -1]
```

### ALR2M (Average Log Return - 2 Months)
Mean of daily log returns over the past ~40 trading days:
```
ALR2M = mean(log(Close[t] / Close[t-1])) for t in [-40, -1]
```

### RSI (Relative Strength Index)
14-period RSI calculated using the `ta` library:
```python
from ta.momentum import RSIIndicator
rsi = RSIIndicator(close=close_prices, window=14)
```

Range: 0 to 100 (typically 30 = oversold, 70 = overbought)

---

## Data Sources

The experiments use data from **LSEG Refinitiv Eikon** (historical stock prices). Alternative data sources include:
- Yahoo Finance (via `yfinance` Python library)
- Alpha Vantage
- Quandl
- Bloomberg Terminal
- Local financial data provider

**Important**: Ensure data is adjusted for stock splits and dividends.

---

## Data Preprocessing

See the following scripts for reference preprocessing:
- `src/datasets/csv_processing/indicators_processing.py` - Technical indicator calculation
- `src/datasets/csv_processing/phase_processing.py` - Market phase partitioning
- `experiments/notebooks/lseg-data-collection.ipynb` - Example data collection

---

## Data Collection Methodology

As described in the paper:
1. Download daily OHLCV data for all S&P 500 constituents
2. Filter stocks with insufficient history (< 11 years of data)
3. Adjust prices for splits and dividends
4. Calculate technical indicators (RSI, ALR1W, ALR2M, NormClose)
5. Partition into 200-day non-overlapping phases
6. Export to CSV format

### Wikidata Caching

- The replication scripts cache Wikidata lookups in `data/SP500/cache/` (created automatically).
- By default external SPARQL requests are **disabled**. Populate caches manually or enable live queries via `GRAPH_OPTIONS["allow_external_requests"] = True` in `experiments/config/experiment_config.py`.
- Cached files include `wikidata_symbol_qids.json` and `wikidata_relations_adj.npy`. Delete them to rebuild from scratch (if external requests are allowed).

---

## Validation

To validate your data files, ensure:
- [ ] All required files are present in correct locations
- [ ] CSV files have the specified column names (exact match)
- [ ] Date formats are consistent (YYYY-MM-DD for data, DD-MM-YYYY for phase filenames)
- [ ] No missing values (NaN) in indicator columns
- [ ] Approximately 500 unique stock symbols
- [ ] Phase files contain exactly 200 trading days each
- [ ] Dates are in chronological order

---

## Contact

For questions about data format or preprocessing, please refer to the paper methodology section or contact the authors.
