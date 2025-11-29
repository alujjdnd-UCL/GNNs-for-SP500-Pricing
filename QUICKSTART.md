# Quick Start Guide

One-page guide for running the replication package from start to finish.

## Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 11.8+ (recommended)
- Your own S&P 500 stock data (see data format below)

## Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install PyTorch
Follow the official selector at https://pytorch.org/get-started/locally/ to obtain the command for your OS, Python version, and CUDA toolkit. Run the suggested command in the activated environment.

# 3. Install PyTorch Geometric extensions
Use the matching wheels from https://data.pyg.org/whl/ (see INSTALL.md for the exact commands).

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Verify installation
cd tests
export PYTHONPATH=..
python validate_setup.py
```

## Data Preparation (variable time)

You must provide your own data. Required format:

### Directory Structure
```
data/SP500/
├── filtered_symbols.csv          # Stock→Industry mapping
├── symbol_industry.csv           # Alternative mapping
└── raw/
    ├── aggregated_cleaned.csv    # Full data with indicators
    ├── prices_500_truncated.csv  # OHLCV prices
    └── phases/                    # Market phases (200-day windows)
        ├── P0#23-08-2013#10-06-2014.csv
        ├── P1#11-06-2014#26-03-2015.csv
        └── ... (15 total phases)
```

### Required Columns
Each CSV must have:
- `Date`, `Symbol`, `Open`, `High`, `Low`, `Close`, `Volume`
- `NormClose`, `ALR1W`, `ALR2M`, `RSI` (technical indicators)

**See `data/documentation/DATA_FORMAT.md` for complete specifications.**

## Run Experiments (2-6 hours on RTX 3090)

```bash
cd experiments
export PYTHONPATH=..  # Windows: set PYTHONPATH=..
python run_all.py
```

This will:
- Process all 15 market phases
- Test 4 graph construction strategies
- Train 6 models (MLP, GRU, LSTM, DCRNN, TGCN, A3TGCN)
- Save metrics to `metrics.csv`
- Write per-run metadata JSON files to `run_metadata/`

## View Results

```bash
# Check metrics
cat experiments/metrics.csv

# Generate visualizations
cd analysis
python model_comparison.py
python graph_centrality.py
```

## Configuration

Edit hyperparameters in `experiments/config/experiment_config.py`:
```python
EPOCHS = 12                # Training epochs
TRAIN_SPLIT = 0.75        # 75% train, 25% test
PAST_WINDOW = 25          # Lookback window
RANDOM_SEED = 42          # For reproducibility
```

## Expected Output

After completion, you should have:
- `experiments/metrics.csv` - Performance metrics (MSE, RMSE, MAE, MRE, portfolio summary)
- `experiments/run_metadata/*.json` - Detailed per-run metadata for reproducibility
- `data/graphs/` - Generated adjacency matrices
- `analysis/figures/` - Visualizations (after running analysis scripts)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size in model files |
| "No module named 'src'" | Set `PYTHONPATH` before running |
| Need live Wikidata edges | Enable `GRAPH_OPTIONS["allow_external_requests"] = True` in config + provide internet access |
| Missing data files | Prepare data per `DATA_FORMAT.md` |
| Slow performance | Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"` |

## Full Documentation

- **README.md** - Complete overview
- **INSTALL.md** - Detailed installation guide
- **MANIFEST.md** - File-by-file description
- **data/documentation/DATA_FORMAT.md** - Data specifications (includes Wikidata caching notes)
- **analysis/README.md** - Visualization guide
- **experiments/config/experiment_config.py** - Hyperparameters, dataset & graph options

## Support

For issues:
1. Check error messages
2. Review documentation
3. Validate setup: `cd tests && python validate_setup.py`
4. Contact authors or open issue
