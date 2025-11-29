# Experiments

This directory contains the main replication script and configuration for reproducing all experimental results.

## Quick Start

```bash
cd experiments
export PYTHONPATH=..  # Windows: set PYTHONPATH=..
python run_all.py
```

## Structure

```
experiments/
├── run_all.py           # Main experiment script
├── config/
│   └── experiment_config.py  # Hyperparameters and settings
└── notebooks/
    └── lseg-data-collection.ipynb  # Data collection methodology
```

## Main Script: `run_all.py`

Orchestrates all experiments across:
- **15 market phases** (2013-2024, 200-day non-overlapping windows)
- **4 graph construction strategies**:
  - Wikidata + Correlation
  - Wikidata + Sectors
  - Correlation + Sectors
  - Wikidata + Correlation + Sectors
- **6 models**: MLP, GRU, LSTM, DCRNN, TGCN, A3TGCN

### Output

Results are saved to `metrics.csv` with columns:
- Phase, Repeat, Model, Edge Formation
- MSE, RMSE, MAE, MRE, Portfolio Performance

Graph adjacency matrices are saved to `../data/graphs/{strategy}/`

## Configuration

All hyperparameters are defined in `config/experiment_config.py`:

### Key Parameters
- `EPOCHS = 12` - Training epochs per model
- `TRAIN_SPLIT = 0.75` - Train/test split ratio (75%/25%)
- `PAST_WINDOW = 25` - Lookback window for predictions
- `RANDOM_SEED = 42` - Random seed for reproducibility

### Modifying Experiments

To change which models are evaluated:
```python
# In config/experiment_config.py
MODELS_TO_EVALUATE = [
    "MLP",
    "LSTM",
    "TGCN",
    "A3TGCN"  # Remove models you don't want to run
]
```

To change graph construction strategies:
```python
EDGE_FORMATION_STRATEGIES = [
    ["wikidata", "correlation"],
    ["wikidata", "sectors"]  # Fewer strategies = faster execution
]
```

## Execution Flow

1. **Initialization**
   - Set random seed for reproducibility
   - Load configuration parameters
   - Validate data directory structure

2. **For each edge formation strategy:**
   - For each market phase:
     - Load phase data (200 days)
     - Construct graph with specified strategy
     - Save adjacency matrix
     - For each model:
       - Split data (75% train, 25% test)
       - Train model on training set
       - Evaluate on test set
       - Calculate metrics (MSE, RMSE, MAE, MRE)
       - Simulate portfolio performance
       - Append results to metrics.csv

3. **Completion**
   - Display summary statistics
   - Save all results

## Expected Runtime

On NVIDIA RTX 3090 GPU:
- **Per model/phase**: ~1-5 minutes
- **Full replication**: ~2-6 hours

Total experiments: 15 phases × 4 strategies × 6 models = **360 experiments**

## Progress Monitoring

The script provides real-time progress updates:
```
[25.0%] Training LSTM...
  Phase: P5 | Period: 2018-05-31 to 2019-03-18
  RMSE: 0.0234 | MAE: 0.0187 | MRE: 0.0156
```

## Checkpointing

Results are written to `metrics.csv` after each model completion, so:
- You can stop and resume (manually edit to continue from last phase)
- Partial results are preserved
- No need to rerun completed experiments

## Data Requirements

Ensure data is prepared before running:
1. Check `../data/SP500/raw/phases/` contains phase CSV files
2. Verify data format matches `../data/documentation/DATA_FORMAT.md`
3. Run validation: `cd ../tests && python validate_setup.py`

## Troubleshooting

### Issue: "Phases directory not found"
**Solution**: Place your data in `data/SP500/raw/phases/` following the naming convention `P*#DD-MM-YYYY#DD-MM-YYYY.csv`

### Issue: CUDA out of memory
**Solution**: 
- Reduce batch size in individual model files
- Use fewer models (edit `MODELS_TO_EVALUATE`)
- Run on CPU (slower): Set `device = 'cpu'` in model files

### Issue: Slow execution
**Solution**:
- Verify GPU is being used: `torch.cuda.is_available()`
- Reduce number of phases or strategies for testing
- Increase batch size if GPU memory allows

### Issue: Different results than paper
**Solution**:
- Ensure `RANDOM_SEED` is set to 42
- Verify data preprocessing matches paper methodology
- Check PyTorch/CUDA versions match requirements

## Notebooks

### `lseg-data-collection.ipynb`

Example notebook showing data collection methodology using LSEG Refinitiv Eikon.

**Note**: This notebook demonstrates the process but requires:
- LSEG Refinitiv Eikon subscription
- Eikon Desktop Application running
- Eikon Python API credentials

The notebook is included for transparency and methodology documentation, but is not required for running experiments with pre-collected data.

## Extending the Package

### Adding New Models

1. Create model class in `../src/models/YourModel.py`
2. Add import to `../src/models/__init__.py`
3. Add to `MODELS_TO_EVALUATE` in `config/experiment_config.py`
4. Run experiments

### Adding New Graph Strategies

1. Modify `../src/datasets/graph_construction/sector_correlation.py`
2. Add new strategy to `EDGE_FORMATION_STRATEGIES`
3. Run experiments

### Custom Evaluation Metrics

1. Modify `../src/utils/evaluate.py`
2. Update `METRICS_COLUMNS` in config
3. Update `run_all.py` to extract new metrics

## Contact

For questions about experiments or configuration:
- See main README.md for overview
- Check INSTALL.md for setup issues
- Refer to paper for methodology details
