# Replication Package Manifest

This document provides a file-by-file description of the package contents.

## Root Directory

| File | Description |
|------|-------------|
| `README.md` | Main documentation and quick start guide |
| `INSTALL.md` | Detailed installation instructions |
| `MANIFEST.md` | This file - complete package inventory |
| `LICENSE` | MIT License |
| `requirements.txt` | Python package dependencies with pinned versions |
| `.gitignore` | Git ignore rules for generated files |

---

## `src/` - Source Code

Core implementation of models, datasets, and utilities.

### `src/datasets/` - Dataset Classes

| File | Description | Key Classes/Functions |
|------|-------------|----------------------|
| `SP500Stocks.py` | PyTorch Geometric dataset for S&P 500 stocks | `SP500Stocks` |
| `dataset_utils.py` | Dataset utility functions | `get_graph_in_pyg_format` |
| `__init__.py` | Package initialization | Exports `SP500Stocks` |

#### `src/datasets/csv_processing/` - Data Preprocessing

| File | Description | Key Functions |
|------|-------------|---------------|
| `indicators_processing.py` | Technical indicator calculation | RSI, ALR, normalization |
| `phase_processing.py` | Market phase partitioning algorithm | 200-day window splitting |
| `__init__.py` | Package initialization | |

#### `src/datasets/graph_construction/` - Graph Building

| File | Description | Key Functions |
|------|-------------|---------------|
| `sector_correlation.py` | Graph edge formation methods | `get_sector_graph_wikidata` |
| `__init__.py` | Package initialization | |

### `src/models/` - Model Implementations

| File | Description | Architecture |
|------|-------------|--------------|
| `MLP.py` | Multi-Layer Perceptron baseline | Feed-forward neural network |
| `GRU.py` | Gated Recurrent Unit | Sequential model with GRU cells |
| `LSTM.py` | Long Short-Term Memory | Sequential model with LSTM cells |
| `DCRNN.py` | Diffusion Convolutional RNN | Spatiotemporal graph model |
| `TGCN.py` | Temporal Graph Convolutional Network | Graph + GRU hybrid |
| `A3TGCN.py` | Attention Temporal GCN | TGCN with attention mechanism |
| `TGCNCell.py` | TGCN cell implementation | Used by TGCN and A3TGCN |
| `GAT.py` | Graph Attention Network | Spatial graph model (not used in paper) |
| `GCN.py` | Graph Convolutional Network | Spatial graph model (not used in paper) |
| `MA_eval.py` | Moving Average baseline | Statistical baseline evaluator |
| `ARIMA_eval.py` | ARIMA baseline | Time series baseline evaluator |
| `__init__.py` | Package initialization | Exports all models |

**Note**: `GAT.py` and `GCN.py` are included for completeness but were not used in the final paper experiments.

### `src/utils/` - Training and Evaluation

| File | Description | Key Functions |
|------|-------------|---------------|
| `train.py` | Model training loop | `train` |
| `evaluate.py` | Evaluation metrics and portfolio simulation | `get_regression_error`, `simulate_long_short_portfolio` |
| `train_and_evaluate.py` | Combined training+evaluation pipeline | `train_and_evaluate_model` |
| `__init__.py` | Package initialization | Exports key functions |

---

## `experiments/` - Experiment Scripts

Scripts for running experiments and reproducing results.

### Main Scripts

| File | Description | Usage |
|------|-------------|-------|
| `run_all.py` | Main replication script | `python run_all.py` |
| `__init__.py` | Package initialization | |

### `experiments/config/` - Configuration

| File | Description | Purpose |
|------|-------------|---------|
| `experiment_config.py` | Hyperparameters and settings | Central configuration for all experiments |
| `__init__.py` | Package initialization | |

**Key Configuration Parameters**:
- `EPOCHS = 12` - Training epochs per model
- `TRAIN_SPLIT = 0.75` - Train/test split ratio
- `PAST_WINDOW = 25` - Lookback window size
- `RANDOM_SEED = 42` - Reproducibility seed
- `MODELS_TO_EVALUATE` - List of models to run
- `EDGE_FORMATION_STRATEGIES` - Graph construction methods

### `experiments/notebooks/` - Jupyter Notebooks

| File | Description | Purpose |
|------|-------------|---------|
| `lseg-data-collection.ipynb` | Data collection example | Demonstrates LSEG Refinitiv data extraction |

**Note**: This notebook shows the methodology but cannot be run without LSEG access.

---

## `data/` - Data Directory

Contains data files and documentation (raw data not included).

### Main Data Script

| File | Description | Purpose |
|------|-------------|---------|
| `get_wikidata.py` | Wikidata knowledge graph fetcher | Retrieves company relationships from Wikidata |

### `data/documentation/` - Data Format Specs

| File | Description |
|------|-------------|
| `DATA_FORMAT.md` | Complete data format specification |

**Documented Formats**:
- `filtered_symbols.csv` - Stock-to-industry mapping
- `aggregated_cleaned.csv` - Full historical data with indicators
- `phases/P*#DD-MM-YYYY#DD-MM-YYYY.csv` - Market phase files
- NumPy adjacency matrices

### `data/SP500/` - S&P 500 Data (User-Provided)

**Directory Structure** (files not included):
```
SP500/
├── filtered_symbols.csv          # Stock symbols and industries
├── symbol_industry.csv           # Alternative industry mapping
└── raw/
    ├── aggregated_cleaned.csv    # Historical data with indicators
    ├── prices_500_truncated.csv  # OHLCV price data
    ├── adj_500_sectors_only.npy  # Pre-computed sector adjacency
    └── phases/                    # 200-day phase files
        ├── P0#23-08-2013#10-06-2014.csv
        ├── P1#11-06-2014#26-03-2015.csv
        └── ...
```

### `data/graphs/` - Generated Graphs (Output)

**Generated during experiments**:
```
graphs/
├── wikidata_correlation/
│   ├── P0_wikidata_correlation.csv
│   └── ...
├── wikidata_sectors/
├── correlation_sectors/
└── wikidata_correlation_sectors/
```

---

## `analysis/` - Post-Processing Scripts

Visualization and analysis scripts for generating paper figures.

| File | Description | Generates |
|------|-------------|-----------|
| `README.md` | Analysis documentation | - |
| `model_comparison.py` | Model performance comparison | Line/bar plots |
| `metrics_heatmap.py` | Performance heatmaps | Metric heatmaps |
| `graph_centrality.py` | Graph centrality analysis | Centrality distributions |
| `graph_centrality_heatmap.py` | Centrality heatmaps | Heatmap visualizations |
| `graph_centrality_indiv.py` | Individual stock centrality | Stock-specific plots |
| `centrality_top.py` | Top central stocks | Rankings |
| `graph_degrees.py` | Degree distribution analysis | Histograms |
| `graph_construction_violin.py` | Edge strategy comparison | Violin plots |
| `graph_cons_error_megaplot.py` | Comprehensive error analysis | Multi-panel plots |
| `market_conditions.py` | Market condition analysis | Statistical summaries |
| `market_conditions_plot.py` | Market condition plots | Time series plots |

**Requires**: Completed experiments (`experiments/metrics.csv`) and graph files.

---

## `tests/` - Testing and Validation

Scripts for validating the environment setup.

| File | Description | Usage |
|------|-------------|-------|
| `validate_setup.py` | Environment validation script | `python validate_setup.py` |

**Checks**:
- Python version (3.8+)
- PyTorch installation and CUDA availability
- PyTorch Geometric and Temporal
- All model/dataset imports
- Directory structure
- Configuration loading

---

## Generated Files (Not Tracked)

These files are created during experiments and analysis:

### Experiment Outputs
- `experiments/metrics.csv` - Aggregated experimental results
- `data/graphs/**/*.csv` - Graph adjacency matrices
- `data/SP500/processed/*.pt` - Processed PyG data objects

### Analysis Outputs
- `analysis/figures/` - Generated plots
- `analysis/*.png`, `*.pdf` - Saved visualizations
- `analysis/*.csv` - Analysis result tables

### Python Artifacts
- `**/__pycache__/` - Compiled Python bytecode
- `*.pyc`, `*.pyo` - Bytecode files

---

## File Count Summary

| Category | File Count | Description |
|----------|------------|-------------|
| Documentation | 5 | README, INSTALL, MANIFEST, data docs, analysis docs |
| Source Code | 23 | Models (11), Datasets (3), Utils (3), Config (1), Init files (5) |
| Experiments | 2 | Main script + notebook |
| Analysis | 13 | Visualization scripts + README |
| Tests | 1 | Validation script |
| Configuration | 3 | requirements.txt, .gitignore, LICENSE |
| **Total** | **47** | Tracked files in repository |

---

## Version Information

- **Package Version**: 1.0.0
- **Python**: 3.8+
- **PyTorch**: 2.5.1+cu118
- **PyTorch Geometric**: 2.6.1
- **PyTorch Geometric Temporal**: 0.54.0

---

## Reproducibility Notes

All code is deterministic when `RANDOM_SEED` is set in configuration:
- Python random seed
- NumPy random seed  
- PyTorch manual seed
- CUDA deterministic operations

---

## Contact

For questions about specific files or package structure, refer to:
- File-specific docstrings and comments
- README.md for overview
- INSTALL.md for setup issues
- Paper for methodology details

---

## Change Log

**v1.0.0** (Initial Release)
- Complete replication package for journal submission
- Cleaned and refactored codebase
- Comprehensive documentation
- Reproducibility features (seeds, validation)
