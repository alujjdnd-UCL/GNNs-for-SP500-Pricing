"""
Configuration file for replication experiments.

This file contains all hyperparameters and experiment settings used in the study.
Modify these values to customize experiments or reproduce results with different settings.
"""

import random
import numpy as np
import torch

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================
RANDOM_SEED = 42  # Set to None for non-deterministic behavior

def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility across runs."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed} for reproducibility")
    else:
        print("Running in non-deterministic mode")


# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_ROOT = "../data/SP500/"
PHASES_DIR = "raw/phases"
PAST_WINDOW = 25  # Number of past timesteps to use for prediction
FUTURE_WINDOW = 1  # Number of future timesteps to predict


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
EPOCHS = 12
REPEAT_TIMES = 1  # Number of times to repeat each experiment
TRAIN_SPLIT = 0.75  # Proportion of data used for training

# Default hyperparameters applied to every model unless overridden below
DEFAULT_TRAINING_PARAMS = {
    "batch_size": 32,
    "lr": 5e-3,
    "weight_decay": 1e-5,
    "num_epochs": EPOCHS,
    "hidden_size": 64,
    "layers_nb": 2,
    "plot": False,
}

# Per-model overrides (update to match paper-specific hyperparameters)
MODEL_TRAINING_OVERRIDES = {
    "MLP": {
        "hidden_size": 128,
        "layers_nb": 3,
        "lr": 1e-3,
    },
    "GRU": {
        "hidden_size": 128,
        "lr": 1e-3,
    },
    "LSTM": {
        "hidden_size": 128,
        "lr": 1e-3,
    },
    "DCRNN": {
        "batch_size": 16,
        "hidden_size": 64,
        "lr": 1e-3,
    },
    "TGCN": {
        "batch_size": 16,
        "hidden_size": 64,
        "lr": 1e-3,
    },
    "A3TGCN": {
        "batch_size": 16,
        "hidden_size": 64,
        "lr": 1e-3,
    },
}

# Additional keyword arguments passed to model constructors
MODEL_INIT_OVERRIDES = {
    "TGCN": {"use_gat": False},
    "A3TGCN": {"use_gat": False},
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Models to evaluate (as used in the paper)
MODELS_TO_EVALUATE = [
    "MLP",
    "GRU", 
    "LSTM",
    "DCRNN",
    "TGCN",
    "A3TGCN"
]

# Baseline models (commented out in paper experiments)
# BASELINE_MODELS = ["MA", "ARIMA"]


# ============================================================================
# GRAPH CONSTRUCTION CONFIGURATION
# ============================================================================
# Edge formation strategies evaluated in the study
EDGE_FORMATION_STRATEGIES = [
    ["wikidata", "correlation"],
    ["wikidata", "sectors"],
    ["correlation", "sectors"],
    ["wikidata", "correlation", "sectors"]
]

# Alternative strategies (not used in paper but available for exploration)
# ADDITIONAL_STRATEGIES = [
#     ["wikidata"],
#     ["correlation"],
#     ["sectors"]
# ]


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# Output configuration
METRICS_OUTPUT_FILE = "metrics.csv"
GRAPHS_OUTPUT_DIR = "data/graphs"

# Metrics to track
METRICS_COLUMNS = [
    "Phase",
    "Repeat", 
    "Model",
    "Edge Formation",
    "MSE",
    "RMSE",
    "MAE",
    "MRE",
    "Portfolio Final Value",
    "Portfolio Curve Length"
]


# ============================================================================
# DATASET & GRAPH OPTIONS
# ============================================================================

# Additional options forwarded to the dataset / graph construction utilities
DATASET_OPTIONS = {
    "verbose": False,
    "force_reload": False,
}

GRAPH_OPTIONS = {
    "visualise": False,
    "enable_wikidata": True,
    "allow_external_requests": False,
    # Relative to the dataset root; will be resolved in run_all.py
    "cache_dir": "cache",
    "verbose": False,
}
