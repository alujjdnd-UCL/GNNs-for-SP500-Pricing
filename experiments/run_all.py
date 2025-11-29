"""
Main experiment script for reproducing results from the paper.

This script trains and evaluates multiple models (MLP, GRU, LSTM, DCRNN, TGCN, A3TGCN)
across different graph construction strategies and market phases.

Usage:
    cd experiments
    export PYTHONPATH=..
    python run_all.py

Results will be saved to metrics.csv in the current directory.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import SP500Stocks
from src.models import MLP, LSTM, GRU, A3TGCN, DCRNN, TGCN
from src.utils.train_and_evaluate import train_and_evaluate_model

# Import configuration
from config.experiment_config import (
    set_seed,
    DATA_ROOT,
    PHASES_DIR,
    PAST_WINDOW,
    FUTURE_WINDOW,
    EPOCHS,
    REPEAT_TIMES,
    TRAIN_SPLIT,
    DEFAULT_TRAINING_PARAMS,
    MODEL_TRAINING_OVERRIDES,
    MODEL_INIT_OVERRIDES,
    MODELS_TO_EVALUATE,
    EDGE_FORMATION_STRATEGIES,
    METRICS_OUTPUT_FILE,
    GRAPHS_OUTPUT_DIR,
    METRICS_COLUMNS,
    DATASET_OPTIONS,
    GRAPH_OPTIONS
)


def main():
    """Run all experiments as described in the paper."""
    
    # Set random seed for reproducibility
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*80)
    print("REPLICATION PACKAGE - EXPERIMENT RUNNER")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Train/Test Split: {TRAIN_SPLIT:.0%}/{1-TRAIN_SPLIT:.0%}")
    print(f"  - Past Window: {PAST_WINDOW}")
    print(f"  - Repeat Times: {REPEAT_TIMES}")
    print(f"  - Models: {', '.join(MODELS_TO_EVALUATE)}")
    print(f"  - Edge Strategies: {len(EDGE_FORMATION_STRATEGIES)}")
    print("="*80)
    
    # Get list of phase files
    phases_path = os.path.join(DATA_ROOT, PHASES_DIR)
    if not os.path.exists(phases_path):
        raise FileNotFoundError(
            f"Phases directory not found: {phases_path}\n"
            f"Please ensure data is organized according to data/documentation/DATA_FORMAT.md"
        )
    
    files = sorted(os.listdir(phases_path))
    files = [f for f in files if f.endswith('.csv')]
    num_phases = len(files)
    
    if num_phases == 0:
        raise ValueError(f"No CSV files found in {phases_path}")
    
    print(f"\nFound {num_phases} market phases to process\n")
    
    # Build model mapping
    model_map = {
        "MLP": MLP,
        "GRU": GRU,
        "LSTM": LSTM,
        "DCRNN": DCRNN,
        "TGCN": TGCN,
        "A3TGCN": A3TGCN
    }
    
    def build_training_params(model_name: str) -> dict:
        params = DEFAULT_TRAINING_PARAMS.copy()
        override = MODEL_TRAINING_OVERRIDES.get(model_name, {})
        params.update(override)
        params.setdefault("num_epochs", EPOCHS)
        return params

    def build_model_kwargs(model_name: str) -> dict:
        return MODEL_INIT_OVERRIDES.get(model_name, {}).copy() if model_name in MODEL_INIT_OVERRIDES else {}

    metrics_file_path = (Path(__file__).parent / METRICS_OUTPUT_FILE).resolve()
    graphs_root = (Path(__file__).parent.parent / GRAPHS_OUTPUT_DIR).resolve()
    metadata_root = (Path(__file__).parent / "run_metadata").resolve()

    metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
    graphs_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(columns=METRICS_COLUMNS)
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"Initialized metrics file: {metrics_file_path}\n")

    # Main experiment loop
    total_experiments = len(EDGE_FORMATION_STRATEGIES) * num_phases * REPEAT_TIMES * len(MODELS_TO_EVALUATE)
    current_experiment = 0

    for strategy_idx, what_edges_to_form in enumerate(EDGE_FORMATION_STRATEGIES, 1):
        print(f"\n{'='*80}")
        print(f"Edge Formation Strategy {strategy_idx}/{len(EDGE_FORMATION_STRATEGIES)}: {what_edges_to_form}")
        print(f"{'='*80}\n")

        for file_idx, file in enumerate(files, 1):
            file_path = f"phases/{file}"
            print(f"\n[Phase {file_idx}/{num_phases}] Processing: {file}")

            current_graph_options = GRAPH_OPTIONS.copy()
            current_dataset_options = DATASET_OPTIONS.copy()

            dataset = SP500Stocks(
                values_file_name=file_path,
                past_window=PAST_WINDOW,
                future_window=FUTURE_WINDOW,
                what_edges_to_form=what_edges_to_form,
                graph_options=current_graph_options,
                dataset_options=current_dataset_options
            )

            if dataset.graph_adj_matrix is None:
                dataset.get_adjacency_matrix()

            # Extract phase and dates from filename (format: P0#DD-MM-YYYY#DD-MM-YYYY.csv)
            file_no_ext = file.split(".")[0]
            phase = file_no_ext.split("#")[0]
            start_date = pd.to_datetime(file_no_ext.split("#")[1], format="%d-%m-%Y")
            end_date = pd.to_datetime(file_no_ext.split("#")[2], format="%d-%m-%Y")

            print(f"  Phase: {phase} | Period: {start_date.date()} to {end_date.date()}")
            print(f"  Dataset size: {len(dataset)} timesteps")

            # Save adjacency matrix for this phase and strategy
            strategy_str = "_".join(what_edges_to_form)
            output_dir = graphs_root / strategy_str
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_name = f"{phase}_{strategy_str}.csv"
            output_path = output_dir / output_file_name
            dataset.save_adj_matrix(file_name=str(output_path))
            print(f"  Saved adjacency matrix to: {output_path}")

            # Run experiments for each model with specified number of repeats
            for repeat in range(REPEAT_TIMES):
                if REPEAT_TIMES > 1:
                    print(f"\n  Repeat {repeat + 1}/{REPEAT_TIMES}")

                for model_name in MODELS_TO_EVALUATE:
                    current_experiment += 1
                    progress = (current_experiment / total_experiments) * 100

                    print(f"\n    [{progress:.1f}%] Training {model_name}...")

                    training_params = build_training_params(model_name)
                    training_params.setdefault("task_title", f"{phase}_{strategy_str}_{model_name}_repeat{repeat + 1}")
                    model_kwargs = build_model_kwargs(model_name)

                    result = train_and_evaluate_model(
                        dataset,
                        model_class=model_map[model_name],
                        train_part=TRAIN_SPLIT,
                        training_params=training_params,
                        model_kwargs=model_kwargs,
                        device=device
                    )

                    test_metrics = result.get("test_metrics", {})
                    train_metrics = result.get("train_metrics", {})
                    portfolio_curve = result.get("portfolio_curve") or []
                    portfolio_final = result.get("portfolio_final_value")

                    mse = test_metrics.get("MSE")
                    rmse = test_metrics.get("RMSE")
                    mae = test_metrics.get("MAE")
                    mre = test_metrics.get("MRE")

                    metrics_row = {
                        "Phase": phase,
                        "Repeat": repeat + 1,
                        "Model": model_name,
                        "Edge Formation": json.dumps(what_edges_to_form),
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MRE": mre,
                        "Portfolio Final Value": portfolio_final,
                        "Portfolio Curve Length": len(portfolio_curve)
                    }

                    metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
                    metrics_df.to_csv(metrics_file_path, index=False)

                    dataset_summary = {
                        "num_timesteps": len(dataset),
                        "past_window": dataset.past_window,
                        "future_window": dataset.future_window,
                        "num_nodes": int(dataset.graph_adj_matrix.shape[0]) if dataset.graph_adj_matrix is not None else None,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d")
                    }

                    metadata = {
                        "phase": phase,
                        "repeat_index": repeat + 1,
                        "edge_formation": what_edges_to_form,
                        "model": model_name,
                        "training_params": training_params,
                        "model_kwargs": model_kwargs,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "portfolio_final_value": portfolio_final,
                        "portfolio_curve_length": len(portfolio_curve),
                        "portfolio_curve_sample": portfolio_curve[:min(10, len(portfolio_curve))] if portfolio_curve else [],
                        "adjacency_matrix_path": str(output_path),
                        "metrics_row": metrics_row,
                        "dataset_summary": dataset_summary,
                        "graph_options": current_graph_options,
                        "dataset_options": current_dataset_options
                    }

                    metadata_path = metadata_root / f"{phase}_{strategy_str}_{model_name}_repeat{repeat + 1}.json"
                    metadata_path.write_text(json.dumps(metadata, indent=2))

                    print(f"      RMSE: {rmse:.4f} | MAE: {mae:.4f} | MRE: {mre:.4f} | Final Portfolio: {portfolio_final}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {metrics_file_path}")
    print(f"Total experiments run: {total_experiments}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
