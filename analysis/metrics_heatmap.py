import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast


def normalize_edge_formation(edge_str):
    """
    Given a string representing a list of edge formation components (e.g.,
    "['wikidata', 'correlation']"), parse it and return a normalized string
    representing the entire configuration, e.g. "wikidata, correlation".
    """
    try:
        lst = ast.literal_eval(edge_str)
        if isinstance(lst, list):
            return ", ".join(lst)
        return edge_str
    except Exception:
        return edge_str


def generate_heatmaps(csv_file, output_dir="heatmaps"):
    """
    Reads the CSV file (metrics_to_plot.csv), processes the data to normalize the
    edge formation configuration, filters out phases beyond P13, and generates separate
    heatmaps for each metric.

    In each heatmap:
      - Rows: Phases (P0 to P13, sorted in order)
      - Columns: Unique edge formation configurations (e.g. "wikidata, correlation",
        "correlation, sectors", etc.)
      - Cells: Average metric value (if multiple entries exist per combination)
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Filter phases: keep only those with Phase number less than 14 (i.e. P0 to P13)
    df = df[df['Phase'].apply(lambda x: int(x.replace("P", "")) < 14)]

    # Create a helper column to sort phases numerically
    df['Phase_num'] = df['Phase'].apply(lambda x: int(x.replace("P", "")))

    # Normalize the edge formation column to represent the entire configuration.
    df['EdgeFormation'] = df['Edge Formation'].apply(normalize_edge_formation)

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # List the metrics to generate heatmaps for.
    metrics = ['MSE', 'RMSE', 'MAE', 'MRE', 'Portfolio Performance']

    # For each metric, create a pivot table and generate a heatmap.
    for metric in metrics:
        pivot = df.pivot_table(index='Phase', columns='EdgeFormation', values=metric, aggfunc='mean')
        # Sort rows by numeric phase order (P0, P1, ..., P13)
        pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.replace("P", ""))))

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"Heatmap of {metric} by Phase and Edge Formation")
        plt.ylabel("Phase")
        plt.xlabel("Edge Formation")
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{metric}_heatmap.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved heatmap for {metric} to {output_file}")


if __name__ == "__main__":
    csv_file = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/metrics_to_plot.csv"
    generate_heatmaps(csv_file)
