import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def create_graph_matrix_histograms(base_dir: str, output_file: str = "graph_matrix_histograms.png") -> None:
    """
    Creates a matrix of degree distribution histograms from graph CSV files,
    with rows as phases and columns as graph construction methods.

    Each histogram is plotted with only a fill colour and no border, so the different
    bar widths don't look weird.

    Assumes a folder structure like:
      base_dir/
          correlation/
              P1_correlation.csv
              P2_correlation.csv
              ...
          sectors/
              P1_sectors.csv
              P2_sectors.csv
              ...

    Each CSV filename starts with the phase (e.g., "P1") and the folder name gives the method.
    Only the top row (columns) and leftmost column (rows) are labeled.
    """
    matrix_data = {}
    methods = set()
    phases = set()

    # Traverse each method folder to build the data matrix.
    for method in os.listdir(base_dir):
        method_dir = os.path.join(base_dir, method)
        if os.path.isdir(method_dir):
            for file in os.listdir(method_dir):
                if file.endswith('.csv'):
                    # Extract phase from filename, e.g., "P5_correlation.csv" -> "P5"
                    phase = file.split('_')[0]
                    phases.add(phase)
                    methods.add(method)
                    full_path = os.path.join(method_dir, file)
                    matrix_data[(method, phase)] = full_path

    # Sort phases numerically (P1, P2, ...) and methods alphabetically.
    phases = sorted(phases, key=lambda x: int(x[1:]))
    methods = sorted(methods)

    n_rows = len(phases)
    n_cols = len(methods)

    # Create a figure with a subplot grid.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    # Plot the histogram for each cell.
    for i, phase in enumerate(phases):
        for j, method in enumerate(methods):
            ax = axes[i][j]
            key = (method, phase)
            if key in matrix_data:
                # Load the CSV and construct the graph.
                df = pd.read_csv(matrix_data[key], index_col=0)
                G = nx.from_pandas_adjacency(df)

                # Compute the degree series.
                degree_series = pd.Series(dict(G.degree()))

                # Determine bins (using integer degree values).
                bins = range(int(degree_series.min()), int(degree_series.max()) + 2)

                # Plot histogram with only fill colour and no border.
                ax.hist(degree_series, bins=bins, edgecolor='none', color='darkblue')
            else:
                ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', fontsize=12)

            # Remove individual x and y labels.
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Label the top row with method names.
            if i == 0:
                ax.set_title(method, fontsize=14)
            # Label the leftmost column with phase names.
            if j == 0:
                ax.set_ylabel(phase, rotation=0, labelpad=40, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()
    print(f"Saved composite graph matrix histograms to {output_file}")


if __name__ == "__main__":
    # Update the base_dir path as needed.
    base_dir = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/data/graphs"
    create_graph_matrix_histograms(base_dir)
