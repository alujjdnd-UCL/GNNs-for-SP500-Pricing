import os
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compute_phase_centralities(method_dir, phase):
    """
    For a given method directory and phase (e.g. 'P0'),
    load the CSV file (if available), build the network, and compute:
      - Betweenness centrality (normalized)
      - Closeness centrality
      - Eigenvector centrality
    Returns a tuple (betw, close, eig) as dictionaries or None if no file is found.
    """
    files = [f for f in os.listdir(method_dir) if f.startswith(phase) and f.endswith('.csv')]
    if not files:
        return None
    filepath = os.path.join(method_dir, files[0])
    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    G = nx.from_pandas_adjacency(df)
    if len(G.nodes()) == 0:
        return None
    betw = nx.betweenness_centrality(G, normalized=True)
    close = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig = {node: np.nan for node in G.nodes()}
    return betw, close, eig


def normalize_dict(d):
    """
    Min–max normalise the values in dictionary d to [0, 1].
    If all values are identical, returns zeros.
    """
    values = np.array(list(d.values()))
    if len(values) == 0:
        return d
    min_val = values.min()
    max_val = values.max()
    if max_val - min_val == 0:
        return {k: 0 for k in d}
    return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}


def generate_rank_distribution_plots(base_dir):
    """
    For each graph construction method (each subdirectory in base_dir):
      1. For each phase (P0..P13), compute betweenness, closeness, and eigenvector centralities.
      2. For each phase, normalize each measure (min–max normalization) and compute a
         combined score (average of the three normalized values) for each stock.
      3. Aggregate the combined scores over phases to compute an overall average per stock.
         Select the top 5 stocks (highest overall average).
      4. For each measure and for each phase, rank stocks (rank 1 = highest normalized value).
         Record the rank (or NaN if a top stock is missing in a phase).
      5. Plot a figure with 3 subplots (one per measure) showing overlapping curves of
         rank evolution for each of the top 5 stocks across phases.
         The lines are coloured using the Seaborn viridis palette (each stock gets one unique, consistent colour).
      6. Add a single global legend on the right side mapping each stock to its colour.
      7. Save the plot as a PNG file.
    """
    # Get method directories
    method_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    method_dirs.sort()

    # Parse method names from directory names
    methods = []
    for dir_name in method_dirs:
        try:
            dir_name_clean = dir_name.replace('\\', '')
            if dir_name_clean.startswith("'") and dir_name_clean.endswith("'"):
                dir_name_clean = dir_name_clean[1:-1]
            method_list = ast.literal_eval(dir_name_clean)
            if isinstance(method_list, list):
                method_name = '_'.join(method_list)
            else:
                method_name = str(method_list)
            methods.append((method_name, dir_name))
        except (SyntaxError, ValueError):
            methods.append((dir_name, dir_name))

    phases = [f'P{i}' for i in range(14)]

    # Process each method
    for method_name, dir_name in methods:
        method_dir = os.path.join(base_dir, dir_name)
        print(f"Processing method: {method_name}")

        # Store per-phase normalized centralities and combined scores
        phase_results = {}  # phase -> {'betw': {}, 'close': {}, 'eig': {}, 'combined': {}}
        overall_combined = {}  # stock -> list of combined scores

        for phase in phases:
            cent = compute_phase_centralities(method_dir, phase)
            if cent is None:
                continue
            betw, close, eig = cent
            norm_betw = normalize_dict(betw)
            norm_close = normalize_dict(close)
            norm_eig = normalize_dict(eig)
            combined = {}
            stocks = set(list(norm_betw.keys()) + list(norm_close.keys()) + list(norm_eig.keys()))
            for stock in stocks:
                nb = norm_betw.get(stock, 0)
                nc = norm_close.get(stock, 0)
                ne = norm_eig.get(stock, 0)
                comb = (nb + nc + ne) / 3
                combined[stock] = comb
                overall_combined.setdefault(stock, []).append(comb)
            phase_results[phase] = {
                'betw': norm_betw,
                'close': norm_close,
                'eig': norm_eig,
                'combined': combined
            }

        # Compute overall average combined score for each stock and select top 5
        overall_avg = {stock: np.mean(scores) for stock, scores in overall_combined.items()}
        top5_stocks = sorted(overall_avg, key=overall_avg.get, reverse=True)[:5]
        print(f"Top 5 stocks for {method_name}: {top5_stocks}")

        # For each measure, record per-phase rank for each top stock.
        # Rank: 1 = highest normalized value.
        measures = ['betw', 'close', 'eig']
        rank_data = {m: {stock: [] for stock in top5_stocks} for m in measures}

        for phase in phases:
            if phase not in phase_results:
                for m in measures:
                    for stock in top5_stocks:
                        rank_data[m][stock].append(np.nan)
                continue
            for m in measures:
                d = phase_results[phase][m]
                if not d:
                    for stock in top5_stocks:
                        rank_data[m][stock].append(np.nan)
                    continue
                # Rank stocks by descending normalized value
                sorted_stocks = sorted(d.items(), key=lambda x: x[1], reverse=True)
                rank_mapping = {stock: rank + 1 for rank, (stock, _) in enumerate(sorted_stocks)}
                for stock in top5_stocks:
                    rank_data[m][stock].append(rank_mapping.get(stock, np.nan))

        # Set up a single viridis palette for the top 5 stocks
        viridis_colors = sns.color_palette("viridis", 5)
        color_mapping = {stock: viridis_colors[i] for i, stock in enumerate(top5_stocks)}

        # Plot rank evolution: 3 subplots (one per measure)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        measure_names = {'betw': 'Betweenness', 'close': 'Closeness', 'eig': 'Eigenvector'}
        x_vals = list(range(len(phases)))  # indices for phases 0...13

        for i, m in enumerate(measures):
            ax = axes[i]
            for stock in top5_stocks:
                y_vals = rank_data[m][stock]
                ax.plot(x_vals, y_vals, marker='o', color=color_mapping[stock])
            ax.set_title(f"{measure_names[m]} Rank Distribution")
            ax.set_ylabel("Rank (1 = highest)")
            ax.invert_yaxis()  # lower rank at the top
            ax.set_xticks(x_vals)
            ax.set_xticklabels(phases)
            ax.grid(True, linestyle='--', alpha=0.5)

        axes[-1].set_xlabel("Temporal Phase")

        # Create a global legend on the right side
        handles = [plt.Line2D([0], [0], color=color_mapping[stock], marker='o', linestyle='-', label=stock)
                   for stock in top5_stocks]
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.92, 0.7))

        # Adjust layout: bring the title closer and reduce extra space for the legend
        fig.suptitle(f"Rank Progression of Top 5 Stocks for {method_name}", fontsize=16, y=0.95)
        plt.tight_layout(rect=[0, 0.4, 0.9, 0.95])

        # Save the plot
        output_dir = os.path.join(base_dir, "rank_distribution_plots")
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, f"top5_rank_distributions_{method_name}.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        print(f"Saved rank distribution plot for {method_name} to {plot_filename}")


if __name__ == "__main__":
    # Set base_dir to your graph CSVs directory
    base_dir = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/data/graphs"
    generate_rank_distribution_plots(base_dir)
