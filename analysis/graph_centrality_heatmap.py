import os
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def generate_centrality_plots(base_dir):
    """
    Generates matrix of plots of graphs colored by centrality measures.
    Each centrality measure (betweenness, closeness, eigenvector) has a separate mega-plot.
    Rows correspond to phases P0 to P13, columns correspond to different graph construction methods.
    """
    # Get list of method directories
    method_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    method_dirs.sort()  # Ensure consistent ordering
    print(f"Found method directories: {method_dirs}")

    # Parse method names
    methods = []
    for dir_name in method_dirs:
        try:
            # Clean up the directory name
            dir_name_clean = dir_name.replace('\\', '')
            if dir_name_clean.startswith("'") and dir_name_clean.endswith("'"):
                dir_name_clean = dir_name_clean[1:-1]
            # Attempt to parse the cleaned directory name as a list
            method_list = ast.literal_eval(dir_name_clean)
            if isinstance(method_list, list):
                method_name = '_'.join(method_list)
            else:
                method_name = str(method_list)
            methods.append((method_name, dir_name))
        except (SyntaxError, ValueError) as e:
            print(f"Failed to parse directory name '{dir_name}': {e}")
            methods.append((dir_name, dir_name))

    ncols = len(methods)
    print(f"Parsed methods: {methods}")

    # Get list of phases from P0 to P13
    phases = [f'P{i}' for i in range(14)]
    nrows = len(phases)

    # Initialize data structures
    centrality_data = {}
    centrality_values = {'Betweenness': [], 'Closeness': [], 'Eigenvector': []}

    # Compute positions using a sample graph
    sample_graph_found = False
    positions = None

    # Loop to find a sample graph to compute positions
    for method_name, dir_name in methods:
        method_dir = os.path.join(base_dir, dir_name)
        for phase in phases:
            # Attempt to find a filename that exists
            for file in os.listdir(method_dir):
                if file.startswith(phase) and file.endswith('.csv'):
                    filename = file
                    filepath = os.path.join(method_dir, filename)
                    df = pd.read_csv(filepath, index_col=0)
                    G_sample = nx.from_pandas_adjacency(df)
                    positions = nx.spring_layout(G_sample, seed=42)
                    sample_graph_found = True
                    break
            if sample_graph_found:
                break
        if sample_graph_found:
            break

    if not sample_graph_found:
        print("No valid graphs found to compute positions. Exiting.")
        return

    # Loop over methods and phases to compute and store centralities
    print("Computing centralities across all methods and phases...")
    for method_name, dir_name in methods:
        centrality_data[method_name] = {}
        method_dir = os.path.join(base_dir, dir_name)
        for phase in phases:
            filename = ""
            # Attempt to find a filename that exists
            for file in os.listdir(method_dir):
                if file.startswith(phase) and file.endswith('.csv'):
                    filename = file
                    break
            if not filename:
                print(f"No file found for method '{method_name}' and phase '{phase}'. Skipping.")
                continue
            filepath = os.path.join(method_dir, filename)
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}. Skipping.")
                continue
            df = pd.read_csv(filepath, index_col=0)
            G = nx.from_pandas_adjacency(df)
            # Compute centralities
            betweenness = nx.betweenness_centrality(G, normalized=True)
            closeness = nx.closeness_centrality(G)
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {node: float('nan') for node in G.nodes}
            # Store centralities
            centrality_data[method_name][phase] = {'Betweenness': betweenness,
                                                   'Closeness': closeness,
                                                   'Eigenvector': eigenvector}
            # Collect centrality values for global normalization
            centrality_values['Betweenness'].extend([v for v in betweenness.values() if not np.isnan(v) and v > 0])
            centrality_values['Closeness'].extend([v for v in closeness.values() if not np.isnan(v)])
            centrality_values['Eigenvector'].extend([v for v in eigenvector.values() if not np.isnan(v)])

            # Print betweenness centrality stats for debugging
            betweenness_values = list(betweenness.values())
            print(
                f"Method: {method_name}, Phase: {phase}, Betweenness centrality stats - min: {np.min(betweenness_values)}, max: {np.max(betweenness_values)}, mean: {np.mean(betweenness_values)}")

    # Compute global min and max for each centrality measure
    global_min_max = {}
    for measure in ['Betweenness', 'Closeness', 'Eigenvector']:
        values = [v for v in centrality_values[measure] if not np.isnan(v) and v > 0]
        if values:
            global_min = min(values)
            global_max = max(values)
        else:
            global_min = 1e-6
            global_max = 1
        global_min_max[measure] = (global_min, global_max)
        print(f"{measure} centrality: min={global_min}, max={global_max}")

    # Generate plots for each centrality measure
    for measure in ['Betweenness', 'Closeness', 'Eigenvector']:
        print(f"Generating plot for {measure} centrality...")
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
        fig.suptitle(f"{measure} Centrality", fontsize=16, y=1.02)

        # Loop over phases and methods to plot the graphs
        for i, phase in enumerate(phases):
            for j, (method_name, dir_name) in enumerate(methods):
                ax = axes[i, j] if nrows > 1 else axes[j]
                # Get the centrality data
                centrality = centrality_data.get(method_name, {}).get(phase, {}).get(measure, None)
                if centrality is None:
                    ax.axis('off')
                    continue
                # Get the graph
                method_dir = os.path.join(base_dir, dir_name)
                filename = ""
                # Attempt to find the correct filename
                for file in os.listdir(method_dir):
                    if file.startswith(phase) and file.endswith('.csv'):
                        filename = file
                        break
                if not filename:
                    ax.axis('off')
                    continue
                filepath = os.path.join(method_dir, filename)
                df = pd.read_csv(filepath, index_col=0)
                G = nx.from_pandas_adjacency(df)
                # Prepare node colors
                node_colors = [centrality.get(node, 0.0) for node in G.nodes()]

                if measure == 'Betweenness':
                    # Replace zero or negative values with a small positive number
                    node_colors_array = np.array(node_colors)
                    node_colors_array[node_colors_array <= 0] = 1e-6

                    # Apply logarithmic normalization
                    norm = mcolors.LogNorm(vmin=global_min_max[measure][0], vmax=global_min_max[measure][1])

                    # Get the colormap
                    cmap = plt.get_cmap('viridis')

                    # Map the normalized values to colors
                    mapped_colors = cmap(norm(node_colors_array))

                    # Draw nodes with mapped colors
                    nodes = nx.draw_networkx_nodes(G, positions, ax=ax, node_color=mapped_colors, node_size=50)
                else:
                    # For other measures, use linear normalization
                    vmin, vmax = global_min_max[measure]
                    nodes = nx.draw_networkx_nodes(
                        G, positions, ax=ax, node_color=node_colors, cmap='viridis',
                        vmin=vmin, vmax=vmax, node_size=50)

                # Draw edges in light grey behind nodes
                nx.draw_networkx_edges(G, positions, ax=ax, edge_color='lightgrey', alpha=0.5)

                # Remove axis
                ax.axis('off')
                # Set column titles
                if i == 0:
                    ax.set_title(method_name, fontsize=12)
                # Set row labels
                if j == 0:
                    ax.text(-0.1, 0.5, phase, transform=ax.transAxes, fontsize=12, fontweight='bold',
                            va='center', ha='right', rotation=0)
        # Adjust layout and add colorbar
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        cbar = fig.colorbar(nodes, ax=axes.ravel().tolist(), shrink=0.6, orientation='horizontal', pad=0.02)
        cbar.set_label(f"{measure} Centrality")
        # Save the figure
        output_dir = 'centrality_plots'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{measure}_centrality_plots.png"), bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_dir}/{measure}_centrality_plots.png")

    print("All plots generated successfully.")


if __name__ == "__main__":
    # Change base_dir to the location of your graphs.
    base_dir = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/data/graphs"
    generate_centrality_plots(base_dir)