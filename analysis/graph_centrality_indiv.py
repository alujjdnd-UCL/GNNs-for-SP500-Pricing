import os
import ast
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def generate_centrality_heatmaps(base_dir):
    method_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    method_dirs.sort()
    print(f"Found method directories: {method_dirs}")

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

    print(f"Parsed methods: {methods}")
    phases = [f'P{i}' for i in range(14)]

    centrality_data = {}
    centrality_values = {'Betweenness': [], 'Closeness': [], 'Eigenvector': []}

    sample_graph_found = False
    positions = None

    for method_name, dir_name in methods:
        method_dir = os.path.join(base_dir, dir_name)
        for phase in phases:
            for file in os.listdir(method_dir):
                if file.startswith(phase) and file.endswith('.csv'):
                    filepath = os.path.join(method_dir, file)
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

    print("Computing centralities across all methods and phases...")
    for method_name, dir_name in methods:
        centrality_data[method_name] = {}
        method_dir = os.path.join(base_dir, dir_name)
        for phase in phases:
            filename = ""
            for file in os.listdir(method_dir):
                if file.startswith(phase) and file.endswith('.csv'):
                    filename = file
                    break
            if not filename:
                print(f"No file found for method '{method_name}' and phase '{phase}'. Skipping.")
                continue
            filepath = os.path.join(method_dir, filename)
            df = pd.read_csv(filepath, index_col=0)
            G = nx.from_pandas_adjacency(df)

            betweenness = nx.betweenness_centrality(G, normalized=True)
            closeness = nx.closeness_centrality(G)
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                eigenvector = {node: float('nan') for node in G.nodes}

            centrality_data[method_name][phase] = {
                'Betweenness': betweenness,
                'Closeness': closeness,
                'Eigenvector': eigenvector
            }

            centrality_values['Betweenness'].extend([v for v in betweenness.values() if not np.isnan(v) and v > 0])
            centrality_values['Closeness'].extend([v for v in closeness.values() if not np.isnan(v)])
            centrality_values['Eigenvector'].extend([v for v in eigenvector.values() if not np.isnan(v)])

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

    output_base = 'graph_centralities_heatmap'
    os.makedirs(output_base, exist_ok=True)

    for measure in ['Betweenness', 'Closeness', 'Eigenvector']:
        print(f"Generating heatmaps for {measure} centrality...")
        measure_dir = os.path.join(output_base, measure.lower())
        os.makedirs(measure_dir, exist_ok=True)

        for method_name, dir_name in methods:
            for phase in phases:
                centrality = centrality_data.get(method_name, {}).get(phase, {}).get(measure, None)
                if centrality is None:
                    continue

                method_dir = os.path.join(base_dir, dir_name)
                filename = ""
                for file in os.listdir(method_dir):
                    if file.startswith(phase) and file.endswith('.csv'):
                        filename = file
                        break
                if not filename:
                    continue

                filepath = os.path.join(method_dir, filename)
                df = pd.read_csv(filepath, index_col=0)
                G = nx.from_pandas_adjacency(df)
                node_colors = [centrality.get(node, 0.0) for node in G.nodes()]

                fig, ax = plt.subplots(figsize=(5, 4))
                if measure == 'Betweenness':
                    node_colors_array = np.array(node_colors)
                    node_colors_array[node_colors_array <= 0] = 1e-6
                    norm = mcolors.LogNorm(vmin=global_min_max[measure][0], vmax=global_min_max[measure][1])
                    cmap = plt.get_cmap('viridis')
                    mapped_colors = cmap(norm(node_colors_array))
                    nx.draw_networkx_nodes(G, positions, ax=ax, node_color=mapped_colors, node_size=80)
                else:
                    vmin, vmax = global_min_max[measure]
                    nx.draw_networkx_nodes(
                        G, positions, ax=ax, node_color=node_colors, cmap='viridis',
                        vmin=vmin, vmax=vmax, node_size=80)

                nx.draw_networkx_edges(G, positions, ax=ax, edge_color='lightgrey', alpha=0.5)
                ax.set_title(f"{method_name} - {phase}", fontsize=10)
                ax.axis('off')

                out_path = os.path.join(measure_dir, f"{method_name}_{phase}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

    print(f"Saved {len(methods)} centrality plots to {output_base}")


if __name__ == "__main__":
    base_dir = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/data/graphs"
    generate_centrality_heatmaps(base_dir)
