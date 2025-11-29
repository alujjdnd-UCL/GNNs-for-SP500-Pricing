import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast

# Load CSV data
df = pd.read_csv("/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/metrics_to_plot.csv")

# Ignore rows with Phase "P14"
df = df[df['Phase'] != "P14"]

# Convert the "Edge Formation" string to an actual list
df['edge_method'] = df['Edge Formation'].apply(lambda x: ast.literal_eval(x))

# Define mapping using frozensets for order‐independent matching.
edge_method_map = {
    frozenset(["correlation", "wikidata"]): "CorrWiki",
    frozenset(["wikidata", "sectors"]): "SecWiki",
    frozenset(["correlation", "sectors"]): "CorrSec",
    frozenset(["wikidata", "correlation", "sectors"]): "CorrSecWiki"
}


def map_edge_method(method_list):
    return edge_method_map.get(frozenset(method_list), None)


# Map the edge methods and drop any rows that don't match.
df['edge_method'] = df['edge_method'].apply(map_edge_method)
df = df.dropna(subset=['edge_method'])

# Define full descriptive titles for each section.
edge_titles = {
    "CorrSecWiki": "correlation_sectors_wikidata",
    "CorrSec": "correlation_sectors",
    "CorrWiki": "correlation_wikidata",
    "SecWiki": "sectors_wikidata"
}

# Define the 2x2 layout (order is important):
# Top-left: CorrSecWiki, Top-right: CorrSec, Bottom-left: CorrWiki, Bottom-right: SecWiki
edge_methods = [
    ("CorrSecWiki", edge_titles["CorrSecWiki"]),
    ("CorrSec", edge_titles["CorrSec"]),
    ("CorrWiki", edge_titles["CorrWiki"]),
    ("SecWiki", edge_titles["SecWiki"])
]

# Create a large figure with a 2x2 gridspec for the main sections.
fig = plt.figure(figsize=(20, 30))
gs_main = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.2)

# Use the viridis colourmap (sampled from 0.2 to 0.8 to avoid the brightest yellow)
viridis = plt.colormaps['viridis']
colours = viridis(np.linspace(0.2, 0.8, 3))  # Colours for RMSE, MAE, MRE


def plot_edge_section(cell, edge_method_key, edge_method_title):
    # Filter the data for this edge formation category.
    df_sub = df[df['edge_method'] == edge_method_key]
    models = sorted(df_sub['Model'].unique())

    # Determine number of rows (at least 1 even if no data).
    n_models = len(models) if len(models) > 0 else 1
    subgrid = cell.subgridspec(n_models, 1, hspace=0.4)

    axes = []
    if len(models) == 0:
        # Create a single empty subplot with a message.
        ax = fig.add_subplot(subgrid[0, 0])
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        ax.set_title(edge_method_title, bold=True)
        ax.axis("off")
        axes.append(ax)
    else:
        # Create one subplot per model.
        for i, model_name in enumerate(models):
            ax = fig.add_subplot(subgrid[i, 0])
            # Extract the numeric part from Phase for correct sorting.
            df_model = df_sub[df_sub['Model'] == model_name].copy()
            df_model['phase_num'] = df_model['Phase'].str.extract('(\d+)').astype(int)
            df_model = df_model.sort_values(by='phase_num')

            # Plot RMSE, MAE, and MRE with the selected colours.
            ax.plot(df_model['phase_num'], df_model['RMSE'], label='RMSE', color=colours[0])
            ax.plot(df_model['phase_num'], df_model['MAE'], label='MAE', color=colours[1])
            ax.plot(df_model['phase_num'], df_model['MRE'], label='MRE', color=colours[2])

            ax.set_ylabel(model_name)
            if i == 0:
                ax.set_title(edge_method_title)
            if i == len(models) - 1:
                ax.set_xlabel("Phase")
            # Set x-ticks using the original Phase labels.
            ax.set_xticks(df_model['phase_num'])
            ax.set_xticklabels(df_model['Phase'])
            axes.append(ax)
    return axes[0]  # Return the top axis for legend handles


# Create subgrids for each of the four main sections.
main_cells = [gs_main[0, 0], gs_main[0, 1], gs_main[1, 0], gs_main[1, 1]]
legend_axes = []
for (edge_key, edge_title), cell in zip(edge_methods, main_cells):
    top_ax = plot_edge_section(cell, edge_key, edge_title)
    legend_axes.append(top_ax)

# Set overall title at the top.
fig.suptitle("Error Statistics of Models Across Graph Construction Techniques", fontsize=24, y=0.94)

# Place the legend between the title and the plots.
handles, labels = legend_axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, title="Error Metrics", bbox_to_anchor=(0.5, 0.92))

# Adjust layout to leave room for the title and legend.
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
