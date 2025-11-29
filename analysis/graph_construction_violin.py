import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

# Load CSV data
df = pd.read_csv("../metrics_to_plot.csv")

# Parse the "Edge Formation" column into a sorted, comma-separated string for grouping
df['EdgeFormationParsed'] = df['Edge Formation'].apply(lambda x: ', '.join(sorted(ast.literal_eval(x))))

# Ensure that the "Return (%)" column is numeric
df['Portfolio Performance'] = pd.to_numeric(df['Portfolio Performance'], errors='coerce')

# 1) Instead of simple alphabetical sorting, we sort by number of items, then alphabetically
unique_methods = df['EdgeFormationParsed'].unique()

def method_key(method_str):
    items = method_str.split(", ")
    # Sort primarily by how many items are in the list, then by alphabetical order
    return (len(items), items)

graph_methods = sorted(unique_methods, key=method_key)

# 2) Create a mapping from each unique graph construction method to a numeric x-axis value
method_to_x = {method: i for i, method in enumerate(graph_methods)}

# Add a jittered x coordinate for each row to avoid overlapping dots
np.random.seed(42)  # For reproducibility
df['x_jitter'] = df['EdgeFormationParsed'].map(method_to_x) + np.random.uniform(-0.1, 0.1, size=len(df))

# Create a color mapping for each model
models = df['Model'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
model_to_color = {model: colors[i] for i, model in enumerate(models)}

# ----- Violin Plot with Scatter Overlay and Summary Statistics -----
# Prepare the violin plot data: one list per graph construction method
violin_data = [df[df['EdgeFormationParsed'] == method]['Portfolio Performance'].values for method in graph_methods]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the violin plots
parts = ax.violinplot(violin_data, positions=list(range(len(graph_methods))), widths=0.8, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('lightgray')
    pc.set_edgecolor('gray')
    pc.set_alpha(0.5)

# Overlay small scatter dots for each model
for model in models:
    subset = df[df['Model'] == model]
    ax.scatter(subset['x_jitter'], subset['Portfolio Performance'],
               color=model_to_color[model],
               label=model,
               s=10,  # small dots
               alpha=0.8)

# Compute and plot the mean and 25th-75th percentiles for each method
for method, xpos in method_to_x.items():
    data = df[df['EdgeFormationParsed'] == method]['Portfolio Performance']
    if len(data) == 0:
        continue
    mean_val = data.mean()
    p25 = np.percentile(data, 25)
    p75 = np.percentile(data, 75)
    # Plot the mean as a black circle with error bars for the 25th-75th percentiles
    ax.errorbar(xpos, mean_val,
                yerr=[[mean_val - p25], [p75 - mean_val]],
                fmt='o', color='black', capsize=5, markersize=6)

# Set x-axis labels and ticks
ax.set_xticks(list(method_to_x.values()))
ax.set_xticklabels(list(method_to_x.keys()), rotation=45)
ax.set_xlabel("Graph Construction Method")
ax.set_ylabel("Portfolio Performance")
ax.set_title("Distribution of Portfolio Performance by Graph Construction Method")

# Add a legend for models (avoid duplicates by using unique labels)
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# ----- Table of Percentiles -----
# Compute percentiles (10% to 90% in 10% increments) for each graph construction method
percentiles = np.arange(0, 100, 5)
table_data = {}
for method in graph_methods:
    data = df[df['EdgeFormationParsed'] == method]['Portfolio Performance']
    pct_values = np.percentile(data, percentiles)
    table_data[method] = pct_values

# Create a DataFrame where rows are construction types and columns are the percentiles
table_df = pd.DataFrame(table_data, index=[f"{p}%" for p in percentiles]).T

print("Table of Percentiles for Portfolio Performance by Graph Construction Method:")
print(table_df)

# Save the table as a CSV file
table_df.to_csv("percentiles_table.csv", index=True)
print("\nPercentile table saved as 'percentiles_table.csv'")
