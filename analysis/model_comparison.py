import pandas as pd
import matplotlib.pyplot as plt
import ast

# -------------------------------
# PART 1: Calculate SP500 Index Price
# -------------------------------

# Load the CSV of S&P500 constituent prices.
df_prices = pd.read_csv("aggregated_cleaned.csv", parse_dates=['Date'])

# For each day, compute an equally weighted index price by averaging all stock prices.
price_cols = df_prices.columns.drop('Date')
df_prices['IndexPrice'] = df_prices[price_cols].mean(axis=1)

# Normalize the index price so that the first day is 100.
df_prices['IndexPriceNorm'] = df_prices['IndexPrice'] / df_prices['IndexPrice'].iloc[0] * 100

# -------------------------------
# PART 2: Compute Cumulative Returns for Each Phase (last 25% of days)
# -------------------------------

# Define the phases (using ISO date strings) as provided.
phases = {
    'P0':  ('2013-08-23', '2014-06-10'),
    'P1':  ('2014-06-11', '2015-03-26'),
    'P2':  ('2015-03-27', '2016-01-11'),
    'P3':  ('2016-01-12', '2016-08-11'),
    'P4':  ('2016-10-26', '2018-05-30'),
    'P5':  ('2018-05-31', '2019-03-18'),
    'P6':  ('2019-03-19', '2019-12-31'),
    'P7':  ('2020-01-02', '2020-10-15'),
    'P8':  ('2020-10-16', '2021-03-18'),
    'P9':  ('2021-08-10', '2022-05-18'),
    'P10': ('2022-05-19', '2023-03-07'),
    'P11': ('2023-03-08', '2023-12-20'),
    'P12': ('2023-12-21', '2024-10-10'),
    'P13': ('2024-10-09', '2024-12-06')
}

# Calculate phase returns for the SP500 index using only the last 25% of days for each phase.
sp500_returns = {}
for phase, (start, end) in phases.items():
    phase_data = df_prices[(df_prices['Date'] >= pd.to_datetime(start)) &
                             (df_prices['Date'] <= pd.to_datetime(end))]
    if len(phase_data) == 0:
        continue
    last_25 = phase_data.iloc[int(0.75 * len(phase_data)):]
    ret = (last_25['IndexPriceNorm'].iloc[-1] / last_25['IndexPriceNorm'].iloc[0] - 1) * 100
    sp500_returns[phase] = ret

# Convert the SP500 returns dictionary into a DataFrame
df_sp500 = pd.DataFrame(list(sp500_returns.items()), columns=['Phase', 'SP500 Return (%)'])

# Order phases naturally (assume phases are labeled like P0, P1, …)
phases_order = sorted(phases.keys(), key=lambda p: int(p.strip('P')))
df_sp500['Phase'] = pd.Categorical(df_sp500['Phase'], categories=phases_order, ordered=True)
df_sp500 = df_sp500.sort_values('Phase')

# Compute the cumulative return by compounding each phase's return.
df_sp500['SP500 Cumulative Return (%)'] = ( (1 + df_sp500['SP500 Return (%)']/100).cumprod() - 1 ) * 100

# -------------------------------
# PART 3: Overlay SP500 Data on the Trading Strategies Graph
# -------------------------------

# Load the adjusted metrics CSV (which has trading performance per phase).
df = pd.read_csv("../metrics_to_plot.csv")

# Exclude any rows where Phase equals 'P14'
df = df[df['Phase'] != 'P14']

# Parse the "Edge Formation" column into a sorted, comma-separated string for grouping.
df['EdgeFormationParsed'] = df['Edge Formation'].apply(lambda x: ', '.join(sorted(ast.literal_eval(x))))

# Ensure that Portfolio Performance is numeric.
df['Portfolio Performance'] = pd.to_numeric(df['Portfolio Performance'], errors='coerce')

# Convert Phase to a categorical variable with natural ordering (e.g. P0, P1, …).
phases_order_metrics = sorted(df['Phase'].unique(), key=lambda p: int(p.strip('P')))
df['Phase'] = pd.Categorical(df['Phase'], categories=phases_order_metrics, ordered=True)

# Filter the DataFrame to include only the models ['DCRNN', 'TGCN', 'A3TGCN']
df = df[df['Model'].isin(['DCRNN', 'TGCN', 'A3TGCN'])]

# Compute cumulative return for trading strategies per group.
def compute_cumulative_return(group):
    group = group.sort_values('Phase')
    group['Cumulative Return'] = (1 + group['Portfolio Performance'] / 100).cumprod() - 1
    group['Cumulative Return'] = group['Cumulative Return'] * 100
    return group

df = df.groupby(['EdgeFormationParsed', 'Model'], group_keys=False).apply(compute_cumulative_return)

# Plot trading strategies against SP500 cumulative returns
graph_methods = sorted(df['EdgeFormationParsed'].unique())

for method in graph_methods:
    df_method = df[df['EdgeFormationParsed'] == method]
    models = ['DCRNN', 'TGCN', 'A3TGCN']  # Ensuring only these models are plotted

    plt.figure(figsize=(10, 6))

    # Plot each selected model's cumulative return.
    for model in models:
        df_model = df_method[df_method['Model'] == model].sort_values('Phase')
        if not df_model.empty:
            plt.plot(df_model['Phase'], df_model['Cumulative Return'], marker='o', label=model)

    # Overlay the SP500 cumulative returns.
    plt.plot(df_sp500['Phase'], df_sp500['SP500 Cumulative Return (%)'], marker='s', linestyle='--',
             label='SP500 Index (Cumulative Return)')

    plt.xlabel("Phase")
    plt.ylabel("Cumulative Return (%)")
    plt.title(f"Cumulative Adjusted Return Across Phases\n(Graph Construction Method: {method})")

    # Set up the legend with only the desired models + SP500
    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_labels = ['DCRNN', 'TGCN', 'A3TGCN', 'SP500 Index (Cumulative Return)']
    ordered_handles = [handles[labels.index(l)] for l in ordered_labels if l in labels]

    plt.legend(ordered_handles, ordered_labels, title="Model")
    plt.tight_layout()
    plt.show()

# -------------------------------
# PART 4: Save Returns to CSV
# -------------------------------

# Save SP500 returns
df_sp500.to_csv("sp500_returns.csv", index=False)

# Save trading strategies returns
df.to_csv("trading_strategies_returns.csv", index=False)
