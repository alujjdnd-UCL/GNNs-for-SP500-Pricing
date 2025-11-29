import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from glob import glob

def jensen_shannon_divergence(p, q):
    """
    Computes the Jensen-Shannon divergence between two probability distributions.
    Both p and q should be arrays that sum to 1.
    """
    p = np.array(p) + 1e-10  # add epsilon to avoid zeros
    q = np.array(q) + 1e-10
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))

def get_phase_number(phase_name):
    """
    Extracts the numeric part of a phase name for sorting purposes.
    """
    return int(phase_name.strip('P'))

def compare_phase_market_conditions(phase_folder, js_output_file="js_divergence.csv", ks_p_output_file="ks_p_values.csv"):
    """
    For each CSV file (phase) in phase_folder, splits the data into the first 75% (earlier dates)
    and the last 25% (later dates), then compares their market conditions.

    Generates:
      - CSV of JS divergences
      - CSV of ks_p_values
    """

    csv_files = glob(os.path.join(phase_folder, "*.csv"))

    # Sort csv_files based on phase numbers
    csv_files = sorted(csv_files, key=lambda x: get_phase_number(os.path.basename(x).split("#")[0]))

    # Initialize phases_list with correct sorting, including P0 to P13
    phases_list = []
    for f in csv_files:
        phase = os.path.basename(f).split("#")[0]
        if phase.startswith('P'):
            phase_num = get_phase_number(phase)
            if 0 <= phase_num <=13:
                phases_list.append(phase)

    # Define the features
    features = ['Close', 'RSI', 'MACD', 'DailyLogReturn']

    # Initialize list to store the statistics
    results = []

    for phase in phases_list:
        # Find the corresponding file
        file = [f for f in csv_files if os.path.basename(f).startswith(phase)][0]

        # Read CSV and parse the Date column
        df = pd.read_csv(file, parse_dates=["Date"])
        df = df.sort_values("Date")

        # Split into first 75% and last 25% of rows
        n = len(df)
        split_index = int(0.75 * n)
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]

        for feature in features:
            if feature not in df.columns:
                # Append NaN to signify missing feature for this phase
                results.append({
                    'phase': phase,
                    'feature': feature,
                    'js_divergence': np.nan,
                    'ks_p_value': np.nan
                })
                continue  # Skip if the feature is not in the dataframe

            train_series = train_data[feature].dropna()
            test_series = test_data[feature].dropna()

            if len(train_series) == 0 or len(test_series) == 0:
                results.append({
                    'phase': phase,
                    'feature': feature,
                    'js_divergence': np.nan,
                    'ks_p_value': np.nan
                })
                continue

            # Perform KS test
            ks_stat, ks_p = ks_2samp(train_series, test_series)

            # Compute histograms with common bins for Jensen–Shannon divergence.
            combined_series = np.concatenate([train_series, test_series])
            bins = np.histogram_bin_edges(combined_series, bins='auto')
            train_hist, _ = np.histogram(train_series, bins=bins, density=True)
            test_hist, _ = np.histogram(test_series, bins=bins, density=True)
            js_div = jensen_shannon_divergence(train_hist, test_hist)

            results.append({
                'phase': phase,
                'feature': feature,
                'js_divergence': js_div,
                'ks_p_value': ks_p
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Now pivot the DataFrame to get the required shape
    # For js_divergence
    js_div_df = results_df.pivot(index='feature', columns='phase', values='js_divergence')
    js_div_df = js_div_df[phases_list]  # Ensure columns are in order

    # For ks_p_value
    ks_p_df = results_df.pivot(index='feature', columns='phase', values='ks_p_value')
    ks_p_df = ks_p_df[phases_list]  # Ensure columns are in order

    # Save the DataFrames to CSV
    js_div_df.to_csv(js_output_file)
    print(f"JS divergence saved to {js_output_file}")

    ks_p_df.to_csv(ks_p_output_file)
    print(f"KS p-values saved to {ks_p_output_file}")

if __name__ == "__main__":
    phase_folder = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/data/SP500/raw/phases"
    compare_phase_market_conditions(phase_folder)