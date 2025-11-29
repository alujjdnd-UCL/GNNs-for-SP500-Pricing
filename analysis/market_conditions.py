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
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compare_phase_market_conditions(phase_folder, output_file="phase_market_comparison.csv", feature_columns=None):
    """
    For each CSV file (phase) in phase_folder, splits the data into the first 75% (earlier dates)
    and the last 25% (later dates), then compares their market conditions.

    Comparisons include:
      - Summary statistics: mean and standard deviation.
      - Distributional differences using the KS test.
      - Distributional differences using Jensen–Shannon divergence on histograms.

    Parameters:
      phase_folder (str): Folder containing the phase CSV files.
      output_file (str): Filename for saving the comparison summary.
      feature_columns (list, optional): List of column names to compare. If None, all numeric columns are used.
    """
    results = []
    csv_files = glob(os.path.join(phase_folder, "*.csv"))

    for file in csv_files:
        # Extract the phase identifier from the filename (e.g., "P0" from "P0#23-08-2013#10-06-2014.csv")
        base = os.path.basename(file)
        phase = base.split("#")[0]

        # Read CSV and parse the Date column
        df = pd.read_csv(file, parse_dates=["Date"])
        df = df.sort_values("Date")

        # Split into first 75% and last 25% of rows
        n = len(df)
        split_index = int(0.75 * n)
        train_data = df.iloc[:split_index]
        test_data = df.iloc[split_index:]

        # Determine which features to compare.
        # If not specified, use numeric columns (excluding Date and non-numeric ones like Symbol)
        if feature_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols
        else:
            features = feature_columns

        # Compare each feature between the two time segments.
        for feature in features:
            train_series = train_data[feature].dropna()
            test_series = test_data[feature].dropna()

            if len(train_series) == 0 or len(test_series) == 0:
                continue

            train_mean = train_series.mean()
            test_mean = test_series.mean()
            train_std = train_series.std()
            test_std = test_series.std()

            # Perform KS test
            ks_stat, ks_p = ks_2samp(train_series, test_series)

            # Compute histograms with common bins for Jensen–Shannon divergence.
            bins = np.histogram_bin_edges(np.concatenate([train_series, test_series]), bins='auto')
            train_hist, _ = np.histogram(train_series, bins=bins, density=True)
            test_hist, _ = np.histogram(test_series, bins=bins, density=True)
            js_div = jensen_shannon_divergence(train_hist, test_hist)

            results.append({
                "phase": phase,
                "feature": feature,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
                "ks_stat": ks_stat,
                "ks_p_value": ks_p,
                "js_divergence": js_div,
                "n_train": len(train_series),
                "n_test": len(test_series)
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Market conditions comparison saved to {output_file}")


if __name__ == "__main__":
    phase_folder = "/cs/student/projects1/2022/jiuyzhan/fyp/src/scripts/data/SP500/raw/phases"
    compare_phase_market_conditions(phase_folder)
