import pandas as pd
import numpy as np
from tqdm import tqdm


# Normalise the closing prices for each stock
def normalise_close(column):
    """
    Normalise the closing prices for each stock.
    :param column: The series of closing prices.
    :return: The series of normalised closing prices.
    """
    return (column - column.min()) / (column.max() - column.min())


# Calculate daily log return
def calculate_log_return(series):
    """
    Calculate the daily log return for a series.
    :param series: The series of closing prices.
    :return: The series of daily log returns.
    """
    return np.log(series / series.shift(1))


# Define technical indicators
def calculate_rsi(prices, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a series of prices.
    :param prices: The series of closing prices.
    :param window: The window size for the RSI calculation.
    :return: The series of RSI values.
    """
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a series of prices.
    :param prices: The series of closing prices.
    :param short_window: The window size for the short-term EMA.
    :param long_window: The window size for the long-term EMA.
    :param signal_window: The window size for the signal line EMA.
    :return: The series of MACD values.
    """
    short_ema = prices.ewm(span=short_window, min_periods=1).mean()
    long_ema = prices.ewm(span=long_window, min_periods=1).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1).mean()
    return macd - signal


def process_indicators(data_csv_path = "../../data/SP500/raw/aggregated_cleaned.csv",
                       output_dir = "../../data/SP500/raw/phases/"):
    """
    Process the technical indicators for the dataset.
    :param data_csv_path: The path to the dataset CSV file.
    :param output_dir: The directory to save the processed CSV files.
    :return: A list of paths to the processed CSV files.
    """

    # Load the dataset
    df = pd.read_csv(data_csv_path, parse_dates=["Date"], dayfirst=True)

    # Calculate all statistics for the entire dataset first
    all_data = []

    print("Processing stocks for statistics...")
    for column in tqdm(df.columns[1:], desc="Stocks Processed"):
        stock_data = df[["Date", column]].rename(columns={column: "Close"}).dropna()
        stock_data["Symbol"] = column
        stock_data["NormClose"] = normalise_close(stock_data["Close"])
        stock_data["DailyLogReturn"] = calculate_log_return(stock_data["Close"])
        stock_data["ALR1W"] = stock_data["NormClose"].rolling(window=5).mean().shift(1).fillna(0)
        stock_data["ALR2W"] = stock_data["NormClose"].rolling(window=10).mean().shift(1).fillna(0)
        stock_data["ALR1M"] = stock_data["NormClose"].rolling(window=20).mean().shift(1).fillna(0)
        stock_data["ALR2M"] = stock_data["NormClose"].rolling(window=40).mean().shift(1).fillna(0)
        stock_data["RSI"] = calculate_rsi(stock_data["Close"])
        stock_data["MACD"] = calculate_macd(stock_data["Close"])
        all_data.append(stock_data)

    # Concatenate all stock data
    all_data = pd.concat(all_data, ignore_index=True)

    # Sort by date and truncate the first 40 days (max window size used for ALR2M)
    print("Truncating initial days...")
    all_data = all_data.sort_values(by=["Date", "Symbol"])
    all_data = all_data.groupby("Symbol").apply(lambda group: group.iloc[40:]).reset_index(drop=True)

    # Partition into 200-day non-overlapping intervals
    n_days = 200

    dates = all_data["Date"].unique()
    intervals = [dates[i:i+n_days] for i in range(0, len(dates), n_days)]

    output_files = []

    # Save each interval as a separate CSV
    print("Saving interval CSVs...")
    for idx, interval in tqdm(enumerate(intervals), desc="Intervals Processed"):
        interval_data = all_data[all_data["Date"].isin(interval)]
        if not interval_data.empty:
            start_date = pd.to_datetime(interval[0]).strftime("%d-%m-%Y")
            end_date = pd.to_datetime(interval[-1]).strftime("%d-%m-%Y")
            output_path = output_dir + f"P{idx}#{start_date}#{end_date}.csv"
            output_files.append(output_path)
            interval_data.to_csv(output_path, index=False)

    print("Processing complete.")

    return output_files