import numpy as np
import torch
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from torch import nn
from torch_geometric.loader import DataLoader

from src.utils import get_regression_error

def evaluate_ARIMA(
        dataset,
        train_part=0.75,
        order=(5, 1, 0),
        plot=False
):
    """
    ARIMA-based forecasting for S&P 500 dataset using `get_regression_error` for metrics.

    Parameters:
    - dataset: SP500Stocks instance.
    - train_part: Proportion of the dataset to use for training (default: 75%).
    - order: (p, d, q) tuple for ARIMA model configuration.
    - get_regression_error: Callable for calculating regression metrics.
    - plot: Whether to plot predictions vs. actuals for 4 random stocks.

    Returns:
    - mse, rmse, mae, mre: Metrics computed using `get_regression_error`.
    """
    #################################
    # 1) Train/test split
    #################################
    train_size = int(train_part * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    print(f"Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}")

    # Use batch_size=1 for the test set
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #################################
    # 2) ARIMA Forecasting
    #################################
    class ARIMAModel(torch.nn.Module):
        """
        Wrapper class to use ARIMA for forecasting with `get_regression_error`.
        """

        def forward(self, close_price):
            """
            ARIMA-based forecasting along the time dimension.

            Parameters:
            - close_price: Tensor [N, past_window].

            Returns:
            - preds: Tensor [N, 1], 1-step-ahead forecast for each stock.
            """
            device = close_price.device  # Get the device of input tensor
            preds = []
            for stock_idx in range(close_price.shape[0]):  # Loop over stocks
                series = close_price[stock_idx].cpu().numpy()  # Convert to NumPy on CPU
                try:
                    model = ARIMA(series, order=(20, 1, 0))
                    model_fit = model.fit(disp=0)
                    forecast = model_fit.forecast(steps=1)  # 1-step-ahead forecast
                    preds.append(forecast[0])
                except Exception:
                    # Fallback to last known value if ARIMA fails
                    preds.append(series[-1])

            # Convert predictions to a PyTorch tensor and move to the correct device
            preds = np.array(preds)  # Convert the list of numpy arrays into a single numpy array
            preds = torch.tensor(preds, dtype=torch.float32, device=device).unsqueeze(dim=1)  # Shape [N, 1]
            return preds

    arima_model = ARIMAModel()

    #################################
    # 3) Compute Metrics
    #################################
    mse, rmse, mae, mre = get_regression_error(arima_model, test_dataloader, graph=False)

    #################################
    # 4) Optional Plotting
    #################################
    if plot:
        print(f"MSE : {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE : {mae:.6f}")
        print(f"MRE : {mre:.6f}")

        # Plot 4 random stocks
        all_preds = []
        all_truth = []
        arima_model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                preds = arima_model(data.close_price)  # Predictions for the current batch
                all_preds.append(preds.squeeze(dim=1).cpu().numpy())
                all_truth.append(data.close_price_y.squeeze(dim=1).cpu().numpy())

        all_preds = np.stack(all_preds, axis=0)  # Shape: [num_test_steps, N]
        all_truth = np.stack(all_truth, axis=0)  # Shape: [num_test_steps, N]

        num_stocks = all_preds.shape[1]
        random_stocks = np.random.choice(num_stocks, size=4, replace=False)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        axes = axes.flatten()

        for i, stock_idx in enumerate(random_stocks):
            ax = axes[i]
            ax.plot(
                all_truth[:, stock_idx],
                label="Actual",
                color="blue"
            )
            ax.plot(
                all_preds[:, stock_idx],
                label="ARIMA",
                color="orange"
            )
            ax.set_title(f"Stock #{stock_idx}")
            ax.set_xlabel("Test Time Step")
            ax.set_ylabel("Price")
            ax.legend()

        plt.tight_layout()
        plt.show()

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MRE": mre
    }