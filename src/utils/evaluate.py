import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import confusion_matrix


def get_regression_error(model: nn.Module, dataloader: DataLoader, graph=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> tuple[float, float, float, float]:
    """
    Computes regression errors
    :param model: Model to test
    :param dataloader: Dataloader to test on
    :param graph: Whether the model uses graph-based inputs
    :param device: Device to run the computation on (default: CUDA if available)
    :return: Mean squared error, rooted mean squared error, mean absolute error, mean relative error
    """
    model.to(device)
    model.eval()

    mse = 0
    rmse = 0
    mae = 0
    mre = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)  # Move data to the correct device

            # Get the model's output
            out = model(data.x, data.edge_index, data.edge_weight) if graph else model(data.x)

            # Ensure the output shape matches the target shape
            if out.shape != data.y.shape:
                out = out.squeeze()  # Remove extra dimensions if needed
                if out.shape != data.y.shape:  # If still mismatched, take mean across dimensions
                    out = out.mean(dim=-1, keepdim=True)

            # Compute metrics
            mse += F.mse_loss(out, data.y).item()
            rmse += torch.sqrt(F.mse_loss(out, data.y)).item()
            mae += F.l1_loss(out, data.y).item()
            mre += (F.l1_loss(out, data.y) / (data.y.abs().mean() + 1e-6)).item()  # Avoid division by zero

    return mse / len(dataloader), rmse / len(dataloader), mae / len(dataloader), mre / len(dataloader)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.data import Data

def plot_regression(model: nn.Module, data: Data, title: str = None, device: str = "cuda") -> None:
    """
    Plot 4 graphs for regression
    :param model: Model to test
    :param data: Data to test on
    :param title: Title of the plot
    :param device: Device to use (default: "cuda")
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Move data to the selected device
    data = data.to(device)

    # Prepare the plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)

    with torch.no_grad():
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_weight)

    # Select random stock indices for plotting
    stocks_idx = np.random.choice(data.x.shape[0] // (len(data.ptr) - 1), 4, replace=False)

    # Reshape predictions and targets
    preds = out.reshape(len(data.ptr) - 1, -1).cpu()
    target = data.y.reshape(len(data.ptr) - 1, -1).cpu()

    for idx, stock_idx in enumerate(stocks_idx):
        ax = axs[idx // 2, idx % 2]
        ax.plot(target[:, stock_idx].detach().numpy(), label="Real")
        ax.plot(preds[:, stock_idx].detach().numpy(), label="Predicted")
        ax.set_title(f"Stock {stock_idx}")
        ax.legend()

    plt.show()

    #


def measure_accuracy(model: nn.Module, data: Data) -> float:
    """
    Measure accuracy
    :param model: Model to test
    :param data: Data to test on
    :return: Accuracy
    """
    out = model(data.x, data.edge_index, data.edge_weight)
    if out.shape[1] == 1:  # Binary classification
        return (F.sigmoid(out).round() == data.y).sum().item() / len(data.y)
    else:  # Multi-class classification
        return (F.softmax(out, dim=-1).argmax(dim=-1) == data.y).sum().item() / len(data.y)


def get_confusion_matrix(model: nn.Module, data: Data) -> np.ndarray:
    """
    Get confusion matrix
    :param model: Model to test
    :param data: Data to test on
    :return: Confusion matrix
    """
    out = model(data.x, data.edge_index, data.edge_weight)
    if out.shape[1] == 1:
        y_pred = F.sigmoid(out).round().detach().numpy()
    else:
        y_pred = F.softmax(out, dim=-1).argmax(dim=-1).detach().numpy()
    y_true = data.y.detach().numpy()
    return confusion_matrix(y_true, y_pred)


def simulate_long_short_portfolio(
        model: nn.Module,
        dataloader: DataLoader,
        device = None,
        initial_capital: float = 100000,
        long_count: int = 10,
        short_count: int = 0,
) -> list[float]:
    """
    Simulate a long/short portfolio based on model predictions.

    For each day (graph in the batched data), the function:
      - Computes the predicted next-day price.
      - Calculates predicted returns from the current price (last close in the past window).
      - Selects the top 10 stocks with the highest predicted returns (to long) and the
        bottom 10 stocks (to short).
      - Allocates the entire capital equally across these 20 positions.
      - Computes the realized profit/loss using the next-day actual prices.

    Assumptions:
      - Trades are entered for a 1-day increment.
      - There are no transaction fees.
      - The Data objects have attributes:
          - close_price: shape [num_stocks, past_window] (last column gives the current price)
          - close_price_y: shape [num_stocks, future_window] (first column gives the next-day price)
          - For batched data, a 'ptr' attribute indicates graph boundaries.

    :param model: Trained model that outputs predicted next-day prices.
    :param dataloader: DataLoader providing Data objects for each day.
    :param device: Device on which to run the simulation.
    :param initial_capital: Starting portfolio value.
    :param long_count: Number of stocks to long.
    :param short_count: Number of stocks to short.
    :return: A list of portfolio values at the end of each day.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    portfolio_values = []
    capital = initial_capital

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            # Get model predictions.
            out = model(data.x, data.edge_index, data.edge_weight)

            # Check if data is batched across multiple days using the ptr attribute.
            if hasattr(data, "ptr") and data.ptr is not None:
                n_days = len(data.ptr) - 1
                # Reshape predictions to (n_days, n_stocks_per_day)
                preds = out.reshape(n_days, -1)
                # Current price: take the last column of close_price.
                current_prices = data.close_price[:, -1].reshape(n_days, -1)
                # Next day actual price: take the first column of close_price_y.
                next_prices = data.close_price_y[:, 0].reshape(n_days, -1)
            else:
                # If data is not batched into days, add a day dimension.
                preds = out.unsqueeze(0)
                current_prices = data.close_price[:, -1].unsqueeze(0)
                next_prices = data.close_price_y[:, 0].unsqueeze(0)

            # Simulate trading for each day.
            for i in range(preds.shape[0]):
                # Predicted next-day prices for all stocks on day i.
                predicted_prices = preds[i]
                curr = current_prices[i]
                next_day = next_prices[i]

                # Compute predicted returns.
                predicted_returns = (predicted_prices - curr) / curr

                # Select indices for long and short positions.
                long_indices = torch.topk(predicted_returns, long_count).indices
                short_indices = torch.topk(-predicted_returns, short_count).indices

                # Allocate capital equally across the 20 positions.
                allocation = capital / (long_count + short_count)

                # Compute realized returns:
                # For long positions, profit is proportional to (next_day - current) / current.
                long_realized = (next_day[long_indices] - curr[long_indices]) / curr[long_indices]
                long_profit = allocation * long_realized.sum()

                # For short positions, profit is proportional to (current - next_day) / current.
                short_realized = (curr[short_indices] - next_day[short_indices]) / curr[short_indices]
                short_profit = allocation * short_realized.sum()

                # Update capital.
                daily_profit = long_profit + short_profit
                capital += daily_profit.item()  # ensure conversion from tensor to float

                portfolio_values.append(capital)

    return portfolio_values
