import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch_geometric.loader import DataLoader

from src.utils import get_regression_error


def evaluate_MA(dataset, train_part=0.75, plot=False):
    #################################
    # 3) Train/test split
    #################################
    train_size = int(train_part * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    print(f"Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}")

    # We will set batch_size=1 for test set
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #################################
    # 4) Moving Average "Model"
    #################################
    class MovingAverageModel(nn.Module):
        """
        A naive moving average approach using close_price across the time dimension.
        For each time-step in the test set, we produce a forecast by averaging all past_window days.
        """

        def __init__(self):
            super().__init__()

        def forward(self, close_price):
            """
            close_price: Tensor [N, past_window], for N stocks, each with 'past_window' days.
            returns: Tensor [N, 1], one-day-ahead forecast for each stock.
            """
            avg_price = close_price.mean(dim=1, keepdim=True)  # [N, 1]
            return avg_price

    model = MovingAverageModel()

    #################################
    # 5) Gather predictions across test set
    #################################
    model.eval()

    all_preds = []
    all_truth = []

    with torch.no_grad():
        for data in test_dataloader:
            # data.close_price: [N, past_window]
            # data.close_price_y: [N, future_window=1]
            close_price = data.close_price  # shape [N, seq_len]
            preds = model(close_price)  # shape [N, 1]

            # Store each day’s predictions/targets
            all_preds.append(preds.squeeze(dim=1).cpu().numpy())  # shape [N]
            all_truth.append(data.close_price_y.squeeze(dim=1).cpu().numpy())  # shape [N]

    # Convert list -> array of shape [num_test_steps, N]
    all_preds = np.stack(all_preds, axis=0)  # [num_test_steps, N]
    all_truth = np.stack(all_truth, axis=0)  # [num_test_steps, N]

    #################################
    # 6) Compute metrics across entire test set
    #################################
    mse, rmse, mae, mre = get_regression_error(model, test_dataloader, graph=False)

    if plot:
        print(f"MSE : {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE : {mae:.6f}")
        print(f"MRE : {mre:.6f}")

        #################################
        # 7) Plot 4 random stocks: Real vs MA
        #################################
        num_stocks = all_preds.shape[1]
        random_stocks = np.random.choice(num_stocks, size=4, replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        axes = axes.flatten()

        for i, stock_idx in enumerate(random_stocks):
            ax = axes[i]
            ax.plot(
                all_truth[:, stock_idx],
                label="Real",
                color="blue"
            )
            ax.plot(
                all_preds[:, stock_idx],
                label="MA",
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