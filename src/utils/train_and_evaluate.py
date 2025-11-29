import torch
from torch import nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.utils import get_regression_error, simulate_long_short_portfolio
from src.utils import train, plot_regression


def train_and_evaluate_model(dataset,
                             model_class,
                             train_part=0.75,
                             training_params: dict | None = None,
                             model_kwargs: dict | None = None,
                             device=None):
    """
    Train and evaluate a temporal graph model on the supplied dataset.

    :param dataset: Dataset instance (implements slicing and returns Data objects)
    :param model_class: Model class to instantiate
    :param train_part: Fraction of timesteps allocated to training
    :param training_params: Hyperparameter overrides (batch size, lr, etc.)
    :param model_kwargs: Extra keyword arguments forwarded to the model constructor
    :param device: torch.device to run on (defaults to CUDA if available)
    :return: Dictionary with evaluation metrics and portfolio performance summary
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    training_params = training_params.copy() if training_params else {}
    model_kwargs = model_kwargs or {}

    batch_size = training_params.get("batch_size", 32)
    lr = training_params.get("lr", 5e-3)
    weight_decay = training_params.get("weight_decay", 1e-5)
    num_epochs = training_params.get("num_epochs", 20)
    hidden_size = training_params.get("hidden_size", 32)
    layers_nb = training_params.get("layers_nb", 2)
    task_title = training_params.get("task_title", "PriceForecasting")
    plot = training_params.get("plot", False)

    train_boundary = int(train_part * len(dataset))
    train_dataset = dataset[:train_boundary]
    test_dataset = dataset[train_boundary:]

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Train/test split produced empty dataset. Adjust TRAIN_SPLIT or ensure sufficient timesteps.")

    print(f"Train dataset: {len(train_dataset)}, Test dataset: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False, drop_last=False)

    sample = dataset[0]
    in_channels, out_channels = sample.x.shape[-2], 1
    model = model_class(in_channels, out_channels, hidden_size, layers_nb, **model_kwargs)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train(model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs, task_title, device=device)

    mse, rmse, mae, mre = get_regression_error(model, train_dataloader, device=device)
    test_mse, test_rmse, test_mae, test_mre = get_regression_error(model, test_dataloader, device=device)

    if plot:
        plot_regression(model, next(iter(train_dataloader)), "Train data", device=str(device))
        plot_regression(model, next(iter(test_dataloader)), "Test data", device=str(device))

    portfolio_curve = simulate_long_short_portfolio(model, test_dataloader, device=device)
    portfolio_final_value = portfolio_curve[-1] if portfolio_curve else None

    return {
        "test_metrics": {
            "MSE": test_mse,
            "RMSE": test_rmse,
            "MAE": test_mae,
            "MRE": test_mre
        },
        "portfolio_curve": portfolio_curve,
        "portfolio_final_value": portfolio_final_value,
        "train_metrics": {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MRE": mre
        }
    }