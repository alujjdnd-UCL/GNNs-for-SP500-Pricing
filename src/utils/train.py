from datetime import datetime

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import trange

import torch

from src.utils.evaluate import measure_accuracy

def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    task_title: str = "",
    measure_acc: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> None:
    """
    Train function for a regression / classification model
    :param model: Model to train
    :param optimizer: Optimizer to use (Adam, ...)
    :param criterion: Loss function to use (MSE, CrossEntropy, ...)
    :param train_dataloader: Train data loader
    :param test_dataloader: Test data loader
    :param num_epochs: Number of epochs to train on the train dataset
    :param task_title: Title of the tensorboard run
    :param measure_acc: Whether to measure accuracy or not (for classification tasks)
    :param device: Device to train on (default: CUDA if available)
    """
    model.to(device)
    writer = SummaryWriter(f'runs-new/{task_title}_{datetime.now().strftime("%d_%m_%Hh%M")}_{model.__class__.__name__}')
    model_name = model.__class__.__name__
    for epoch in (pbar := trange(num_epochs, desc=f"{model_name} Epochs")):
        train_iteration(model, optimizer, pbar, criterion, train_dataloader, epoch, writer, measure_acc, device)
        test_iteration(model, criterion, test_dataloader, epoch, writer, measure_acc, device)

def test_iteration(
    model: nn.Module,
    criterion: nn.Module,
    test_dataloader: DataLoader,
    epoch: int,
    writer: SummaryWriter,
    measure_acc: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> None:
    """
    Test iteration
    :param model: Model to test
    :param criterion: Loss function to use (MSE, CrossEntropy, ...)
    :param test_dataloader: Test data loader
    :param epoch: Current epoch
    :param writer: Tensorboard writer
    :param measure_acc: Whether to measure accuracy or not (for classification tasks)
    :param device: Device to test on (default: CUDA if available)
    """
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_weight)
            loss = criterion(out, data.y)
            writer.add_scalar("Loss/Test Loss", loss.item(), epoch * len(test_dataloader) + idx)
            if measure_acc:
                acc = measure_accuracy(model, data)
                writer.add_scalar("Accuracy/Test Accuracy", acc, epoch * len(test_dataloader) + idx)

def train_iteration(
    model: nn.Module,
    optimizer: optim.Optimizer,
    pbar: trange,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    epoch: int,
    writer: SummaryWriter,
    measure_acc: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> None:
    """
    Train iteration
    :param model: Model to train
    :param optimizer: Optimizer to use (Adam, ...)
    :param pbar: tqdm progress bar
    :param criterion: Loss function to use (MSE, CrossEntropy, ...)
    :param train_dataloader: Train data loader
    :param epoch: Current epoch
    :param writer: Tensorboard writer
    :param measure_acc: Whether to measure accuracy or not (for classification tasks)
    :param device: Device to train on (default: CUDA if available)
    """
    model.train()
    for idx, data in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Batch": f"{(idx + 1) / len(train_dataloader) * 100:.1f}%"})
        writer.add_scalar("Loss/Train Loss", loss.item(), epoch * len(train_dataloader) + idx)
        if measure_acc:
            acc = measure_accuracy(model, data)
            writer.add_scalar("Accuracy/Train Accuracy", acc, epoch * len(train_dataloader) + idx)
