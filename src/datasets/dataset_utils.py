import numpy as np
import pandas as pd
import torch


def get_graph_in_pyg_format(values_pd, adj_pd, verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Creates the PyTorch Geometric graph data from the stock price data and adjacency matrix.
	:param values_pd: PD dataframe of the stocks price data
	:param adj_pd: PD adjacency matrix of the stocks price data
	:return: The graph data in PyTorch Geometric format
		x: Node features (nodes_nb, timestamps_nb, features_nb)
		close_prices: Close prices (nodes_nb, timestamps_nb)
		edge_index: Edge index (2, edge_nb)
		edge_weight: Edge weight (edge_nb,)
	"""

	nodes_nb, edge_nb = len(adj_pd), np.count_nonzero(adj_pd)

	if verbose:
		size = values_pd.to_numpy().size
		print(f"Size of the data: {size}")
		print(f"Number of datapoints per stock: {size / len(values_pd.index.levels[0])}")
		print(f"Reshaping to {nodes_nb}, {-1}, {values_pd.shape[1] - 1}")

	x = torch.tensor(
		values_pd.drop(columns=["Close"]).to_numpy().reshape((nodes_nb, -1, values_pd.shape[1] - 1)), dtype=torch.float32
	)
	x = x.transpose(1, 2)
	close_prices = torch.tensor(
		values_pd[["Close"]].to_numpy().reshape((nodes_nb, -1)), dtype=torch.float32
	)
	edge_index, edge_weight = torch.zeros((2, edge_nb), dtype=torch.long), torch.zeros((edge_nb,), dtype=torch.float32)
	count = 0
	for i in range(nodes_nb):
		for j in range(nodes_nb):
			if (weight := adj_pd[i, j]) != 0:
				edge_index[0, count], edge_index[1, count] = i, j
				edge_weight[count] = weight
				count += 1

	return x, close_prices, edge_index, edge_weight


def get_stocks_labels() -> list[str]:
	"""
	Retrieves the labels (symbols) of the dataset stocks
	:return: The list of stock labels
	"""
	return pd.read_csv("../data/SP100/raw/values.csv")["Symbol"].unique().tolist()
