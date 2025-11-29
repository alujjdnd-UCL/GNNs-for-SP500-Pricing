import os
import os.path as osp
from typing import Callable

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from src.datasets.dataset_utils import get_graph_in_pyg_format
from src.datasets.graph_construction.sector_correlation import get_sector_graph_wikidata


class SP500Stocks(Dataset):
    """
	Stock price data for the S&P 500 companies.
	The graph data built from the notebooks is used.
	"""

    def __init__(self, root: str = "../data/SP500/", values_file_name: str = "None",
                 adj_file_name: str = "adj_500_sectors_only.npy", past_window: int = 25, future_window: int = 1,
                 force_reload: bool = False, transform: Callable = None, what_edges_to_form = None,
                 graph_options: dict | None = None, dataset_options: dict | None = None) -> None:
        dataset_options = dataset_options or {}
        if "force_reload" in dataset_options:
            force_reload = dataset_options["force_reload"]
        self.symbols = None
        self.stock_prices = None
        self.symbol_industry_map = None
        self.graph_adj_matrix = None
        self.values_file_name = values_file_name
        self.past_window = past_window
        self.future_window = future_window
        self.what_edges_to_form = what_edges_to_form
        self.graph_options = graph_options or {}
        self.dataset_options = dataset_options
        self._verbose = bool(dataset_options.get("verbose", False) or self.graph_options.get("verbose", False))
        super().__init__(root, force_reload=force_reload, transform=transform)

    @property
    def raw_file_names(self) -> list[str]:
        return [
            self.values_file_name
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return [
            f'timestep_{idx}.pt' for idx in range(len(self))
        ]

    def get_adjacency_matrix(self):
        symbols_path = osp.join(self.root, 'filtered_symbols.csv')
        phases_path = osp.join(self.root, 'raw', self.values_file_name)

        if not osp.exists(symbols_path):
            raise FileNotFoundError(f"Expected symbol metadata at {symbols_path}")
        if not osp.exists(phases_path):
            raise FileNotFoundError(f"Expected phase data at {phases_path}")

        self.symbol_industry_map = pd.read_csv(symbols_path).set_index('Symbol')
        self.stock_prices = pd.read_csv(phases_path)

        # Identify which columns to pivot
        indicator_cols = [
            "NormClose",
            "ALR1W",
            "ALR2M",
            "RSI",
        ]

        # Pivot 'Close' separately, as it is the main one used by stocks_prices_pd
        pivot_stock_prices = self.stock_prices.pivot(index='Date', columns='Symbol', values='Close')

        # Here is where you capture the columns (symbol order).
        # We'll assume get_sector_graph_wikidata() does NOT reorder them.
        self.symbols = pivot_stock_prices.columns.tolist()

        # Create a list of pivoted DataFrames for each indicator
        indicator_pivot_pds = []
        for col in indicator_cols:
            col_pivot = self.stock_prices.pivot(index='Date', columns='Symbol', values=col)
            indicator_pivot_pds.append(col_pivot)

        # Pass everything into get_sector_graph_wikidata
        cache_dir = self.graph_options.get("cache_dir")
        if cache_dir:
            cache_dir = osp.join(self.root, cache_dir)

        graph_kwargs = {
            "visualise": self.graph_options.get("visualise", False),
            "what_edges_to_form": self.what_edges_to_form,
            "indicator_pivot_pds": indicator_pivot_pds,
            "enable_wikidata": self.graph_options.get("enable_wikidata", True),
            "cache_dir": cache_dir,
            "allow_external_requests": self.graph_options.get("allow_external_requests", True),
            "verbose": self.graph_options.get("verbose", False),
        }

        self.graph_adj_matrix = get_sector_graph_wikidata(
            symbol_industry_map_pd=self.symbol_industry_map,
            stocks_prices_pd=pivot_stock_prices,
            **graph_kwargs
        )

        return self.graph_adj_matrix

    def save_adj_matrix(self, file_name: str) -> None:
        """
        Saves the adjacency matrix (self.graph_adj_matrix) to a CSV file with
        the actual ticker symbols as row/column labels.
        """
        import pandas as pd
        import numpy as np

        if not hasattr(self, 'graph_adj_matrix') or self.graph_adj_matrix is None:
            raise ValueError("graph_adj_matrix not found or hasn't been created.")

        # self.graph_adj_matrix should be square: NxN
        n = self.graph_adj_matrix.shape[0]
        if self.graph_adj_matrix.shape[0] != self.graph_adj_matrix.shape[1]:
            raise ValueError("graph_adj_matrix must be square (NxN).")

        # You must have a list of symbols in the same order as the adjacency matrix
        # e.g. if your adjacency matrix is built in the order of these symbols
        # (from pivot_stock_prices.columns or some 'common_symbols' list).
        # We'll assume you stored it in self.symbols during adjacency creation.
        if not hasattr(self, 'symbols') or len(self.symbols) != n:
            raise ValueError("self.symbols not found or does not match adjacency size.")

        # Build a DataFrame so we can save row/col labels
        adjacency_df = pd.DataFrame(
            self.graph_adj_matrix,
            index=self.symbols,
            columns=self.symbols
        )

        directory = osp.dirname(file_name)
        if directory:
            os.makedirs(directory, exist_ok=True)

        adjacency_df.to_csv(file_name)
        print(f"Adjacency matrix saved to {file_name}")

    def download(self) -> None:
        pass

    def process(self) -> None:
        self.get_adjacency_matrix()

        if not isinstance(self.stock_prices.index, pd.MultiIndex):
            self.stock_prices = self.stock_prices.set_index(['Date', 'Symbol'])
        self.stock_prices = self.stock_prices.sort_index(level=['Symbol', 'Date'])

        x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
            values_pd=self.stock_prices,
            adj_pd=self.graph_adj_matrix,
            verbose=self._verbose
        )
        timestamps = [
            Data(
                x=x[:, :, idx:idx + self.past_window],
                edge_index=edge_index,
                edge_weight=edge_weight,
                close_price=close_prices[:, idx:idx + self.past_window],
                y=x[:, 0, idx + self.past_window:idx + self.past_window + self.future_window],
                close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
            ) for idx in range(x.shape[2] - self.past_window - self.future_window)
        ]
        for t, timestep in enumerate(timestamps):
            torch.save(
                timestep, osp.join(self.processed_dir, f"timestep_{t}.pt")
            )

    def len(self) -> int:
        values = pd.read_csv(osp.join(self.root, 'raw', self.values_file_name)).set_index(['Symbol', 'Date'])
        return len(values.loc[values.index[0][0]]) - self.past_window - self.future_window

    def get(self, idx: int) -> Data:
        data = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'), weights_only=False)
        return data