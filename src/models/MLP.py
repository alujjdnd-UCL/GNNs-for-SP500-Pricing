import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_size: int,
            layers_nb: int = 2,
            output_activation: nn.Module = None,
            use_gat: bool = True  # Not used but retained for compatibility
    ):
        super(MLP, self).__init__()

        self.hidden_size = hidden_size
        self.layers_nb = layers_nb

        # Define layers
        self.layers = nn.ModuleList(
            [nn.Linear(in_channels, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.layers_nb - 1)]
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_channels),
            output_activation if output_activation is not None else nn.Identity(),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, n1, n2) -> torch.Tensor:
        """
        Performs a forward pass on the MLP.
        :param x: Input tensor of shape (Nodes_nb, Features_nb, SeqLength).
        :return: Output tensor of shape (Nodes_nb, out_channels).
        """
        # Flatten the sequence dimension
        for t in range(x.shape[-1]):
            h = x[:, :, t]  # Extract features for time step t
            for i, layer in enumerate(self.layers):
                h = self.activation(layer(h))
        return self.out(h)
