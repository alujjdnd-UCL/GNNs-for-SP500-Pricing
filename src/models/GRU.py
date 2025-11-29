import torch
from torch import nn


class GRU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        layers_nb: int = 2,
        output_activation: nn.Module = None,
        use_gat: bool = True  # Not used but retained for compatibility
    ):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.layers_nb = layers_nb

        # GRU layers
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=layers_nb,
            batch_first=True
        )
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_channels),
            output_activation if output_activation is not None else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, n1, n2) -> torch.Tensor:
        """
        Performs a forward pass on the GRU model.
        :param x: Input tensor of shape (Nodes_nb, Features_nb, SeqLength).
        :return: Output tensor of shape (Nodes_nb, out_channels).
        """
        # Rearrange x to (batch_size, seq_length, feature_dim)
        x = x.permute(0, 2, 1)

        # Pass through the GRU
        h, _ = self.gru(x)  # h: (Nodes_nb, SeqLength, Hidden_Size)

        # Use the final hidden state from the last time step
        h_last = h[:, -1, :]  # (Nodes_nb, Hidden_Size)

        # Pass the last hidden state through the output layer
        return self.out(h_last)
