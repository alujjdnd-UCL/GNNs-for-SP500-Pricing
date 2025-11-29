import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent.dcrnn import DConv as PyG_DConv

class DCRNN(nn.Module):
    """
    Wrapper for torch_geometric_temporal.nn.convolutional.DConv

    Parameters:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        hidden_size (int): Number of hidden units in the recurrent layer.
        layers_nb (int): Number of recurrent layers. Default is 2.
        output_activation (nn.Module): Optional activation function applied to the output. Default is None.
        use_gat (bool): Optional flag to indicate if GAT is used (not applicable in this wrapper).
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, layers_nb: int = 2,
                 output_activation: nn.Module = None, use_gat: bool = True):
        super(DCRNN, self).__init__()

        self.hidden_size = hidden_size
        self.layers_nb = layers_nb
        self.output_activation = output_activation

        # Create DConv layers using PyG implementation
        self.dconv_layers = nn.ModuleList([
            PyG_DConv(in_channels if i == 0 else hidden_size, hidden_size, K=2) for i in range(layers_nb)
        ])

        # Output layer
        self.out_layer = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DCRNN.

        Parameters:
            x (torch.Tensor): Input node features of shape [num_nodes, in_channels, sequence_length].
            edge_index (torch.Tensor): Graph edge indices of shape [2, num_edges].
            edge_weight (torch.Tensor): Edge weights of shape [num_edges].

        Returns:
            torch.Tensor: Output node features of shape [num_nodes, out_channels].
        """
        device = x.device
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        h_prev = [
            torch.zeros(x.shape[0], self.hidden_size, device=device) for _ in range(self.layers_nb)
        ]

        for t in range(x.shape[-1]):
            h = x[:, :, t].to(device)  # Extract features for the current timestep

            for i, layer in enumerate(self.dconv_layers):
                h = layer(h, edge_index, edge_weight)
                h_prev[i] = h

        # Final output projection
        h = self.out_layer(h_prev[-1])

        if self.output_activation:
            h = self.output_activation(h)

        return h
