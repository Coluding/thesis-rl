import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import GCNConv, global_mean_pool, Sequential
import torch
from typing import Sequence
from dataclasses import dataclass


@dataclass
class CriticGNNOutput:
    x: torch.Tensor
    pooled_x: torch.Tensor
    value: torch.Tensor


class CriticGNN(nn.Module):
    def __init__(self, feature_dim_node: int = 3, out_channels: int = 24, hidden_channels: int = 12,
                 fc_hidden_dim: int = 128,
                 num_layers: int = 4, num_nodes: int = None):
        super(CriticGNN, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(feature_dim_node, hidden_channels)
        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_convs.append(GCNConv(hidden_channels, hidden_channels))

        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.value = nn.Sequential(
            nn.Linear(out_channels, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, 1)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def _compute_conv_output_dim(self, input_shape: Sequence[int]):
        x = torch.rand(input_shape).unsqueeze(0)
        edges = torch.zeros(2, 2).to(torch.int64)
        x = self.conv1(x, edges)
        for conv in self.hidden_convs:
            x = conv(x, edges)

        x = self.conv2(x, edges)

        return torch.prod(torch.tensor(x.shape)).item()

    def forward(self, data: torch_geometric.data.Data,
                batch: torch.Tensor = None,) -> CriticGNNOutput:
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.hidden_convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.conv2(x, edge_index)

        if batch is None and self.num_nodes is not None:
            batch = self.create_batch_tensor(self.num_nodes, x.shape[0]).to(self.device)

        elif batch is None and self.num_nodes is None:
            raise ValueError("Either batch or num_nodes must be provided.")

        pooled_x = global_mean_pool(x, batch)

        return CriticGNNOutput(x, pooled_x, self.value(pooled_x))

    @staticmethod
    def create_batch_tensor(nodes_per_graph, total_nodes):
        num_graphs = total_nodes // nodes_per_graph

        # Check for any inconsistency
        if total_nodes % nodes_per_graph != 0:
            raise ValueError("Total number of nodes is not a multiple of nodes per graph.")

        batch_tensor = torch.cat([torch.full((nodes_per_graph,), i) for i in range(num_graphs)])

        return batch_tensor