import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import AGNNConv, global_mean_pool, GAT, GATConv, GlobalAttention, Linear
from torch_geometric_temporal.nn import GraphAAGCN
import torch
from typing import Sequence, Union
from dataclasses import dataclass
import numpy as np


class SemiTemporalSwapGNN(nn.Module):
    def __init__(self, feature_dim_node: int = 3,
                 out_channels: int = 24,
                 hidden_channels: int = 12,
                 fc_hidden_dim: int = 128,
                 num_gat_layers: int = 4,
                 num_mlp_layers:int = 3,
                 num_heads=2,
                 activation=nn.ReLU,
                 num_nodes: int = None,
                 num_locations: int = 15,
                 for_active: bool = True,
                 device: str = "cuda",
                 lr: float = 3e-4,
                 optimizer: nn.Module = torch.optim.Adam,
                 ):
        super().__init__()
        self.num_nodes = num_nodes
        self.for_active = for_active
        self.num_locations = num_locations
        out_dim = 1
        self.type_embedding = nn.Embedding(4, feature_dim_node)
        self.activation = activation()
        self.att = GATConv(feature_dim_node + 2, hidden_channels//num_heads, heads=num_heads)
        self.hidden_atts = nn.ModuleList()
        for _ in range(num_gat_layers - 2):
            self.hidden_atts.append(GATConv(hidden_channels, hidden_channels//num_heads, heads=num_heads))

        self.final_att = GATConv(hidden_channels, hidden_channels//num_heads, heads=num_heads)

        self.temporal_layer = nn.Sequential(
            nn.LSTM(hidden_channels, hidden_channels, batch_first=True),
            activation(),
            nn.LSTM(hidden_channels, hidden_channels, batch_first=True),
            activation(),
            nn.LSTM(hidden_channels, hidden_channels, batch_first=True),
            activation(),
        )

        self.first_mlp = nn.Sequential(
            Linear(hidden_channels, fc_hidden_dim),
            activation()
        )
        self.mlps = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.mlps.append(Linear(fc_hidden_dim, fc_hidden_dim))
            self.mlps.append(activation())

        self.final_mlp = Linear(fc_hidden_dim, out_dim)

        self.hidden_state_projector = nn.Linear(hidden_channels, hidden_channels)

        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)


    def forward(self, data_full: torch_geometric.data.Data,
                batch: torch.Tensor = None,):

        time_step_embeddings = []
        for i in range(len(data_full.features)):
            data = data_full[i]
            x = self.type_embedding(data.x[0].long())
            time_index = data.x[2].unsqueeze(1).to(self.device)
            # normalize the requests
            mean_requests = torch.mean(data.x[1][self.num_locations:], dim=0)
            std_requests = torch.std(data.x[1][self.num_locations:], dim=0)
            requests_norm = (data.x[1][self.num_locations:] - mean_requests) / std_requests
            requests_final = torch.cat([data.x[1][:self.num_locations], requests_norm], dim=0)
            x = torch.cat([x, requests_final.unsqueeze(1), time_index], dim=-1)
            edge_index = data.edge_index
            x = self.att(x, edge_index, edge_attr=data.edge_attr)
            x = self.activation(x)
            for att in self.hidden_atts:
                x = att(x, edge_index, edge_attr=data.edge_attr)
                x = self.activation(x)
            spatial_embedding = self.final_att(x, edge_index, edge_attr=data.edge_attr)
            time_step_embeddings.append(spatial_embedding)


        print("hello")
        # now we will perform temporal modeling on the static embeddings (the static nodes, which are the locations)


