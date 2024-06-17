import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import GCNConv, global_mean_pool, GAT, GATConv, GlobalAttention, Linear
import torch
from typing import Sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class CriticGNNOutput:
    x: torch.Tensor
    pooled_x: torch.Tensor
    value: torch.Tensor

@dataclass
class ActorGNNOutput:
    x: torch.Tensor
    pooled_x: torch.Tensor
    pa: torch.Tensor


class CriticGCNN(nn.Module):
    def __init__(self, feature_dim_node: int = 3, out_channels: int = 24, hidden_channels: int = 12,
                 fc_hidden_dim: int = 128,
                 num_layers: int = 4, num_heads=4,
                 num_nodes: int = None):
        super(CriticGCNN, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = GATConv(feature_dim_node, hidden_channels, heads=num_heads)
        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads))

        self.conv2 = GATConv(hidden_channels, out_channels)

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


class ActorGCNN(nn.Module):
    def __init__(self,
                 n_actions: int,
                 feature_dim_node: int = 3,
                 out_channels: int = 24,
                 hidden_channels: int = 12,
                 fc_hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_nodes: int = None):
        super(ActorGCNN, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = GCNConv(feature_dim_node, hidden_channels)
        self.hidden_convs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_convs.append(GCNConv(hidden_channels, hidden_channels))

        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.actor = nn.Sequential(
            nn.Linear(out_channels, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, n_actions)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, data: torch_geometric.data.Data,
                batch: torch.Tensor = None,) -> ActorGNNOutput:

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

        pa = self.actor(pooled_x)

        return ActorGNNOutput(x, pooled_x, nn.Softmax()(pa))

    @staticmethod
    def create_batch_tensor(nodes_per_graph, total_nodes):
        num_graphs = total_nodes // nodes_per_graph

        # Check for any inconsistency
        if total_nodes % nodes_per_graph != 0:
            raise ValueError("Total number of nodes is not a multiple of nodes per graph.")

        batch_tensor = torch.cat([torch.full((nodes_per_graph,), i) for i in range(num_graphs)])

        return batch_tensor


class SwapGNN(nn.Module):
    def __init__(self, feature_dim_node: int = 3,
                 out_channels: int = 24,
                 hidden_channels: int = 12,
                 fc_hidden_dim: int = 128,
                 num_gat_layers: int = 4,
                 num_mlp_layers:int = 3,
                 num_heads=4,
                 activation=nn.ReLU,
                 num_nodes: int = None,
                 num_locations: int = 15,
                 for_active: bool = True
                 ):
        super(SwapGNN, self).__init__()
        self.num_nodes = num_nodes
        self.for_active = for_active
        self.num_locations = num_locations
        out_dim = 1
        self.type_embedding = nn.Embedding(4, feature_dim_node)
        self.activation = activation()
        self.att = GATConv(feature_dim_node + 1, hidden_channels//num_heads, heads=num_heads)
        self.hidden_atts = nn.ModuleList()
        for _ in range(num_gat_layers - 2):
            self.hidden_atts.append(GATConv(hidden_channels, hidden_channels//num_heads, heads=num_heads))

        self.final_att = GATConv(hidden_channels, hidden_channels//num_heads, heads=num_heads)

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
                batch: torch.Tensor = None,):
        #x, edge_index = data.x, data.edge_index
        if len(data.type.shape) > 2:
            raise ValueError("Type should be a 1D tensor. Be sure it is not one hot encoded.")
        x = self.type_embedding(data.type.long())
        # normalize the requests
        mean_requests = torch.mean(data.requests[self.num_locations:], dim=0)
        std_requests = torch.std(data.requests[self.num_locations:], dim=0)
        requests_norm = (data.requests[self.num_locations:] - mean_requests) / std_requests
        requests_final = torch.cat([data.requests[:self.num_locations], requests_norm], dim=0)
        x = torch.cat([x, requests_final.unsqueeze(1)], dim=-1)
        edge_index = data.edge_index
        x = self.att(x, edge_index)
        x = self.activation(x)
        for att in self.hidden_atts:
            x = att(x, edge_index)
            x = self.activation(x)
        h_l1 = self.final_att(x, edge_index)

        x = self.first_mlp(h_l1)
        for mlp in self.mlps:
            x = mlp(x)

        # It should be possible that removed facility is the same as the new facility, then no movement is done
        mask = data.active_mask if self.for_active else data.passive_mask
        remove_mask = mask.clone()
        remove_mask[:self.num_locations][mask[:self.num_locations] == 0] = -np.inf
        remove_mask[:self.num_locations][mask[:self.num_locations] == -np.inf] = 0

        removed_facility_logits = self.final_mlp(x)
        removed_facility_logits = removed_facility_logits.squeeze() + remove_mask
        pi1 = torch.distributions.Categorical(logits=removed_facility_logits.squeeze())
        action1 = pi1.sample()

        # TODO: maybe add a mask for the new facility, such that the model cannot select a facility that
        #  was not passive before

        # Remove the action1 facility from the mask so that we allow for no reconfiguration
        # In essence, this allows the model to select the same node for removing and selecting which is a no-op
        mask[action1] = 0
        new_facility_logits = h_l1 @ nn.Tanh()(
            self.hidden_state_projector(h_l1[action1]).unsqueeze(0)).transpose(-2, -1)
        new_facility_logits = new_facility_logits.squeeze() + mask
        pi2 = torch.distributions.Categorical(logits=new_facility_logits.squeeze())
        action2 = pi2.sample()

        logits = torch.stack([removed_facility_logits, new_facility_logits]).squeeze()
        actions = torch.stack([action1, action2],)

        # Return action can be read as follows: Remove facility at index action1 and add facility at index action2
        return logits, actions

    @staticmethod
    def create_batch_tensor(nodes_per_graph, total_nodes):
        num_graphs = total_nodes // nodes_per_graph

        # Check for any inconsistency
        if total_nodes % nodes_per_graph != 0:
            raise ValueError("Total number of nodes is not a multiple of nodes per graph.")

        batch_tensor = torch.cat([torch.full((nodes_per_graph,), i) for i in range(num_graphs)])

        return batch_tensor