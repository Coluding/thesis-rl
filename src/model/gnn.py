import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.data import Data, Batch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import (
    GCNConv, global_mean_pool, GAT, GATConv, GlobalAttention, TransformerConv, TopKPooling, global_max_pool
)
import torch
from typing import Sequence, Union
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

@dataclass
class SwapGNNOutput:
    logits: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor



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
                 for_active: bool = True,
                 device: str = "cuda",
                 lr: float = 3e-4,
                 optimizer: nn.Module = torch.optim.Adam,
                 ):
        super(SwapGNN, self).__init__()
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


    def _compute_conv_output_dim(self, input_shape: Sequence[int]):
        x = torch.rand(input_shape).unsqueeze(0)
        edges = torch.zeros(2, 2).to(torch.int64)
        x = self.conv1(x, edges)
        for conv in self.hidden_convs:
            x = conv(x, edges)

        x = self.conv2(x, edges)

        return torch.prod(torch.tensor(x.shape)).item()

    def forward(self, data: torch_geometric.data.Data,
                batch: torch.Tensor = None,) -> SwapGNNOutput:
        #x, edge_index = data.x, data.edge_index
        if len(data.type.shape) > 2:
            raise ValueError("Type should be a 1D tensor. Be sure it is not one hot encoded.")
        x = self.type_embedding(data.type.long())
        time_index = data.update_step.unsqueeze(1).to(self.device)
        # normalize the requests
        mean_requests = torch.mean(data.requests[self.num_locations:], dim=0)
        std_requests = torch.std(data.requests[self.num_locations:], dim=0)
        requests_norm = (data.requests[self.num_locations:] - mean_requests) / std_requests
        requests_final = torch.cat([data.requests[:self.num_locations], requests_norm], dim=0)
        x = torch.cat([x, requests_final.unsqueeze(1), time_index], dim=-1)
        edge_index = data.edge_index
        x = self.att(x, edge_index, edge_attr=data.latency)
        x = self.activation(x)
        for att in self.hidden_atts:
            x = att(x, edge_index, edge_attr=data.latency)
            x = self.activation(x)
        h_l1 = self.final_att(x, edge_index, edge_attr=data.latency)

        x = self.first_mlp(h_l1)
        for mlp in self.mlps:
            x = mlp(x)
        if batch is not None:
            logits_list = []
            actions_list = []
            log_probs_list = []

            unique_batches = data.batch.unique()

            for graph_idx in unique_batches:
                action_mask = data.active_mask[data.batch == graph_idx] if self.for_active else data.passive_mask[
                    data.batch == graph_idx]
                mask = action_mask.clone()
                remove_mask = mask.clone()
                remove_mask[:self.num_locations][mask[:self.num_locations] == 0] = -float('inf')
                remove_mask[:self.num_locations][mask[:self.num_locations] == -float('inf')] = 0

                removed_facility_logits = self.final_mlp(x[data.batch == graph_idx])
                removed_facility_logits = removed_facility_logits.squeeze() + remove_mask
                pi1 = torch.distributions.Categorical(logits=removed_facility_logits.squeeze())
                action1 = pi1.sample()

                mask[action1] = 0
                new_facility_logits = h_l1[data.batch == graph_idx] @ nn.Tanh()(
                    self.hidden_state_projector(h_l1[data.batch == graph_idx][action1]).unsqueeze(0)).transpose(-2, -1)
                new_facility_logits = new_facility_logits.squeeze() + mask
                pi2 = torch.distributions.Categorical(logits=new_facility_logits.squeeze())
                action2 = pi2.sample()

                log_probs = torch.stack([pi1.log_prob(action1), pi2.log_prob(action2)])
                logits = torch.stack([removed_facility_logits, new_facility_logits]).squeeze()
                actions = torch.stack([action1, action2])

                logits_list.append(logits)
                actions_list.append(actions)
                log_probs_list.append(log_probs)

            logits = torch.stack(logits_list)
            actions = torch.stack(actions_list)
            log_probs = torch.stack(log_probs_list)
        else:
            # It should be possible that removed facility is the same as the new facility, then no movement is done
            action_mask = data.active_mask if self.for_active else data.passive_mask

            #We need to clone the mask to avoid changing the original mask
            mask = action_mask.clone()

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

            log_probs = torch.stack([pi1.log_prob(action1), pi2.log_prob(action2)])
            logits = torch.stack([removed_facility_logits, new_facility_logits]).squeeze()
            actions = torch.stack([action1, action2],)

        # Return action can be read as follows: Remove facility at index action1 and add facility at index action2
        return SwapGNNOutput(logits, actions, log_probs)

    @staticmethod
    def create_batch_tensor(nodes_per_graph, total_nodes):
        num_graphs = total_nodes // nodes_per_graph

        # Check for any inconsistency
        if total_nodes % nodes_per_graph != 0:
            raise ValueError("Total number of nodes is not a multiple of nodes per graph.")

        batch_tensor = torch.cat([torch.full((nodes_per_graph,), i) for i in range(num_graphs)])

        return batch_tensor


class CriticSwapGNN(nn.Module):
    def __init__(self,
                 feature_dim_node: int = 3,
                 hidden_channels: int = 12,
                 fc_hidden_dim: int = 128,
                 num_gat_layers: int = 4,
                 num_mlp_layers:int = 3,
                 num_heads=4,
                 activation=nn.ReLU,
                 num_nodes: int = None,
                 num_locations: int = 15,
                 for_active: bool = True,
                 device: str = "cuda",
                 optimizer: nn.Module = torch.optim.Adam,
                 lr: float = 3e-4,
                 ):
        super(CriticSwapGNN, self).__init__()
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

        critic_layer = [
            nn.Sequential(
                Linear(hidden_channels, fc_hidden_dim),
                activation())
        ]
        for _ in range(num_mlp_layers - 2):
            critic_layer.append(Linear(fc_hidden_dim, fc_hidden_dim))
            critic_layer.append(activation())

        critic_layer.append(nn.Sequential(
            Linear(fc_hidden_dim, 1),
            activation()))

        self.critic = nn.Sequential(*critic_layer)
        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)

    def forward(self, data: Union[Data, Batch], batch: torch.Tensor = None):
        if len(data.type.shape) > 2:
            raise ValueError("Type should be a 1D tensor. Be sure it is not one-hot encoded.")
        x = self.type_embedding(data.type.long())
        time_index = data.update_step.unsqueeze(1).to(self.device)
        # Normalize the requests
        mean_requests = torch.mean(data.requests[self.num_locations:], dim=0)
        std_requests = torch.std(data.requests[self.num_locations:], dim=0)
        requests_norm = (data.requests[self.num_locations:] - mean_requests) / (std_requests + 1e-6)
        requests_final = torch.cat([data.requests[:self.num_locations], requests_norm], dim=0)

        x = torch.cat([x, requests_final.unsqueeze(1), time_index], dim=-1)
        edge_index = data.edge_index

        x = self.att(x, edge_index, edge_attr=data.latency)
        x = self.activation(x)
        for att in self.hidden_atts:
            x = att(x, edge_index, edge_attr=data.latency)
            x = self.activation(x)
        x = self.final_att(x, edge_index, edge_attr=data.latency)

        # Node values
        node_values = self.critic(x)

        # Apply global mean pooling to get a graph-level embedding
        graph_value = global_mean_pool(node_values, batch)

        return graph_value


class TransformerGNN(nn.Module):
    def __init__(self,
                 device: str = "cuda",
                 feature_size: int = 4,
                 embedding_size: int = 32,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 top_k_every_n: int = 1,
                 dropout_rate: float = 0.,
                 top_k_ratio: float = 0.1,
                 dense_neurons: int = 128,
                 edge_dim: int = 1,
                 lr: float = 0.0001,
                 optimizer: nn.Module = torch.optim.Adam):

        super().__init__()
        self.top_k_every_n = top_k_every_n
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        self.conv1 = TransformerConv(
            feature_size, embedding_size, heads=n_heads, dropout=dropout_rate, edge_dim=1, beta=True
        )

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        # Linear layers
        self.linear1 = Linear(embedding_size, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 1)

        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)

        self.device = device
        self.to(self.device)

    def forward(self, data, batch_index=None):
        # Initial transformation
        x = torch.cat([data.type.unsqueeze(1), data.requests.unsqueeze(1)], dim=1)
        edge_index = data.edge_index
        edge_attr = data.latency.unsqueeze(1).to(torch.float)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                # Add current representation
                #global_representation.append(torch.cat([global_max_pool(x, batch_index),
                #                                        global_mean_pool(x, batch_index)], dim=1))

                global_representation.append(global_mean_pool(x, batch_index))

        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x


class TransformerSwapGNN(nn.Module):
    def __init__(self,
                 device: str = "cuda",
                 feature_size: int = 4,
                 embedding_size: int = 32,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 dropout_rate: float = 0.,
                 top_k_ratio: float = 0.1,
                 dense_neurons: int = 128,
                 edge_dim: int = 1,
                 num_locations: int = 15,
                 for_active: bool = True,
                 lr: float = 0.0001,
                 optimizer: nn.Module = torch.optim.Adam
                 ):

        super().__init__()
        self.num_locations = 15
        self.for_active = for_active
        self.n_layers = n_layers
        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        self.conv1 = TransformerConv(
            feature_size, embedding_size, heads=n_heads, dropout=dropout_rate, edge_dim=1, beta=True
        )

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size,
                                                    embedding_size,
                                                    heads=n_heads,
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

        # Linear transformation layers to prepare attention embedding for the location selection logits
        self.linear1 = Linear(embedding_size, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))

        self.remove_facility_layer = Linear(dense_neurons // 2, 1)
        self.add_facility_projector = Linear(dense_neurons // 2, dense_neurons // 2)

        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)

    def forward(self, data, batch_index=None):
        # Initial transformation
        x = torch.cat([data.type.unsqueeze(1), data.requests.unsqueeze(1)], dim=1)
        edge_index = data.edge_index
        edge_attr = data.latency.unsqueeze(1).to(torch.float)
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))

        if batch_index is None:
            action_mask = data.active_mask if self.for_active else data.passive_mask
            mask = action_mask.clone()
            remove_mask = mask.clone()
            remove_mask[:self.num_locations][mask[:self.num_locations] == 0] = -np.inf
            remove_mask[:self.num_locations][mask[:self.num_locations] == -np.inf] = 0

            removed_facility_logits = self.remove_facility_layer(x)
            removed_facility_logits = removed_facility_logits.squeeze() + remove_mask
            pi1 = torch.distributions.Categorical(logits=removed_facility_logits.squeeze())
            action1 = pi1.sample()

            mask[action1] = 0
            # the idea is here that the dot product uses the embedding of the removed facility to select the new one
            new_facility_logits = x @ nn.Tanh()(self.add_facility_projector(x[action1]).unsqueeze(0)).transpose(-2, -1)
            new_facility_logits = new_facility_logits.squeeze() + mask
            pi2 = torch.distributions.Categorical(logits=new_facility_logits.squeeze())
            action2 = pi2.sample()

            log_probs = torch.stack([pi1.log_prob(action1), pi2.log_prob(action2)])
            logits = torch.stack([removed_facility_logits, new_facility_logits]).squeeze()
            actions = torch.stack([action1, action2])

        else:
            logits_list = []
            actions_list = []
            log_probs_list = []

            unique_batches = batch_index.unique()

            for graph_idx in unique_batches:
                action_mask = data.active_mask[data.batch == graph_idx] if self.for_active else data.passive_mask[
                    data.batch == graph_idx]
                mask = action_mask.clone()
                remove_mask = mask.clone()
                remove_mask[:self.num_locations][mask[:self.num_locations] == 0] = -float('inf')
                remove_mask[:self.num_locations][mask[:self.num_locations] == -float('inf')] = 0

                removed_facility_logits = self.remove_facility_layer(x[data.batch == graph_idx])
                removed_facility_logits = removed_facility_logits.squeeze() + remove_mask
                pi1 = torch.distributions.Categorical(logits=removed_facility_logits.squeeze())
                action1 = pi1.sample()

                mask[action1] = 0
                new_facility_logits = x[data.batch == graph_idx] @ nn.Tanh()(
                    self.add_facility_projector(x[data.batch == graph_idx][action1]).unsqueeze(0)).transpose(-2, -1)
                new_facility_logits = new_facility_logits.squeeze() + mask
                pi2 = torch.distributions.Categorical(logits=new_facility_logits.squeeze())
                action2 = pi2.sample()

                log_probs = torch.stack([pi1.log_prob(action1), pi2.log_prob(action2)])
                logits = torch.stack([removed_facility_logits, new_facility_logits]).squeeze()
                actions = torch.stack([action1, action2])

                logits_list.append(logits)
                actions_list.append(actions)
                log_probs_list.append(log_probs)

            logits = torch.stack(logits_list)
            actions = torch.stack(actions_list)
            log_probs = torch.stack(log_probs_list)

        return SwapGNNOutput(logits, actions, log_probs)


class QNetworkSwapGNN:
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
                 for_active: bool = True,
                 device: str = "cuda",
                 optimizer: nn.Module = torch.optim.Adam,
                 lr: float = 3e-4,
                 ):
        super(QNetworkSwapGNN, self).__init__()
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

        critic_layer = []
        for _ in range(num_mlp_layers):
            critic_layer.append(Linear(hidden_channels, hidden_channels))
            critic_layer.append(activation())

        critic_layer.append(Linear(hidden_channels, 1))
        self.critic = nn.Sequential(*critic_layer)

        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)