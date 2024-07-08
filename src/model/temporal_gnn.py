import torch
import torch.nn as nn
import torch_geometric
from torch.nn import Linear
import numpy as np
from typing import List
from torch_geometric.data import Data, Batch

from src.model.gnn import NetworkGraphEmbeddingConstructor, QNetworkOutput


class SemiTemporalEmbeddingConstructor(nn.Module):
    """
    This class constructs embeddings that capture both spatial and temporal information from graph-structured data.
    """
    def __init__(self,
                 device: str = "cuda",
                 feature_size: int = 4,
                 embedding_size: int = 32,
                 n_heads: int = 4,
                 n_time_layers: int = 3,
                 n_layers: int = 3,
                 dropout_rate: float = 0.,
                 dense_neurons: int = 128,
                 edge_dim: int = 1,
                 num_locations: int = 15,
                 for_active: bool = True,
                 ):
        """
        Initializes the SemiTemporalEmbeddingConstructor.

        Args:
            device (str): Device to run the model on, e.g., "cuda" or "cpu".
            feature_size (int): Size of the input features for each node.
            embedding_size (int): Size of the embedding vector for each node.
            n_heads (int): Number of attention heads in the graph embedding constructor.
            n_time_layers (int): Number of LSTM layers for temporal modeling.
            n_layers (int): Number of layers in the graph embedding constructor.
            dropout_rate (float): Dropout rate for regularization.
            dense_neurons (int): Number of neurons in the dense layers of the graph embedding constructor.
            edge_dim (int): Dimensionality of edge features.
            num_locations (int): Number of unique locations to consider in the graph.
            for_active (bool): Whether to use active masks or passive masks.
        """
        super().__init__()
        self.for_active = for_active
        self.num_locations = num_locations

        # Graph embedding constructor
        self.node_embedding_constructor = NetworkGraphEmbeddingConstructor(
            feature_size=feature_size,
            embedding_size=embedding_size,
            edge_dim=edge_dim,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            dense_neurons=dense_neurons,
            n_layers=n_layers
        )

        # Linear transformation for node embeddings
        self.node_embedding_transformation = nn.Sequential(
            Linear(embedding_size, embedding_size),
        )

        # LSTM for temporal modeling
        self.temporal_layer = nn.LSTM(embedding_size, embedding_size, batch_first=True, num_layers=n_time_layers)
        self.time_horizon = None
    def _extract_correct_location_embedding(self, full_embedding, mask, data):
        chunks = []
        mask_chunks = []
        start = 0
        for batch in data:
            inner_start = start
            for d in batch:
                chunk = (start, self.num_locations + start)
                chunks.append(full_embedding[chunk[0]: chunk[1]])
                inner_start += len(d.active_mask)
                start += len(d.active_mask)
            mask_chunks.append(mask[chunk[0]: chunk[1]])

        return torch.cat(chunks), torch.stack(mask_chunks, dim=0)

    def _extract_time_varying_nodes(self, full_embedding):
        final_embedding = []
        """
        for loc in range(self.num_locations):
            inner_loc_counter = 0
            for step in range(0, full_embedding.shape[0], self.time_horizon * self.num_locations):
                node_indices = torch.arange(loc,full_embedding.shape[0], self.num_locations)
                final_embedding.append(full_embedding[
                                           self.num_locations * self.time_horizon * inner_loc_counter:
                                           self.num_locations * self.time_horizon * (inner_loc_counter + 1)]
                                       [node_indices])
        """
        last_step = 0
        for step in range(self.time_horizon * self.num_locations,
                          full_embedding.shape[0] + self.time_horizon * self.num_locations,
                          self.time_horizon * self.num_locations):
            inner_embedding = []
            for loc in range(self.num_locations):
                node_indices = torch.arange(loc + last_step, loc + step, self.num_locations)
                inner_embedding.append(full_embedding[node_indices])
            last_step = step
            final_embedding.append(torch.stack(inner_embedding))

        return torch.stack(final_embedding)

    def forward(self, data_full: List[torch_geometric.data.Data]):
        """
        Forward pass to compute spatial and temporal embeddings.

        Args:
            data_full (List[torch_geometric.data.Data]): List of graph data at different time steps.

        Returns:
            final_output (torch.Tensor): Final embeddings after temporal modeling.
            masks (torch.Tensor): Masks indicating active/passive nodes.
        """
        if not isinstance(data_full[0], list):
            data_full = [data_full]

        self.time_horizon = len(data_full[0])

        temp_graph = Batch.from_data_list([Batch.from_data_list(d) for d in data_full])
        location_spatial_embeddings_list = []

        #for temp_graph in data:
        # Use active or passive mask based on the flag
        mask = temp_graph.active_mask if self.for_active else temp_graph.passive_mask
        # Combine node type and update step features
        x = torch.cat([
            temp_graph.type.unsqueeze(1),
            temp_graph.requests.unsqueeze(1),
            temp_graph.update_step.unsqueeze(1)
        ], dim=1)
        edge_index = temp_graph.edge_index
        edge_attr = temp_graph.latency.unsqueeze(1).to(torch.float)
        # Create node embeddings using the graph embedding constructor
        x = self.node_embedding_constructor(x, edge_index, edge_attr)
        # Apply linear transformation to node embeddings
        location_spatial_embedding = self.node_embedding_transformation(x)
        location_spatial_embeddings_list.append(location_spatial_embedding[:self.num_locations])

        # Stack location-specific embeddings
        location_spatial_embeddings, mask_chunked = self._extract_correct_location_embedding(location_spatial_embedding, mask, data_full)
        location_spatial_embeddings_time_varying_node = self._extract_time_varying_nodes(location_spatial_embeddings)

        #merge the first two dimensions
        location_spatial_embeddings_time_varying_node = location_spatial_embeddings_time_varying_node.view(
            location_spatial_embeddings_time_varying_node.shape[0] *
            location_spatial_embeddings_time_varying_node.shape[1],
            location_spatial_embeddings_time_varying_node.shape[2],
            location_spatial_embeddings_time_varying_node.shape[3]
        )

        # **What do we input in the LSTM?** We have (batch_size x num_locations x num_timesteps x embedding_dim)
        # dimension We merge the first two dimensions to get huge embedding that contains for each location its
        # embedding over the time steps. We do this over a whole batch, so in the batch we have a all node states
        # over all the time steps for each graph in the batch. This is then used for the LSTM and we model the time
        # sequence of nodes with the LSTM (time_horizon node instances per element in the huge batch). Temporal
        # modeling with LSTM
        lstm_out, (hn, cn) = self.temporal_layer(location_spatial_embeddings_time_varying_node)
        final_embedding = hn[-1]  # Take the last hidden state of the top layer
        final_embedding = torch.tanh(final_embedding)

        return final_embedding, mask_chunked


class QNetworkSemiTemporal(nn.Module):
    """
    This class defines a Q-network that uses spatial and temporal embeddings to compute Q-values for a reinforcement learning task.
    """
    def __init__(self,
                 device: str = "cuda",
                 feature_size: int = 4,
                 embedding_size: int = 32,
                 n_heads: int = 4,
                 n_time_layers: int = 3,
                 n_layers: int = 3,
                 dropout_rate: float = 0.,
                 dense_neurons: int = 128,
                 edge_dim: int = 1,
                 num_locations: int = 15,
                 for_active: bool = True,
                 lr: float = 0.0001,
                 optimizer: object = torch.optim.Adam,
                 reduce_action_space: bool = True
                 ):
        """
        Initializes the QNetworkSemiTemporal.

        Args:
            device (str): Device to run the model on, e.g., "cuda" or "cpu".
            feature_size (int): Size of the input features for each node.
            embedding_size (int): Size of the embedding vector for each node.
            n_heads (int): Number of attention heads in the graph embedding constructor.
            n_time_layers (int): Number of LSTM layers for temporal modeling.
            n_layers (int): Number of layers in the graph embedding constructor.
            dropout_rate (float): Dropout rate for regularization.
            dense_neurons (int): Number of neurons in the dense layers of the graph embedding constructor.
            edge_dim (int): Dimensionality of edge features.
            num_locations (int): Number of unique locations to consider in the graph.
            for_active (bool): Whether to use active masks or passive masks.
            lr (float): Learning rate for the optimizer.
            optimizer (object): Optimizer class for training.
            reduce_action_space (bool): Flag for action space reduction.
        """
        super().__init__()
        out_dim = 2
        # Initialize the embedding constructor
        self.spatial_and_temporal_embedding_constructor = SemiTemporalEmbeddingConstructor(
            feature_size=feature_size,
            num_locations=num_locations,
            n_layers=n_layers,
            n_time_layers=n_time_layers,
            dropout_rate=dropout_rate,
            embedding_size=embedding_size,
            dense_neurons=dense_neurons,
            edge_dim=edge_dim,
            for_active=for_active,
            n_heads=n_heads
        )

        # Transformation layers for embedding to Q-values
        self.node_embedding_transformation = nn.Sequential(
            Linear(embedding_size, dense_neurons),
            nn.GELU(),
            Linear(dense_neurons, dense_neurons // 2),
            nn.GELU(),
            Linear(dense_neurons // 2, dense_neurons // 4),
            nn.GELU(),
            Linear(dense_neurons // 4, out_dim)
        )
        self.num_locations = num_locations
        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)

    def construct_masks(self, action_mask):
        """
        Constructs masks for valid actions.

        Args:
            action_mask (torch.Tensor): Mask indicating valid actions.

        Returns:
            mask (torch.Tensor): Mask for valid actions.
            remove_mask (torch.Tensor): Mask with invalid actions set to -inf.
        """
        mask = action_mask.clone()
        remove_mask = mask.clone()
        remove_mask[mask == 0] = -np.inf
        remove_mask[mask == -np.inf] = 0

        return mask, remove_mask

    def forward(self, data: List[torch_geometric.data.Data], action=None):
        """
        Forward pass to compute Q-values from spatial and temporal embeddings.

        Args:
            data (List[torch_geometric.data.Data]): List of graph data at different time steps.

        Returns:
            masked_output (torch.Tensor): Q-values with action masks applied.
        """
        # Get spatial and temporal embeddings and masks
        embedding, masks = self.spatial_and_temporal_embedding_constructor(data)

        # reshape into batch format
        embedding = embedding.view(embedding.shape[0] // self.num_locations, self.num_locations, -1)

        # Construct action masks
        mask, remove_mask = self.construct_masks(masks)
        # Transform embeddings to Q-values
        q_values_raw = self.node_embedding_transformation(embedding)
        remove_action = torch.argmax(q_values_raw[:, :, 0] + remove_mask, dim=1)
        mask[torch.arange(len(mask)), remove_action] = 0
        if action is not None:
            action_tensor = torch.tensor(action)
            inds = torch.where(action_tensor[:, 0] == action_tensor[:, 1])[0]
            mask[inds, action_tensor[inds, 0]] = 0
        # Apply action masks to the Q-values
        action_mask = torch.stack((remove_mask, mask), dim=2)
        masked_output = q_values_raw + action_mask
        return QNetworkOutput(q_values=masked_output, reduced_graph_node_mapping_remove=None,
                              reduced_graph_node_mapping_add=None)


class SemiTemporalCritic(nn.Module):
    def __init__(self,
                 device: str = "cuda",
                 feature_size: int = 4,
                 embedding_size: int = 32,
                 n_heads: int = 4,
                 n_time_layers: int = 3,
                 n_layers: int = 3,
                 dropout_rate: float = 0.,
                 dense_neurons: int = 128,
                 edge_dim: int = 1,
                 num_locations: int = 15,
                 for_active: bool = True,
                 lr: float = 0.0001,
                 optimizer: object = torch.optim.Adam,
                 reduce_action_space: bool = True
                 ):
        """
        Initializes the QNetworkSemiTemporal.

        Args:
            device (str): Device to run the model on, e.g., "cuda" or "cpu".
            feature_size (int): Size of the input features for each node.
            embedding_size (int): Size of the embedding vector for each node.
            n_heads (int): Number of attention heads in the graph embedding constructor.
            n_time_layers (int): Number of LSTM layers for temporal modeling.
            n_layers (int): Number of layers in the graph embedding constructor.
            dropout_rate (float): Dropout rate for regularization.
            dense_neurons (int): Number of neurons in the dense layers of the graph embedding constructor.
            edge_dim (int): Dimensionality of edge features.
            num_locations (int): Number of unique locations to consider in the graph.
            for_active (bool): Whether to use active masks or passive masks.
            lr (float): Learning rate for the optimizer.
            optimizer (object): Optimizer class for training.
            reduce_action_space (bool): Flag for action space reduction.
        """
        super().__init__()
        out_dim = 1
        # Initialize the embedding constructor
        self.spatial_and_temporal_embedding_constructor = SemiTemporalEmbeddingConstructor(
            feature_size=feature_size,
            num_locations=num_locations,
            n_layers=n_layers,
            n_time_layers=n_time_layers,
            dropout_rate=dropout_rate,
            embedding_size=embedding_size,
            dense_neurons=dense_neurons,
            edge_dim=edge_dim,
            for_active=for_active,
            n_heads=n_heads
        )

        # Transformation layers for embedding to Q-values
        self.node_embedding_transformation = nn.Sequential(
            Linear(embedding_size, dense_neurons),
            nn.GELU(),
            Linear(dense_neurons, dense_neurons // 2),
            nn.GELU(),
            Linear(dense_neurons // 2, dense_neurons // 4),
            nn.GELU(),
            Linear(dense_neurons // 4, out_dim)
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.num_locations = num_locations
        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.device = device
        self.to(self.device)

    def forward(self, data: List[torch_geometric.data.Data], action=None):
        """
        Forward pass to compute Q-values from spatial and temporal embeddings.

        Args:
            data (List[torch_geometric.data.Data]): List of graph data at different time steps.

        Returns:
            masked_output (torch.Tensor): Q-values with action masks applied.
        """
        # Get spatial and temporal embeddings and masks
        embedding, masks = self.spatial_and_temporal_embedding_constructor(data)

        # reshape into batch format
        embedding = embedding.view(embedding.shape[0] // self.num_locations, self.num_locations, -1)
        pooled_embedding = self.pooling(embedding.permute(0,2,1))
        values = self.node_embedding_transformation(pooled_embedding.squeeze())

        return values