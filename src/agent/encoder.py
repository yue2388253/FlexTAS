import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNModel(nn.Module):
    # todo: use GNN such as GCNConv
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        # define the layers and operations of your GNN here
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, adj_matrix, node_features):
        """
        adj_matrix: Batched adjacency matrices of the graphs.
            Shape: [batch_size, num_nodes, num_nodes]
        node_feats: Batched node feature matrices of the graphs.
            Shape: [batch_size, num_nodes, input_dim]
        """
        # First graph convolution layer
        node_feats = F.relu(self.conv1(node_features))
        node_feats = torch.bmm(adj_matrix, node_feats)

        # Second graph convolution layer
        node_feats = F.relu(self.conv2(node_feats))
        node_feats = torch.bmm(adj_matrix, node_feats)

        # Flatten the output
        batch_size, num_nodes, num_dim = node_feats.shape
        node_feats = node_feats.view(batch_size, num_nodes * num_dim)

        return node_feats


def dict_to_batch(obs):
    # Convert the dict observations to PyTorch Geometric Data objects
    data_list = []
    num_batch = obs['adjacency_matrix'].shape[0]
    print(num_batch)
    for i in range(obs['adjacency_matrix'].shape[0]):
        edge_index = obs['adjacency_matrix'][i]
        x = obs['features_matrix'][i]
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    # Convert the list of Data objects to a Batch
    batch = Batch.from_data_list(data_list)

    return batch


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        state = observation_space.sample()
        num_nodes, num_features = state['features_matrix'].shape

        embedding_dim = 64
        super().__init__(observation_space, features_dim=num_nodes * embedding_dim)

        self.graph_encoder = GNNModel(num_features, 64, embedding_dim)

    def forward(self, observations) -> torch.Tensor:
        batch = dict_to_batch(observations)
        return self.graph_encoder(observations['adjacency_matrix'],
                                  observations['features_matrix'])
