import gymnasium as gym
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GinModel(torch.nn.Module):
    def __init__(self, num_features, dim1=32, dim2=64, embed_dim=64):
        super(GinModel, self).__init__()
        nn1 = Sequential(Linear(num_features, dim1), ReLU(), Linear(dim1, dim2))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim2)
        nn2 = Sequential(Linear(dim2, dim2), ReLU(), Linear(dim2, embed_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(embed_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Filter out padded edges
        valid_edge_mask = (edge_index >= 0).all(dim=0)
        edge_index = edge_index[:, valid_edge_mask]

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = global_mean_pool(x, batch)  # Add average pooling here
        return x


def dict_to_batch(obs):
    # Convert the dict observations to PyTorch Geometric Data objects
    data_list = []
    num_batch = obs['adjacency_matrix'].shape[0]
    for i in range(num_batch):
        edge_index = obs['adjacency_matrix'][i]

        int_dtypes = (torch.uint8, torch.int8, torch.int32, torch.int64)
        if edge_index.dtype not in int_dtypes:
            # stable baselines3 always cast an int64 numpy array to a tensor of float.
            # We should cast the tensor of float to int for GINConv to work.
            edge_index = edge_index.to(torch.int64)

        x = obs['features_matrix'][i]
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    # Convert the list of Data objects to a Batch
    batch = Batch.from_data_list(data_list)
    return batch


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        embedding_dim = 128
        array_embedding = 64
        graph_embedding = embedding_dim - array_embedding
        super().__init__(observation_space, features_dim=embedding_dim)

        self.array_encoder = nn.Sequential(
            nn.Linear(
                observation_space['flow_feature'].shape[0] + observation_space['link_feature'].shape[0],
                array_embedding),
            nn.ReLU()
        )

        _, num_features = observation_space['features_matrix'].shape
        self.graph_encoder = GinModel(num_features, embed_dim=graph_embedding)

    def forward(self, observations) -> torch.Tensor:
        array_encoded = self.array_encoder(
            torch.cat([
                observations['flow_feature'],
                observations['link_feature']
            ], dim=1)
        )

        batch = dict_to_batch(observations)
        graph_encoded = self.graph_encoder(batch)

        return torch.cat([array_encoded, graph_encoded], dim=1)
