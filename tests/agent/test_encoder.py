import logging
from sb3_contrib import MaskablePPO
import torch
from torch_geometric.data import Data, Batch
import unittest

from src.agent.encoder import GinModel, FeaturesExtractor
from src.env.env import NetEnv


class TestGinModel(unittest.TestCase):
    def setUp(self):
        self.num_features = 27
        self.model = GinModel(self.num_features)

    def test_forward(self):
        # Create a batch of 5 graphs with 80 nodes each and 27-dimensional node features
        data_list = []
        num_nodes = 80
        num_edges = 200
        for _ in range(5):
            x = torch.randn(num_nodes, self.num_features)
            edge_index = torch.randint(num_nodes, (2, num_edges))  # Randomly connect the nodes
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)

        # Pass the batch through the model
        out = self.model(batch)

        # Check that the output has the correct shape
        self.assertEqual(out.shape, (5, 64))


class TestFeaturesEncoder(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.env = NetEnv()
        self.observation_space = self.env.observation_space
        self.features_extractor = FeaturesExtractor(self.observation_space)

    def test_embedding(self):
        obs, _ = self.env.reset()
        # convert to tensor
        obs = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        out = self.features_extractor(obs)
        self.assertEqual(out.shape, (1, 192))

    def test_integration(self):
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )
        model = MaskablePPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        model.learn(500)
