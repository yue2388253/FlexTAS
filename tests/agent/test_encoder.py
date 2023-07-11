import logging
import networkx as nx
import numpy as np
import os
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
import torch
from torchviz import make_dot
import unittest

from definitions import OUT_DIR
from src.agent.encoder import GNNModel, FeaturesExtractor
from src.env.env import NetEnv


class TestGNNModel(unittest.TestCase):
    def test_gnn_model(self):
        num_batch = 5
        num_nodes = 200
        num_features = 30

        graph = np.stack([nx.to_numpy_array(nx.fast_gnp_random_graph(num_nodes, 0.5)) for _ in range(num_batch)])
        features = np.random.normal(size=(num_batch, num_nodes, num_features))

        graph_conv_units = 32
        embedding_dim = 16
        gnn_model = GNNModel(num_features, graph_conv_units, embedding_dim)

        output = gnn_model(torch.from_numpy(graph).float(), torch.tensor(features).float())

        self.assertEqual(output.shape, (num_batch, num_nodes * embedding_dim))

        # Plot the model
        make_dot(output,
                 params=dict(list(gnn_model.named_parameters()))).render(os.path.join(OUT_DIR, 'model_GNN'),
                                                                         format="png")


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
        self.assertEqual(out.shape, (1, (self.env.state_encoder.num_operations + 2) * 64))

    def test_integration(self):
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )
        model = MaskablePPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)
        model.learn(500)

