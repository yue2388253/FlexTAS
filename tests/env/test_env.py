import logging
import os.path

from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
import unittest

from definitions import ROOT_DIR
from src.env.env import NetEnv
from src.network.from_json import generate_net_flows_from_json


class TestEnvState(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/smt_output/FlexTAS_50_1204.json'))
        self.env = NetEnv(graph, flows)
        self.state_encoder = self.env.state_encoder

    def test_graph(self):
        graph = self.state_encoder.graph

        adjacency_matrix = self.state_encoder.adjacency_matrix
        logging.debug(adjacency_matrix)

    def test_state(self):
        state = self.state_encoder.state()
        logging.debug(state)
        logging.debug(type(state))
        for key, value in state.items():
            logging.debug((key, value.shape))

        features_matrix = state['features_matrix']
        logging.debug(features_matrix)
        self.assertTrue(self.state_encoder.observation_space.contains(state))


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/smt_output/FlexTAS_50_1204.json'))
        self.env = NetEnv(graph, flows)

    def test_check_env(self):
        check_env(self.env)

    def test_action(self):
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))

        gating_list = [0, 1, 0]
        for gating in gating_list:
            obs, reward, done, _, info = self.env.step([0, gating])
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertFalse(done)

        obs, reward, done, _, info = self.env.step([1, 1])
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertFalse(done)

        obs, reward, done, _, info = self.env.step([0, 1])
        self.assertFalse(done)

    def test_action_masks(self):
        model = MaskablePPO("MultiInputPolicy", self.env, verbose=1)
        model.learn(500)

