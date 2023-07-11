import logging
import os.path
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
import unittest

from definitions import ROOT_DIR
from src.env.env import NetEnv
from src.network.from_json import generate_net_flows_from_json


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
        model = MaskablePPO("MlpPolicy", self.env, verbose=1)
        model.learn(5000)

