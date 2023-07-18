import logging
import os.path

from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
import unittest

from definitions import ROOT_DIR, OUT_DIR
from src.env.env import NetEnv
from src.network.from_json import generate_net_flows_from_json
from src.network.net import generate_linear_5, Flow


class TestEnvState(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(
            os.path.join(ROOT_DIR, 'data/input/smt_output/FlexTAS_50_1204.json'))
        self.env = NetEnv(graph, flows)
        self.state_encoder = self.env.state_encoder

    def test_edge_lists(self):
        edge_lists = self.state_encoder.edge_lists
        self.assertEqual(edge_lists.shape, (2, len(self.env.line_graph.edges)))

    def test_state(self):
        state = self.state_encoder.state()
        logging.debug(state)
        for key, value in state.items():
            logging.debug((key, value.shape))

        self.assertTrue(self.state_encoder.observation_space.contains(state))


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(
            os.path.join(ROOT_DIR, 'data/input/smt_output/FlexTAS_50_1204.json'))
        self.env = NetEnv(graph, flows)

    def test_check_env(self):
        check_env(self.env)

    def test_action(self):
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))

        gating_list = [0, 1, 0]
        for gating in gating_list:
            obs, reward, done, _, info = self.env.step(gating)
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertFalse(done)

        obs, reward, done, _, info = self.env.step(1)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertFalse(done)

    def test_action_masks(self):
        model = MaskablePPO("MultiInputPolicy", self.env, verbose=1)
        model.learn(5000)


class TestEnvInfo(unittest.TestCase):
    def test_success(self):
        graph = generate_linear_5()
        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flow = Flow(f"F0", "E1", "E2", path, payload=64, period=2000)
        flows = [flow]
        env = NetEnv(graph, flows)
        for i in range(3):
            _, _, done, _, info = env.step(1)
            if i == 2:
                self.assertTrue(done)
                self.assertTrue(info['success'])
                env.save_results(os.path.join(OUT_DIR, "schedule.txt"))
            else:
                self.assertFalse(done)
                self.assertFalse(info['success'])
