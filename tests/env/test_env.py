import logging
import os.path

from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
import unittest

from definitions import ROOT_DIR, OUT_DIR
from src.env.env import NetEnv
from src.network.net import generate_linear_5, Flow, Network


class TestEnvState(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.env = NetEnv()
        self.state_encoder = self.env.state_encoder

    def test_state(self):
        state = self.state_encoder.state()
        logging.debug(state)
        for key, value in state.items():
            logging.debug((key, value.shape))

        logging.debug(self.state_encoder.observation_space)
        self.assertTrue(self.state_encoder.observation_space.contains(state))


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.env = NetEnv()

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

    @unittest.skipIf(os.getenv('RUN_LONG_TESTS') != '1', "Skipping long running test")
    def test_action_masks(self):
        model = MaskablePPO("MultiInputPolicy", self.env, verbose=1)
        model.learn(5000)


class TestEnvInfo(unittest.TestCase):
    def setUp(self):
        self.graph = generate_linear_5()

        path = [("E1", "S1"), ("S1", "S2"), ("S2", "E2")]
        flow0 = Flow(f"F0", "E1", "E2", path, payload=64, period=2000, jitter=2000)
        flow1 = Flow(f"F1", "E1", "E2", path, payload=64, period=4000, jitter=2000)
        self.flows = [flow0, flow1]

    def test_success_all_gate(self):
        network = Network(self.graph, self.flows)
        env = NetEnv(network)
        for i in range(6):
            _, _, done, _, info = env.step(1)
            if i == 5:
                self.assertTrue(done)
                self.assertTrue(info['success'])
                env.save_results(os.path.join(OUT_DIR, "schedule_drl_l5_all_gate.txt"))
            else:
                self.assertFalse(done)
                self.assertFalse(info['success'])

    def test_success_no_gate(self):
        network = Network(self.graph, self.flows)
        env = NetEnv(network)
        for i in range(6):
            _, _, done, _, info = env.step(0)
            if i == 5:
                self.assertTrue(done)
                self.assertTrue(info['success'])
                env.save_results(os.path.join(OUT_DIR, "schedule_drl_l5_no_gate.txt"))
            else:
                self.assertFalse(done)
                self.assertFalse(info['success'])

    def test_limited_gcl(self):
        network = Network(self.graph, self.flows)
        for link in network.links_dict.values():
            link.gcl_capacity = 5
        env = NetEnv(network)
        for i in range(4):
            _, _, done, _, info = env.step(1)
            if i == 3:
                # gating is invalid since there is no gcl available.
                self.assertTrue(done)
            else:
                self.assertFalse(done)
