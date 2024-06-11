import logging
import unittest
from src.network.net import generate_cev, generate_flows, Network
from src.app.drl_scheduler import DrlScheduler


class TestDrl(unittest.TestCase):
    def test_drl(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph, 5)
        network = Network(graph, flows)
        scheduler = DrlScheduler(network, num_envs=1)
        scheduler.schedule()

    def test_time_limit(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph, 150)
        network = Network(graph, flows)
        scheduler = DrlScheduler(network, timeout_s=5)
        self.assertFalse(scheduler.schedule())
