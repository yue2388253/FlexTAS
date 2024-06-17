import logging
import unittest
from src.network.net import generate_cev, generate_flows, Network
from src.app.drl_scheduler import DrlScheduler
from src.app.scheduler import ResAnalyzer


class TestDrl(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.graph = generate_cev()

    def test_drl(self):
        flows = generate_flows(self.graph, 5)
        network = Network(self.graph, flows)
        scheduler = DrlScheduler(network, num_envs=1)
        self.assertTrue(scheduler.schedule())

    def test_heterogeneous_true(self):
        flows = generate_flows(self.graph, 5, jitters=1)
        network = Network(self.graph, flows)

        # disable all nodes
        network.disable_gcl(46)

        scheduler = DrlScheduler(network, num_envs=1)
        self.assertTrue(scheduler.schedule())
        res_analyzer = ResAnalyzer(network, scheduler.get_res())
        analyze_res = res_analyzer.analyze()

        # gcl of all nodes should be zero
        self.assertEqual(analyze_res['gcl_avg'], 0)
        self.assertEqual(analyze_res['gcl_max'], 0)

    def test_heterogeneous_false(self):
        flows = generate_flows(self.graph, 5, period_set=[2000], jitters=0.01)
        network = Network(self.graph, flows)

        # disable all nodes
        network.disable_gcl(20)

        scheduler = DrlScheduler(network, num_envs=1, timeout_s=5)
        # should fail since jitter constraints cannot be satisfied.
        self.assertFalse(scheduler.schedule())

    def test_drl_multi_envs(self):
        flows = generate_flows(self.graph, 5)
        network = Network(self.graph, flows)
        scheduler = DrlScheduler(network, num_envs=4)
        self.assertTrue(scheduler.schedule())
