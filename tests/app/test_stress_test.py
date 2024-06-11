import logging
import unittest

from src.app.StressTester import GCLTester, SchedulerTester, stress_test
from src.network.net import generate_cev, generate_flows, Network
from src.app.no_wait_tabu_scheduler import TimeTablingScheduler


class TestStressTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph, 100)
        self.network = Network(graph, flows)

    def test_gcl(self):
        tester = GCLTester(self.network)
        logging.info(tester.stress_test())

    def test_link(self):
        scheduler = TimeTablingScheduler(self.network)
        tester = SchedulerTester(self.network, scheduler)
        logging.info(tester.stress_test())


class TestBatch(unittest.TestCase):
    def test_batch(self):
        logging.basicConfig(level=logging.DEBUG)
        df = stress_test(["RRG", "ERG"], [10], 100, 5, ['gcl', 'uti'])
        has_nan = df.isnull().any().any()
        # an easy scheduling task, all tests should pass
        self.assertFalse(has_nan)
        # df.to_csv("tmp.csv")
