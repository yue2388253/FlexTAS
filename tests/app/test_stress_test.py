import logging
import unittest

from src.app.StressTester import GCLTester, SchedulerTester
from src.network.net import generate_cev, generate_flows, Network


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
        tester = SchedulerTester(self.network)
        logging.info(tester.stress_test())
