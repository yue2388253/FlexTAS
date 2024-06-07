import logging
import unittest

from src.app.StressTester import GCLTester, LinkTester
from src.network.net import generate_cev, generate_flows


class TestStressTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.graph = generate_cev()
        self.flows = generate_flows(self.graph, 100)

    def test_gcl(self):
        tester = GCLTester(self.graph, self.flows)
        logging.info(tester.stress_test())

    def test_link(self):
        tester = LinkTester(self.graph, self.flows)
        logging.info(tester.stress_test())
