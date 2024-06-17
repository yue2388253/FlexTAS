import unittest

from src.app.drl_scheduler import DrlScheduler
from src.network.net import *


class TestLimitedGcl(unittest.TestCase):
    def setUp(self):
        self.graph = generate_linear_5()

        flow1 = Flow(
            "f1", "E1", "E2",
            [("E1", "S1"), ("S1", "S2"), ("S2", "S1")],
            period=4000
        )
        flow2 = Flow(
            "f2", "E1", "E2",
            [("E1", "S1"), ("S1", "S2"), ("S2", "S1")],
            period=8000
        )
        self.flows = [flow1, flow2]

    def _test_single_cls(self, scheduler_cls, num_gcl_max):
        network = Network(self.graph, self.flows)
        network.set_gcl(num_gcl_max)
        scheduler = scheduler_cls(network)
        self.assertTrue(scheduler.schedule())

    def test_ok(self):
        self._test_single_cls(DrlScheduler, 6)

    def test_ok_again(self):
        # Drl should pass since it do not require all gate
        self._test_single_cls(DrlScheduler, 2)
