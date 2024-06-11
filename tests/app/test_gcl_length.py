import unittest

from src.app.no_wait_tabu_scheduler import TimeTablingScheduler
from src.app.drl_scheduler import DrlScheduler
from src.network.net import *


class TestTimeTablingScheduler(unittest.TestCase):
    def setUp(self):
        self.old_gcl = Net.GCL_LENGTH_MAX

    def tearDown(self):
        Net.GCL_LENGTH_MAX = self.old_gcl

    def _construct_scheduler(self):
        self.graph = generate_linear_5()

        flow1 = Flow(
            "f1", "E1", "E2",
            [("E1", "S1"), ("S1", "S2"), ("S2", "S1")],
            period=2000
        )
        flow2 = Flow(
            "f2", "E1", "E2",
            [("E1", "S1"), ("S1", "S2"), ("S2", "S1")],
            period=4000
        )
        self.flows = [flow1, flow2]

    def _test_single_cls(self, scheduler_cls, num_gcl_max, expected_res):
        Net.GCL_LENGTH_MAX = num_gcl_max
        self._construct_scheduler()
        network = Network(self.graph, self.flows)
        scheduler = scheduler_cls(network)

        if expected_res:
            # expected pass
            self.assertTrue(scheduler.schedule())
        else:
            with self.assertRaises(RuntimeError):
                scheduler.schedule()

    def test_gcl(self):
        list_scheduler_cls = [
            TimeTablingScheduler,
            DrlScheduler
        ]
        num_gcl_max = 6
        for scheduler_cls in list_scheduler_cls:
            self._test_single_cls(scheduler_cls, num_gcl_max, True)

        # Fail due to not enough gcl
        self._test_single_cls(TimeTablingScheduler, num_gcl_max - 1, False)

        # Drl should pass since it use FlexTAS
        self._test_single_cls(DrlScheduler, num_gcl_max - 1, True)
