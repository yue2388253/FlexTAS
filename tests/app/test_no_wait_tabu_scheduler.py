import unittest

from src.app.no_wait_tabu_scheduler import *
from src.network.net import generate_cev, generate_flows, generate_linear_5


class TestTimeTablingScheduler(unittest.TestCase):
    def setUp(self):
        self.old_gcl = Net.GCL_LENGTH_MAX

    def tearDown(self):
        Net.GCL_LENGTH_MAX = self.old_gcl

    def _construct_scheduler(self):
        graph = generate_linear_5()
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
        self.scheduler = TimeTablingScheduler(graph, [flow1, flow2])

    def test_gcl_limit_true(self):
        self._construct_scheduler()
        self.assertTrue(self.scheduler.schedule())

        num_link = 18
        self.assertEqual(len(self.scheduler.get_gcl_length()), num_link)

        num_gcl_max = 6
        self.assertEqual(max(self.scheduler.get_gcl_length()), num_gcl_max)

        num_gcl_avg = 1
        self.assertEqual(self.scheduler.get_gcl_length().mean(), num_gcl_avg)

    def test_gcl_limit_false(self):
        Net.GCL_LENGTH_MAX = 5
        self._construct_scheduler()
        self.assertFalse(self.scheduler.schedule())

    def test_schedule(self):
        logging.basicConfig(level=logging.DEBUG)
        Net.GCL_LENGTH_MAX = sys.maxsize
        graph = generate_cev()
        flows = generate_flows(graph,  100)
        scheduler = TimeTablingScheduler(graph, flows)
        self.assertTrue(scheduler.schedule())
        scheduler.dump_res()


class TestNoWaitTabuScheduler(unittest.TestCase):
    def test_generate_neighbourhood(self):
        flows = [1, 3, 2]
        critial_flow_idx = 1
        nbh = generate_neighbourhood(flows, critial_flow_idx)
        self.assertEqual(len(nbh), 4)
        self.assertEqual(nbh[0], [3, 1, 2])
        self.assertEqual(nbh[1], [1, 3, 2])
        self.assertEqual(nbh[2], [3, 1, 2])
        self.assertEqual(nbh[3], [1, 2, 3])

    def test_schedule(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph,  10)
        scheduler = NoWaitTabuScheduler(graph, flows)
        self.assertTrue(scheduler.schedule())
        scheduler.dump_res()
