import unittest

from src.app.no_wait_tabu_scheduler import *
from src.network.net import generate_cev, generate_flows, generate_linear_5


class TestTimeTablingScheduler(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.old_gcl_limit = Net.GCL_LENGTH_MAX
        Net.GCL_LENGTH_MAX = sys.maxsize
        self.graph = generate_cev()
        self.flows = generate_flows(self.graph,  10)

    def tearDown(self):
        Net.GCL_LENGTH_MAX = self.old_gcl_limit

    def test_schedule(self):
        scheduler = TimeTablingScheduler(self.graph, self.flows)
        self.assertTrue(scheduler.schedule())
        scheduler.dump_res()

    def test_no_gate(self):
        scheduler = TimeTablingScheduler(self.graph, self.flows, GatingStrategy.NoGate)
        self.assertTrue(scheduler.schedule())
        scheduler.dump_res()

    def test_random_gating(self):
        scheduler = TimeTablingScheduler(self.graph, self.flows, GatingStrategy.RandomGate)
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
