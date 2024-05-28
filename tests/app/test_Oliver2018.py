import logging
import unittest
from src.network.net import generate_cev, generate_flows
from src.network.net import Net
from src.app.Oliver2018_scheduler import Oliver2018Scheduler


class TestOliver2018(unittest.TestCase):
    def test_another_schedule(self):
        logging.basicConfig(level=logging.DEBUG)  # Overwrite the log file on each run
        Net.GCL_LENGTH_MAX = 10
        graph = generate_cev()
        flows = generate_flows(graph,  5, period_set=[2000])
        scheduler = Oliver2018Scheduler(graph, flows)
        self.assertTrue(scheduler.schedule())

    def test_time_limit(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph,  10)
        scheduler = Oliver2018Scheduler(graph, flows, timeout_s=5)
        self.assertFalse(scheduler.schedule())
