import logging
import os
import unittest
from src.network.net import generate_cev, generate_flows
from src.network.net import Net, Network
from src.app.Oliver2018_scheduler import Oliver2018Scheduler


class TestOliver2018(unittest.TestCase):
    def setUp(self):
        self.old_gcl = Net.GCL_LENGTH_MAX

    def tearDown(self):
        Net.GCL_LENGTH_MAX = self.old_gcl

    def test_another_schedule(self):
        logging.basicConfig(level=logging.DEBUG)  # Overwrite the log file on each run
        Net.GCL_LENGTH_MAX = 10
        graph = generate_cev()
        flows = generate_flows(graph,  5, period_set=[2000])
        network = Network(graph, flows)
        scheduler = Oliver2018Scheduler(network)
        self.assertTrue(scheduler.schedule())

    @unittest.skipIf(os.getenv('RUN_LONG_TESTS') != '1', "Skipping long running test")
    def test_time_limit(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph,  10)
        network = Network(graph, flows)
        scheduler = Oliver2018Scheduler(network, timeout_s=5)
        self.assertFalse(scheduler.schedule())
