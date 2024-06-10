import logging
import os
import unittest
from src.network.net import generate_cev, generate_flows
from src.app.smt_scheduler import SmtScheduler


class TestSmt(unittest.TestCase):
    def test_schedule(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph,  10)
        scheduler = SmtScheduler(graph, flows)
        self.assertTrue(scheduler.schedule())

    @unittest.skipIf(os.getenv('RUN_LONG_TESTS') != '1', "Skipping long running test")
    def test_time_limit(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph,  100)
        scheduler = SmtScheduler(graph, flows, timeout_s=5)
        self.assertFalse(scheduler.schedule())
