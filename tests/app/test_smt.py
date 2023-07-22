import logging
import os
import unittest
from definitions import ROOT_DIR
from src.network.from_json import generate_net_flows_from_json
from src.app.smt_scheduler import SmtScheduler


class TestSmt(unittest.TestCase):
    def test_schedule(self):
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/smt_output.json'))
        scheduler = SmtScheduler(graph, flows)
        self.assertTrue(scheduler.schedule())

    def test_time_limit(self):
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/FlexTAS_CEV_100_100.json'))
        scheduler = SmtScheduler(graph, flows, timeout_s=5)
        self.assertFalse(scheduler.schedule())
