import logging
import os
import unittest
from definitions import ROOT_DIR
from src.network.from_json import generate_net_flows_from_json
from src.app.drl_scheduler import DrlScheduler


class TestDrl(unittest.TestCase):
    def test_drl(self):
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/smt_output.json'))
        scheduler = DrlScheduler(graph, flows, num_envs=4)
        scheduler.schedule()

    def test_time_limit(self):
        logging.basicConfig(level=logging.DEBUG)
        graph, flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/FlexTAS_CEV_100_100.json'))
        scheduler = DrlScheduler(graph, flows, 5)
        self.assertFalse(scheduler.schedule())
