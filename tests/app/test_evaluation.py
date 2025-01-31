import logging
import os.path
import unittest

from definitions import ROOT_DIR
from src.app.evaluation import GCLTester, SchedulerTester, evaluate_experiments
from src.network.net import generate_cev, generate_flows, Network
from src.app.no_wait_tabu_scheduler import TimeTablingScheduler
from src.app.smt_scheduler import NoWaitSmtScheduler


class TestStressTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        graph = generate_cev()
        flows = generate_flows(graph, 10)
        self.network = Network(graph, flows)

    def test_gcl(self):
        tester = GCLTester(self.network)
        logging.info(tester.stress_test())

    def test_link(self):
        scheduler = TimeTablingScheduler(self.network)
        tester = SchedulerTester(self.network, scheduler)
        logging.info(tester.stress_test())

    def test_smt(self):
        scheduler = NoWaitSmtScheduler(self.network)
        tester = SchedulerTester(self.network, scheduler)
        logging.info(tester.stress_test())


class TestBatch(unittest.TestCase):
    def test_batch(self):
        logging.basicConfig(level=logging.DEBUG)
        df = evaluate_experiments(
            ["RRG", "ERG"],
            [10],
            100,
            5,
            ['gcl', 'all_gate', 'drl', 'smt'],
            os.path.join(ROOT_DIR, r'model/best_model.zip')
        )
        has_nan = df.isnull().any().any()
        # an easy scheduling task, all tests should pass
        # df.to_csv("tmp.csv")
        self.assertFalse(has_nan)
