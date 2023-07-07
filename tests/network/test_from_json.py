import os.path
import unittest
from definitions import ROOT_DIR
from src.network.from_json import *


class TestJson(unittest.TestCase):
    def test_from_json(self):
        filename = os.path.join(ROOT_DIR, 'data/input/smt_output.json')
        graph, flows = generate_net_flows_from_json(filename)

        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(len(graph.nodes), 10)

        line_graph = nx.line_graph(graph)
        self.assertEqual(len(line_graph.nodes), 18)

        for flow in flows:
            self.assertIsInstance(flow, Flow)
        self.assertEqual(len(flows), 20)
