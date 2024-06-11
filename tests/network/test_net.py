import json
import jsons
import os.path
import unittest

from definitions import ROOT_DIR, OUT_DIR
from src.network.from_json import generate_net_flows_from_json
from src.network.net import *


class TestCEV(unittest.TestCase):
    def test_cev(self):
        cev = generate_cev()

        # # user should check the graph manually.
        # from pyvis.network import Network
        # net = Network(notebook=True)
        # net.from_nx(cev)
        # net.show(os.path.join(OUT_DIR, 'cev.html'))

        self.assertEqual(len(cev.nodes), 46)
        self.assertEqual(len(cev.edges), 2 * (31 + 24))


class TestRandomGraph(unittest.TestCase):
    def test_random_graph(self):
        num_test = 10
        num_nodes = 20
        for i in range(num_test):
            d = 4
            graph = generate_graph("RRG", 100)
            self.assertIsNotNone(graph)
            self.assertTrue(len(graph.nodes) == num_nodes)
            self.assertEqual(len(graph.edges), num_nodes * d)

        for i in range(num_test):
            graph = generate_graph("ERG", 100)
            self.assertIsNotNone(graph)
            self.assertTrue(len(graph.nodes) == num_nodes)

        for i in range(num_test):
            graph = generate_graph("BAG", 100)
            self.assertIsNotNone(graph)
            self.assertTrue(len(graph.nodes) == num_nodes)
            self.assertEqual(len(graph.edges), (num_nodes - 3) * 3 * 2)


class TestDuration(unittest.TestCase):
    def test_conflict(self):
        d1 = Duration(0, 0, 4)
        d2 = Duration(2, 2, 4)
        d3 = Duration(1, 1, 2)
        self.assertFalse(d1.is_conflict(d2))
        self.assertFalse(d1.is_conflict(d3))
        self.assertFalse(d2.is_conflict(d3))

        d4 = Duration(0, 0, 5)
        self.assertTrue(d1.is_conflict(d4))
        self.assertTrue(d2.is_conflict(d4))
        self.assertTrue(d3.is_conflict(d4))


class TestLink(unittest.TestCase):
    def test_conflict(self):
        d1 = Duration(0, 0, 4)
        d2 = Duration(2, 2, 4)
        d3 = Duration(1, 1, 2)
        link = Link('link0', 100)
        link.add_reserved_duration(d1, check=True)
        link.add_reserved_duration(d2, check=True)
        link.add_reserved_duration(d3, check=True)

        link = Link('link1', 100)
        d4 = Duration(0, 0, 5)
        link.add_reserved_duration(d4, check=True)
        with self.assertRaises(RuntimeError) as _:
            link.add_reserved_duration(d1, check=True)
        with self.assertRaises(RuntimeError) as _:
            link.add_reserved_duration(d2, check=True)
        with self.assertRaises(RuntimeError) as _:
            link.add_reserved_duration(d3, check=True)

    def test_embedding(self):
        link = Link('link', 100)
        link.add_reserved_duration(Duration(0, 0, 4))
        link.add_reserved_duration(Duration(1, 1, 8))
        embedding = link.embedding()
        self.assertTrue(np.all((embedding >= 0) & (embedding <= 1)))


class TestFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.G, self.flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/smt_output.json'))

    def test_flow_wait_time_budget(self):
        flow = self.flows[0]
        self.assertEqual(flow._wait_time_allowed(100), 1643)

    def test_path(self):
        print(self.flows[0].path)

    def test_export_and_import(self):
        filename = os.path.join(OUT_DIR, 'flows.json')
        with open(filename, 'w') as f:
            f.write(jsons.dumps(self.flows, {"indent": 4}))

        with open(filename, 'r') as f:
            flows_json = f.read()

        # Parse the JSON string to a Python object.
        flows_list = json.loads(flows_json)

        self.import_flows = jsons.load(flows_list, list[Flow])
        print([str(flow) for flow in self.import_flows])


class TestFlowGenerator(unittest.TestCase):
    def test_random_seed(self):
        graph = generate_linear_5()
        num_flows = 10
        flows1 = generate_flows(graph, num_flows, 1)

        # same seed, should be all the same
        flows2 = generate_flows(graph, num_flows, 1)
        self.assertTrue(all([str(flows1[i] == str(flows2[i]) for i in range(num_flows))]))

        # different seed, there must be at least a difference.
        flows3 = generate_flows(graph, num_flows, 2)
        self.assertTrue(any([str(flows1[i] != str(flows3[i]) for i in range(num_flows))]))

    def test_flow_jitter(self):
        graph = generate_linear_5()
        num_flows = 100

        flow_generator = FlowGenerator(graph, jitters=0, period_set=[2000])
        flows = flow_generator(num_flows)
        [self.assertEqual(flow.jitter, 0) for flow in flows]

        flow_generator = FlowGenerator(graph, jitters=0.1, period_set=[2000])
        flows = flow_generator(num_flows)
        [self.assertEqual(flow.jitter, 200) for flow in flows]

        flow_generator = FlowGenerator(graph, jitters=[0.1, 0.2], period_set=[2000])
        flows = flow_generator(num_flows)
        [self.assertIn(flow.jitter, [200, 400]) for flow in flows]


    def test_random_graph(self):
        graph = generate_graph("RRG", 100)
        num_flows = 10
        flows = generate_flows(graph, num_flows)
        self.assertEqual(len(flows), num_flows)


if __name__ == '__main__':
    unittest.main()
