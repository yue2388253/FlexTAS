from copy import deepcopy
import os.path
import unittest

from definitions import ROOT_DIR, OUT_DIR
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


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.graph = generate_cev()
        self.flows = generate_flows(self.graph, 5)

    def test_node_links(self):
        network = Network(self.graph, self.flows)
        node = "SW11"
        links = [link for link in network.links_dict.values() if node == link.link_id[0]]

        # SW11 has 5 ports
        self.assertEqual(len(links), 5)

        node = "SW7"
        links = [link for link in network.links_dict.values() if node == link.link_id[0]]
        self.assertEqual(len(links), 4)

    def test_disable_gcl(self):
        network = Network(self.graph, self.flows)
        network.disable_gcl(2)
        self.assertTrue(any(link.gcl_capacity == 0 for link in network.links_dict.values()))

        # disable all nodes
        network.disable_gcl(46)
        self.assertTrue(all(link.gcl_capacity == 0 for link in network.links_dict.values()))

        # try to disable larger than number of nodes would raise exception
        with self.assertRaises(ValueError):
            network.disable_gcl(47)

    def test_deepcopy(self):
        network = Network(self.graph, self.flows)
        network_copy = deepcopy(network)
        network_copy.disable_gcl(2)
        # disable the copy class should not affect the original class.
        self.assertTrue(all(link.gcl_capacity != 0 for link in network.links_dict.values()))
