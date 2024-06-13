import logging
import math
import networkx as nx
import numpy as np
import random
import typing
from enum import Enum, auto

PERIOD_SET = [2000, 4000, 8000, 16000, 32000, 64000, 128000]


class Net:
    MTU = 1522
    JITTER_MAX = 1.0

    SYNC_PRECISION = 0

    # switch processing delay
    DELAY_PROC_MIN = 1
    DELAY_PROC_MAX = 6
    DELAY_PROC_JITTER = DELAY_PROC_MAX - DELAY_PROC_MIN

    # propagation delay
    DELAY_PROP = 1

    # This value should be set carefully. A large value like 128,000 would make the program slow.
    GCL_CYCLE_MAX = 128000
    GCL_LENGTH_MAX = 256
    BINARY_LENGTH_MAX = GCL_CYCLE_MAX
    PERIOD_MAX = GCL_CYCLE_MAX

    # num of st queues
    ST_QUEUES = 2


class Link:
    embedding_length = 3

    def __init__(self, link_id, link_rate):
        self.link_id = link_id
        self.gcl_capacity = Net.GCL_LENGTH_MAX
        self.link_rate = link_rate

    def __hash__(self):
        return hash(self.link_id)

    def __eq__(self, other):
        if isinstance(other, Link):
            return self.link_id == other.link_id
        return False

    def __repr__(self):
        return f"Link{self.link_id}"

    def interference_time(self) -> int:
        return self.transmission_time(Net.MTU)

    def transmission_time(self, payload: int) -> int:
        # 12 for interframe gap and 8 for preamble
        return math.ceil((payload + 12 + 8) * 8 / self.link_rate)

class Flow:
    def __init__(self, flow_id, src_id, dst_id, path,
                 period=2000, payload=100, e2e_delay=None, jitter=None):
        self.flow_id = flow_id
        self.src_id = src_id
        self.dst_id = dst_id
        self.path = path

        assert period in PERIOD_SET, f"Invalid period {period}"
        self.period = period
        self.payload = payload
        self.e2e_delay = period if e2e_delay is None else e2e_delay

        if jitter is None:
            self.jitter = int(0.1 * self.period)
        else:
            self.jitter = jitter
        assert type(self.jitter) is int, f"jitter ({self.jitter}) must be an integer."

    def __hash__(self):
        return hash(self.flow_id)

    def __eq__(self, other):
        if isinstance(other, Flow):
            return self.flow_id == other.flow_id
        return False

    def __str__(self):
        return f"Flow {self.flow_id} [{self.src_id}-->{self.dst_id}] with period {self.period}, payload {self.payload}B," \
               f" e2e delay {self.e2e_delay}us, jitter {self.jitter}us"

    def embedding(self) -> np.ndarray:
        return np.array([self.period / Net.PERIOD_MAX,
                         self.payload / Net.MTU,
                         self.e2e_delay / self.period,
                         self.jitter / self.period])

    def _wait_time_allowed(self, link_rate: int) -> int:
        assert isinstance(link_rate, int) and link_rate > 0
        num_hops = len(self.path)
        num_switches = num_hops - 1
        transmission_time = num_hops * math.ceil(self.payload * 8 / link_rate)
        return self.e2e_delay - transmission_time - num_hops * (Net.SYNC_PRECISION + Net.DELAY_PROP) \
            - num_switches * Net.DELAY_PROC_MAX


def generate_linear_5(link_rate=100) -> nx.DiGraph:
    graph = nx.DiGraph()

    # The id of nodes should be unique.
    graph.add_node('S1', node_type='SW')
    graph.add_node('S2', node_type='SW')
    graph.add_node('S3', node_type='SW')
    graph.add_node('S4', node_type='SW')
    graph.add_node('S5', node_type='SW')

    graph.add_node('E1', node_type='ES')
    graph.add_node('E2', node_type='ES')
    graph.add_node('E3', node_type='ES')
    graph.add_node('E4', node_type='ES')
    graph.add_node('E5', node_type='ES')

    for i in range(5):
        n0, n1 = f'S{i + 1}', f'E{i + 1}'
        graph.add_edge(n0, n1, link_id=f"{n0}-{n1}", link_rate=link_rate)
        graph.add_edge(n1, n0, link_id=f"{n1}-{n0}", link_rate=link_rate)

    for i in range(4):
        n0, n1 = f'S{i + 1}', f'S{i + 2}'
        graph.add_edge(n0, n1, link_id=f"{n0}-{n1}", link_rate=link_rate)
        graph.add_edge(n1, n0, link_id=f"{n1}-{n0}", link_rate=link_rate)

    return graph


def generate_cev(link_rate=100) -> nx.DiGraph:
    edges = [
        ("DU1", "SW11"),
        ("DU2", "SW11"),
        ("DU3", "SW11"),
        ("CMRIU1", "SW21"),
        ("FCM1", "SW31"),
        ("LCM1", "SW31"),
        ("RCM1", "SW31"),
        ("CM1CA", "SW41"),
        ("CM1CB", "SW41"),
        ("SM1CA", "SW51"),
        ("SM1CB", "SW51"),
        ("SMRIU1", "SW6"),
        ("SMRIU2", "SW6"),
        ("SM2CB", "SW52"),
        ("SM2CA", "SW52"),
        ("CM2CB", "SW42"),
        ("CM2CA", "SW42"),
        ("RCM2", "SW32"),
        ("LCM2", "SW32"),
        ("FCM2", "SW32"),
        ("CMRIU2", "SW22"),
        ("BFCU", "SW22"),
        ("DU5", "SW14"),
        ("DU4", "SW14"),
        ("StarTr2", "SW13"),
        ("StarTr1", "SW13"),
        ("MIMU3", "SW13"),
        ("MIMU2", "SW13"),
        ("MIMU1", "SW13"),
        ("SBAND2", "SW12"),
        ("SBAND1", "SW12"),

        # links between switches
        ("SW11", "SW21"),
        ("SW11", "SW22"),
        ("SW12", "SW21"),
        ("SW12", "SW22"),
        ("SW13", "SW21"),
        ("SW13", "SW22"),
        ("SW14", "SW21"),
        ("SW14", "SW22"),
        ("SW21", "SW31"),
        ("SW21", "SW31"),
        ("SW22", "SW32"),
        ("SW21", "SW7"),
        ("SW22", "SW7"),
        ("SW31", "SW7"),
        ("SW32", "SW7"),
        ("SW31", "SW41"),
        ("SW31", "SW8"),
        ("SW31", "SW6"),
        ("SW32", "SW8"),
        ("SW32", "SW6"),
        ("SW32", "SW42"),
        ("SW41", "SW51"),
        ("SW42", "SW52"),
        ("SW8", "SW51"),
        ("SW8", "SW52"),
    ]

    graph = nx.from_edgelist(edges)
    leaf_dict = {node: 'ES' if degree == 1 else 'SW' for node, degree in graph.degree()}
    nx.set_node_attributes(graph, leaf_dict, 'node_type')

    graph = nx.DiGraph(graph)

    for edge in graph.edges:
        graph.edges[edge]['link_rate'] = link_rate

    return graph


class RandomGraph(Enum):
    RRG = auto()
    ERG = auto()
    BAG = auto()


def _generate_graph(graph_type: RandomGraph, link_rate=100, **kwargs):
    graph = None
    while graph is None:
        if graph_type == RandomGraph.RRG:
            graph = nx.random_regular_graph(**kwargs)
        elif graph_type == RandomGraph.ERG:
            graph = nx.erdos_renyi_graph(**kwargs)
        elif graph_type == RandomGraph.BAG:
            graph = nx.barabasi_albert_graph(**kwargs)
        else:
            assert False
        if not nx.is_connected(graph):
            graph = None

    graph = nx.DiGraph(graph)

    # all nodes can send traffic
    nx.set_node_attributes(graph, 'ES', 'node_type')

    # set link rate
    nx.set_edge_attributes(graph, link_rate, 'link_rate')

    return graph


def generate_graph(topo: str, link_rate):
    if topo == "CEV":
        graph = generate_cev(link_rate)
    elif topo == "L5":
        graph = generate_linear_5(link_rate)
    elif topo == "RRG":
        graph = _generate_graph(RandomGraph.RRG, link_rate, d=4, n=20)
    elif topo == "ERG":
        graph = _generate_graph(RandomGraph.ERG, link_rate, n=20, p=0.25)
    elif topo == "BAG":
        graph = _generate_graph(RandomGraph.BAG, link_rate, n=20, m=3)
    elif topo == "Random":
        topo = random.choice(["RRG", "ERG", "BAG"])
        return generate_graph(topo, link_rate)
    else:
        raise ValueError(f"Unknown topo {topo}")

    return graph


def _transform_line_graph(graph) -> (nx.Graph, typing.Dict):
    line_graph = nx.line_graph(graph)
    links_dict = {}
    for node in line_graph.nodes:
        links_dict[node] = Link(node, graph.edges[node]['link_rate'])
    return line_graph, links_dict


class Network:
    def __init__(self, graph, flows) -> None:
        self.graph = graph
        self.flows = flows
        # construct links
        self.line_graph, self.links_dict = _transform_line_graph(graph)

    def disable_gcl(self, num_nodes: int):
        list_nodes = random.sample(list(self.graph.nodes), num_nodes)
        list_links = [
            [link for link in self.links_dict.values() if node == link.link_id[0]]
            for node in list_nodes
        ]

        # flatten the list
        list_links = [item for sublist in list_links for item in sublist]

        for link in list_links:
            link.gcl_capacity = 0

    def set_gcl(self, num_gcl: int):
        for link in self.links_dict.values():
            link.gcl_capacity = num_gcl


class FlowGenerator:
    def __init__(self, graph, seed:int=None, period_set=None, jitters=None) -> None:
        self.graph = graph

        if seed is not None:
            random.seed(seed)

        if period_set is None:
            period_set = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
        for period in period_set:
            assert isinstance(period, int) and period > 0
        self.period_set = period_set

        self.jitters = jitters

        # get the nodes whose node_type is 'ES'
        self.es_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'ES']
        self.num_generated_flows = 0

    def _generate_flow(self):
        # Select two random nodes from the es_nodes list
        random_nodes = random.sample(self.es_nodes, 2)
        src_id, dst_id = random_nodes[0], random_nodes[1]

        # calculate the shortest path
        path = nx.shortest_path(self.graph, src_id, dst_id)
        path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        period=random.choice(self.period_set)

        if self.jitters is not None:
            jitter_percentage = None
            if isinstance(self.jitters, float) or isinstance(self.jitters, int):
                jitter_percentage = self.jitters
            else:
                jitter_percentage = random.choice(self.jitters)
            assert 0 <= jitter_percentage
            jitter = math.ceil(jitter_percentage * period)
        else:
            jitter = None

        res = Flow(
            f"F{self.num_generated_flows}", src_id, dst_id, path, 
            payload=random.randint(64, 1518), 
            period=period, 
            jitter=jitter
        )

        self.num_generated_flows += 1
        return res

    def __call__(self, num_flows=1):
        return [self._generate_flow() for _ in range(num_flows)]


def generate_flows(graph, num_flows: int = 50, seed: int = None,
                   period_set=None, jitters=None) -> list[Flow]:
    generator = FlowGenerator(graph, seed, period_set, jitters)
    return generator(num_flows)
