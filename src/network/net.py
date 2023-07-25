import math
import networkx as nx
import numpy as np
import random
import typing


PERIOD_SET = [2000, 4000, 8000, 16000, 32000, 64000, 128000]


class Net:
    PAYLOAD_MAX = 1522
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


class Duration:
    PRECISION = 1000  # 1ms
    LENGTH_EMBEDDING = int(Net.GCL_CYCLE_MAX / PRECISION)

    def __init__(self, start, end, cycle):
        if start is None and end is None:
            # an empty duration.
            if cycle != Net.GCL_CYCLE_MAX:
                raise RuntimeError("Empty duration should have a cycle of GCL_CYCLE_MAX.")

            self._time_slot = np.zeros(cycle, dtype=np.uint8)
            return

        if start > end:
            raise RuntimeError(f"start is greater than end. ({start} > {end})")

        if end >= cycle:
            raise RuntimeError(f"end is greater than or equal to cycle. ({end} >= {cycle})")

        if Net.GCL_CYCLE_MAX % cycle != 0:
            raise RuntimeError(f"GCL_CYCLE_MAX ({Net.GCL_CYCLE_MAX}) is not divisible by cycle ({cycle}).")

        self.start = start
        self.end = end
        self.cycle = cycle

        _extend_times = Net.GCL_CYCLE_MAX // cycle

        uint8_arr = np.zeros(cycle, dtype=np.uint8)
        uint8_arr[start: end + 1] = 1

        self._time_slot = np.tile(uint8_arr, _extend_times)

        assert len(self._time_slot) == Net.GCL_CYCLE_MAX

    def is_conflict(self, other: 'Duration') -> bool:
        return np.any(np.bitwise_and(self._time_slot, other._time_slot) == 1)

    def add_duration(self, other: 'Duration'):
        self._time_slot = np.bitwise_or(self._time_slot, other._time_slot)

    def utilization(self):
        return np.sum(self._time_slot) / Net.GCL_CYCLE_MAX

    def embedding(self):
        reshaped_arr = self._time_slot.reshape(-1, self.PRECISION)
        percentage_arr = reshaped_arr.mean(axis=1).astype(np.float32)
        return percentage_arr


class Link:
    embedding_length = 3

    def __init__(self, link_id, link_rate):
        self.link_id = link_id
        self.gcl_cycle = 1
        self.gcl_length = 0
        self.max_gcl_length = Net.GCL_LENGTH_MAX
        self.link_rate = link_rate

        self.reserved_durations: list[Duration] = []
        self.reserved_binaries = Duration(None, None, Net.GCL_CYCLE_MAX)

    def __hash__(self):
        return hash(self.link_id)

    def __eq__(self, other):
        if isinstance(other, Link):
            return self.link_id == other.link_id
        return False

    def __repr__(self):
        return f"Link{self.link_id}"

    def reset(self):
        self.gcl_cycle = 1
        self.gcl_length = 0
        self.reserved_durations = []
        self.reserved_binaries = Duration(None, None, Net.GCL_CYCLE_MAX)

    def interference_time(self) -> int:
        return math.ceil((Net.PAYLOAD_MAX * 8 + 96 + 8 * 8) / self.link_rate)  # IFG + preamble + interference packet

    def transmission_time(self, payload: int) -> int:
        return math.ceil(payload * 8 / self.link_rate)

    def safe_distance(self) -> int:
        # inter-frame gap
        return self.transmission_time(12)

    def embedding(self) -> np.ndarray:
        return np.array([self.reserved_binaries.utilization(),
                         self.gcl_cycle / Net.GCL_CYCLE_MAX,
                         self.gcl_length / self.max_gcl_length])

    def add_reserved_duration(self, duration: Duration, check=True):
        if check and not self.check_isolation(duration):
            raise RuntimeError("Frame isolation conflict.")

        self.reserved_durations.append(duration)
        self.reserved_binaries.add_duration(duration)

    def check_isolation(self, duration: Duration):
        """
        :param duration: Duration.
        :return: True if not conflict.
        """
        return not self.reserved_binaries.is_conflict(duration)

    def add_gating(self, period: int, attempt=False) -> bool:
        """
        :return: True if can enable gating, otherwise depend on the value of attempt
            return False if attempt is True,
            otherwise raise RuntimeError
        """
        new_cycle = math.lcm(self.gcl_cycle, period)
        new_length = self.gcl_length * (new_cycle // self.gcl_cycle)
        new_length += ((new_cycle // period) * 2)
        if new_length > self.max_gcl_length:
            if attempt:
                return False
            else:
                raise RuntimeError("Gating constraint is not satisfied.")
        elif not attempt:
            self.gcl_cycle = new_cycle
            self.gcl_length = new_length
        return True


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
        assert self.jitter <= self.period, f"jitter ({self.jitter}) must be not greater than period ({self.period})."

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
                         self.payload / Net.PAYLOAD_MAX,
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


def transform_line_graph(graph) -> (nx.Graph, typing.Dict):
    line_graph = nx.line_graph(graph)
    links_dict = {}
    for node in line_graph.nodes:
        links_dict[node] = Link(node, graph.edges[node]['link_rate'])
    return line_graph, links_dict


def generate_flows(graph, num_flows: int = 50, seed: int = None) -> list[Flow]:
    if seed is not None:
        random.seed(seed)

    # get the nodes whose node_type is 'ES'
    es_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'ES']

    res = []
    period_set = [2000, 4000, 8000, 16000, 32000, 64000, 128000]

    for i in range(num_flows):
        # Select two random nodes from the es_nodes list
        random_nodes = random.sample(es_nodes, 2)
        src_id, dst_id = random_nodes[0], random_nodes[1]

        # calculate the shortest path
        path = nx.shortest_path(graph, src_id, dst_id)
        path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        res.append(
            Flow(f"F{i}", src_id, dst_id,
                 path, payload=random.randint(64, 1518), period=random.choice(period_set))
        )

    return res
