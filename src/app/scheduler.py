import networkx as nx
from src.network.net import Flow


class BaseScheduler:
    def __init__(self, graph: nx.DiGraph, flows: list[Flow], timeout_s: int = 300):
        self.graph = graph
        self.flows = flows
        self.timeout_s = timeout_s

    def schedule(self) -> bool:
        """
        :return:
            True means successfully scheduled.
        """
        pass

    def get_num_gcl_max(self):
        pass
