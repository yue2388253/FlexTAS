import networkx as nx
from src.network.net import Flow, Link


class BaseScheduler:
    def __init__(self, graph: nx.Graph, flows: list[Flow], timeout_s: int = None):
        """TODO: replace the arguments of graph and flows with a function that construct a env"""
        self.graph = graph
        self.flows = flows
        self.links_dict: dict[str, Link] = {link_id: Link(link_id) for link_id in nx.line_graph(self.graph).nodes}
        self.timeout = timeout_s

    def schedule(self) -> bool:
        """
        :return:
            True means successfully scheduled.
        """
        pass
