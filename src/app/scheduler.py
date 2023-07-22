import networkx as nx
from src.lib.config import ConfigManager
from src.network.net import Flow


class BaseScheduler:
    def __init__(self, graph: nx.DiGraph, flows: list[Flow], link_rate: int = None, timeout_s: int = 300):
        self.graph = graph
        self.flows = flows
        self.timeout_s = timeout_s

        if link_rate is None:
            self.link_rate = ConfigManager().config.getint('Net', 'link_rate')
        else:
            self.link_rate = int(link_rate)

    def schedule(self) -> bool:
        """
        :return:
            True means successfully scheduled.
        """
        pass
