import networkx as nx
from src.network.net import Flow, Network


class BaseScheduler:
    def __init__(self, network: Network, timeout_s: int = 300):
        self.graph = network.graph
        self.flows = network.flows
        self.links_dict = network.links_dict
        self.timeout_s = timeout_s

    def schedule(self) -> bool:
        """
        :return:
            True means successfully scheduled.
        """
        pass

    def get_num_gcl_max(self):
        pass
