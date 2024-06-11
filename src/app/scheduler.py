from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import List, Dict, Tuple

from src.network.net import Flow, Link, Network
from src.lib.operation import Operation


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

    def get_res(self):
        pass

    def get_num_gcl_max(self):
        pass


LinkOperations = List[Tuple[Flow, Operation]]
ScheduleRes = Dict[Link, LinkOperations]


class ResAnalyzer:
    def __init__(self, network: Network, links_operations: ScheduleRes):
        self.graph = network.graph
        self.flows = network.flows
        self.network = network
        self.links_operations = links_operations

    @dataclass
    class UtilizationStat:
        link_utilization_min: float
        link_utilization_max: float
        link_utilization_avg: float
        link_utilization_std: float

    def analyze_link_utilization(self):
        list_link_utilization = []
        links_operations = self.links_operations
        for link in self.network.links_dict.values():
            if link not in links_operations:
                list_link_utilization.append(0)
                continue

            operations = links_operations[link]
            gcl_cycle = math.lcm(*[f.period for f, _ in operations])
            expansion = np.array([gcl_cycle // f.period for f, _ in operations])
            trans_time = np.array([
                operation.end_time - operation.start_time
                for _, operation in operations
            ])
            link_utilization = np.dot(expansion, trans_time) / gcl_cycle
            assert link_utilization <= 1
            assert link_utilization != 0
            list_link_utilization.append(link_utilization)

        list_link_utilization = np.array(list_link_utilization)

        return asdict(self.UtilizationStat(
            list_link_utilization.min(),
            list_link_utilization.max(),
            list_link_utilization.mean(),
            list_link_utilization.std()
        ))
