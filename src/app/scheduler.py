from dataclasses import dataclass, asdict
import math
import numpy as np
from typing import List, Dict, Tuple

from src.network.net import Flow, Link, Network
from src.lib.operation import Operation


LinkOperations = List[Tuple[Flow, Operation]]
ScheduleRes = Dict[Link, LinkOperations]


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

    def get_res(self) -> ScheduleRes:
        pass


class ResAnalyzer:
    def __init__(self, network: Network, links_operations: ScheduleRes):
        self.graph = network.graph
        self.flows = network.flows
        self.network = network
        self.links_operations = links_operations

    def analyze(self) -> Dict:
        return self._analyze_link_uti() | self._analyze_gcl()

    @dataclass
    class UtilizationStat:
        link_utilization_min: float
        link_utilization_max: float
        link_utilization_avg: float
        link_utilization_std: float

    def _analyze_link_uti(self) -> Dict:
        list_link_utilization = []
        links_operations = self.links_operations
        for link in self.network.links_dict.values():
            if link not in links_operations:
                list_link_utilization.append(0)
                continue

            operations = links_operations[link]
            if len(operations) == 0:
                link_utilization = 0
            else:
                gcl_cycle = math.lcm(*[f.period for f, _ in operations])
                expansion = np.array([gcl_cycle // f.period for f, _ in operations])
                trans_time = np.array([
                    operation.end_time - operation.start_time
                    for _, operation in operations
                ])
                link_utilization = np.dot(expansion, trans_time) / gcl_cycle

            assert 0 <= link_utilization <= 1
            list_link_utilization.append(link_utilization)

        list_link_utilization = np.array(list_link_utilization)

        return asdict(self.UtilizationStat(
            list_link_utilization.min(),
            list_link_utilization.max(),
            list_link_utilization.mean(),
            list_link_utilization.std()
        ))

    @dataclass
    class GCLStat:
        gcl_min: int
        gcl_max: int
        gcl_avg: float
        gcl_std: float

    def _analyze_gcl(self) -> Dict:
        list_gcl = []
        links_operations = self.links_operations
        for link in self.network.links_dict.values():
            if link not in links_operations:
                list_gcl.append(0)
                continue

            operations = links_operations[link]
            if len(operations) == 0:
                gcl_length = 0
            else:
                gcl_cycle = math.lcm(*[f.period for f, _ in operations])
                expansion = np.array([gcl_cycle // f.period for f, _ in operations])
                gcl_length = sum(expansion) * 2
            assert 0 <= gcl_length <= link.gcl_capacity
            list_gcl.append(gcl_length)

        list_gcl = np.array(list_gcl)

        return asdict(self.GCLStat(
            list_gcl.min(),
            list_gcl.max(),
            list_gcl.mean(),
            list_gcl.std()
        ))
