from collections import defaultdict
from dataclasses import dataclass, asdict
import logging
import math
import numpy as np
import os
from typing import List, Dict, Tuple

from definitions import OUT_DIR
from src.network.net import Flow, Link, Network, Net
from src.lib.operation import Operation, check_operation_isolation


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
        self._check_valid()
        filename = os.path.join(OUT_DIR, f'schedule_res_{id(self)}.log')
        self.dump_res(filename)

    def _check_valid(self):
        for operations in self.links_operations.values():
            num_operations = len(operations)
            for i in range(num_operations - 1):
                flow_0, operation_0 = operations[i]
                for j in range(i+1, num_operations):
                    flow_1, operation_1 = operations[j]
                    assert (check_operation_isolation((operation_0, flow_0.period),
                                                      (operation_1, flow_1.period))
                            is None)

    def dump_res(self, filename):
        flows_operations = defaultdict(lambda: defaultdict(Operation))
        for link, operations in self.links_operations.items():
            for flow, operation in operations:
                flows_operations[flow][link.link_id] = operation

        res = []
        for flow, operations in flows_operations.items():
            path = flow.path
            s = str(len(res)) + '. ' + str(flow) + '\n'
            for link_id in path:
                assert link_id in operations
                operation = operations[link_id]
                s += f"\t{link_id}, {operation}\n"
            res.append(s)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.writelines(res)
            logging.info(f"Schedule result dump to {filename}")

    def analyze(self) -> Dict:
        return self._analyze_link_uti() | self._analyze_gcl() | self._analyze_flows()

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
            operations = [fo for fo in operations if fo[1].gating_time is not None]

            if len(operations) == 0:
                gcl_length = 0
            else:
                gcl_cycle = math.lcm(*[f.period for f, _ in operations])
                expansion = np.array([gcl_cycle // f.period for f, _ in operations])
                gcl_length = sum(expansion) * 2
            list_gcl.append(gcl_length)

        list_gcl = np.array(list_gcl)

        return asdict(self.GCLStat(
            list_gcl.min(),
            list_gcl.max(),
            list_gcl.mean(),
            list_gcl.std()
        ))

    @dataclass
    class FlowStat:
        e2e_delay_min: int
        e2e_delay_max: int
        e2e_delay_avg: float
        e2e_delay_std: float
        jitter_min: int
        jitter_max: int
        jitter_avg: float
        jitter_std: float
        jitter_ratio_min: float
        jitter_ratio_max: float
        jitter_ratio_avg: float
        jitter_ratio_std: float

    def _analyze_flows(self):
        def _get_operation(_l, _f):
            return next(o for f, o in self.links_operations[_l] if f == _f)

        list_e2e_delay = []
        list_jitter = []
        list_jitter_req = []

        for flow in self.network.flows:
            path = flow.path
            first_link = path[0]
            last_link = path[-1]
            first_link = self.network.links_dict[first_link]
            last_link = self.network.links_dict[last_link]
            first_link_oper = _get_operation(first_link, flow)
            last_link_oper = _get_operation(last_link, flow)
            last_link_t4_max = last_link_oper.latest_time

            e2e_delay = last_link_t4_max - first_link_oper.start_time + Net.DELAY_PROP + last_link.transmission_time(flow.payload)
            list_e2e_delay.append(e2e_delay)

            last_link_t4_min = last_link_oper.start_time
            jitter = last_link_t4_max - last_link_t4_min if last_link_oper.gating_time is None else 0
            list_jitter.append(jitter)

            jitter_req = flow.jitter
            list_jitter_req.append(jitter_req)

        list_e2e_delay = np.array(list_e2e_delay)
        list_jitter = np.array(list_jitter)
        list_jitter_req = np.array(list_jitter_req)

        # set ratio to 0 if the divisor (jitter_req) is 0
        list_jitter_ratio = np.zeros_like(list_jitter, dtype=float)
        np.divide(list_jitter, list_jitter_req, out=list_jitter_ratio, where=list_jitter_req!= 0)

        return asdict(self.FlowStat(
            list_e2e_delay.min(),
            list_e2e_delay.max(),
            list_e2e_delay.mean(),
            list_e2e_delay.std(),
            list_jitter.min(),
            list_jitter.max(),
            list_jitter.mean(),
            list_jitter.max(),
            list_jitter_ratio.min(),
            list_jitter_ratio.max(),
            list_jitter_ratio.mean(),
            list_jitter_ratio.std()
        ))
