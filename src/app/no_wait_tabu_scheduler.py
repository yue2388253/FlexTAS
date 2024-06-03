import math
from enum import Enum, auto
import random
import sys
from typing import Optional
from collections import defaultdict
import logging
import networkx as nx

from src.network.net import Net, Flow, Link, transform_line_graph
from src.app.scheduler import BaseScheduler
from src.lib.operation import Operation, check_operation_isolation


class TimeTablingScheduler(BaseScheduler):
    """
    schedule the flows in order
    """
    def __init__(self, graph: nx.DiGraph, flows: list[Flow], **kwargs):
        super().__init__(graph, flows, **kwargs)

        _, self.link_dict = transform_line_graph(graph)

        self.flows_operations = {}
        self.links_operations = defaultdict(list)

        self.makespan = 0
        self.critical_flow = None

    def schedule(self) -> bool:
        # time-tabling
        for flow in self.flows:
            ok = (self._try_schedule_flow(flow) and self._check_gcl_limit())
            if not ok:
                return False
        return True

    def _check_gcl_limit(self) -> bool:
        for link, flow_operations in self.links_operations.items():
            gcl_cycle = math.lcm(*[flow.period for flow, _ in flow_operations])
            num_entries = 0
            for flow, _ in flow_operations:
                # one for open and one for close
                num_entries += 2 * (gcl_cycle // flow.period)

                if num_entries > link.max_gcl_length:
                    # exceed the limit
                    return False

        # all good
        return True

    def _try_schedule_flow(self, flow) -> bool:
        """
        try to schedule flow at start_time, return True if success, else False
        """
        path = flow.path
        earliest_enqueue_time = 0
        latest_enqueue_time = 0

        operations = {}

        # construct operations
        for link_id in path:
            link = self.link_dict[link_id]
            trans_time = link.transmission_time(flow.payload)

            end_trans_time = latest_enqueue_time + trans_time

            operations[link] = Operation(
                earliest_enqueue_time,
                latest_enqueue_time,    # always enable gating right after the latest enqueue time.
                latest_enqueue_time,
                end_trans_time
            )

            if end_trans_time > self.makespan:
                self.makespan = end_trans_time
                self.critical_flow = flow

            earliest_enqueue_time = end_trans_time + Net.DELAY_PROP + Net.DELAY_PROC_MIN
            latest_enqueue_time = earliest_enqueue_time + Net.DELAY_PROC_JITTER

        # find the earliest possible operations
        while True:
            offset = self._check_operations(operations, flow)
            if offset is None:
                # successfully schedule, apply changes
                self.flows_operations[flow] = operations
                for link, operation in operations.items():
                    self.links_operations[link].append((flow, operation))

                break

            # add offset to each operation and try again.
            for operation in operations.values():
                operation.add(offset)
                if operation.end_time > flow.period:
                    # fail to schedule
                    return False

        return True

    def _check_operations(self, operations, flow) -> Optional[int]:
        for link, operation in operations.items():
            offset = self._check_valid_link(link, operation, flow)
            if offset:
                # there is a conflict, return immediately
                return offset
        return None

    def _check_valid_link(self, link: Link, operation: Operation, flow: Flow) -> Optional[int]:
        for flow_rhs, operation_rhs in self.links_operations[link]:
            offset = check_operation_isolation(
                (operation, flow.period),
                (operation_rhs, flow_rhs.period)
            )
            if offset is not None:
                return offset
        return None

    def get_res(self):
        return self.flows_operations

    def dump_res(self):
        for flow, operations in self.flows_operations.items():
            logging.debug(flow)
            i = 0
            for link, operation in operations.items():
                logging.debug(f"{' ' * i}link: {str(link).ljust(25)}{operation}")
                i += 1

    def get_makespan(self):
        return self.makespan

    def get_critical_flow(self):
        return self.critical_flow


def generate_neighbourhood(flows, critical_flow_idx):
    res = []
    critical_flow = flows[critical_flow_idx]
    num_flows = len(flows)

    # insertion
    tmp = flows.copy()
    del tmp[critical_flow_idx]
    for i in range(num_flows - 1):
        t = tmp.copy()
        t.insert(i, critical_flow)
        res.append(t)

    # swapping
    tmp = flows.copy()
    for i in range(num_flows):
        if i == critical_flow_idx:
            continue

        t = tmp.copy()
        # swap
        t[i], t[critical_flow_idx] = t[critical_flow_idx], t[i]
        res.append(t)

    return res


class NoWaitTabuScheduler(BaseScheduler):
    X = 5
    Y = 5

    class InitialHeuristic(Enum):
        SumAsc = auto()
        SumDec = auto()
        LongestAsc = auto()
        LongestDec = auto()
        Random = auto()

    def __init__(self, graph: nx.DiGraph, flows: list[Flow], **kwargs):
        super().__init__(graph, flows, **kwargs)
        self.best_scheduler = None

    def _generate_initial_sln(self, heur: InitialHeuristic):
        def sort_by_sum(flow):
            return flow.payload * len(flow.path)

        def sort_by_longest(flow):
            return flow.payload

        if heur == self.InitialHeuristic.SumAsc:
            return sorted(self.flows, key=sort_by_sum)
        elif heur == self.InitialHeuristic.SumDec:
            return sorted(self.flows, key=sort_by_sum, reverse=True)
        elif heur == self.InitialHeuristic.LongestAsc:
            return sorted(self.flows, key=sort_by_longest)
        elif heur == self.InitialHeuristic.LongestDec:
            return sorted(self.flows, key=sort_by_longest, reverse=True)
        elif heur == self.InitialHeuristic.Random:
            return random.sample(self.flows, len(self.flows))
        else:
            assert False

    def _sequencing_alg(self, flows):
        """
        Alg. 2 Sequencing Algorithm
        """
        best_makespan = sys.maxsize
        best_scheduler = None
        current_critical_flow = flows[-1]

        num_not_improve = 0
        tabu_list = {current_critical_flow}
        logging.info("Start iterate.")
        while num_not_improve < self.Y:
            # Termination criteria not satisfied

            nhd = generate_neighbourhood(flows, flows.index(current_critical_flow))
            list_slns = []
            for soln in nhd:
                scheduler = TimeTablingScheduler(self.graph, soln)
                ok = scheduler.schedule()
                if ok:
                    list_slns.append(scheduler)

            # solution selection
            # sort the scheduelr based on makespan first
            list_slns = sorted(list_slns, key=lambda x: x.get_makespan())
            # find the first one that does not violate the tabu list or
            # satisfies the aspiration criterion
            selected_soln = None
            for scheduler in list_slns:
                if (scheduler.get_critical_flow() not in tabu_list) or \
                        (scheduler.get_makespan() < best_makespan):
                    current_critical_flow = scheduler.get_critical_flow()
                    flows = scheduler.flows
                    # update tabu list
                    tabu_list.add(current_critical_flow)

                    selected_soln = scheduler
                    break

            if selected_soln is None:
                logging.info("No selectedSoln found.")
                break

            if selected_soln.get_makespan() < best_makespan:
                # yield an improvement
                best_scheduler = selected_soln
                best_makespan = selected_soln.get_makespan()
                logging.info(f"selectedSoln better than bestOrder (makespan: {best_scheduler.get_makespan()})")
                num_not_improve = 0
            else:
                num_not_improve += 1
                logging.info(f"selectedSoln worse than bestOrder (num not improved: {num_not_improve})")

        return best_scheduler

    def schedule(self):
        list_res = []
        for initial_heur in self.InitialHeuristic:
            logging.info("Heuristic: {}".format(initial_heur))
            flows = self._generate_initial_sln(initial_heur)
            logging.info("Sequencing alg...")
            scheduler = self._sequencing_alg(flows)
            list_res.append(scheduler)

        if all(v is None for v in list_res):
            logging.info("Fail to schedule")
            return False

        self.best_scheduler = min(list_res, key=lambda x: x.get_makespan())
        return True

    def dump_res(self):
        if self.best_scheduler is None:
            raise ValueError("No best scheduler found.")
        self.best_scheduler.dump_res()
