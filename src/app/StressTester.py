import os.path
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
import itertools
import logging
import math
import numpy as np
import pandas as pd

from definitions import OUT_DIR
from src.app.no_wait_tabu_scheduler import TimeTablingScheduler, GatingStrategy
from src.lib.execute import execute_from_command_line
from src.network.net import Flow, generate_graph, generate_flows, Network


class IStressTester:
    """
    An interface class for stress test.
    Note that this is not a scheduler!
    """

    def __init__(self, network: Network):
        self.graph = network.graph
        self.flows = network.flows
        self.link_dict = network.links_dict

    def stress_test(self) -> dict:
        pass


class GCLTester(IStressTester):
    @dataclass
    class GCLStat:
        num_links: int
        num_links_active: int
        num_flows_per_active_link: float
        num_flows_per_link_max: int
        gcl_max: int
        gcl_avg: float
        gcl_min: int
        num_hops_per_flow: float

    """
    A GCL Tester that tests how many GCLs are needed for the input.
    """
    def __init__(self, network: Network):
        super().__init__(network)

    def stress_test(self) -> dict:
        """
        Always return True. Only need to test how many GCLs needed.
        :return:
            True means successfully scheduled.
        """
        link_flows = defaultdict(list)
        for flow in self.flows:
            for link in flow.path:
                link_flows[link].append(flow)

        list_num_gcls = []
        for link, flows in link_flows.items():
            gcl_cycle = math.lcm(*[f.period for f in flows])
            expansion = [gcl_cycle // f.period for f in flows]
            gcl_length = sum(expansion) * 2
            list_num_gcls.append(gcl_length)

        list_num_gcls = np.array(list_num_gcls)

        num_hops_per_flow_avg = np.array([len(flow.path) for flow in self.flows]).mean()

        num_flows_per_active_link = np.array([len(v) for v in link_flows.values()])

        return asdict(self.GCLStat(
            len(self.link_dict),
            len(link_flows),
            num_flows_per_active_link.mean(),
            num_flows_per_active_link.max(),
            list_num_gcls.max(),
            list_num_gcls.mean(),
            list_num_gcls.min(),
            num_hops_per_flow_avg
        ))


class LinkTester(IStressTester):
    @dataclass
    class UtilizationStat:
        link_utilization_min: float
        link_utilization_max: float
        link_utilization_avg: float
        link_utilization_std: float

    """
    A stress tester that tests how much link_utilization can achieve.
    """

    def __init__(self, network: Network, gating_strategy: GatingStrategy=GatingStrategy.AllGate):
        super().__init__(network)
        self.scheduler = TimeTablingScheduler(network, gating_strategy)
        if gating_strategy == GatingStrategy.AllGate:
            # we only test link utilization, thus ignore the gcl limit
            for link in self.scheduler.links_dict.values():
                link.max_gcl_length = sys.maxsize

    def stress_test(self) -> dict:
        ok = self.scheduler.schedule()
        if not ok:
            return {}

        list_link_utilization = []
        links_operations = self.scheduler.links_operations
        for link in self.link_dict.values():
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


@dataclass
class StressTestSettings:
    topo: str
    test_gcl: bool
    test_uti: bool
    num_flows: int
    link_rate: int


def stress_test_single(settings: StressTestSettings,
                       num_tests: int):
    list_stats = []
    topo, num_flows, link_rate = settings.topo, settings.num_flows, settings.link_rate
    dict_settings = asdict(settings)
    for _ in range(num_tests):
        graph = generate_graph(topo, link_rate)
        flows = generate_flows(graph, num_flows)
        network = Network(graph, flows)

        stat = dict_settings.copy()
        if settings.test_gcl:
            tester = GCLTester(network)
            stat |= tester.stress_test()

        if settings.test_uti:
            tester = LinkTester(network)
            stat |= tester.stress_test()

            tester = LinkTester(network, GatingStrategy.NoGate)
            stat_no_gate = tester.stress_test()
            stat |= {f"{k}_no_gate": v for k, v in stat_no_gate.items()}

        assert len(stat) > 0
        list_stats.append(stat)

    # Create a DataFrame
    df = pd.DataFrame(list_stats)
    return df


def stress_test(topos: list[str], list_num_flows: list[int],
                link_rate: int, num_tests: int,
                list_obj: list[str]):
    list_df = []
    test_gcl = "gcl" in list_obj
    test_uti = "uti" in list_obj

    list_settings = [
        StressTestSettings(
            topo, test_gcl, test_uti, num_flow, link_rate
        )
        for topo, num_flow in itertools.product(topos, list_num_flows)
    ]

    logging.info("Starting stress tests.")
    for i, settings in enumerate(list_settings):
        logging.info(f"Progress: {i/len(list_settings)*100:.2f}%\t{settings}")
        df = stress_test_single(settings, num_tests)
        list_df.append(df)

    df = pd.concat(list_df, ignore_index=True)
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = execute_from_command_line(stress_test)
    filename = os.path.join(OUT_DIR, "stress_test.csv")
    df.to_csv(filename)
    print(f"results saved to {filename}")
