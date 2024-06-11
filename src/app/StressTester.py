import time
from collections import defaultdict
from dataclasses import dataclass, asdict
import itertools
import logging
import math
import numpy as np
import os.path
import pandas as pd
import random
from typing import Optional

from definitions import OUT_DIR
from src.app.drl_scheduler import DrlScheduler
from src.app.no_wait_tabu_scheduler import TimeTablingScheduler, GatingStrategy
from src.app.scheduler import BaseScheduler, ResAnalyzer
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
    """
    A GCL Tester that tests how many GCLs are needed for the input.
    """

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


class SchedulerTester(IStressTester):
    """
    A tester that use a scheduler to schedule and then analyze the schedule res if success.
    """

    @dataclass
    class UtilizationStat:
        link_utilization_min: float
        link_utilization_max: float
        link_utilization_avg: float
        link_utilization_std: float

    def __init__(self, network: Network, scheduler: BaseScheduler):
        super().__init__(network)
        self.network = network
        self.scheduler = scheduler

    def stress_test(self) -> dict:
        try:
            start_time = time.time()
            ok = self.scheduler.schedule()
            scheduling_time = time.time() - start_time
        except RuntimeError as e:
            logging.info(e)
            ok = False

        if not ok:
            return {}

        res = ResAnalyzer(self.network, self.scheduler.get_res()).analyze()
        res['scheduling_time'] = scheduling_time
        return res


@dataclass
class StressTestSettings:
    topo: str
    num_flows: int
    link_rate: int
    timeout: int = 5
    test_gcl: bool = False              # test how many GCLs needed, ignoring the scheduling
    # the following flags involves the corresponding scheduler to schedule
    # and give an analysis of the schedule, e.g., GCLs needed, link_utilization, etc.
    test_all_gate: bool = False
    test_no_gate: bool = False
    test_random_gate: bool = False
    test_drl: Optional[str] = None


def stress_test_single(settings: StressTestSettings,
                       num_tests: int, seed: int):
    list_stats = []
    topo, num_flows, link_rate = settings.topo, settings.num_flows, settings.link_rate
    dict_settings = asdict(settings)
    for i in range(num_tests):
        graph = generate_graph(topo, link_rate)
        flows = generate_flows(graph, num_flows)
        network = Network(graph, flows)

        stat = dict_settings.copy()
        stat['test_id'] = seed + i

        if settings.test_gcl:
            tester = GCLTester(network)
            stat |= tester.stress_test()

        list_schedulers = []

        if settings.test_all_gate:
            scheduler = TimeTablingScheduler(network, GatingStrategy.AllGate, timeout_s=settings.timeout)
            list_schedulers.append(("all_gate", scheduler))

        if settings.test_no_gate:
            scheduler = TimeTablingScheduler(network, GatingStrategy.NoGate, timeout_s=settings.timeout)
            list_schedulers.append(("no_gate", scheduler))

        if settings.test_random_gate:
            scheduler = TimeTablingScheduler(network, GatingStrategy.RandomGate, timeout_s=settings.timeout)
            list_schedulers.append(("random_gate", scheduler))

        if settings.test_drl is not None:
            scheduler = DrlScheduler(network, timeout_s=settings.timeout)
            best_model_path = settings.test_drl
            assert os.path.isfile(best_model_path), "Cannot find the best model"
            scheduler.load_model(best_model_path, "MaskablePPO")
            list_schedulers.append(("drl", scheduler))

        for name, scheduler in list_schedulers:
            tester = SchedulerTester(network, scheduler)
            logging.info(f"Testing {name} scheduler... Settings: {settings}")
            res = tester.stress_test()
            stat |= {f"{k}_{name}": v for k, v in res.items()}

        assert len(stat) > 0
        list_stats.append(stat)

    # Create a DataFrame
    df = pd.DataFrame(list_stats)
    return df


def stress_test(topos: list[str], list_num_flows: list[int],
                link_rate: int, num_tests: int,
                list_obj: list[str],
                drl_model: str=None,
                seed=None):
    """
    Args:
        list_obj: valid options: "gcl", "uti", "drl"
    """
    list_df = []
    test_gcl = "gcl" in list_obj
    test_uti = "uti" in list_obj

    test_drl = "drl" in list_obj
    if test_drl:
        assert drl_model is not None, "Should specify the drl model"
        assert os.path.isfile(drl_model), "Cannot find the drl model"

    list_settings = [
        StressTestSettings(
            topo, num_flow, link_rate,
            test_gcl=test_gcl,
            test_all_gate=test_uti,
            test_no_gate=test_uti,
            test_random_gate=test_uti,
            test_drl=drl_model,
        )
        for topo, num_flow in itertools.product(topos, list_num_flows)
    ]

    logging.info("Starting stress tests.")

    if seed is not None:
        assert isinstance(seed, int), "Seed must be an integer"
    else:
        seed = random.randint(1, 10000)

    df = None
    for i, settings in enumerate(list_settings):
        logging.info(f"Progress: {i/len(list_settings)*100:.2f}%\t{settings}")
        df = stress_test_single(settings, num_tests, seed)
        list_df.append(df)

        # save the results each time
        df = pd.concat(list_df, ignore_index=True)

    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = execute_from_command_line(stress_test)
    filename = os.path.join(OUT_DIR, "stress_test.csv")
    df.to_csv(filename)
    print(f"results saved to {filename}")
