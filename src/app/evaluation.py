import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
import itertools
import logging
import math
import numpy as np
import os.path
import pandas as pd
import random
from typing import Optional, List

from definitions import OUT_DIR
from src.app.drl_scheduler import DrlScheduler
from src.app.no_wait_tabu_scheduler import TimeTablingScheduler, GatingStrategy
from src.app.scheduler import BaseScheduler, ResAnalyzer
from src.app.smt_scheduler import SmtScheduler
from src.lib.execute import execute_from_command_line
from src.lib.log_config import log_config
from src.network.net import generate_graph, Network, FlowGenerator


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
        start_time = time.time()

        try:
            ok = self.scheduler.schedule()
        except RuntimeError as e:
            logging.info(e)
            ok = False

        scheduling_time = time.time() - start_time
        res = {'scheduling_time': scheduling_time}

        if not ok:
            return res

        res |= ResAnalyzer(self.network, self.scheduler.get_res()).analyze()
        return res


@dataclass
class ExpSettings:
    topo: str
    num_flows: int
    link_rate: int
    jitters: List[float] = field(default_factory=lambda: [0.1])
    periods: List[int] = field(default_factory=lambda: [2000, 4000, 8000, 16000, 32000, 64000, 128000])
    timeout: int = 5
    num_non_tsn_devices: int = 0        # for heterogeneous network test
    test_gcl: bool = False              # test how many GCLs needed, ignoring the scheduling
    # the following flags involves the corresponding scheduler to schedule
    # and give an analysis of the schedule, e.g., GCLs needed, link_utilization, etc.
    test_all_gate: bool = False
    test_no_gate: bool = False
    test_random_gate: bool = False
    test_drl: Optional[str] = None
    test_smt: bool = False


def evaluate_single(settings: ExpSettings,
                    num_tests: int, seed: int):
    list_stats = []
    topo, num_flows, link_rate = settings.topo, settings.num_flows, settings.link_rate
    dict_settings = asdict(settings)
    for i in range(seed, seed+num_tests):
        graph = generate_graph(topo, link_rate)
        flow_generator = FlowGenerator(
            graph,
            seed=i,
            period_set=settings.periods,
            jitters=settings.jitters
        )
        flows = flow_generator(num_flows)
        network = Network(graph, flows)

        if settings.num_non_tsn_devices > 0:
            # heterogeneous network
            assert isinstance(settings.num_non_tsn_devices, int)
            network.disable_gcl(settings.num_non_tsn_devices)

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

        if settings.test_smt:
            scheduler = SmtScheduler(network, timeout_s=settings.timeout)
            list_schedulers.append(("smt", scheduler))

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


def evaluate_experiments(topos: List[str], list_num_flows: List[int],
                         link_rate: int, num_tests: int,
                         list_obj: List[str],
                         drl_model: str=None,
                         jitters: List[float]=None,
                         periods: List[int]=None,
                         num_non_tsn_devices: int=0,
                         to_csv: str=None,
                         seed: int=None,
                         timeout: int=None):
    """
    Args:
        list_obj: valid options: "gcl", "all_gate", "no_gate", "random_gate", "drl", "smt"
    """
    list_df = []
    test_gcl = "gcl" in list_obj
    test_all_gate = "all_gate" in list_obj
    test_no_gate = "no_gate" in list_obj
    test_random_gate = "random_gate" in list_obj

    test_drl = "drl" in list_obj
    if test_drl:
        assert drl_model is not None, "Should specify the drl model"
        assert os.path.isfile(drl_model), "Cannot find the drl model"
    else:
        drl_model = None

    test_smt = "smt" in list_obj

    list_settings = [
        ExpSettings(
            topo, num_flow, link_rate,
            num_non_tsn_devices=num_non_tsn_devices,
            test_gcl=test_gcl,
            test_all_gate=test_all_gate,
            test_no_gate=test_no_gate,
            test_random_gate=test_random_gate,
            test_drl=drl_model,
            test_smt=test_smt,
        )
        for topo, num_flow in itertools.product(topos, list_num_flows)
    ]

    if jitters is not None:
        for s in list_settings:
            s.jitters = jitters

    if periods is not None:
        for s in list_settings:
            s.periods = periods

    if timeout is not None:
        assert isinstance(timeout, int), "timeout must be an integer"
        for s in list_settings:
            s.timeout = timeout

    logging.info("Starting stress tests.")

    if seed is not None:
        assert isinstance(seed, int), "Seed must be an integer"
    else:
        seed = random.randint(1, 10000)

    df = None
    for i, settings in enumerate(list_settings):
        logging.info(f"Progress: {i/len(list_settings)*100:.2f}%\t{settings}")
        df = evaluate_single(settings, num_tests, seed)
        list_df.append(df)

        if to_csv:
            # save the results each time
            df = pd.concat(list_df, ignore_index=True)
            df.to_csv(to_csv)

    df = pd.concat(list_df, ignore_index=True)
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = os.path.join(OUT_DIR, 'test.log')
    print(f"Log file: {log_file}")
    log_config(log_file, level=logging.INFO)

    # run the tests
    df = execute_from_command_line(evaluate_experiments)

    filename = os.path.join(OUT_DIR, "stress_test.csv")
    df.to_csv(filename)
    logging.info(f"results saved to {filename}")
