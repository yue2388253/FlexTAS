from collections import defaultdict
import itertools
import logging
import os
import pandas as pd
import random
import re
import time

from definitions import OUT_DIR
from src.app.smt_scheduler import SmtScheduler, NoWaitSmtScheduler
from src.app.drl_scheduler import DrlScheduler
from src.lib.log_config import log_config
from src.lib.execute import execute_from_command_line
from src.network.net import generate_linear_5, generate_cev, generate_flows


class SchedulerManager:
    def __init__(self, topo, best_model, time_limit, num_flows, seed, num_tests, link_rate):
        # [scheduler, [seed, # of flows]]
        self.scheduler_capacity = defaultdict(lambda: defaultdict(None))

        self.topo = topo
        self.best_model = best_model
        self.time_limit = time_limit
        self.num_flows = sorted(num_flows)
        self.seed = seed
        self.num_tests = num_tests
        self.link_rate = link_rate

    def can_schedule(self, scheduler_str, num_flows, seed):
        max_flows = self.scheduler_capacity[scheduler_str][seed]
        return max_flows is None or num_flows <= max_flows

    def update_capacity(self, scheduler_str, num_flows, seed, success):
        max_flows = self.scheduler_capacity[scheduler_str][seed]
        if success:
            if max_flows is None or num_flows > max_flows:
                self.scheduler_capacity[scheduler_str][seed] = num_flows
        elif max_flows is None or num_flows < max_flows:
            self.scheduler_capacity[scheduler_str][seed] = num_flows - 1

    def run_dichotomy(self) -> pd.DataFrame:
        schedulers = [
            ('smt', SmtScheduler),
            ('smt_no_wait', NoWaitSmtScheduler),
            ('drl', DrlScheduler)
        ]
        # [scheduler, [seed, [num_flows, (is_scheduled, time)]]]
        res_dict = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: None)
            )
        )

        for scheduler_str, scheduler_cls in schedulers:
            for seed in range(self.seed, self.seed+self.num_tests):
                low = 0
                high = len(self.num_flows) - 1
                while low <= high:
                    mid = (low + high) // 2

                    num_flows = self.num_flows[mid]
                    graph = get_graph(self.topo)
                    flows = generate_flows(graph, num_flows, seed=seed)

                    scheduler = scheduler_cls(graph, flows, timeout_s=self.time_limit, link_rate=self.link_rate)
                    if isinstance(scheduler, DrlScheduler):
                        scheduler.load_model(self.best_model, 'MaskablePPO')

                    start_time = time.time()
                    logging.info(f"Use [{scheduler_str}] Scheduling {num_flows} flows, seed {seed}")
                    is_scheduled = scheduler.schedule()

                    elapsed_time = time.time() - start_time

                    res_dict[scheduler_str][seed][num_flows] = is_scheduled, elapsed_time

                    if is_scheduled:
                        for i in range(low, mid):
                            res_dict[scheduler_str][seed][self.num_flows[i]] = (True, 0)
                        low = mid + 1
                    else:
                        for i in range(mid+1, high+1):
                            res_dict[scheduler_str][seed][self.num_flows[i]] = (False, self.time_limit)
                        high = mid - 1

        # convert res_dict to df
        column_names = ['topo', 'num_flows', 'seed',
                        'smt_scheduled', 'smt_time',
                        'smt_scheduled_no_wait', 'smt_time_no_wait',
                        'drl_scheduled', 'drl_time'
                        ]

        res = []
        for num_flows, seed in itertools.product(self.num_flows, range(self.seed, self.seed+self.num_tests)):
            res.append(
                [self.topo, num_flows, seed,
                 *res_dict['smt'][seed][num_flows],
                 *res_dict['smt_no_wait'][seed][num_flows],
                 *res_dict['drl'][seed][num_flows],
                 ]
            )

        return pd.DataFrame(columns=column_names, data=res)


def schedule(scheduler):
    start_time = time.time()
    is_scheduled = scheduler.schedule()
    elapsed_time = time.time() - start_time
    logging.debug(f"result: {'success' if is_scheduled else 'failure'}")
    return is_scheduled, elapsed_time


def get_graph(topo):
    if topo == 'L5':
        return generate_linear_5()
    elif topo == 'CEV':
        return generate_cev()
    raise ValueError(f"Unknown graph type {topo}")


def main(num_flows: str, num_tests: int, best_model: str, seed: int = None,
         link_rate: int = None, time_limit: int = 300, topo: str = 'L5'):
    seed = seed or random.randint(0, 10000)
    num_flows = map(int, num_flows.split(',')) if isinstance(num_flows, str) else num_flows

    scheduler_manager = SchedulerManager(topo, best_model, time_limit, num_flows, seed, num_tests, link_rate)
    df = scheduler_manager.run_dichotomy()

    filename = os.path.join(OUT_DIR, f'schedule_stat_{topo}_{num_tests}_{seed}_{time_limit}.csv')
    df.to_csv(filename, index=False)
    logging.info(f"scheduling statistics is saved to {filename}")


def run_tests(topo, num_flow, i, best_model, time_limit, seed):
    graph = get_graph(topo)
    logging.info(f"scheduling instance: {topo}, {num_flow} flows, {i}-th")
    flows = generate_flows(graph, num_flow, seed=seed + i)

    schedulers = [
        ('smt', SmtScheduler(graph, flows, timeout_s=time_limit)),
        ('no_wait_smt', NoWaitSmtScheduler(graph, flows, timeout_s=time_limit)),
        ('drl', DrlScheduler(graph, flows, timeout_s=time_limit, num_envs=1))
    ]
    schedulers[-1][1].load_model(best_model, alg=get_alg(best_model))

    results = [topo, num_flow, i, seed + i]
    for scheduler_str, scheduler in schedulers:
        logging.info(f"using {scheduler_str} to schedule...")
        is_scheduled, elapsed_time = schedule(scheduler)
        results.extend([is_scheduled, elapsed_time])

    results.append(results[-1] / results[-3])  # ratio_drl2smt
    return results


def get_alg(best_model):
    return re.search(r"best_model_([^_\.]+)", os.path.basename(best_model)).group(1)


if __name__ == '__main__':
    log_config(os.path.join(OUT_DIR, 'exp.log'), logging.DEBUG)
    execute_from_command_line(main)
