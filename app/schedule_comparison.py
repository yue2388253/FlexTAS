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
from src.app.no_wait_tabu_scheduler import NoWaitTabuScheduler
from src.app.drl_scheduler import DrlScheduler
from src.lib.log_config import log_config
from src.lib.execute import execute_from_command_line
from src.network.net import generate_flows, generate_graph


def run_test(topo, num_flows, scheduler_str, scheduler_cls, seed, timeout, link_rate, best_model=None):
    graph = generate_graph(topo, link_rate)
    flows = generate_flows(graph, num_flows, seed=seed)

    scheduler = scheduler_cls(graph, flows, timeout_s=timeout)
    if isinstance(scheduler, DrlScheduler):
        assert best_model is not None
        scheduler.load_model(best_model, 'MaskablePPO')

    start_time = time.time()
    logging.info(f"Use [{scheduler_str}] Scheduling {num_flows} flows, seed {seed}")
    is_scheduled = scheduler.schedule()

    elapsed_time = time.time() - start_time

    result = [[topo, num_flows, seed, str(scheduler_str), is_scheduled, elapsed_time]]
    return result


class SchedulerManager:
    scheduler_dict = {
        'smt': SmtScheduler,
        'smt_no_wait': NoWaitSmtScheduler,
        'Tabu': NoWaitTabuScheduler,
        'drl': DrlScheduler
    }
    
    def __init__(self, topo, best_model, time_limit, num_flows, seed, num_tests, link_rate, schedulers: list[str]):
        # [scheduler, [seed, # of flows]]
        self.scheduler_capacity = defaultdict(lambda: defaultdict(None))

        self.topo = topo
        self.best_model = best_model
        self.time_limit = time_limit
        self.num_flows = sorted(num_flows)
        self.seed = seed
        self.num_tests = num_tests
        self.link_rate = link_rate
        self.schedulers = schedulers if schedulers is not None else self.scheduler_dict.keys()

    def run_normally(self):
        column_names = ['topo', 'num_flows', 'seed',
                        'scheduler', 'is_scheduled', 'consuming_time']
        df = pd.DataFrame(columns=column_names)
        for scheduler_str, num_flows, seed in \
                itertools.product(self.schedulers, self.num_flows, range(self.seed, self.seed + self.num_tests)):
            result = run_test(
                self.topo, num_flows, scheduler_str, self.scheduler_dict[scheduler_str], seed,
                self.time_limit, self.link_rate, self.best_model
            )
            filename = os.path.join(OUT_DIR, f'schedule_stat_{self.topo}_{self.link_rate}_{self.seed}_{self.time_limit}.csv')
            df_temp = pd.DataFrame(result, columns=column_names)
            df = pd.concat([df, df_temp], ignore_index=True)
            df.to_csv(filename, index=False)
            logging.info(f"scheduling statistics is saved to {filename}")

    def run_dichotomy(self):
        column_names = ['topo', 'num_flows', 'seed',
                        'scheduler', 'is_scheduled', 'consuming_time']
        df = pd.DataFrame(columns=column_names)

        for scheduler_str in self.schedulers:
            for seed in range(self.seed, self.seed + self.num_tests):
                low = 0
                high = len(self.num_flows) - 1
                while low <= high:
                    mid = (low + high) // 2

                    num_flows = self.num_flows[mid]
                    result = run_test(
                        self.topo, num_flows, scheduler_str, self.scheduler_dict[scheduler_str], seed,
                        self.time_limit, self.link_rate, self.best_model
                    )

                    is_scheduled = result[0][-2]

                    if is_scheduled:
                        for i in range(low, mid):
                            result.append([self.topo, self.num_flows[i], seed, scheduler_str, True, 0])
                        low = mid + 1
                    else:
                        for i in range(mid + 1, high + 1):
                            result.append([self.topo, self.num_flows[i], seed, scheduler_str, False, self.time_limit])
                        high = mid - 1

                    filename = os.path.join(OUT_DIR, f'schedule_stat_{self.topo}_{self.seed}_{self.time_limit}.csv')
                    df_temp = pd.DataFrame(result, columns=column_names)
                    df = pd.concat([df, df_temp], ignore_index=True)
                    df.to_csv(filename, index=False)
                    logging.info(f"scheduling statistics is saved to {filename}")


def schedule(scheduler):
    start_time = time.time()
    is_scheduled = scheduler.schedule()
    elapsed_time = time.time() - start_time
    logging.debug(f"result: {'success' if is_scheduled else 'failure'}")
    return is_scheduled, elapsed_time


def main(num_flows: str, num_tests: int, best_model: str, seed: int = None,
         link_rate: int = None, time_limit: int = 300, topo: str = 'L5', test_time: bool = False,
         schedulers: str = None):
    seed = seed or random.randint(0, 10000)
    num_flows = map(int, num_flows.split(',')) if isinstance(num_flows, str) else num_flows

    if schedulers is not None:
        schedulers = schedulers.split(',')

    scheduler_manager = SchedulerManager(
        topo, best_model, time_limit, num_flows,
        seed, num_tests, link_rate, schedulers
    )

    if test_time:
        logging.info("test time consumed.")
        scheduler_manager.run_normally()
    else:
        logging.info("saving time mode.")
        scheduler_manager.run_dichotomy()


def get_alg(best_model):
    return re.search(r"best_model_([^_\.]+)", os.path.basename(best_model)).group(1)


if __name__ == '__main__':
    log_config(os.path.join(OUT_DIR, 'exp.log'), logging.DEBUG)
    execute_from_command_line(main)
