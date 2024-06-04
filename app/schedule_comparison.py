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

    result = [topo, num_flows, seed, str(scheduler_str), is_scheduled, elapsed_time]
    return result


class SchedulerManager:
    scheduler_dict = {
        'smt': SmtScheduler,
        'smt_no_wait': NoWaitSmtScheduler,
        'Tabu': NoWaitTabuScheduler,
        'drl': DrlScheduler
    }
    
    def __init__(self, topos, best_model, time_limit, num_flows, seed, num_tests, link_rate, schedulers: list[str]):
        # [scheduler, [seed, # of flows]]
        self.scheduler_capacity = defaultdict(lambda: defaultdict(None))

        self.topos = topos
        self.best_model = best_model
        self.time_limit = time_limit
        self.num_flows = sorted(num_flows)
        self.seed = seed
        self.num_tests = num_tests
        self.link_rate = link_rate
        self.schedulers = schedulers if schedulers is not None else self.scheduler_dict.keys()

    def run(self):
        column_names = ['topo', 'num_flows', 'seed',
                        'scheduler', 'is_scheduled', 'consuming_time']
        results = []
        filename = os.path.join(OUT_DIR, f'schedule_stat_{self.link_rate}_{self.seed}_{self.time_limit}.csv')
        for topo, scheduler_str, num_flows, seed in \
                itertools.product(self.topos, self.schedulers, self.num_flows, range(self.seed, self.seed + self.num_tests)):
            result = run_test(
                topo, num_flows, scheduler_str, self.scheduler_dict[scheduler_str], seed,
                self.time_limit, self.link_rate, self.best_model
            )
            results.append(result)

            df = pd.DataFrame(results, columns=column_names)
            df.to_csv(filename, index=False)
            logging.info(f"scheduling statistics is saved to {filename}")


def main(num_flows: str, num_tests: int, best_model: str, seed: int = None,
         link_rate: int = None, time_limit: int = 300, topos: str = 'L5',
         schedulers: str = None):
    seed = seed or random.randint(0, 10000)
    num_flows = map(int, num_flows.split(',')) if isinstance(num_flows, str) else num_flows

    if schedulers is not None:
        schedulers = schedulers.split(',')

    topos = topos.split(',')

    scheduler_manager = SchedulerManager(
        topos, best_model, time_limit, num_flows,
        seed, num_tests, link_rate, schedulers
    )

    scheduler_manager.run()


def get_alg(best_model):
    return re.search(r"best_model_([^_\.]+)", os.path.basename(best_model)).group(1)


if __name__ == '__main__':
    log_config(os.path.join(OUT_DIR, 'exp.log'), logging.DEBUG)
    execute_from_command_line(main)
