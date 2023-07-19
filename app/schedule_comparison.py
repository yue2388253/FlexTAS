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
from src.lib.execute import execute_from_command_line
from src.network.net import generate_linear_5, generate_cev, generate_flows


def schedule(scheduler):
    start_time = time.time()
    is_scheduled = scheduler.schedule()
    elapsed_time = time.time() - start_time
    logging.debug(f"result: {'success' if is_scheduled else 'failure'}")
    return is_scheduled, elapsed_time


def main(num_flows: str, num_tests: int, best_model: str,
         seed: int = None, time_limit: int = 300, topo: str = 'L5',
         ):
    seed = seed if seed is not None else random.randint(0, 10000)
    if isinstance(num_flows, str):
        num_flows = list(map(int, num_flows.split(',')))

    topos = list(topo.split(','))

    results = []
    for topo, num_flow, i in itertools.product(topos, num_flows, range(num_tests)):
        # generate graph and flows.
        if topo == 'L5':
            graph = generate_linear_5()
        elif topo == 'CEV':
            graph = generate_cev()
        else:
            raise ValueError(f"Unknown graph type {topo}")

        logging.info(f"scheduling instance: {topo}, {num_flow} flows, {i}-th")

        flows = generate_flows(graph, num_flow, seed=seed + i)

        # use smt to schedule
        logging.info("using smt to schedule...")
        scheduler = SmtScheduler(graph, flows, timeout_s=time_limit)
        is_scheduled_smt, smt_time = schedule(scheduler)

        # use no-wait smt to schedule
        logging.info("using no-wait smt to schedule...")
        scheduler = NoWaitSmtScheduler(graph, flows, timeout_s=time_limit)
        is_scheduled_smt_no_wait, smt_no_wait_time = schedule(scheduler)

        # use drl to schedule
        logging.info("using drl to schedule...")
        alg, num_envs = re.search(r"_([^_]+)_(\d+)(\.zip)?$", os.path.basename(best_model)).group(1, 2)
        scheduler = DrlScheduler(graph, flows, time_limit, num_envs=int(num_envs))
        scheduler.load_model(best_model, alg=alg)

        is_scheduled_drl, drl_time = schedule(scheduler)

        ratio_drl2smt = drl_time / smt_time

        results.append([num_flow, i, seed + i,
                        is_scheduled_smt, smt_time,
                        is_scheduled_smt_no_wait, smt_no_wait_time,
                        is_scheduled_drl, drl_time,
                        ratio_drl2smt])

    df = pd.DataFrame(results,
                      columns=['num_flows', 'index', 'seed',
                               'smt_scheduled', 'smt_time',
                               'smt_scheduled_no_wait', 'smt_time_no_wait',
                               'drl_scheduled', 'drl_time',
                               'ratio_drl2smt'])
    filename = os.path.join(OUT_DIR, f'schedule_stat_{num_tests}_{seed}_{time_limit}.csv')
    df.to_csv(filename)
    logging.info(f"scheduling statistics is saved to {filename}")

    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    execute_from_command_line(main)
