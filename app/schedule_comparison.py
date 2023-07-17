import logging
import os
import pandas as pd
import random
import time

from definitions import OUT_DIR
from src.app.smt_scheduler import SmtScheduler
from src.app.drl_scheduler import DrlScheduler
from src.lib.execute import execute_from_command_line
from src.network.net import generate_linear_5, generate_flows


def schedule(graph, flows, scheduler_class, time_limit):
    scheduler = scheduler_class(graph, flows, timeout_s=time_limit)
    start_time = time.time()
    is_scheduled = scheduler.schedule()
    elapsed_time = time.time() - start_time
    logging.debug(f"result: {'success' if is_scheduled else 'failure'}")
    return is_scheduled, elapsed_time


def main(num_flows: str, num_tests: int, seed: int = None, time_limit: int = 300):
    seed = seed if seed is not None else random.randint(0, 10000)
    if isinstance(num_flows, str):
        num_flows = list(map(int, num_flows.split(',')))

    results = []
    for num_flow in num_flows:
        for i in range(num_tests):
            # generate graph and flows.
            graph = generate_linear_5()
            flows = generate_flows(graph, num_flow, seed=seed + i)

            # use smt to schedule
            logging.debug("using smt to schedule...")
            is_scheduled_smt, smt_time = schedule(graph, flows, SmtScheduler, time_limit)

            # use drl to schedule
            logging.debug("using drl to schedule...")
            is_scheduled_drl, drl_time = schedule(graph, flows, DrlScheduler, time_limit)

            ratio_drl2smt = drl_time / smt_time

            results.append([num_flow, i, seed + i,
                            is_scheduled_smt, smt_time, is_scheduled_drl, drl_time,
                            ratio_drl2smt])

        df = pd.DataFrame(results,
                          columns=['num_flows', 'index', 'seed',
                                   'smt_scheduled', 'smt_time', 'drl_scheduled', 'drl_time', 'ratio_drl2smt'])
        filename = os.path.join(OUT_DIR, f'schedule_stat_{num_tests}_{seed}_{time_limit}.csv')
        df.to_csv(filename)
        logging.info(f"scheduling statistics is saved to {filename}")

    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    execute_from_command_line(main)
