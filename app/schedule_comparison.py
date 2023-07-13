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


def main(num_flows: int, num_tests: int, seed: int = None, time_limit: int = 300):
    seed = seed if seed is not None else random.randint(0, 10000)
    results = []

    for i in range(num_tests):
        # generate graph and flows.
        graph = generate_linear_5()
        flows = generate_flows(graph, num_flows, seed=seed + i)

        # use smt to schedule
        logging.debug("using smt to schedule...")
        smt_scheduler = SmtScheduler(graph, flows, timeout_s=time_limit)
        start_time = time.time()
        is_scheduled_smt = smt_scheduler.schedule()
        smt_time = time.time() - start_time
        logging.debug(f"result: {'success' if is_scheduled_smt else 'failure'}")

        # use drl to schedule
        logging.debug("using drl to schedule...")
        drl_scheduler = DrlScheduler(graph, flows, timeout_s=time_limit)
        start_time = time.time()
        is_scheduled_drl = drl_scheduler.schedule()
        drl_time = time.time() - start_time
        logging.debug(f"result: {'success' if is_scheduled_drl else 'failure'}")

        results.append([is_scheduled_smt, smt_time, is_scheduled_drl, drl_time])

    df = pd.DataFrame(results, columns=['smt_scheduled', 'smt_time', 'drl_scheduled', 'drl_time'])
    filename = os.path.join(OUT_DIR, 'schedule_stat.csv')
    df.to_csv(filename)
    logging.info(f"scheduling statistics is saved to {filename}")
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    execute_from_command_line(main)
