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
from src.lib.config import ConfigManager
from src.lib.execute import execute_from_command_line
from src.network.net import generate_linear_5, generate_cev, generate_flows


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
    if link_rate is not None:
        ConfigManager().config.set('Net', 'link_rate', str(link_rate))

    seed = seed or random.randint(0, 10000)
    num_flows = map(int, num_flows.split(',')) if isinstance(num_flows, str) else num_flows

    filename = os.path.join(OUT_DIR, f'schedule_stat_{topo}_{num_tests}_{seed}_{time_limit}.csv')
    column_names = ['topo', 'num_flows', 'index', 'seed',
                    'smt_scheduled', 'smt_time',
                    'smt_scheduled_no_wait', 'smt_time_no_wait',
                    'drl_scheduled', 'drl_time',
                    'ratio_drl2smt']
    df = pd.DataFrame(columns=column_names)

    for num_flow, i in itertools.product(num_flows, range(num_tests)):
        result = run_tests(topo, num_flow, i, best_model, time_limit, seed)

        # continuously write the results to avoid losing the results in case the process gets terminated unexpectedly.
        df_temp = pd.DataFrame([result], columns=column_names)
        df = pd.concat([df, df_temp], ignore_index=True)
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
    logging.basicConfig(level=logging.INFO)
    execute_from_command_line(main)
