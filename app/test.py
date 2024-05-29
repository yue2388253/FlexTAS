import logging

from src.lib.timing_decorator import timing_decorator
from src.lib.execute import execute_from_command_line
from src.network.net import generate_cev, generate_flows, generate_linear_5
from src.app.drl_scheduler import DrlScheduler


@timing_decorator(logging.info)
def test(topo: str, num_flows: int, num_envs: int,
         best_model_path: str = None, alg: str = 'PPO', link_rate: int = 100):
    logging.basicConfig(level=logging.DEBUG)

    if topo == "L5":
        graph = generate_linear_5(link_rate)
    elif topo == "CEV":
        graph = generate_cev(link_rate)
    else:
        raise ValueError(f"Unknown topo {topo}")

    flows = generate_flows(graph, num_flows)
    scheduler = DrlScheduler(graph, flows, num_envs=num_envs)

    if best_model_path is not None:
        scheduler.load_model(best_model_path, alg=alg)

    logging.info("Start scheduling...")
    is_scheduled = scheduler.schedule()
    if is_scheduled:
        logging.info("Successfully scheduling the flows.")
    else:
        logging.error("Fail to find a valid solution.")

    return is_scheduled


if __name__ == '__main__':
    execute_from_command_line(test)