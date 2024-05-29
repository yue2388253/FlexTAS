import argparse
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os

from app.test import test
from definitions import OUT_DIR
from src.agent.encoder import FeaturesExtractor
from src.env.env import NetEnv, TrainingNetEnv
from src.lib.log_config import log_config
from src.lib.timing_decorator import timing_decorator
from src.app.drl_scheduler import DrlScheduler
from src.network.net import generate_linear_5, generate_cev, generate_flows, Net

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

TOPO = 'CEV'
NUM_ENVS = multiprocessing.cpu_count()
NUM_FLOWS = 50

DRL_ALG = 'A2C'

MONITOR_ROOT_DIR = os.path.join(OUT_DIR, "monitor")


def get_best_model_path():
    return os.path.join(OUT_DIR, f"best_model_{TOPO}_{DRL_ALG}")


def make_env(num_flows, rank: int, topo: str, training: bool = True, link_rate: int = 100):
    def _init():
        if topo == "CEV":
            graph = generate_cev(link_rate)
        elif topo == "L5":
            graph = generate_linear_5(link_rate)
        else:
            raise ValueError(f"Unknown topo {topo}")

        if training:
            env = TrainingNetEnv(graph, generate_flows, num_flows, 1.0, 0.0)
        else:
            flows = generate_flows(graph, num_flows)
            env = NetEnv(graph, flows)

        # Wrap the environment with Monitor
        env = Monitor(env, os.path.join(MONITOR_DIR, f'{"train" if training else "eval"}_{rank}'))

        return env

    return _init


@timing_decorator(logging.info)
def train(topo: str, num_time_steps, num_flows=NUM_FLOWS, pre_trained_model=None, link_rate=100):
    os.makedirs(OUT_DIR, exist_ok=True)

    n_envs = NUM_ENVS  # Number of environments to create
    env = SubprocVecEnv([make_env(num_flows, i, topo, link_rate=link_rate) for i in range(n_envs)])

    if pre_trained_model is not None:
        model = DrlScheduler.SUPPORTING_ALG[DRL_ALG].load(pre_trained_model, env)
    else:
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )

        if DRL_ALG == 'DQN':
            # limit the buffer size.
            model = DrlScheduler.SUPPORTING_ALG[DRL_ALG]("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                                                         buffer_size=500_000)
        else:
            model = DrlScheduler.SUPPORTING_ALG[DRL_ALG]("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    eval_env = SubprocVecEnv([make_env(num_flows, i, topo, training=False, link_rate=link_rate) for i in range(n_envs)])
    callback = EvalCallback(eval_env, best_model_save_path=get_best_model_path(),
                            log_path=OUT_DIR, eval_freq=max(10000 // n_envs, 1))

    # logging.debug(model.policy)

    # Train the agent
    model.learn(total_timesteps=num_time_steps, callback=callback)

    logging.info("------Finish learning------")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.savefig(os.path.join(log_folder, "reward.png"))
    plt.show()


if __name__ == "__main__":
    # specify an existing model to train.
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, required=True)
    parser.add_argument('--num_flows', type=int, nargs='?', default=NUM_FLOWS)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
    parser.add_argument('--alg', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--topo', type=str, default="CEV")
    parser.add_argument('--link_rate', type=int, default=100)
    args = parser.parse_args()

    if args.alg is not None:
        assert args.alg in DrlScheduler.SUPPORTING_ALG, ValueError(f"Unknown alg {args.alg}")
        DRL_ALG = args.alg

    if args.link_rate is not None:
        support_link_rates = [100, 1000]
        assert args.link_rate in support_link_rates, \
            f"Unknown link rate {args.link_rate}, which is not in supported link rates {support_link_rates}"

    TOPO = args.topo

    NUM_ENVS = args.num_envs

    log_config(os.path.join(OUT_DIR, f"train.log"), logging.DEBUG)

    logging.info(args)

    done = False
    i = 0
    MONITOR_DIR = None
    while not done:
        try:
            MONITOR_DIR = os.path.join(MONITOR_ROOT_DIR, str(i))
            os.makedirs(MONITOR_DIR, exist_ok=False)
            done = True
        except OSError:
            i += 1
            continue
    assert MONITOR_DIR is not None

    logging.info("start training...")
    train(args.topo, args.time_steps,
          num_flows=args.num_flows,
          pre_trained_model=args.model,
          link_rate=args.link_rate)

    # MONITOR_DIR = os.path.join(MONITOR_ROOT_DIR, str(1))
    plot_results(MONITOR_DIR)

    test(args.topo, args.num_flows, NUM_ENVS,
         os.path.join(get_best_model_path(), "best_model"), DRL_ALG, args.link_rate)
