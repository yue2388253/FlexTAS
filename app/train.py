import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from sb3_contrib import MaskablePPO
import sys

from app.test import test
from definitions import OUT_DIR
from src.agent.encoder import FeaturesExtractor
from src.env.env_helper import generate_env
from src.lib.timing_decorator import timing_decorator
from src.network.net import generate_cev, generate_flows
from src.app.drl_scheduler import DrlScheduler

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

NUM_TIME_STEPS = 10000_00
NUM_ENVS = 2
NUM_FLOWS = 50

DRL_ALG = 'A2C'


def get_best_model_path():
    return os.path.join(OUT_DIR, f"best_model_{DRL_ALG}_{NUM_ENVS}")


def make_env(num_flows, rank: int):
    def _init():
        log_subdir = os.path.join(OUT_DIR, str(rank))
        os.makedirs(log_subdir, exist_ok=True)
        env = generate_env("CEV", num_flows, rank)
        env = Monitor(env, log_subdir)  # Wrap the environment with Monitor
        return env

    return _init


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dirs: (list[str]) Paths to the folder where the file created by the ``Monitor`` is generated.
    :param verbose: (int)
    """

    def __init__(self, rank: int, check_freq: int, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rank = rank
        self.log_dirs = [os.path.join(OUT_DIR, str(i)) for i in range(rank)]
        self.save_path = get_best_model_path()
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            for i in range(self.rank):
                self._check_and_update_model(i)

        return True

    def _check_and_update_model(self, rank):
        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dirs[rank]), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print(
                    "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                             mean_reward))

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Example for saving best model
                if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                self.model.save(self.save_path)


@timing_decorator(logging.info)
def train(num_time_steps=NUM_TIME_STEPS, num_flows=NUM_FLOWS):
    os.makedirs(OUT_DIR, exist_ok=True)

    n_envs = NUM_ENVS  # Number of environments to create
    env = SubprocVecEnv([make_env(num_flows, i) for i in range(n_envs)])  # or SubprocVecEnv

    policy_kwargs = dict(
        features_extractor_class=FeaturesExtractor,
    )

    model = DrlScheduler.SUPPORTING_ALG[DRL_ALG]("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    callback = SaveOnBestTrainingRewardCallback(n_envs, check_freq=1000)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, default=NUM_TIME_STEPS)
    parser.add_argument('--num_flows', type=int, nargs='?', default=NUM_FLOWS)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_envs', type=int, default=NUM_ENVS)
    parser.add_argument('--alg', type=str, default=None)
    args = parser.parse_args()

    if args.alg is not None:
        assert args.alg in DrlScheduler.SUPPORTING_ALG, ValueError(f"Unknown alg {args.alg}")
        DRL_ALG = args.alg

    NUM_ENVS = args.num_envs
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    train(args.time_steps, num_flows=args.num_flows)
    for i in range(NUM_ENVS):
        plot_results(os.path.join(OUT_DIR, str(i)))

    test('CEV', args.num_flows, NUM_ENVS, get_best_model_path(), DRL_ALG)
