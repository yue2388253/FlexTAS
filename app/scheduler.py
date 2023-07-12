import argparse
import inspect
import logging
import multiprocessing
import os
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

from src.agent.encoder import FeaturesExtractor
from src.env.env_helper import from_file
from src.lib.timing_decorator import timing_decorator


class SuccessCallback(BaseCallback):
    """
    A custom callback that stops training when the agent successfully completes the game.
    """

    def __init__(self, time_limit: int = 3600, verbose=0):
        """

        :param time_limit: the max training timme in seconds unit.
        """
        super(SuccessCallback, self).__init__(verbose)
        self.start_time = time.time()
        self.time_limit = time_limit
        self.is_scheduled = False

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if game is done and game was a success
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if any([infos[i].get('success') if done else False for i, done in enumerate(dones)]):
            logging.info("Game successfully completed, stopping training...")
            self.is_scheduled = True
            return False  # False means "stop training"

        # Check if time limit has been reached
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            logging.info(f"Time limit {self.time_limit}s reached, fail to find a solution. Stopping training...")
            return False  # False means "stop training"

        return True  # True means "continue training"

    def get_result(self) -> bool:
        return self.is_scheduled


@timing_decorator(logging.info)
def schedule(
        filename: os.PathLike | str,
        num_envs: int = multiprocessing.cpu_count() - 1,
        time_steps: int = 10000,
        time_limit: int = 3600
):
    assert os.path.isfile(filename), f"No such file {filename}"

    logging.basicConfig(level=logging.INFO)

    env = SubprocVecEnv([lambda: from_file(filename) for _ in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=FeaturesExtractor,
    )

    model = MaskablePPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    callback = SuccessCallback(time_limit=time_limit)

    # Train the agent
    model.learn(total_timesteps=time_steps, callback=callback)

    is_scheduled = callback.get_result()
    if is_scheduled:
        logging.info("Successfully scheduling the flows.")
    else:
        logging.error("Fail to find a valid solution.")

    return is_scheduled


if __name__ == '__main__':
    # Get the signature of the function
    sig = inspect.signature(schedule)

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    for name, param in sig.parameters.items():
        if param.default is param.empty:  # it's a required argument
            parser.add_argument('--' + name, required=True)
        else:  # it's an optional argument, use default value from function definition
            parser.add_argument('--' + name, default=param.default)

    args = parser.parse_args()

    # Convert args to a dictionary
    args_dict = vars(args)

    # Call the function with the arguments
    schedule(**args_dict)
