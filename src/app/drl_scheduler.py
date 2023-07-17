import logging
import networkx as nx
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

from src.agent.encoder import FeaturesExtractor
from src.env.env import NetEnv
from src.lib.timing_decorator import timing_decorator
from src.network.net import Flow
from src.app.scheduler import BaseScheduler


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


class DrlScheduler(BaseScheduler):
    def __init__(self, graph: nx.Graph, flows: list[Flow], timeout_s: int = 3600,
                 num_envs: int = 1, time_steps=100000):
        super().__init__(graph, flows, timeout_s)
        self.num_envs = num_envs
        self.time_steps = time_steps

    @timing_decorator(logging.info)
    def schedule(self):
        env = SubprocVecEnv([lambda: NetEnv(self.graph, self.flows) for _ in range(self.num_envs)])
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )

        model = MaskablePPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

        callback = SuccessCallback(time_limit=self.timeout)

        # Train the agent
        model.learn(total_timesteps=self.time_steps, callback=callback)

        is_scheduled = callback.get_result()
        if is_scheduled:
            logging.info("Successfully scheduling the flows.")
        else:
            logging.error("Fail to find a valid solution.")

        return is_scheduled
