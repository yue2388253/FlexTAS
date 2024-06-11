import logging
import os
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

from src.agent.encoder import FeaturesExtractor
from src.env.env import NetEnv
from src.lib.timing_decorator import timing_decorator
from src.network.net import Flow, Network
from src.app.scheduler import BaseScheduler, ScheduleRes


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

        self.res = None

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if game is done and game was a success
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if any(dones):
            for i, done in enumerate(dones):
                if done and infos[i].get('success'):
                    logging.info("Game successfully completed, stopping training...")
                    self.is_scheduled = True
                    self.res = infos[i].get('ScheduleRes')
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
    SUPPORTING_ALG = {
        'A2C': A2C,
        'DQN': DQN,
        'PPO': PPO,
        'MaskablePPO': MaskablePPO
    }

    def __init__(self, network: Network,
                 num_envs: int = 1, time_steps=100000, **kwargs):
        super().__init__(network,  **kwargs)
        self.num_envs = num_envs
        self.time_steps = time_steps

        self.env = SubprocVecEnv([lambda: NetEnv(network) for _ in range(self.num_envs)])
        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )
        self.model = MaskablePPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1)

        self.res = None

    def load_model(self, filepath: str, alg: str = 'PPO'):
        del self.model
        if filepath.endswith(r".zip"):
            filepath = filepath[:-4]
        assert os.path.isfile(f"{filepath}.zip"), f"No such file {filepath}"
        logging.info(f"loading model at {filepath}.zip")
        self.model = self.SUPPORTING_ALG[alg].load(filepath, self.env)

    @timing_decorator(logging.info)
    def schedule(self):
        callback = SuccessCallback(time_limit=self.timeout_s)

        # Train the agent
        self.model.learn(total_timesteps=self.time_steps, callback=callback)

        is_scheduled = callback.get_result()
        if is_scheduled:
            self.res = callback.res
            logging.info("Successfully scheduling the flows.")
        else:
            logging.error("Fail to find a valid solution.")

        return is_scheduled

    def get_res(self) -> ScheduleRes:
        return self.res
