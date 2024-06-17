import logging
import os
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import time
import torch

from src.agent.encoder import FeaturesExtractor
from src.env.env import NetEnv
from src.lib.timing_decorator import timing_decorator
from src.network.net import Flow, Network
from src.app.scheduler import BaseScheduler, ScheduleRes


class DrlScheduler(BaseScheduler):
    # todo: only Maskable PPO
    SUPPORTING_ALG = {
        'A2C': A2C,
        'DQN': DQN,
        'PPO': PPO,
        'MaskablePPO': MaskablePPO
    }

    # todo: should specify model, otherwise load the default model.
    def __init__(self, network: Network,
                 num_envs: int = 1, time_steps=100000, **kwargs):
        super().__init__(network,  **kwargs)
        self.num_envs = num_envs
        self.time_steps = time_steps

        # Choose the device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = SubprocVecEnv([lambda: NetEnv(network) for _ in range(self.num_envs)])
        # for debug
        # self.env = DummyVecEnv([lambda: NetEnv(network) for _ in range(self.num_envs)])

        policy_kwargs = dict(
            features_extractor_class=FeaturesExtractor,
        )
        self.model = MaskablePPO("MultiInputPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1, device=self.device)

        self.res = None

    def load_model(self, filepath: str, alg: str = 'PPO'):
        del self.model
        if filepath.endswith(r".zip"):
            filepath = filepath[:-4]
        assert os.path.isfile(f"{filepath}.zip"), f"No such file {filepath}"
        logging.info(f"loading model at {filepath}.zip")
        self.model = self.SUPPORTING_ALG[alg].load(filepath, self.env, device=self.device)  # Specify device

    @timing_decorator(logging.info)
    def schedule(self):

        # todo: learn to schedule
        # Simulate some scheduling logic here
        is_scheduled = self._simulate_scheduling()

        if is_scheduled:
            logging.info("Successfully scheduling the flows.")
        else:
            logging.error("Fail to find a valid solution.")

        return is_scheduled

    def _simulate_scheduling(self):
        logging.info("Scheduling without training the model.")
        start_time = time.time()
        self.res = None
        for _ in range(self.time_steps):
            obs = self.env.reset()
            done = False
            while not done:
                action_masks = np.array(self.env.env_method('action_masks'))
                action, _states = self.model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, rewards, dones, infos = self.env.step(action)
                # Implementing SuccessCallback logic directly here
                if any(dones):
                    for i, done in enumerate(dones):
                        if done:
                            if infos[i].get('success'):
                                logging.info("Game successfully completed, stopping simulation...")
                                self.res = infos[i].get('ScheduleRes')
                                return True  # Successfully scheduled

                            # Reset the done environment
                            obs[i] = self.env.env_method('reset', indices=i)[0]

                # Check if time limit has been reached
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout_s:
                    logging.info(f"Time limit {self.timeout_s}s reached, fail to find a solution. Stopping simulation...")
                    return False  # Failed to schedule within the time limit

        return False  # Failed to schedule within the given time steps

    def get_res(self) -> ScheduleRes:
        return self.res
