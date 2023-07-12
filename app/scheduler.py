import logging
import os
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent.encoder import FeaturesExtractor
from src.env.env_helper import from_file
from src.lib.timing_decorator import timing_decorator


class SuccessCallback(BaseCallback):
    """
    A custom callback that stops training when the agent successfully completes the game.
    """
    def __init__(self, verbose=0):
        super(SuccessCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if game is done and game was a success
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if any([infos[i].get('success') if done else False for i, done in enumerate(dones)]):
            print("Game successfully completed, stopping training...")
            return False  # False means "stop training"
        return True  # True means "continue training"


@timing_decorator(logging.info)
def schedule(filename: os.PathLike | str, num_envs: int = 1, time_steps: int = 10000):
    assert os.path.isfile(filename), f"No such file {filename}"

    env = DummyVecEnv([lambda: from_file(filename) for _ in range(num_envs)])

    policy_kwargs = dict(
        features_extractor_class=FeaturesExtractor,
    )

    model = MaskablePPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    callback = SuccessCallback()

    # Train the agent
    model.learn(total_timesteps=time_steps, callback=callback)

    logging.info("------Finish learning------")

    return True

