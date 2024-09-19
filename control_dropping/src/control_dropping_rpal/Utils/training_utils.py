import os
import logging

from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np


class SaveOnBestRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve episode info
            episode_rewards = self.model.ep_info_buffer
            if len(episode_rewards) > 0:
                mean_reward = np.mean([ep['r'] for ep in episode_rewards])
                if self.verbose > 0:
                    logging.info(f"Num timesteps: {self.num_timesteps}")
                    logging.info(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # New best model, save the agent
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        logging.info(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True