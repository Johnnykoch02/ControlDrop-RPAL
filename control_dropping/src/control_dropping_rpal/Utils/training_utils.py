import os
import logging

from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import numpy as np


class SaveOnBestRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, env=None):
        super(SaveOnBestRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path_reward = os.path.join(log_dir, 'best_reward_model')
        self.save_path_accuracy = os.path.join(log_dir, 'best_accuracy_model')
        self.best_mean_reward = -np.inf
        self.best_accuracy = -np.inf
        self.env = env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path_reward is not None:
            os.makedirs(self.save_path_reward, exist_ok=True)
        if self.save_path_accuracy is not None:
            os.makedirs(self.save_path_accuracy, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                save_path = os.path.join(self.log_dir, f"checkpoint_{self.n_calls}_steps")
                self.model.save(save_path)
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
                        logging.info(f"Saving new best model to {self.save_path_reward}")
                    self.model.save(self.save_path_reward)

                accuracy = self.env.current_accuracy()
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    if self.verbose > 0:
                        logging.info(f"Saving new best accuracy to {self.save_path_accuracy}")
                    self.model.save(self.save_path_accuracy)
            except Exception as e:
                logging.error(f"Error in SaveOnBestRewardCallback: {e}")

        return True