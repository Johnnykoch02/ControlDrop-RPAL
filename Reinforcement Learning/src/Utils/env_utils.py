from ray.rllib.env import BaseEnv
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict
from typing import List, Tuple, Union, Callable
import gymnasium as gym
import threading


@PublicAPI
class AsyncVectorEnv(BaseEnv):
    def __init__(self, make_env: Callable, num_envs: int):
        self.num_envs = num_envs
        self.envs = [make_env() for _ in range(num_envs)]
        self.observation_spaces = [env.observation_space for env in self.envs]
        self.action_spaces = [env.action_space for env in self.envs]
        self.reset_pending = [True] * num_envs
        self.current_obs = [None] * num_envs
        self.current_rewards = [0] * num_envs
        self.current_dones = [False] * num_envs
        self.current_infos = [{}] * num_envs

        # Threading:
        self.locks = [threading.Lock() for _ in range(num_envs)]
        self.threads = [None] * num_envs

        # Initialization:
        for i in range(self.num_envs):
            with self.locks[i]:
                self.threads[i] = threading.Thread(target=self.try_reset, args=(i,))
                self.threads[i].start()

    def poll(
        self,
    ) -> Tuple[
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        MultiAgentDict,
        List[MultiAgentDict],
    ]:
        obs, rewards, dones, infos = {}, {}, {}, {}
        for i in range(self.num_envs):
            # if self.threads[i] is not None:
            #     self.threads[i].join()
            #     self.threads[i] = None

            if not self.locks[i].locked():
                with self.locks[i]:
                    if self.current_obs[i] is not None and not self.reset_pending[i]:
                        obs[i] = self.current_obs[i]
                        rewards[i] = self.current_rewards[i]
                        dones[i] = self.current_dones[i]
                        if dones[i]:  # Reset:
                            self.reset_pending[i] = True
                            self.threads[i] = threading.Thread(
                                target=self.try_reset, args=(i,)
                            )
                            self.threads[i].start()

                        infos[i] = self.current_infos[i]
        return obs, rewards, dones, infos, {}

    def send_actions(self, action_dict: MultiAgentDict) -> None:
        for i, action in action_dict.items():
            with self.locks[i]:
                if self.reset_pending[i]:
                    continue

                obs, reward, terminated, truncated, info = self.envs[i].step(action)
                done = terminated or truncated
                self.current_obs[i] = obs
                self.current_rewards[i] = reward
                self.current_dones[i] = done
                self.current_infos[i] = info

    def try_reset(self, env_id: int) -> Union[None, gym.Env]:
        print("[DBG] Resetting!")
        obs, _ = self.envs[env_id].reset()
        with self.locks[env_id]:
            self.current_obs[env_id] = obs
            self.current_rewards[env_id] = 0
            self.current_dones[env_id] = False
            self.current_infos[env_id] = {}
            self.reset_pending[env_id] = False
        self.threads[env_id].join(0.01)
        # self.threads[env_id] = None

    def get_sub_environments(self) -> List[gym.Env]:
        return self.envs

    def try_render(self, env_id: Union[int, None] = None) -> None:
        if env_id is None:
            for env in self.envs:
                env.render()
        else:
            self.envs[env_id].render()

    def close(self):
        for env in self.envs:
            if env is not None:
                env.close()
