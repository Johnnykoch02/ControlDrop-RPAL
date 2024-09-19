import torch
import numpy as np
import torch.nn as nn


from tensordict.tensordict import TensorDict

from control_dropping_rpal.RL.Networks.ExtractorNetworks import (
    TemporalObjectTactileEncoder_Additive,
)
from control_dropping_rpal.RL.control_dropping_env import (
    base_observation_space,
    default_action_space,
    T_buffer,
)

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Normal

from control_dropping_rpal.Utils.env_utils import is_vectorized_observation

# TODO: There is an issue using the observation space and action space inside of the ActorCriticNetwork. What we should do is fix this ASAP. For now it is hard coded.


class ControlDropPolicy(ActorCriticPolicy):
    def __init__(
        self, observation_space, action_space, lr_schedule, model_config=None, **kwargs
    ):
        self.model_config = model_config if model_config is not None else {}

        # Initialize device before calling super().__init__()
        # self.device = self.model_config.get("device", "auto")

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self._device = None

        # Initialize the policy components
        self._init_policy_components()

    def _init_policy_components(self):
        # Initialize components from the original ControlDroppingPolicy
        self.obs_space = base_observation_space
        self.action_space = default_action_space
        self.num_outputs = np.prod(default_action_space.shape) * 2

        temporal_dim = self.model_config.get("temporal_dim", T_buffer)
        obj_encoder_vec_encoding_size = self.model_config.get(
            "obj_encoder_vec_encoding_size", 8
        )
        dynamix_num_tsf_layer = self.model_config.get("obj_encoder_num_tsf_layer", 8)
        obj_encoder_load_path = self.model_config.get("obj_encoder_load_path", None)
        obj_encoder_freeze_params = self.model_config.get(
            "obj_encoder_freeze_params", False
        )

        self.features = TemporalObjectTactileEncoder_Additive(
            observation_space=self.obs_space,
            device=self.device,
            vec_encoding_size=obj_encoder_vec_encoding_size,
            num_tsf_layer=dynamix_num_tsf_layer,
            t_dim_size=temporal_dim,
            load_pretrain=False,
            use_mask=True,
        )

        if obj_encoder_load_path is not None:
            print("[ControlDropPolicy]: loading feature encoder from checkpoint.")
            self.features.load_checkpoint(obj_encoder_load_path)
            if obj_encoder_freeze_params:
                self.features.freeze_parameters()

        # state_attrib_size = self.obs_space["state_attrib"].shape[0]
        # self.estimator = nn.Sequential(
        #     nn.Linear(self.features.vec_encoding_size + state_attrib_size, self.features.vec_encoding_size),
        #     nn.GELU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
        #     nn.GELU(),
        #     nn.Dropout(0.05),
        # )
        _actions_projected_size = 64
        self.action_projector = nn.Linear(40, _actions_projected_size)  # Projecting from 40 to 64

        self.estimator = nn.Sequential(
            nn.Linear(self.features.vec_encoding_size + _actions_projected_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Linear(
                self.features.vec_encoding_size, self.features.vec_encoding_size // 2
            ),
            nn.GELU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Linear(self.features.vec_encoding_size, 1),
        )

        # Initialize action distribution components
        action_dim = self.action_space.shape[0]
        self.action_net = nn.Linear(self.features.vec_encoding_size // 2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

    def forward(self, obs, deterministic=False):
        features = self._extract_features(obs)
        actions = obs["actions"]
        latent_pi, latent_vf = self._get_latent(features, actions)

        # Process latent_pi through policy_head
        policy_features = self.policy_head(latent_pi)

        # Compute mean action
        mean_actions = self.action_net(policy_features)

        # Compute the standard deviation
        log_std = self.log_std.expand_as(mean_actions)
        std = torch.exp(log_std)

        # Sample actions
        if deterministic:
            actions = mean_actions
        else:
            actions = Normal(mean_actions, std).rsample()

        log_prob = Normal(mean_actions, std).log_prob(actions).sum(axis=-1)

        # Compute value
        values = self.value_head(latent_vf)

        return actions, values, log_prob

    def _extract_features(self, obs):
        features = self.features(obs)
        return features

    def _get_latent(self, features, actions):
        B, T, D = actions.shape
        actions_reshaped = actions.view(B, T * D)
        projected_actions = self.action_projector(actions_reshaped)
        shared_latent = self.estimator(torch.cat((features, projected_actions), dim=-1))
        return shared_latent, shared_latent

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self._extract_features(obs)
        actions = obs["actions"]
        _, latent_vf = self._get_latent(features, actions)
        return self.value_head(latent_vf)

    def evaluate_actions(self, obs, actions):
        features = self._extract_features(obs)
        actions = obs["actions"]
        latent_pi, latent_vf = self._get_latent(features, actions)

        policy_features = self.policy_head(latent_pi)
        mean_actions = self.action_net(policy_features)

        log_std = self.log_std.expand_as(mean_actions)
        std = torch.exp(log_std)

        distribution = Normal(mean_actions, std)
        log_prob = distribution.log_prob(actions).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)

        values = self.value_head(latent_vf)

        return values, log_prob, entropy

    def predict(self, observation, state=None, episode_start=None, deterministic=False):

        vectorized_env = is_vectorized_observation(observation, self.observation_space)
        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = torch.as_tensor(observation).float().to(self.device)
        with torch.no_grad():
            actions, values, log_prob, entropy = self.forward(
                observation, deterministic=deterministic
            )
        actions = actions.cpu().numpy()

        if not vectorized_env:
            actions = actions[0]

        return actions, state

    # @property
    # def device(self) -> torch.device:
    #     """Infer which device this policy lives on by inspecting its parameters.
    #     If it has no parameters, the 'cpu' device is used as a fallback.

    #     :return:
    #     """
    #     for param in self.parameters():
    #         return param.device
    #     return torch.device("cpu")
