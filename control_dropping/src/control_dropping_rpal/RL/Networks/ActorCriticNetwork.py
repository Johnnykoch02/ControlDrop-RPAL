import torch
import numpy as np
import torch.nn as nn
from torchrl.modules import TensorDictModule, ProbabilisticActor, ValueOperator
from tensordict.tensordict import TensorDict

from control_dropping_rpal.RL.Networks.ExtractorNetworks import (
    TemporalObjectTactileEncoder_Additive,
)
from control_dropping_rpal.RL.control_dropping_env import (
    base_observation_space,
    default_action_space,
    T_buffer,
)

# TODO: There is an issue using the observation space and action space inside of the ActorCriticNetwork. What we should do is fix this ASAP. For now it is hard coded.


class ControlDroppingPolicy(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.obs_space = base_observation_space
        self.action_space = default_action_space
        self.num_outputs = (
            np.prod(default_action_space.shape) * 2
        )  # MEAN, STD for i = 1:num_actions)

        temporal_dim = model_config.get("temporal_dim", T_buffer)
        obj_encoder_vec_encoding_size = model_config.get(
            "obj_encoder_vec_encoding_size", 8
        )
        dynamix_num_tsf_layer = model_config.get("obj_encoder_num_tsf_layer", 8)
        obj_encoder_load_path = model_config.get("obj_encoder_load_path", None)
        obj_encoder_freeze_params = model_config.get("obj_encoder_freeze_params", False)

        self.device = model_config.get("device", "cpu")
        torch.set_default_device(self.device)

        self.gelu = nn.GELU()

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
            print("[ControlDroppingPolicy]: loading feature encoder from checkpoint.")
            self.features.load_checkpoint(obj_encoder_load_path)

            if obj_encoder_freeze_params:
                self.features.freeze_parameters()

        state_attrib_size = self.obs_space["state_attrib"].shape[0]
        self.estimator = nn.Sequential(
            nn.Linear(
                self.features.vec_encoding_size + state_attrib_size,
                self.features.vec_encoding_size,
            ),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Dropout(0.05),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Linear(self.features.vec_encoding_size, self.num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.features.vec_encoding_size, self.features.vec_encoding_size),
            nn.GELU(),
            nn.Linear(self.features.vec_encoding_size, 1),
        )

        self.to(self.device)

        # Not being used atm:
        # self.features = DynamixModel(
        #     state_space=obs_space,
        #     device=device,
        #     vec_encoding_size=dynamix_vec_encoding_size,
        #     num_tsf_layer=dynamix_num_tsf_layer,
        #     num_residual_blocks=dynamix_num_residual_blocks,

        #     pretrain=False,
        # )
        # transformer_layer = nn.TransformerEncoderLayer(
        #     batch_first=True,
        #     d_model=self.features.object_encoder.flatten_size,
        #     nhead=action_ff_head,
        #     dim_feedforward=action_ff_dim,
        #     dropout=0.08,
        #     device=self.device,
        # )
        # # Stack 4 of these layers together
        # self.actions_encoder = nn.TransformerEncoder(
        #     transformer_layer,
        #     num_layers=4,
        # )

        # self.action_output_dim = (
        # self.features.object_encoder.flatten_size * self.action_encoder_num_action
        # )
        # self.lower_feature_dim_cast = nn.Linear(
        #     self.action_output_dim, self.features.object_encoder.flatten_size
        # )

    def forward(self, tensordict: TensorDict):
        obs = tensordict.get("observation")
        features = self.features(obs)
        state_estimation = self.estimator(
            torch.cat([features, obs["state_attrib"]], dim=1)
        )

        action_params = self.policy_head(state_estimation)
        value = self.value_head(state_estimation)

        return TensorDict(
            {
                "action_params": action_params,
                "state_value": value,
            },
            batch_size=tensordict.batch_size,
        )
