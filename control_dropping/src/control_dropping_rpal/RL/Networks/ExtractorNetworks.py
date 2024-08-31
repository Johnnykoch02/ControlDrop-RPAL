# from .control_dropping_env import BarrettHandGym
from stable_baselines3 import PPO
import os
from typing import Dict, List, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
from control_dropping_rpal.RL.control_dropping_env import BerrettHandGym, T_buffer, base_observation_space
import math 
from math import inf
from torch import nn
import json
from torch.nn import functional as F
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ------------------------------------------------------------------------- #
# ---------------------------- MMExtractor -------------------------------- #
# ------------------------------------------------------------------------- #


class MMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(MMExtractor, self).__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "hand_config":
                extractors[key] = nn.Sequential(
                    nn.BatchNorm1d(7), nn.Linear(7, 8), nn.LeakyReLU()
                )
                total_concat_size += 8
            elif key == "hand_torque":
                extractors[key] = nn.Sequential(
                    nn.BatchNorm1d(3), nn.Linear(3, 4), nn.LeakyReLU()
                )
                total_concat_size += 4
            # elif key == 'palm_tactile':
            #     extractors[key] = nn.Sequential(#8*3*1
            #          nn.BatchNorm2d(subspace.shape[0]),
            #          nn.Conv2d(subspace.shape[0], 32, (3, 3), padding='same'),
            #     )
            elif "tactile" in key:
                extractors[key] = nn.Sequential(
                    nn.BatchNorm2d(subspace.shape[0]),
                    nn.Conv2d(subspace.shape[0], 32, (3, 3), padding="same"),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 64, (3, 3), padding="same"),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 128, (3, 3), padding="same"),
                    nn.LeakyReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                total_concat_size += 128
            elif key == "ball_count":
                extractors[key] = nn.Sequential(
                    nn.BatchNorm1d(subspace.shape[0]), nn.Linear(10, 16), nn.LeakyReLU()
                )
                total_concat_size += 16
            elif key == "ball_location":
                extractors[key] = nn.Sequential(
                    nn.BatchNorm1d(subspace.shape[0]),
                    nn.Linear(42, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 128),  # TODO: this change should be tested
                    nn.LeakyReLU(),
                    nn.LayerNorm(),
                )
                total_concat_size += 128

        self.extractors = nn.ModuleDict(extractors)
        self._LinearLayers = nn.Sequential(
            nn.Linear(total_concat_size, 3072),
            nn.LeakyReLU(),
            nn.Dropout(0.01),
            nn.Linear(3072, 1536),
            nn.LeakyReLU(),
        )

        self._features_dim = 1536

    def forward(self, observations):
        encoded_tensor_list = []
        """extractors contain nn.Modules that do all of our processing """
        for key, extractor in self.extractors.items():
            # print('Key:', key, 'Extractor:', extractor)
            encoded_tensor_list.append(extractor(observations[key]))

        return self._LinearLayers(th.cat(encoded_tensor_list, dim=1))


# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
# ----------------------- TestExtractorVelocity --------------------------- #
# ------------------------------------------------------------------------- #


class TestExtractorVelocity(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(TestExtractorVelocity, self).__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "hand_config":
                extractors[key] = nn.Sequential(nn.Linear(7, 42), nn.LeakyReLU())
                total_concat_size += 42
            elif key == "hand_torque":
                extractors[key] = nn.Sequential(
                    nn.Linear(3, 42),
                    nn.LeakyReLU(),
                )
                total_concat_size += 42
            # elif key == 'palm_tactile':
            #     extractors[key] = nn.Sequential(#8*3*1
            #          nn.BatchNorm2d(subspace.shape[0]),
            #          nn.Conv2d(subspace.shape[0], 32, (3, 3), padding='same'),
            #     )
            elif "tactile" in key:
                extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 32, (3, 3), padding="same"),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 64, (3, 3), padding="same"),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 128, (3, 3), padding="same"),
                    nn.LeakyReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                total_concat_size += 128
            elif key == "ball_count":
                extractors[key] = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.Tanh(),
                )
                total_concat_size += 16
            elif key == "ball_location":
                extractors[key] = nn.Sequential(
                    nn.Linear(42, 128),  # TODO: this change should be tested
                    nn.Tanh(),  # test this change as opposed to the
                )
                total_concat_size += 128
            elif key == "obj_velocity":  # Same as Location
                extractors[key] = nn.Sequential(
                    nn.Linear(42, 128),
                    nn.Tanh(),
                )
                total_concat_size += 128
            elif key == "obj_angular_velocity":
                extractors[key] = nn.Sequential(
                    nn.Linear(10, 128),
                    nn.Tanh(),
                )
                total_concat_size += 128
            elif key == "progress_bar":
                extractors[key] = nn.Sequential(nn.Linear(1, 24), nn.Tanh())
                total_concat_size += 24

            elif key == "previous_actions":
                extractors[key] = nn.Sequential(
                    nn.Linear(15, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.LayerNorm(
                        256,
                    ),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.LayerNorm(
                        256,
                    ),
                )
                total_concat_size += 256
        self.encoding_size = 1092
        self.extractors = nn.ModuleDict(extractors)
        # Pre-encoding for LSTM
        self._LinearLayers = nn.Sequential(
            nn.Linear(total_concat_size, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(0.1),  # Maybe we can get rid of dropout
            nn.LayerNorm(self.encoding_size),
            nn.Linear(self.encoding_size, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.encoding_size),
            nn.Linear(self.encoding_size, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.encoding_size),
        )
        # Layer Weight Optimization
        prev_layer = None
        for layer in self._LinearLayers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
                prev_layer = layer

        self._features_dim = self.encoding_size  # total_concat_size
        self.output_dim = self.encoding_size  # total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []
        """extractors contain nn.Modules that do all of our processing """
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return self._LinearLayers(th.cat(encoded_tensor_list, dim=1))


# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
# -------------------- BaseTransformerFeatureEncoder ---------------------- #
# ------------------------------------------------------------------------- #


class BaseTransformerFeatureEncoder(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, vec_encoding_size=128, decoder_size=1024, device="cuda"
    ):
        super(BaseTransformerFeatureEncoder, self).__init__(
            observation_space, features_dim=1
        )
        extractors = {}
        total_concat_size = 0
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)

        self.pos_keys = ["tactile_pos", "ball_location", "state_attrib"]

        self.features_keys = [
            "palm_tactile",
            "finger_1_tactile",
            "finger_2_tactile",
            "finger_3_tactile",
            "obj_velocity",
        ]
        self.feature_encoders = {}
        self.positional_encoding_size = 0
        for key, subspace in observation_space.spaces.items():
            if key in self.pos_keys:
                self.positional_encoding_size += subspace.shape[0]
            elif key in self.features_keys:
                self.feature_encoders[key] = {
                    "value_enc": nn.Sequential(
                        nn.Linear(
                            subspace.shape[0], self.vec_encoding_size, device=device
                        ),
                        nn.Sigmoid(),
                        nn.Linear(
                            self.vec_encoding_size,
                            self.vec_encoding_size,
                            device=device,
                        ),
                        nn.Sigmoid(),
                        nn.Linear(
                            self.vec_encoding_size,
                            int(self.vec_encoding_size / 2),
                            device=device,
                        ),
                        nn.Sigmoid(),
                    )
                }
        for key in self.features_keys:
            self.feature_encoders[key]["pos_enc"] = nn.Sequential(
                nn.Linear(
                    self.positional_encoding_size,
                    self.positional_encoding_size,
                    device=device,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.positional_encoding_size, self.vec_encoding_size, device=device
                ),
                nn.ReLU(),
                nn.Linear(
                    self.vec_encoding_size,
                    int(self.vec_encoding_size / 2),
                    device=device,
                ),
                nn.ReLU(),
            )

        self.trns_encoder = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=self.vec_encoding_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.15,
            device=device,
        )

        output_shape = (1, len(self.features_keys), self.vec_encoding_size)

        flatten_size = output_shape[1] * output_shape[2]

        self.decoder_size = decoder_size

        self.decoder = nn.Sequential(
            nn.Flatten(),  # Flatten the transformer output
            nn.Linear(flatten_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.LayerNorm(self.decoder_size),
            nn.Dropout(p=0.1),
        )  # Triple Layer Decoder

        self._features_dim = self.decoder_size
        self.output_dim = self.decoder_size

    def preprocess(self, observations):
        positional_tensors = []
        encoded_features_tensors = {}
        trf_input = []
        """extractors contain nn.Modules that do all of our processing """
        for key in self.features_keys:
            encoded_features_tensors[key] = self.feature_encoders[key]["value_enc"](
                observations[key].to(self.device)
            )
        for key in self.pos_keys:
            positional_tensors.append(observations[key].to(self.device))

        pos_tensor = th.cat(positional_tensors, dim=1)
        for key in self.features_keys:
            encoded_pos_tensor = self.feature_encoders[key]["pos_enc"](
                pos_tensor
            ).unsqueeze(dim=1)
            trf_input.append(
                th.cat(
                    [
                        encoded_features_tensors[key].unsqueeze(dim=1),
                        encoded_pos_tensor,
                    ],
                    dim=2,
                ).to(self.device)
            )

        trf_input_tensor = th.cat(trf_input, dim=1).to(self.device)

        return trf_input_tensor

    def forward(self, observations):
        trf_input_tensor = self.preprocess(observations)
        trf_output = self.trns_encoder(trf_input_tensor)
        features_output = self.decoder(trf_output)
        _norm = nn.LayerNorm(
            features_output.shape
        )  # Exploding Gradient: https://github.com/hill-a/stable-baselines/issues/340
        features_normalized = _norm(features_output)
        return th.nan_to_num(features_normalized)


# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
# -------------------- ObjectTactileEncoder_Additive ---------------------- #
# ------------------------------------------------------------------------- #


class ObjectTactileEncoder_Additive(nn.Module):
    def __init__(
        self,
        observation_space,
        vec_encoding_size=1024,
        tactile_dim=2 + 34 * 3,
        object_dim=6,
        num_tsf_layer=4,
        device="cuda",
        load_pretrain=True,
        use_mask=False,
    ):
        super(ObjectTactileEncoder_Additive, self).__init__()
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        self.observation_space = observation_space
        self.save_path = os.path.join(os.path.dirname(__file__), "pretrained_models")
        self.tactile_dim = tactile_dim
        self.object_dim = object_dim
        # TODO: Add an Angular Velocity to the Value vector
        self.tactile_values = [
            "palm_tactile",
            "finger_1_tactile",
            "finger_2_tactile",
            "finger_3_tactile",
        ]
        self.tactile_positions = [
            "palm_location",
            "finger_1_location",
            "finger_2_location",
            "finger_3_location",
        ]
        self.object_projections = ["obj_velocity"]
        self.object_positions = ["obj_location"]
        # Used to project our features to the right dimension
        self.tactile_val_projection = nn.Sequential(
            nn.Linear(
                self.observation_space["palm_tactile"].shape[0],
                self.vec_encoding_size * 2,
                self.device,
            ),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
        )
        self.tactile_pos_projection = nn.Sequential(
            nn.Linear(tactile_dim, self.vec_encoding_size * 2, self.device),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
        )
        self.object_val_projection = nn.Sequential(
            nn.Linear(self.object_dim, self.vec_encoding_size * 2, self.device),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
        )
        self.object_pos_projection = nn.Sequential(
            nn.Linear(self.object_dim, self.vec_encoding_size * 2, self.device),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=self.vec_encoding_size,
            nhead=4,
            dim_feedforward=256,
            dropout=0.08,
            device=self.device,
        )
        # Stack 4 of these layers together
        self.trns_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_tsf_layer,
        )
        self.output_shape = (
            1,
            self.observation_space["obj_velocity"].shape[0] + len(self.tactile_values),
            self.vec_encoding_size,
        )
        self.flatten_size = self.output_shape[1] * self.output_shape[2]
        if load_pretrain and os.path.exists(self.save_path):
            self.load_checkpoint()
        elif load_pretrain:
            os.makedirs(self.save_path)

        self.use_mask = use_mask

    def load_checkpoint(self):
        if os.path.exists(
            os.path.join(self.save_path, "object_tactile_encoder_add.pt")
        ):
            self.load_state_dict(
                th.load(os.path.join(self.save_path, "object_tactile_encoder_add.pt"))
            )

    def save_checkpoint(self, file=None):
        print("[TactileObjectEncoder] Saving Checkpoint...")
        if file != None:
            th.save(self.state_dict(), file)
        else:
            th.save(
                self.state_dict(),
                self.save_path + "/" + "object_tactile_encoder_add.pt",
            )

    def apply_random_mask(self, input_tensor):
        batch_size, K, _ = input_tensor.shape

        # Generate a random mask with values 0 or 1 based on the 15% probability
        # We use a uniform distribution and compare it against the 0.15 threshold
        # A value of 1 in the mask means "keep", and 0 means "mask"
        # Note: th.rand generates values in the range [0, 1)
        mask = (th.rand(batch_size, K, 1, device=input_tensor.device) >= 0.1).float()

        # Apply the mask
        # Elements to be masked are multiplied by 0, effectively masking them
        masked_tensor = input_tensor * mask

        return masked_tensor

    def preprocess_observations(self, observations):
        observations = {
            key: th.nan_to_num(observations[key])
            for key in self.tactile_values
            + self.tactile_positions
            + self.object_projections
            + self.object_positions
        }
        tac_tensors = []
        for t_val, t_pos in zip(
            self.tactile_values, self.tactile_positions
        ):  # First we must pad our tactile pos vectors to the same length
            pad = max(0, self.tactile_dim - observations[t_pos].shape[1])
            # tac_tensors: (Value, Position)
            tac_tensors.append(
                (
                    observations[t_val],  # Value
                    F.pad(observations[t_pos], (0, pad), "constant", 0),
                )  # Position
            )
        tac_tensors = [
            (
                (val.unsqueeze(dim=1), pos.unsqueeze(dim=1))
                if len(val.shape) < 3
                else (val, pos)
            )
            for val, pos in tac_tensors
        ]  # Batch Seq.len Dim
        # Projections to the right dimension
        """ Tactiles """
        tac_proj_vectors = []
        for t_val_vec, t_pos_vec in tac_tensors:
            # Add the tactile val projection and the and pos projection
            tac_proj_vectors.append(
                (
                    self.tactile_val_projection(t_val_vec)
                    + self.tactile_pos_projection(t_pos_vec)
                ).to(self.device)
            )  # [(1, EMBED DIM), (1, EMBED DIM), ... ]
        tac_proj_vectors = th.cat(tac_proj_vectors, dim=1).to(
            self.device
        )  # (n, 4, EMBED DIM)
        """ Objects """
        obj_pos_tensor = observations["obj_location"]
        obj_val_tensor = observations["obj_velocity"]
        # Add the Object val projection and the and pos projection
        obj_proj_vectors = (
            self.object_val_projection(obj_val_tensor)
            + self.object_pos_projection(obj_pos_tensor)
        ).to(
            self.device
        )  # (7, EMBED DIM)

        state_output_tensor = th.zeros_like(tac_proj_vectors[:, 0, :]).unsqueeze(dim=1)

        trf_input_tensor = th.concatenate(
            [tac_proj_vectors, obj_proj_vectors, state_output_tensor], dim=1
        )

        if self.use_mask:
            trf_input_tensor = self.apply_random_mask(trf_input_tensor)

        trf_input_tensor = trf_input_tensor.to(self.device)

        return trf_input_tensor

    def forward(self, observations) -> th.Tensor:

        trf_input_tensor = self.preprocess_observations(observations)

        trf_output = self.trns_encoder(trf_input_tensor)
        trf_output = th.nan_to_num(trf_output)

        state_representation_tnsor = trf_output[:, -1, :]

        return state_representation_tnsor


# ------------------------------------------------------------------------- #


# ------------------------------------------------------------------------- #
# ----------------- ObjectTactileEncoder_TFREE ---------------------------- #
# ------------------------------------------------------------------------- #


class ObjectTactileEncoder_TFREE(nn.Module):
    """
    Transformer free Object Encoder
    """

    def __init__(
        self,
        observation_space,
        vec_encoding_size=64,
        tactile_dim=2 + 34 * 3,
        object_dim=6,
        num_transforme=512,
        device="cuda",
        load_pretrain=True,
    ):
        super(ObjectTactileEncoder_Additive, self).__init__()
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        self.observation_space = observation_space
        self.save_path = os.path.join(os.path.dirname(__file__), "pretrained_models")
        self.tactile_dim = tactile_dim
        self.object_dim = object_dim
        # TODO: Add an Angular Velocity to the Value vector
        self.tactile_values = [
            "palm_tactile",
            "finger_1_tactile",
            "finger_2_tactile",
            "finger_3_tactile",
        ]
        self.tactile_positions = [
            "palm_location",
            "finger_1_location",
            "finger_2_location",
            "finger_3_location",
        ]
        self.object_projections = ["obj_velocity"]
        self.object_positions = ["obj_location"]
        # Used to project our features to the right dimension
        self.tactile_val_projection = nn.Sequential(
            nn.Linear(
                self.observation_space["palm_tactile"].shape[0],
                self.vec_encoding_size * 2,
                self.device,
            ),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size * 2, self.device
            ),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
            nn.GELU(),
        )
        self.tactile_pos_projection = nn.Sequential(
            nn.Linear(tactile_dim, self.vec_encoding_size * 2, self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size * 2, self.device
            ),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
            nn.GELU(),
        )
        self.object_val_projection = nn.Sequential(
            nn.Linear(self.object_dim, self.vec_encoding_size * 2, self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size * 2, self.device
            ),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
            nn.GELU(),
        )
        self.object_pos_projection = nn.Sequential(
            nn.Linear(self.object_dim, self.vec_encoding_size * 2, self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size * 2, self.device
            ),
            nn.GELU(),
            nn.Linear(self.vec_encoding_size * 2, self.vec_encoding_size, self.device),
            nn.GELU(),
        )

        self.output_shape = (
            1,
            self.observation_space["obj_velocity"].shape[0] + len(self.tactile_values),
            self.vec_encoding_size,
        )
        self.flatten_size = self.output_shape[1] * self.output_shape[2]
        if load_pretrain and os.path.exists(self.save_path):
            self.load_checkpoint()
        elif load_pretrain:
            os.makedirs(self.save_path)

    def load_checkpoint(self):
        if os.path.exists(
            os.path.join(self.save_path, "object_tactile_encoder_tfree.pt")
        ):
            self.load_state_dict(
                th.load(os.path.join(self.save_path, "object_tactile_encoder_tfree.pt"))
            )

    def save_checkpoint(self, file=None):
        print("[TactileObjectEncoder] Saving Checkpoint...")
        if file != None:
            th.save(self.state_dict(), file)
        else:
            th.save(
                self.state_dict(),
                self.save_path + "/" + "object_tactile_encoder_tfree.pt",
            )

    def forward(self, observations) -> th.Tensor:
        observations = {
            key: th.nan_to_num(observations[key])
            for key in self.tactile_values
            + self.tactile_positions
            + self.object_projections
            + self.object_positions
        }
        tac_tensors = []
        for t_val, t_pos in zip(
            self.tactile_values, self.tactile_positions
        ):  # First we must pad our tactile pos vectors to the same length
            pad = max(0, self.tactile_dim - observations[t_pos].shape[1])
            # tac_tensors: (Value, Position)
            tac_tensors.append(
                (
                    observations[t_val],  # Value
                    F.pad(observations[t_pos], (0, pad), "constant", 0),
                )  # Position
            )
        tac_tensors = [
            (
                (val.unsqueeze(dim=1), pos.unsqueeze(dim=1))
                if len(val.shape) < 3
                else (val, pos)
            )
            for val, pos in tac_tensors
        ]  # Batch Seq.len Dim
        # Projections to the right dimension
        """ Tactiles """
        tac_proj_vectors = []
        for t_val_vec, t_pos_vec in tac_tensors:
            # Add the tactile val projection and the and pos projection
            tac_proj_vectors.append(
                (
                    self.tactile_val_projection(t_val_vec)
                    + self.tactile_pos_projection(t_pos_vec)
                ).to(self.device)
            )  # [(1, 128), (1, 128), ... ]
        tac_proj_vectors = th.cat(tac_proj_vectors, dim=1).to(
            self.device
        )  # (n, 4, 128)
        """ Objects """
        obj_pos_tensor = observations["obj_location"]
        obj_val_tensor = observations["obj_velocity"]
        # Add the Object val projection and the and pos projection
        obj_proj_vectors = (
            self.object_val_projection(obj_val_tensor)
            + self.object_pos_projection(obj_pos_tensor)
        ).to(
            self.device
        )  # (7, 128)
        trf_input_tensor = th.concatenate(
            [tac_proj_vectors, obj_proj_vectors], dim=1
        ).to(self.device)
        trf_output = self.trns_encoder(trf_input_tensor)
        return th.nan_to_num(trf_output)


# ------------------------------------------------------------------------- #
# ------------------------- ObjectTactileEncoder -------------------------- #
# ------------------------------------------------------------------------- #


class ObjectTactileEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        vec_encoding_size=64,
        tactile_dim=2 + 34 * 3,
        object_dim=6,
        decoder_size=512,
        device="cuda",
        load_pretrain=True,
    ):
        super(ObjectTactileEncoder, self).__init__()
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        self.observation_space = observation_space
        self.save_path = os.path.join(os.path.dirname(__file__), "pretrained_models")
        self.tactile_dim = tactile_dim
        self.object_dim = object_dim
        # TODO: Add an Angular Velocity to the Value vector
        self.tactile_values = [
            "palm_tactile",
            "finger_1_tactile",
            "finger_2_tactile",
            "finger_3_tactile",
        ]
        self.tactile_positions = [
            "palm_location",
            "finger_1_location",
            "finger_2_location",
            "finger_3_location",
        ]
        self.object_projections = ["obj_velocity"]
        self.object_positions = ["obj_location"]
        # Used to project our features to the right dimension
        self.tactile_val_projection = nn.Linear(
            self.observation_space["palm_tactile"].shape[0],
            self.vec_encoding_size // 2,
            self.device,
        )
        self.tactile_pos_projection = nn.Linear(
            self.tactile_dim, self.vec_encoding_size // 2, self.device
        )
        self.object_val_projection = nn.Linear(
            self.object_dim, self.vec_encoding_size // 2, self.device
        )
        self.object_pos_projection = nn.Linear(
            self.object_dim, self.vec_encoding_size // 2, self.device
        )
        transformer_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=self.vec_encoding_size,
            nhead=8,
            dim_feedforward=256,
            dropout=0.08,
            device=self.device,
        )
        # Stack 4 of these layers together
        self.trns_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=4,
        )
        self.output_shape = (
            1,
            self.observation_space["obj_velocity"].shape[0] + len(self.tactile_values),
            self.vec_encoding_size,
        )
        self.flatten_size = self.output_shape[1] * self.output_shape[2]
        if load_pretrain and os.path.exists(self.save_path):
            self.load_checkpoint()
        elif load_pretrain:
            os.makedirs(self.save_path)

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.save_path, "object_tactile_encoder.pt")):
            self.load_state_dict(
                th.load(os.path.join(self.save_path, "object_tactile_encoder.pt"))
            )
            print("\n" * 5 + "[TactileObjectEncoder] Loaded Checkpoint..." + "\n" * 5)

    def save_checkpoint(self, file=None):
        print("[TactileObjectEncoder] Saving Checkpoint...")
        if file != None:
            th.save(self.state_dict(), file)
        else:
            th.save(
                self.state_dict(), self.save_path + "/" + "object_tactile_encoder.pt"
            )

    def forward(self, observations) -> th.Tensor:
        observations = {
            key: th.nan_to_num(observations[key])
            for key in self.tactile_values
            + self.tactile_positions
            + self.object_projections
            + self.object_positions
        }
        tac_tensors = []
        for t_val, t_pos in zip(
            self.tactile_values, self.tactile_positions
        ):  # First we must pad our tactile pos vectors to the same length
            pad = max(0, self.tactile_dim - observations[t_pos].shape[1])
            # tac_tensors: (Value, Position)
            tac_tensors.append(
                (
                    observations[t_val],  # Value
                    F.pad(observations[t_pos], (0, pad), "constant", 0),
                )  # Position
            )
        tac_tensors = [
            (
                (val.unsqueeze(dim=1), pos.unsqueeze(dim=1))
                if len(val.shape) < 3
                else (val, pos)
            )
            for val, pos in tac_tensors
        ]  # Batch Seq.len Dim
        # Projections to the right dimension
        """ Tactiles """
        tac_proj_vectors = []
        for t_val_vec, t_pos_vec in tac_tensors:
            # Cat the tactile val projection and the and pos projection
            tac_proj_vectors.append(
                th.concatenate(
                    [
                        self.tactile_val_projection(t_val_vec),
                        self.tactile_pos_projection(t_pos_vec),
                    ],
                    dim=2,
                )
            )  # [(1, 128), (1, 128), ... ]
        tac_proj_vectors = th.cat(tac_proj_vectors, dim=1).to(
            self.device
        )  # (n, 4, 128)
        """ Objects """
        obj_pos_tensor = observations["obj_location"]
        obj_val_tensor = observations["obj_velocity"]
        obj_proj_vectors = th.concatenate(
            [
                self.object_val_projection(obj_val_tensor),
                self.object_pos_projection(obj_pos_tensor),
            ],
            dim=2,
        ).to(
            self.device
        )  # (7, 128)
        trf_input_tensor = th.concatenate(
            [tac_proj_vectors, obj_proj_vectors], dim=1
        ).to(self.device)
        trf_output = self.trns_encoder(trf_input_tensor)
        return th.nan_to_num(trf_output)


# ------------------------------------------------------------------------- #
# ------------------ MultiModalTransformerFeatureEncoder ------------------ #
# ------------------------------------------------------------------------- #


class MultiModalTransformerFeatureEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        vec_encoding_size=64,
        tactile_dim=2 + 34 * 3,
        object_dim=6,
        decoder_size=512,
        device="cuda",
    ):
        super(MultiModalTransformerFeatureEncoder, self).__init__(
            observation_space, features_dim=1
        )
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        self.observation_space = observation_space
        self.join_keys = ["state_attrib"]
        self.pos_encoder = ObjectTactileEncoder_Additive(
            observation_space,
            vec_encoding_size=8,
            load_pretrain=False,
        )
        flatten_size = (
            self.pos_encoder.output_shape[1] * self.pos_encoder.output_shape[2]
            + self.observation_space["state_attrib"].shape[0]
        )
        self.decoder_size = decoder_size
        self.decoder = nn.Sequential(
            nn.Linear(flatten_size, self.decoder_size, device=device),
            nn.GELU(),
            # nn.Dropout(p=0.0),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.GELU(),
            # nn.Dropout(p=0.0),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.GELU(),
            # nn.LayerNorm(self.decoder_size),
            # nn.Dropout(p=0.0),
        )  # Triple Layer Decoder
        self._features_dim = self.decoder_size
        self.output_dim = self.decoder_size
        self.__loaded_trf = False

    def forward(self, observations):
        if not self.__loaded_trf:
            # self.pos_encoder.load_checkpoint()
            self.__loaded_trf = True
            self.pos_encoder.eval()
        trf_output = self.pos_encoder(observations).detach()
        # trf_output = self.pos_encoder(observations) # Train the TRF
        state_attrib = th.nan_to_num(observations["state_attrib"])
        decoder_input = th.concatenate(
            [trf_output.view((trf_output.shape[0], -1)), state_attrib], dim=1
        ).to(self.device)
        features_output = self.decoder(decoder_input)
        return th.nan_to_num(features_output)


# ------------------------------------------------------------------------- #
# -------------------------- Residual Networks ---------------------------- #
# ------------------------------------------------------------------------- #

class ResidualLayer1D(nn.Module):
    def __init__(self, feature_dim: int, embed_dim=512, dropout_p=0.1):
        super(ResidualLayer1D, self).__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(
            feature_dim,
            embed_dim,
        )
        self.fc2 = nn.Linear(
            embed_dim,
            feature_dim,
        )
        self.n = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.gelu(self.fc2(out))
        out = self.n(x + out)
        return out


class ResidualBlocks1D(nn.Module):
    def __init__(self, feature_dim: int, num_blocks: int, embed_dim=512, dropout_p=0.1):
        super(ResidualBlocks1D, self).__init__()
        self.feature_dim = feature_dim
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.layers = nn.Sequential(
            *[
                ResidualLayer1D(feature_dim, embed_dim, dropout_p=dropout_p)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        out = self.layers(x)
        return out

# ------------------------------------------------------------------------- #
# ---------------- Sinosoudal Positional Encoding ------------------------- #
# ------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=16):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_length = max_length

        # Create a matrix of shape (max_length, d_model) where each row represents the positional encoding of a position in the sequence
        position = th.arange(max_length).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = th.zeros(max_length, d_model)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)

        # pe does not require gradient computation)
        self.register_buffer("pe", pe)

    def forward(self):
        # x is a tensor of shape (batch_size, sequence_length, d_model)
        x = self.pe[: self.max_length, :].unsqueeze(0)
        return x


# ------------------------------------------------------------------------- #
# ----------------- TemporalObjectTactileEncoder_Additive ----------------- #
# ------------------------------------------------------------------------- #


class TemporalObjectTactileEncoder_Additive(nn.Module):
    """Uses Time, and N number of scene objects to produce a temporally and spatially motivated vector embedding of the scene."""

    def __init__(
        self,
        observation_space,
        vec_encoding_size=1024,
        tactile_dim=2 + 34 * 3,
        object_dim=6,
        state_attrib_dim=45,
        num_tsf_layer=4,
        t_dim_size=3,
        device="cuda",
        load_pretrain=True,
        use_mask=False,
    ):
        super(TemporalObjectTactileEncoder_Additive, self).__init__()
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        self.observation_space = observation_space
        print("Obs Space:", observation_space)
        self.save_path = os.path.join(os.path.dirname(__file__), "pretrained_models")
        self.tactile_dim = tactile_dim
        self.object_dim = object_dim
        self.state_attrib_dim = state_attrib_dim

        # TODO: Add an Angular Velocity to the Value vector
        self.tactile_values = [
            "palm_tactile",
            "finger_1_tactile",
            "finger_2_tactile",
            "finger_3_tactile",
        ]
        self.tactile_positions = [
            "palm_location",
            "finger_1_location",
            "finger_2_location",
            "finger_3_location",
        ]
        self.object_projections = ["obj_velocity"]
        self.object_positions = ["obj_location"]

        self.t_dim_size = t_dim_size

        # Used to project our features to the right dimension
        self.tactile_val_projection = nn.Sequential(
            nn.Linear(
                self.observation_space["palm_tactile"].shape[1],
                self.vec_encoding_size * 2,
                device=self.device,
            ),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size, device=self.device
            ),
        )
        self.tactile_pos_projection = nn.Sequential(
            nn.Linear(tactile_dim, self.vec_encoding_size * 2, device=self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size, device=self.device
            ),
        )
        self.temporal_projection = PositionalEncoding(
            self.vec_encoding_size, self.t_dim_size
        )
        self.object_val_projection = nn.Sequential(
            nn.Linear(self.object_dim, self.vec_encoding_size * 2, device=self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size, device=self.device
            ),
        )
        self.object_pos_projection = nn.Sequential(
            nn.Linear(self.object_dim, self.vec_encoding_size * 2, device=self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size, device=self.device
            ),
        )
        
        self.state_attrib_projection = nn.Sequential(
            nn.Linear(self.state_attrib_dim, self.vec_encoding_size * 2, device=self.device),
            nn.GELU(),
            nn.Linear(
                self.vec_encoding_size * 2, self.vec_encoding_size, device=self.device
            ),
        )

        # Transformer layer used for encoding the objects
        transformer_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=self.vec_encoding_size,
            nhead=4,
            dim_feedforward=256,  # TODO: make modular
            dropout=0.08,
            device=self.device,
        )
        self.trns_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_tsf_layer,
        )
        self.output_shape = (
            1,
            self.observation_space["obj_velocity"].shape[0] + len(self.tactile_values),
            self.vec_encoding_size,
        )
        self.flatten_size = self.output_shape[1] * self.output_shape[2]
        if load_pretrain and os.path.exists(self.save_path):
            self.load_checkpoint()
        elif load_pretrain:
            os.makedirs(self.save_path)

        self.use_mask = use_mask

    def load_checkpoint(self, file=None):
        file = (
            file
            if file is not None
            else os.path.join(self.save_path, "object_tactile_encoder_add.pt")
        )

        if os.path.exists(file):
            self.load_state_dict(th.load(file))

    def save_checkpoint(self, file=None):
        print("[TemporalObjectTactileEncoder_Additive] Saving Checkpoint...")
        if file != None:
            th.save(self.state_dict(), file)
        else:
            th.save(
                self.state_dict(),
                self.save_path + "/" + "object_tactile_encoder_add.pt",
            )

    def apply_random_mask(self, input_tensor):
        batch_size, K, _ = input_tensor.shape

        # Generate a random mask with values 0 or 1 based on the 15% probability
        # We use a uniform distribution and compare it against the 0.15 threshold
        # A value of 1 in the mask means "keep", and 0 means "mask"
        # Note: th.rand generates values in the range [0, 1)
        mask = (th.rand(batch_size, K, 1, device=input_tensor.device) >= 0.11).float()

        # Apply the mask
        # Elements to be masked are multiplied by 0, effectively masking them
        masked_tensor = input_tensor * mask

        return masked_tensor

    def normalize_observations(self, observations):
        """Normalizes the input data to avoid NAN numerics"""
        observations = {
            k: th.nan_to_num(
                th.tensor(observations[k], dtype=th.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for k in observations.keys()
        }

        return observations

    def preprocess_observations(self, observations):
        """Input Dict Tensor for the Transformer:
        finger_1_location: (batch_size, T, embed_dim*)
        finger_2_location: (batch_size, T, embed_dim*)
        finger_3_location: (batch_size, T, embed_dim*)
        palm_location: (batch_size, T, embed_dim*)
        finger_1_tactile: (batch_size, T, embed_dim*)
        finger_2_tactile: (batch_size, T, embed_dim*)
        finger_3_tactile: (batch_size, T, embed_dim*)
        palm_tactile : (batch_size, T, embed_dim*)
        obj_location: (batch_size, T, OBJECTS, embed_dim*)
        obj_velocity: (batch_size, T, OBJECTS, embed_dim*)
        state_attrib: (batch_size, embed_dim*) 
        state_embedding: zeros_like(batch_size, T, OBJECTS, embed_dim*)
        """
        # with open("./tmp_shape_buffer", "w") as f:
        #     f.write(json.dumps(
        #         {k: v.shape for k, v in observations.items()}
        #     ))
        # Normalize the observations (Nan -> 0.0f):
        observations = self.normalize_observations(observations)

        ps_enc = self.temporal_projection.forward().to(self.device)

        tac_tensors = []

        for t_val, t_pos in zip(
            self.tactile_values, self.tactile_positions
        ):  # First we must pad our tactile pos vectors to the same length
            # print(f"tactile: {self.tactile_dim}\nobs: {observations[t_pos].shape}")
            pad = max(0, self.tactile_dim - observations[t_pos].shape[2])
            # tac_tensors: (Value, Position)
            tac_tensors.append(
                (
                    observations[t_val].to(self.device),  # Value
                    F.pad(observations[t_pos], (0, pad), "constant", 0).to(self.device),
                )  # Position
            )
        tac_tensors = [
            (
                (val.unsqueeze(dim=1), spatial_pos.unsqueeze(dim=1))
                if len(val.shape) < 4
                else (val, spatial_pos)
            )
            for val, spatial_pos in tac_tensors
        ]  # Batch Seq.len Dim
        # Projections to the right dimension

        """ Tactiles """
        tac_proj_vectors = []
        for t_val_vec, t_pos_vec in tac_tensors:
            # Add the tactile val projection and the and pos projection
            tac_proj_vectors.append(
                (
                    self.tactile_val_projection(t_val_vec)
                    + self.tactile_pos_projection(t_pos_vec)
                ).to(self.device)
            )  # [(1, EMBED DIM), (1, EMBED DIM), ... ]
        tac_proj_vectors = th.cat(tac_proj_vectors, dim=1).to(
            self.device
        )  # (n, 4, T, EMBED DIM)

        # Apply positional encodings and reshape to object dimension

        tac_proj_vectors = ps_enc + tac_proj_vectors

        B, N, T, embed_dim = tac_proj_vectors.shape

        tac_proj_vectors = tac_proj_vectors.reshape(B, N * T, embed_dim)

        """ Objects """
        obj_pos_tensor = observations["obj_location"]
        obj_val_tensor = observations["obj_velocity"]

        B, T, N, _ = obj_val_tensor.shape
        # Apply positional encodings to objects
        # Add the Object val projection and the and pos projection

        # temporal_pos_enc = ps_enc.unsqueeze(2).expand(B, T, N, embed_dim)
        # val_enc = self.object_val_projection(obj_val_tensor)
        # spat_pos_enc = self.object_pos_projection(obj_pos_tensor)

        # obj_proj_vectors = (temporal_pos_enc + val_enc + spat_pos_enc).to(self.device)

        # Project Val, Project Spatial Pos, Expand Temporal Pos;
        # Obj Encoding < - temporal enc + val enc + spatial enc
        
        ps_enc = ps_enc[:, 0:T]
        
        ps_enc = ps_enc.unsqueeze(2).expand(B, T, N, embed_dim).to(self.device)

        obj_proj_vectors = (
            ps_enc
            + self.object_val_projection(obj_val_tensor)
            + self.object_pos_projection(obj_pos_tensor)
        ).to(
            self.device
        )  # (B, T, 7, EMBED DIM)
        
        state_attributes_tensor = self.state_attrib_projection(observations["state_attrib"].to(self.device)).unsqueeze(dim=1).to(self.device)

        B, T, N, embed_dim = obj_proj_vectors.shape
        obj_proj_vectors = obj_proj_vectors.reshape(B, N * T, embed_dim).to(self.device)

        state_output_tensor = (
            th.zeros_like(tac_proj_vectors[:, 0, :]).unsqueeze(dim=1).to(self.device)
        )

        trf_input_tensor = th.concatenate(
            [tac_proj_vectors, obj_proj_vectors, state_attributes_tensor, state_output_tensor], dim=1
        )

        if self.use_mask:
            trf_input_tensor = self.apply_random_mask(trf_input_tensor)

        trf_input_tensor = trf_input_tensor.to(self.device)

        return trf_input_tensor

    def forward(self, observations) -> th.Tensor:

        trf_input_tensor = self.preprocess_observations(observations)

        trf_output = self.trns_encoder(trf_input_tensor)
        trf_output = th.nan_to_num(trf_output)

        state_representation_tnsor = trf_output[:, -1, :]

        return state_representation_tnsor

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad_ = False


# ------------------------------------------------------------------------- #
# ---------------------- DynamixModel ------------------------------------- #
# ------------------------------------------------------------------------- #

DYNAMIX_OUTPUT_SIZES_DICT = {
    "finger_1_location": 27,
    "finger_2_location": 27,
    "finger_3_location": 27,
    "palm_location": 27,
    "finger_1_tactile": 9,
    "finger_2_tactile": 9,
    "finger_3_tactile": 9,
    "palm_tactile": 9,
    "obj_location": 42,
    "hand_config": 7,
    "obj_count": 5,
    "progress_bar": 1,
    "reward": 1,
}

PRED_OUTPUT_SIZES_DICT = {
    "action": 5,
    "reward": 1,
}


class DynamixModel(nn.Module):
    """Given (State, Action): Predict (N_state)"""

    def __init__(
        self,
        embed_dim_high=1024,
        embed_dim_low=256,
        device="cuda",
        dropout_prob=0.05,
        num_tsf_layer=4,
        num_residual_blocks=4,
        vec_encoding_size=8,
        use_mask=False,
        encoder: Optional[TemporalObjectTactileEncoder_Additive] = None,
    ):
        super(DynamixModel, self).__init__()
        self.device = device
        self.to(self.device)

        self.object_encoder = (
            encoder
            if encoder
            else TemporalObjectTactileEncoder_Additive(
                observation_space=base_observation_space,
                vec_encoding_size=vec_encoding_size,
                t_dim_size=T_buffer,
                load_pretrain=False,
                num_tsf_layer=num_tsf_layer,
                use_mask=use_mask,
            )
        )

        self.act_size = 5
        self.cat_size = self.object_encoder.flatten_size + 5  # action shape
        self.embed_dim_high = embed_dim_high
        self.embed_dim_low = embed_dim_low
        self.activation = nn.GELU
        self.join_keys = ["action"]

        self.delta_state_network = nn.Sequential(
            # CAST
            ResidualBlocks1D(
                feature_dim=vec_encoding_size + 5,
                num_blocks=num_residual_blocks,
                embed_dim=embed_dim_high,
            ),
            # GATING
            nn.Linear(vec_encoding_size + 5, vec_encoding_size),
        )

        modules = {}
        for k, size in DYNAMIX_OUTPUT_SIZES_DICT.items():
            modules[k] = nn.Sequential(
                nn.Linear(vec_encoding_size, self.embed_dim_high),
                self.activation(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.embed_dim_high, self.embed_dim_low),
                self.activation(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.embed_dim_low, size),
            )

        self.networks = nn.ModuleDict(modules)


    def forward(self, obs):
        tac_encoding = self.object_encoder(obs)
        tac_encoding = tac_encoding.view((tac_encoding.shape[0], -1))
        cat_tensor = th.concatenate([tac_encoding, obs["action"]], dim=1).to(
            self.device
        )

        state_delta = self.delta_state_network(cat_tensor)

        # Intuition:
        # Or critiq model is trying to find a relationship between one state embedding and another state embedding to produce an action and a reward
        # Here we are trying to find the delta between our original state and the new state
        # caused by the action such that we can predict it
        embed_tnsor = state_delta + tac_encoding

        # Benifets: I can take a state at T[0] and pass S[0] through our network to
        # obtain an embedding e_0
        # With e_0 and A_0 I can obtain e_1: use e_0 CONCAT A_0 with delta state network
        # with e_1, I can obtain e_2 using A_1 via a similar process.
        # Hence, this modeling scheme is powerful if future trajectories can be reliably obtained in the embedding space of S

        # TODO: Test Utitlity
        # embed_tnsor = F.normalize(F.tanh(state_delta) + F.sigmoid(tac_encoding), eps=1e-7) #TODO: Test Utitlity
        # embed_tnsor = F.tanh((state_delta + tac_encoding ) * (1 / self.object_encoder.vec_encoding_size**0.5))
        # embed_tnsor = state_delta

        logit_map = {k: net(embed_tnsor) for k, net in self.networks.items()}
        return logit_map

    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.object_encoder.save_path, "dynamix.pt")

        checkpoint = th.load(path, map_location=self.device)
        self.object_encoder.load_state_dict(checkpoint["object_encoder"])
        self.delta_state_network.load_state_dict(checkpoint["delta_state_network"])
        self.networks.load_state_dict(checkpoint["networks"])

    def save_checkpoint(self, file=None):
        print("[DynamixModel] Saving Checkpoint...")
        if file is None:
            file = os.path.join(self.object_encoder.save_path, "dynamix.pt")

        save_dict = {
            "object_encoder": self.object_encoder.state_dict(),
            "delta_state_network": self.delta_state_network.state_dict(),
            "networks": self.networks.state_dict(),
        }
        th.save(save_dict, file)


class CritiqModel(nn.Module):
    """Given (State, N_state): Predict (Action, Reward)"""

    def __init__(
        self,
        embed_dim_high=1024,
        embed_dim_low=256,
        device="cuda",
        dropout_prob=0.05,
        num_tsf_layer=4,
        num_residual_blocks=4,
        vec_encoding_size=8,
        use_mask=False,
        encoder: Optional[TemporalObjectTactileEncoder_Additive] = None,
    ):
        super(CritiqModel, self).__init__()
        self.device = device
        self.to(self.device)

        self.object_encoder = (
            encoder
            if encoder
            else TemporalObjectTactileEncoder_Additive(
                observation_space=base_observation_space,
                vec_encoding_size=vec_encoding_size,
                t_dim_size=T_buffer,
                load_pretrain=False,
                num_tsf_layer=num_tsf_layer,
                use_mask=use_mask,
            )
        )

        self.act_size = 5
        self.cat_size = self.object_encoder.flatten_size + 5  # action shape
        self.embed_dim_high = embed_dim_high
        self.embed_dim_low = embed_dim_low
        self.activation = nn.GELU
        self.join_keys = ["action"]

        self.predictor = nn.Sequential(
            # Recieves Delta State
            ResidualBlocks1D(
                feature_dim=vec_encoding_size * 2,
                num_blocks=5,
                embed_dim=embed_dim_high,
            ),
            nn.Linear(vec_encoding_size * 2, vec_encoding_size),
            self.activation(),
            nn.Dropout(p=dropout_prob),
        )

        # Time estimation:
        self.time_estimator = nn.Sequential(
            # Recieves Delta State
            nn.Linear(vec_encoding_size, vec_encoding_size),
            self.activation(),
            nn.Linear(vec_encoding_size, 1),
        )

        modules = {}
        for k, size in PRED_OUTPUT_SIZES_DICT.items():
            modules[k] = nn.Sequential(
                nn.Linear(vec_encoding_size, self.embed_dim_high),
                self.activation(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.embed_dim_high, self.embed_dim_low),
                self.activation(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(self.embed_dim_low, size),
            )

        self.networks = nn.ModuleDict(modules)

    def forward(self, state, n_state):
        # Feed forward S[T] and S[T+1]
        embed_state = self.object_encoder(state)
        n_embed_state = self.object_encoder(n_state)
        embed_state = embed_state.view((embed_state.shape[0], -1))
        n_embed_state = n_embed_state.view((n_embed_state.shape[0], -1))

        # We obtain a delta
        delta_state = n_embed_state - embed_state
        # delta_estimation = (1 / self.time_estimator(embed_state)) * self.predictor(delta_state)
        delta_concat = th.cat([embed_state, delta_state], dim=1).to(self.device)

        delta_estimation = self.predictor(delta_concat)

        logit_map = {k: net(delta_estimation) for k, net in self.networks.items()}

        return logit_map

    def load_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.object_encoder.save_path, "critiq.pt")

        checkpoint = th.load(path, map_location=self.device)
        self.object_encoder.load_state_dict(checkpoint["object_encoder"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.time_estimator.load_state_dict(checkpoint["time_estimator"])
        self.networks.load_state_dict(checkpoint["networks"])

    def save_checkpoint(self, file=None):
        print("[CritiqModel] Saving Checkpoint...")
        if file is None:
            file = os.path.join(self.object_encoder.save_path, "critiq.pt")

        save_dict = {
            "object_encoder": self.object_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "time_estimator": self.time_estimator.state_dict(),
            "networks": self.networks.state_dict(),
        }
        th.save(save_dict, file)


def make_dynamix_and_predictor(
    model_args: Dict[str, Any]
) -> Tuple[DynamixModel, CritiqModel]:
    # Extract common arguments
    embed_dim_high = model_args.get("embed_dim_high", 1024)
    embed_dim_low = model_args.get("embed_dim_low", 256)
    device = model_args.get("device", "cuda")
    dropout_prob = model_args.get("dropout_prob", 0.05)
    num_tsf_layer = model_args.get("num_tsf_layer", 4)
    num_residual_blocks = model_args.get("num_residual_blocks", 4)
    vec_encoding_size = model_args.get("vec_encoding_size", 8)
    use_mask = model_args.get("use_mask", False)

    # Create the shared encoder
    encoder = TemporalObjectTactileEncoder_Additive(
        observation_space=model_args.get("state_space"),
        vec_encoding_size=vec_encoding_size,
        t_dim_size=model_args.get("T_buffer"),
        load_pretrain=False,
        num_tsf_layer=num_tsf_layer,
        use_mask=use_mask,
    )

    # Create DynamixModel
    dynamix_model = DynamixModel(
        embed_dim_high=embed_dim_high,
        embed_dim_low=embed_dim_low,
        device=device,
        dropout_prob=dropout_prob,
        num_tsf_layer=num_tsf_layer,
        num_residual_blocks=num_residual_blocks,
        vec_encoding_size=vec_encoding_size,
        use_mask=use_mask,
        encoder=encoder,
    )

    # Create CritiqModel
    critiq_model = CritiqModel(
        embed_dim_high=embed_dim_high,
        embed_dim_low=embed_dim_low,
        device=device,
        dropout_prob=dropout_prob,
        num_tsf_layer=num_tsf_layer,
        num_residual_blocks=num_residual_blocks,
        vec_encoding_size=vec_encoding_size,
        use_mask=use_mask,
        encoder=encoder,
    )

    return dynamix_model, critiq_model