## Initialization:: Use env stbl3 or raylib
import pickle
import os
import sys
import numpy as np

from gymnasium.spaces import Box
from control_dropping_rpal.RL.control_dropping_env import BerrettHandGym, T_buffer
from typing import Dict, List, Optional, Tuple, Any
from pytorch_lightning.strategies import DDPStrategy


from math import inf, radians, degrees
from stable_baselines3 import PPO, A2C

import matplotlib.pyplot as plt

# Torch
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from torchmetrics import Accuracy

## Dataloader for Model Training
from stable_baselines3.common.utils import obs_as_tensor
from torch.utils.data import DataLoader, random_split

### PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

## Model for predictions
from control_dropping_rpal.RL.Networks.ExtractorNetworks import (
    DYNAMIX_OUTPUT_SIZES_DICT,
    TemporalObjectTactileEncoder_Additive,
    DynamixModel,
    CritiqModel,
)

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import TQDMProgressBar

class ProgressBar(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # Log custom metrics here
        self.main_progress_bar.set_postfix({
            'loss': outputs['loss'].item(),
            'accuracy': outputs['accuracy'].item()
        })

class LoggingCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:  # Log every 100 batches
            print(f"Batch {batch_idx}/{len(trainer.train_dataloader)}, Loss: {outputs['loss'].item():.4f}")


torch.set_float32_matmul_precision('medium')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

BATCH_SIZE = 512
DATA_SAVE_PATH = os.path.join(os.getcwd(), "..", "Data_Collection")
MODEL_PATH = "/media/rpal/Drive_10TB/John/Control Drop/control_dropping/src/RL/Training/Checkpoints/TransformerFeatureEncoder/Expert_rl_5000_steps.zip"
PATH_DYNAMIX = (
    "/media/rpal/Drive_10TB/John/Control Drop/Data_Collection/Time_Dependent_Samples_4/"
)
PATH_CRITIQ = "/media/rpal/Drive_10TB/John/Control Drop/Data_Collection/Action_Pred_Time_Dependent_Samples_4/"

GAMMA = 0.5

state_space = {
    "palm_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    "finger_1_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    "finger_2_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    "finger_3_tactile": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            24,
        ),
    ),  # Value
    # 'tactile_pos': Box(low= -inf, high= inf, shape=(378, )), # Position
    "finger_1_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 34 * 3,
        ),
    ),  # Joint pos [Theta_1, Theta_2] + [xyz*34]
    "finger_2_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 34 * 3,
        ),
    ),
    "finger_3_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 34 * 3,
        ),
    ),
    "palm_location": Box(
        low=-inf,
        high=inf,
        shape=(
            T_buffer,
            2 + 24 * 3,
        ),
    ),
    "obj_location": Box(low=-inf, high=inf, shape=(T_buffer, 7, 6)),  # Position
    "obj_velocity": Box(
        low=-inf, high=inf, shape=(T_buffer, 7, 6)
    ),  # Value, Concat with angular velocity
    "state_attrib": Box(
        low=-inf, high=inf, shape=(45,)
    ),  # Ball Cnt, Progress, Prev.Actions, hand_cfg, hand_trq (44)
}

start_lr = 0.001
end_lr = 0.00005
factor = 0.999

# removes inhomogeneous elements from a dataset:
def remove_inhomogeneous_elements(aggregated_data):
    removal_idxs = set()

    # Detect inhomogeneous elements
    for data in aggregated_data:
        for key, arr_list in data.items():
            shapes = [arr.shape for arr in arr_list]
            most_common_shape = max(set(shapes), key=shapes.count)

            for idx, shape in enumerate(shapes):
                if shape != most_common_shape:
                    removal_idxs.add(idx)

    # Remove inhomogeneous elements
    for data in aggregated_data:
        for key in data:
            data[key] = [
                arr for idx, arr in enumerate(data[key]) if idx not in removal_idxs
            ]

    return aggregated_data, list(removal_idxs)


def mult(arr, idx=0):
    if idx == len(arr):
        return 1
    return arr[idx] * mult(arr, idx + 1)


def save_data(
    data,
    file_path,
):
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    # Extract data from each data point and create numpy arrays
    state_list = [state[0].copy() for state in data]
    pred_state_list = [state[1].copy() for state in data]

    chunk_data = {
        "states": state_list,
        "pred_states": pred_state_list,
    }
    import re

    i = (
        sorted([int(re.sub(r"\D", "", s)) for s in os.listdir(file_path)] + [-1])[-1]
        + 1
    )
    chunk_file_path = os.path.join(file_path, f"Actions_data_{i}.pkl")

    with open(chunk_file_path, "wb") as f:
        pickle.dump(chunk_data, f)
    print("Saved to", chunk_file_path)


def mult(shape):
    """Utility function to compute the product of elements in a shape tuple."""
    product = 1
    for dim in shape:
        product *= dim
    return product


def load_data_node_format(file_path, target_range=4):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # Load data from pickle files into a list of states
    states = []
    new_keys = {f"finger_{i}_tactile": f"finger_{i+1}_tactile" for i in range(3)}
    new_keys.update(
        {f"finger_{i}_locaction": f"finger_{i+1}_location" for i in range(3)}
    )
    new_keys.update({"palm_locaction": "palm_location"})

    for f_path in [
        f for f in os.listdir(file_path) if "Actions_data" in f
    ]:  # For each episode
        with open(os.path.join(file_path, f_path), "rb") as file:
            data = pickle.load(file)

        temp_states = []
        for state, pred in zip(
            data["states"], data["pred_states"]
        ):  # For each state[t-1], state[t] pair
            if not isinstance(pred, dict):
                pred = {key: pred[key] for key in pred.dtype.names}
            if "finger_0_tactile" in pred:
                # Update keys:
                pred = {new_keys.get(k, k): v for k, v in pred.items()}
            temp_states.append((state, pred))

        sts = [s[0] for s in temp_states]
        preds = [s[1] for s in temp_states]
        # for i in range(len(preds) - 2, -1, -1):
        #     preds[i]["reward"] += GAMMA * preds[i + 1]["reward"]

        states += [(s, p) for s, p in zip(sts, preds)]

    # Aggregate data by key
    aggregated_data = [{}, {}]
    for d in states:
        for key, value in d[0].items():
            if key not in aggregated_data[0]:
                aggregated_data[0][key] = []
            aggregated_data[0][key].append(value)
        for key, value in d[1].items():
            if key not in aggregated_data[1]:
                aggregated_data[1][key] = []
            aggregated_data[1][key].append(value)

    # Perform surgery: Some values are input wrong
    reward_data = []
    for i in range(len(aggregated_data[1]["reward"])):
        reward_data.append(
            np.array([aggregated_data[1]["reward"][i]])
        )  # Convert to list
    aggregated_data[1]["reward"] = np.array(reward_data)

    # Flatten and format:
    aggregated_data = [
        {
            k: [np.array(v).squeeze() for v in value]
            for k, value in aggregated_data[0].items()
        },
        {
            k: [np.array(v).flatten() for v in value]
            for k, value in aggregated_data[1].items()
        },
    ]

    stacked_values = np.vstack(aggregated_data[0]["action"])
    stacked_values /= 3.5
    aggregated_data[0]["action"] = [
        a.squeeze() for a in np.split(stacked_values, len(aggregated_data[0]["action"]))
    ]

    # Find inhomogeneous data:
    removal_idxs = set()

    for data in aggregated_data:
        for key, arr_list in data.items():
            shapes = [arr.shape for arr in arr_list]
            most_common_shape = max(
                set(shapes), key=shapes.count
            )  # Detect inhomogeneous elements

            for idx, shape in enumerate(shapes):
                if shape != most_common_shape:
                    removal_idxs.add(idx)

    # Remove inhomogeneous elements:
    for data in aggregated_data:
        for key in data:
            data[key] = [
                arr for idx, arr in enumerate(data[key]) if idx not in removal_idxs
            ]

    print("Inhomogeneous data:", removal_idxs)

    for data in aggregated_data:
        for key, arr in data.items():
            arr = np.where(np.isnan(arr), 0, arr)
            arr = np.where(arr < -(4**2), 0, arr)
            arr = np.where(arr > 4**2, 0, arr)
            data[key] = arr

    print("Shape Min Maxs:\n")
    for key in aggregated_data[0].keys():
        print(
            f"{key}: {aggregated_data[0][key][0].shape}, {np.min(aggregated_data[0][key])}, {np.max(aggregated_data[0][key])}"
        )
    for key in aggregated_data[1].keys():
        print(
            f"{key}: {np.array(aggregated_data[1][key][0]).shape}, {np.min(aggregated_data[1][key])}, {np.max(aggregated_data[1][key])}"
        )

    return aggregated_data


def load_data_state_format(file_path, target_range=4):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # Load data from pickle files into a list of states
    states = []
    new_keys = {f"finger_{i}_tactile": f"finger_{i+1}_tactile" for i in range(3)}
    new_keys.update(
        {f"finger_{i}_locaction": f"finger_{i+1}_location" for i in range(3)}
    )
    new_keys.update({"palm_locaction": "palm_location"})

    for f_path in [
        f for f in os.listdir(file_path) if "Actions_data" in f
    ]:  # For each episode
        with open(os.path.join(file_path, f_path), "rb") as file:
            data = pickle.load(file)

        temp_states = []
        for state, pred in zip(
            data["states"], data["pred_states"]
        ):  # For each state[t-1], state[t] pair
            if not isinstance(pred, dict):
                pred = {key: pred[key] for key in pred.dtype.names}
            if "finger_0_tactile" in pred:
                # Update keys:
                pred = {new_keys.get(k, k): v for k, v in pred.items()}

            # We collect prediction data for an entire state, we are going to restrict the prediction to a T_buffer of 1 so that it encapsulates only S[T] and enables better representations during the forward pass:
            for key in [
                k
                for k in pred.keys()
                if k != "reward"
                and T_buffer in pred[k].shape
                and len(pred[k].shape) > 1
            ]:
                arr = pred[key]
                pred[key] = arr[-1:, :]

            action = state["action"] if "action" in state else pred["action"]
            reward = np.array(
                [(pred["reward"] if "reward" in pred else state["reward"])]
            )

            state = {k: v for k, v in state.items() if k not in ("action", "reward")}
            n_state = {k: v for k, v in pred.items() if k not in ("action", "reward")}
            pred = {"action": action, "reward": reward}

            temp_states.append((state, n_state, pred))

        states += [(s, n_s, p) for s, n_s, p in temp_states]

    # Aggregate data by key
    aggregated_data = [{}, {}, {}]
    for d in states:
        # S[T - 1]
        for key, value in d[0].items():
            if key not in aggregated_data[0]:
                aggregated_data[0][key] = []
            aggregated_data[0][key].append(value)
        # S[T]
        for key, value in d[1].items():
            if key not in aggregated_data[1]:
                aggregated_data[1][key] = []
            aggregated_data[1][key].append(value)

        # Prediction:
        for key, value in d[2].items():
            if key not in aggregated_data[2]:
                aggregated_data[2][key] = []
            aggregated_data[2][key].append(value)

    # Flatten and format:
    aggregated_data = [
        {k: [np.array(v) for v in value] for k, value in aggregated_data[0].items()},
        {k: [np.array(v) for v in value] for k, value in aggregated_data[1].items()},
        {
            k: (
                [np.array(v).flatten() for v in value]
            )  # if k != "reward" else [np.array(v) for v in value])
            for k, value in aggregated_data[2].items()
        },
    ]

    stacked_values = np.vstack(aggregated_data[2]["action"])
    stacked_values *= 1 / 3.5
    aggregated_data[2]["action"] = [
        a.squeeze() for a in np.split(stacked_values, len(aggregated_data[2]["action"]))
    ]

    for data in aggregated_data:
        for key, arr in data.items():
            arr = np.where(np.isnan(arr), 0, arr)
            arr = np.where(arr < -(4**2), 0, arr)
            arr = np.where(arr > 4**2, 0, arr)
            data[key] = arr

    print("Shape Min Maxs:\n")
    for key in aggregated_data[0].keys():
        print(
            f"{key}: {aggregated_data[0][key][0].shape}, {np.min(aggregated_data[0][key])}, {np.max(aggregated_data[0][key])}"
        )
    print("\n----------\n")
    for key in aggregated_data[1].keys():
        print(
            f"{key}: {np.array(aggregated_data[1][key][0]).shape}, {np.min(aggregated_data[1][key])}, {np.max(aggregated_data[1][key])}"
        )
    print("\n----------\n")
    for key in aggregated_data[2].keys():
        print(
            f"{key}: {np.array(aggregated_data[2][key][0]).shape}, {np.min(aggregated_data[2][key])}, {np.max(aggregated_data[2][key])}"
        )

    return aggregated_data


import torch.nn as nn
import torch.nn.functional as F


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

def normalize_data_dynamix(data):
    # Compute mean and standard deviation for each key
    mean_dict = [{}, {}]
    std_dict = [{}, {}]

    for idx in range(len(mean_dict)):
        for key in data[idx].keys():
            all_values = np.concatenate(
                [np.array(arr).flatten() for arr in data[idx][key]]
            )
            mean_dict[idx][key] = np.mean(all_values, axis=0)
            std_dict[idx][key] = np.std(all_values, axis=0)

    # Normalize the data
    NON_NORM_KEYS = [
        "obj_count",
        "progress_bar",
        "state_attrib"
    ]

    normalized_data = [data[0], {}]
    for key in data[1].keys():
        if key in NON_NORM_KEYS:
            normalized_data[1][key] = data[1][key]
            continue
        normalized_data[1][key] = []
        for arr in data[1][key]:
            normalized_arr = (arr - mean_dict[1][key]) / std_dict[1][key]
            normalized_data[1][key].append(normalized_arr)

    print("Dynamix:")
    print("MEAN:", mean_dict)
    print("STD:", std_dict)

    return normalized_data


def normalize_data_critiq(data):
    # Compute mean and standard deviation for each key
    mean_dict = [{}, {}, {}]
    std_dict = [{}, {}, {}]

    for idx in range(len(mean_dict)):
        for key in data[idx].keys():
            all_values = np.concatenate(
                [np.array(arr).flatten() for arr in data[idx][key]]
            )
            mean_dict[idx][key] = np.mean(all_values, axis=0)
            std_dict[idx][key] = np.std(all_values, axis=0)

    # Normalize the data
    NON_NORM_KEYS = ["action", "state_attrib"]

    normalized_data = [data[0], data[1], {}]
    for key in data[2].keys():
        if key in NON_NORM_KEYS:
            normalized_data[2][key] = data[2][key]
            continue
        normalized_data[2][key] = []
        for arr in data[2][key]:
            normalized_arr = (arr - mean_dict[2][key]) / std_dict[2][key]
            normalized_data[2][key].append(normalized_arr)

    print("Critiq:")
    print("MEAN:", mean_dict)
    print("STD:", std_dict)

    return normalized_data

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


## Augments Data into Correct Format (Converts the Dict objects from Key, Idx, value to Idx, Key, value format)


def convert_dataset_dynamix(
    data,
):
    num_samples = len(data[0]["palm_location"])
    state_elements = {
        "palm_tactile": [],
        "finger_1_tactile": [],
        "finger_2_tactile": [],
        "finger_3_tactile": [],
        "palm_location": [],
        "finger_1_location": [],
        "finger_2_location": [],
        "finger_3_location": [],
        "obj_location": [],
        "obj_velocity": [],
        "action": [],
        "state_attrib": [],
    }
    pred_elements = {
        "palm_tactile": [],
        "finger_1_tactile": [],
        "finger_2_tactile": [],
        "finger_3_tactile": [],
        "palm_location": [],
        "finger_1_location": [],
        "finger_2_location": [],
        "finger_3_location": [],
        "obj_location": [],
        "obj_count": [],
        "reward": [],
        "progress_bar": [],
    }

    state = data[0]
    prediction = data[1]
    for i in range(num_samples):
        for k in state_elements.keys():
            state_elements[k].append(
                th.nan_to_num(
                    th.tensor(state[k][i], dtype=th.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
        for k in pred_elements.keys():
            pred_elements[k].append(
                th.nan_to_num(
                    th.tensor(
                        np.array(
                            prediction[k][i],
                        ).flatten(),
                        dtype=th.float32,
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )

    return state_elements, pred_elements


def convert_dataset_critiq(
    data,
):
    num_samples = len(data[0]["palm_location"])
    state_elements = {
        "palm_tactile": [],
        "finger_1_tactile": [],
        "finger_2_tactile": [],
        "finger_3_tactile": [],
        "palm_location": [],
        "finger_1_location": [],
        "finger_2_location": [],
        "finger_3_location": [],
        "obj_location": [],
        "obj_velocity": [],
        "state_attrib": [],
    }

    n_state_elements = {
        "palm_tactile": [],
        "finger_1_tactile": [],
        "finger_2_tactile": [],
        "finger_3_tactile": [],
        "palm_location": [],
        "finger_1_location": [],
        "finger_2_location": [],
        "finger_3_location": [],
        "obj_location": [],
        "obj_velocity": [],
        "state_attrib": [],
    }

    pred_elements = {
        "reward": [],
        "action": [],
    }

    state = data[0]
    n_state = data[1]
    prediction = data[2]

    for i in range(num_samples):
        for k in state_elements.keys():
            state_elements[k].append(
                th.nan_to_num(
                    th.tensor(state[k][i], dtype=th.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )

        for k in n_state_elements.keys():
            n_state_elements[k].append(
                th.nan_to_num(
                    th.tensor(n_state[k][i], dtype=th.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )

        for k in pred_elements.keys():
            pred_elements[k].append(
                th.nan_to_num(
                    th.tensor(
                        np.array(
                            prediction[k][i],
                        ).flatten(),
                        dtype=th.float32,
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )

    return state_elements, n_state_elements, pred_elements


## Dataset Class for Object Motion Prediction

class ObjDataset(Dataset):
    def __init__(self, data):
        self.elements, self.preds = data
        self.num_samples = len(self.elements["action"])
        self.rand_sort()

    def rand_sort(self):
        permutations = np.random.permutation(self.num_samples)
        self.elements = {
            k: [val[p] for p in permutations] for k, val in self.elements.items()
        }
        self.preds = {
            k: [val[p] for p in permutations] for k, val in self.preds.items()
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_elements = {key: value[idx] for key, value in self.elements.items()}
        y_output = {key: value[idx] for key, value in self.preds.items()}
        return sample_elements, y_output


class CritiqDataset(Dataset):
    def __init__(self, data):
        self.states, self.n_states, self.preds = data
        self.num_samples = len(self.states["palm_location"])
        self.rand_sort()

    def rand_sort(self):
        permutations = np.random.permutation(self.num_samples)
        self.states = {k: [val[p] for p in permutations] for k, val in self.states.items()}
        self.n_states = {k: [val[p] for p in permutations] for k, val in self.n_states.items()}
        self.preds = {k: [val[p] for p in permutations] for k, val in self.preds.items()}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        state_batch = {key: value[idx] for key, value in self.states.items()}
        n_state_batch = {key: value[idx] for key, value in self.n_states.items()}
        pred_batch = {key: value[idx] for key, value in self.preds.items()}
        return state_batch, n_state_batch, pred_batch

def load_dynamix_and_critiq_data(train_test_split=0.92, random_seed=42) -> Tuple[ObjDataset, ObjDataset, CritiqDataset, CritiqDataset]:
    # Initial load from file:
    dynamix_data = load_data_node_format(PATH_DYNAMIX)
    pred_data = load_data_state_format(PATH_CRITIQ)
    
    # Normalizes the data:
    dynamix_data = normalize_data_dynamix(dynamix_data)
    pred_data = normalize_data_critiq(pred_data)
    
    # Converts it into tensors (ready for batch)
    dynamix_data = convert_dataset_dynamix(dynamix_data)
    pred_data = convert_dataset_critiq(pred_data)
    
    
    dynamix_dataset = ObjDataset(dynamix_data)
    train_size = int(train_test_split * len(dynamix_dataset))
    validation_size = len(dynamix_dataset) - train_size
    dynamix_train_dataset, dynamix_validation_dataset = random_split(
        dynamix_dataset, [train_size, validation_size]
    )

    critiq_dataset = CritiqDataset(pred_data)
    train_size = int(train_test_split * len(critiq_dataset))
    validation_size = len(critiq_dataset) - train_size
    critiq_train_dataset, critiq_validation_dataset = random_split(
        critiq_dataset, [train_size, validation_size]
    )
    
    return dynamix_train_dataset, dynamix_validation_dataset, critiq_train_dataset, critiq_validation_dataset


def lr_schedule():
    global start_lr, end_lr, factor
    ret_lr = start_lr
    start_lr *= factor
    return ret_lr

# Try Including/Excluding Finger Values (2)
# Diff between loss (Mean/Sum) ()
# Use huggingface model
# Model Predictive Control: Accuracy (1)
#
key_losses_dynamix = {
    "palm_tactile": lambda y_pred, y_target: 1 - F.cosine_similarity(y_pred, y_target).mean(),
    "finger_1_tactile": lambda y_pred, y_target: 1 - F.cosine_similarity(y_pred, y_target).mean(),
    "finger_2_tactile": lambda y_pred, y_target: 1 - F.cosine_similarity(y_pred, y_target).mean(),
    "finger_3_tactile": lambda y_pred, y_target: 1 - F.cosine_similarity(y_pred, y_target).mean(),
    "palm_location": lambda y_pred, y_target: torch.zeros_like(
        F.mse_loss(y_pred, y_target)
    ),  # F.mse_loss(y_pred, y_target),
    "finger_1_location": lambda y_pred, y_target: F.smooth_l1_loss(y_pred, y_target),
    "finger_2_location": lambda y_pred, y_target: F.smooth_l1_loss(y_pred, y_target),
    "finger_3_location": lambda y_pred, y_target: F.smooth_l1_loss(y_pred, y_target),
    "obj_location": lambda y_pred, y_target: F.smooth_l1_loss(y_pred, y_target),
    "obj_count": lambda y_pred, y_target: F.cross_entropy(y_pred, y_target),
    "reward": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
} 

key_losses_critiq = {
    "action": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
    "reward": lambda y_pred, y_target: F.mse_loss(y_pred, y_target),
}

class JointModel(pl.LightningModule):
    def __init__(self, dynamix_model, critiq_model, learning_rate=0.005):
        super().__init__()
        self.automatic_optimization = False
        self.dynamix_model = dynamix_model
        self.critiq_model = critiq_model
        self.params = 0
        
        self.learner = None
        self.scheduler = None
        
        self.enable_progress_bar = True,  
        self.learning_rate = learning_rate
        self.val_count_accuracy = Accuracy(task="multiclass", num_classes=DYNAMIX_OUTPUT_SIZES_DICT["obj_count"])  # Assuming 5 classes for obj_count
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # print(f"BATCH {batch_idx}")
        # print_dict(batch)
        for model in (self.dynamix_model, self.critiq_model):
            model.zero_grad()
            
        dynamix_data, dynamix_target = batch['dynamix']
        critiq_state, critiq_n_state, critiq_target = batch['critiq']

        dynamix_pred = self.dynamix_model(dynamix_data)
        dynamix_loss = sum([key_losses_dynamix[key](dynamix_pred[key], dynamix_target[key]) 
                            for key in key_losses_dynamix.keys() 
                            if key in dynamix_pred and key in dynamix_target])

        critiq_pred = self.critiq_model(critiq_state, critiq_n_state)
        critiq_loss = sum([key_losses_critiq[key](critiq_pred[key], critiq_target[key]) 
                           for key in key_losses_critiq.keys() 
                           if key in critiq_pred and key in critiq_target])

        dynamix_loss.backward()
        critiq_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.params, 2.0)
        
        self.learner.step()
        
        self.scheduler.step(self.trainer.current_epoch)

        self.log('train_dynamix_loss', dynamix_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_critiq_loss', critiq_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_total_loss', dynamix_loss.item() + critiq_loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": th.Tensor([dynamix_loss.item() + critiq_loss.item()])}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            dynamix_data, dynamix_target = batch['dynamix']
            critiq_state, critiq_n_state, critiq_target = batch['critiq']

            dynamix_pred = self.dynamix_model(dynamix_data)
            dynamix_loss = sum([key_losses_dynamix[key](dynamix_pred[key], dynamix_target[key]) 
                                for key in key_losses_dynamix.keys() 
                                if key in dynamix_pred and key in dynamix_target])

            pred_count = torch.argmax(F.softmax(dynamix_pred["obj_count"], dim=1), dim=1)
            true_count = torch.argmax(dynamix_target["obj_count"], dim=1)
            self.val_count_accuracy(pred_count, true_count)

            critiq_pred = self.critiq_model(critiq_state, critiq_n_state)
            critiq_loss = sum([key_losses_critiq[key](critiq_pred[key], critiq_target[key]) 
                            for key in key_losses_critiq.keys() 
                            if key in critiq_pred and key in critiq_target])

        self.log('val_dynamix_loss', dynamix_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_critiq_loss', critiq_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_count_accuracy', self.val_count_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        self.params = []
        self.params.extend(self.dynamix_model.parameters())
        self.params.extend(self.critiq_model.parameters())
        self.learner = Adam(self.params, lr=self.learning_rate)
        self.scheduler = CosineAnnealingLR(self.learner, T_max=self.trainer.max_epochs, eta_min=0)
        return {
            "optimizer": self.learner,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dynamix_dataset, critiq_dataset):
        self.dynamix_dataset = dynamix_dataset
        self.critiq_dataset = critiq_dataset
        self.length = min(len(self.dynamix_dataset), len(self.critiq_dataset))

    def __getitem__(self, index):
        return {
            'dynamix': self.dynamix_dataset[index],
            'critiq': self.critiq_dataset[index]
        }

    def __len__(self):
        return self.length
    
def custom_collate(batch):
    dynamix_data = [item['dynamix'] for item in batch]
    critiq_data = [item['critiq'] for item in batch]
    
    dynamix_elements = {k: torch.stack([d[0][k] for d in dynamix_data]) for k in dynamix_data[0][0].keys()}
    dynamix_preds = {k: torch.stack([d[1][k] for d in dynamix_data]) for k in dynamix_data[0][1].keys()}
    
    critiq_state = {k: torch.stack([d[0][k] for d in critiq_data]) for k in critiq_data[0][0].keys()}
    critiq_n_state = {k: torch.stack([d[1][k] for d in critiq_data]) for k in critiq_data[0][1].keys()}
    critiq_preds = {k: torch.stack([d[2][k] for d in critiq_data]) for k in critiq_data[0][2].keys()}
    
    return {
        'dynamix': (dynamix_elements, dynamix_preds),
        'critiq': (critiq_state, critiq_n_state, critiq_preds)
    }

def print_dict(d, indent=0):
    if isinstance(d, dict):
        for k,v in d.items():
            t = '\t'*indent
            print(f"{t}{k}:")
            print_dict(v, indent+1)
    else:
        if isinstance(d, tuple):
            for i in d:
                print_dict(i, indent+1)
        else:
            print('\t'*indent, d.shape)  
                

def train_models(dynamix_model, critiq_model, dynamix_train_dataset, dynamix_val_dataset, 
                 critiq_train_dataset, critiq_val_dataset, batch_size=16, epochs=1000, 
                 learning_rate=5e-3, num_gpus=1):

    train_dataset = CombinedDataset(dynamix_train_dataset, critiq_train_dataset)
    val_dataset = CombinedDataset(dynamix_val_dataset, critiq_val_dataset)

    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate)

    for batch in train_loader:
        b = batch
        print_dict(b)
        break 
    
    for batch in val_loader:
        b = batch
        print_dict(b)
        break


    joint_model = JointModel(dynamix_model, critiq_model, learning_rate)

    logger = TensorBoardLogger("lightning_logs", name="joint_model")
    checkpoint_callback = ModelCheckpoint(monitor="val_count_accuracy")
    

    trainer = pl.Trainer(
        max_epochs=epochs,
        # devices=-1,  # Use all available GPUs
        # accelerator='gpu' if num_gpus > 0 else 'cpu',
        # devices=num_gpus if num_gpus > 0 else None,
        logger=logger,
        callbacks=[checkpoint_callback, LoggingCallback()],
        # strategy='ddp' if num_gpus > 1 else None,
        # strategy=DDPStrategy(find_unused_parameters=True),
        accelerator='gpu',
        devices="0",
        enable_progress_bar=True,
    )

    trainer.fit(joint_model, train_loader, val_loader)

    return joint_model

## Params
NUM_LAYERS_TRANSFORMER = 4
NUM_RESIDUALS = 3
EPOCHS = 1000
VEC_ENCODING_SIZE = 512

MODEL_ARGS = {
    "vec_encoding_size": VEC_ENCODING_SIZE,
    "num_residuals": NUM_RESIDUALS,
    "num_tsf_layer": NUM_LAYERS_TRANSFORMER,
    "use_mask": True,
    "dropout_prob": 0.01,
    "embed_dim_low": VEC_ENCODING_SIZE,
    "T_buffer": T_buffer,
    "state_space": state_space,
}

dynamix_model, critiq_model = make_dynamix_and_predictor(MODEL_ARGS)

dynamix_train_dataset, dynamix_validation_dataset, critiq_train_dataset, critiq_validation_dataset = load_dynamix_and_critiq_data()

def debug_datasets(dynamix_dataset, critiq_dataset, batch_size=16):
    combined_dataset = CombinedDataset(dynamix_dataset, critiq_dataset)
    
    print(f"Combined dataset length: {len(combined_dataset)}")
    
    # Simulate a batch
    batch = [combined_dataset[i] for i in range(batch_size)]
    
    # Use our custom collate function
    collated_batch = custom_collate(batch)
    
    print("\nDynamix data:")
    for key, value in collated_batch['dynamix'][0].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")
    
    print("\nDynamix predictions:")
    for key, value in collated_batch['dynamix'][1].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")
    
    print("\nCritiq state:")
    for key, value in collated_batch['critiq'][0].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")
    
    print("\nCritiq next state:")
    for key, value in collated_batch['critiq'][1].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")
    
    print("\nCritiq predictions:")
    for key, value in collated_batch['critiq'][2].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

# print("Debugging training datasets:")
# debug_datasets(dynamix_train_dataset, critiq_train_dataset)

# print("\nDebugging validation datasets:")
# debug_datasets(dynamix_validation_dataset, critiq_validation_dataset)

## Params
NUM_LAYERS_TRANSFORMER = 8
NUM_RESIDUALS = 8
EPOCHS = 1000
VEC_ENCODING_SIZE = 512

MODEL_ARGS = {
    "vec_encoding_size": VEC_ENCODING_SIZE,
    "num_residuals": NUM_RESIDUALS,
    "num_tsf_layer": NUM_LAYERS_TRANSFORMER,
    "use_mask": True,
    "dropout_prob": 0.01,
    "embed_dim_low": VEC_ENCODING_SIZE,
    "T_buffer": T_buffer,
    "state_space": state_space,
}

# Usage
num_gpus = torch.cuda.device_count()

joint_model = train_models(
    dynamix_model, 
    critiq_model, 
    dynamix_train_dataset, 
    dynamix_validation_dataset, 
    critiq_train_dataset, 
    critiq_validation_dataset, 
    batch_size=BATCH_SIZE, 
    epochs=EPOCHS, 
    learning_rate=0.0008,
    num_gpus=num_gpus,
)

# Save the trained models
torch.save(joint_model.dynamix_model.state_dict(), "./trained_dynamix_model.pt")
torch.save(joint_model.critiq_model.state_dict(), "./trained_critiq_model.pt")