import pickle
import os

CONTROL_DROP_DIR = os.environ["CONTROL_DROP_DIR"]

import numpy as np
from typing import Tuple

# Torch
import torch as th
from torch.utils.data import Dataset, random_split

from control_dropping_rpal.RL.control_dropping_env import T_buffer


def mult(shape):
    """Utility function to compute the product of elements in a shape tuple."""
    product = 1
    for dim in shape:
        product *= dim
    return product


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
    NON_NORM_KEYS = ["obj_count", "progress_bar", "state_attrib"]

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
        self.states = {
            k: [val[p] for p in permutations] for k, val in self.states.items()
        }
        self.n_states = {
            k: [val[p] for p in permutations] for k, val in self.n_states.items()
        }
        self.preds = {
            k: [val[p] for p in permutations] for k, val in self.preds.items()
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        state_batch = {key: value[idx] for key, value in self.states.items()}
        n_state_batch = {key: value[idx] for key, value in self.n_states.items()}
        pred_batch = {key: value[idx] for key, value in self.preds.items()}
        return state_batch, n_state_batch, pred_batch


def load_dynamix_and_critiq_data(
    path_dynamix, path_critiq, train_test_split=0.92, random_seed=42
) -> Tuple[ObjDataset, ObjDataset, CritiqDataset, CritiqDataset]:
    # Initial load from file:
    dynamix_data = load_data_node_format(path_dynamix)
    pred_data = load_data_state_format(path_critiq)

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

    return (
        dynamix_train_dataset,
        dynamix_validation_dataset,
        critiq_train_dataset,
        critiq_validation_dataset,
    )


class CombinedDataset(Dataset):
    def __init__(self, dynamix_dataset, critiq_dataset):
        self.dynamix_dataset = dynamix_dataset
        self.critiq_dataset = critiq_dataset
        self.length = min(len(self.dynamix_dataset), len(self.critiq_dataset))

    def __getitem__(self, index):
        return {
            "dynamix": self.dynamix_dataset[index],
            "critiq": self.critiq_dataset[index],
        }

    def __len__(self):
        return self.length


def debug_datasets(dynamix_dataset, critiq_dataset, batch_size=16):
    combined_dataset = CombinedDataset(dynamix_dataset, critiq_dataset)

    print(f"Combined dataset length: {len(combined_dataset)}")

    # Simulate a batch
    batch = [combined_dataset[i] for i in range(batch_size)]

    # Use our custom collate function
    collated_batch = custom_dynamix_critiq_collate(batch)

    print("\nDynamix data:")
    for key, value in collated_batch["dynamix"][0].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

    print("\nDynamix predictions:")
    for key, value in collated_batch["dynamix"][1].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

    print("\nCritiq state:")
    for key, value in collated_batch["critiq"][0].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

    print("\nCritiq next state:")
    for key, value in collated_batch["critiq"][1].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

    print("\nCritiq predictions:")
    for key, value in collated_batch["critiq"][2].items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")


def custom_dynamix_critiq_collate(batch):
    dynamix_data = [item["dynamix"] for item in batch]
    critiq_data = [item["critiq"] for item in batch]

    dynamix_elements = {
        k: th.stack([d[0][k] for d in dynamix_data]) for k in dynamix_data[0][0].keys()
    }
    dynamix_preds = {
        k: th.stack([d[1][k] for d in dynamix_data]) for k in dynamix_data[0][1].keys()
    }

    critiq_state = {
        k: th.stack([d[0][k] for d in critiq_data]) for k in critiq_data[0][0].keys()
    }
    critiq_n_state = {
        k: th.stack([d[1][k] for d in critiq_data]) for k in critiq_data[0][1].keys()
    }
    critiq_preds = {
        k: th.stack([d[2][k] for d in critiq_data]) for k in critiq_data[0][2].keys()
    }

    return {
        "dynamix": (dynamix_elements, dynamix_preds),
        "critiq": (critiq_state, critiq_n_state, critiq_preds),
    }


def get_joint_dataset(
    dynamix_path,
    critiq_path,
    train_test_split=0.92,
):
    (
        dynamix_train_dataset,
        dynamix_validation_dataset,
        critiq_train_dataset,
        critiq_validation_dataset,
    ) = load_dynamix_and_critiq_data(
        dynamix_path, critiq_path, train_test_split=train_test_split
    )
    train_dataset = CombinedDataset(dynamix_train_dataset, critiq_train_dataset)
    val_dataset = CombinedDataset(dynamix_validation_dataset, critiq_validation_dataset)

    return train_dataset, val_dataset
