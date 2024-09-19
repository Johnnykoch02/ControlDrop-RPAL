import os
import numpy as np
import pickle
import time
import argparse
from tqdm import tqdm
from control_dropping_rpal.RL.control_dropping_env import BerrettHandGym
import multiprocessing as mp

# Ensure CONTROL_DROP_DIR is set in your environment variables
CONTROL_DROP_DIR = os.environ.get("CONTROL_DROP_DIR")
if not CONTROL_DROP_DIR:
    raise EnvironmentError("CONTROL_DROP_DIR environment variable is not set")

DATA_SAVE_PATH = os.path.join(
    CONTROL_DROP_DIR, "Data_Collection", "Action_Pred_Time_Dependent_Samples_5"
)
MAX_ACTION_COEF = 2.5


def save_data(data, file_path, save_data_lock):
    with save_data_lock:
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)

        state_list = [state[0].copy() for state in data]
        pred_state_list = [state[1].copy() for state in data]

        chunk_data = {
            "states": state_list,
            "pred_states": pred_state_list,
        }

        import re

        i = (
            sorted([int(re.sub(r"\D", "", s)) for s in os.listdir(file_path)] + [-1])[
                -1
            ]
            + 1
        )
        chunk_file_path = os.path.join(file_path, f"Actions_data_{i}.pkl")

        with open(chunk_file_path, "wb") as f:
            pickle.dump(chunk_data, f)
        print(f"Saved to {chunk_file_path}")


def _get_sampled_action(action_space):
    action_coef = np.random.uniform() * MAX_ACTION_COEF
    action = (
        action_coef * action_space.sample()
        if np.random.uniform() > 0.5
        else np.zeros(action_space.shape)
    )
    return action


def collect_critiq_data(num_steps, save_interval=1000, save_data_lock=None):
    env = BerrettHandGym(detailed_training=True, is_val=True)
    data = []
    data_point = env.reset()[0].copy()

    for i in tqdm(
        range(num_steps),
        desc=f"Collecting data (PID: {os.getpid()})",
        position=mp.current_process()._identity[0] - 1,
    ):
        try:
            action = _get_sampled_action(env.action_space)
            data_point["action"] = action
            state, reward, done, _ = env.step(action)

            pred_state = state.copy()
            pred_state["reward"] = reward
            pred_state["action"] = action

            data.append((data_point, pred_state))
            data_point = state.copy()

            if done or (i > 0 and i % save_interval == 0):
                save_data(data, DATA_SAVE_PATH, save_data_lock)
                data = []
                if done:
                    data_point = env.reset()[0].copy()

        except Exception as e:
            print(f"An error occurred: {e}")
            data = []
            data_point = env.reset()[0].copy()

    # Save any remaining data
    if data:
        save_data(data, DATA_SAVE_PATH, save_data_lock)

    env.close()


def main(args):
    if args.num_instances > 1:
        save_data_lock = mp.Lock()
        processes = []
        for _ in range(args.num_instances):
            p = mp.Process(
                target=collect_critiq_data,
                args=(args.num_steps, args.save_interval, save_data_lock),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        collect_critiq_data(args.num_steps, args.save_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect critiq data")
    parser.add_argument(
        "--num_steps", type=int, default=30000, help="Number of steps to collect data"
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=1,
        help="Number of parallel data collection instances",
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Interval for saving data"
    )
    args = parser.parse_args()

    main(args)
