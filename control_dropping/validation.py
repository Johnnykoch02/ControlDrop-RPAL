import os
from re import I
import shutil
import sys
import multiprocessing
import subprocess
import argparse
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch as th
from src.Utils.network_utils import find_open_port
from src.Utils.data_utils import plot_results

th.manual_seed(42)

# from stable_baselines3.common import env_checker
# from src.RL.callbacks import TrainAndLoggingCallback
from src.RL.control_dropping_env import (
    BerrettHandGym,
    SceneDifficulty,
    NAME,
    CHECKPOINT_DIR,
    LOG_DIR,
    SAVE_FREQ,
)
from src.SimController.sim_controller import SimController
from stable_baselines3 import PPO, DDPG
from sb3_contrib import RecurrentPPO
import time

COPPELIA_LOCATION = "/home/rpal/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04"
SCENE_LOCATION = "./40mm_Sphere_Random_color.ttt"
VALIDATION_DATA_PATH = os.path.join(
    os.getcwd(), "Data_Collection", "ValidationDifficultys"
)
server_port = find_open_port()

# CMD_ARGS = [
#     os.path.join(COPPELIA_LOCATION, "coppeliaSim.sh"), # Coppelia Process
#     SCENE_LOCATION, "-s" , # Scene
#     f"-gREMOTEAPISERVERPORT={server_port}" # Remote Server
# ]
CMD_ARGS = [f"./start_sim_on_port.sh", f"{server_port}"]
ARGS = argparse.ArgumentParser(
    description="CLI Tool for Performing Validation on the BerretHandGym"
)

ARGS.add_argument("--algo", type=str, help="Name of Algorithm being Used")
ARGS.add_argument(
    "--server_port", type=str, help="Port of the server to run the model on."
)
ARGS.add_argument(
    "--model_path",
    type=str,
    help="Path to the model being Validated",
)
ARGS.add_argument(
    "--use_sbl",
    type=bool,
    help="Sets algorithm (In House or Stable Baselines)",
    default=True,
)
ARGS.add_argument(
    "--data_save_path",
    type=str,
    help="Path to Save Data to, otherwise None.",
    default=None,
)
ARGS.add_argument(
    "--plot_val",
    type=bool,
    help="True if plotting current validation data, otherwise False.",
    default=False,
)


def run_copellia_processs():
    subprocess.run(
        args=CMD_ARGS,
    )


# def _start_scene():
#     global server_port
#     simController = SimController(port=server_port)
#     simController.start_sim()


def validate(algo, model_path, use_sbl, data_save_path, plot_val):
    global server_port
    path = (
        model_path
        if os.path.exists(model_path)
        else os.path.join(CHECKPOINT_DIR, model_path)
    )
    if use_sbl:
        if algo == "PPO":
            model = PPO.load(path)
        elif algo == "RecurrentPPO":
            model = RecurrentPPO.load(
                path,
                use_sde=True,
            )
        elif algo == "DDPG":
            model = DDPG.load(path)
        else:
            return
        env = BerrettHandGym(
            object_type="Sphere",
            object_quantity=7,
            detailed_training=True,
            difficulties=[
                SceneDifficulty.EASY,
                SceneDifficulty.MEDIUM,
                SceneDifficulty.HARD,
            ],
            algorithm=algo,
            is_val=True,
        )
        # env.configs = env.get_detailed_training_data(path=VALIDATION_DATA_PATH)
        model.set_env(env)
        obs = env.reset()
        reward = []
        Dones = []
        num_episodes = 0
        done = False
        DROPPING_STEP_SUCCESS = 8
        save_flag = False
        drop_stats = {}
        Difficulty_Stats = {
            # (Difficulty, Goal Count) : success rate
        }

        while num_episodes < 150:  # 150 Episode Sample

            # Count number of Successful Steps post Drop
            if env.current_obj_count == env.goal_obj_count:
                drop_stats[env.post_drop] = drop_stats.get(env.post_drop, 0) + 1
            if (
                not save_flag
                and env.current_obj_count == env.goal_obj_count
                and env.post_drop > DROPPING_STEP_SUCCESS
            ):
                save_flag = True
                ds = Difficulty_Stats.get(
                    (env.current_index_difficulty, env.goal_obj_count), []
                )
                ds.append(1)
                Difficulty_Stats[env.current_index_difficulty, env.goal_obj_count] = ds

            if done:
                if not save_flag:
                    ds = Difficulty_Stats.get(
                        (env.current_index_difficulty, env.goal_obj_count), []
                    )
                    ds.append(0)
                    Difficulty_Stats[
                        env.current_index_difficulty, env.goal_obj_count
                    ] = ds
                save_flag = False
                Dones.append(done)
                env.reset()
                num_episodes += 1
            action, _states = model.predict(obs.copy(), deterministic=False)  # Or True
            obs, rewards, done, info = env.step(action.squeeze())
            reward.append(rewards)
        stat_data = {"Post_Drop": [v / num_episodes for v in drop_stats.values()]}
        plot_results(
            [stat_data],
            ["Post_Drop"],
            ["# of Steps Post Drop"],
            range_spc=range(len(stat_data["Post_Drop"])),
            file_name=os.path.join(env.performance_save_path, "Post_Drop.png"),
        )
        file_data = f'{"="*10}' * 3 + "\n"
        file_data += f"{model_path}\n"
        for k, v in Difficulty_Stats.items():
            file_data += f"{k[0]}, {k[1]} Balls: {np.mean(v):.2f}\n"
        file_data += f'{"="*10}' * 3 + "\n"
        file_num = len(
            os.listdir(os.path.join("control_dropping", "Validations"))
        )

        os.makedirs(
            os.path.join("control_dropping", "Validations", f"{file_num}"),
            exist_ok=True,
        )
        with open(
            os.path.join(
                f"control_dropping/Validations/{file_num}/Val_Difficulty_Stats.txt"
            ),
            "w",
        ) as f:
            f.write(file_data)
        shutil.copy(
            path, os.path.join("control_dropping", "Validations", f"{file_num}")
        )


def main(algo, model_path, use_sbl, data_save_path, plot_val):
    # print(f'Starting server on port {server_port}')
    # _cop = multiprocessing.Process(target=run_copellia_processs,)
    # _val = multiprocessing.Process(target=validate, args=(algo, model_path, use_sbl, data_save_path, plot_val))

    # Start Sim, Wait for Coppelia to open, Start Scene, Run Validation
    # _cop.start()
    # time.sleep(30)
    # _start_scene()
    # time.sleep(5)
    # _val.start()

    # Wait for Validation to finish and kill Sim
    # _val.join()
    # _cop.terminate()
    validate(algo, model_path, use_sbl, data_save_path, plot_val)
    print("Exiting Validation.")


if __name__ == "__main__":
    _args = ARGS.parse_args()
    exit(
        main(
            _args.algo,
            _args.model_path,
            _args.use_sbl,
            _args.data_save_path,
            _args.plot_val,
        )
    )
