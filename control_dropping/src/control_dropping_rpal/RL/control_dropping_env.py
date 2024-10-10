import json
import gzip
import logging
import os

CONTROL_DROP_DIR = os.environ["CONTROL_DROP_DIR"]

import re
import subprocess
import time
from collections import deque
from enum import IntEnum
from math import degrees, inf, radians
from operator import truediv
from typing import Callable, Dict, List, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from control_dropping_rpal.Utils.network_utils import find_open_port

from control_dropping_rpal.SimController import SimController, SensorBuffer
from control_dropping_rpal.Utils.data_utils import plot_results
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

# NAME = 'DDPG_TransformerFeatureEncoder'
NAME = "TransformerFeatureEncoder"

CHECKPOINT_DIR = os.path.join(
    CONTROL_DROP_DIR,
    "control_dropping",
    "src",
    "RL",
    "Training",
    "Checkpoints",
    NAME,
)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

LOG_DIR = os.path.join(CONTROL_DROP_DIR, "Training", "Logs", NAME)


def create_sim(sim_port, **kwargs) -> subprocess.Popen:
    logging.info(f"Creating SIM on port {sim_port}")
    CMD_ARGS = [
        f"{CONTROL_DROP_DIR}/scripts/start_sim_on_port.sh",
        f"{sim_port}",
        "--is_gui",
        os.environ.get("SIM_GUI", "false"),
    ]  #                                       Supress the STDOUT from the sims
    return subprocess.Popen(
        args=CMD_ARGS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


SAVE_FREQ = 2500


class SceneDifficulty(IntEnum):
    """In a labelled training dataset, we classify Scenes as easy, medium or hard"""
    EASY = 0
    MEDIUM = 1
    HARD = 2
    UNKNOWN = 3

class GraspShape(IntEnum):
    """In a labelled training dataset, we classify grasps to a specific type"""
    CYLINDRICAL = 0
    FUNNEL = 1
    CUP = 2
    TRACKS = 3

class GraspFunction(IntEnum):
    """In a labelled training dataset, we classify grasps to a specific function"""
    ABDUCTION_CLIP = 0
    FINGER_PALM_CLIP = 1
    FINGER_FINGER_CLIP = 2
    FINGERTIP_FINGER_TIP_PINCH = 3
    FINGERTIP_FINGER_PINCH = 4


class Spec:
    def __init__(self, id, max_steps):
        self.id = id
        self.max_episode_steps = max_steps

@dataclass
class EpisodicFloatHook:
    key: str
    fn_ptr: callable

    val: float = None
    info = {}
    _val_history = []

    def get(self) -> float:
        val, info = self.fn_ptr()
        self.info = info
        self.val = val
        self._val_history.append(val)
        return val

    def reset(self) -> None:
        self.val = None
        self.info = {}
        self._val_history = []


OBJECT_QUANTITY = 7
"""Max # of items in the scene"""

K_NUM_ACTIONS = 128
"""Number of actions to sample for the policy"""

T_buffer = 5
"""Time Buffer for Simulation"""

ACTIONS_TIME_LEN = 8
"""Num actions tracked in the module"""

# spaces = {
#     'hand_config': Box(low= -inf, high= inf, shape=(7, )), # Position
#     'hand_torque': Box(low= -inf, high= inf, shape=(3, )), # Value
#     'palm_tactile': Box(low= -inf, high= inf, shape=(1,3,8)), # Value
#     'finger_1_tactile': Box(low= -inf, high= inf, shape=(1,3,8)), # Value
#     'finger_2_tactile': Box(low= -inf, high= inf, shape=(1,3,8)), # Value
#     'finger_3_tactile': Box(low= -inf, high= inf, shape=(1,3,8)), # Value
#     'tactile_pos': Box(low= -inf, high= inf, shape=(378, )), # Position
#     'ball_count': Box(low = 0, high = 1, shape = (1, )), # Position
#     'ball_location': Box(low= -inf, high= inf, shape=(42, )), # Position
#     'obj_velocity': Box(low= -inf, high= inf, shape=(42, )), # Value
#     'obj_angular_velocity': Box(low=0, high= inf, shape=(10,)), # Value
#     'progress_bar': Box(low= -inf, high= inf, shape = (1,)), # Position
#     'previous_actions': Box(low= -inf, high= inf, shape=(32,)), # Position
# }

base_observation_space = Dict(
    {
        "palm_tactile": Box(
            low=-inf,
            high=inf,
            shape=(
                T_buffer,
                24,
            ),
        ),
        "finger_1_tactile": Box(
            low=-inf,
            high=inf,
            shape=(
                T_buffer,
                24,
            ),
        ),
        "finger_2_tactile": Box(
            low=-inf,
            high=inf,
            shape=(
                T_buffer,
                24,
            ),
        ),
        "finger_3_tactile": Box(
            low=-inf,
            high=inf,
            shape=(
                T_buffer,
                24,
            ),
        ),
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
        "obj_location": Box(low=-inf, high=inf, shape=(T_buffer, OBJECT_QUANTITY, 6)),
        "obj_velocity": Box(
            low=-inf, high=inf, shape=(T_buffer, OBJECT_QUANTITY, 6)
        ),  # Value, Concat with angular velocity
        "state_attrib": Box(
            low=-inf, high=inf, shape=(45,)
        ),  # Ball Cnt, Progress, Prev.Actions, hand_cfg, hand_trq (44)
        "actions": Box(
            low=-inf, high=inf, shape=(ACTIONS_TIME_LEN, 5)
        ),  # Action Samples to pick from
    }
)

default_action_space = Box(low=-7, high=7, shape=(5,), dtype=np.float32)

K_discrete_action_space = Discrete(K_NUM_ACTIONS)

class SceneManager:
    def __init__(self, detailed_training=False, difficulties=None, is_test=False):
        self.detailed_training = detailed_training
        self.difficulties = difficulties or [
            SceneDifficulty.EASY,
            SceneDifficulty.MEDIUM,
            SceneDifficulty.HARD,
        ]
        self.is_test = is_test
        self.data, self.metadata = self.load_scenes()

    def load_scenes(self):
        if self.detailed_training:
            return self.load_detailed_training_data()
        else:
            return self.load_default_data()

    def dataset_size(self):
        return len(self.data)

    def get_data_index(self, index):
        scene = self.data[index]
        metadata = self.metadata[index]
        return scene, metadata

    def set_metadata_at_index(self, index: int, new_metadata: dict):
        current_metadata = self.metadata[index]
        for key in new_metadata:
            if key != "index":  # Skip 'index' key
                current_metadata[key] = new_metadata[key]
        self.save_metadata()

    def save_metadata(self):
        path = os.path.join(CONTROL_DROP_DIR, "Data_Collection", "Difficultys")
        json_path = os.path.join(path, "metadata_difficulties.json")
        with open(json_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    # Adjusted load_detailed_training_data to load self.data and self.metadata
    def load_detailed_training_data(self):
        path = os.path.join(CONTROL_DROP_DIR, "Data_Collection", "Difficultys")

        # Load data
        data = np.load(os.path.join(path, "difficulties.npy"), allow_pickle=True)
        self.data = data

        # Load metadata
        with open(os.path.join(path, "metadata_difficulties.json"), "r") as f:
            metadata = json.load(f)
        self.metadata = metadata

        return self.data, self.metadata

    @classmethod
    def load_default_data(cls, is_test=False):
        file_name = "40_mm_sphere_control_drop_scenes.npy.gz" if is_test else "40_mm_sphere_train_control_drop_scenes.npy.gz"
        file_path = os.path.join(CONTROL_DROP_DIR, "control_dropping", "src", "RL", "Training", "Data", file_name)
        with gzip.GzipFile(file_path) as f:
            data =  np.load(f, allow_pickle=True)
            return data, [{"difficulty": SceneDifficulty.UNKNOWN} * len(data)]

    @classmethod
    def load_cluster(cls, cluster_id):
        file_path = os.path.join(CONTROL_DROP_DIR, "Data_Collection", "Clusters", f"{cluster_id}.npy")
        return np.load(file_path, allow_pickle=True)

    def get_random_scene(self):
        index = np.random.randint(0, len(self.data))
        scene = self.data[index]
        difficulty = self.metadata[index]["difficulty"]

        logging.debug("Setting up the Scene...\n Index:", index)        
        return scene, difficulty

    def get_scene_config(self, scene):
        hand_config, ball_locations = [np.degrees(i) for i in scene[-7:]], scene
        hand_config = [hand_config[i] for i in (0, 3, 2, 1, 6, 5, 4)]
        return hand_config, ball_locations


# Reward Function Parameters
TERMINATION_PENALTY = 150
LONG_DROP_PENALTY = 120
SUCESS_REWARD = 175
STABILIZING_REWARD = -20
FAILURE_PENALTY = 500
MOVEMENT_PENALTY_SCALE_FACTOR = 0.005
PALM_ANGLE_PENALTY_SCALE_FACTOR = 0.05
BASE_ALLOTED_DROPPING_TIME = 15  # Steps given to start a drop
PREDROP_PENALTY = 25  # Base penalty for a predropped object
POST_DROP_REWARD = 20  # Reward per step for the model after a drop has occured


class BerrettHandGym(gym.Env):
    """This class is designed to implement control of the Berret Hand through use of SimController"""

    def __init__(
        self,
        test=False,  # If trying to iterate through grasping data, set to True
        cluster_index=3,  # Used for setting the data cluster for use in training
        sim_port=None,  # Port to connect the sim on, or None if auto-create environment.
        object_type="Sphere",  # This is for the Sim to generate the Specific type of Object for Grasp
        object_quantity=7,  # Number of Objects being generated
        detailed_training=False,  # Controls the type of data and data detailing being performed during Training
        detailed_save_dir=None,  # Overrides the Path of Trial Saving
        worker_idx=0,  # Number of the worker associated with this env instance
        plot_params=[
            "history",
            "avg_ke",
            "avg_vel",
            "avg_rew",
        ],  # Specify which parameters to include in the Figures
        difficulties=[
            SceneDifficulty.EASY,
            SceneDifficulty.MEDIUM,
            SceneDifficulty.HARD,
        ],  # Difficulties to Train on
        algorithm="RecurrentPPO",  # Algorithm for loading in validation
        is_val=False,  # is Running Validation
    ):

        super().__init__()
        self.max_steps = 256
        if sim_port == None:
            self.sim_port = None
            self.simController = None
            self.sim_process = None

            def restart_trigger():
                self.sim_port = find_open_port()
                if self.sim_process:
                    self.sim_process.kill()
                self.sim_process = create_sim(self.sim_port)
                time.sleep(25)
                self.simController = SimController(
                    port=self.sim_port,
                    object_type=object_type,
                    object_quantity=object_quantity,
                    time_seq_len=T_buffer,
                )

            self.restart_trigger = restart_trigger
            self.restart_trigger()
        else:

            def restart_trigger():
                self.simController.restart_sim()

            self.sim_process = None
            self.restart_trigger = restart_trigger
            self.sim_port = sim_port
            self.simController = SimController(
                port=self.sim_port,
                object_type=object_type,
                object_quantity=object_quantity,
                time_seq_len=T_buffer,
            )

        # Misc Simulation Env Parameters
        self.obj_quantity = object_quantity
        self.test = test
        self.counter = {"index": 0, "total_steps": 0, "episodes": 0}
        self.configs_index = 0
        self.reset_counter = 0
        self.successes = []
        self.algorithm = algorithm  # not used?
        self.val_process = None
        self.is_val = is_val
        self.num_vals = 0
        self.clock = pygame.time.Clock()
        self.target_fps = 3
        self.worker_idx = worker_idx

        # Flag to indicate the necesity of checking sim gravity
        self.restart_check = False
        self.current_index_difficulty = None

        # Config parameters for Detailed Training Metrics:
        self.detailed_training = detailed_training

        self.difficulties = difficulties or [SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD]
        self.is_val = is_val
        self.scene_loader = SceneManager(detailed_training=detailed_training, 
                                        difficulties=self.difficulties, 
                                        is_test=self.is_val)
        
        if self.detailed_training:
            len_of_dtsve = len(
                os.listdir(
                    os.path.join(
                        CONTROL_DROP_DIR,
                        "Data_Collection",
                        "Model_Performance",
                    )
                )
            )
            self.performance_save_path = (
                os.path.join(
                    CONTROL_DROP_DIR,
                    "Data_Collection",
                    "Model_Performance",
                    f"Trial_{len_of_dtsve}",
                )
                if detailed_save_dir is None
                else detailed_save_dir
            )
            if not os.path.exists(self.performance_save_path):
                os.makedirs(self.performance_save_path)

            self.num_detailed_files = 0
            # Model Adds to performance, remembers history of previous saves and the performance at those time steps.
            self.model_performance = {
                SceneDifficulty.EASY: {
                    "performance": [],
                    "history": [],
                    "avg_ke": [],
                    "avg_vel": [],
                    "avg_rew": [],
                    "use_data": False,
                },
                SceneDifficulty.MEDIUM: {
                    "performance": [],
                    "history": [],
                    "avg_ke": [],
                    "avg_vel": [],
                    "avg_rew": [],
                    "use_data": False,
                },
                SceneDifficulty.HARD: {
                    "performance": [],
                    "history": [],
                    "avg_ke": [],
                    "avg_vel": [],
                    "avg_rew": [],
                    "use_data": False,
                },
            }
            self.current_run_vel = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.current_run_ke = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.validation_runs = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.episodic_reward = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.current_episodic_rew = 0
            self.plot_params = plot_params


        #
        self.detailed_save_frequency = 100
        self.spec = Spec(id="BerrettHandGym-v0", max_steps=self.max_steps)

        # Establish Data Spaces:
        self.observation_space = base_observation_space
        self.action_space = default_action_space
        # self.action_space = Box(low=-32, high=32, shape=(5,), dtype=np.float32) #

        self.previous_actions = deque()
        for _ in range(ACTIONS_TIME_LEN):
            self.previous_actions.append(np.zeros(shape=(5,)).astype(float))

        self.current_hand_config = self.simController.get_hand_config()
        self.current_obj_count = None
        self.goal_obj_count = None
        self.step_len = 0
        self.post_drop = 0
        self.bad_scenes = set()

        # Reward Function Parameters
        self.termination_penalty = TERMINATION_PENALTY
        self.long_drop_penalty = LONG_DROP_PENALTY
        self.success_reward = SUCESS_REWARD
        self.stabilizing_reward = STABILIZING_REWARD
        self.failure_penalty = FAILURE_PENALTY
        self.movement_penalty_scale_factor = MOVEMENT_PENALTY_SCALE_FACTOR
        self.palm_angle_penalty_scale_factor = PALM_ANGLE_PENALTY_SCALE_FACTOR
        self.success_step_count = 9 if is_val else 15  # Steps after drop
        self.base_alloted_dropping_time = (
            BASE_ALLOTED_DROPPING_TIME  # Steps given to start a drop
        )
        self.predrop_penalty = PREDROP_PENALTY
        self.post_drop_reward = POST_DROP_REWARD

        ## Register Penalty/Reward Hooks
        ## Register Penalty/Reward Hooks
        self.__termination_hook = EpisodicFloatHook(
            key="terminations",
            fn_ptr=self.terminations_hook
        )
        self.__limit_signals_hook = EpisodicFloatHook(
            key="limit_signals",
            fn_ptr=self.limit_signals_hook
        )
        self.__palm_displacement_hook = EpisodicFloatHook(
            key="palm_displacement",
            fn_ptr=self.palm_displacement_hook
        )
        self.__velocity_penalty_hook = EpisodicFloatHook(
            key="velocity_penalty",
            fn_ptr=self.velocity_penalty_hook
        )
        self.__drop_status_hook = EpisodicFloatHook(
            key="drop_status",
            fn_ptr=self.drop_status_hook
        )
        self.__step_bonus_hook = EpisodicFloatHook(
            key="step_bonus",
            fn_ptr=self.step_bonus_hook
        )
        self.episodic_penalty_hooks = {
            "terminations": self.__termination_hook,
            "limit_signals": self.__limit_signals_hook,
            "palm_displacement": self.__palm_displacement_hook,
            "velocity_penalty": self.__velocity_penalty_hook,
            "drop_status": self.__drop_status_hook,
            "step_bonus": self.__step_bonus_hook
        } # Hooks let us easily track reward values so we can understand their distributions

    def reset_data_cntrs(self):
        try:
            self.current_run_vel = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.current_run_ke = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.episodic_reward = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }
            self.validation_runs = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM: [],
                SceneDifficulty.HARD: [],
            }

        except Exception as e:
            logging.error("Error While resetting Data Cntrs..." + str(e))

    def step(self, action):
        """Defines how to step in the berret hand gym."""
        logging.debug(
            "Step: "
            + str(self.step_len)
            + "Current Difficulty: "
            + str(self.current_index_difficulty)
        )
        action *= 1.5
        self.step_len += 1
        # if self.counter['total_steps'] != 0:
        #     self.remove_extra_files()
        """ Action Space [-1:1] """
        action = np.round(action)
        main_joints = action[1:4]
        hand_step = np.concatenate(
            ([action[0]], main_joints, np.array(main_joints) * (1 / 3))
        )
        arm_step = np.array([0.0, 0.0, 0.0, 0.0, action[4], 0.0])
        logging.info("Hand Step:" + str(hand_step))
        logging.info("Arm Step:" + str(arm_step))
        self.simController.step_hand_config(hand_step)
        d_t = self.clock.tick(self.target_fps) / 1000.0

        # Update current state:
        self.current_hand_config = self.simController.get_hand_config()
        self.current_obj_count = self.simController.get_object_count()
        reward, done = self.get_current_reward()

        # Keep track of action encoding
        self.previous_actions.append(np.array(action) / 7)
        self.previous_actions.popleft()
        action_state = np.array(self.previous_actions).copy().reshape((40,))
        state = self.simController.get_state_encoding(action_state)

        if self.detailed_training:
            self.current_episodic_rew += reward
        return state, reward, done, {}

    def close(self):
        """Closes simulation"""
        if self.sim_process:
            self.sim_process.kill()

    def plot_detailed_training_data(self):
        """Plots detailed training data collected during step() iterations."""
        plotters = []
        params = self.plot_params
        _range = None
        labels = []
        _path = (
            os.path.join(self.performance_save_path, "Detailed_Training")
            if not self.is_val
            else os.path.join(self.performance_save_path, "Detailed_Validation")
        )
        if not os.path.exists(_path):
            os.makedirs(_path)
        _path = os.path.join(_path, f"{self.num_detailed_files}.png")
        """ Update the Current Values for each of our potential Data Plots"""
        if self.model_performance[SceneDifficulty.EASY]["use_data"]:
            try:
                easy_data = self.model_performance[SceneDifficulty.EASY]["performance"][
                    -25::
                ]
                self.model_performance[SceneDifficulty.EASY]["history"].append(
                    sum(easy_data) / len(easy_data)
                )
                self.model_performance[SceneDifficulty.EASY]["avg_vel"].append(
                    sum(self.current_run_vel[SceneDifficulty.EASY])
                    / len(self.current_run_vel[SceneDifficulty.EASY])
                )
                self.model_performance[SceneDifficulty.EASY]["avg_ke"].append(
                    sum(self.current_run_ke[SceneDifficulty.EASY])
                    / len(self.current_run_vel[SceneDifficulty.EASY])
                )
                self.model_performance[SceneDifficulty.EASY]["avg_rew"].append(
                    sum(self.episodic_reward[SceneDifficulty.EASY])
                    / len(self.current_run_vel[SceneDifficulty.EASY])
                )
                plotters.append(self.model_performance[SceneDifficulty.EASY])
                labels.append("Easy")
                _range = range(
                    len(self.model_performance[SceneDifficulty.EASY]["history"])
                )
            except Exception as e:
                logging.error(f"Error with Saving Easy Data... {e}")
        if self.model_performance[SceneDifficulty.MEDIUM]["use_data"]:
            try:
                med_data = self.model_performance[SceneDifficulty.MEDIUM][
                    "performance"
                ][-25::]
                self.model_performance[SceneDifficulty.MEDIUM]["history"].append(
                    sum(med_data) / len(med_data)
                )
                self.model_performance[SceneDifficulty.MEDIUM]["avg_vel"].append(
                    sum(self.current_run_vel[SceneDifficulty.MEDIUM])
                    / len(self.current_run_vel[SceneDifficulty.MEDIUM])
                )
                self.model_performance[SceneDifficulty.MEDIUM]["avg_ke"].append(
                    sum(self.current_run_ke[SceneDifficulty.MEDIUM])
                    / len(self.current_run_vel[SceneDifficulty.MEDIUM])
                )
                self.model_performance[SceneDifficulty.MEDIUM]["avg_rew"].append(
                    sum(self.episodic_reward[SceneDifficulty.MEDIUM])
                    / len(self.current_run_vel[SceneDifficulty.MEDIUM])
                )
                plotters.append(self.model_performance[SceneDifficulty.MEDIUM])
                labels.append("Medium")
                _range = range(
                    len(self.model_performance[SceneDifficulty.MEDIUM]["history"])
                )
            except Exception as e:
                logging.error(f"Error with Saving Medium Data...{e}")

        if self.model_performance[SceneDifficulty.HARD]["use_data"]:
            try:
                hard_data = self.model_performance[SceneDifficulty.HARD]["performance"][
                    -25::
                ]
                self.model_performance[SceneDifficulty.HARD]["history"].append(
                    sum(hard_data) / len(hard_data)
                )
                self.model_performance[SceneDifficulty.HARD]["avg_vel"].append(
                    sum(self.current_run_vel[SceneDifficulty.HARD])
                    / len(self.current_run_vel[SceneDifficulty.HARD])
                )
                self.model_performance[SceneDifficulty.HARD]["avg_ke"].append(
                    sum(self.current_run_ke[SceneDifficulty.HARD])
                    / len(self.current_run_vel[SceneDifficulty.HARD])
                )
                self.model_performance[SceneDifficulty.HARD]["avg_rew"].append(
                    sum(self.episodic_reward[SceneDifficulty.HARD])
                    / len(self.current_run_vel[SceneDifficulty.HARD])
                )
                plotters.append(self.model_performance[SceneDifficulty.HARD])
                labels.append("Hard")
                _range = range(
                    len(self.model_performance[SceneDifficulty.HARD]["history"])
                )
            except Exception as e:
                logging.error(f"Error with Saving Hard Data... {e}")
        if _range != None:
            plot_results(
                plotters,
                params=params,
                y_labels=labels,
                range_spc=_range,
                file_name=_path,
            )
            logging.info(
                f'\n{"-"*10}\n{"-"*10}\n{"-"*10}\nSaving Performance Data\n{"-"*10}\n{"-"*10}\n{"-"*10}'
            )
        self.num_detailed_files += 1
        self.reset_data_cntrs()

    def reset(self):
        self.step_len = 0
        self.post_drop = 0
        self.simController.reset_arm_config()
        for h in self.episodic_penalty_hooks.values():
            h.reset()

        # Logging: updates the episodic reward list with the previous episode's reward
        if self.detailed_training:
            if self.current_episodic_rew != 0:
                self.episodic_reward[self.current_index_difficulty].append(
                    self.current_episodic_rew
                )
            self.current_episodic_rew = 0

        valid = False
        self.counter["episodes"] += 1

        self.simController.progress_bar = 0
        self.simController.drop_detected = False

        self.reset_counter += 1
        if (self.reset_counter) % 10 == 0 or self.restart_check:
            try:
                logging.info(
                    f"Current Accuracy: {(sum(self.successes)/len(self.successes)):.2%}"
                )
            except:
                pass
            self.restart_check = False
            self.simController.is_sim_working()

        self.current_obj_count = self.simController.get_object_count()
        if self.current_obj_count > 1 and not self.test:
            self.current_hand_config = self.simController.get_hand_config()
            logging.info("Using the Same Scene...")
            valid = True

        reset_atmpt_cntr = 0

        while not valid:
            if reset_atmpt_cntr + 1 % 16 == 0:
                logging.debug("Reset Counter Casusing Reset on Sim...")
                self.restart_trigger()
            reset_atmpt_cntr += 1

            new_scene, self.current_index_difficulty = self.scene_loader.get_random_scene()
            
            if self.set_sim_scene(new_scene):
                if self.current_obj_count == self.current_obj_count or self.current_obj_count > 1:
                    valid = True
                    logging.info("Scene Validated. Starting Dropping...")
            else:
                # self.bad_scenes.add(self.configs_index)
                pass

        self.goal_obj_count = self.current_obj_count - 1
        self.previous_actions.clear()
        for _ in range(8):
            self.previous_actions.append(np.zeros(shape=(5,)).astype(float))
        action_state = np.array(self.previous_actions).copy().reshape((40,))
        state = self.simController.get_state_encoding(action_state)

        return state, {}
    
    def set_sim_scene(self, scene):
        hand_config, ball_locations = self.scene_loader.get_scene_config(scene)
        success = self.simController.setup_scene(hand_config, ball_locations)
        if success:
            time.sleep(3)
            self.current_hand_config = self.simController.get_hand_config()
            self.current_obj_count = self.simController.get_object_count()
            logging.info("Scene set successfully.")
            return True
        else:
            logging.error("Failed to set scene.")
            return False
    
    def drop_status_hook(self) -> float:
        if self.current_obj_count > self.goal_obj_count:
            return -self.predrop_penalty, {"status": "predrop"}
        elif self.current_obj_count == self.goal_obj_count and self.post_drop < self.success_step_count:
            self.post_drop += 1
            return self.post_drop_reward, {"status": "drop_detected"}
        elif (self.current_obj_count == self.goal_obj_count 
              and self.post_drop >= self.success_step_count 
              and self.__velocity_penalty_hook.info["cum_velocity"] < 0.001):
            return self.success_reward, {"status": "success"}
        elif self.current_obj_count == self.goal_obj_count and self.post_drop < 21:
            self.post_drop += 1
            return 0, {"status": "stabilizing"}
        else:
            return -self.failure_penalty, {"status": "failure"}
        
    def step_bonus_hook(self) -> float:
        bonus = 0
        if self.detailed_training:
            bonus += int(self.current_index_difficulty) * 5
        if self.is_val:
            bonus += 10
        return bonus, {"bonus": bonus}

    def palm_displacement_hook(self) -> float:
        palm_displacement = self.simController.get_angle_from_normal()
        scaled_palm_displacement = -max(
            palm_displacement * self.palm_angle_penalty_scale_factor,
            5,
        )
        return scaled_palm_displacement, {"palm_displacement": palm_displacement}

    def terminations_hook(self) -> float:
        terminations = self.simController.get_terminations()
        return sum(terminations), {"terminations": terminations}

    def limit_signals_hook(self) -> float:
        limit_signals = self.simController.get_limit_signals()
        return sum(limit_signals) / 2, {"limit_signals": limit_signals}

    def velocity_penalty_hook(self) -> float:
        cum_velocity = self.simController.get_cumulative_velocity()[-1] ** 2
        hand_ke = self.simController.get_KE()

        if self.detailed_training:
            self.current_run_vel[self.current_index_difficulty].append(cum_velocity)
            self.current_run_ke[self.current_index_difficulty].append(hand_ke)

        penalty = (
            hand_ke * self.movement_penalty_scale_factor + cum_velocity
            if self.post_drop == 0
            else (hand_ke * self.movement_penalty_scale_factor + cum_velocity) ** 2
        )
        
        return penalty, {
            "cum_velocity": cum_velocity,
            "hand_kinetic_energy": hand_ke
        }
    
    def get_current_reward(self):
        # Get penalty values
        penalty = 25 + sum([h.get() for h in self.episodic_penalty_hooks.values()])

        self.simController.reset_termination_signals()
        self.simController.reset_limit_signals()

        reward = -penalty
        done = False

        terminations = self.__termination_hook.info["terminations"]
        drop_status = self.__drop_status_hook.info["status"]

        if all(terminations):
            logging.info("Hand or Palm Penalty...")
            done = True
            reward -= self.termination_penalty
        elif drop_status == "predrop":
            done = False
        elif drop_status == "drop_detected":
            logging.info("Drop Detected...")
            done = False
        elif drop_status == "success":
            logging.info("Successful Drop!")
            done = True 
            reward += self.success_reward
            if self.detailed_training:
                self.model_performance[self.current_index_difficulty]["performance"].append(1)
        elif drop_status == "stabilizing":
            logging.info("Hand Stabilization...")
            done = False
        else:  # failure
            logging.info("Failed at Dropping Single Object")
            done = True
            reward -= self.failure_penalty
            if self.detailed_training:
                self.model_performance[self.current_index_difficulty]["performance"].append(0)

        return reward / 150, done

    # def get_current_reward(self):
    #     # Get penalty values
    #     penalty = -2 + sum([h.get() for h in self.episodic_penalty_hooks.values()])

    #     self.simController.reset_termination_signals()
    #     self.simController.reset_limit_signals()

    #     reward = -penalty
    #     done = False

    #     terminations = self.__termination_hook.info["terminations"]
    #     cum_velocity = self.__velocity_penalty_hook.info["cum_velocity"]
    #     step_bonus = self.__step_bonus_hook.info["bonus"]
    #     drop_status = self.__drop_status_hook.info["status"]

    #     if all(terminations):
    #         logging.info("Hand or Palm Penalty...")
    #         done = True
    #         self.restart_check = True
    #         reward -= self.termination_penalty

    #     elif (self.step_len > 21 + step_bonus and self.post_drop < 1) or self.post_drop > 35:
    #         logging.info("Taking too Long to Drop...")
    #         done = True
    #         self.successes.append(False)
    #         if self.detailed_training:
    #             self.model_performance[self.current_index_difficulty]["performance"].append(0)
    #         self.restart_check = True
    #         reward -= self.long_drop_penalty

    #     elif drop_status == "predrop":
    #         done = False

    #     elif drop_status == "drop_detected":
    #         logging.info("Drop Detected...")
    #         done = False

    #     elif drop_status == "success":
    #         logging.info("Successful Drop!")
    #         done = True
    #         self.successes.append(True)
    #         if self.detailed_training:
    #             self.model_performance[self.current_index_difficulty]["performance"].append(1)

    #     elif drop_status == "stabilizing":
    #         logging.info("Hand Stabilization...")
    #         done = False

    #     else:  # failure
    #         logging.info("Failed at Dropping Single Object")
    #         done = True
    #         self.successes.append(False)
    #         if self.detailed_training:
    #             self.model_performance[self.current_index_difficulty]["performance"].append(0)

    #     return reward / 150, done
    
    def render(self, mode="human"):
        pass

    def close(self):
        if self.sim_process:
            self.sim_process.kill()

    """
    Cmd for running Validation on a seperate process
    """

    @staticmethod
    def validate_agent(algo, model_path, use_sbl, save_path=None, plot_data=False):
        model_files = [f for f in os.listdir(model_path) if ".zip" in f]
        for f in model_files:
            most_recent_model = max(
                model_files,
                key=lambda f: os.path.getctime(os.path.join(CONTROL_DROP_DIR, f)),
            )
        _args = [
            "python",
            os.path.join(
                CONTROL_DROP_DIR,
                "control_dropping",
                "validation.py",
            ),
            "--algo",
            algo,
            "--model_path",
            most_recent_model,
        ]
        if not use_sbl:
            _args.append("--use_sbl")
            _args.append(False)
        if save_path != None:
            _args.append("--data_save_path")
            _args.append(save_path)
        if plot_data:
            _args.append("--plot_val")
            _args.append(True)
        return subprocess.Popen(_args)


class BerretHandGymRayLibWrapper(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}

        self.cfg = config
        self.episode_len = 100
        self.current_step = 0

        # Initialize the agent (your custom simulator)
        self.agent = BerrettHandGym(
            test=config.get("test", False),
            cluster_index=config.get("cluster_index", 3),
            sim_port=config.get("sim_port", None),
            object_type=config.get("object_type", "Sphere"),
            object_quantity=config.get("object_quantity", 7),
            detailed_training=config.get("detailed_training", False),
            detailed_save_dir=config.get("detailed_save_dir", None),
            plot_params=config.get(
                "plot_params", ["history", "avg_ke", "avg_vel", "avg_rew"]
            ),
            difficulties=config.get(
                "difficulties",
                [SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD],
            ),
            algorithm=config.get("algorithm", "RecurrentPPO"),
            is_val=config.get("is_val", False),
            worker_idx=config.get("worker_idx"),
        )

        self.action_buffer = SensorBuffer(
            name="action_buffer",
            data_shape=(5,),  # Assuming actions are 5-dimensional
            time_seq_len=8,  # Store 8 previous actions
            update_fnc=self._update_action_buffer,
        )

        # Define action and observation spaces
        self._action_space = self.agent.action_space
        self._observation_space = self.agent.observation_space

        self.current_action = np.zeros(5)

    def _update_action_buffer(self):
        return self.current_action

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_action = np.zeros(5)
        self.current_step = 0
        obs, info = self.agent.reset()
        self.action_buffer.reset_sensor()

        obs["actions"] = self.action_buffer.get_updated_sensor_reading()
        return obs, info  # Gym API expects only obs, not (obs, info)

    def step(self, action):
        self.current_action = action
        obs, reward, done, info = self.agent.step(action)
        obs["actions"] = self.action_buffer.get_updated_sensor_reading()

        self.current_step += 1
        terminated = done
        truncated = self.current_step >= self.episode_len
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.agent.close()

    
    def current_accuracy(self, window=32):
        return sum(self.agent.successes[-window:]) / len(self.agent.successes[-window:])

    @property
    def observation_space(self) -> gym.Space:
        """Returns the observation space for each agent.

        Note: samples from the observation space need to be preprocessed into a
            `MultiEnvDict` before being used by a policy.

        Returns:
            The observation space for each environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the action space for each agent.

        Note: samples from the action space need to be preprocessed into a
            `MultiEnvDict` before being passed to `send_actions`.

        Returns:
            The observation space for each environment.
        """
        return self._action_space
    
    @property
    def current_obj_count(self) -> int:
        return self.agent.current_obj_count
    
    @property
    def goal_obj_count(self) -> int:
        return self.agent.goal_obj_count

    @property
    def current_index_difficulty(self) -> Union[SceneDifficulty, int]:
        return self.agent.current_index_difficulty

    @property
    def post_drop(self) -> int:
        return self.agent.post_drop
    
    @property
    def current_episodic_rew(self) -> int:
        return self.agent.current_episodic_rew
    
    @property
    def current_run_vel(self) -> int:
        return self.agent.current_run_vel

# Vectorized Environment
from gymnasium.vector import VectorEnv


class VectorizedBerrettHandGym(VectorEnv):
    """This class is designed to implement control of the Berret Hand through use of SimController in a vectorized manner"""

    def __init__(
        self,
        sim_ports: List[
            int
        ],  # Ports to connect the sim on, or None if auto-create environment.
        test=False,  # If trying to iterate through grasping data, set to True
        cluster_index=3,  # Used for setting the data cluster for use in training
        object_type="Sphere",  # This is for the Sim to generate the Specific type of Object for Grasp
        object_quantity=7,  # Number of Objects being generated
        detailed_training=False,  # Controls the type of data and data detailing being performed during Training
        detailed_save_dir=None,  # Overrides the Path of Trial Saving
        plot_params=[
            "history",
            "avg_ke",
            "avg_vel",
            "avg_rew",
        ],  # Specify which parameters to include in the Figures
        difficulties=[
            SceneDifficulty.EASY,
            SceneDifficulty.MEDIUM,
            SceneDifficulty.HARD,
        ],  # Difficulties to Train on
        algorithm="RecurrentPPO",  # Algorithm for loading in validation
        is_val=False,  # is Running Validation
    ):
        """
        - test: If trying to iterate through grasping data, set to True
        - cluster_index: Used for setting the data cluster for use in training
        - sim_ports: # Ports to connect the sim on, or None if auto-create environment.
        - object_type: This is for the Sim to generate the Specific type of Object for Grasp
        - object_quantity=7: Number of Objects being generated
        - detailed_training: Controls the type of data and data detailing being performed during Training
        - detailed_save_dir: Overrides the Path of Trial Saving
        - plot_params: Specify which parameters to include in the Figures
        - difficulties=[SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD], :Difficulties to Train on
        - algorithm: Algorithm for loading in validation
        - is_val: is Running Validation
        """
        # Initialize the first environment to get the observation_space and action_space
        temp_env = BerrettHandGym(
            sim_port=sim_ports[0],
        )
        logging.info("Successfully Created Env.")
        super().__init__(
            num_envs=len(sim_ports),
            observation_space=temp_env.observation_space,
            action_space=temp_env.action_space,
        )

        # Now initialize the rest of the environments
        self.envs = [temp_env] + [
            BerrettHandGym(
                sim_port=port,
                test=test,
                cluster_index=cluster_index,
                object_type=object_type,
                object_quantity=object_quantity,
                detailed_training=detailed_training,
                detailed_save_dir=detailed_save_dir,
                plot_params=plot_params,
                difficulties=difficulties,
                algorithm=algorithm,
                is_val=False,
            )
            for port in sim_ports[1:]
        ]

    def step(self, actions):
        """
        Performs the step operation for all the environments with the corresponding action.

        Parameters:
        actions (List[Action]): List of actions for each environment.

        Returns:
        Tuple: Observations, Rewards, Dones, and Info as specified by the gym API.
        """
        assert len(actions) == self.num_envs

        # Initialize empty lists for observations, rewards, dones and infos
        observations, rewards, dones, infos = [], [], [], []

        # Perform step in each environment with corresponding action
        for env, action in zip(self.envs, actions):
            obs, rew, done, info = env.step(action)
            observations.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        # Convert to numpy arrays and return
        return np.array(observations), np.array(rewards), np.array(dones), infos

    def reset(self):
        """
        Resets all the environments.

        Returns:
        List[Observation]: List of observations for each environment after resetting.
        """
        return np.array([env.reset() for env in self.envs])

    def close_extras(self, **kwargs):
        """Close all environments."""
        pass
        # for env in self.envs:
        #    env.close()
