import sys
import os
import numpy as np
import ray
import logging

sys.path.append("/media/rpal/Drive_10TB/John/Control Drop/Reinforcement Learning")

from src.SimController.sim_controller import SensorBuffer, DictSensorBuffer
from src.RL.control_dropping_env import BerretHandGymRayLibWrapper
from src.Utils.env_utils import AsyncVectorEnv

MOCK_T_BUFFER = 5
SIZE = 24


FINGER_KEYS = ("f1", "f2", "f3", "palm")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_WORKERS = 8
ENVS_PER_WORKER = 1
SAVE_INTERVAL = 2
NUM_INTERACTIONS = 1000
CHECKPOINT_DIR = (
    "/media/rpal/Drive_10TB/John/Control Drop/Reinforcement Learning/ray_checkpoints/"
)

# Initialize Ray
ray.init(local_mode=True)
logger.info("Ray initialized")
os.environ["SIM_GUI"] = "false"


def test_sensor_buffer():
    pass


def test_dict_sensor_buffer():
    hand_values_buffer = DictSensorBuffer(
        "hand_values",
        {k: (SIZE,) for k in FINGER_KEYS},
        MOCK_T_BUFFER,
        lambda: {
            k: np.random.sample(
                size=(SIZE),
            )
            for k in FINGER_KEYS
        },
    )

    readings = hand_values_buffer.get_updated_sensor_reading()
    for key in FINGER_KEYS:
        assert MOCK_T_BUFFER in readings[key].shape and SIZE in readings[key].shape

    print("test_dict_sensor_buffer All tests passed.")


def test_create_async_vec_env():
    # Ensure we can create an instance of the vec environment
    _ = AsyncVectorEnv(
        lambda: BerretHandGymRayLibWrapper(
            {
                "test": False,
                "cluster_index": 3,
                "sim_port": None,
                "object_type": "Sphere",
                "object_quantity": 7,
                "detailed_training": False,
                "detailed_save_dir": None,
                "plot_params": ["history", "avg_ke", "avg_vel", "avg_rew"],
                "is_val": False,
            }
        ),
        num_envs=NUM_WORKERS,
    )

    print("test_async_vec_env All test passed.")


test_dict_sensor_buffer()
test_create_async_vec_env()
