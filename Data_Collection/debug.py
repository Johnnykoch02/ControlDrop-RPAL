# ## Initialization:: Use env stbl3 or raylib
import os

CONTROL_DROP_DIR = os.environ["CONTROL_DROP_DIR"]

import torch as th
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from src.RL.Networks.ActorCriticNetwork import RayDroppingModel
from src.RL.control_dropping_env import BerretHandGymRayLibWrapper, T_buffer
from ray.tune.logger import pretty_print
from src.Utils.env_utils import AsyncVectorEnv

# from pprint import p/
import ray
import logging


from stable_baselines3.common.env_checker import check_env

# from ray.rllib.utils.pre_checks.env import ch


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

th.set_default_device("cuda")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
DATA_SAVE_PATH = os.path.join(os.getcwd(), "..", "Data_Collection")
NUM_LAYERS_TRANSFORMER = 8
NUM_RESIDUALS = 4
EPOCHS = 1000
VEC_ENCODING_SIZE = 256

NUM_WORKERS = 8
ENVS_PER_WORKER = 1
SAVE_INTERVAL = 2
NUM_INTERACTIONS = 1000
CHECKPOINT_DIR = os.path.join(CONTROL_DROP_DIR, "control_dropping/checkpoints/")

# Initialize Ray
ray.init(local_mode=True)
logger.info("Ray initialized")

# Register the custom model and environment
ModelCatalog.register_custom_model("RayDroppingModel", RayDroppingModel)

register_env("ControlDroppingEnv", lambda config: BerretHandGymRayLibWrapper(config))
register_env(
    "AsyncControlDroppingEnv",
    lambda _: AsyncVectorEnv(
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
    ),
)

logger.info("Custom model and environment registered")
device = "cpu" if not th.cuda.is_available() else "cuda"
# Define the PPO configuration
config = (
    PPOConfig()
    .environment(
        env="AsyncControlDroppingEnv",
    )
    .experimental(_disable_preprocessor_api=True)
    .framework("torch")
    .rollouts(
        rollout_fragment_length="auto",
        batch_mode="truncate_episodes",
        observation_filter="NoFilter",
        num_rollout_workers=NUM_WORKERS,
        num_envs_per_worker=ENVS_PER_WORKER,
    )
    .resources(
        num_gpus=1 if device == "cuda" else 0,
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
    )
    .exploration(
        explore=True,
        exploration_config={"type": "StochasticSampling"},
    )
    .evaluation(
        evaluation_interval=None,
        evaluation_duration=1,
        evaluation_duration_unit="episodes",
        evaluation_config={"explore": False},
    )
    .training(
        train_batch_size=1048,
        sgd_minibatch_size=128,
        num_sgd_iter=16,
        gamma=0.9,
        optimizer={"lr": 5e-5},
    )
    .debugging(log_level="INFO")
)

config.model = {
    "custom_model": "RayDroppingModel",
    "custom_model_config": {
        "action_encoder_ff_dim": 512,
        "action_encoder_num_action": 5 * 2,
        "action_encoder_head": 4,
        "temporal_dim": T_buffer,
        "obj_encoder_vec_encoding_size": VEC_ENCODING_SIZE,
        "obj_encoder_num_tsf_layer": NUM_LAYERS_TRANSFORMER,
        "obj_encoder_load_path": os.path.join(
            CONTROL_DROP_DIR,
            f"pretrained_object_encoder-{NUM_LAYERS_TRANSFORMER}_layers-{NUM_RESIDUALS}_residuals-{VEC_ENCODING_SIZE}-vecencoding_size-{EPOCHS}_epochs.pt",
        ),
        "device": device,
        "max_seq_len": 1,
    },
    "max_seq_len": 1,
}

# Create the PPO trainer
try:
    trainer = config.build()
    logger.info("PPO trainer created successfully")
except Exception as e:
    logger.error(f"Failed to create PPO trainer: {e}")
    raise

# Load an existing model if available
if os.path.exists(CHECKPOINT_DIR):
    try:
        latest_checkpoint = max(
            [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR)],
            key=os.path.getctime,
        )
        trainer.restore(latest_checkpoint)
        logger.info(f"Loaded from {latest_checkpoint}")
    except Exception as e:
        logger.error(f"Failed to load from {CHECKPOINT_DIR}: {e}")

# Create a checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Train the model
for i in range(NUM_INTERACTIONS):
    try:
        result = trainer.train()
        logger.info(f"Iteration {i + 1} result: {result}")
        print(pretty_print(result))
    except Exception as e:
        logger.error(f"Error during training iteration {i + 1}: {e}")
        break

    # Save the model periodically
    if (i + 1) % SAVE_INTERVAL == 0:
        try:
            checkpoint_path = trainer.save(CHECKPOINT_DIR)
            logger.info(f"Model saved at: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save model at iteration {i + 1}: {e}")

# Manual test loop
print("Finished training. Running manual test/inference loop.")

env = BerretHandGymRayLibWrapper(config["env_config"])
obs, info = env.reset()
done = False

# Run one iteration until done
print(f"BerretHandGymRayLibWrapper with {config['env_config']}")
while not done:
    action = trainer.compute_single_action(obs)
    next_obs, reward, done, truncated, _ = env.step(action)
    print(f"Obs: {obs}, Action: {action}")
    obs = next_obs

print("Finished successfully.")
ray.shutdown()

### DEBUG:
# import os
# import sys
# # os.chdir("..")
# import torch as th
# from gymnasium.spaces import Box
# from src.RL.control_dropping_env import BerrettHandGym, T_buffer
# from math import inf, radians, degrees
# # from stable_baselines3 import PPO, A2C

# DATA_SAVE_PATH = os.path.join(os.getcwd(), '..', 'Data_Collection')
# MODEL_PATH = "/media/rpal/Drive_10TB/John/Control Drop/control_droppingL/Training/Checkpoints/TransformerFeatureEncoder/Expert_rl_5000_steps.zip"

# # model = PPO.load(MODEL_PATH)
# # env = BerrettHandGym(detailed_training=True, is_val=True)
# # model.set_env(env)

# from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

# class CustomPPOTorchPolicy(PPOTorchPolicy):
#     def loss(self, model, dist_class, train_batch):
#         train_batch = {k: v.to(self.device) for k, v in train_batch.items()}
#         return super().loss(model, dist_class, train_batch)


# ## Params
# NUM_LAYERS_TRANSFORMER = 8
# NUM_RESIDUALS = 4
# EPOCHS = 1000
# VEC_ENCODING_SIZE = 256

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import os
# import ray
# from ray import tune
# from ray.rllib.algorithms.ppo import PPO, PPOConfig
# from ray.rllib.models import ModelCatalog
# from ray.tune.registry import register_env

# from src.RL.Networks.ActorCriticNetwork import RayDroppingModel
# from src.RL.control_dropping_env import BerretHandGymRayLibWrapper, T_buffer, default_action_space, base_observation_space
# from src.RL.Networks.ActorCriticNetwork import RayDroppingModel
# from src.RL.control_dropping_env import BerrettHandGym, T_buffer

# from ray.rllib.utils.typing import ModelConfigDict

# # Register the custom model and environment
# ModelCatalog.register_custom_model("RayDroppingModel", RayDroppingModel)
# register_env("ControlDroppingEnv", lambda config: BerretHandGymRayLibWrapper(config))

# NUM_WORKERS = 1
# ENVS_PER_WORKER = 1
# SAVE_INTERVAL = 3
# NUM_INTERACTIONS = 1000

# ray.init(local_mode=True)

# # Define the PPO configuration
# ppo_config = dict({
#     "env": "ControlDroppingEnv",
#     "env_config": {
#         "test": False,
#         "cluster_index": 3,
#         "sim_port": None,
#         "object_type": "Sphere",
#         "object_quantity": 7,
#         "detailed_training": False,
#         "detailed_save_dir": None,
#         "plot_params": ["history", "avg_ke", "avg_vel", "avg_rew"],
#         "is_val": False,
#     },
#     "model": {
#         "custom_model": "RayDroppingModel",
#         "custom_model_config": {
#             "action_encoder_ff_dim": 512,
#             "action_encoder_num_action": 5*2,
#             "action_encoder_head": 4,
#             "temporal_dim": T_buffer,
#             "obj_encoder_vec_encoding_size": VEC_ENCODING_SIZE,
#             "obj_encoder_num_tsf_layer": NUM_LAYERS_TRANSFORMER,
#             "obj_encoder_load_path": f"/media/rpal/Drive_10TB/John/Control Drop/pretrained_object_encoder-{NUM_LAYERS_TRANSFORMER}_layers-{NUM_RESIDUALS}_residuals-{VEC_ENCODING_SIZE}-vecencoding_size-{EPOCHS}_epochs.pt",
#             "device": "cuda",
#         },
#     },
#      "policy_class": CustomPPOTorchPolicy,
#     "num_workers": NUM_WORKERS,
#     "num_envs_per_worker": ENVS_PER_WORKER,
#     "framework": "torch",

#     # Rollout settings
#     "rollout_fragment_length": "auto",
#     "batch_mode": "truncate_episodes",
#     "observation_filter": "NoFilter",

#     # Training settings
#     "train_batch_size": 1048,
#     "sgd_minibatch_size": 128,
#     "num_sgd_iter": 16,
#     "lr": 5e-5,
#     "lr_schedule": None,
#     "clip_param": 0.2,
#     "vf_clip_param": tune.grid_search([1.0, 10.0]),
#     "vf_loss_coeff": tune.grid_search([0.7, 1.0]),
#     "entropy_coeff": tune.grid_search([0.0, .01]),
#     "entropy_coeff_schedule": None,
#     "kl_coeff": tune.grid_search([0.0, 0.001]),
#     "kl_target": tune.grid_search([0.001, 0.05]),

#     # Exploration settings
#     "explore": True,
#     "exploration_config": {
#         "type": "StochasticSampling",
#     },

#     # Evaluation settings
#     "evaluation_interval": None,
#     "evaluation_duration": 1,
#     "evaluation_duration_unit": "episodes",
#     "evaluation_config": {
#         "explore": False,
#     },

#     # Debugging settings
#     "log_level": "INFO",
#     # "callbacks": {
#     #     "on_episode_start": None,
#     #     "on_episode_step": None,
#     #     "on_episode_end": None,
#     #     "on_sample_end": None,
#     #     "on_train_result": None,
#     # },
#     "ignore_worker_failures": True,
#     "recreate_failed_workers": False,

#     # Parallelism settings
#     "num_gpus": 1,
#     "num_cpus_per_worker": 1,
#     "num_gpus_per_worker": 1,
# })
# # # PPOConfig()
# # # Create the PPO trainer

# # ppo_config = (
# #     PPOConfig()
# #     .environment("ControlDroppingEnv", action_space=default_action_space, observation_space=base_observation_space, clip_rewards=True)
# #     .framework("torch")
# #     .resources(num_gpus=1)
# #     .env_runners(enable_connectors=False)
# #     .reporting(keep_per_episode_custom_metrics=True)
# #     )

# # ppo_config.model = {
# #         "custom_model": "RayDroppingModel",
# #         "custom_model_config": {
# #             "action_encoder_ff_dim": 512,
# #             "action_encoder_num_action": 5*2,
# #             "action_encoder_head": 4,
# #             "temporal_dim": T_buffer,
# #             "obj_encoder_vec_encoding_size": VEC_ENCODING_SIZE,
# #             "obj_encoder_num_tsf_layer": NUM_LAYERS_TRANSFORMER,
# #             "obj_encoder_load_path": f"/media/rpal/Drive_10TB/John/Control Drop/pretrained_object_encoder-{NUM_LAYERS_TRANSFORMER}_layers-{NUM_RESIDUALS}_residuals-{VEC_ENCODING_SIZE}-vecencoding_size-{EPOCHS}_epochs.pt",
# #             "device": "cuda",
# #         },
# #         "max_seq_len": 250,
# #     }

# # ppo_config.env_config = {
# #         "test": False,
# #         "cluster_index": 3,
# #         "sim_port": None,
# #         "object_type": "Sphere",
# #         "object_quantity": 7,
# #         "detailed_training": False,
# #         "detailed_save_dir": None,
# #         "plot_params": ["history", "avg_ke", "avg_vel", "avg_rew"],
# #         "is_val": False,
# #     }

# # ppo_config.num_gpus = 2
# # ppo_config.num_cpus_per_worker = 1
# # ppo_config.num_gpus_per_worker = 0


# trainer = PPO(config=ppo_config)


# # Load an existing model if available
# checkpoint_path = "/media/rpal/Drive_10TB/John/Control Drop/control_droppingheckpoints/"

# if os.path.exists(checkpoint_path):
#     try:
#         trainer.restore(checkpoint_path)
#         print(f"Loaded from {checkpoint_path}")
#     except:
#         pass

# # Train the model
# for _ in range(NUM_INTERACTIONS):
#     result = trainer.train()
#     print(result)

#     # Save the model periodically
#     if (_ + 1) % SAVE_INTERVAL == 0:
#         checkpoint_path = trainer.save()
#         print(f"Model saved at: {checkpoint_path}")
