import os

CONTROL_DROP_DIR = os.environ["CONTROL_DROP_DIR"]

import warnings
import torch as th
import gymnasium as gym
from torchrl.envs import GymWrapper
import logging
import argparse
from stable_baselines3 import PPO
from control_dropping_rpal.RL.control_dropping_env import (
    BerretHandGymRayLibWrapper,
    T_buffer,
)
from control_dropping_rpal.RL.Networks.ActorCriticNetwork import ControlDropPolicy
from control_dropping_rpal.Utils.training_utils import SaveOnBestRewardCallback

warnings.filterwarnings("ignore")
th.set_default_device("cuda")

from gymnasium.envs.registration import register

register(
    id='BerretHandGymRayLib-v0',
    entry_point=lambda **kwargs: BerretHandGymRayLibWrapper(kwargs.get('env_config', {})),
    max_episode_steps=100,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENV_CONFIG = dict(
    detailed_training=True,
    object_quantity=7,
)

NUM_LAYERS_TRANSFORMER = 6
NUM_RESIDUALS = 3
EPOCHS = 1000
VEC_ENCODING_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser(description="RL Control Dropping Training")
    parser.add_argument(
        "--embedding_chkpoint",
        type=str,
        required=True,
        help="Embedding checkpoint file name",
    )
    parser.add_argument(
        "--rl_chkpoint",
        type=str,
        default=None,
        required=False,
        help="Embedding checkpoint file name",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="dynamix_control_drop",
        help="Experiment name",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Freeze the pretrained encoder"
    )

    parser.add_argument(
        "--unlock_encoder", action="store_true", help="Forces unlocked pretrained encoder"
    )

    # PPO hyperparameters
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to run in parallel",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--n_steps", type=int, default=2048, help="Number of steps per update"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epoch when optimizing the surrogate loss",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="Factor for trade-off of bias vs variance for GAE",
    )
    parser.add_argument(
        "--clip_range", type=float, default=0.2, help="Clipping parameter"
    )
    parser.add_argument(
        "--clip_range_vf",
        type=float,
        default=None,
        help="Clipping parameter for the value function",
    )
    parser.add_argument(
        "--normalize_advantage", action="store_true", help="Normalize advantage"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.0, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum value for gradient clipping",
    )
    parser.add_argument(
        "--use_sde",
        action="store_true",
        help="Use generalized State Dependent Exploration",
    )
    parser.add_argument(
        "--sde_sample_freq",
        type=int,
        default=-1,
        help="Sample a new noise matrix every n steps when using gSDE",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="Limit the KL divergence between updates",
    )
    parser.add_argument(
        "--stats_window_size",
        type=int,
        default=100,
        help="Window size for the rollout logging",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000000,
        help="Total timesteps for training",
    )
    parser.add_argument("--save_freq", type=int, default=10000, help="Save frequency")
    return parser.parse_args()


def main(args):
    DATA_SAVE_PATH = os.path.join(CONTROL_DROP_DIR, "Data_Collection")

    model_config = {
        "temporal_dim": T_buffer,
        "obj_encoder_vec_encoding_size": VEC_ENCODING_SIZE,
        "obj_encoder_num_tsf_layer": NUM_LAYERS_TRANSFORMER,
        "obj_encoder_load_path": args.embedding_chkpoint,
        "obj_encoder_freeze_params": args.freeze_encoder,
        "device": "cuda" if th.cuda.is_available() else "cpu",
    }

    logging.info(f"Model Config: {model_config}")

    # Create the log dir and tensorboard log directory
    log_dir = os.path.join("./rl_logs", args.experiment_name)
    if os.path.exists(os.path.join(log_dir)):
        experiment_cnt = len(
            [i for i in os.listdir("./rl_logs") if i.startswith(args.experiment_name)]
        )
        log_dir = os.path.join("./rl_logs", f"{args.experiment_name}_{experiment_cnt+1}")
        logging.warn(f"{args.experiment_name} already exists, new logging directory: {log_dir}")

    tensorboard_log_dir = os.path.join(
        log_dir, "tensorboard"
    )

    checkpoint_save_dir = os.path.join(
        log_dir, "checkpoints"
    )

    if args.num_envs > 1:
        env = gym.vector.make("BerretHandGymRayLib-v0", num_envs=args.num_envs, env_config=ENV_CONFIG)
    else:
        env = gym.make("BerretHandGymRayLib-v0", env_config=ENV_CONFIG)


    # Initialize the PPO model with the custom policy and tensorboard logging
    model = PPO(
        policy=ControlDropPolicy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=args.normalize_advantage,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        target_kl=args.target_kl,
        stats_window_size=args.stats_window_size,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs={"model_config": model_config},
        verbose=args.verbose,

    )

    if args.rl_chkpoint is not None:
        model.load(args.rl_chkpoint)

    if args.unlock_encoder:
        model.policy.features.unlock_parameters()

    callback = SaveOnBestRewardCallback(
        check_freq=args.save_freq, 
        log_dir=checkpoint_save_dir, 
        verbose=1,
        env=env
        )

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps, 
        callback=callback,
        tb_log_name=args.experiment_name)

    # Save the model
    model.save(os.path.join(CONTROL_DROP_DIR, f"{args.experiment_name}_model"))

if __name__ == "__main__":
    args = parse_args()
    main(args)