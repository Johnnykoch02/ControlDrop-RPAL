import os
import argparse
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from torch.cuda.amp import autocast
from control_dropping_rpal.RL.control_dropping_env import (
    BerretHandGymRayLibWrapper,
    SceneDifficulty,
)
from control_dropping_rpal.Utils.data_utils import plot_results
from control_dropping_rpal.RL.Networks.ActorCriticNetwork import ControlDropPolicy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
th.set_default_device(device)


CONTROL_DROP_DIR = os.environ["CONTROL_DROP_DIR"]

def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for RL Control Dropping")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file to load",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=150,
        help="Number of episodes to run for validation",
    )
    
    args =  parser.parse_args()
    setattr(args, "output_dir", os.path.join(args.experiment_path, "validation_results"))
    return args

def validate(policy, env, num_episodes, device):
    rewards = []
    dones = []
    drop_stats = {(difficulty, count): {} for count in range(6) for difficulty in SceneDifficulty}
    difficulty_stats = {}

    for episode in tqdm(range(num_episodes), desc="Validating"):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        save_flag = False
        policy.to(device)

        while not done:
            # Convert observation to tensor
            obs_tensor = obs_as_tensor(obs, device)
            # Unsqueeze the observation tensor or tensor dict
            if isinstance(obs_tensor, dict):
                obs_tensor = {k: v.unsqueeze(0).to(device).half() if isinstance(v, th.Tensor) else v for k, v in obs_tensor.items()}
            elif isinstance(obs_tensor, th.Tensor):
                obs_tensor = obs_tensor.unsqueeze(0).to(device).half()
            else:
                raise ValueError(f"Unexpected obs_tensor type: {type(obs_tensor)}")

            with th.cuda.amp.autocast():
                with th.no_grad():
                    actions, _, _ = policy(obs_tensor, deterministic=False)

            # Convert actions back to numpy if necessary
            action = actions.cpu().numpy().squeeze()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            post_drop = env.post_drop - 1

            if env.current_obj_count == env.goal_obj_count:
                key = (env.current_index_difficulty, env.goal_obj_count)
                drop_stats[key][post_drop] = drop_stats[key].get(post_drop, 0) + 1

            if not save_flag and env.current_obj_count == env.goal_obj_count and post_drop > 8:
                save_flag = True
                key = (env.current_index_difficulty, env.goal_obj_count)
                difficulty_stats[key] = difficulty_stats.get(key, []) + [1]

        if not save_flag:
            key = (env.current_index_difficulty, env.goal_obj_count)
            difficulty_stats[key] = difficulty_stats.get(key, []) + [0]

        rewards.append(episode_reward)
        dones.append(done)

    return rewards, dones, drop_stats, difficulty_stats

def plot_post_drop_by_difficulty(drop_stats, difficulty_stats, num_episodes, output_dir):
    difficulties = [SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD]
    base_colors = ['#00FF00', '#FFA500', '#FF0000']  # Brighter base colors
    labels = ['Easy', 'Medium', 'Hard']

    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')  # Use dark background style

    for difficulty, base_color, label in zip(difficulties, base_colors, labels):
        colormap = cm.get_cmap('Greens' if base_color == '#00FF00' else 'Oranges' if base_color == '#FFA500' else 'Reds')
        
        for obj_count in range(1, 8):
            success_rates = []
            confidence_intervals = []

            key = (difficulty, obj_count)
            if key in drop_stats:
                total_attempts = drop_stats[key].get(0, 0)  # Total attempts from step 0

                for steps in range(24):
                    successes_at_steps = drop_stats[key].get(steps, 0)
                    
                    if total_attempts > 0:
                        success_rate = successes_at_steps / total_attempts
                    else:
                        success_rate = 0

                    success_rates.append(success_rate)

                # Calculate confidence intervals
                ci = 1.96 * np.std(success_rates) / np.sqrt(len(success_rates))  # 95% confidence interval
                confidence_intervals = [ci] * len(success_rates)

                x_range = range(24)
                shade = colormap((obj_count - 1) / 7 + 0.2)  # Adjust color intensity
                plt.plot(x_range, success_rates, color=shade, label=f'{label} Difficulty, Obj Count {obj_count}', linewidth=2)
                plt.fill_between(x_range, 
                                 np.array(success_rates) - np.array(confidence_intervals),
                                 np.array(success_rates) + np.array(confidence_intervals),
                                 color=shade, alpha=0.3)

    plt.xlabel('Steps Post Drop', color='white', fontsize=12)
    plt.ylabel('Probability of Success', color='white', fontsize=12)
    plt.title('Probability of Success by Difficulty and Steps Post Drop', color='white', fontsize=14)
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.ylim(0, 1)

    # Customize tick labels
    plt.tick_params(colors='white')
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Post_Drop_By_Difficulty.png")
    plt.savefig(output_path, facecolor='#333333', edgecolor='none')
    plt.close()

def main(args):
    # Load the model
    model = PPO.load(args.checkpoint, policy=ControlDropPolicy)
    policy = model.policy
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    # policy.device = device
    policy.to(device)
    # Create the environment
    env_config = dict(
        detailed_training=True,
        object_quantity=7,
        difficulties=[SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD],
        is_val=True,
    )
    env = BerretHandGymRayLibWrapper(config=env_config)

    # Run validation
    rewards, dones, drop_stats, difficulty_stats = validate(policy, env, args.num_episodes, device)

    # Process and save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot post-drop statistics by difficulty
    plot_post_drop_by_difficulty(drop_stats, difficulty_stats, args.num_episodes, args.output_dir)

    # Calculate total episodes and dataset percentages
    total_episodes = sum(len(v) for v in difficulty_stats.values())
    
    # Save difficulty statistics with additional information
    file_data = f"{'='*50}\n"
    file_data += f"Checkpoint: {args.checkpoint}\n"
    file_data += f"Total Episodes: {total_episodes}\n"
    file_data += f"{'='*50}\n\n"

    file_data += "Difficulty Statistics:\n"
    file_data += f"{'Difficulty':<10} {'Balls':<6} {'Success Rate':<15} {'Episodes':<10} {'Dataset %':<10}\n"
    file_data += f"{'-'*50}\n"

    for difficulty in [SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD]:
        for obj_count in range(1, 8):
            key = (difficulty, obj_count)
            if key in difficulty_stats:
                episodes = len(difficulty_stats[key])
                success_rate = np.mean(difficulty_stats[key])
                dataset_percentage = (episodes / total_episodes) * 100
                file_data += f"{difficulty.name} {obj_count} {success_rate:.2f} {episodes} {dataset_percentage:.2f}%\n"
        file_data += f"{'-'*50}\n"

    file_data += f"\n{'='*50}\n"
    file_data += "Overall Statistics:\n"
    file_data += f"Average Reward: {np.mean(rewards):.2f}\n"
    file_data += f"Overall Success Rate: {np.mean(dones):.2f}\n"

    for difficulty in [SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD]:
        success_rate = np.mean([np.mean(difficulty_stats.get((difficulty, obj_count), [0])) 
                                for obj_count in range(1, 8)])
        episodes = sum(len(difficulty_stats.get((difficulty, obj_count), [])) for obj_count in range(1, 8))
        dataset_percentage = (episodes / total_episodes) * 100
        file_data += f"{difficulty.name} Difficulty:\n"
        file_data += f"  Success Rate: {success_rate:.2f}\n"
        file_data += f"  Episodes: {episodes}\n"
        file_data += f"  Dataset %: {dataset_percentage:.2f}%\n"

    with open(os.path.join(args.output_dir, "Val_Difficulty_Stats.txt"), "w") as f:
        f.write(file_data)

    # Print summary
    print(f"Validation complete. Results saved in {args.output_dir}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Overall success rate: {np.mean(dones):.2f}")

    # Print success rates by difficulty
    for difficulty in [SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD]:
        success_rate = np.mean([np.mean(difficulty_stats.get((difficulty, obj_count), [0])) 
                                for obj_count in range(1, 8)])
        print(f"{difficulty.name} difficulty success rate: {success_rate:.2f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)