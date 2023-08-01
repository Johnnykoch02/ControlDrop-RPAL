from typing import Union, Callable, Tuple
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch as th

th.autograd.set_detect_anomaly(True)
th.manual_seed(42)

# from stable_baselines3.common import env_checker
# from src.RL.callbacks import TrainAndLoggingCallback 
from src.RL.custom_gym import BerrettHandGym, VectorizedBerrettHandGym, SceneDifficulty, NAME, CHECKPOINT_DIR, LOG_DIR, SAVE_FREQ
from src.RL.Networks.ExtractorNetworks import MMExtractor, TestExtractorVelocity, BaseTransformerFeatureEncoder, MultiModalTransformerFeatureEncoder
from stable_baselines3 import PPO
from gym.vector import VectorEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

# https://npitsillos.github.io/blog/2021/recurrent-ppo/ This Talks about Recurrent PPO

# PPO.load(os.path.join(CHECKPOINT_DIR, 'Exploration_Model.zip'))

import threading as t
import time
import numpy as np
import subprocess
from src.Utils.NetworkUtils import find_open_port


factor = 0.99999
stop_factor_lr = 5e-5
lr = 0.001

def lr_call(step):
    global lr, stop_factor_lr, factor
    lr = max(stop_factor_lr, lr * factor)
    return lr

# TODO: LR Scheduler

def create_sim(sim_port, **kwargs):
    print('Creating SIM from ')
    CMD_ARGS = [
        f"./start_sim_on_port.sh", f"{sim_port}"
    ]   #                                       Supress the STDOUT from the sims
    return subprocess.Popen( args = CMD_ARGS, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)



def extract_command(command:str):
    cmd = command.split(' ')[0]
    arguments = command.replace(cmd,'').split(' -')
    args = {}
    for i in arguments:
        try:
            tmp = i.replace(' ', '#',1).split('#')
            args[tmp[0]] = tmp[1].strip()
        except:
            pass
    return cmd, args

def Demo(model_loc):
    model = PPO.load(os.path.join(CHECKPOINT_DIR, 'Exploration_Model.zip'))
    env = BerrettHandGym()
    env.configs = np.load(os.path.join(os.getcwd(), "Data_Collection", "Difficultys"," easy.npy"), allow_pickle= True)
    model.set_env(env)
    obs = env.reset()
    i = 0
    reward = []
    Dones = []
    done = False
    while i < 10000:
        if done:
            Dones.append(done)
            env.reset()
        i+=1
        action, _states = model.predict(obs.copy())
        obs, rewards, done, info = env.step(action)
        reward.append(rewards)
        if i % 60 == 0:
            print("Percent Success: {:.2f}%".format((sum(Dones)/ len(Dones)) * 100))
            print('Mean Rew. for Past 60 timesteps:', np.mean(reward[-60::]),'Demo Info:', info )

def save_data(data, path):
    hand_config = []
    hand_torque = []
    finger1 = []
    finger2 = []
    finger3 = []
    palm = []
    obj_count = []
    current_obj_locations = []
    obj_velocity = []
    rot_velocity = []
    progress = []
    for idx in data:
        hand_config.append(idx['hand_config'])
        hand_torque.append(idx['hand_torque'])
        finger1.append(idx['finger_1_tactile'])
        finger2.append(idx['finger_2_tactile'])
        finger3.append(idx['finger_3_tactile'])
        palm.append(idx['palm_tactile'])
        obj_count.append(idx['ball_count'])
        current_obj_locations.append(idx['ball_location'])
        obj_velocity.append(idx['obj_velocity'])
        rot_velocity.append(idx['obj_angular_velocity'])
        progress.append(idx['progress_bar'])

    np.savez(os.path.join(path, 'states.npz'), hand_config=np.array(hand_config), hand_torque=np.array(hand_torque),
     finger1=np.array(finger1), finger2=np.array(finger2), finger3=np.array(finger3), palm=np.array(palm), 
     obj_count=np.array(obj_count), current_obj_locations=np.array(current_obj_locations),
     obj_velocity=np.array(obj_velocity),rot_velocity=np.array(rot_velocity), progress=np.array(progress))

def load_data(path):
    npzfile = np.load(os.path.join(path, 'states.npz'))
    hand_config = npzfile['hand_config']
    hand_torque = npzfile['hand_torque']
    finger1 = npzfile['finger1']
    finger2 = npzfile['finger2']
    finger3 = npzfile['finger3']
    palm = npzfile['palm']
    obj_count = npzfile['obj_count']
    current_obj_locations = npzfile['current_obj_locations']
    obj_velocity = npzfile['obj_velocity']
    rot_velocity = npzfile['rot_velocity']
    progress = npzfile['progress']
    return hand_config, hand_torque, finger1, finger2, finger3, palm, obj_count, current_obj_locations, obj_velocity, rot_velocity, progress

def DataCollect(num_steps):
    env = BerrettHandGym()
    done = False
    data = []
    data.append(env.reset())
    i = 0
    while i < num_steps:
        if done:
            data.append(env.reset())
        i+=1
        state, reward, done, _ = env.step(env.action_space.sample())
        data.append(state)
        time.sleep(0.02)
    
    save_data(data, os.path.join(os.getcwd(), 'Data_Collection', 'Enviornment_Samples'))

def create_vector_env(num_envs=4, env_name="BerretHandDropping-v0", entry_point= BerrettHandGym):
    from gym.envs.registration import register
    from gym.vector import AsyncVectorEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    register(
        id=env_name,
        entry_point=entry_point
    )
    
    ports = [ find_open_port() for _ in range(num_envs)]
    sim_processes = []
    for port in ports:
        sim_process = create_sim(port)
        time.sleep(25)
        sim_processes.append(sim_process)
    
    if entry_point==BerrettHandGym:
        envs = VectorizedBerrettHandGym(
                    sim_ports=ports,
                    object_type='Sphere',
                    object_quantity=7,
                    detailed_training=True,
                    difficulties=[
                        SceneDifficulty.EASY,
                        SceneDifficulty.MEDIUM,
                        SceneDifficulty.HARD
                    ],
                    algorithm="RecurrentPPO",
                    is_val=False
                )
        
        return envs, sim_processes
        
    return None
    
def get_model(model_key='Stable',detailed_training=False, num_envs=3,):
    '''As We implement more Architectures, we want to keep Track of training histories. This section is dedicated for that.'''
    vec_env, sim_ps = create_vector_env(num_envs=num_envs,)
    if model_key == "Stable_LSTM_Adam_LinearEncoder": # LSTM Implementation with Adam Optimizer 
        policy_kwargs = dict(
        features_extractor_class=TestExtractorVelocity,# pi: action, vf: value
        net_arch=[512, dict(pi=[256, 256], qf=[256, 256])],
        optimizer_class=th.optim.Adam,
        lstm_hidden_size=128,
        optimizer_kwargs=dict(
            eps=1e-7,
            weight_decay=0.001,
            ),  
        )
        return RecurrentPPO('MultiInputLstmPolicy',
            env=vec_env.envs[0],
            policy_kwargs=policy_kwargs, 
            verbose = 1,
            learning_rate = 7e-4,
            gae_lambda=1,
            normalize_advantage=False,
            n_epochs=4,
            clip_range_vf=None,
            n_steps = 128*4,
            batch_size = 128,
            ent_coef = 0.001,
            tensorboard_log=LOG_DIR,
        ), sim_ps, vec_env
        
    elif model_key == "Stable_Adam_LinearEncoder":
        policy_kwargs = dict(
        features_extractor_class=TestExtractorVelocity,# pi: action, vf: value
        net_arch=[512, dict(pi=[256, 256], vf=[256, 256])],
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=dict(
            eps=1e-7,
            weight_decay=0.001,
            ),  
        )
        return PPO('MultiInputPolicy',
            env=vec_env.envs[0],
            policy_kwargs=policy_kwargs, 
            verbose = 1,
            learning_rate = 7e-4,
            gae_lambda=1,
            normalize_advantage=False,
            n_epochs=4,
            clip_range_vf=None,
            n_steps = 128*4,
            batch_size = 128,
            ent_coef = 0.001,
            tensorboard_log=LOG_DIR,
        ), sim_ps, vec_env
     
    elif model_key == "something_else":
        policy_kwargs = None
        return None
    
    return None


'''train -num_steps <int> -load_dir <path>'''
def Train(num_steps:int=100000,detailed_training=False, num_envs=3):
    def custom_train(model:Union[PPO, RecurrentPPO], vec_env:VectorEnv, total_timesteps, episodes_per_checkpoint, log_interval):
        timesteps = 0
        episode = 0

        while timesteps < total_timesteps:
            obs = vec_env.reset()
            ep_timesteps = 0
            ep_rewards = np.zeros(vec_env.num_envs)
            done = np.array([False] * vec_env.num_envs)
            # model.policy.set_train_mode(True)

            while not np.all(done):
                action, _ = model.predict(obs)
                new_obs, reward, done, info = vec_env.step(action)
                model.learn(total_timesteps=vec_env.num_envs, reset_num_timesteps=False)

                ep_rewards += reward
                ep_timesteps += 1
                obs = new_obs
                timesteps += vec_env.num_envs

            episode += 1
            if episode % log_interval == 0:
                print(f"Episode: {episode}, Timesteps: {timesteps}, Avg. Episode Reward: {ep_rewards.mean()}")

            if (episode+1) % episodes_per_checkpoint == 0:
                len_of_dtsve = len(os.listdir(os.path.join(os.getcwd(), 'Data_Collection', 'Model_Performance',)))
                performance_save_path = os.path.join(os.getcwd(), 'Data_Collection', 'Model_Performance', f'Trial_{len_of_dtsve-1}')
            
                BerrettHandGym.validate_agent('RecurrentPPO', CHECKPOINT_DIR, use_sbl=True, plot_data=True)
    
    # model, sim_ps = get_model(model_key='Stable_LSTM_Adam_LinearEncoder', detailed_training=detailed_training, num_envs=num_envs)
    # custom_train(model, envs, num_steps, 10050, 228)
    model, sim_ps, envs = get_model(model_key='Stable_Adam_LinearEncoder', detailed_training=detailed_training, num_envs=num_envs)
    custom_train(model, envs, num_steps, 10050, 228)
    
    # model.learn(num_steps, )
    
    # print(f"Final Accuracy: {(sum(model.env.successes[-250::])/1000):.2%}")

def temp_train():
    policy_kwargs= dict(
        features_extractor_class=MultiModalTransformerFeatureEncoder,# pi: action, vf: value
        net_arch=[512, dict(pi=[256, 64], vf=[256, 64])],
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=0.0001,
            ),  
        )
    
    # model = PPO('MultiInputPolicy',
    #         env=BerrettHandGym(
    #                 object_type='Sphere',
    #                 object_quantity=7,
    #                 detailed_training=True,
    #                 difficulties=[
    #                     SceneDifficulty.EASY,
    #                     SceneDifficulty.MEDIUM,
    #                 #    SceneDifficulty.HARD
    #                 ],
    #                 algorithm="PPO",
    #                 is_val=False
    #             ),
    #         policy_kwargs=policy_kwargs, 
    #         verbose = 1,
    #         learning_rate = 7e-3,
    #         gae_lambda=0.98,
    #         gamma=0.93,
    #         normalize_advantage=False,
    #         n_epochs=4,
    #         clip_range_vf=None,
    #         n_steps = 128*4,
    #         batch_size = 128,
    #         ent_coef = 0.001,
    #         tensorboard_log=LOG_DIR,
    #     )
    model = PPO.load("/media/rpal/SSD512/John/Control Drop/Reinforcement Learning/src/RL/Training/Checkpoints/TransformerFeatureEncoder/baseline_rl_2500_steps.zip")
    env= BerrettHandGym(
                    object_type='Sphere',
                    object_quantity=7,
                    detailed_training=True,
                    difficulties=[
                        SceneDifficulty.EASY,
                        SceneDifficulty.MEDIUM,
                    #    SceneDifficulty.HARD
                    ],
                    algorithm="PPO",
                    is_val=False
                )
    model.set_env(env)
    
    model = model.learn(1000000,callback=CheckpointCallback(SAVE_FREQ, CHECKPOINT_DIR, 'baseline_rl'))
    model.save('./FinalResultTransformer.zip')
        
def main():
    print('CALLING MAIN')
    global commands
    
    
    temp_train()#(num_steps=1000000, detailed_training=True, num_envs=1)


if __name__ == '__main__':
    main()
    
