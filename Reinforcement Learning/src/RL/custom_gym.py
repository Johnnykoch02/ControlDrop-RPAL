from typing import List, Union, Callable, Dict
from operator import truediv
from gym import Env
from gym.spaces import MultiDiscrete, Box, Dict
import re
from math import inf, radians, degrees
import numpy as np
import os
import time
import gzip
import re
from collections import deque
from enum import Enum
import subprocess
import multiprocessing
import pygame

from ..SimController import SimController
from ..Utils.DataUtils import plot_results
from src.Utils.NetworkUtils import find_open_port

NAME = 'TransformerFeatureEncoder'

CHECKPOINT_DIR = os.path.join(os.getcwd(), 'Reinforcement Learning','src','RL','Training', 'Checkpoints', NAME)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
LOG_DIR = os.path.join(os.getcwd(), 'Training', 'Logs', NAME)
'''Creating Environments Automatically; Returns Process and Port'''


def create_sim(sim_port, **kwargs):
    print('Creating SIM from ')
    CMD_ARGS = [
        f"./start_sim_on_port.sh", f"{sim_port}"
    ]   #                                       Supress the STDOUT from the sims
    return subprocess.Popen( args = CMD_ARGS, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


SAVE_FREQ = 2500

class SceneDifficulty(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

class BerrettHandGym(Env):
    '''This class is designed to implement control of the Berret Hand through use of SimController'''
    def __init__(
            self,
            test = False,  # If trying to iterate through grasping data, set to True
            cluster_index = 3, # Used for setting the data cluster for use in training
            sim_port= None, # Port to connect the sim on, or None if auto-create environment.
            object_type='Sphere', # This is for the Sim to generate the Specific type of Object for Grasp
            object_quantity=7, # Number of Objects being generated
            detailed_training=False, # Controls the type of data and data detailing being performed during Training
            detailed_save_dir=None, # Overrides the Path of Trial Saving
            plot_params= ['history', 'avg_ke', 'avg_vel', 'avg_rew'], # Specify which parameters to include in the Figures
            difficulties=[SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD], # Difficulties to Train on
            algorithm="RecurrentPPO", # Algorithm for loading in validation
            is_val=False # is Running Validation
            ):
        
        super().__init__()
        if sim_port == None:
            self.sim_port = find_open_port()
            self.sim_process = create_sim(self.sim_port)
            time.sleep(25)
        else:
            self.sim_port = sim_port
        self.simController = SimController(port=self.sim_port, object_type=object_type, object_quantity=object_quantity)
        self.obj_quantity = object_quantity
        self.test = test
        self.counter = {'index': 0, 'total_steps': 0, 'episodes': 0}
        self.configs_index = 0
        self.reset_counter = 0
        self.successes = []
        self.algorithm = algorithm
        self.val_process = None
        self.is_val = is_val
        self.num_vals = 0
        self.clock = pygame.time.Clock()
        self.target_fps = 3
        
        # Flag to indicate the necesity of checking sim gravity
        self.restart_check = False
        
        self.detailed_training = detailed_training
        if self.detailed_training:
            len_of_dtsve = len(os.listdir(os.path.join(os.getcwd(), 'Data_Collection', 'Model_Performance',)))
            self.performance_save_path = os.path.join(os.getcwd(), 'Data_Collection', 'Model_Performance', f'Trial_{len_of_dtsve}') if detailed_save_dir is None else detailed_save_dir
            if not os.path.exists(self.performance_save_path):
                os.makedirs(self.performance_save_path)
            self.num_detailed_files = 0
            self.model_performance = { # Model Adds to performance, remembers history of previous saves and the performance at those time steps.
                SceneDifficulty.EASY: {'performance': [], 'history': [], 'avg_ke': [], 'avg_vel': [], 'avg_rew': [],'use_data': False},
                SceneDifficulty.MEDIUM:  {'performance': [], 'history': [],'avg_ke': [], 'avg_vel': [], 'avg_rew': [],'use_data': False},
                SceneDifficulty.HARD: {'performance': [], 'history': [], 'avg_ke': [], 'avg_vel': [], 'avg_rew': [],'use_data': False},
            }
            self.current_run_vel = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.current_run_ke = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.validation_runs = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.episodic_reward = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.current_episodic_rew = 0
            self.current_index_difficulty = None
            self.plot_params = plot_params
            
        self.configs = self.get_cluster(cluster_index) if not detailed_training else self.get_detailed_training_data(desired_difficulties=difficulties)
        self.detailed_save_frequency = 100
        
    
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
        
        spaces = {
            'palm_tactile': Box(low= -inf, high= inf, shape=(24, )), # Value
            'finger_1_tactile': Box(low= -inf, high= inf, shape=(24,)), # Value
            'finger_2_tactile': Box(low= -inf, high= inf, shape=(24,)), # Value
            'finger_3_tactile': Box(low= -inf, high= inf, shape=(24,)), # Value
            # 'tactile_pos': Box(low= -inf, high= inf, shape=(378, )), # Position
            'finger_1_location': Box(low=-inf, high=inf, shape=(2 + 34*3,)), # Joint pos [Theta_1, Theta_2] + [xyz*34]
            'finger_2_location': Box(low=-inf, high=inf, shape=(2 + 34*3,)), 
            'finger_3_location': Box(low=-inf, high=inf, shape=(2 + 34*3,)),
            'palm_location': Box(low= -inf, high= inf, shape=(2 + 24*3,)),
            'obj_location': Box(low= -inf, high= inf, shape=(object_quantity, 6)), # Position
            'obj_velocity': Box(low= -inf, high= inf, shape=(object_quantity, 6)), # Value, Concat with angular velocity
            'state_attrib': Box(low=-inf, high=inf, shape=(45, )) # Ball Cnt, Progress, Prev.Actions, hand_cfg, hand_trq (44)
        }
        """
        NOTES:
        - Previous Actions 1 hot encoded into a vector, in a matrix of 5 Previous Actions {OUTDATED}
        """
        
        self.observation_space = Dict(spaces)
        #TODO Change this !!
        # self.action_space = MultiDiscrete([3, 3, 3])
        # self.action_space = MultiDiscrete([3, 3, 3, 3, 3, 3])
        self.action_space = Box(low=-7, high=7, shape=(5,), dtype=np.float32) # Continuous Action Space

        self.previous_actions = deque()
        for _ in range(8):
            self.previous_actions.append(np.zeros(shape=(5,)).astype(float))
       
        self.current_hand_config = self.simController.get_hand_config()
        self.current_obj_count = None
        self.goal_obj_count = None
        self.step_len = 0
        self.post_drop = 0
        self.bad_scenes = set()
        
        #Reward Function Parameters
        self.termination_penalty = 150
        self.long_drop_penalty = 65
        self.success_reward = 175
        self.stabilizing_reward = 10
        self.failure_penalty = 120
        self.movement_penalty_scale_factor = 0.3
        self.palm_angle_penalty_scale_factor = 0.1

        
    def reset_data_cntrs(self):
        try:
            self.current_run_vel = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.current_run_ke = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.episodic_reward = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            self.validation_runs = {
                SceneDifficulty.EASY: [],
                SceneDifficulty.MEDIUM:[],
                SceneDifficulty.HARD: [],
            }
            
        except Exception as e:
            print("Error While resetting Data Cntrs...", e)
    
    def read_files(self):
        file_name = os.path.join(os.getcwd(),'Reinforcement Learning','src','RL','Training','Data','40_mm_sphere_control_drop_scenes.npy.gz') if self.test else os.path.join(os.getcwd(),'Reinforcement Learning','src','RL', 'Training','Data','40_mm_sphere_train_control_drop_scenes.npy.gz')
        with gzip.GzipFile(file_name) as f:
            data = np.load(f, allow_pickle= True)
        train_path = os.path.join(CHECKPOINT_DIR)
        names = os.listdir(train_path)
        names = [re.sub('[a-z_.]', '', name) for name in names]
        try:
            self.counter['total_steps'] = int(names[-1])
        except:
            pass
        
        return data
    
    def get_detailed_training_data(self, path=None, desired_difficulties=[SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD]):
        data = []
        path = os.path.join(os.path.join(os.getcwd(), "Data_Collection", "Difficultys")) if path == None else path
        data_srlzation = { 'easy.npy': SceneDifficulty.EASY,  'medium.npy': SceneDifficulty.MEDIUM, 'hard.npy': SceneDifficulty.HARD}
        for file in os.listdir(path):
            if data_srlzation[file] in desired_difficulties:
                self.model_performance[data_srlzation[file]]['use_data'] = True
                d = np.load(os.path.join(os.getcwd(), "Data_Collection", "Difficultys", file), allow_pickle= True)
                for scene in d: #npy file, Difficulty Enum
                    data.append({'scene': scene, 'difficulty': data_srlzation[file]})
        return data
            
    def get_cluster(self, cluster_id):
        return np.load(os.path.join(os.getcwd(), "Data_Collection", "Clusters", f"{cluster_id}.npy"), allow_pickle= True)
    

    def step(self, action):
        '''Defines how a step occurs in an enviornment'''
        d_t = self.clock.tick(self.target_fps) / 1000.0
        print('Current Scene:', self.configs_index, 'Step Len:', self.step_len, 'Current Difficulty:', self.current_index_difficulty)
        self.step_len+=1
        # if self.counter['total_steps'] != 0:
        #     self.remove_extra_files()
        ''' Action Space [-1:1] '''
        action = np.round(action)
        main_joints = action[1:4]
        step = np.concatenate(([action[0]], main_joints, np.array(main_joints) * (1/3)))
        print('Step:', step)
        
        self.simController.step_hand_config(step)
        # self.simController.step_arm_config(np.array([0, 0, 0, 0, action[4], 0]))

        self.current_hand_config = self.simController.get_hand_config()
            
        self.current_obj_count = self.simController.get_object_count()        
        
        reward, done = self.get_current_reward()
        
        # Keep track of action encoding
        self.previous_actions.append(np.array(action) / 7)
        self.previous_actions.popleft()
        action_state = np.array(self.previous_actions).copy().reshape((40,))
        state = self.simController.get_state_encoding(action_state)
        
        self.current_episodic_rew+=reward
        return state, reward, done, {}
    
    def one_hot_encode(self, action):#[0-14, 0-14, 0-14]
        encoded_action = np.zeros(shape=(45,))
        for i in range(len(action)):
            offset = (i*15)
            action_encode_index = action[i] + offset
            encoded_action[action_encode_index] = 1
        return encoded_action
                
    def plot_detailed_training_data(self):
        plotters = []
        params = self.plot_params
        _range = None
        labels = []
        _path= os.path.join(self.performance_save_path,'Detailed_Training') if not self.is_val else os.path.join(self.performance_save_path,'Detailed_Validation')
        if not os.path.exists(_path):
            os.makedirs(_path)
        _path = os.path.join(_path,  f'{self.num_detailed_files}.png')
        ''' Update the Current Values for each of our potential Data Plots'''
        if self.model_performance[SceneDifficulty.EASY]['use_data']:
            try:
                easy_data = self.model_performance[SceneDifficulty.EASY]['performance'][-25::]
                self.model_performance[SceneDifficulty.EASY]['history'].append(sum(easy_data)/len(easy_data))
                self.model_performance[SceneDifficulty.EASY]['avg_vel'].append(sum(self.current_run_vel[SceneDifficulty.EASY])/len(self.current_run_vel[SceneDifficulty.EASY]))
                self.model_performance[SceneDifficulty.EASY]['avg_ke'].append(sum(self.current_run_ke[SceneDifficulty.EASY])/len(self.current_run_vel[SceneDifficulty.EASY]))
                self.model_performance[SceneDifficulty.EASY]['avg_rew'].append(sum(self.episodic_reward[SceneDifficulty.EASY])/len(self.current_run_vel[SceneDifficulty.EASY]))
                plotters.append(self.model_performance[SceneDifficulty.EASY])
                labels.append('Easy')
                _range = range(len(self.model_performance[SceneDifficulty.EASY]['history']))
            except Exception as e:
                print("Error with Saving Easy Data...", e)
        if self.model_performance[SceneDifficulty.MEDIUM]['use_data']:
            try:
                med_data = self.model_performance[SceneDifficulty.MEDIUM]['performance'][-25::]
                self.model_performance[SceneDifficulty.MEDIUM]['history'].append(sum(med_data)/len(med_data))
                self.model_performance[SceneDifficulty.MEDIUM]['avg_vel'].append(sum(self.current_run_vel[SceneDifficulty.MEDIUM])/len(self.current_run_vel[SceneDifficulty.MEDIUM]))
                self.model_performance[SceneDifficulty.MEDIUM]['avg_ke'].append(sum(self.current_run_ke[SceneDifficulty.MEDIUM])/len(self.current_run_vel[SceneDifficulty.MEDIUM]))
                self.model_performance[SceneDifficulty.MEDIUM]['avg_rew'].append(sum(self.episodic_reward[SceneDifficulty.MEDIUM])/len(self.current_run_vel[SceneDifficulty.MEDIUM]))
                plotters.append(self.model_performance[SceneDifficulty.MEDIUM])
                labels.append('Medium')
                _range = range(len(self.model_performance[SceneDifficulty.MEDIUM]['history']))
            except Exception as e:
                print("Error with Saving Medium Data...", e)
        if self.model_performance[SceneDifficulty.HARD]['use_data']:
            try:
                hard_data = self.model_performance[SceneDifficulty.HARD]['performance'][-25::]
                self.model_performance[SceneDifficulty.HARD]['history'].append(sum(hard_data)/len(hard_data))
                self.model_performance[SceneDifficulty.HARD]['avg_vel'].append(sum(self.current_run_vel[SceneDifficulty.HARD])/len(self.current_run_vel[SceneDifficulty.HARD]))
                self.model_performance[SceneDifficulty.HARD]['avg_ke'].append(sum(self.current_run_ke[SceneDifficulty.HARD])/len(self.current_run_vel[SceneDifficulty.HARD]))
                self.model_performance[SceneDifficulty.HARD]['avg_rew'].append(sum(self.episodic_reward[SceneDifficulty.HARD])/len(self.current_run_vel[SceneDifficulty.MEDIUM]))
                plotters.append(self.model_performance[SceneDifficulty.HARD])
                labels.append('Hard')
                _range = range(len(self.model_performance[SceneDifficulty.HARD]['history']))
            except Exception as e:
                print("Error with Saving Hard Data...", e)     
        if _range != None:
            plot_results(plotters,params=params, y_labels=labels, range_spc=_range, file_name=_path)
            print(f'\n{"-"*10}\n{"-"*10}\n{"-"*10}\nSaving Performance Data\n{"-"*10}\n{"-"*10}\n{"-"*10}')
        self.num_detailed_files+=1
        self.reset_data_cntrs()
        
    def reset(self):
        self.step_len = 0
        self.post_drop = 0 
        self.simController.reset_arm_config()
        if self.detailed_training:
            if self.current_episodic_rew != 0:
                self.episodic_reward[self.current_index_difficulty].append(self.current_episodic_rew)
            self.current_episodic_rew = 0
        
        valid = False
        self.counter['episodes'] += 1
        
        self.simController.progress_bar = 0
        self.simController.drop_detected = False
        
        
        self.reset_counter+=1
        if (self.reset_counter) % 10 == 0 or self.restart_check:
            try:
                print(f"Current Accuracy: {(sum(self.successes)/len(self.successes)):.2%}")
            except:
                pass
            self.restart_check = False
            self.simController.is_sim_working()
            
        self.current_obj_count = self.simController.get_object_count()
        if self.current_obj_count > 1 and not self.test:
            self.current_hand_config = self.simController.get_hand_config()
            print('Using the Same Scene...')
            valid = True  
        
        reset_atmpt_cntr = 0
        
        while not valid:
            if reset_atmpt_cntr+1 % 16 == 0:
                print('Reset Counter Casusing Reset on Sim...')
                self.simController.restart_sim()
            reset_atmpt_cntr+=1
        
            self.configs_index = np.random.randint(0, len(self.configs))
            ''' Avoid using known bad scenes '''

            print('Setting up the Scene...\n Index:', self.configs_index)
            new_scene = None
            if self.detailed_training:
                if self.counter['episodes'] % self.detailed_save_frequency == 0:
                    self.plot_detailed_training_data()
                new_scene = self.configs[self.configs_index]['scene']
                self.current_index_difficulty = self.configs[self.configs_index]['difficulty']
                    
            else:
                new_scene = self.configs[self.configs_index]
                
            hand_config, ball_locations = [degrees(i) for i in new_scene[-7::]], new_scene
            hand_config = [hand_config[i] for i in (0, 3, 2, 1, 6, 5, 4)]
            if self.simController.setup_scene(hand_config, ball_locations):
                cnt = self.simController.get_object_count()
                time.sleep(3)
                self.current_hand_config = self.simController.get_hand_config()
                self.current_obj_count = self.simController.get_object_count()
                if self.current_obj_count == cnt or  self.current_obj_count > 1:
                    valid = True
                    print("Scene Validated. Starting Dropping...")
            else:
                # self.bad_scenes.add(self.configs_index)  
                pass
            # self.configs_index+=1
            # if self.configs_index == len(self.configs):
            #     self.configs_index = 0
            
        self.goal_obj_count = self.current_obj_count - 1
        self.previous_actions.clear()
        for _ in range(8):
            self.previous_actions.append(np.zeros(shape=(5,)).astype(float))
        action_state = np.array(self.previous_actions).copy().reshape((40,))
        state = self.simController.get_state_encoding(action_state) 

        return state
    
    #TODO: 
    #   - implement penalty for all moving balls after drop.
    #   - implement penalty for n>1 moving balls before drop.
    #   - 
    #CHANGELOG:
    # - 7/28/2023: Added base penalty for step, added scale factor for palm displacement.
    def get_current_reward(self):
        terminations = self.simController.get_terminations()
        # terminations and palm stress
        palm_displacement = -max(self.simController.get_angle_from_normal()*self.palm_angle_penalty_scale_factor, 5)# self.simController.check_palm_orientation()[1]
                
        cum_velocity = self.simController.get_cumulative_velocity()
        hand_ke = self.simController.get_KE()
        
        penalty = -0.2 + sum(self.simController.get_limit_signals()) / 2 + sum(terminations) + (palm_displacement) + hand_ke * self.movement_penalty_scale_factor + cum_velocity
        self.current_run_vel[self.current_index_difficulty].append(cum_velocity)
        self.current_run_ke[self.current_index_difficulty].append(hand_ke)
        self.simController.reset_termination_signals()
        self.simController.reset_limit_signals()

        print(f"Cumulative Velocity: {cum_velocity}\nHand KE: {hand_ke}\n")
        reward = -penalty
        done = False
        
        if all(terminations) or palm_displacement > 0:
            print("Hand or Palm Penalty...")
            done = True
            reward -= self.termination_penalty
            
        elif self.step_len > 175 and self.post_drop < 1 or self.post_drop > 50:
            print("Taking too Long to Drop...")
            done = True
            self.successes.append(False)
            if self.detailed_training:
                self.model_performance[self.current_index_difficulty]['performance'].append(0)
            self.restart_check = True
            reward -= self.long_drop_penalty
        elif self.current_obj_count > self.goal_obj_count:
            done = False 
            reward += ((self.goal_obj_count - self.current_obj_count)*2.5)
            
        elif self.current_obj_count == self.goal_obj_count and self.post_drop < 35:
            print("Drop Detected...")
            self.post_drop+=1
            done = False
            reward += (self.post_drop / 2)
            
        elif self.current_obj_count == self.goal_obj_count and self.post_drop >= 35 and cum_velocity < 0.01:
            print("Succesful Drop!")
            done = True
            self.successes.append(True)
            if self.detailed_training:
                self.model_performance[self.current_index_difficulty]['performance'].append(1)
            reward += self.success_reward
        elif self.current_obj_count == self.goal_obj_count and self.post_drop < 50:
            print("Hand Stablazation...")
            done = False
            self.post_drop+=1
            reward += self.stabilizing_reward 
        else:
            print("Failed at Dropping Single Object")
            done = True
            self.successes.append(False) 
            if self.detailed_training:
                self.model_performance[self.current_index_difficulty]['performance'].append(0)
            reward -= self.failure_penalty
            
        # Normalized Reward

        
        return (reward)/150, done
    
    def render(self, mode='human'):
        pass
    '''
    Cmd for running Validation on a seperate process
    '''
    @staticmethod
    def validate_agent(algo, model_path, use_sbl, save_path=None, plot_data=False):
        model_files = [f for f in os.listdir(model_path) if '.zip' in f]
        for f in model_files:
            most_recent_model = max(model_files, key=lambda f: os.path.getctime(os.path.join(os.getcwd(), f)))
        _args = [
            'python', os.path.join(os.getcwd(), 'Reinforcement Learning', 'validation.py'), '--algo', algo, '--model_path', most_recent_model,
        ]
        if not use_sbl:
            _args.append('--use_sbl')
            _args.append(False)
        if save_path != None:
            _args.append('--data_save_path')
            _args.append(save_path)
        if plot_data:
            _args.append('--plot_val')
            _args.append(True)
        return subprocess.Popen(_args)
        
    #TODO: Done on finger termination
    #TODO: Stop Threshold gets -15
    #TODO: Graph NN
     

# Vectorized Environment
from gym.vector import VectorEnv
class VectorizedBerrettHandGym(VectorEnv):
    '''This class is designed to implement control of the Berret Hand through use of SimController in a vectorized manner'''

    def __init__(self, 
            sim_ports: List[int], # Ports to connect the sim on, or None if auto-create environment.  
            test = False,  # If trying to iterate through grasping data, set to True
            cluster_index = 3, # Used for setting the data cluster for use in training
            object_type='Sphere', # This is for the Sim to generate the Specific type of Object for Grasp
            object_quantity=7, # Number of Objects being generated
            detailed_training=False, # Controls the type of data and data detailing being performed during Training
            detailed_save_dir=None, # Overrides the Path of Trial Saving
            plot_params= ['history', 'avg_ke', 'avg_vel', 'avg_rew'], # Specify which parameters to include in the Figures
            difficulties=[SceneDifficulty.EASY, SceneDifficulty.MEDIUM, SceneDifficulty.HARD], # Difficulties to Train on
            algorithm="RecurrentPPO", # Algorithm for loading in validation
            is_val=False # is Running Validation
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
        temp_env = BerrettHandGym(sim_port=sim_ports[0],
                                  
                                  )
        print('Created Env')
        super().__init__(num_envs=len(sim_ports), observation_space=temp_env.observation_space, action_space=temp_env.action_space)

        # Now initialize the rest of the environments
        self.envs = [temp_env] + [BerrettHandGym(sim_port=port,test = test, cluster_index=cluster_index,
                                                 object_type=object_type, object_quantity=object_quantity, 
                                                 detailed_training=detailed_training, detailed_save_dir=detailed_save_dir,
                                                 plot_params=plot_params, difficulties=difficulties,
                                                 algorithm=algorithm, is_val=False) for port in sim_ports[1:]]

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
        #for env in self.envs:
        #    env.close()