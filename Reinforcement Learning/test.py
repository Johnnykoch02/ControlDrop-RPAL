import os
import sys

import numpy as np

import sklearn

import sklearn
import gzip

from sklearn.cluster import KMeans
def read_files():
        file_name = os.path.join(os.getcwd(),'Reinforcement Learning','src','RL', 'Training','Data','40_mm_sphere_train_control_drop_scenes.npy.gz')
        with gzip.GzipFile(file_name) as f:
            data = np.load(f, allow_pickle= True,)
        # train_path = os.path.join(CHECKPOINT_DIR)
        # names = os.listdir(train_path)
        # names = [re.sub('[a-z_.]', '', name) for name in names]
        
        return data


os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from stable_baselines3.common import env_checker
from src.RL.callbacks import TrainAndLoggingCallback 
from src.RL.custom_gym import  NAME, CHECKPOINT_DIR, LOG_DIR, SAVE_FREQ, BerrettHandGym
from src.RL.Networks.ExtractorNetworks import MMExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import threading as t
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import glob
from PIL import Image
import re
from math import inf, radians, degrees


import time as t

import cv2


     


gym = BerrettHandGym(test=True, sim_port=19876)
sim_controller = gym.simController
# sim_controller.restart_sim()
# state = sim_controller.get_state() 
# file_num = 0




def Clustering():
    global sim_controller
    global gym
    data = read_files()
    print("Shape of data:", data.shape)

    clusters = []
    inputStr = ''
    while inputStr != 'n':
        sim_controller.set_hand_config([120, 0, 0, 0, 0, 0, 0])        
        configs_index = np.random.randint(0, len(data))
        new_scene = data[configs_index]
        hand_config, ball_locations = [degrees(i) for i in new_scene[-7::]], new_scene
        hand_config = [hand_config[i] for i in (0, 3, 2, 1, 6, 5, 4)]
        if sim_controller.setup_scene(hand_config, ball_locations):
            pass
        else:
            inputStr = input()
        inputStr = input()
        if inputStr == 's':
            continue
        if inputStr == 'y':
            pass
            clusters.append(new_scene)

save_dir = os.path.join(os.getcwd(), "Data_Collection", "ValidationDifficultys")

easy = []
medium = []
hard = []

inStr = ''
for _ in range(len(gym.configs)):
    # sim_controller.set_hand_config([120, 0, 0, 0, 0, 0, 0])        
    gym.reset()
    inStr = input()
    if inStr == 'e':
        easy.append(gym.configs[gym.configs_index])
    elif inStr == 'm':
        medium.append(gym.configs[gym.configs_index])
    elif inStr == 'h':
        hard.append(gym.configs[gym.configs_index])
    elif inStr == 'x':
        break

'''Save the Data'''
np.save(os.path.join(save_dir, 'easy.npy'), np.array(easy))
np.save(os.path.join(save_dir, 'medium.npy'), np.array(medium))
np.save(os.path.join(save_dir, 'hard.npy'), np.array(hard))

            


def generate_tactile_sensor_info(sensor_info):
        '''
        palm * 24, finger2 * 34, finger3 * 34, finger1 * 34
        '''
        palm = sensor_info[0:24]
        finger2 = sensor_info[24:58]
        finger3 = sensor_info[58:92]
        finger1 = sensor_info[92:]
        # Figure out what create images does
        _, finger2, finger3, finger1 = reduce_tactile(palm, finger2, finger3, finger1)

        _, finger2, finger3, finger1 = create_images(palm, finger2, finger3, finger1)

        return palm, finger2, finger3, finger1
    
def reduce_tactile(palm, finger2, finger3, finger1):
        '''
        Returns palm, finger2, finger3, finger1
        '''
        palm_reading = np.array(palm, dtype = np.float32)
        finger2_reading = np.zeros(24, dtype = np.float32)
        finger3_reading = np.zeros(24, dtype = np.float32)
        finger1_reading = np.zeros(24, dtype = np.float32)
        for iteration in range(24):
            if iteration < 18:
                finger2_reading[iteration] = (finger2[iteration] + finger2[iteration + 3]) / 2
                finger3_reading[iteration] = (finger3[iteration] + finger3[iteration + 3]) / 2
                finger1_reading[iteration] = (finger1[iteration] + finger1[iteration + 3]) / 2
            elif iteration > 17 and iteration < 21:
                finger2_reading[iteration] = (finger2[iteration + 2] + (finger2[iteration + 5] + finger2[iteration + 6]) /2 ) / 2
                finger3_reading[iteration] = (finger3[iteration + 2] + (finger3[iteration + 5] + finger3[iteration + 6]) /2 ) / 2  
                finger1_reading[iteration] = (finger1[iteration + 2] + (finger1[iteration + 5] + finger1[iteration + 6]) /2 ) / 2
            else:
                finger2_reading[iteration] = (finger2[iteration + 6] + finger2[iteration + 9]) / 2
                finger3_reading[iteration] = (finger3[iteration + 6] + finger3[iteration + 9]) / 2
                finger1_reading[iteration] = (finger1[iteration + 6] + finger1[iteration + 9]) / 2
        
        return palm_reading, finger2_reading, finger3_reading, finger1_reading
    
def create_images(palm, finger2, finger3, finger1):
        palm_image = np.zeros((7, 4))
        palm_image[0, 0] = -10
        palm_image[0, 3] = -10
        palm_image[6, 0] = -10
        palm_image[6, 3] = -10
        for i in range(len(palm)):
            palm_image[0, 1] = palm[0]
            palm_image[0, 2] = palm[1]
            palm_image[1:6, :] = np.reshape(palm[2:22], (5, 4))
            palm_image[6, 1] = palm[22]
            palm_image[6, 2] = palm[23]
        # Fixed
        # TODO CHANGE TO 1, 3, 8 when testting
        finger3 = np.reshape(finger3, (1, 3, 8))
        # lower
        finger2 = np.reshape(finger2, (1, 3, 8))
        finger1 = np.reshape(finger1, (1, 3, 8))
        return palm_image, finger2, finger3, finger1
    
def rotate180(arr):
    return image[::-1,::-1] #rotate the image 180 degrees

img_num = 0
def generate_sense_image(palm, f1, f2, f3, spread):
    global img_num
    # rotated_image = image[::-1,::-1] #rotate the image 180 degrees
    palm = np.reshape(palm, (3, 8)) * 25
    f1 = np.reshape(f1, (8, 3)) * 25
    f2 = np.reshape(f2, (8, 3)) * 25
    f3 = np.reshape(f3, (8, 3)) * 25
    # sensor_imgs: [f1, palm, f2, f3]
    sensor_imgs = [np.array(cv2.applyColorMap(cv2.convertScaleAbs(sensor,alpha=15),cv2.COLORMAP_JET)) for sensor in (f1, palm, f2, f3)]
    f1_img = np.zeros(shape= (20, 10, 3), dtype=np.uint8)
    f2_img = np.zeros(shape= (20, 10, 3), dtype=np.uint8)
    f3_img = np.zeros(shape= (20, 10, 3), dtype=np.uint8)
    
    #f1img
    for i in range(1, 9):
        for j in range(1, 4):
            f1_img[i,j] = sensor_imgs[0][i-1, j-1]
    # center_f1 = (8,2)
    # th_f1 = 360 - spread/2
    
    # rot_mat_f1 = cv2.getRotationMatrix2D(center=center_f1, angle=th_f1, scale=1)
    # f1_img = np.array(cv2.warpAffine(src=f1_img, M=rot_mat_f1, dsize=f1_img.shape[:2])).reshape(f1_img.shape)
    
    #f2img
    for i in range(1, 9):
        for j in range(6, 9):
            f2_img[i,j] = sensor_imgs[2][i-1, j-6]
    
    #f3_img 
    for i in range(7, 10):
        for j in range(1, 9):
            f3_img[i, j] = sensor_imgs[1][i-7, j-1] 
    for i in range(11, 19):
        for j in range(4, 7):
            f3_img[i, j] = sensor_imgs[3][i-11, j-4]
    
    full_img = np.concatenate((f2_img, f3_img, f1_img), axis=1)
    print(full_img.shape)
    # print(f1_img.shape)      
    # full_image =Image.fromarray(full_img)
    # full_image.save('./imgs/{}.png'.format(img_num))
    img_num+=1

policy_kwargs = dict(
    features_extractor_class=MMExtractor,# pi: action, vf: value
    net_arch=[512, 256, dict(pi=[128,64], vf=[128,64])]
)
factor = 0.999
stop_factor_lr = 1e-5
lr = 0.005

def lr_call(step):
    global lr, stop_factor_lr, factor
    lr = max(stop_factor_lr, lr * factor)
    return lr

def Demo(model_loc):
    model = PPO('MultiInputPolicy',
            BerrettHandGym(),
            policy_kwargs=policy_kwargs, 
            verbose = 1,
            learning_rate = lr_call,
            n_steps = 128,
            tensorboard_log=LOG_DIR
        )
    env = model.get_env()
    obs = env.reset()
    i = 0
    done = False
    while i < 10000:
        if done:
            env.reset()
        i+=1
        action, _states = model.predict(obs.copy())
        obs, rewards, done, info = env.step(action)





# Demo(None)




# import gzip

# observations = np.load( gzip.GzipFile('Reinforcement Learning/src/RL/Training/PreTraining/observations.npy.gz', 'r'))

# i = 0
# starts = 0
# for observation in observations:
#     sensor_info = observation[54:180]
#     start = t.time()
#     palm, f1, f2, f3 = generate_tactile_sensor_info(sensor_info)
#     # sensors = map(lambda x: (th.from_numpy(np.reshape(x, (8,3))).to("cuda").absolute() * 25).clamp(0,255).to(th.uint8).cpu().numpy(), sensors)
#     # sensors = map(lambda x: (th.from_numpy(np.reshape(x, (8,3))).to("cuda")* 25).cpu().numpy(), sensors)
#     generate_sense_image(palm, f1, f2, f3, 120)
#     i+=1
#     if i > 100:
#         break

# print("Total Time for Conversion on {} Observations: {}".format(i, starts) )

# print("Time per single conversion set: {} ms".format((starts/i)*1000))