import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from math import inf, radians, degrees
from random import randint
from sklearn.preprocessing import Normalizer
import threading as thr
import sys
sys.path.append('PythonAPI')
# sys.path.append('python')
from ..PythonAPI import sim
import time
from collections import OrderedDict
import os
# import cv2
from PIL import Image
import concurrent.futures



class SimController():
    def __init__(self, port=21313, object_type='Sphere', object_quantity=7):
        '''Get all the required handles'''
        _conn_attempts = 0
        try:
            if _conn_attempts < 2:
                sim.simxFinish(-1) # just in case, close all opened connections
                clientID=sim.simxStart('127.0.0.1', port, True, True, 23000, 5) # Connect to CoppeliaSim
            else:
                exit(1)
            _conn_attempts+=1
            if clientID!=-1:
                print ('Connected to remote API server')
            else:
                print ('Failed connecting to remote API server')
                sys.exit('Could not connect to Coppelia')
        except Exception:
            print('Failed connection')
            exit()
        self.clientID = clientID
        self.normalizer = Normalizer()
        self.arm = self._get_arm_handles()
        self.hand =  self._get_hand_handles()
        self.object_type = object_type
        self.object_quantity = object_quantity
        self.objects = self._get_object_handles(self.object_type, self.object_quantity)
        self.tactile = self._get_tactile_handles()
        self.palm_handle = sim.simxGetObjectHandle(self.clientID, 'BarrettHand_handSensorShape', sim.simx_opmode_blocking)[1]
        print('PALM HANDLE:', self.palm_handle)
        sim.simxGetObjectOrientation(self.clientID, self.palm_handle, -1, sim.simx_opmode_streaming)

        # get the handle of the vison sensors
        self.v1, self.v2, self.v3, self.v4, self.v5, self.v6 = self.get_vision_handles()
        print('CAM HANDLES:', self.v1, self.v2, self.v3, self.v4, self.v5, self.v6)


        # enable the streaming of palm view
        sim.simxGetVisionSensorImage(self.clientID, self.v5, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v3, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v1, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v2, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v4, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v6, 0, sim.simx_opmode_streaming)
        
        # for i in range(5):
        #     time.sleep(2)
        #     print(sim.simxGetObjectOrientation(self.clientID, self.palm_handle, -1, sim.simx_opmode_buffer))

        self.img_num = 0
        self.terminations = [False] * 3
        
        self.hand_KE = 0
        self.cumulative_velocity = 0

        #Data Normalizers
        hand_config, hand_torque, finger1, finger2, finger3, palm, obj_count, current_obj_locations, obj_velocity, rot_velocity, progress = self.load_state_data(os.path.join(os.getcwd(), 'Data_Collection', 'Enviornment_Samples'))
        
        self.normalizers = {
            'hand_config': Normalizer(),
            'hand_torque': Normalizer(),
            'finger_1_tactile': Normalizer(),
            'finger_2_tactile': Normalizer(),
            'finger_3_tactile': Normalizer(),
            'palm_tactile': Normalizer(),
            'ball_count': Normalizer(),
            'ball_location': Normalizer(),
            'obj_velocity': Normalizer(),
            'obj_angular_velocity': Normalizer(),
            'progress_bar': Normalizer()
            }
        data_arr = [hand_config, hand_torque, finger1, finger2, finger3, palm, obj_count, current_obj_locations, obj_velocity, rot_velocity, progress]
        for idx, key in enumerate(['hand_config', 'hand_torque', 'finger_1_tactile', 'finger_2_tactile', 'finger_3_tactile', 'palm_tactile', 'ball_count', 'ball_location', 'obj_velocity', 'obj_angular_velocity', 'progress_bar']):
            if 'tactile' in key:
                data_arr[idx] = np.reshape(data_arr[idx], (data_arr[idx].shape[0], 24))
            self.normalizers[key].fit(data_arr[idx])
        
        #Set up termination threshold
        self.base_threshold_set = [-0.8, -0.9, -1, -1.1, -1.2]
        self.base_threshold = -0.8

        # Get current hand_config
        # Spread, f1, f2, f3 ...
        self.current_hand_config = self.get_hand_config()

        #KEEP TRACK OF LIMITS AND PROVIDE PENALTY IN CASE LIMIT IS EXCEEDED
        self.limits = [0, 132]

        self.limit_signals = [False] * 3

        self.current_arm_config = None
        self.reset_arm_config()

        # Holds previous object locations for comparison
        self.prev_obj_locations = None
        self.stack = [True, True, True]
        
        self.prev_obj_locations = None
        self.prev_sense_map = None
        self.obj_velocity = None
        self.current_sense_map = None
        self.progress_bar = 0
        self.drop_detected = False
        self.current_tactile_locations = None
        
        state = self.get_state_encoding(np.zeros((5*8,)))
        self.current_hand_config = self.get_hand_config()
        self.current_obj_locations = state['obj_location']
        self.prev_obj_locations = np.zeros(shape= self.current_obj_locations.shape)
        
    def normalize_tactile_image(self, image, key):
        original_shape = image.shape
        image = np.reshape(image, (original_shape[0], 24))
        image = self.normalizers[key].transform(image)
        return image.reshape(original_shape)

    def normalize_tactile(self, image, key):
        return self.normalizers[key].transform(np.reshape(image, (1, 24)))
    
    def get_vision_handles(self):
        v1 = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor1', sim.simx_opmode_blocking)[1]
        v2 = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor2', sim.simx_opmode_blocking)[1]
        v3 = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor3', sim.simx_opmode_blocking)[1]
        v4 = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor4', sim.simx_opmode_blocking)[1]
        v5 = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor5', sim.simx_opmode_blocking)[1]
        v6 = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor6', sim.simx_opmode_blocking)[1]
        return v1, v2, v3, v4, v5, v6
        

    def load_state_data(self, path):
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

    def check_palm_orientation(self):
        # print(sim.simxGetObjectOrientation(self.clientID, self.palm_handle, -1, sim.simx_opmode_buffer))
        orientation = degrees(abs(sim.simxGetObjectOrientation(self.clientID, self.palm_handle, -1, sim.simx_opmode_buffer)[1][0]))
        # print(orientation)
        return orientation < 172, (172 - orientation)

    def reinitialize(self):
        '''Get all the required handles'''
        self.arm = self._get_arm_handles()
        self.hand =  self._get_hand_handles()
        self.objects = self._get_object_handles(self.object_type, self.object_quantity)
        self.tactile = self._get_tactile_handles()
        self.palm_handle = sim.simxGetObjectHandle(self.clientID, 'BarrettHand_handSensorShape', sim.simx_opmode_blocking)[1]
        print('PALM HANDLE:', self.palm_handle)
        sim.simxGetObjectOrientation(self.clientID, self.palm_handle, -1, sim.simx_opmode_streaming)


        # get the handle of the vison sensors
        self.v1, self.v2, self.v3, self.v4, self.v5, self.v6 = self.get_vision_handles()
        print('CAM HANDLES:', self.v1, self.v2, self.v3, self.v4, self.v5, self.v6)

        # enable the streaming of palm view
        sim.simxGetVisionSensorImage(self.clientID, self.v5, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v3, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v1, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v2, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v4, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(self.clientID, self.v6, 0, sim.simx_opmode_streaming)

        # Get current hand_config
        # Spread, f1, f2, f3 ...
        self.current_hand_config = self.get_hand_config()

        #KEEP TRACK OF LIMITS AND PROVIDE PENALTY IN CASE LIMIT IS EXCEEDED

        self.reset_arm_config()

    def take_top_image(self, path):
        try:
            err, resolution, image5 = sim.simxGetVisionSensorImage(self.clientID, self.v5, 0, sim.simx_opmode_buffer)
            img5 = np.array(image5, dtype=np.uint8)
            img5.resize([resolution[1],resolution[0],3])
            plt.imsave(os.path.join(path, 'top.jpg'), img5)
        except Exception as e:
            print(e)
            print("ERROR SAVING TOP PHOTO")

    def take_images(self):
        err, resolution1, image1 = sim.simxGetVisionSensorImage(self.clientID, self.v1, 0, sim.simx_opmode_buffer)
        err, resolution2, image2 = sim.simxGetVisionSensorImage(self.clientID, self.v2, 0, sim.simx_opmode_buffer)
        err, resolution3, image3 = sim.simxGetVisionSensorImage(self.clientID, self.v3, 0, sim.simx_opmode_buffer)
        err, resolution4, image4 = sim.simxGetVisionSensorImage(self.clientID, self.v4, 0, sim.simx_opmode_buffer)
        err, resolution5, image5 = sim.simxGetVisionSensorImage(self.clientID, self.v5, 0, sim.simx_opmode_buffer)

        return image1, resolution1, image2, resolution2, image3, resolution3, image4, resolution4, image5, resolution5

    def reset_arm_config(self):
        config = np.array([0, 0, 50, 40, -90, 0])
        sim.simxPauseCommunication(self.clientID,True)
        for handle, angle in zip(self.arm, config):
            sim.simxSetJointTargetPosition(self.clientID, handle, round(radians(angle), 3), sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID,False)
        time.sleep(6)
    
    def step_arm_config(self, step):
        config = []
        for i, (angle, step) in enumerate(zip(self.get_arm_config(), step)):
            if i == 4:
                config.append(angle+step) if -180 <= (angle+step) <= 0 else config.append(angle)
            else:
                config.append(angle)        
                
        sim.simxPauseCommunication(self.clientID,True)
        for handle, angle in zip(self.arm, config):
            sim.simxSetJointTargetPosition(self.clientID, handle, round(radians(angle), 3), sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID,False)


    def reset_termination_signals(self):
        self.terminations = [False] * 3
    
    def reset_limit_signals(self):
        self.limit_signals = [False] * 3
        
    def _check_limits(self, value):
        '''
        Resets angle to 0 if lesser than 0 or resets angle to 140 if angle exceeds
        140. Also passes the flag confirming if limit was exceeded
        '''
        if value < self.limits[0]:
            return True, self.limits[0]
        elif value > self.limits[1]:
            return True, self.limits[1]
        return False, value

    def step_hand_config(self, step, override=False):
        '''
        Checks for terminations, if higher than threshold flip termination
        receives spread, f1, f2, f3, ...
        Return current hand config in degrees
        '''
        self.hand_KE = 0
        for s in step:
            self.hand_KE += s **2
        self.hand_KE = self.hand_KE * 2 * 3.14159 / 180 
            
        config = []
        for i in range(len(step)):
            if i in (1, 2, 3) and (self.terminations[i-1]) and not override:  #and step[i] > 0 
                config.append(self.current_hand_config[i])
                print("Preventing Movement.")
                continue
            elif i == 0 and ((step[i] < 0 and self.current_hand_config[i] + step[i] < 0) or (step[i] > 0 and self.current_hand_config[i] + step[i] > 340.0)):
                config.append(self.current_hand_config[i])
                print("Preventing Movement.")
                continue
            new_angle = self.current_hand_config[i] + step[i]
            config.append(new_angle) if new_angle < self.limits[1] or i not in (1, 2, 3) else config.append(self.limits[1])
            

        self.current_hand_config = np.array(config)
        
        for i in range(1, 4):
            self.limit_signals[i-1], self.current_hand_config[i] = self._check_limits(self.current_hand_config[i])
        
        self.set_hand_config(self.current_hand_config)
        return self.current_hand_config

    def set_hand_config(self, target):
        '''
        input in degrees
        spread, f1, f2, f3, f1c, f2c, f3c
        handle order [finger3base, finger3couple, spread, finger2base, finger2 couple, spread, finger1base, finger1couple]
        '''
        print('setting hand configuration to:', target)
        sim.simxPauseCommunication(self.clientID,True)
        target = [radians(i) for i in target]
        sim.simxSetJointTargetPosition(self.clientID, self.hand[2], target[0] / 2.0 - 1.57, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[5], target[0] / 2.0 - 1.57, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[6], target[1], sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[3], target[2], sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[0], target[3], sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[7], target[4] + 0.733, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[4], target[5] + 0.733, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetPosition(self.clientID, self.hand[1], target[6] + 0.733, sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[2], target[0] / 2.0 - 1.57, sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[5], target[0] / 2.0 - 1.57, sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[0], target[1], sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[3], target[2], sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[6], target[3], sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[1], target[4] + 0.733, sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[4], target[5] + 0.733, sim.simx_opmode_oneshot)
        # sim.simxSetJointTargetPosition(self.clientID, self.hand[7], target[6] + 0.733, sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID,False)
        time.sleep(0.05)
        if sim.simxGetJointForce(self.clientID, self.hand[6], sim.simx_opmode_buffer)[1] < self.base_threshold:
            print('Threshold finger1:', sim.simxGetJointForce(self.clientID, self.hand[6], sim.simx_opmode_buffer)[1])
            self.terminations[0] = True
        else:
            self.terminations[0] = False
        if sim.simxGetJointForce(self.clientID, self.hand[3], sim.simx_opmode_buffer)[1] < self.base_threshold:
            print('Threshold finger2:', sim.simxGetJointForce(self.clientID, self.hand[3], sim.simx_opmode_buffer)[1])
            self.terminations[1] = True
        else:
            self.terminations[1] = False
        if sim.simxGetJointForce(self.clientID, self.hand[0], sim.simx_opmode_buffer)[1] < self.base_threshold:
            print('Threshold finger3:', sim.simxGetJointForce(self.clientID, self.hand[0], sim.simx_opmode_buffer)[1])
            self.terminations[2] = True
        else:
            self.terminations[2] = False
            
    def get_hand_config(self):
        '''
        [spread, f1, f2, f3, f1c, f2c, f3c]
        Returns in degrees
        '''
        raw = [sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_buffer)[1] for handle in self.hand]
        # raw = [(raw[2] + 1.57) * 2, raw[0], raw[3], raw[6], raw[1] - 0.733, raw[4] - 0.733, raw[7] - 0.733]
        raw = [(raw[2] + 1.57) * 2, raw[6], raw[3], raw[0], raw[7] - 0.733, raw[4] - 0.733, raw[1] - 0.733]
        return [degrees(i) for i in raw] 
    
    def get_arm_config(self):
        self.current_arm_config = np.array([round(degrees(sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_buffer)[1])) for handle in self.arm])
        return self.current_arm_config
    
    def get_base_angle_reward(self):
        hand_config = self.get_hand_config()
        # Sum the finger values
        summation = -sum(hand_config[1:4])
        return (summation/420) + 1
        
    def get_torque_info(self):
        '''
        [finger3base, finger3couple, spread, finger2base, finger2 couple, spread, finger1base, finger1couple]
        couple joint torque values
        f1 couple, f2c, f3c
        '''
        return [sim.simxGetJointForce(self.clientID, self.hand[i], sim.simx_opmode_buffer)[1] for i in (7, 4, 1)]
    
    def get_tactile_info(self):
        '''
        palm * 24, finger2 * 34, finger3 * 34, finger1 * 34
        '''
        sensor_info = [sim.simxReadForceSensor(self.clientID, handle, sim.simx_opmode_buffer)[2][2] for handle in self.tactile]
        palm = sensor_info[0:24]
        finger2 = sensor_info[24:58]
        finger3 = sensor_info[58:92]
        finger1 = sensor_info[92:]
        # Figure out what create images does
        _, finger2, finger3, finger1 = self.reduce_tactile(palm, finger2, finger3, finger1)

        _, finger2, finger3, finger1 = self._create_images(palm, finger2, finger3, finger1)

        return palm, finger2, finger3, finger1

    def generate_tactile_sensor_info(self, sensor_info):
        '''
        palm * 24, finger2 * 34, finger3 * 34, finger1 * 34
        '''
        palm = sensor_info[0:24]
        finger2 = sensor_info[24:58]
        finger3 = sensor_info[58:92]
        finger1 = sensor_info[92:]
        # Figure out what create images does
        _, finger2, finger3, finger1 = self.reduce_tactile(palm, finger2, finger3, finger1)

        _, finger2, finger3, finger1 = self._create_images(palm, finger2, finger3, finger1)

        return palm, finger2, finger3, finger1
    
    def reduce_tactile(self, palm, finger2, finger3, finger1):
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
    
    def _create_images(self, palm, finger2, finger3, finger1):
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
    
    def get_object_count(self, height = 0.625):
        count = 0
        for handle in self.objects:
            cur_height = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_buffer)[1][2]
            if cur_height > height:
                # print(cur_height)
                count += 1
        return count

    def get_object_state(self):
        count = self.get_object_count()
        c = np.zeros((10,))
        c[count] = 1
        return c
        
    def get_object_locations(self, height = 0.625):
        locations = []
        count = 0
        # If ball not in hand give location [0,0,0] 
        for handle in self.objects:
            position = sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_buffer)[1]
            if position[2] > height:
                count += 1
                orientation = sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_buffer)[1]
                position += orientation
                locations.append(position)
        for _ in range(len(locations), 7):
            locations.append([0,0,0,0,0,0])
        return locations
    
    def get_sensor_locations(self):
        sensor_info = [sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_buffer)[1] for handle in self.tactile]
        palm = sensor_info[0:24]
        finger2 = sensor_info[24:58]
        finger3 = sensor_info[58:92]
        finger1 = sensor_info[92:]
        # return np.array(sensor_info).flatten()
        return np.array(palm).flatten(), np.array(finger1).flatten(), np.array(finger2).flatten(), np.array(finger3).flatten()
    
    def get_state(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_sensor_locations = executor.submit(self.get_sensor_locations)
            future_locations = executor.submit(self.get_object_locations)
            future_hand_config = executor.submit(self.get_hand_config)
            future_hand_torque = executor.submit(self.get_torque_info)
            future_tactile_info = executor.submit(self.get_tactile_info)
            future_object_count = executor.submit(self.get_object_count)
            sensor_locations = future_sensor_locations.result()
            locations = future_locations.result()
            hand_config = np.array(future_hand_config.result(), dtype = np.float32)
            hand_torque = np.array(future_hand_torque.result(), dtype = np.float32)
            palm, finger2, finger3, finger1 = future_tactile_info.result()
            
        object_count = future_object_count.result()
        # if self.prev_obj_locations is not None:
        self.prev_obj_locations = np.array(self.current_obj_locations)
        self.current_obj_locations = np.array(locations, dtype = np.float32)
        r_values = [] 
        if self.prev_obj_locations is not None:
            self.obj_velocity = self.current_obj_locations - self.prev_obj_locations
            for i in range(3, len(locations), 6):
                v = np.subtract(locations[i:i+3], self.prev_obj_locations[i:i+3])
                r_values.append(v.dot(v)**0.5)
        else:
            self.obj_velocity = np.zeros(shape=self.current_obj_locations.shape)
            r_values = np.zeros(shape=len(self.objects))
        rot_velocity = np.zeros(shape=(10,), dtype= np.float32)
        for i in range(len(r_values)):
            rot_velocity[i] = r_values[i]
        palm = np.reshape(palm, (1,3,8))
        #TODO 
        # normalize hand config value
        # normalize other data using grasping data
        
        #TODO map the tactile locations to the actual locations
        if self.current_sense_map is not None:
            self.prev_sense_map = self.current_sense_map
        else:
            self.prev_sense_sense_map = np.zeros(shape= (20, 30, 3), dtype=np.uint8)
        # self.current_sense_map = self.generate_sense_image(palm, finger1, finger2, finger3, hand_config[0])
        self.current_sense_map = np.zeros(shape= (20, 30, 3), dtype=np.uint8) #TODO: Not in use Right now, not finished!!
        if self.drop_detected:
            self.progress_bar+=1
        
        self.cumulative_velocity = 0 
        for i in range(0, len(self.obj_velocity), 3):
            x = self.obj_velocity[i]
            y = self.obj_velocity[i+1]
            z = self.obj_velocity[i+2]
            r = np.sqrt(x**2 + y**2 + z**2)
            self.cumulative_velocity += r
        

        return {
            'hand_config': self.normalizers['hand_config'].transform(np.array([hand_config])).squeeze(),
            'hand_torque': self.normalizers['hand_torque'].transform(np.array([hand_torque])).squeeze(),
            'finger_1_tactile': self.normalize_tactile_image(finger1, 'finger_1_tactile'),
            'finger_2_tactile': self.normalize_tactile_image(finger2, 'finger_2_tactile'),
            'finger_3_tactile': self.normalize_tactile_image(finger3, 'finger_3_tactile'),
            'palm_tactile': self.normalize_tactile_image(palm, 'palm_tactile'),
            'tactile_pos': self.normalizers['ball_location'].transform(np.array([sensor_locations])).squeeze(),
            'ball_count': np.array([object_count/10]),
            'ball_location': self.normalizers['ball_location'].transform(np.array([self.current_obj_locations])).squeeze(),
            'obj_velocity': self.normalizers['obj_velocity'].transform(np.array([self.obj_velocity])).squeeze(),
            'obj_angular_velocity': self.normalizers['obj_angular_velocity'].transform(np.array([rot_velocity])).squeeze(),
            'progress_bar': np.array([self.progress_bar/35])
            }
    def get_state_encoding(self, action_state):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_sensor_locations = executor.submit(self.get_sensor_locations)
            future_locations = executor.submit(self.get_object_locations)
            future_hand_config = executor.submit(self.get_hand_config)
            future_hand_torque = executor.submit(self.get_torque_info)
            future_tactile_info = executor.submit(self.get_tactile_info)
            future_object_count = executor.submit(self.get_object_count)

            palm_locs, f1_locs, f2_locs, f3_locs = np.nan_to_num(future_sensor_locations.result())
            locations = np.nan_to_num(future_locations.result())
            hand_config = np.nan_to_num(np.array(future_hand_config.result(), dtype = np.float32))
            hand_torque = np.nan_to_num(np.array(future_hand_torque.result(), dtype = np.float32))
            palm, finger2, finger3, finger1 = future_tactile_info.result()
            palm, finger2, finger3, finger1 = np.nan_to_num(palm), np.nan_to_num(finger2), np.nan_to_num(finger3), np.nan_to_num(finger1)
            object_count = future_object_count.result()

        if self.prev_obj_locations is not None:
            self.prev_obj_locations = np.array(self.current_obj_locations)
        self.current_obj_locations = np.array(locations, dtype = np.float32)
        r_values = [] 
        if self.prev_obj_locations is not None:
            self.obj_velocity = self.current_obj_locations - self.prev_obj_locations
            # for i in range(3, len(locations), 6): # Vector Magnitude Calculation
            #     v = np.subtract(locations[i:i+3], self.prev_obj_locations[i:i+3])
            #     r_values.append(v.dot(v)**0.5)
        else:
            self.obj_velocity = np.zeros(shape=self.current_obj_locations.shape)
            # r_values = np.zeros(shape=len(self.objects))
        # rot_velocity = np.zeros(shape=(10,), dtype= np.float32)
        # for i in range(len(r_values)):
            # rot_velocity[i] = r_values[i]
        #   = np.reshape(palm, (1,3,8))
        
        if self.drop_detected:
            self.progress_bar+=1
        
        self.cumulative_velocity = 0 
        for i in range(0, len(self.obj_velocity), 3):
            x = self.obj_velocity[i][0]
            y = self.obj_velocity[i][1]
            z = self.obj_velocity[i][2]
            r = np.sqrt(x**2 + y**2 + z**2)
            self.cumulative_velocity += r
            
        ball_count = np.array([object_count/5 ])
        progress_bar = np.array([self.progress_bar/35])
        hand_config = self.normalizers['hand_config'].transform(np.array([hand_config])).squeeze()
        hand_torque = self.normalizers['hand_torque'].transform(np.array([hand_torque])).squeeze()
        
        # Modify the Positions of each finger:
        # spread, f1, f2, f3, f1c, f2c, f3c
        palm_locs = np.concatenate([[(self.get_arm_config()[4] + 90) / 90, hand_config[0]], palm_locs], axis=0).squeeze() 
        f1_locs = np.concatenate([[hand_config[1], hand_config[4]], f1_locs], axis=0).squeeze()
        f2_locs = np.concatenate([[hand_config[2], hand_config[5]], f2_locs], axis=0).squeeze()
        f3_locs = np.concatenate([[hand_config[3], hand_config[6]], f3_locs], axis=0).squeeze() 

        # Concatenate into a single array
        state_attrib = np.nan_to_num(np.concatenate([ball_count, progress_bar, action_state, hand_torque]))


        return {key: np.nan_to_num(val) for key, val in {
            'finger_1_tactile': self.normalize_tactile(finger1, 'finger_1_tactile'),
            'finger_2_tactile': self.normalize_tactile(finger2, 'finger_2_tactile'),
            'finger_3_tactile': self.normalize_tactile(finger3, 'finger_3_tactile'),
            'finger_1_location': f1_locs,
            'finger_2_location': f2_locs,
            'finger_3_location': f3_locs,
            'palm_location': palm_locs,
            'palm_tactile': self.normalize_tactile(palm, 'palm_tactile'),
            # 'tactile_pos': sensor_locations,
            'obj_location': self.current_obj_locations,
            'obj_velocity': self.obj_velocity,
            'state_attrib': state_attrib
            }.items()}
        # return {
        #     'hand_config': hand_config,
        #     'hand_torque': hand_torque,
        #     'finger_1_tactile': finger1,
        #     'finger_2_tactile': finger2,
        #     'finger_3_tactile': finger3,
        #     'palm_tactile': palm,
        #     'ball_count': np.array([sum(object_count)/10]),
        #     'ball_location': self.current_obj_locations,
        #     'obj_velocity': self.obj_velocity,
        #     'obj_angular_velocity': rot_velocity,
        #     'progress_bar': np.array([self.progress_bar/35])
        #     # 'c_sense_map': self.current_sense_map,
        #     # 'p_sense_map': self.prev_sense_sense_map
        #     }
    
    def get_terminations(self):
        return self.terminations

    def get_limit_signals(self):
        return self.limit_signals

    def get_KE(self):
        return self.hand_KE
    
    def get_cumulative_velocity(self):
        return self.cumulative_velocity*1.2

    def set_object_location(self, object_locations):
        print('Setting object locations')
        sim.simxPauseCommunication(self.clientID,True)
        count = object_locations[0]
        object_locations = object_locations[1:]
        for i in range(int(count)):
            pos = [object_locations[0 + (i*3)], object_locations[1 + (i*3)], object_locations[2+(i*3)]]
            # print(pos)
            sim.simxSetObjectPosition(self.clientID, self.objects[i], -1, pos, sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.clientID,False)
        return count
    
    def _reset_object_location(self):
        print('Resetting object locations')
        for handle in self.objects:
            sim.simxSetObjectPosition(self.clientID, handle, -1, [0,0,0], sim.simx_opmode_oneshot)
        time.sleep(3)

#TODO: Implement a randomization of Sim Parameters: 
    def setup_scene(self, hand_config, object_locations):
        '''input in degrees of hand_config'''
        self._reset_object_location()
        # Set hand
        self.set_hand_config([hand_config[0], 0,0,0,0,0,0])
        time.sleep(3)
        self.set_hand_config(hand_config)
        time.sleep(5)
        self.current_hand_config = hand_config

        self.prev_object_locations = None
        self.stack = [True, True, True]

        #set objects
        self.set_object_location(object_locations)

        # Reducing stress on hand by opening it
        open_incrementer = [0, -1, -1, -1, 0, 0, 0]
        time.sleep(0.5)
        while self.exceeding_start_threshold() == True:
                self.step_hand_config(open_incrementer, override=True)
        time.sleep(0.5)       
        if self.check_palm_orientation()[1] > 0:
            print('Hand orientation to high!')
            return False

        time.sleep(5)

        object_count = self.get_object_count()

        print('object count for initialized scene = ', object_count)
        if object_count < 2:
            return False

        # Reset signals
        self.reset_termination_signals()
        self.reset_limit_signals()
        self.hand_KE = 0
        self.cumulative_velocity = 0
        state = self.get_state_encoding(np.zeros(shape=5*8))
        self.current_obj_locations = state['obj_location']
        self.prev_obj_locations = np.zeros(shape= self.current_obj_locations.shape)
        return True

    def start_threshold_exceeded(self):
        for handle in self.hand:
            force = sim.simxGetJointForce(self.clientID, handle, sim.simx_opmode_buffer)[1]
            if force < -1.5:
                print('Exceeding:', force)
                return True
        return False

    def is_ball_normal(self, object_locations):
        '''receives an array of size 7 and checks z value for every ball location'''
        palm_height = sim.simxGetObjectPosition(self.clientID, self.tactile[0], -1, sim.simx_opmode_oneshot)[1][2]
        for i in range(2, len(object_locations), 3):
            if object_locations[i] >= palm_height:
                return False
        return True

    def _get_arm_handles(self):
        armjoint_handle = []
        for i in range(6):
            err_code, current_armjoint_handle = sim.simxGetObjectHandle(self.clientID, "UR5_joint" + str(i+1), sim.simx_opmode_blocking)
            armjoint_handle.append(current_armjoint_handle)
            sim.simxSetObjectIntParameter(self.clientID, current_armjoint_handle, 2000, 1, sim.simx_opmode_oneshot)
            sim.simxSetObjectIntParameter(self.clientID, current_armjoint_handle, 2001, 1, sim.simx_opmode_oneshot)
            sim.simxGetJointPosition(self.clientID, current_armjoint_handle, sim.simx_opmode_streaming)
        print('ARMJOINT HANDLE:', armjoint_handle)
        return armjoint_handle

    def _get_hand_handles(self):
        handjoint_handle = [sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)[1] for name in ("BarrettHand_jointB_1", 'BarrettHand_jointC_1', 'BarrettHand_jointA_0', 'BarrettHand_jointB_0', \
            'BarrettHand_jointC_0', 'BarrettHand_jointA_2', 'BarrettHand_jointB_2', 'BarrettHand_jointC_2')]
        print('HANDJOINT HANDLE:', handjoint_handle)
        for handle in handjoint_handle:
            sim.simxSetObjectIntParameter(self.clientID, handle, 2000, 1, sim.simx_opmode_oneshot)
            sim.simxSetObjectIntParameter(self.clientID, handle, 2001, 1, sim.simx_opmode_oneshot)
            sim.simxGetJointPosition(self.clientID, handle, sim.simx_opmode_streaming)
            sim.simxGetJointForce(self.clientID, handle, sim.simx_opmode_streaming)
        return handjoint_handle
    
    def get_angle_from_normal(self, ) -> float:
        """ Returns the angle in Degrees from the normal of the palm """
        t6, t14, t17 = self.tactile[7], self.tactile[15], self.tactile[18]
        p_6, p_14, p_17 =  (sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_buffer)[1] for handle in (t6, t14, t17))
        v1, v2 = np.subtract(p_6, p_14), np.subtract(p_17, p_14)
        palm_norm = np.cross(v1, v2)
        palm_norm /= np.linalg.norm(palm_norm)
        world_norm = np.array([0, 0, 1])
        pndotwn = np.clip(np.dot(palm_norm, world_norm) / (np.linalg.norm(palm_norm) * np.linalg.norm(world_norm)), -1.0, 1.0)
        theta = np.degrees(np.arccos(pndotwn))
        print('V1, V2:', v1, v2)
        print('Palm Angle Displacement:', theta)
        return max(min(theta, 180 - theta), 0)
         
    def _get_object_handles(self, object_type, object_quantity):
        object_handle = [sim.simxGetObjectHandle(self.clientID, object_type, sim.simx_opmode_blocking)[1]] + [sim.simxGetObjectHandle(self.clientID,object_type + str(i), sim.simx_opmode_blocking)[1] for i in range(object_quantity - 1)]
        # object_handle = [sim.simxGetObjectHandle(self.clientID, f'/{object_type}[{str(i)}]', sim.simx_opmode_blocking)[1] for i in range(object_quantity)]
        print('OBJECT HANDLES', object_handle)
        for handle in object_handle:
            sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_streaming)
            sim.simxGetObjectOrientation(self.clientID, handle, -1, sim.simx_opmode_streaming)
        return object_handle

    def _get_tactile_handles(self):
        tactile_names = ["BarrettHand_handSensor" + str(i) for i in range(24)] + ["BarrettHand_fingerTipSensor"+ str(finger) + "_" + str(i) for finger in range(3) for i in range(34)]
        tactile_sensor_handle = [sim.simxGetObjectHandle(self.clientID, name, sim.simx_opmode_blocking)[1] for name in tactile_names]
        print('TACTILE HANDLES:', tactile_sensor_handle)
        for handle in tactile_sensor_handle:
            sim.simxReadForceSensor(self.clientID, handle, sim.simx_opmode_streaming)
            sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_streaming)
        return tactile_sensor_handle

    def is_sim_working(self):
        '''
        Checks if simulation gravity is broken
        returns False if objects in the sim are not moving
        returns true if objects in the sim are moving
        
        '''
        print('Checking Sim...')
        
        current_location = [
                [round(i, 3) for i in sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_buffer)[1]]
                for handle in self.objects
        ]
        
        self.prev_object_locations = current_location
        self.set_hand_config([120, 0, 0, 0, 0, 0, 0])
        
        time.sleep(5)
        if self.get_object_count() > 0:  
            self.restart_sim() 
            
        current_location = [
                [round(i, 3) for i in sim.simxGetObjectPosition(self.clientID, handle, -1, sim.simx_opmode_buffer)[1]]
                for handle in self.objects
        ]
        try:
            for l1, l2 in zip(self.prev_object_locations, current_location):
                if l1 != l2:
                    return True
        except:
            print("Restart Succesful")
            return True
            
        
        print('Fixing Simulation...')
        self.restart_sim()      
        
    def exceeding_start_threshold(self):
        for handle in self.hand:
            force = sim.simxGetJointForce(self.clientID, handle, sim.simx_opmode_buffer)[1]
            if force < -1.5:
                print('Exceeding:', force)
                return True
        return False
    
    def reset_sim_objects(self):
        returnCode,outInts,outFloats,outStrings,outBuffer=sim.simxCallScriptFunction(self.clientID, 'Box', sim.sim_scripttype_childscript, "reset_objects",[],[],[], bytearray(), sim.simx_opmode_blocking)
        print('Return Code:', returnCode)
           
    def check_sim_integrity(self):  
        finger_a = sim.simxGetObjectHandle(self.clientID, "BarrettHand_fingerBase0", sim.simx_opmode_blocking)[1]
        finger_b = sim.simxGetObjectHandle(self.clientID, "BarrettHand_fingerBase2", sim.simx_opmode_blocking)[1]

        err_code = sim.simxGetObjectHandle(self.clientID,"Sphere" + str(1), sim.simx_opmode_blocking)[0]

        if sim.simxCheckCollision(self.clientID, finger_a, finger_b, sim.simx_opmode_blocking)[1] or  err_code!= 0:
            if err_code==0:
                print("Detected Hand Abnormalties...")
            else:
                print("Sim was manually stopped...")
            self.restart_sim()
            
    def start_sim(self,):
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        
    def restart_sim(self):
        print("Restarting Sim.")
        # returnCode,outInts,outFloats,outStrings,outBuffer=sim.simxCallScriptFunction(clientID, 'UR5', sim.sim_scripttype_childscript, "reset_hand",[],[],[], bytearray(), sim.simx_opmode_blocking)
        # sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
        # # sim.simxCloseScene(self.clientID, sim.simx_opmode_oneshot)
        # # sim.simxLoadScene(self.clientID, os.path.join(os.getcwd(),'40mm_Sphere_Random_color.ttt'), 0xFF, sim.simx_opmode_blocking)
        # # self.reset_sim_objects()
        # # self.objects = self._get_object_handles(self.object_type, self.object_quantity)
        # time.sleep(10)
        # # _, baseHandle = sim.simxGetObjectHandle(clientID, 'Vehicle', sim.simx_opmode_blocking)
        # # set robot pose
        # # sim.simxSetObjectPosition(clientID, baseHandle, -1, jpos, sim.simx_opmode_blocking)
        # # sim.simxSetObjectOrientation(clientID, baseHandle, -1, jyaw, sim.simx_opmode_oneshot_wait)

        # print ("load scene : " + '40mm_Sphere_Random_color.ttt')
        # # start simulation
        
        # time.sleep(10.0)
        # self.reinitialize()

        # Call Lua script function to reset objects
        _, _, _, _, _ = sim.simxCallScriptFunction(
            self.clientID, 
            'Box', 
            sim.sim_scripttype_childscript, 
            "reset_objects", 
            [],[],[], bytearray(), sim.simx_opmode_blocking
        )
        time.sleep(10)
        print('Reset Objects.')


    # def generate_sense_image(self, palm, f1, f2, f3, spread):
    #     # rotated_image = image[::-1,::-1] #rotate the image 180 degrees
    #     palm = np.reshape(palm, (3, 8)) * 25
    #     f1 = np.reshape(f1, (8, 3)) * 25
    #     f2 = np.reshape(f2, (8, 3)) * 25
    #     f3 = np.reshape(f3, (8, 3)) * 25
    #     # sensor_imgs: [f1, palm, f2, f3]
    #     sensor_imgs = [np.array(cv2.applyColorMap(cv2.convertScaleAbs(sensor,alpha=15),cv2.COLORMAP_JET)) for sensor in (f1, palm, f2, f3)]
    #     f1_img = np.zeros(shape= (20, 10, 3), dtype=np.uint8)
    #     f2_img = np.zeros(shape= (20, 10, 3), dtype=np.uint8)
    #     f3_img = np.zeros(shape= (20, 10, 3), dtype=np.uint8)

    #     #f1img TODO: needs flip along axis 1 of row
    #     for i in range(1, 9):
    #         for j in range(1, 4):
    #             f1_img[i,j] = sensor_imgs[0][i-1, j-1]
    #     #f2img TODO: needs flip along axis 1 of row
    #     for i in range(1, 9):
    #         for j in range(6, 9):
    #             f2_img[i,j] = sensor_imgs[2][i-1, j-6]

    #     #f3_img 
    #     for i in range(7, 10):
    #         for j in range(1, 9):
    #             f3_img[i, j] = sensor_imgs[1][i-7, j-1] 
    #     for i in range(11, 19):
    #         for j in range(4, 7):
    #             f3_img[i, j] = sensor_imgs[3][i-11, j-4]

    #     full_img = np.concatenate((f2_img, f3_img, f1_img), axis=1)
    #     # print(f1_img.shape)      
    #     full_image =Image.fromarray(full_img)
    #     # f2_image = Image.fromarray(f2_img)
    #     full_image.save('./imgs/{}.png'.format(self.img_num))
    #     # f2_image.save('./imgs/f2_{}.png'.format(img_num))
    #     self.img_num+=1
    #     return full_img


# sim_controller = SimController(clientID, )
# print(sim_controller.is_sim_working())
# print(sim_controller.is_sim_working())
# print(sim_controller.is_sim_working())
# print(sim_controller.is_sim_working())
# print(sim_controller.is_sim_working())



