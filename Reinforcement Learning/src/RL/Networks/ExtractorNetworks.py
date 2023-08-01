# from .custom_gym import BarrettHandGym
from stable_baselines3 import PPO
import os
from typing import Dict, List
import gym
from gym import spaces

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(MMExtractor, self).__init__(observation_space, features_dim= 1)
        extractors = {}
        total_concat_size = 0               
        for key, subspace in observation_space.spaces.items():
                if key == 'hand_config':
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm1d(7),
                        nn.Linear(7, 8),
                        nn.LeakyReLU()
                    )
                    total_concat_size+=8
                elif key == 'hand_torque':
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm1d(3),
                        nn.Linear(3, 4),
                        nn.LeakyReLU()
                    )
                    total_concat_size+=4
                # elif key == 'palm_tactile':
                #     extractors[key] = nn.Sequential(#8*3*1
                #          nn.BatchNorm2d(subspace.shape[0]),
                #          nn.Conv2d(subspace.shape[0], 32, (3, 3), padding='same'),
                #     )
                elif  'tactile' in key:
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm2d(subspace.shape[0]),
                        nn.Conv2d(subspace.shape[0], 32, (3, 3), padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(32, 64, (3,3), padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(64, 128, (3, 3), padding= 'same'),
                        nn.LeakyReLU(),
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),
                    )
                    total_concat_size+=128
                elif key == 'ball_count':
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm1d(subspace.shape[0]),
                        nn.Linear(10, 16),
                        nn.LeakyReLU()
                    )
                    total_concat_size+=16
                elif key == 'ball_location':
                    extractors[key] = nn.Sequential(
                        nn.BatchNorm1d(subspace.shape[0]),
                        nn.Linear(42, 64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 128), # TODO: this change should be tested
                        nn.LeakyReLU(),
                        nn.LayerNorm()
                    )
                    total_concat_size+=128

        self.extractors = nn.ModuleDict(extractors)
        self._LinearLayers = nn.Sequential(
            nn.Linear(total_concat_size, 3072),
            nn.LeakyReLU(),
            nn.Dropout(0.01),
            nn.Linear(3072, 1536),
            nn.LeakyReLU()
        )
        
        self._features_dim = 1536
    
    def forward(self, observations):
        encoded_tensor_list = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key, extractor in self.extractors.items():
            # print('Key:', key, 'Extractor:', extractor)
            encoded_tensor_list.append(extractor(observations[key]))
    
        return self._LinearLayers(th.cat(encoded_tensor_list, dim= 1))
    
class TestExtractorVelocity(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(TestExtractorVelocity, self).__init__(observation_space, features_dim= 1)
        extractors = {}
        total_concat_size = 0               
        for key, subspace in observation_space.spaces.items():
                if key == 'hand_config':
                    extractors[key] = nn.Sequential(
                        nn.Linear(7, 42),
                        nn.LeakyReLU()
                    )
                    total_concat_size+=42
                elif key == 'hand_torque':
                    extractors[key] = nn.Sequential(
                        nn.Linear(3, 42),
                        nn.LeakyReLU(),
                    )
                    total_concat_size+=42
                # elif key == 'palm_tactile':
                #     extractors[key] = nn.Sequential(#8*3*1
                #          nn.BatchNorm2d(subspace.shape[0]),
                #          nn.Conv2d(subspace.shape[0], 32, (3, 3), padding='same'),
                #     )
                elif  'tactile' in key:
                    extractors[key] = nn.Sequential(
                        nn.Conv2d(subspace.shape[0], 32, (3, 3), padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(32, 64, (3,3), padding='same'),
                        nn.LeakyReLU(),
                        nn.Conv2d(64, 128, (3, 3), padding= 'same'),
                        nn.LeakyReLU(),
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),
                    )
                    total_concat_size+=128
                elif key == 'ball_count':
                    extractors[key] = nn.Sequential(
                        nn.Linear(1, 16),
                        nn.Tanh(),
                    )
                    total_concat_size+=16
                elif key == 'ball_location':
                    extractors[key] = nn.Sequential(
                        nn.Linear(42, 128), # TODO: this change should be tested
                        nn.Tanh(), # test this change as opposed to the 
                    )
                    total_concat_size+=128
                elif key == 'obj_velocity': # Same as Location
                    extractors[key] = nn.Sequential(
                        nn.Linear(42, 128), 
                        nn.Tanh(),
                    )
                    total_concat_size+=128
                elif key == 'obj_angular_velocity':
                    extractors[key] = nn.Sequential(
                    nn.Linear(10, 128),
                    nn.Tanh(),
                    )
                    total_concat_size+=128
                elif key == 'progress_bar':
                    extractors[key] = nn.Sequential(
                    nn.Linear(1, 24),
                    nn.Tanh()
                    )
                    total_concat_size+=24
                
                
                elif key == 'previous_actions':
                    extractors[key] = nn.Sequential(
                        nn.Linear(15, 256),
                        nn.Tanh(),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.LayerNorm(256,),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.LayerNorm(256,),
                    )
                    total_concat_size+=256
        self.encoding_size = 1092
        self.extractors = nn.ModuleDict(extractors)
        # Pre-encoding for LSTM
        self._LinearLayers = nn.Sequential(
            nn.Linear(total_concat_size, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(0.1), # Maybe we can get rid of dropout
            nn.LayerNorm(self.encoding_size),
            nn.Linear(self.encoding_size, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.encoding_size),
            nn.Linear(self.encoding_size, self.encoding_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.encoding_size),
        )
        # Layer Weight Optimization
        prev_layer = None
        for layer in self._LinearLayers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
                prev_layer = layer
        
        self._features_dim = self.encoding_size#total_concat_size
        self.output_dim = self.encoding_size#total_concat_size
    
    def forward(self, observations):
        encoded_tensor_list = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return self._LinearLayers(th.cat(encoded_tensor_list, dim= 1))
    
   
class BaseTransformerFeatureEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, vec_encoding_size=128, decoder_size=1024, device='cuda'):
        super(BaseTransformerFeatureEncoder, self).__init__(observation_space, features_dim= 1)
        extractors = {}
        total_concat_size = 0   
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        
        self.pos_keys = [
            'tactile_pos', 'ball_location', 'state_attrib'
        ]
        
        self.features_keys = [
            'palm_tactile', 'finger_1_tactile', 'finger_2_tactile', "finger_3_tactile", 'obj_velocity'
        ]
        self.feature_encoders = {}
        self.positional_encoding_size = 0
        for key, subspace in observation_space.spaces.items():
                if key in self.pos_keys:
                    self.positional_encoding_size+= subspace.shape[0]
                elif key in self.features_keys:
                    self.feature_encoders[key] = {
                        'value_enc': nn.Sequential (
                            nn.Linear(subspace.shape[0], self.vec_encoding_size, device=device),
                            nn.Sigmoid(),
                            nn.Linear(self.vec_encoding_size, self.vec_encoding_size, device=device),
                            nn.Sigmoid(),
                            nn.Linear(self.vec_encoding_size, int(self.vec_encoding_size/2), device=device),
                            nn.Sigmoid(),
                        )
                    }
        for key in self.features_keys:
            self.feature_encoders[key]['pos_enc'] = nn.Sequential (
                            nn.Linear(self.positional_encoding_size, self.positional_encoding_size, device=device),
                            nn.ReLU(),
                            nn.Linear(self.positional_encoding_size, self.vec_encoding_size, device=device),
                            nn.ReLU(),
                            nn.Linear(self.vec_encoding_size, int(self.vec_encoding_size/2), device=device),
                            nn.ReLU(),
                        )
            
        self.trns_encoder = nn.TransformerEncoderLayer (
            batch_first=True,
            d_model=self.vec_encoding_size, 
            nhead=8,
            dim_feedforward=2048,
            dropout=0.15,
            device=device,
        )
        
        output_shape = (1, len(self.features_keys), self.vec_encoding_size)
        
        flatten_size = output_shape[1] * output_shape[2]
        
        self.decoder_size = decoder_size
        
        self.decoder = nn.Sequential(
            nn.Flatten(), # Flatten the transformer output
            nn.Linear(flatten_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.LayerNorm(self.decoder_size),
            nn.Dropout(p=0.1),
        ) # Triple Layer Decoder
            
        self._features_dim = self.decoder_size
        self.output_dim = self.decoder_size
    
    def forward(self, observations):
        positional_tensors = []
        encoded_features_tensors = {}
        trf_input = []
        '''extractors contain nn.Modules that do all of our processing '''
        for key in self.features_keys:
            encoded_features_tensors[key] = self.feature_encoders[key]['value_enc'](observations[key].to(self.device))
        for key in self.pos_keys:
            positional_tensors.append(observations[key].to(self.device))
            
        pos_tensor = th.cat(positional_tensors, dim= 1)
        for key in self.features_keys:
            encoded_pos_tensor = self.feature_encoders[key]['pos_enc'](pos_tensor).unsqueeze(dim=1)
            trf_input.append(th.cat([encoded_features_tensors[key].unsqueeze(dim=1), encoded_pos_tensor], dim=2).to(self.device))     
        
        trf_input_tensor = th.cat(trf_input, dim=1).to(self.device)
        trf_output = self.trns_encoder(trf_input_tensor)
        features_output = self.decoder(trf_output)
        
        return th.nan_to_num(features_output)

class ObjectTactileEncoder(nn.Module):
    def __init__(self, vec_encoding_size=64, tactile_dim= 2+34*3, object_dim = 6, decoder_size=512, device='cuda'):
        super(ObjectTactileEncoder, self).__init__()
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        
        self.tactile_dim = tactile_dim
        self.object_dim = object_dim

        self.tactile_values = [
            'palm_tactile', 'finger_1_tactile', 'finger_2_tactile', "finger_3_tactile",
        ]
        self.tactile_positions = [
            'palm_location', 'finger_1_location', 'finger_2_location', 'finger_3_location'
        ]
        
        self.object_projections = [
             'obj_velocity'
        ]
        
        self.object_positions = [
            'obj_location'
        ]

        # Used to project our features to the right dimension
        self.tactile_val_projection = nn.Linear(self.observation_space['palm_tactile'].shape[0], self.vec_encoding_size//2, self.device)
        self.tactile_pos_projection = nn.Linear(self.tactile_dim, self.vec_encoding_size//2, self.device)
        self.object_val_projection = nn.Linear(self.object_dim, self.vec_encoding_size//2, self.device)
        self.object_pos_projection = nn.Linear(self.object_dim, self.vec_encoding_size//2, self.device)

        transformer_layer = nn.TransformerEncoderLayer (
            batch_first=True,
            d_model=self.vec_encoding_size, 
            nhead=8,
            dim_feedforward=1028,
            dropout=0.15,
            device=self.device,
        )
# Stack 4 of these layers together
        self.trns_encoder  = nn.TransformerEncoder(transformer_layer, num_layers=4, device=self.device)
        self.output_shape = (1, self.observation_space['obj_velocity'].shape[0] + len(self.tactile_values), self.vec_encoding_size)
        self.flatten_size = self.output_shape[1] * self.output_shape[2]
    
    def forward(self, observations):
        positional_tensors = []
        encoded_features_tensors = {}
        observations = {key: th.nan_to_num(observations[key]) for key in observations.keys()}
        tac_tensors = []
        for t_val, t_pos in zip(self.tactile_values, self.tactile_positions): # First we must pad our tactile pos vectors to the same length
            pad = max(0, self.tactile_dim - observations[t_pos].shape[1])
            # tac_tensors: (Value, Position)
            tac_tensors.append(
                (observations[t_val],
                F.pad(observations[t_pos], (0, pad), "constant", 0)) # Position
                )
        
        # Projections to the right dimension
        ''' Tactiles '''
        tac_proj_vectors = []
        for t_val_vec, t_pos_vec in tac_tensors:
            # Cat the tactile val projection and the and pos projection
            tac_proj_vectors.append(th.concatenate([self.tactile_val_projection(t_val_vec).unsqueeze(dim=1),self.tactile_pos_projection(t_pos_vec).unsqueeze(dim=1)],dim=2)) # [(1, 128), (1, 128), ... ]
        tac_proj_vectors = th.cat(tac_proj_vectors, dim=1).to(self.device) # (n, 4, 128)

        ''' Objects '''
        obj_pos_tensor = observations['obj_location']
        obj_val_tensor = observations['obj_velocity']
        obj_proj_vectors = th.concatenate([self.object_val_projection(obj_val_tensor),
            self.object_pos_projection(obj_pos_tensor)], dim=2).to(self.device) # (7, 128)

        trf_input_tensor = th.concatenate([tac_proj_vectors, obj_proj_vectors], dim=1).to(self.device)
        trf_output = self.trns_encoder(trf_input_tensor)



    
class MultiModalTransformerFeatureEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, vec_encoding_size=64, tactile_dim= 2+34*3, object_dim = 6, decoder_size=512, device='cuda'):
        super(MultiModalTransformerFeatureEncoder, self).__init__(observation_space, features_dim= 1)
        extractors = {}
        total_concat_size = 0   
        self.vec_encoding_size = vec_encoding_size
        self.device = device
        self.to(self.device)
        
        self.observation_space = observation_space
        self.tactile_dim = tactile_dim
        self.object_dim = object_dim
        
        #TODO: Add an Angular Velocity to the Value vector
        
        self.tactile_values = [
            'palm_tactile', 'finger_1_tactile', 'finger_2_tactile', "finger_3_tactile",
        ]
        self.tactile_positions = [
            'palm_location', 'finger_1_location', 'finger_2_location', 'finger_3_location'
        ]
        
        self.object_projections = [
             'obj_velocity'
        ]
        
        self.object_positions = [
            'obj_location'
        ]
        
        self.join_keys = [
            'state_attrib'
        ]
        
        # Used to project our features to the right dimension
        self.tactile_val_projection = nn.Linear(self.observation_space['palm_tactile'].shape[0], self.vec_encoding_size//2, self.device)
        self.tactile_pos_projection = nn.Linear(self.tactile_dim, self.vec_encoding_size//2, self.device)
        self.object_val_projection = nn.Linear(self.object_dim, self.vec_encoding_size//2, self.device)
        self.object_pos_projection = nn.Linear(self.object_dim, self.vec_encoding_size//2, self.device)
        
        self.trns_encoder = nn.TransformerEncoderLayer (
            batch_first=True,
            d_model=self.vec_encoding_size, 
            nhead=8,
            dim_feedforward=1028,
            dropout=0.15,
            device=device,
        )
        
        output_shape = (1, self.observation_space['obj_velocity'].shape[0] + len(self.tactile_values), self.vec_encoding_size)
        
        flatten_size = output_shape[1] * output_shape[2] + self.observation_space['state_attrib'].shape[0]
        self.decoder_size = decoder_size
        
        self.decoder = nn.Sequential(
            nn.Linear(flatten_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.decoder_size, self.decoder_size, device=device),
            nn.ReLU(),
            # nn.LayerNorm(self.decoder_size),
            nn.Dropout(p=0.1),
        ) # Triple Layer Decoder
            
        self._features_dim = self.decoder_size
        self.output_dim = self.decoder_size
    
    def forward(self, observations):
        positional_tensors = []
        encoded_features_tensors = {}
        observations = {key: th.nan_to_num(observations[key]) for key in observations.keys()}
        tac_tensors = []
        for t_val, t_pos in zip(self.tactile_values, self.tactile_positions): # First we must pad our tactile pos vectors to the same length
            pad = max(0, self.tactile_dim - observations[t_pos].shape[1])
            # tac_tensors: (Value, Position)
            tac_tensors.append(
                (observations[t_val],
                F.pad(observations[t_pos], (0, pad), "constant", 0)) # Position
                )
        
        # Projections to the right dimension
        ''' Tactiles '''
        tac_proj_vectors = []
        for t_val_vec, t_pos_vec in tac_tensors:
            # Cat the tactile val projection and the and pos projection
            tac_proj_vectors.append(th.concatenate([self.tactile_val_projection(t_val_vec).unsqueeze(dim=1),self.tactile_pos_projection(t_pos_vec).unsqueeze(dim=1)],dim=2)) # [(1, 128), (1, 128), ... ]
        tac_proj_vectors = th.cat(tac_proj_vectors, dim=1).to(self.device) # (n, 4, 128)

        ''' Objects '''
        obj_pos_tensor = observations['obj_location']
        obj_val_tensor = observations['obj_velocity']
        obj_proj_vectors = th.concatenate([self.object_val_projection(obj_val_tensor),
            self.object_pos_projection(obj_pos_tensor)], dim=2).to(self.device) # (7, 128)

        trf_input_tensor = th.concatenate([tac_proj_vectors, obj_proj_vectors], dim=1).to(self.device)
        trf_output = self.trns_encoder(trf_input_tensor)
        decoder_input = th.concatenate([trf_output.view((trf_output.shape[0], -1)), observations['state_attrib']], dim=1).to(self.device)
        features_output = self.decoder(decoder_input)
        return th.nan_to_num(features_output)