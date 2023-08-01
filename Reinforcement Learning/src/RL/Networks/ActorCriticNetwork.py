import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, FeaturesExtractor, beta=0.008, fc1_dims=256, fc2_dims=256, actions=[], ModelName="ActorNet",checkpoint_dir=None):
        super(ActorNetwork, self).__init__()
        if checkpoint_dir!= None:
            self.checkpoint_file = os.path.join(checkpoint_dir, ModelName)
        else:
            self.checkpoint_file = None
        
        self._is_multi_discrete = False
        self._action_nums = actions
        self._num_action_sets = len(self._action_nums)
        self._is_multi_discrete = self._num_action_sets > 1
        idx = 0
        
        self.n_actions = n_actions
        self.FeaturesExtractor = FeaturesExtractor
        self.actor_network = nn.Sequential(
            nn.Linear(FeaturesExtractor.output_dim, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, sum(self._action_nums)),   
            nn.Softmax(dim=1)
        )
        
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=beta)
        
        
        
    def forward(self, state):
        return Categorical(self.actor_network(self.FeaturesExtractor(state)))
    
    def save_checkpoint(self):
        if self.checkpoint_file != None:
            th.save(self.actor_network.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(th.load(self.checkpoint_file))
    
class CriticNetwork(nn.Module):
    def __init__(self, FeaturesExtractor, alpha=0.024, fc1_dims=256, fc2_dims=256, ModelName="CriticNet", checkpoint_dir=None):
        super(CriticNetwork, self).__init__()
        if checkpoint_dir!= None:
            self.checkpoint_file = os.path.join(checkpoint_dir, ModelName)
        else:
            self.checkpoint_file = None
        
        self.FeaturesExtractor = FeaturesExtractor
        self.critic_network = nn.Sequential(
            nn.Linear(FeaturesExtractor.output_dim, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=alpha)
    
    def forward(self, state):
        return self.critic_network(self.FeaturesExtractor(state))
    
    def save_checkpoint(self):
        if self.checkpoint_file != None:
            th.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(th.load(self.checkpoint_file))