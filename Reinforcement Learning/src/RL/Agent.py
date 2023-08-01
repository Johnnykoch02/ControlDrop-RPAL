from .Algorithms.Utils.ReplayBuffer import ReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor
import torch as th

class Agent:
    def __init__(self, n_actions, actor, critic, gamma=0.99, alpha=0.0003, policy_clip=0.1, batch_size=512, gae_lambda=0.95, N=2048, n_epochs= 10): 
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.policy_clip = policy_clip
        self.replay_buffer = ReplayBuffer(batch_size)
        self.actor = actor
        self.critic = critic
        self.device = critic.device
    
        self.gae_lambda = gae_lambda
        self.N = N
        self.n_epochs = n_epochs
        
    def store_transition(self, state, action, probs, vals, reward, done):
        self.replay_buffer.store_memory(state, action, probs, vals, reward, done)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
    def choose_action(self, observation):
        obs_tensor = obs_as_tensor(observation, self.actor.device)
        
        dist = self.actor(obs_tensor)
        value = self.critic(obs_tensor)
        action = (dist.sample())
        
        probs = th.squeeze(dist.log_prob(action)).item()
        action = th.squeeze(action).item()
        value = th.squeeze(value).item()
        
        return action, probs, value
    