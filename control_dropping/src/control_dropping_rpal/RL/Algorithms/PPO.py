import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ..Agent import Agent
from ..Networks.ActorNetwork import ActorNetwork
from ..Networks.ActorCriticNetwork import ActorNetwork, CriticNetwork
from ..Networks.ExtractorNetworks import TestExtractorVelocity

NUM_ACTIONS = np.array([15, 15, 15])


def PPO(env, checkpoint_dir):
    global NUM_ACTIONS
    """Initiaize the PPO Algorithm"""
    n_epochs = 10
    extractor = TestExtractorVelocity(env.observation_space)
    agent = Agent(
        NUM_ACTIONS,
        ActorNetwork(NUM_ACTIONS, extractor, checkpoint_dir=checkpoint_dir),
        CriticNetwork(extractor, checkpoint_dir=checkpoint_dir),
    )

    def learn():
        for _ in range(n_epochs):
            states, actions, probs, values, rewards, dones, batches = (
                agent.replay_buffer.generate_batches()
            )
            advantage = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (
                        rewards[k]
                        + agent.gamma * values[k + 1] * (1 - int(dones[k]))
                        - values[k]
                    )
                    discount *= agent.gamma * agent.gae_lambda
                advantage[t] = a_t
            advantage = th.from_numpy(advantage).to(agent.device)
            values = th.from_numpy(values).to(agent.device)
            for batch in batches:
                eval_states = th.tensor(states[batch], dtype=th.float).to(agent.device)
                eval_old_probs = th.tensor(probs[batch]).to(agent.device)
                eval_actions = th.tensor(actions[batch]).to(agent.device)

                dist = agent.actor(eval_states)
                critic_value = agent.critic(eval_states)
                critic_value = th.squeeze(critic_value)
                new_probs = dist.log_prob(eval_actions)
                prob_ratio = new_probs.exp() / eval_old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    th.clamp(prob_ratio, 1 - agent.policy_clip, 1 + agent.policy_clip)
                    * advantage[batch]
                )

                actor_loss = -th.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                agent.actor.optimizer.zero_grad()
                agent.critic.optimizer.zero_grad()
                total_loss.backward()
                agent.actor.optimizer.step()
                agent.critic.optimizer.step()

    """Collect Rollout Buffer"""
