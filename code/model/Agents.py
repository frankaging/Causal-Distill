# coding=utf-8

# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function

import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 3)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

class DistillEnv():
    """
    Distillation environment that the agent can act on.
    """
    def __init__(self, verbose = False):
        self.internal_state = []

    def reset(self):
        self.internal_state = []

    def update(self, new_state):
        self.internal_state = new_state

    def step(self, action):
        imitation_states = []
        for layer_idx in range(action.shape[0]):
            batch_imitation_states = []
            for batch_idx in range(action.shape[1]):
                imitation_target = action[layer_idx, batch_idx]
                imitation_state = self.internal_state[imitation_target][batch_idx]
                batch_imitation_states.append(imitation_state)
            batch_imitation_states = torch.stack(batch_imitation_states, dim=0) # [b, dim]
            imitation_states.append(batch_imitation_states)
        return imitation_states

    def step_fix(self, action):
        """
        we step rl agent with fixed policy, selecting on certain layers
        """
        layer_count = action.shape[0]
        return self.internal_state[-layer_count:] # we simply pick the last layers

if __name__ == '__main__':
    actor = Actor(768,3)
    critic = Critic(768,3)