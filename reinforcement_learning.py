##
## EPITECH PROJECT, 2023
## Deep-learning-cars
## File description:
## reinforcement_learning.py
##

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


lr = 1e-3
gamma = 0.995

episodes = 10

env = game_environnement()

class NeuralNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.lin1 = nn.Linear(1, 4)
        self.batch1 = nn.BatchNorm1d(4)
        self.lin2 = nn.Linear(4, 8)
        self.batch2 = nn.BatchNorm1d(8)
        self.lin3 = nn.Linear(8, 4) #le 4 signifie le nombre d'actions que l'ont peut faire 

        self.actions, self.states, self.rewards = [],[],[]
    
    def forward(self, x):
        x = x.float()
        x = F.relu(self.batch1(self.lin1(x)))
        x = F.relu(self.batch2(self.lin2(x)))
        x = F.softmax(self.lin3(x), dim = -1)
        return x
    
    def policy_action(self, state):
        probability = self.forward(state)
        categories = Categorical(probability)
        act = m.sample()
        return act.item()
    
    def remember(self, Action, State, Reward):
        self.actions.append(Action)
        self.states.append(State)
        self.rewards.append(Reward)

    def discount_rewards(self):
        dc_reward = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            dc_reward.insert(0, R)
        self.rewards = dc_reward

rl = NeuralNetwork()


