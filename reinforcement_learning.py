##
## EPITECH PROJECT, 2023
## Deep-learning-cars
## File description:
## reinforcement_learning.py
##
import environment
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


lr = 1e-3
gamma = 0.995

episodes = 500

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(2, 64)
        self.batch1 = nn.LayerNorm(64)

        self.lin2 = nn.Linear(64, 128)
        self.batch2 = nn.LayerNorm(128)

        self.lin3 = nn.Linear(128, 256)
        self.batch3 = nn.LayerNorm(256)

        self.lin4 = nn.Linear(256, 128)  
        self.batch4 = nn.LayerNorm(128)

        self.lin5 = nn.Linear(128, 64)  
        self.batch5 = nn.LayerNorm(64)

        self.lin6 = nn.Linear(64, 4)  

        self.actions, self.states, self.rewards = [],[],[]
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.950
    
    def forward(self, x):
        # x = x.float()
        x = F.relu(self.batch1(self.lin1(x)))
        x = F.relu(self.batch2(self.lin2(x)))
        x = F.relu(self.batch3(self.lin3(x)))
        x = F.relu(self.batch4(self.lin4(x)))
        x = F.relu(self.batch5(self.lin5(x)))
        x = F.softmax(self.lin6(x), dim = -1)
        return x
    
    def policy_action(self, state):
        state = torch.FloatTensor([state])
        probability = self.forward(state)
        categories = Categorical(probability)
        act = categories.sample()
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
        return dc_reward

    def gradient_ascent(self, dc_reward):
        optim.zero_grad()
        for state, action, G in zip(self.states, self.actions, dc_reward):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state_tensor)
            m = torch.distributions.Categorical(probs)
            log_prob = m.log_prob(torch.tensor(action))

            loss = -log_prob * G
            loss.backward()
        optim.step()


train_rewards = []
recent_rewards = []
env = environment.game_environnement()
rl = NeuralNetwork()
optim = torch.optim.Adam(rl.parameters(), lr=lr)

for episode in range(episodes):
    state = env.reset()
    rl.actions, rl.states, rl.rewards = [], [], []

    for i in range(1000):
        if random.random() < rl.epsilon:
            action = random.randrange(4)
        else :
            action = rl.policy_action(state)

        new_state, reward , done = env.step(action)

        rl.remember(action, state, reward)

        state = new_state

        if done:
            break

    rl.epsilon = max(rl.epsilon_end, rl.epsilon_decay * rl.epsilon)
    rl.gradient_ascent(rl.discount_rewards())

    train_rewards.append(np.sum(rl.rewards))
    recent_rewards.append(train_rewards[-1])

    print(f"Episode {episode:>6}: \tR:{np.mean(recent_rewards):>6.3f}")

fig, ax = plt.subplots()

ax.plot(train_rewards)
ax.plot(gaussian_filter1d(train_rewards, sigma=20), linewidth=4)
ax.set_title('Rewards')

fig.show()

# # env = gym.make(env_name, render_mode="human")

# for _ in range(5):
#     Rewards = []
    
#     state = env.reset()
#     done = False
    
#     for _ in range(1000):
#         # Calculate the probabilities of taking each action using the trained
#         # neural network
#         state_tensor = torch.from_numpy(state).float()
#         probs = rl.forward(state_tensor)
        
#         # Sample an action from the resulting distribution using the 
#         # torch.distributions.Categorical() method
#         m = torch.distributions.Categorical(probs)
#         action = m.sample().item()  # Sample an action from the distribution
    
#         new_state, reward, done= env.step(action)
    
#         state = new_state

#         Rewards.append(reward)

#         if done:
#             break
    
#     # Print the total rewards for the current episode
#     print(f'Reward: {sum(Rewards)}')

# # Close the environment
env.close()