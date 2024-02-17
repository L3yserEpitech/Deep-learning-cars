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
optim = torch.optim.Adam(network.parameters(), lr=lr)

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

    def gradient_ascent(self, discounted_rewards):
        # Perform gradient ascent to update the probabilities in the distribution
        optim.zero_grad()  # Clear the gradients before backward pass
        for state, action, G in zip(self.states, self.actions, discounted_rewards):
            # Convert the state to a PyTorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get the probabilities for the current state
            probs = self.forward(state_tensor)
            
            # Create a distribution according to the probabilities
            m = torch.distributions.Categorical(probs)
            
            # Calculate the negative log probability of the chosen action
            log_prob = m.log_prob(torch.tensor(action))
            
            # Calculate the loss as the negative log probability of the chosen action
            # multiplied by the discounted return
            loss = -log_prob * G
            
            # Perform backpropagation
            loss.backward()
        
        # Update the network parameters
        optim.step()

rl = NeuralNetwork()


# Iterate over the number of episodes
for episode in range(episodes):
    # Reset the environment and initialize empty lists for actions, states, and rewards
    state, _  = env.reset()
    network.actions, network.states, network.rewards = [], [], []

    # Train the agent for a single episode
    for _ in range(1000):
        action = network.policy_action(state)

        # Take the action in the environment and get the new state, reward, and done flag
        new_state, reward, termination, truncation, _ = env.step(action)

        # Save the action, state, and reward for later
        network.remember(action, state, reward)

        state = new_state

        # If the episode is done or the time limit is reached, stop training
        if termination or truncation:
            break

    # Perform gradient ascent
    network.gradient_ascent(network.discount_rewards())

    # Save the total reward for the episode and append it to the recent rewards queue
    train_rewards.append(np.sum(network.rewards))
    recent_rewards.append(train_rewards[-1])

    # Print the mean recent reward every 50 episodes
    if episode % 50 == 0:
        print(f"Episode {episode:>6}: \tR:{np.mean(recent_rewards):>6.3f}")

    # if np.mean(recent_rewards) > 400:
    #     break

fig, ax = plt.subplots()

ax.plot(train_rewards)
ax.plot(gaussian_filter1d(train_rewards, sigma=20), linewidth=4)
ax.set_title('Rewards')

fig.show()

env = gym.make(env_name, render_mode="human")

for _ in range(5):
    Rewards = []
    
    state, _ = env.reset()
    done = False
    
    for _ in range(1000):
        # Calculate the probabilities of taking each action using the trained
        # neural network
        state_tensor = torch.from_numpy(state).float()
        probs = network.forward(state_tensor)
        
        # Sample an action from the resulting distribution using the 
        # torch.distributions.Categorical() method
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()  # Sample an action from the distribution
    
        new_state, reward, termination, truncation, _ = env.step(action)
    
        state = new_state

        Rewards.append(reward)

        if termination or truncation:
            break
    
    # Print the total rewards for the current episode
    print(f'Reward: {sum(Rewards)}')

# Close the environment
env.close()