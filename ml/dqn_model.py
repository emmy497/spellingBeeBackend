# dqn_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.2  # 20% random exploration

    def get_action(self, state_vector):
        state = torch.FloatTensor(state_vector).unsqueeze(0)
        if random.random() < self.epsilon:
            # Occasionally show mastered words
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()

    def train(self, state_vector, action, reward, next_state_vector):
        state = torch.FloatTensor(state_vector).unsqueeze(0)
        next_state = torch.FloatTensor(next_state_vector)._
