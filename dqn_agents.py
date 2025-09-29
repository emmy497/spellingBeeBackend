# dqn_agents.py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import os
# import pickle

# # ----------------------
# # Simple DQN Model
# # ----------------------
# class DQN(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_size)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # ----------------------
# # DQN Agent
# # ----------------------
# class DQNAgent:
#     def __init__(self, state_size, action_size, user_id, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.user_id = user_id

#         self.memory = []
#         self.model = DQN(state_size, action_size)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.criterion = nn.MSELoss()

#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay

#     def act(self, state):
#         """ Choose next word index """
#         if np.random.rand() <= self.epsilon:
#             # Explore: pick random action
#             return random.randrange(self.action_size)
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         q_values = self.model(state_tensor)
#         return torch.argmax(q_values).item()

#     def remember(self, state, action, reward, next_state):
#         self.memory.append((state, action, reward, next_state))

#     def replay(self, batch_size=32):
#         if len(self.memory) < batch_size:
#             return
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state in minibatch:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

#             target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
#             current = self.model(state_tensor)[0][action]

#             loss = self.criterion(current, torch.tensor(target))
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     def save(self, path):
#         with open(path, "wb") as f:
#             pickle.dump(self, f)

#     @staticmethod
#     def load(path):
#         with open(path, "rb") as f:
#             return pickle.load(f)


import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models

class DQNAgent:
    def __init__(self, state_size, action_size, user_id, gamma=0.95, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.user_id = user_id

        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return model

    def act(self, state):
        state = np.array(state).reshape(1, -1)

        # Exploration vs Exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state, verbose=0)[0]

        # ---  Bias reinforcement ---
        # Lower Q-values for mastered words (1), higher Q-values for unmastered (0)
        for i, val in enumerate(state[0]):
            if val == 1:  # mastered word
                q_values[i] -= 1.0  # discourage
            else:  # unmastered word
                q_values[i] += 1.0  # encourage

        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, -1)
            next_state = np.array(next_state).reshape(1, -1)

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
