# import random
# import numpy as np
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size  # [accuracy, attempts]
#         self.action_size = action_size  # number of difficulty levels
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95    # discount rate
#         self.epsilon = 1.0   # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.001
#         self.model = self._build_model()

#     def _build_model(self):
#         model = Sequential()
#         model.add(Dense(24, input_dim=self.state_size, activation='relu'))
#         model.add(Dense(24, activation='relu'))
#         model.add(Dense(self.action_size, activation='linear'))
#         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
#         return model

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         act_values = self.model.predict(np.array([state]), verbose=0)
#         return np.argmax(act_values[0])

#     def remember(self, state, action, reward, next_state):
#         self.memory.append((state, action, reward, next_state))

#     def replay(self, batch_size=32):
#         minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
#         for state, action, reward, next_state in minibatch:
#             target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
#             target_f = self.model.predict(np.array([state]), verbose=0)
#             target_f[0][action] = target
#             self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.layers(x)

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=0.1):
        self.model = DQN(state_size, action_size)
        self.epsilon = epsilon
        self.action_size = action_size
        self.memory = []  # (state, action, reward, next_state)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # exploit

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def train(self, batch_size=32, gamma=0.9):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in samples:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward + gamma * torch.max(self.model(next_state_tensor)).item()
            predicted = self.model(state_tensor)[0][action]

            loss = self.criterion(predicted, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

