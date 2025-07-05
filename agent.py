# Agent class for Deep Q-Learning
import random
from collections import deque
import torch


MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class DQNAgent:
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.n_games = 0
        self.epsilon = 0  # Exploration rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Replay buffer

    def get_action(self, state):
        self.epsilon = 80 - self.n_games  # Decrease over time
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
