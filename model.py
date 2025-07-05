import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize pygame
pygame.init()

# Constants
BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [RIGHT, DOWN, LEFT, UP]  # Clockwise order for turning

class SnakeGame:
    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.head = [self.w // 2, self.h // 2]
        self.snake = [self.head[:], [self.head[0] - BLOCK_SIZE, self.head[1]], [self.head[0] - 2 * BLOCK_SIZE, self.head[1]]]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = [x, y]
            if self.food not in self.snake:
                break

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head[:])

        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(15)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:] or pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        return False

    def _move(self, action):
        idx = DIRECTIONS.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = DIRECTIONS[(idx - 1) % 4]  # left turn
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = DIRECTIONS[idx]  # straight
        else:  # [0, 0, 1]
            new_dir = DIRECTIONS[(idx + 1) % 4]  # right turn

        self.direction = new_dir
        x = self.head[0] + self.direction[0] * BLOCK_SIZE
        y = self.head[1] + self.direction[1] * BLOCK_SIZE
        self.head = [x, y]

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

    def get_state(self):
        head = self.snake[0]
        point_l = [head[0] + DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % 4][0] * BLOCK_SIZE,
                   head[1] + DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % 4][1] * BLOCK_SIZE]
        point_r = [head[0] + DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % 4][0] * BLOCK_SIZE,
                   head[1] + DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % 4][1] * BLOCK_SIZE]
        point_s = [head[0] + self.direction[0] * BLOCK_SIZE, head[1] + self.direction[1] * BLOCK_SIZE]

        danger_l = self._is_collision(point_l)
        danger_r = self._is_collision(point_r)
        danger_s = self._is_collision(point_s)

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        food_l = self.food[0] < head[0]
        food_r = self.food[0] > head[0]
        food_u = self.food[1] < head[1]
        food_d = self.food[1] > head[1]

        state = [
            danger_l,
            danger_s,
            danger_r,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_l,
            food_r,
            food_u,
            food_d
        ]
        return np.array(state, dtype=int)

# Q-network (model.py)
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # Reshape to (1, input_size)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
