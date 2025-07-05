# Install dependencies (run this in terminal, not in script):
# pip install pygame numpy

import pygame
import random
import numpy as np

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

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [RIGHT, DOWN, LEFT, UP]  # Clockwise order

class SnakeGame:
    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 25)
        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.head = [self.w // 2, self.h // 2]
        self.snake = [self.head[:], [self.head[0] - BLOCK_SIZE, self.head[1]], [self.head[0] - 2 * BLOCK_SIZE, self.head[1]]]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.loop_history = []
        self.no_progress_counter = 0
        self.prev_food_dist = self._get_food_distance()

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = [x, y]
            if self.food not in self.snake:
                break

    def _get_food_distance(self):
        return abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

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

        # Looping penalty
        self.loop_history.append(tuple(self.head))
        if len(self.loop_history) > 50:
            self.loop_history.pop(0)
        if self.loop_history.count(tuple(self.head)) > 5:
            reward -= 5

        # Proximity penalty
        curr_dist = self._get_food_distance()
        if curr_dist < self.prev_food_dist:
            reward += 1
            self.no_progress_counter = 0
        else:
            reward -= 1
            self.no_progress_counter += 1
        self.prev_food_dist = curr_dist

        if self.no_progress_counter > 30:
            reward -= 5
            self.no_progress_counter = 0

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 25
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(15000)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        return pt in self.snake[1:] or pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h

    def _move(self, action):
        idx = DIRECTIONS.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = DIRECTIONS[(idx - 1) % 4]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = DIRECTIONS[idx]
        else:
            new_dir = DIRECTIONS[(idx + 1) % 4]
        self.direction = new_dir
        x = self.head[0] + self.direction[0] * BLOCK_SIZE
        y = self.head[1] + self.direction[1] * BLOCK_SIZE
        self.head = [x, y]

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [5, 5])
        pygame.display.flip()

    def get_state(self):
        head = self.snake[0]
        idx = DIRECTIONS.index(self.direction)

        point_l = [head[0] + DIRECTIONS[(idx - 1) % 4][0] * BLOCK_SIZE,
                   head[1] + DIRECTIONS[(idx - 1) % 4][1] * BLOCK_SIZE]
        point_r = [head[0] + DIRECTIONS[(idx + 1) % 4][0] * BLOCK_SIZE,
                   head[1] + DIRECTIONS[(idx + 1) % 4][1] * BLOCK_SIZE]
        point_s = [head[0] + self.direction[0] * BLOCK_SIZE,
                   head[1] + self.direction[1] * BLOCK_SIZE]

        state = [
            self._is_collision(point_l),
            self._is_collision(point_s),
            self._is_collision(point_r),
            self.direction == LEFT,
            self.direction == RIGHT,
            self.direction == UP,
            self.direction == DOWN,
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]   # food down
        ]
        return np.array(state, dtype=int)
