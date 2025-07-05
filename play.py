import torch
import numpy as np
from snake_game import SnakeGame  # or snake_game_rl_env if that’s your file
from model import LinearQNet

model_path = "model.pth"

class AgentPlay:
    def __init__(self):
        self.model = LinearQNet(11, 256, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # inference mode

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state_tensor)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

if __name__ == "__main__":
    game = SnakeGame()
    agent = AgentPlay()

    while True:
        state = game.get_state()
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)

        if done:
            print("Game Over. Score:", score)
            game.reset()
