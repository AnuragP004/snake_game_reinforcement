import torch
import matplotlib.pyplot as plt
import csv
import os
from snake_game import SnakeGame
from model import LinearQNet, QTrainer
from agent import DQNAgent

# Plotting helper
def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.1)

# Save training stats to CSV
def save_stats_csv(game_num, score, mean_score, record, filename="train_stats.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Game", "Score", "Mean Score", "Record"])
        writer.writerow([game_num, score, mean_score, record])

# Main training loop
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = SnakeGame()
    model = LinearQNet(11, 256, 3)
    trainer = QTrainer(model=model, lr=0.001, gamma=0.9)
    agent = DQNAgent(model=model, trainer=trainer)

    while True:
        state_old = game.get_state()

        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                torch.save(model.state_dict(), 'model.pth')

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            save_stats_csv(agent.n_games, score, mean_score, record)

if __name__ == '__main__':
    plt.ion()
    train()
