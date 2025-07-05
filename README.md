# 🐍 Snake Game AI using Deep Q-Learning (DQN)

This project implements a self-learning Snake game agent using Deep Q-Learning (DQN) built with Python and PyTorch. The agent learns to play Snake from scratch through rewards and punishments without any hardcoded rules.

![Snake Game](https://user-images.githubusercontent.com/your-image-url.gif)

## 🎮 Features

* Built with **Pygame** for real-time rendering
* Deep Q-Network (**DQN**) from scratch using **PyTorch**
* Reward system based on food, proximity, loop penalties, and survival
* Real-time score display and game visualization
* Tracks training performance (score, mean score)
* Supports saving metrics and model checkpoints

## 🧠 Reinforcement Learning Setup

* **State space**: 11-dimensional vector:

  * Danger in left, straight, right
  * Current direction (4 bits)
  * Food location relative to the head (4 bits)
* **Actions**: `[Left, Straight, Right]` relative to current direction
* **Rewards**:

  * +25 for eating food
  * -10 for dying (collision with wall or itself)
  * -5 for repeated loops / stalling
  * +1 for moving closer to food
  * -1 for moving away from food

## 📊 Training Metrics

* Average score over time
* High score (record)
* Stats saved to `stats.csv`
* Model saved as `model.pth`

## 🧪 Installation

```bash
git clone https://github.com/your-username/snake_game_rl.git
cd snake_game_rl
pip install -r requirements.txt
```

> Make sure Python ≥ 3.8 is installed

### `requirements.txt`

```txt
pygame
numpy
torch
matplotlib
pandas
```

## 🚀 Run the Game

* **Play manually** (optional, not default):

  ```bash
  python play.py
  ```

* **Train the agent**:

  ```bash
  python train.py
  ```

> Training can take some time depending on your hardware. Use `Ctrl+C` to stop and resume from saved model.

## 📁 File Structure

```
snake_game_rl/
├── snake_game.py        # Environment and UI logic
├── model.py             # Neural network (LinearQNet)
├── agent.py             # DQN Agent and memory
├── train.py             # Training loop
├── stats.csv            # Logs score, mean score per game
├── model.pth            # Saved PyTorch model
├── README.md
└── requirements.txt
```

## 📈 Example Training Plot

Scores and mean scores plotted during training.
![Training Plot](https://github.com/user-attachments/assets/b6a0f474-daf1-4cc4-b93c-513cb70a84a4)

## 📌 Future Improvements

* Switch to CNN-based state representation (vision)
* Use Prioritized Experience Replay
* Add curriculum learning with increasing difficulty
* Deploy as a web-based demo

## 🧑‍💻 Author

* [Anurag Pokhariyal](https://github.com/AnuragP004)

## 📝 License

This project is licensed under the MIT License.
