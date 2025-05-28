from tetris_game import TetrisGame, SCORE_HISTORY, REWARD_HISTORY, PIECE_ORDER, BOARD_WIDTH, BOARD_HEIGHT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sklearn.linear_model import LinearRegression
import gymnasium as gym
import sys

# Hyperparamaters
TRAIN_STEPS = 5000000
ENTROPY = .02
LEARNING_RATE = 2e-4
USE_CNN = False

def display_stat_history():
    score_data = pd.Series(SCORE_HISTORY, dtype=int, name='Score')
    reward_data = pd.Series(REWARD_HISTORY, dtype=float, name='Reward')
    print(score_data.describe())
    print(reward_data.describe())
    
    plt.figure(figsize=(10, 5))
    plt.scatter(score_data.index, score_data.values, alpha=0.5)
    plt.title("Score History")

    # Reshape data for sklearn
    X = np.array(range(len(score_data))).reshape(-1, 1)
    y = score_data.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    y_pred = model.predict(X)
    plt.plot(score_data.index, y_pred, color='red', label='Regression Line')
    plt.legend()
    plt.show()

    # --------------------

    plt.figure(figsize=(10, 5))
    plt.scatter(reward_data.index, reward_data.values, alpha=0.5)
    plt.title("Reward History")

    # Reshape data for sklearn
    X = np.array(range(len(reward_data))).reshape(-1, 1)
    y = reward_data.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    y_pred = model.predict(X)
    plt.plot(reward_data.index, y_pred, color='red', label='Regression Line')
    plt.legend()
    plt.show()

# Trains and saves the model
def train(env: TetrisGame, model_file = None, output_file = "ppo_tetris_custom_net"):
    # Use gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    if model_file:
        model = PPO.load(model_file, env=env)
        print("Model loaded from file", model_file)
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 512], vf=[256, 256]),
            activation_fn=nn.ReLU
        )
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, ent_coef=ENTROPY, learning_rate=LEARNING_RATE, gamma=.995, verbose=1, device=device, n_steps=10240, batch_size=512)
    # tensorboard_log="./ppo_tetris_tensorboard/", if want logging

    print("Training Started")

    # Train the model
    model.learn(total_timesteps=TRAIN_STEPS)

    # Save the model
    model.save(output_file)

    display_stat_history()

if __name__ == "__main__":
    # Create the environment
    env = TetrisGame(train=True)
    
    args = sys.argv[1:]

    # Train the model
    train(env, model_file=(args[0] if len(args) >= 1 else None), output_file=(args[1] if len(args) >= 2 else None))
    
    # Close the environment
    env.close()