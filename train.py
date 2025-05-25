from tetris_game import TetrisGame, SCORE_HISTORY, REWARD_HISTORY, PIECE_ORDER, BOARD_WIDTH, BOARD_HEIGHT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sklearn.linear_model import LinearRegression
import gymnasium as gym
import sys

# Hyperparamaters
TRAIN_STEPS = 1000000
ENTROPY = .02
LEARNING_RATE = 2e-4
USE_CNN = False


# --- Custom Feature Extractor ---
class CustomTetrisFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Tetris.
    It uses a CNN for the board and an MLP for the piece coordinates.
    The outputs are then concatenated.
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 64, mlp_output_dim: int = 32):
        super().__init__(observation_space, features_dim=cnn_output_dim + mlp_output_dim + mlp_output_dim)
        
        # Get device once and use consistently
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        extractors = {}

        # --- CNN for the board ---
        # The board is (BOARD_HEIGHT, BOARD_WIDTH)
        # We need to add a channel dimension for the CNN: (1, BOARD_HEIGHT, BOARD_WIDTH)
        board_shape = observation_space["board"].shape
        cnn_input_shape = (1, board_shape[0], board_shape[1]) # (channels, height, width)

        # Define a simple CNN
        # Kernel sizes and strides might need tuning based on BOARD_HEIGHT/WIDTH
        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(2,1), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1,2), padding=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(self.device)

        # Calculate the flattened size after CNN
        # Pass a dummy tensor to determine the output shape
        with torch.no_grad():
            dummy_board_input = torch.as_tensor(observation_space["board"].sample()[None]).float().to(self.device)
            # Add channel dimension: (batch_size, height, width) -> (batch_size, 1, height, width)
            dummy_board_input = dummy_board_input.unsqueeze(1)
            cnn_flattened_size = self.board_cnn(dummy_board_input).shape[1]

        # Define a linear layer to get to the desired cnn_output_dim
        self.cnn_linear_output = nn.Linear(cnn_flattened_size, cnn_output_dim).to(self.device)

        # Linear variant
        self.board_mlp = nn.Sequential(
            nn.Linear(board_shape[0] * board_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, cnn_output_dim),
        ).to(self.device)

        if USE_CNN:
            extractors["board"] = lambda x: self.cnn_linear_output(self.board_cnn(x.unsqueeze(1).float().to(self.device)))
        else:
            extractors["board"] = lambda x: self.board_mlp(x.float().view(x.shape[0], -1).to(self.device))


        # --- MLP for piece coordinates ---
        self.piece_mlp = nn.Sequential(
            nn.Linear(len(PIECE_ORDER), 32), # 4 pairs of coordinates (x, y)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, mlp_output_dim),
            nn.ReLU()
        ).to(self.device)
        
        extractors["piece"] = lambda x: self.piece_mlp(x.float().to(self.device))

        # --- Height processing (optional) ---
        self.height_mlp = nn.Sequential(
            nn.Linear(BOARD_WIDTH, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim // 2),
            nn.ReLU()
        ).to(self.device)

        extractors["height"] = lambda x: self.height_mlp(x.float().to(self.device))

        # --- Extra processing (optional) ---
        self.extra_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim // 2),
            nn.ReLU()
        ).to(self.device)

        extractors["extra"] = lambda x: self.extra_mlp(x.float().to(self.device))

        self._features_dim = cnn_output_dim + mlp_output_dim + mlp_output_dim # Concatenate all features

    def forward(self, observations: gym.spaces.Dict) -> torch.Tensor:
        board_obs = observations["board"].to(self.device)
        piece_obs = observations["piece"].to(self.device)
        height_obs = observations["height"].to(self.device)

        # Board processing
        if USE_CNN:
            board_features = board_obs.unsqueeze(1).float() # Ensure float and add channel
            board_features = self.cnn_linear_output(self.board_cnn(board_features))
        else:
            board_features = self.board_mlp(board_obs.float().view(board_obs.shape[0], -1))

        # Piece MLP processing
        piece_features = self.piece_mlp(piece_obs.float())

        # Height processing (optional, can be added to board_features)
        height_features = self.height_mlp(height_obs.float())

        # Extra processing (optional, can be added to board_features)
        extra_features = self.extra_mlp(observations["extra"].float())

        # Concatenate the features
        concatenated_features = torch.cat((board_features, piece_features, height_features, extra_features), dim=1)
        return concatenated_features

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
def train(env: TetrisGame, model_file = None, output_file = None):
    # Use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_file:
        model = PPO.load(model_file, env=env)
        print("Model loaded from file", model_file)
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomTetrisFeatureExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128, mlp_output_dim=128),
            net_arch=dict(pi=[512, 512], vf=[256, 256]),
            activation_fn=nn.ReLU
        )
        model = PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, ent_coef=ENTROPY, learning_rate=LEARNING_RATE, gamma=.995, verbose=1, device=device, n_steps=2048, batch_size=512)
    # tensorboard_log="./ppo_tetris_tensorboard/", if want logging

    print("Training Started")

    # Train the model
    model.learn(total_timesteps=TRAIN_STEPS)

    # Save the model
    output_file = output_file or "ppo_tetris_custom_net"
    model.save(output_file)

    display_stat_history()

if __name__ == "__main__":
    # Create the environment
    env = TetrisGame(train=True)
    
    args = sys.argv[1:]

    # Train the model
    train(env, model_file=args[1] if len(args) > 1 else None, output_file="ppo_tetris_custom_net")
    
    # Close the environment
    env.close()