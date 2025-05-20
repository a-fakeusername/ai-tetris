import random
from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os
import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from queue import Queue

# --- Flask App Setup ---
app = Flask(__name__)
# In a real app, use a secret key from environment variables
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
# Enable CORS for your Vue frontend origin (adjust in production)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}) # Adjust port if needed
socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173")

# --- Game Constants ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
EMPTY_CELL = 0
# Score multiplier per line cleared at once
LINE_SCORES = {1: 40, 2: 100, 3: 300, 4: 1200}
POSSIBLE_ACTIONS = ['move_left', 'move_right', 'move_down', 'hard_drop', 'rotate', 'rotate_reverse', 'rotate_180', 'no_op']
PIECE_ORDER = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
ROTATIONS = ['no_op', 'rotate', 'rotate_180', 'rotate_reverse']
PIECE_INDEXES = {piece: i for i, piece in enumerate(PIECE_ORDER)}
SIMULATION_DELAY = .1 # Delay in seconds for bot simulation

# Hyperparamaters
TRAIN_STEPS = 500000
ENTROPY = .03
LEARNING_RATE = 3e-4
HIGH_COL_THRESHOLD = .5 # Height threshold for high columns
USE_CNN = False

SCORE_HISTORY = []
REWARD_HISTORY = []

# --- Tetromino Shapes ---
# Define shapes as matrices (0 = empty, 1 = block)
# Rotations are defined clockwise
TETROMINOES = {
    'I': {
        'shape': [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        'color': '#00FFFF' # Cyan
    },
    'O': {
        'shape': [[1, 1], [1, 1]],
        'color': '#FFFF00' # Yellow
    },
    'T': {
        'shape': [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
        'color': '#800080' # Purple
    },
    'S': {
        'shape': [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
        'color': '#008000' # Green
    },
    'Z': {
        'shape': [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
        'color': '#FF0000' # Red
    },
    'J': {
        'shape': [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
        'color': '#0000FF' # Blue
    },
    'L': {
        'shape': [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
        'color': '#FFA500' # Orange
    }
}

# Add rotated versions to each tetromino definition for easier lookup
def get_rotations(shape):
    """Generates all 4 rotations for a given shape matrix."""
    rotations = [shape]
    
    current_shape = shape
    for _ in range(3):
        rows, cols = len(current_shape), len(current_shape[0])
        new_shape = [[0] * rows for _ in range(cols)]
        for r in range(rows):
            for c in range(cols):
                new_shape[c][rows - 1 - r] = current_shape[r][c]
        if new_shape not in rotations: # For O piece duplicates only
            rotations.append(new_shape)
        current_shape = new_shape
    return rotations

for key in TETROMINOES:
    if ('rotations' not in TETROMINOES[key]):
        TETROMINOES[key]['rotations'] = get_rotations(TETROMINOES[key]['shape'])
    # Set initial shape to the first rotation
    TETROMINOES[key]['shape'] = TETROMINOES[key]['rotations'][0]


# --- Game State Management ---
# Store game state per client session ID (sid)
game_states = {}
# Lock to prevent race conditions when modifying game_states or individual states
state_locks = {}
# Store background task references per sid
background_tasks = {}

# --- Helper Functions ---
def create_empty_board():
    """Creates a new empty game board."""
    return [[EMPTY_CELL for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

def generate_7bag():
    """Generates a random sequence of tetrominoes in a 7-bag format."""
    pieces = list(TETROMINOES.keys())
    random.shuffle(pieces)
    return pieces

def create_new_piece(game):
    """Creates the next piece and sets its starting position."""
    # if (game.piece_queue.empty()):
        # Generate a new 7-bag if the queue is empty
        # new_pieces = generate_7bag()
        # for piece in new_pieces:
        #     game.piece_queue.put(piece)
    # piece_type = game.piece_queue.get()

    piece_type = random.choice(PIECE_ORDER) # Pure Random
    piece_data = TETROMINOES[piece_type]
    return {
        'type': piece_type,
        'rotations': piece_data['rotations'],
        'rotation_index': 0,
        'shape': piece_data['rotations'][0], # Current shape based on rotation
        'color': piece_data['color'],
        'x': BOARD_WIDTH // 2 - len(piece_data['rotations'][0][0]) // 2, # Start roughly centered
        'y':0 # Start at the top
    }

def is_valid_position(board, piece, offset_x=0, offset_y=0, rotation_offset=0):
    """Checks if a piece's potential position/rotation is valid."""
    new_x = piece['x'] + offset_x
    new_y = piece['y'] + offset_y
    new_rotation_index = (piece['rotation_index'] + rotation_offset) % len(piece['rotations'])
    shape_to_check = piece['rotations'][new_rotation_index]

    for r, row in enumerate(shape_to_check):
        for c, cell in enumerate(row):
            if cell: # If it's part of the piece shape
                board_y = new_y + r
                board_x = new_x + c

                # Check boundaries
                if not (0 <= board_x < BOARD_WIDTH and 0 <= board_y < BOARD_HEIGHT):
                    return False
                # Check collision with existing blocks on the board
                if board[board_y][board_x] != EMPTY_CELL:
                    return False
    return True

def freeze_piece(board, piece):
    """Locks the current piece onto the board."""
    shape = piece['shape']
    color_value = piece['color'] # Store color directly or map to an index if preferred
    for r, row in enumerate(shape):
        for c, cell in enumerate(row):
            if cell:
                board_y = piece['y'] + r
                board_x = piece['x'] + c
                # Prevent writing outside bounds (safety check)
                if 0 <= board_y < BOARD_HEIGHT and 0 <= board_x < BOARD_WIDTH:
                    board[board_y][board_x] = color_value # Use color or index
    return board

def clear_lines(board):
    """Checks for completed lines, clears them, and returns the number cleared."""
    lines_cleared = 0
    new_board = [row for row in board if any(cell == EMPTY_CELL for cell in row)]
    lines_cleared = BOARD_HEIGHT - len(new_board)

    # Add new empty lines at the top
    for _ in range(lines_cleared):
        new_board.insert(0, [EMPTY_CELL for _ in range(BOARD_WIDTH)])

    return new_board, lines_cleared

def calculate_speed(level):
    """Calculates game speed based on score. Returns delay in seconds."""
    # Example: Speed increases every 10 lines points, minimum delay 0.1s
    delay = max(0.1, 0.8 - level * 0.05)
    return delay

def count_holes(board):
    """Counts the number of holes in the board."""
    holes = 0
    for c in range(BOARD_WIDTH):
        has_block = False
        for r in range(BOARD_HEIGHT):
            if board[r][c] != EMPTY_CELL:
                has_block = True
            elif has_block:
                holes += 1
    return holes

def count_high_columns(board):
    """Counts the number of high columns in the board."""
    high_columns = 0
    for c in range(BOARD_WIDTH):
        height = 0
        for r in range(BOARD_HEIGHT):
            if board[r][c] != EMPTY_CELL:
                height = BOARD_HEIGHT - r
                break
        if height > HIGH_COL_THRESHOLD * BOARD_HEIGHT:
            high_columns += 1
    return high_columns

def count_uneven_height(board):
    heights = [0] * BOARD_WIDTH
    for c in range(BOARD_WIDTH):
        height = 0
        for r in range(BOARD_HEIGHT):
            if board[r][c] != EMPTY_CELL:
                height = BOARD_HEIGHT - r
                break
        heights[c] = height
    
    diffs = []
    for c in range(1, BOARD_WIDTH):
        diffs.append(abs(heights[c] - heights[c - 1]))
    
    diffs.sort()
    uneven_height = 0
    for i in range(2, len(diffs) - 2):
        uneven_height += diffs[i]
    return uneven_height


# --- Game Logic Class (Optional but good for structure) ---
# Alternatively, keep functions operating on the state dictionary directly
class TetrisGame(gym.Env):
    def __init__(self, sid=0, train=False, run_bot=False):
        super().__init__()
        self.sid = sid
        self.mode = 'player'
        self.run_bot = run_bot
        self.reset()
        board_space = gym.spaces.MultiBinary((BOARD_HEIGHT, BOARD_WIDTH)) # Binary representation of the board
        # piece = gym.spaces.Box(low=0, high=len(PIECE_ORDER), shape=(len(PIECE_ORDER),), dtype=np.int8) # One-hot encoding of piece type
        piece_space = gym.spaces.Box(low=0, high=max(BOARD_WIDTH, BOARD_HEIGHT), shape=(8,), dtype=np.int8) # 4 pairs of coordinates (x, y)
        height_space = gym.spaces.Box(low=0, high=BOARD_HEIGHT, shape=(BOARD_WIDTH,), dtype=np.int8) # Height of each column
        extra_space = gym.spaces.Box(low=0, high=BOARD_HEIGHT * BOARD_WIDTH, shape=(4,), dtype=np.float32) # Extra space for future use
        # Store 4 pairs of (x, y)
        self.observation_space = gym.spaces.Dict({
            'board': board_space,
            # 'piece': piece
            'piece': piece_space,
            'height': height_space,
            'extra': extra_space
        })
        # rotation, X position + 5 [-5, 4], slide pos + 2 [-2, 2], slide rotate
        self.action_space = gym.spaces.MultiDiscrete([4, 10, 5, 4])
        if not train:
            self.model = PPO.load("ppo_tetris_custom_net", env=self)

    def _get_obs(self):
        """Returns the current observation of the game."""
        # Convert board and piece to binary representation
        piece = self.current_piece
        board_obs = np.array([[1 if cell != EMPTY_CELL else 0 for cell in row] for row in self.board], dtype=np.int8)
        piece_obs = np.array([0] * 8, dtype=np.int8) # 4 pairs of coordinates (x, y) for piece
        height_obs = np.array([0] * BOARD_WIDTH, dtype=np.int8)
        if piece:
            counter = 0
            for i, row in enumerate(piece['shape']):
                for j, cell in enumerate(row):
                    if cell:
                        piece_obs[2 * counter] = j + piece['x']
                        piece_obs[2 * counter + 1] = i + piece['y']
                        counter += 1
        total_height = 0
        for c in range(BOARD_WIDTH):
            height = 0
            for r in range(BOARD_HEIGHT):
                if self.board[r][c] != EMPTY_CELL:
                    height = BOARD_HEIGHT - r
                    break
            height_obs[c] = height
            total_height += height

        return {
            'board': board_obs,
            # 'piece': [1 if piece and piece['type'] == p else 0 for p in PIECE_ORDER] # One-hot encoding of piece type
            'piece': piece_obs,
            'height': height_obs,
            'extra': np.array([count_holes(self.board), count_high_columns(self.board), count_uneven_height(self.board), total_height], dtype=np.float32) # Extra space for future use
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        """Resets the game state."""
        self.board = create_empty_board()
        self.score = 0
        self.total_reward = 0
        self.prev_board_reward = 0
        self.is_game_over = False
        self.game_active = True # Flag to stop the loop
        self.fall_delay = calculate_speed(0)
        self.lines_cleared = 0
        self.pieces = 0
        self.piece_queue = Queue()
        self.current_piece = create_new_piece(self)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def get_state(self):
        """Returns the current game state dictionary."""
        return {
            "board": self.board,
            # Send only necessary info about the current piece
            "current_piece": {
                "shape": self.current_piece['shape'],
                "x": self.current_piece['x'],
                "y": self.current_piece['y'],
                "color": self.current_piece['color']
            } if self.current_piece else None,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "is_game_over": self.is_game_over,
            "board_width": BOARD_WIDTH,
            "board_height": BOARD_HEIGHT
            # Optional: Add next_piece preview
        }

    def _get_info(self):
        """Returns additional game info."""
        return {
            'score': self.score,
            'lines_cleared': self.lines_cleared
        }

    def move(self, dx, dy):
        """Attempts to move the current piece."""
        if self.current_piece and is_valid_position(self.board, self.current_piece, offset_x=dx, offset_y=dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def rotate(self, amt = 1):
        """Attempts to rotate the current piece."""
        if self.current_piece and is_valid_position(self.board, self.current_piece, rotation_offset=amt):
            self.current_piece['rotation_index'] = (self.current_piece['rotation_index'] + amt) % len(self.current_piece['rotations'])
            self.current_piece['shape'] = self.current_piece['rotations'][self.current_piece['rotation_index']]
            return True, False
        return False

    def game_step(self):
        """Advances the game by one tick."""
        if self.is_game_over or not self.current_piece:
            return False, False # Game already ended or no piece

        # Try moving down
        if self.move(0, 1):
            return True, False # Piece moved down successfully
        else:
            # Piece couldn't move down, freeze it
            self.board = freeze_piece(self.board, self.current_piece)

            self.pieces += 1 # Increment piece count

            # Clear completed lines
            self.board, lines_cleared = clear_lines(self.board)
            if lines_cleared > 0:
                self.score += LINE_SCORES.get(lines_cleared, 0) * (self.lines_cleared // 10 + 1) # Add level bonus
                self.fall_delay = calculate_speed(self.lines_cleared // 10) # Update speed
                self.lines_cleared += lines_cleared # Track total lines cleared

            # Spawn a new piece
            self.current_piece = create_new_piece(self)

            # Check for game over (new piece overlaps immediately)
            if not is_valid_position(self.board, self.current_piece):
                self.is_game_over = True
                self.game_active = False # Stop the loop
                self.current_piece = None # No more falling piece
                # print(f"Game Over for SID: {self.sid}. Final Score: {self.score}")
                SCORE_HISTORY.append(self.score) # Store score history
                REWARD_HISTORY.append(self.total_reward) # Store reward history
                return False, True
            return True, True

    # Board heuristics can be added here
    def get_board_reward(self):
        hole_reward = 0
        density_reward = 0
        high_col_reward = 0
        uneven_height_reward = 0

        # Negative reward for holes
        hole_reward = -count_holes(self.board) / 3 # Normalize to a reasonable range

        # # Negative reward for too many taken cells
        # taken_cells = sum(1 for row in self.board for cell in row if cell != EMPTY_CELL)
        # if taken_cells > .4 * (BOARD_WIDTH * BOARD_HEIGHT):
        #     density_reward -= (taken_cells / (BOARD_WIDTH * BOARD_HEIGHT) - .4)

        # Negative reward for high columns
        high_col_reward = -count_high_columns(self.board) / 2
        

        # Negative reward for uneven heights
        uneven_height_reward = -max(count_uneven_height(self.board) - 5, 0) / 5

        reward = hole_reward + high_col_reward + uneven_height_reward
        # reward = 0
        # reward = hole_reward

        # print(f"Board Reward: {reward:.2f} (hole: {hole_reward:.2f}, density: {density_reward:.2f}, high_col: {high_col_reward:.2f}, uneven_height: {uneven_height_reward:.2f})") # Debugging

        delta_reward = reward - self.prev_board_reward
        self.prev_board_reward = reward
        if delta_reward > 0:
            delta_reward /= 2

        return delta_reward

    # Function that the model calls to train
    def step(self, action):
        reward = 0

        old_score = self.score
        old_lines = self.lines_cleared

        rot = action[0]
        x_pos = action[1] - 5
        slide_x = action[2] - 2
        slide_rot = action[3]

        drop_count = 0

        if self.do_action('move_down'):
            # Initial rotation and moving
            self.do_action(ROTATIONS[rot])
            for i in range(x_pos):
                self.do_action('move_right')
            for i in range(-x_pos):
                self.do_action('move_left')
            
            # Hard drop if no extra
            # if slide_x == 0 and slide_rot == 0:
            if True:
                self.do_action('hard_drop')
            else:
                while self.current_piece and is_valid_position(self.board, self.current_piece, offset_y=1):
                    self.do_action('move_down')
                    drop_count += 1

                for i in range(slide_x):
                    self.do_action('move_right')
                for i in range(-slide_x):
                    self.do_action('move_left')
                
                self.do_action(ROTATIONS[slide_rot])
                self.game_step()
                
                # for i in range(STEP_MOVES):
                #     old_piece_count = self.pieces
                #     self.do_action(POSSIBLE_ACTIONS[action[i]]) # Perform the action
                #     if self.pieces > old_piece_count:
                #         break
                #     success, frozen = self.game_step()

                #     if callback:
                #         # Call the callback function with the current state
                #         callback(self)

                #     if not success or frozen:
                #         break

        if self.is_game_over:
            reward = -10 # Large negative reward for game over
        else:
            # reward += max(self.score - max(300, old_score), 0) / 1000  # Reward for score increase
            reward += (self.lines_cleared - old_lines) * 5 # Reward for lines cleared
            reward += 1 # Small reward for each tick survived
            reward += self.get_board_reward() # Add board heuristics
        
        self.total_reward += reward

        socketio.emit('game_update', self.get_state(), room=self.sid) # Emit state update

        observation = self._get_obs()
        terminated = self.is_game_over
        truncated = False # Not used in this game
        info = self._get_info()
        return observation, reward, terminated, truncated, info
        

    def do_action(self, action: str) -> bool:
        if action == 'restart':
            # Restart the game
            self.reset()
            return True
        
        if self.run_bot:
            socketio.sleep(SIMULATION_DELAY)

        updated = False

        if not self.is_game_over and self.current_piece:
            # print(f"Received action '{action}' from SID: {sid}") # Debug
            if action == 'move_left':
                updated = self.move(-1, 0)
            elif action == 'move_right':
                updated = self.move(1, 0)
            elif action == 'move_down':
                # Move down attempts to step, potentially freezing piece
                updated = self.game_step() # Use step logic for consistency
                self.score += 1
            elif action == 'rotate':
                updated = self.rotate()
            elif action == 'hard_drop':
                amt = 0
                while self.move(0, 1):
                    amt += 1
                updated = self.game_step() # Final step to freeze and check lines/game over
                self.score += amt * 2
            elif action == 'rotate_reverse':
                updated = self.rotate(3)
            elif action == 'rotate_180':
                updated = self.rotate(2)
            updated = updated or self.is_game_over
        if updated:
            # Emit the updated game state
            socketio.emit('game_update', self.get_state(), room=self.sid)
        return updated
    
    def print(self):
        obs_data = self._get_obs()
        piece = self.current_piece
        print("Board State:")
        for i, row in enumerate(obs_data["board"]):
            row_str = ""
            for j, cell in enumerate(row):
                if piece['y'] <= i and piece['y'] + len(piece['shape']) > i and piece['x'] <= j and piece['x'] + len(piece['shape'][0]) > j:
                    # Check if the cell is part of the current piece
                    piece_row = i - piece['y']
                    piece_col = j - piece['x']
                    if piece['shape'][piece_row][piece_col] == 1:
                        row_str += "%"
                        continue
                if cell == 1:
                    row_str += "#"
                else:
                    row_str += "."
            print(row_str)
        print(f"Heights: {obs_data['height']}")
        print(f"Piece Locs: {obs_data['piece']}")

    def close(self):
        """Cleans up the game state."""
        self.game_active = False
        self.is_game_over = True
        self.current_piece = None # No more falling piece
        self.board = None # Clear board reference

def emit_game_update(game, sid):
    """Emit game update to the client."""
    socketio.emit('game_update', game.get_state(), room=sid)
    socketio.sleep(SIMULATION_DELAY) # Sleep briefly to avoid busy waiting

# --- Game Loop Task ---
def game_loop_task(sid):
    """Background task that runs the game loop for a specific client."""
    print(f"Starting game loop for SID: {sid}")
    while True:
        # Retrieve game state and lock safely
        with state_locks[sid]:
            game: TetrisGame = game_states.get(sid)
            if not game or not game.game_active:
                print(f"Stopping game loop for SID: {sid} (Game not active or not found)")
                break # Exit loop if game ended or state removed

            # If bot, then perform action
            if game.mode == 'bot':
                if not game.model:
                    print(f"Model not loaded for SID: {sid}, skipping action.")
                    socketio.sleep(SIMULATION_DELAY)
                    continue
                # Get the current observation
                obs = game._get_obs()
                # Predict action using the model
                action, _states = game.model.predict(obs, deterministic=True)
                # Perform the action
                game.step(action)
            else:
                # --- Perform Game Step ---
                game.game_step()
                fall_delay = game.fall_delay # Get current delay

            current_state = game.get_state()

        # Emit update outside the lock to avoid holding it during network I/O
        socketio.emit('game_update', current_state, room=sid)
        # print(f"Tick for {sid}, Delay: {fall_delay:.2f}s") # Debugging

        # Check again if game should continue before sleeping
        if not game.game_active:
            print(f"Waiting game loop for SID: {sid} after emitting final state.")
            while not game.game_active:
                socketio.sleep(0.1) # Sleep briefly to avoid busy waiting
            continue # Continue to next tick
        
        if game.mode == 'bot':
            socketio.sleep(SIMULATION_DELAY)
        else:
            # Wait for the next tick
            socketio.sleep(fall_delay) # Use socketio.sleep for compatibility with async modes

    print(f"Game loop task ended for SID: {sid}")
    # Clean up task reference
    if sid in background_tasks:
        del background_tasks[sid]


# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    sid = request.sid
    print(f"Client connected: {sid}")
    # Initialize game state for the new client
    with state_locks.setdefault(sid, Lock()): # Create lock if it doesn't exist
        if sid not in game_states:
            game_states[sid] = TetrisGame(sid)
            print(f"Initialized new game state for SID: {sid}")
        else:
            # Reconnection? Reset or resume? For simplicity, reset.
            print(f"Reconnected client {sid}, resetting game state.")
            game_states[sid] = TetrisGame(sid)

        initial_state = game_states[sid].get_state()

    # Join a room specific to this client
    join_room(sid)

    # Send the initial state
    emit('game_update', initial_state, room=sid)

    # Start the game loop in a background task if not already running
    if sid not in background_tasks or not background_tasks[sid]:
        background_tasks[sid] = socketio.start_background_task(game_loop_task, sid)
        print(f"Started background task for SID: {sid}")
    else:
        print(f"Background task already running for SID: {sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    sid = request.sid
    print(f"Client disconnected: {sid}")

    # Clean up game state and stop the loop
    if sid in state_locks:
        with state_locks[sid]:
            if sid in game_states:
                game_states[sid].game_active = False # Signal loop to stop
                del game_states[sid]
                print(f"Removed game state for SID: {sid}")
        # Remove lock after releasing it
        del state_locks[sid]

    # Leave the client's room (optional, happens automatically)
    leave_room(sid)

    # Background task should stop itself, but remove reference
    if sid in background_tasks:
        # Note: We don't explicitly kill the thread here,
        # it checks game.game_active to exit gracefully.
        print(f"Ensuring background task reference is cleared for SID: {sid}")
        del background_tasks[sid] # Task cleans itself up now

@socketio.on('player_action')
def handle_player_action(data):
    """Handles player input actions from the client."""
    sid = request.sid
    action = data.get('action')

    if not action:
        return

    updated = False
    # Get lock and game state
    if sid in state_locks:
        with state_locks[sid]:
            game: TetrisGame = game_states.get(sid)
            if not game:
                print(f"Game state not found for SID: {sid}, ignoring action.")
                return

            if action == 'change_mode':
                game.mode = data.get('mode')
                print(f"Changed game mode for SID: {sid} to {game.mode}")
                game.do_action('restart') # Restart game on mode change


            updated = game.do_action(action)
            if updated:
                socketio.emit('game_update', game.get_state(), room=sid)

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
            nn.Linear(8, 32), # 4 pairs of coordinates (x, y)
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
def train(env: TetrisGame):
    # Use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model.save("ppo_tetris_custom_net")

    display_stat_history()

def simulate(env: TetrisGame):
    loaded_model = PPO.load("ppo_tetris_custom_net", env=env)
    obs, info = env.reset()
    print("\nInitial Observation:")
    env.print() # Using our basic render

    # Simulate a few steps
    for i in range(50):
        action, _states = loaded_model.predict(obs, deterministic=True)
        # action = env.action_space.sample() # Random action
        print(f"Step {i+1}, Piece: {env.current_piece['type']}\nAction: {f"{action[0]}, {action[1] - 5}, {action[2] - 2}, {action[3]}"}")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Game Over or Truncated!")
            break
        env.print()
        print(f"Reward: {reward} (board: {env.prev_board_reward})")

# --- Main Execution ---
if __name__ == '__main__':
    args = sys.argv[1:]

    env = TetrisGame(train=('train' in args), run_bot=(len(args) == 0))

    if 'train' in args:
        train(env)

    if 'simulate' in args:
        simulate(env)

    if len(args) == 0:
        print("Starting Flask-SocketIO server...")
        # Use host='0.0.0.0' to make it accessible on your network
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)

    env.close()
    