import random
import sys

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

from queue import Queue

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

HIGH_COL_THRESHOLD = .3 # Height threshold for high columns

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

def count_high_column_score(board):
    """Calculates a score based on the height of high columns."""
    score = 0
    for c in range(BOARD_WIDTH):
        height = 0
        for r in range(BOARD_HEIGHT):
            if board[r][c] != EMPTY_CELL:
                height = BOARD_HEIGHT - r
                break
        if height > HIGH_COL_THRESHOLD * BOARD_HEIGHT:
            score += (height - (HIGH_COL_THRESHOLD * BOARD_HEIGHT)) / (BOARD_HEIGHT - (HIGH_COL_THRESHOLD * BOARD_HEIGHT))
    return score

def count_uneven_height(board):
    """Calculates the uneven height score based on column heights."""
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
    def __init__(self, sid=0, train=False, model_file="ppo_tetris_custom_net"):
        super().__init__()
        self.sid = sid
        self.mode = 'player'
        self.reset()
        # Heights, piece, extra 4
        self.observation_space = gym.spaces.Box(low=0, high=BOARD_HEIGHT * BOARD_WIDTH, shape=(15,), dtype=np.float32);
        # rotation, X position + 5 [-5, 4], slide pos + 2 [-2, 2], slide rotate
        self.action_space = gym.spaces.MultiDiscrete([4, 10])
        if not train:
            self.model = PPO.load(model_file, env=self)

    def _get_obs(self):
        """Returns the current observation of the game."""
        # Convert board and piece to binary representation
        piece = PIECE_INDEXES[self.current_piece['type']] if self.current_piece else len(PIECE_ORDER)
        height_obs = [0] * BOARD_WIDTH
        # One-hot encoding of piece type
        total_height = 0
        for c in range(BOARD_WIDTH):
            height = 0
            for r in range(BOARD_HEIGHT):
                if self.board[r][c] != EMPTY_CELL:
                    height = BOARD_HEIGHT - r
                    break
            height_obs[c] = height
            total_height += height

        obs: list[float] = height_obs
        obs.extend([piece, count_holes(self.board), count_high_column_score(self.board), count_uneven_height(self.board), total_height])
        return obs
    
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
        hole_reward = -count_holes(self.board) / 2 # Normalize to a reasonable range

        # # Negative reward for too many taken cells
        # taken_cells = sum(1 for row in self.board for cell in row if cell != EMPTY_CELL)
        # if taken_cells > .4 * (BOARD_WIDTH * BOARD_HEIGHT):
        #     density_reward -= (taken_cells / (BOARD_WIDTH * BOARD_HEIGHT) - .4)

        # Negative reward for high columns
        high_col_reward = -count_high_column_score(self.board)
        

        # Negative reward for uneven heights
        uneven_height_reward = -max(count_uneven_height(self.board) - 8, 0) / 4

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
    def step(self, action, callback=None):
        reward = 0

        old_score = self.score
        old_lines = self.lines_cleared

        rot = action[0]
        x_pos = action[1] - 5
        # slide_x = action[2] - 2
        # slide_rot = action[3]

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
            self.do_action('hard_drop')

                # while self.current_piece and is_valid_position(self.board, self.current_piece, offset_y=1):
                #     self.do_action('move_down')
                #     drop_count += 1

                # for i in range(slide_x):
                #     self.do_action('move_right')
                # for i in range(-slide_x):
                #     self.do_action('move_left')
                
                # self.do_action(ROTATIONS[slide_rot])
                # self.game_step()
                
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
            reward = -20 # Large negative reward for game over
        else:
            # reward += max(self.score - max(300, old_score), 0) / 1000  # Reward for score increase
            reward += (self.lines_cleared - old_lines) * 10 # Reward for lines cleared
            reward += 1 # Small reward for each tick survived
            reward += self.get_board_reward() # Add board heuristics
            if (self.pieces * 4 >= BOARD_HEIGHT * BOARD_WIDTH):
                reward += 3 # Extra reward for placing pieces after filling board
        self.total_reward += reward

        if callback:
            # Call the callback function with the current state
            callback(self)

        observation = self._get_obs()
        terminated = self.is_game_over
        truncated = False # Not used in this game
        info = self._get_info()
        return observation, reward, terminated, truncated, info
        

    def do_action(self, action: str, callback=None) -> bool:
        if action == 'restart':
            # Restart the game
            self.reset()
            return True

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
        if updated and callback:
            callback(self)
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

def simulate(env: TetrisGame, model_file = None):
    model_file = model_file or "ppo_tetris_custom_net"
    loaded_model = PPO.load(model_file, env=env)
    obs, info = env.reset()
    print("\nInitial Observation:")
    env.print() # Using our basic render

    # Simulate a few steps
    for i in range(50):
        action, _states = loaded_model.predict(obs, deterministic=True)
        # action = env.action_space.sample() # Random action
        print(f"Step {i+1}, Piece: {env.current_piece['type']}\nAction: {f"{action[0]}, {action[1] - 5}"}")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Game Over or Truncated!")
            break
        env.print()
        print(f"Reward: {reward} (board: {env.prev_board_reward})")

# --- Main Execution ---
if __name__ == '__main__':
    args = sys.argv[1:]

    env = TetrisGame(model_file=args[1] if len(args) > 1 else None)

    simulate(env, args[1] if len(args) > 1 else None)
    env.close()
    