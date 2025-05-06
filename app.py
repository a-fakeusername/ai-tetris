import random
from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os

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
    if (game.piece_queue.empty()):
        # Generate a new 7-bag if the queue is empty
        new_pieces = generate_7bag()
        for piece in new_pieces:
            game.piece_queue.put(piece)
    
    piece_type = game.piece_queue.get()
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

# --- Game Logic Class (Optional but good for structure) ---
# Alternatively, keep functions operating on the state dictionary directly
class TetrisGame:
    def __init__(self, sid):
        self.sid = sid
        self.board = create_empty_board()
        self.score = 0
        self.is_game_over = False
        self.game_active = True # Flag to stop the loop
        self.fall_delay = calculate_speed(0)
        self.lines_cleared = 0
        self.piece_queue = Queue()
        self.current_piece = create_new_piece(self)

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
            return True
        return False

    def step(self):
        """Advances the game by one tick."""
        if self.is_game_over or not self.current_piece:
            return False # Game already ended or no piece

        # Try moving down
        if self.move(0, 1):
            return True # Piece moved down successfully
        else:
            # Piece couldn't move down, freeze it
            self.board = freeze_piece(self.board, self.current_piece)

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
                print(f"Game Over for SID: {self.sid}. Final Score: {self.score}")
                return False
            return True

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

            # --- Perform Game Step ---
            game.step()
            current_state = game.get_state()
            fall_delay = game.fall_delay # Get current delay

        # Emit update outside the lock to avoid holding it during network I/O
        socketio.emit('game_update', current_state, room=sid)
        # print(f"Tick for {sid}, Delay: {fall_delay:.2f}s") # Debugging

        # Check again if game should continue before sleeping
        if not game.game_active:
            print(f"Stopping game loop for SID: {sid} after emitting final state.")
            break

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
        # del background_tasks[sid] # Task cleans itself up now

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
            if game and not game.is_game_over and game.current_piece:
                # print(f"Received action '{action}' from SID: {sid}") # Debug
                if action == 'move_left':
                    updated = game.move(-1, 0)
                elif action == 'move_right':
                    updated = game.move(1, 0)
                elif action == 'move_down':
                    # Move down attempts to step, potentially freezing piece
                    updated = game.step() # Use step logic for consistency
                    game.score += 1
                elif action == 'rotate':
                    updated = game.rotate()
                elif action == 'hard_drop':
                    amt = 0
                    while game.move(0, 1):
                        amt += 1
                    updated = game.step() # Final step to freeze and check lines/game over
                    game.score += amt * 2
                elif action == 'rotate_reverse':
                    updated = game.rotate(3)
                elif action == 'rotate_180':
                    updated = game.rotate(2)

                # If the action resulted in a state change, emit update
                if updated or game.is_game_over:
                    current_state = game.get_state()
                    # Emit update outside the lock? No, emit immediately after valid action
                    socketio.emit('game_update', current_state, room=sid)
                # else:
                    # print(f"Action '{action}' was invalid or had no effect for SID: {sid}")

            # else:
                # print(f"Action '{action}' ignored for SID: {sid} (Game Over or No Piece)")


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # Use host='0.0.0.0' to make it accessible on your network
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)