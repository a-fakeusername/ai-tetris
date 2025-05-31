from threading import Lock
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import os
import sys
from tetris_game import TetrisGame, NUM_WEIGHTS
from train import output_to_action
from stable_baselines3 import PPO
import neat
import pickle

# --- Flask App Setup ---
app = Flask(__name__)
# In a real app, use a secret key from environment variables
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
# Enable CORS for your Vue frontend origin (adjust in production)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}) # Adjust port if needed
socketio = SocketIO(app, cors_allowed_origins="http://localhost:5173")

model_file = "ppo_tetris_custom_net"
genome_file = "best_genome.pkl"

CONFIG_PATH = "config-feedforward.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    CONFIG_PATH)

SIMULATION_DELAY = .1 # Delay in seconds for bot simulation

# --- Game State Management ---
# Store game state per client session ID (sid)
game_states: dict[int, TetrisGame] = {}
rl_models: dict[int, PPO] = {}
neat_models: dict[int, neat.nn.FeedForwardNetwork] = {}
weights = None
# Lock to prevent race conditions when modifying game_states or individual states
state_locks: dict[int, Lock] = {}
# Store background task references per sid
background_tasks = {}


def emit_game_update(game: TetrisGame, sid):
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
            game: TetrisGame = game_states[sid]
            if not game or not game.game_active:
                print(f"Stopping game loop for SID: {sid} (Game not active or not found)")
                break # Exit loop if game ended or state removed

            # If bot, then perform action
            if game.mode == 'bot':
                # Get the current observation
                # obs = game._get_obs()

                # Predict action using the model
                # action, _states = rl_models[sid].predict(obs, deterministic=True)
                # action = output_to_action(neat_models[sid].activate(obs))
                action = game.heuristic_move()
                
                # Perform the action
                game.step(action, callback=socketio.emit('game_update', game.get_state(), room=game.sid))
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
        print(weights)
        if sid not in game_states:
            game_states[sid] = TetrisGame(sid, weights=weights)
            # rl_models[sid] = PPO.load(model_file, env=game_states[sid])
            # with open(genome_file, 'rb') as input_file:
            #     loaded_genome = pickle.load(input_file)
                # neat_models[sid] = neat.nn.FeedForwardNetwork.create(loaded_genome, config)
            print(f"Initialized new game state for SID: {sid}")
        else:
            # Reconnection? Reset or resume? For simplicity, reset.
            print(f"Reconnected client {sid}, resetting game state.")
            game_states[sid] = TetrisGame(sid, weights=weights)

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
                # del rl_models[sid]
                # del neat_models[sid]
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


            updated = game.do_action(action, callback=socketio.emit('game_update', game.get_state(), room=sid))
            if updated:
                socketio.emit('game_update', game.get_state(), room=sid)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        model_file = args[0]
    with open("best_weights.txt") as weights_file:
        weights = list(map(float, weights_file.read().strip().split()))
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)