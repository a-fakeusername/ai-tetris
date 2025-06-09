from tetris_game import TetrisGame, SCORE_HISTORY, REWARD_HISTORY, PIECE_ORDER, BOARD_WIDTH, BOARD_HEIGHT, NUM_WEIGHTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sklearn.linear_model import LinearRegression
import gymnasium as gym
import os
from dotenv import load_dotenv
import sys
import neat
import pickle
import random
import pygad

# RL Hyperparamaters
TRAIN_STEPS = 20000000
ENTROPY = .02
LEARNING_RATE = 2e-4

# NEAT Hyperparameters
GENERATIONS = 200

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

def output_to_action(output: list[float]):
    # id = np.argmax(output)
    # action = [id // 10, id % 10]
    rot = max(0, min(3, int(output[0])))
    pos = max(0, min(9, int(output[1])))
    action = [rot, pos]
    return action

# -------------------- Reinforcement Learning (using stable_baselines3) --------------------

# Trains and saves the model
def run_rl(env: TetrisGame, model_file = None, output_file = "ppo_tetris_custom_net"):
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

# -------------------- Evolutionary Neural Network (using NEAT) --------------------

rng = random.Random()
gen = 0
best_genome = {}
best_fitness = -1e9
def eval_genomes(genomes, config):
    """
    Evaluates the fitness of each genome in the `genomes` list.
    `genomes` is a list of (genome_id, genome) tuples.
    `config` is the NEAT configuration object.
    """
    global gen
    gen += 1
    global rng
    seed = rng.randrange(0, 1000000000)
    global best_genome
    global best_fitness
    fitnesses = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0  # Start with fitness 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = TetrisGame(seed=seed)

        # --- Run multiple episodes for more stable fitness ---
        num_episodes = 3 # Average fitness over a few episodes
        total_episode_reward = 0

        for _ in range(num_episodes):
            observation, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            max_steps_per_episode = 500 # Adjust as needed for the environment

            for _ in range(max_steps_per_episode):
                inputs = observation

                # --- Get network output ---
                output = net.activate(inputs)

                action = output_to_action(output)

                # --- Step the environment ---
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break
            total_episode_reward += episode_reward
            SCORE_HISTORY.append(env.score)
            REWARD_HISTORY.append(episode_reward)

        genome.fitness = total_episode_reward / num_episodes
        
        if genome.fitness > best_fitness:
            best_genome = genome
        fitnesses.append(genome.fitness)
        env.close()
    
    data = pd.DataFrame(fitnesses)
    print("Finished evaluating generation", gen)
    print(data.describe())

def run_neat(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    winner = p.run(eval_genomes, GENERATIONS)

    global best_genome
    with open("best_genome.pkl", "wb") as output_file:
        pickle.dump(best_genome, output_file)
    with open("last_genome.pkl", "wb") as output_file:
        pickle.dump(winner, output_file)

    display_stat_history()

# -------------------- Genetic Algorithm on Heuristics (using PyGAD) --------------------

def eval_pygad(ga_instance: pygad.GA, weights, solution_idx):
    env = TetrisGame(weights=weights, seed=rng.randrange(0, 1000000000))
    # --- Run multiple episodes for more stable fitness ---
    num_episodes = 5 # Average fitness over a few episodes
    episode_rewards = []

    for _ in range(num_episodes):
        env.reset()
        terminated = False
        truncated = False
        max_steps_per_episode = 500 # Adjust as needed for the environment

        for _ in range(max_steps_per_episode):
            action = env.heuristic_move()

            # --- Step the environment ---
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break
        episode_rewards.append(env.score)
        SCORE_HISTORY.append(env.score)
        REWARD_HISTORY.append(env.total_reward)
    
    env.close()
    return np.median(episode_rewards)  # Return the average reward as fitness

def print_pygad_generation(ga_instance: pygad.GA):
    print(f"Generation = {ga_instance.generations_completed}")

    # Get fitness values of the last generation
    last_gen_fitness = ga_instance.last_generation_fitness
    if last_gen_fitness is not None and len(last_gen_fitness) > 0:
        fitness_df = pd.DataFrame(last_gen_fitness, columns=['Fitness'])
        print("Last Generation Fitness Stats:")
        print(fitness_df.describe())
    else:
        print("No fitness data available for the last generation yet.")

    print("------------------------------") # Separator

def run_pygad():
    # PyGAD training on heuristic weights
    # Configure the GA parameters
    ga_instance = pygad.GA(num_generations=50,
                           num_parents_mating=6,
                           fitness_func=eval_pygad,
                           sol_per_pop=30,
                           num_genes=NUM_WEIGHTS,
                           gene_type=float,
                           mutation_type="adaptive",
                           mutation_probability=[.1, .4],
                           mutation_percent_genes=15,
                           parent_selection_type="rank",
                           keep_elitism=2,
                           crossover_type="single_point",
                           crossover_probability=0.9,
                           initial_population=None,
                           gene_space=[{"low": 0, "high": 2.0}] * NUM_WEIGHTS,
                           on_generation=print_pygad_generation,
                           save_best_solutions=True,
                           )
    # Run the GA
    ga_instance.run()

    index = np.argmax(ga_instance.best_solutions_fitness)
    solution = ga_instance.best_solutions[index]

    print("Best solution found: ", solution)
    print("Fitness of the best solution: ", ga_instance.best_solutions_fitness[index])
    with open("best_weights.txt", "w") as f:
        f.write(" ".join(map(str, solution)))
    display_stat_history()

if __name__ == "__main__":
    # Create the environment
    env = TetrisGame()
    
    model_type = os.environ.get("MODEL")
    print(model_type)

    if model_type == 'RL':
        # Reinforcement Learning training
        args = sys.argv[1:]
        # Train the model
        run_rl(env, model_file=(args[0] if len(args) >= 1 else None), output_file=(args[1] if len(args) >= 2 else "ppo_tetris_custom_net"))
        # Close the environment
        env.close()
    else:
        # PyGAD training with heuristics
        run_pygad()

    # NEAT training with genomes
    # local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward.txt')
    # run_neat(config_path)