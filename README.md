# AI Tetris Bot

## Summary
### Reinforcement Learning (Stable Baselines3)
I attempted to make a tetris bot using AI, starting with reinforcement learning. Initially, I had observation space using a binary grid, but changed it to heights, holes, etc. I trained for around 7 hours total, and it plays decently well, averaging 1k score.
### Neuroevolution (NEAT)
I switched to a neuroevolution-based neural network, and after training using the same configuration as reinforcement learning, it unfortunately yielded unimpressive results.
### Heuristic + Genetic Algorithm (PyGAD)
Lastly, since most Tetris bots are heuristic-based, I decided to create a heuristic algorithm with 0 lookahead (1 is too slow to train). I optimized the weights using a genetic algorithm, which improved average scores by 30%.

## Setup Info

- Current model uses heuristics tuned with a genetic algorithm, averages >10k score per game
- Researched other reinforcement learning bots
- Reinforcement Learning model averages just 1k score per game
- Has inputs of certain board info and states
- Genetic model using NEAT fails to play optimally

## Instructions

- pip install -r requirements.txt
- cd frontend, npm i
- call python server.py for backend
- have another terminal to /frontend and run npm run dev and connect to localhost:5173
- python train.py to train
- Also can do docker compose up -d --build to build and run


