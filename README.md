# AI Tetris Bot

### Info

- Current model uses heuristics tuned with a genetic algorithm, averages >10k score per game
- Researched other reinforcement learning bots
- Reinforcement Learning model averages just 1k score per game
- Has inputs of certain board info and states
- Genetic model using NEAT fails to play optimally

### Instructions

- pip install -r requirements.txt
- cd frontend, npm i
- call python server.py for backend
- have another terminal to /frontend and run npm run dev and connect to localhost:5173
- python train.py to train
- Also can do docker compose up -d --build to build and run


