# Flappy Bird Reinforcement Learning

This project implements a Flappy Bird game with a reinforcement learning agent and real-time visualization of neural network activations.

## Requirements

- Python 3.7+
- PyTorch
- Pygame
- Matplotlib
- Seaborn
- NumPy

## Installation

1. Clone the repository

2. Install the required packages:
```bash
pip install -r requirements.txt
```
## Running the Game

1. Navigate to the project directory

2. Run the main script:
```bash
python src/main.py
```
## Project Structure

- `assets/`: Contains game images (bird, pipe, background)
- `src/`: Source code files
- `game.py`: Flappy Bird game implementation
- `agent.py`: DQN agent implementation
- `visualizer.py`: Neural network activation visualizer
- `main.py`: Main script to run the game and training loop

## How it Works

The DQN agent learns to play Flappy Bird through trial and error. As it plays, you can observe the game and the real-time visualization of the neural network activations.

- The left plot shows input layer activations
- The middle plot shows hidden layer activations
- The right plot shows output layer activations

The agent's performance should improve over time, resulting in higher scores and longer gameplay duration.