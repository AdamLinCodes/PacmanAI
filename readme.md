# Deep Q-Learning with Convolutional Neural Network on Ms. Pacman

This project implements a deep Q-learning agent with a convolutional neural network (CNN) to play the Atari game Ms. Pacman. The project uses PyTorch for neural network implementation and Gymnasium’s Atari environment for game interaction.

## Overview

The agent uses Deep Q-Learning (DQN) to learn the optimal policy through interaction with the Ms. Pacman environment. The agent leverages CNN layers to process visual input frames and predict Q-values, representing the estimated rewards for each possible action. This enables the agent to learn a strategy to maximize its score over time.

Key techniques:
- **Experience Replay**: The agent stores experiences and learns from random samples to reduce correlation.
- **Epsilon-Greedy Policy**: A policy that balances exploration and exploitation by gradually reducing exploration.
- **Target Network**: A separate network to stabilize training, updated less frequently than the primary network.

### Model Architecture

- **Input**: Resized, grayscale frames of size 128x128.
- **Convolutional Layers**: Extract spatial features through four layers.
- **Fully Connected Layers**: Map features to Q-values for each action.

### Key Hyperparameters

- **Learning Rate**: `5e-4`
- **Discount Factor**: `0.99`
- **Epsilon Decay**: `0.995`
- **Batch Size**: `64`

## Requirements

This project requires Python 3 and the following libraries. You can install dependencies with the provided `requirements.txt`, or manually as shown below.

### Manual Installation

```bash
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install ale-py
apt-get install -y swig
pip install gymnasium[box2d]
```

## Install from requirements.txt
To install all dependencies at once:

```bash
pip install -r requirements.txt
```
