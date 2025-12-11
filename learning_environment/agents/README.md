# Agents

This folder contains the reinforcement learning agents used for dynamic ticket pricing.

## Contents

- **`base_agent.py`**: Abstract base class for all RL agents. Defines the interface that all agents must implement (select_action, update, save, load).

- **`dqn_agent.py`**: Deep Q-Network (DQN) agent implementation. This is the main agent used for training. Features:
  - Neural network Q-function approximation
  - Experience replay buffer
  - Target network for stable learning
  - Epsilon-greedy exploration strategy

- **`train_dqn/`**: Training infrastructure for the DQN agent
  - **`trainer.py`**: Main training loop that coordinates agent-environment interaction, checkpointing, and metrics tracking

## Usage

The DQN agent is used by the training scripts (`train_headless.py`) and visualization tools. Agents are saved/loaded as PyTorch checkpoints in the `checkpoints/` folder.

