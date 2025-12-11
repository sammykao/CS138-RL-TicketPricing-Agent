# Checkpoints

This folder contains saved RL agent checkpoints from training.

## Contents

- **`dqn_ticket_pricing.pt`**: PyTorch checkpoint containing:
  - Q-network weights
  - Target network weights
  - Optimizer state
  - Training step count
  - Epsilon schedule parameters

- **`episode_metadata.json`**: Training metadata including:
  - Total episodes trained
  - Average rewards
  - Training configuration

## Usage

Checkpoints are automatically saved during training (every 100 episodes by default). They can be loaded by:
- Visualization tool (to watch trained agent behavior)
- Training scripts (to resume training)
- Evaluation scripts (to test agent performance)

To load a checkpoint:
```python
from agents.dqn_agent import DQNAgent
agent = DQNAgent.load(checkpoint_path)
```

