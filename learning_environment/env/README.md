# Environment

This folder contains the Markov Decision Process (MDP) environment for reinforcement learning.

## Contents

- **`ticket_pricing_env.py`**: Main environment implementation. Defines:
  - **State space**: 5D vector (time remaining, price ratio, quality, weekend, playoff)
  - **Action space**: 7 discrete actions (price changes: -20%, -10%, -5%, 0%, +5%, +10%, +20%)
  - **Transition dynamics**: Time progression, sale probability computation, reward calculation
  - **Reward function**: Percentage price change if sold, -1.0 if expired

- **`feature_builder.py`**: Builds feature vectors for the demand model from environment state. Converts state information into the 21-dimensional feature vector used by the demand model.

## Key Features

The environment integrates the trained demand model to compute realistic sale probabilities. It applies:
- Temperature scaling for sharper probability transitions
- Time-based penalties (reduced probability in last 48 hours, price-dependent)
- Price-based penalties (anti-gouging for high prices near event)
- Demand scaling to control environment difficulty
- Hazard rate conversion (event-level → per-step probability)

## Usage

The environment is used by:
- Training scripts (`train_headless.py`)
- Visualization tools (`visualization/app.py`)
- Graph generation scripts (`generate_env_demand_graphs.py`)

