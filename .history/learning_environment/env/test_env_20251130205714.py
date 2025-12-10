from pathlib import Path

from agents.dqn_agent import DQNAgent
from env.ticket_pricing_env import TicketPricingEnv

# Create environment
env = TicketPricingEnv(
    demand_model_path=Path('models/demand_model_v1.pkl'),
    initial_price_range=(100.0, 500.0),
    quality_range=(0.0, 1.0),
    time_horizon=720.0,
    time_step=6.0,
    price_bounds=(0.3, 3.0),
    random_seed=42
)

# Create or load agent
checkpoint_path = Path('checkpoints/dqn_ticket_pricing.pt')
if checkpoint_path.exists():
    print("Loading existing checkpoint...")
    agent = DQNAgent.load(env, checkpoint_path)
else:
    print("Creating new agent...")
    agent = DQNAgent(
        env=env,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        min_buffer_size=1_000,
        target_update_freq=1000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
    )

# Train (now explicitly resets env at start)
rewards = agent.train(n_episodes=1000)

# Save checkpoint
agent.save(checkpoint_path)

# Evaluate
eval_metrics = agent.evaluate(n_episodes=100)
print(f"Mean reward: ${eval_metrics['mean_reward']:.2f}")