from ticket_pricing_env import TicketPricingEnv
from pathlib import Path
 
env = TicketPricingEnv(
    demand_model_path=Path('../models/demand_model_v1.pkl'),
    initial_price_range=(100.0, 500.0),
    quality_range=(0.0, 1.0),
    time_horizon=720.0,  # 30 days
    time_step=6.0,        # 6 hours per step
    price_bounds=(0.3, 3.0),
    random_seed=42
)



print(env._get_obs())