"""
Helper functions for computing data needed for visualization.

Includes demand curve calculations and agent behavior tracking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.feature_builder import build_features_from_state
from demand_modeling.model_serializer import load_model


def compute_demand_curve(
    env,
    n_price_points: int = 50,
    fixed_time_remaining: Optional[float] = None,
    price_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute demand curve: P(sale) vs price for current episode state.
    
    Args:
        env: TicketPricingEnv instance
        n_price_points: Number of price points to sample
        fixed_time_remaining: If provided, use this time instead of current
        price_range: (min_price, max_price) - if None, uses env price bounds
    
    Returns:
        prices: Array of price values
        probabilities: Array of sale probabilities at each price
    """
    # Use current episode state
    if fixed_time_remaining is None:
        time_remaining = env.time_remaining
    else:
        time_remaining = fixed_time_remaining
    
    initial_price = env.initial_price
    quality_score = env.quality_score
    event_context = env.event_context
    
    # Determine price range
    if price_range is None:
        min_price = initial_price * env.price_bounds[0]
        max_price = initial_price * env.price_bounds[1]
    else:
        min_price, max_price = price_range
    
    # Sample prices
    prices = np.linspace(min_price, max_price, n_price_points)
    probabilities = np.zeros(n_price_points)
    
    # Compute probability for each price
    for i, price in enumerate(prices):
        try:
            # Build features with this price
            features = build_features_from_state(
                time_remaining=time_remaining,
                current_price=price,
                initial_price=initial_price,
                quality_score=quality_score,
                event_context=event_context,
                include_interactions=True
            )
            
            # Query demand model
            features_2d = features.reshape(1, -1)
            p_sale_bin = env.demand_model.predict_proba(features_2d)[0, 1]
            
            # Apply scaling and capping (matching env logic)
            p_sale_scaled = p_sale_bin * env.demand_scale
            p_sale_capped = min(p_sale_scaled, env.max_probability)
            
            # Convert bin-level to per-step probability
            # (simplified - using same conversion as env)
            from env.feature_builder import compute_time_bin_log_scale
            current_bin = compute_time_bin_log_scale(time_remaining, env.time_horizon)
            
            bin_boundaries = {
                0: (0, 24),
                1: (24, 72),
                2: (72, 168),
                3: (168, 336),
                4: (336, 720),
                5: (720, float('inf'))
            }
            
            bin_start, bin_end = bin_boundaries[current_bin]
            bin_end = min(bin_end, env.time_horizon)
            bin_start = min(bin_start, env.time_horizon)
            
            hours_in_bin = max(time_remaining - bin_start, env.time_step)
            steps_in_bin = max(1, int(np.ceil(hours_in_bin / env.time_step)))
            
            if p_sale_capped >= 1.0:
                p_step = 1.0
            elif p_sale_capped <= 0.0:
                p_step = 0.0
            else:
                p_step = 1.0 - np.power(1.0 - p_sale_capped, 1.0 / steps_in_bin)
            
            probabilities[i] = float(np.clip(p_step, 0.0, 1.0))
        except Exception as e:
            # Fallback to 0 if prediction fails
            probabilities[i] = 0.0
    
    return prices, probabilities


class BehaviorTracker:
    """Track agent's cumulative behavior over multiple episodes."""
    
    def __init__(self, max_time_steps: int = 200):
        """
        Initialize tracker for cumulative tracking across episodes.
        
        Args:
            max_time_steps: Maximum number of time steps to track
        """
        self.max_time_steps = max_time_steps
        # Cumulative data: track by time step across all episodes
        # step -> list of price_change_pct values at that step
        self.price_changes_by_step: Dict[int, List[float]] = defaultdict(list)
        self.rewards_by_step: Dict[int, List[float]] = defaultdict(list)
        self.total_episodes = 0
    
    def reset(self):
        """Reset for new episode (but keep cumulative data)."""
        # Don't clear cumulative data, just prepare for new episode
        pass
    
    def add_step(
        self,
        price_change_pct: float,
        reward: float,
        step: int
    ):
        """
        Add a step to cumulative history.
        
        Args:
            price_change_pct: Percentage change in price from previous step (e.g., +10.0 for +10%)
            reward: Reward received at this step
            step: Time step index (0-based)
        """
        # Track price changes by step (cumulative across episodes)
        if step not in self.price_changes_by_step:
            self.price_changes_by_step[step] = []
        self.price_changes_by_step[step].append(price_change_pct)
        
        # Track rewards by step (cumulative)
        if step not in self.rewards_by_step:
            self.rewards_by_step[step] = []
        self.rewards_by_step[step].append(reward)
        
        # Limit history size per step
        if len(self.price_changes_by_step[step]) > 1000:
            self.price_changes_by_step[step].pop(0)
        if len(self.rewards_by_step[step]) > 1000:
            self.rewards_by_step[step].pop(0)
    
    def finish_episode(self):
        """Mark that an episode has finished."""
        self.total_episodes += 1
    
    def get_avg_price_change_by_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get average price change % at each time step (cumulative across episodes).
        
        Returns:
            steps: Array of time step indices
            avg_price_changes: Array of average price change % at each step
        """
        if len(self.price_changes_by_step) == 0:
            return np.array([]), np.array([])
        
        steps = sorted(self.price_changes_by_step.keys())
        avg_changes = []
        
        for step in steps:
            changes = self.price_changes_by_step[step]
            if len(changes) > 0:
                avg_changes.append(np.mean(changes))
            else:
                avg_changes.append(0.0)
        
        return np.array(steps), np.array(avg_changes)
    
    def get_avg_reward_by_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get average reward at each time step (cumulative across episodes).
        
        Returns:
            steps: Array of time step indices
            avg_rewards: Array of average reward at each step
        """
        if len(self.rewards_by_step) == 0:
            return np.array([]), np.array([])
        
        steps = sorted(self.rewards_by_step.keys())
        avg_rewards = []
        
        for step in steps:
            rewards = self.rewards_by_step[step]
            if len(rewards) > 0:
                avg_rewards.append(np.mean(rewards))
            else:
                avg_rewards.append(0.0)
        
        return np.array(steps), np.array(avg_rewards)

