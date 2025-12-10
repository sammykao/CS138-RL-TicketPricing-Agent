"""
Feature builder for constructing model features from environment state.

Replicates the exact feature engineering logic from demand_modeling/feature_engineer.py
to ensure feature vectors match the trained model's expectations.
"""

import numpy as np
from typing import Dict


# Time bin centers (matching feature_engineer.py)
TIME_BIN_CENTERS = {
    0: 12.0,    # 0-24h -> 12h
    1: 48.0,    # 24-72h -> 48h
    2: 120.0,   # 72-168h -> 120h
    3: 252.0,   # 168-336h -> 252h
    4: 528.0,   # 336-720h -> 528h
    5: 1080.0   # 720h+ -> 1080h (approximate)
}

QUALITY_TIERS = ['Low', 'Medium', 'High', 'Premium']
DAYS_OF_WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def compute_time_bin_log_scale(time_to_event: float, max_hours: float = 720.0) -> int:
    """
    Compute time bin using log-scale (matching data_extractor.py).
    
    Args:
        time_to_event: Hours until event
        max_hours: Maximum time to consider (default 30 days)
    
    Returns:
        Bin index (0-5)
    """
    time_clipped = max(0.0, min(time_to_event, max_hours))
    
    if time_clipped < 24:
        return 0
    elif time_clipped < 72:
        return 1
    elif time_clipped < 168:
        return 2
    elif time_clipped < 336:
        return 3
    elif time_clipped < 720:
        return 4
    else:
        return 5


def quality_score_to_tier(quality_score: float) -> str:
    """Convert quality score (0-1) to tier string."""
    if quality_score >= 0.75:
        return 'Premium'
    elif quality_score >= 0.50:
        return 'High'
    elif quality_score >= 0.25:
        return 'Medium'
    else:
        return 'Low'


def quality_tier_to_one_hot(quality_tier: str) -> np.ndarray:
    """Convert quality tier to one-hot encoding."""
    one_hot = np.zeros(4)
    if quality_tier in QUALITY_TIERS:
        idx = QUALITY_TIERS.index(quality_tier)
        one_hot[idx] = 1.0
    return one_hot


def day_of_week_to_one_hot(day_of_week: str) -> np.ndarray:
    """Convert day of week to one-hot encoding."""
    one_hot = np.zeros(7)
    if day_of_week in DAYS_OF_WEEK:
        idx = DAYS_OF_WEEK.index(day_of_week)
        one_hot[idx] = 1.0
    return one_hot


def time_bin_to_one_hot(time_bin: int) -> np.ndarray:
    """Convert time bin to one-hot encoding."""
    one_hot = np.zeros(6)
    if 0 <= time_bin < 6:
        one_hot[time_bin] = 1.0
    return one_hot


def build_features_from_state(
    time_remaining: float,
    current_price: float,
    initial_price: float,
    quality_score: float,
    event_context: Dict,
    include_interactions: bool = True
) -> np.ndarray:
    """
    Build feature vector from environment state (NO TIME BINS - continuous time only).
    
    Predicts P(sale before event) directly, not P(sale in time bin).
    
    Feature order (matching model.feature_names):
    1. time_to_event (1 dim)
    2. price_log_rel (1 dim)
    3. price_rel (1 dim)
    4. price_log_rel_squared (1 dim) - Phase 1.1
    5. quality_score (1 dim, continuous)
    6. quality_price_interaction (1 dim) - Phase 1.3
    7. is_weekend (1 dim)
    8. is_playoff (1 dim)
    9. day_Mon through day_Sun (7 dims)
    10. price_time_interaction (1 dim, if include_interactions)
    11. price_urgency_interaction (1 dim, if include_interactions) - Phase 1.2
    12. price_time_inverse (1 dim, if include_interactions) - FIX: penalizes high prices near event
    13. quality_time_interaction (1 dim, if include_interactions)
    
    Args:
        time_remaining: Hours until event
        current_price: Current ticket price
        initial_price: Initial ticket price (used as reference price)
        quality_score: Quality score (0-1)
        event_context: Dict with 'is_weekend', 'is_playoff', 'day_of_week'
        include_interactions: Whether to include interaction terms
    
    Returns:
        Feature vector (21 dims if interactions, 18 dims otherwise)
    """
    # 1. Time features (continuous, linear only - no log transformation)
    # Removed time_log to fix time bias
    
    # 2. Price features
    price_rel = current_price / initial_price
    price_log_rel = np.log(np.clip(price_rel, 0.1, 10.0))  # Clip to avoid log(0)
    
    # INCREASE PRICE SENSITIVITY: Multiply by same factor as training
    PRICE_SENSITIVITY_MULTIPLIER = 5.0
    price_log_rel = price_log_rel * PRICE_SENSITIVITY_MULTIPLIER
    
    # Phase 1.1: Non-linear price feature
    price_log_rel_squared = price_log_rel ** 2
    
    # Price threshold features for sharp demand curve
    price_below_0_8 = 1.0 if price_rel < 0.8 else 0.0
    price_below_0_9 = 1.0 if price_rel < 0.9 else 0.0
    price_above_1_2 = 1.0 if price_rel > 1.2 else 0.0
    price_above_1_5 = 1.0 if price_rel > 1.5 else 0.0
    
    # Phase 1.2: Urgency factor (time-decay: higher urgency as event approaches)
    urgency_factor = 1.0 / (1.0 + time_remaining / 24.0)
    
    # FIX: Add inverse time feature to capture price decrease as event approaches
    # Real-world pattern: prices peak early then decrease as event approaches
    time_inverse = 1.0 / (time_remaining + 1.0)  # Avoid division by zero
    price_time_inverse = price_log_rel * time_inverse
    
    # FIX: Add time penalty feature to reduce probability near event
    # Penalize sales when time is very low (last 24-48 hours)
    # NEGATIVE feature: higher absolute value = more penalty = lower probability
    if time_remaining < 48.0:
        time_penalty_near_event = -(48.0 - time_remaining) / 48.0  # NEGATIVE: 0 at 48h, -1 at 0h
    else:
        time_penalty_near_event = 0.0
    
    # Stronger price penalty near event - multiply price sensitivity by urgency
    # NEGATIVE feature: penalizes high prices when time is low
    price_penalty_near_event = -price_log_rel * abs(time_penalty_near_event) * 10.0  # Strong NEGATIVE penalty
    
    # 3. Quality features (use continuous quality_score directly)
    
    # Phase 1.3: Quality-adjusted price feature
    quality_price_interaction = price_log_rel * quality_score
    
    # 4. Event context
    is_weekend = float(event_context.get('is_weekend', 0))
    is_playoff = float(event_context.get('is_playoff', 0))
    day_of_week = event_context.get('day_of_week', 'Mon')
    day_onehot = day_of_week_to_one_hot(day_of_week)
    
    # Build feature vector in exact order (matching feature_engineer.py)
    features = [
        time_remaining,                     # time_to_event (continuous, linear)
        price_log_rel,                      # price_log_rel (multiplied by 5.0 for sensitivity)
        price_rel,                          # price_rel
        price_log_rel_squared,              # Phase 1.1: price_log_rel_squared
        price_below_0_8,                    # price_below_0.8 (threshold feature)
        price_below_0_9,                    # price_below_0.9 (threshold feature)
        price_above_1_2,                    # price_above_1.2 (threshold feature)
        price_above_1_5,                    # price_above_1.5 (threshold feature)
        quality_score,                      # quality_score (continuous, 0-1)
        quality_price_interaction,          # Phase 1.3: quality_price_interaction
        is_weekend,                         # is_weekend
        is_playoff,                         # is_playoff
        *day_onehot,                        # day_Mon through day_Sun
    ]
    
    # 5. Interaction terms (if enabled)
    if include_interactions:
        # Price × Time interaction (using linear time, not log)
        price_time_interaction = price_log_rel * time_remaining
        features.append(price_time_interaction)
        
        # Phase 1.2: Urgency-based price sensitivity
        price_urgency_interaction = price_log_rel * urgency_factor
        features.append(price_urgency_interaction)
        
        # FIX: Price-time inverse interaction - penalizes high prices when time is low
        # This captures the real-world pattern where prices decrease as event approaches
        features.append(price_time_inverse)
        
        # FIX: Time penalty near event - reduces probability in last 48 hours
        features.append(time_penalty_near_event)
        
        # FIX: Strong price penalty near event - prevents price gouging
        features.append(price_penalty_near_event)
        
        # Quality × Time interaction (using continuous quality_score and linear time)
        quality_time_interaction = quality_score * time_remaining
        features.append(quality_time_interaction)
    
    return np.array(features, dtype=np.float32)


