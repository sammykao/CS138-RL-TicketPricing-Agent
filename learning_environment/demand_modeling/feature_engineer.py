"""
Feature engineering for demand probability modeling.

Transforms aggregated sales data into feature vectors suitable for
logistic regression or other probability models.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def quality_tier_to_one_hot(quality_tier: str) -> List[int]:
    """Convert quality tier to one-hot encoding."""
    tiers = ['Low', 'Medium', 'High', 'Premium']
    return [1 if quality_tier == tier else 0 for tier in tiers]


def day_of_week_to_one_hot(day_of_week: str) -> List[int]:
    """Convert day of week to one-hot encoding."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if day_of_week in days:
        idx = days.index(day_of_week)
        return [1 if i == idx else 0 for i in range(7)]
    return [0] * 7  # Unknown day


def time_bin_to_one_hot(time_bin: int) -> List[int]:
    """Convert time bin to one-hot encoding."""
    bins = list(range(6))  # 0-5
    return [1 if time_bin == b else 0 for b in bins]


def build_features(
    df: pd.DataFrame,
    include_interactions: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build feature vectors from aggregated sales data.
    
    Feature design:
    - Time: log(time_to_event), time_to_event (continuous) - NO TIME BINS
    - Price: log(price_rel), price_rel (continuous)
    - Quality: quality_score (continuous, 0-1)
    - Context: is_weekend, is_playoff, day_of_week one-hot
    - Interactions: price × time, quality × time (optional)
    
    Note: Model predicts P(sale before event) not P(sale in time bin)
    
    Args:
        df: DataFrame from data_extractor.extract_sales_data()
        include_interactions: Whether to include interaction terms
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target probabilities (n_samples,)
        sample_weights: Exposure weights (n_samples,)
        feature_names: List of feature names
    """
    # Use continuous time_to_event directly (no bins, no log transformation)
    df = df.copy()
    if 'time_to_event' not in df.columns:
        raise ValueError("time_to_event column not found. Make sure data_extractor provides continuous time.")
    
    # Price features
    df['price_log_rel'] = np.log(df['price_rel'].clip(0.1, 10.0))  # Clip to avoid log(0)
    
    # INCREASE PRICE SENSITIVITY: Multiply price features by large factor
    # This makes the sigmoid steeper (more sensitive to price changes)
    PRICE_SENSITIVITY_MULTIPLIER = 5.0  # Increase from 1.0 to 5.0 for sharper transitions
    df['price_log_rel'] = df['price_log_rel'] * PRICE_SENSITIVITY_MULTIPLIER
    
    # Phase 1.1: Non-linear price features
    df['price_log_rel_squared'] = df['price_log_rel'] ** 2
    
    # Add price threshold features for sharp transitions
    # These create step-like behavior: 0 if price too high, 1 if price low enough
    df['price_below_0.8'] = (df['price_rel'] < 0.8).astype(float)  # 20% discount threshold
    df['price_below_0.9'] = (df['price_rel'] < 0.9).astype(float)  # 10% discount threshold
    df['price_above_1.2'] = (df['price_rel'] > 1.2).astype(float)  # 20% premium threshold
    df['price_above_1.5'] = (df['price_rel'] > 1.5).astype(float)  # 50% premium threshold
    
    # Phase 1.2: Urgency factor (time-decay: higher urgency as event approaches)
    # urgency_factor = 1.0 / (1.0 + time_remaining / 24.0) ranges from [0, 1]
    # When time_remaining is large (2000h), urgency ≈ 0.02 (low urgency)
    # When time_remaining is small (24h), urgency ≈ 0.5 (high urgency)
    df['urgency_factor'] = 1.0 / (1.0 + df['time_to_event'] / 24.0)
    
    # FIX: Add inverse time feature to capture price decrease as event approaches
    # Real-world pattern: prices peak early (250-750h) then decrease as event approaches
    # This feature penalizes high prices when time is low
    df['time_inverse'] = 1.0 / (df['time_to_event'] + 1.0)  # Avoid division by zero
    df['price_time_inverse'] = df['price_log_rel'] * df['time_inverse']
    
    # FIX: Add time penalty feature to reduce probability near event
    # Penalize sales when time is very low (last 24-48 hours)
    # This prevents unrealistic high probabilities near event
    # NEGATIVE feature: higher value = more penalty = lower probability
    df['time_penalty_near_event'] = np.where(
        df['time_to_event'] < 48.0,  # Last 48 hours
        -(48.0 - df['time_to_event']) / 48.0,  # NEGATIVE: ranges from 0 (at 48h) to -1 (at 0h)
        0.0
    )
    
    # Stronger price penalty near event - multiply price sensitivity by urgency
    # NEGATIVE feature: penalizes high prices when time is low
    df['price_penalty_near_event'] = -df['price_log_rel'] * np.abs(df['time_penalty_near_event']) * 10.0  # Strong NEGATIVE penalty
    
    # Build feature vectors
    features_list = []
    feature_names = []
    
    # 1. Time features (continuous only - linear time, no log)
    # Removed time_log to fix time bias - using only linear time_to_event
    features_list.append(df['time_to_event'].values)
    feature_names.append('time_to_event')
    
    # 3. Price features
    features_list.append(df['price_log_rel'].values)
    feature_names.append('price_log_rel')
    
    features_list.append(df['price_rel'].values)
    feature_names.append('price_rel')
    
    # Phase 1.1: Non-linear price feature
    features_list.append(df['price_log_rel_squared'].values)
    feature_names.append('price_log_rel_squared')
    
    # Price threshold features for sharp demand curve
    features_list.append(df['price_below_0.8'].values)
    feature_names.append('price_below_0.8')
    features_list.append(df['price_below_0.9'].values)
    feature_names.append('price_below_0.9')
    features_list.append(df['price_above_1.2'].values)
    feature_names.append('price_above_1.2')
    features_list.append(df['price_above_1.5'].values)
    feature_names.append('price_above_1.5')
    
    # 4. Quality score (continuous, 0-1)
    features_list.append(df['quality_score'].values)
    feature_names.append('quality_score')
    
    # Phase 1.3: Quality-adjusted price feature
    features_list.append((df['price_log_rel'].values * df['quality_score'].values))
    feature_names.append('quality_price_interaction')
    
    # 5. Event context
    features_list.append(df['is_weekend'].values)
    feature_names.append('is_weekend')
    
    features_list.append(df['is_playoff'].values)
    feature_names.append('is_playoff')
    
    # 6. Day of week (one-hot)
    day_onehot = np.array([day_of_week_to_one_hot(d) for d in df['day_of_week']])
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
        features_list.append(day_onehot[:, i])
        feature_names.append(f'day_{day}')
    
    # 7. Interaction terms (optional)
    if include_interactions:
        # Price × Time interaction (using linear time_to_event, not log)
        price_time = df['price_log_rel'].values * df['time_to_event'].values
        features_list.append(price_time)
        feature_names.append('price_time_interaction')
        
        # Phase 1.2: Urgency-based price sensitivity
        # Price sensitivity increases as event approaches (urgency factor)
        # NOTE: This should make demand MORE elastic (less tolerant of high prices) near event
        price_urgency = df['price_log_rel'].values * df['urgency_factor'].values
        features_list.append(price_urgency)
        feature_names.append('price_urgency_interaction')
        
        # FIX: Price-time inverse interaction - penalizes high prices when time is low
        # This captures the real-world pattern where prices decrease as event approaches
        features_list.append(df['price_time_inverse'].values)
        feature_names.append('price_time_inverse')
        
        # FIX: Time penalty near event - reduces probability in last 48 hours
        features_list.append(df['time_penalty_near_event'].values)
        feature_names.append('time_penalty_near_event')
        
        # FIX: Strong price penalty near event - prevents price gouging
        features_list.append(df['price_penalty_near_event'].values)
        feature_names.append('price_penalty_near_event')
        
        # Quality × Time (using continuous quality_score and linear time)
        quality_time = df['quality_score'].values * df['time_to_event'].values
        features_list.append(quality_time)
        feature_names.append('quality_time_interaction')
    
    # Stack into feature matrix
    X = np.column_stack(features_list).astype(np.float32)
    
    # Target: empirical probability
    y = df['empirical_prob'].values
    
    # Sample weights: exposure * time_weight (to correct time bias)
    # exposure: more observations = higher weight
    # time_weight: give more weight to far-future observations to learn correct time pattern
    if 'time_weight' in df.columns:
        sample_weights = (df['exposure'].values * df['time_weight'].values).astype(np.float32)
    else:
        sample_weights = df['exposure'].values.astype(np.float32)
    
    return X, y, sample_weights, feature_names


def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Note: One-hot encoded features are already in [0,1] range,
    but we normalize them anyway for consistency.
    
    Args:
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
    
    Returns:
        X_train_scaled: Normalized training features
        X_test_scaled: Normalized test features
        scaler: Fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def get_feature_importance_info(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features for interpretability.
    
    Returns:
        Dict mapping category -> list of feature names
    """
    categories = {
        'time': [f for f in feature_names if 'time' in f.lower()],
        'price': [f for f in feature_names if 'price' in f.lower()],
        'quality': [f for f in feature_names if 'quality' in f.lower()],
        'context': [f for f in feature_names if f in ['is_weekend', 'is_playoff'] or f.startswith('day_')],
        'interactions': [f for f in feature_names if 'interaction' in f.lower()]
    }
    return categories


if __name__ == '__main__':
    # Test feature engineering
    from data_extractor import extract_sales_data
    from pathlib import Path
    
    db_path = Path(__file__).parent.parent / 'data_generation' / 'db.sqlite'
    df = extract_sales_data(db_path)
    
    X, y, weights, feature_names = build_features(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample weights shape: {weights.shape}")
    print(f"\nNumber of features: {len(feature_names)}")
    print("\nFeature categories:")
    categories = get_feature_importance_info(feature_names)
    for category, features in categories.items():
        print(f"  {category}: {len(features)} features")
        if len(features) <= 10:
            print(f"    {features}")

