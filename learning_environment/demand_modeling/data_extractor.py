"""
Extract and aggregate sales data from SQLite database for demand modeling.

Key design decisions:
- Time binning: Log-scale bins (no surge assumption for NBA tickets)
- Price normalization: Per (event, quality_tier) median reference price
- Aggregation: Binomial-style (sold_count, exposure) for probability modeling
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def load_database(db_path: Path) -> sqlite3.Connection:
    """Load SQLite database connection."""
    return sqlite3.connect(str(db_path))


def compute_time_bin_log_scale(time_to_event: float, max_hours: float = 720.0) -> int:
    """
    Compute time bin using log-scale to handle long-tail distribution.
    
    NBA tickets sell mostly 30+ days before event, so we use log-scale bins
    rather than urgency-focused bins.
    
    Bins (hours):
    - Bin 0: [0, 24)      - Last day
    - Bin 1: [24, 72)     - 1-3 days
    - Bin 2: [72, 168)    - 3-7 days
    - Bin 3: [168, 336)   - 7-14 days
    - Bin 4: [336, 720)   - 14-30 days
    - Bin 5: [720, max]   - 30+ days
    
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


def compute_reference_prices(conn: sqlite3.Connection, target_hours: float = 2000.0, window_hours: float = 50.0) -> Dict[Tuple[int, str], float]:
    """
    Compute reference price for each (event_id, quality_tier) pair.
    
    Uses median price in a VERY narrow window around target_hours (default 2000h) before event.
    Window is ±25 hours (default 50h total) to capture prices very close to 2000h.
    
    Args:
        conn: Database connection
        target_hours: Target time to event for reference price (default 2000h)
        window_hours: Window size around target (default 50h, so 1975-2025h)
    
    Returns:
        Dict mapping (event_id, quality_tier) -> reference_price
    """
    window_start = max(0, target_hours - window_hours / 2)  # 1975h
    window_end = target_hours + window_hours / 2  # 2025h
    
    query = """
        SELECT 
            event_id,
            CASE 
                WHEN CAST(ticket_quality AS REAL) >= 0.75 THEN 'Premium'
                WHEN CAST(ticket_quality AS REAL) >= 0.50 THEN 'High'
                WHEN CAST(ticket_quality AS REAL) >= 0.25 THEN 'Medium'
                ELSE 'Low'
            END as quality_tier,
            Price
        FROM ticket_sales
        WHERE time_to_event >= ? AND time_to_event <= ?
            AND Price IS NOT NULL
            AND ticket_quality IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn, params=(window_start, window_end))
    
    # Compute median price per (event_id, quality_tier)
    ref_prices = {}
    for (event_id, quality_tier), group in df.groupby(['event_id', 'quality_tier']):
        if len(group) > 0:
            ref_prices[(event_id, quality_tier)] = group['Price'].median()
    
    print(f"Reference prices computed from {len(df)} sales in window [{window_start:.1f}h, {window_end:.1f}h]")
    print(f"  Found {len(ref_prices)} unique (event_id, quality_tier) combinations")
    
    return ref_prices


def extract_sales_data(
    db_path: Path,
    max_time_hours: float = 2000.0,
    min_sales_per_event: int = 10,
    reference_target_hours: float = 2000.0,
    reference_window_hours: float = 50.0  # Very narrow window around target_hours
) -> pd.DataFrame:
    """
    Extract and aggregate sales data from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        max_time_hours: Maximum time_to_event to consider (default 30 days)
        min_sales_per_event: Minimum sales required per event (filter sparse events)
    
    Returns:
        DataFrame with columns:
        - event_id, quality_tier, time_bin, price_rel, sold_count, exposure,
        - day_of_week, is_weekend, is_playoff, year, month
    """
    conn = load_database(db_path)
    
    # Load all sales with event context
    query = """
        SELECT 
            ts.event_id,
            ts.time_to_event,
            ts.Price,
            ts.ticket_quality,
            ts.Qty,
            e.day_of_week,
            e.year,
            e.month,
            e.away_team,
            e.home_team
        FROM ticket_sales ts
        JOIN events e ON ts.event_id = e.event_id
        WHERE ts.time_to_event IS NOT NULL
            AND ts.time_to_event >= 0
            AND ts.time_to_event <= ?
            AND ts.Price IS NOT NULL
            AND ts.ticket_quality IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn, params=(max_time_hours,))
    
    if len(df) == 0:
        raise ValueError("No sales data found in database")
    
    # Use quality score directly (continuous 0-1)
    df['quality_score'] = df['ticket_quality'].astype(float)
    
    # Still compute quality_tier for reference price calculation (backward compatibility)
    df['quality_tier'] = df['quality_score'].apply(
        lambda x: 'Premium' if x >= 0.75 
        else 'High' if x >= 0.50 
        else 'Medium' if x >= 0.25 
        else 'Low'
    )
    
    # Compute reference prices around target time (e.g., 2000h)
    ref_prices = compute_reference_prices(conn, target_hours=reference_target_hours, window_hours=reference_window_hours)
    
    # Compute relative prices
    def get_price_rel(row):
        key = (row['event_id'], row['quality_tier'])
        if key in ref_prices and ref_prices[key] > 0:
            return row['Price'] / ref_prices[key]
        return np.nan
    
    df['price_rel'] = df.apply(get_price_rel, axis=1)
    
    # Filter out rows with invalid price_rel
    df = df.dropna(subset=['price_rel'])
    
    # Compute time bins for aggregation (but won't use in features)
    df['time_bin'] = df['time_to_event'].apply(compute_time_bin_log_scale)
    
    # Add event context features
    df['is_weekend'] = df['day_of_week'].isin(['Fri', 'Sat', 'Sun']).astype(int)
    df['is_playoff'] = (df['month'] >= 4).astype(int)  # April+ = playoffs
    
    # Filter events with too few sales
    event_counts = df.groupby('event_id').size()
    valid_events = event_counts[event_counts >= min_sales_per_event].index
    df = df[df['event_id'].isin(valid_events)]
    
    # Aggregate by (event_id, quality_tier, time_bin) but use continuous time in features
    # FIXED: Account for time remaining to avoid bias towards higher probability near event
    aggregated = []
    
    for (event_id, quality_tier), event_group in df.groupby(['event_id', 'quality_tier']):
        # Sort by time_to_event descending (far to near)
        event_group = event_group.sort_values('time_to_event', ascending=False)
        
        # Compute cumulative inventory (exposure)
        total_sold = event_group['Qty'].sum()
        cumulative_sold = event_group['Qty'].cumsum()
        exposure = total_sold - cumulative_sold + event_group['Qty']
        
        # Group by time_bin for exposure calculation
        for time_bin, bin_group in event_group.groupby('time_bin'):
            sold_count = bin_group['Qty'].sum()
            # Exposure for this bin = remaining before bin starts
            bin_exposure = exposure[bin_group.index].iloc[0] if len(bin_group) > 0 else 0
            
            # Average price_rel in this bin
            avg_price_rel = bin_group['price_rel'].mean()
            
            # Average quality_score in this bin (continuous, 0-1)
            avg_quality_score = bin_group['quality_score'].mean()
            
            # Average time_to_event (continuous, not bin) for features
            avg_time_to_event = bin_group['time_to_event'].mean()
            
            # FIXED: Compute time-normalized probability
            # Instead of raw sold_count/exposure, we account for time remaining
            # The model should learn: more time = higher probability of eventual sale
            # We normalize by time remaining to avoid bias
            
            # Get time bin boundaries to compute time remaining in bin
            bin_boundaries = {
                0: (0, 24),
                1: (24, 72),
                2: (72, 168),
                3: (168, 336),
                4: (336, 720),
                5: (720, float('inf'))
            }
            bin_start, bin_end = bin_boundaries.get(time_bin, (0, float('inf')))
            bin_end = min(bin_end, max_time_hours)
            bin_start = min(bin_start, max_time_hours)
            
            # Time remaining in this bin (from start of bin to end)
            time_in_bin = bin_end - bin_start
            
            # Get event context (same for all rows in event)
            row = bin_group.iloc[0]
            
            aggregated.append({
                'event_id': event_id,
                'quality_tier': quality_tier,  # Keep for reference price lookup
                'quality_score': avg_quality_score,  # Continuous quality score (0-1)
                'time_to_event': avg_time_to_event,  # Continuous time (not bin) for features
                'time_bin': time_bin,  # Keep for aggregation but not in features
                'time_in_bin': time_in_bin,  # Time span of this bin (for normalization)
                'price_rel': avg_price_rel,
                'sold_count': int(sold_count),
                'exposure': max(1, int(bin_exposure)),  # At least 1 to avoid division by zero
                'day_of_week': row['day_of_week'],
                'is_weekend': row['is_weekend'],
                'is_playoff': row['is_playoff'],
                'year': row['year'],
                'month': row['month'],
                'away_team': row['away_team'],
                'home_team': row['home_team']
            })
    
    conn.close()
    
    result_df = pd.DataFrame(aggregated)
    
    # FIXED: Compute time-corrected empirical probability
    # The issue: raw sold_count/exposure creates bias (late bins have low exposure → high prob)
    # Solution: Compute probability of eventual sale, accounting for time remaining
    
    # Raw probability in this bin (sold_count / exposure)
    result_df['raw_prob'] = result_df['sold_count'] / result_df['exposure']
    
    # FIXED APPROACH: Compute probability of eventual sale given time remaining
    # Instead of bin-level probability, we want: P(sale before event | time_remaining = T)
    # 
    # Key insight: If a ticket hasn't sold yet at time T, the probability it will eventually sell
    # should DECREASE as T decreases (less time = less opportunity)
    #
    # We compute this by:
    # 1. For each (event_id, quality_tier), compute total eventual sale rate
    # 2. Adjust by time remaining: more time = higher prob, less time = lower prob
    # 3. Use time-weighted adjustment to counteract exposure bias
    
    # Compute overall sale rate per (event_id, quality_tier)
    event_sale_rates = {}
    for event_id in result_df['event_id'].unique():
        event_data = result_df[result_df['event_id'] == event_id]
        for quality_tier in event_data['quality_tier'].unique():
            tier_data = event_data[event_data['quality_tier'] == quality_tier]
            total_sold = tier_data['sold_count'].sum()
            total_exposure = tier_data['exposure'].max()  # Initial exposure (at start)
            if total_exposure > 0:
                event_sale_rates[(event_id, quality_tier)] = total_sold / total_exposure
            else:
                event_sale_rates[(event_id, quality_tier)] = 0.0
    
    # FIXED: Use raw_prob as target, but use time-weighted sample weights
    # The key insight: raw_prob has exposure bias, but we can correct it via sample weights
    # instead of modifying the target variable directly
    
    # Use raw_prob as the target (it's the empirical probability in this bin)
    result_df['empirical_prob'] = result_df['raw_prob'].clip(0.0, 1.0)
    
    # Apply time-based sample weighting to counteract exposure bias
    # More time remaining → higher weight (helps model learn correct time pattern)
    time_weight = (result_df['time_to_event'] / max_time_hours) ** 1.5
    result_df['time_weight'] = time_weight
    
    return result_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Get summary statistics of extracted data."""
    return {
        'n_observations': len(df),
        'n_events': df['event_id'].nunique(),
        'n_quality_tiers': df['quality_tier'].nunique(),
        'time_range': (df['time_to_event'].min(), df['time_to_event'].max()),
        'total_sold': df['sold_count'].sum(),
        'total_exposure': df['exposure'].sum(),
        'overall_sale_rate': df['sold_count'].sum() / df['exposure'].sum(),
        'price_rel_range': (df['price_rel'].min(), df['price_rel'].max()),
        'price_rel_median': df['price_rel'].median()
    }


if __name__ == '__main__':
    # Test extraction
    db_path = Path(__file__).parent.parent / 'data_generation' / 'db.sqlite'
    df = extract_sales_data(db_path)
    print("Extracted data shape:", df.shape)
    print("\nSummary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nFirst few rows:")
    print(df.head())

