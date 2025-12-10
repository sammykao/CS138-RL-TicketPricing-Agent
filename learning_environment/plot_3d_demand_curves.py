"""
Generate 3D plots of demand curve fitting for each quality bin.

Creates:
1. 3D surface plots: Time to Event vs Price Change % vs Probability of Sale (one per quality tier)
2. 2D plot: Average Ticket Sale Price vs Time (accounting for quality)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Add learning_environment directory to path to allow imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from demand_modeling.model_serializer import load_model
from env.feature_builder import build_features_from_state, quality_score_to_tier


def generate_3d_demand_surface(
    model_path: Path,
    quality_tier: str,
    initial_price: float = 300.0,
    time_range: tuple = (0, 720),
    price_change_range: tuple = (-50, 50),
    n_time_points: int = 50,
    n_price_points: int = 50,
    is_weekend: bool = False,
    is_playoff: bool = False
):
    """
    Generate 3D surface data for demand curve.
    
    Args:
        model_path: Path to demand model .pkl file
        quality_tier: Quality tier ('Low', 'Medium', 'High', 'Premium')
        initial_price: Initial ticket price
        time_range: (min_time, max_time) in hours
        price_change_range: (min_change_pct, max_change_pct) percentage change
        n_time_points: Number of time points to sample
        n_price_points: Number of price change points to sample
        is_weekend: Whether it's a weekend
        is_playoff: Whether it's a playoff game
    
    Returns:
        time_grid: 2D array of time values
        price_change_grid: 2D array of price change % values
        prob_grid: 2D array of sale probabilities
    """
    # Load model
    model = load_model(model_path)
    
    # Map quality tier to quality score (use midpoint of tier range)
    quality_mapping = {
        'Low': 0.125,      # Midpoint of [0.0, 0.25)
        'Medium': 0.375,   # Midpoint of [0.25, 0.50)
        'High': 0.625,     # Midpoint of [0.50, 0.75)
        'Premium': 0.875   # Midpoint of [0.75, 1.0]
    }
    quality_score = quality_mapping.get(quality_tier, 0.5)
    
    # Create grids
    time_values = np.linspace(time_range[0], time_range[1], n_time_points)
    price_change_pct_values = np.linspace(price_change_range[0], price_change_range[1], n_price_points)
    
    time_grid, price_change_grid = np.meshgrid(time_values, price_change_pct_values)
    prob_grid = np.zeros_like(time_grid)
    
    # Event context
    event_context = {
        'is_weekend': is_weekend,
        'is_playoff': is_playoff,
        'day_of_week': 'Sat' if is_weekend else 'Tue'
    }
    
    # Compute probabilities for each point
    for i in range(n_price_points):
        for j in range(n_time_points):
            time_remaining = time_grid[i, j]
            price_change_pct = price_change_grid[i, j]
            
            # Convert price change % to actual price
            current_price = initial_price * (1 + price_change_pct / 100.0)
            
            # Build features and predict
            try:
                features = build_features_from_state(
                    time_remaining=time_remaining,
                    current_price=current_price,
                    initial_price=initial_price,
                    quality_score=quality_score,
                    event_context=event_context,
                    include_interactions=True
                )
                
                prob = model.predict_proba(features.reshape(1, -1))[0, 1]
                prob_grid[i, j] = prob
            except Exception as e:
                print(f"Warning: Failed to predict at time={time_remaining}, price_change={price_change_pct}: {e}")
                prob_grid[i, j] = 0.0
    
    return time_grid, price_change_grid, prob_grid


def plot_3d_demand_curves_by_quality(model_path: Path, output_path: Path = None):
    """
    Create 3D surface plots for each quality tier.
    
    Args:
        model_path: Path to demand model
        output_path: Optional path to save figure
    """
    quality_tiers = ['Low', 'Medium', 'High', 'Premium']
    
    # Create figure with subplots for each quality tier
    fig = plt.figure(figsize=(20, 15))
    
    for idx, quality_tier in enumerate(quality_tiers):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Generate surface data
        time_grid, price_change_grid, prob_grid = generate_3d_demand_surface(
            model_path=model_path,
            quality_tier=quality_tier,
            initial_price=300.0,
            time_range=(0, 720),
            price_change_range=(-50, 50),
            n_time_points=40,
            n_price_points=40
        )
        
        # Create surface plot
        surf = ax.plot_surface(
            time_grid,
            price_change_grid,
            prob_grid,
            cmap='viridis',
            alpha=0.8,
            linewidth=0,
            antialiased=True
        )
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='P(Sale)')
        
        # Labels and title
        ax.set_xlabel('Time to Event (hours)', fontsize=10)
        ax.set_ylabel('Price Change %', fontsize=10)
        ax.set_zlabel('Probability of Sale', fontsize=10)
        ax.set_title(f'Demand Curve: {quality_tier} Quality', fontsize=12, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
    
    plt.suptitle('3D Demand Curves by Quality Tier', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3D plots saved to {output_path}")
    else:
        plt.show()


def compute_avg_sale_price(
    model_path: Path,
    time_remaining: float,
    quality_tier: str,
    initial_price: float = 300.0,
    price_range: tuple = (0.3, 3.0),
    n_price_points: int = 200,
    is_weekend: bool = False,
    is_playoff: bool = False
) -> float:
    """
    Compute average sale price at a given time.
    
    Computes weighted average: E[price] = sum(price * P(sale | price)) / sum(P(sale | price))
    This represents the expected price at which a ticket would sell at this time,
    weighted by the probability of sale at each price point.
    
    Args:
        model_path: Path to demand model
        time_remaining: Time to event in hours
        quality_tier: Quality tier
        initial_price: Initial ticket price
        price_range: (min_multiplier, max_multiplier) relative to initial_price
        n_price_points: Number of price points to sample
        is_weekend: Whether it's a weekend
        is_playoff: Whether it's a playoff game
    
    Returns:
        Average sale price (weighted by sale probability)
    """
    model = load_model(model_path)
    
    # Map quality tier to quality score
    quality_mapping = {
        'Low': 0.125,
        'Medium': 0.375,
        'High': 0.625,
        'Premium': 0.875
    }
    quality_score = quality_mapping.get(quality_tier, 0.5)
    
    # Sample prices
    min_price = initial_price * price_range[0]
    max_price = initial_price * price_range[1]
    prices = np.linspace(min_price, max_price, n_price_points)
    
    event_context = {
        'is_weekend': is_weekend,
        'is_playoff': is_playoff,
        'day_of_week': 'Sat' if is_weekend else 'Tue'
    }
    
    # Compute weighted average price
    total_prob = 0.0
    weighted_price_sum = 0.0
    
    for price in prices:
        try:
            features = build_features_from_state(
                time_remaining=time_remaining,
                current_price=price,
                initial_price=initial_price,
                quality_score=quality_score,
                event_context=event_context,
                include_interactions=True
            )
            
            prob = model.predict_proba(features.reshape(1, -1))[0, 1]
            total_prob += prob
            weighted_price_sum += price * prob
        except Exception:
            continue
    
    if total_prob > 1e-6:  # Avoid division by zero
        avg_price = weighted_price_sum / total_prob
    else:
        avg_price = initial_price
    
    return avg_price


def plot_avg_sale_price_vs_time(model_path: Path, output_path: Path = None):
    """
    Plot average ticket sale price versus time for each quality tier.
    
    Args:
        model_path: Path to demand model
        output_path: Optional path to save figure
    """
    quality_tiers = ['Low', 'Medium', 'High', 'Premium']
    time_values = np.linspace(0, 720, 100)  # 0 to 30 days
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Different colors for each tier
    
    for idx, quality_tier in enumerate(quality_tiers):
        avg_prices = []
        
        for time_remaining in time_values:
            avg_price = compute_avg_sale_price(
                model_path=model_path,
                time_remaining=time_remaining,
                quality_tier=quality_tier,
                initial_price=300.0
            )
            avg_prices.append(avg_price)
        
        ax.plot(
            time_values,
            avg_prices,
            label=f'{quality_tier} Quality',
            linewidth=2.5,
            color=colors[idx]
        )
    
    ax.set_xlabel('Time to Event (hours)', fontsize=12)
    ax.set_ylabel('Average Sale Price ($)', fontsize=12)
    ax.set_title('Average Ticket Sale Price vs Time to Event\n(by Quality Tier)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at common time points
    for time_marker in [24, 168, 336, 720]:
        ax.axvline(x=time_marker, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        if time_marker == 24:
            ax.text(time_marker, ax.get_ylim()[1] * 0.95, '1 day', rotation=90, ha='right', fontsize=8)
        elif time_marker == 168:
            ax.text(time_marker, ax.get_ylim()[1] * 0.95, '1 week', rotation=90, ha='right', fontsize=8)
        elif time_marker == 336:
            ax.text(time_marker, ax.get_ylim()[1] * 0.95, '2 weeks', rotation=90, ha='right', fontsize=8)
        elif time_marker == 720:
            ax.text(time_marker, ax.get_ylim()[1] * 0.95, '30 days', rotation=90, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Average sale price plot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main function to generate all plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate 3D demand curve plots and average sale price graphs")
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path(__file__).parent / 'models' / 'demand_model_v1.pkl',
        help='Path to demand model .pkl file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'plots',
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively (in addition to saving)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating 3D demand curve plots by quality tier...")
    plot_3d_demand_curves_by_quality(
        model_path=args.model_path,
        output_path=args.output_dir / '3d_demand_curves_by_quality.png'
    )
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print("Generating average sale price vs time plot...")
    plot_avg_sale_price_vs_time(
        model_path=args.model_path,
        output_path=args.output_dir / 'avg_sale_price_vs_time.png'
    )
    
    if args.show:
        plt.show()
    else:
        plt.close()
    
    print("All plots generated successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

