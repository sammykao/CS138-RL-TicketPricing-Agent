"""
Generate 3D plot of price vs probability vs time based on fitted demand curve.

Shows how sale probability changes with price and time to event.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demand_modeling.model_serializer import load_model
from env.feature_builder import build_features_from_state


def generate_3d_surface(
    model_path: Path,
    initial_price: float = 200.0,
    quality_score: float = 0.5,
    time_range: tuple = (24.0, 2000.0),
    price_range: tuple = (50.0, 500.0),
    n_time_points: int = 50,
    n_price_points: int = 50
):
    """
    Generate 3D surface data for price vs probability vs time.
    
    Args:
        model_path: Path to trained demand model
        initial_price: Initial ticket price (for computing price_rel)
        quality_score: Quality score (0-1)
        time_range: (min_time, max_time) in hours
        price_range: (min_price, max_price) in dollars
        n_time_points: Number of time points to sample
        n_price_points: Number of price points to sample
    
    Returns:
        time_grid, price_grid, prob_grid: 3D meshgrids for plotting
    """
    # Load model
    model = load_model(model_path)
    
    # Create grids
    time_values = np.linspace(time_range[0], time_range[1], n_time_points)
    price_values = np.linspace(price_range[0], price_range[1], n_price_points)
    time_grid, price_grid = np.meshgrid(time_values, price_values)
    
    # Event context (fixed for this plot)
    event_context = {
        'is_weekend': 0,
        'is_playoff': 0,
        'day_of_week': 'Sat'
    }
    
    # Compute probabilities for each (time, price) combination
    prob_grid = np.zeros_like(time_grid)
    
    print("Computing probabilities for 3D surface...")
    for i in range(n_price_points):
        for j in range(n_time_points):
            time_remaining = time_grid[i, j]
            current_price = price_grid[i, j]
            
            # Build features
            features = build_features_from_state(
                time_remaining=time_remaining,
                current_price=current_price,
                initial_price=initial_price,
                quality_score=quality_score,
                event_context=event_context,
                include_interactions=True
            )
            
            # Predict probability
            features_2d = features.reshape(1, -1)
            p_sale_before_event = model.predict_proba(features_2d)[0, 1]
            
            prob_grid[i, j] = p_sale_before_event
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_price_points} price points")
    
    return time_grid, price_grid, prob_grid


def plot_3d_surface(time_grid, price_grid, prob_grid, save_path: Path):
    """Plot 3D surface of price vs probability vs time."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(
        time_grid, price_grid, prob_grid,
        cmap='viridis',
        alpha=0.8,
        linewidth=0,
        antialiased=True
    )
    
    # Labels
    ax.set_xlabel('Time to Event (hours)', fontsize=12, labelpad=10)
    ax.set_ylabel('Price ($)', fontsize=12, labelpad=10)
    ax.set_zlabel('P(sale before event)', fontsize=12, labelpad=10)
    
    # Title
    ax.set_title('Demand Curve: Price vs Probability vs Time', fontsize=14, pad=20)
    
    # Color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='P(sale before event)')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n3D plot saved to {save_path}")
    plt.close()


def plot_2d_slices(time_grid, price_grid, prob_grid, save_path: Path):
    """Plot 2D slices of the 3D surface at different time points."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Select time slices
    n_slices = 6
    time_indices = np.linspace(0, time_grid.shape[1] - 1, n_slices, dtype=int)
    
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        time_value = time_grid[0, time_idx]
        prices = price_grid[:, time_idx]
        probs = prob_grid[:, time_idx]
        
        ax.plot(prices, probs, 'b-', linewidth=2)
        ax.set_xlabel('Price ($)', fontsize=10)
        ax.set_ylabel('P(sale before event)', fontsize=10)
        ax.set_title(f'Time to Event: {time_value:.0f} hours', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.suptitle('Demand Curves at Different Times', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D slices saved to {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D price-probability-time plot')
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('learning_environment/models/demand_model_v6_timecorrected.pkl'),
        help='Path to trained demand model'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('learning_environment/plots'),
        help='Directory to save plots'
    )
    parser.add_argument(
        '--initial-price',
        type=float,
        default=200.0,
        help='Initial ticket price for computing price_rel'
    )
    parser.add_argument(
        '--quality-score',
        type=float,
        default=0.5,
        help='Quality score (0-1)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("3D PRICE-PROBABILITY-TIME PLOT GENERATION")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Initial Price: ${args.initial_price:.2f}")
    print(f"Quality Score: {args.quality_score:.2f}")
    print()
    
    # Generate 3D surface
    time_grid, price_grid, prob_grid = generate_3d_surface(
        model_path=args.model_path,
        initial_price=args.initial_price,
        quality_score=args.quality_score,
        time_range=(24.0, 2000.0),
        price_range=(50.0, 500.0),
        n_time_points=50,
        n_price_points=50
    )
    
    # Plot 3D surface
    plot_3d_surface(
        time_grid, price_grid, prob_grid,
        save_path=args.output_dir / 'demand_3d_price_prob_time.png'
    )
    
    # Plot 2D slices
    plot_2d_slices(
        time_grid, price_grid, prob_grid,
        save_path=args.output_dir / 'demand_2d_slices_by_time.png'
    )
    
    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()



