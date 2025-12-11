"""
Generate 3D demand graphs using the environment's actual probability computation.
This includes all penalties, temperature scaling, and other environment logic.

This shows what the RL agent actually sees, not just raw model predictions.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from env.ticket_pricing_env import TicketPricingEnv
from demand_modeling.model_serializer import load_model


def generate_3d_surface_with_env(
    model_path: Path,
    initial_price: float = 200.0,
    quality_score: float = 0.5,
    time_range: tuple = (0, 2000.0),
    price_range: tuple = (50.0, 500.0),
    n_time_points: int = 60,
    n_price_points: int = 60,
    demand_scale: float = 0.5
):
    """
    Generate 3D surface data using the environment's actual probability computation.
    
    Args:
        model_path: Path to trained demand model
        initial_price: Initial ticket price (for computing price_rel)
        quality_score: Quality score (0-1)
        time_range: (min_time, max_time) in hours
        price_range: (min_price, max_price) in dollars
        n_time_points: Number of time points to sample
        n_price_points: Number of price points to sample
        demand_scale: Demand scale factor (default 0.5)
    
    Returns:
        time_grid, price_grid, prob_grid: 3D meshgrids for plotting
    """
    # Create environment (we'll reset it for each point)
    event_context = {
        'is_weekend': False,
        'is_playoff': False,
        'day_of_week': 'Tue'
    }
    
    env = TicketPricingEnv(
        demand_model_path=model_path,
        initial_price_range=(initial_price, initial_price),  # Fixed initial price
        quality_range=(quality_score, quality_score),  # Fixed quality
        demand_scale=demand_scale,
        time_horizon=2000.0,
        time_step=6.0
    )
    
    # Create grids
    time_values = np.linspace(time_range[0], time_range[1], n_time_points)
    price_values = np.linspace(price_range[0], price_range[1], n_price_points)
    time_grid, price_grid = np.meshgrid(time_values, price_values)
    
    # Compute probabilities for each (time, price) combination
    prob_grid = np.zeros_like(time_grid)
    
    print("Computing probabilities using environment logic...")
    print("This includes: temperature scaling, time penalties, price penalties, demand scaling")
    print()
    
    for i in range(n_price_points):
        for j in range(n_time_points):
            time_remaining = time_grid[i, j]
            current_price = price_grid[i, j]
            
            # Reset environment
            obs = env.reset()
            
            # Manually set state to desired values (bypassing normal reset)
            env.time_remaining = max(0.0, time_remaining)
            env.current_price = current_price
            env.initial_price = initial_price
            env.quality_score = quality_score
            env.event_context = event_context
            
            # Use environment's probability computation (includes all penalties)
            prob = env._compute_sale_probability()
            prob_grid[i, j] = prob
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{n_price_points} price points")
    
    return time_grid, price_grid, prob_grid


def plot_3d_surface(time_grid, price_grid, prob_grid, save_path: Path):
    """Plot 3D surface of price vs probability vs time."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(
        time_grid, price_grid, prob_grid,
        cmap='viridis',
        alpha=0.85,
        linewidth=0,
        antialiased=True,
        edgecolor='none'
    )
    
    # Labels
    ax.set_xlabel('Time to Event (hours)', fontsize=14, labelpad=12)
    ax.set_ylabel('Price ($)', fontsize=14, labelpad=12)
    ax.set_zlabel('P(sale per step)', fontsize=14, labelpad=12)
    
    # Title
    ax.set_title('Environment Demand Curve: Price vs Probability vs Time\n(Includes all penalties and scaling)', 
                 fontsize=16, pad=20, fontweight='bold')
    
    # Color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=25, label='P(sale per step)')
    cbar.ax.tick_params(labelsize=11)
    
    # Set viewing angle for better visibility
    ax.view_init(elev=25, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n3D plot saved to {save_path}")
    plt.close()


def plot_2d_slices(time_grid, price_grid, prob_grid, save_path: Path):
    """Plot 2D slices of the 3D surface at different time points."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Select time slices - focus on near-event times to show the penalty effect
    time_slices = [2000, 168, 72, 48, 24, 0]  # Far future, 1 week, 3 days, 2 days, 1 day, event time
    
    for idx, target_time in enumerate(time_slices):
        ax = axes[idx]
        
        # Find closest time index
        time_idx = np.argmin(np.abs(time_grid[0, :] - target_time))
        time_value = time_grid[0, time_idx]
        
        prices = price_grid[:, time_idx]
        probs = prob_grid[:, time_idx]
        
        # Plot
        ax.plot(prices, probs, 'b-', linewidth=2.5, label='Sale Probability')
        ax.set_xlabel('Price ($)', fontsize=11)
        ax.set_ylabel('P(sale per step)', fontsize=11)
        ax.set_title(f'Time to Event: {time_value:.0f} hours ({time_value/24:.1f} days)', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(0.1, prob_grid.max() * 1.1))
        
        # Add vertical line at initial price for reference
        initial_price = prices[len(prices)//2]  # Approximate initial price
        ax.axvline(x=initial_price, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Initial Price')
        
        # Highlight cheap prices (50% discount or more)
        cheap_threshold = initial_price * 0.5
        ax.axvspan(0, cheap_threshold, alpha=0.1, color='green', label='Fire Sale Zone')
        
        ax.legend(fontsize=9, loc='best')
    
    plt.suptitle('Demand Curves at Different Times (Environment Logic)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D slices saved to {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D demand graphs using environment logic')
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('learning_environment/models/demand_model_v12_anti_gouge.pkl'),
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
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=0.5,
        help='Demand scale factor (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        print("Please specify correct model path with --model-path")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("3D DEMAND GRAPH GENERATION (USING ENVIRONMENT LOGIC)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Initial Price: ${args.initial_price:.2f}")
    print(f"Quality Score: {args.quality_score:.2f}")
    print(f"Demand Scale: {args.demand_scale:.2f}")
    print()
    
    # Generate 3D surface
    time_grid, price_grid, prob_grid = generate_3d_surface_with_env(
        model_path=args.model_path,
        initial_price=args.initial_price,
        quality_score=args.quality_score,
        time_range=(0.0, 2000.0),  # From event time to far future
        price_range=(50.0, 500.0),  # Wide price range to show discount effects
        n_time_points=60,
        n_price_points=60,
        demand_scale=args.demand_scale
    )
    
    # Plot 3D surface
    plot_3d_surface(
        time_grid, price_grid, prob_grid,
        save_path=args.output_dir / 'demand_3d_price_prob_time_env.png'
    )
    
    # Plot 2D slices
    plot_2d_slices(
        time_grid, price_grid, prob_grid,
        save_path=args.output_dir / 'demand_2d_slices_by_time_env.png'
    )
    
    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"\nKey observations:")
    print(f"- Low prices (fire sales) should show HIGH probability near event time")
    print(f"- High prices should show LOW probability near event time (price gouging penalty)")
    print(f"- The 48-hour penalty is now price-dependent (cheap tickets less penalized)")


if __name__ == '__main__':
    main()

