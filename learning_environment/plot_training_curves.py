"""
Generate RL Paper Graphs from Training Data

Creates publication-quality plots for RL papers:
- Learning curves (rewards over episodes)
- Loss curves
- Epsilon decay
- Convergence analysis
- Performance metrics

Usage:
    uv run python plot_training_curves.py --checkpoint checkpoints/dqn_ticket_pricing.pt
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.train_dqn.trainer import DQNTrainer


# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_training_data(checkpoint_path: Path) -> dict:
    """Load training data from checkpoint and metadata."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.parent / 'episode_metadata.json'
    
    # Try to load from trainer if available
    # For now, we'll need to re-train or load from saved metrics
    # This is a placeholder - you may need to save metrics during training
    return {
        'episode_count': 0,
        'checkpoint_path': checkpoint_path
    }


def plot_learning_curves(trainer: DQNTrainer, save_dir: Path):
    """Plot learning curves: rewards, loss, epsilon."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN Training Curves', fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Episode Rewards with Rolling Average
    ax1 = axes[0, 0]
    if len(trainer.episode_rewards) > 0:
        episodes = np.arange(1, len(trainer.episode_rewards) + 1)
        
        # Raw rewards (transparent)
        ax1.plot(episodes, trainer.episode_rewards, alpha=0.2, color='blue', linewidth=0.5, label='Episode Reward')
        
        # Rolling averages with different window sizes
        window_sizes = [100, 1000]
        colors = ['red', 'darkred']
        for window, color in zip(window_sizes, colors):
            if len(trainer.episode_rewards) >= window:
                rolling_avg = np.convolve(
                    trainer.episode_rewards, 
                    np.ones(window) / window, 
                    mode='valid'
                )
                rolling_episodes = episodes[window-1:]
                ax1.plot(rolling_episodes, rolling_avg, color=color, linewidth=2, 
                        label=f'Rolling Avg ({window})')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward (%)', fontsize=12)
        ax1.set_title('Learning Curve: Episode Rewards', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add convergence indicator
        if len(trainer.episode_rewards) > 1000:
            recent_avg = np.mean(trainer.episode_rewards[-1000:])
            ax1.axhline(y=recent_avg, color='green', linestyle='--', alpha=0.5, 
                       label=f'Recent Avg: {recent_avg:.2f}%')
    
    # 2. Training Loss
    ax2 = axes[0, 1]
    valid_losses = [(i, loss) for i, loss in enumerate(trainer.episode_losses) if loss is not None]
    if valid_losses:
        loss_episodes, losses = zip(*valid_losses)
        ax2.plot(loss_episodes, losses, alpha=0.5, color='orange', linewidth=1, label='Episode Avg Loss')
        
        # Rolling average
        if len(losses) > 100:
            window = min(100, len(losses))
            loss_rolling = np.convolve(losses, np.ones(window) / window, mode='valid')
            rolling_episodes = list(loss_episodes)[window-1:]
            ax2.plot(rolling_episodes, loss_rolling, color='red', linewidth=2, 
                    label=f'Rolling Avg ({window})')
        
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No loss data available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
    
    # 3. Epsilon Decay
    ax3 = axes[1, 0]
    if len(trainer.epsilon_values) > 0:
        episodes = np.arange(1, len(trainer.epsilon_values) + 1)
        ax3.plot(episodes, trainer.epsilon_values, color='green', linewidth=2)
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
        ax3.set_title('Exploration-Exploitation Trade-off', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.1])
    
    # 4. Episode Lengths
    ax4 = axes[1, 1]
    if len(trainer.episode_lengths) > 0:
        episodes = np.arange(1, len(trainer.episode_lengths) + 1)
        ax4.plot(episodes, trainer.episode_lengths, alpha=0.3, color='purple', linewidth=0.5, label='Episode Length')
        
        # Rolling average
        if len(trainer.episode_lengths) > 100:
            window = min(100, len(trainer.episode_lengths))
            rolling_avg = np.convolve(
                trainer.episode_lengths, 
                np.ones(window) / window, 
                mode='valid'
            )
            rolling_episodes = episodes[window-1:]
            ax4.plot(rolling_episodes, rolling_avg, color='red', linewidth=2, 
                    label=f'Rolling Avg ({window})')
        
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Steps per Episode', fontsize=12)
        ax4.set_title('Episode Lengths', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_analysis(trainer: DQNTrainer, save_dir: Path):
    """Plot convergence analysis: reward stability over time."""
    if len(trainer.episode_rewards) < 100:
        print("Not enough data for convergence analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Analysis', fontsize=18, fontweight='bold')
    
    # 1. Reward variance over time (rolling window)
    ax1 = axes[0]
    window = 100
    episodes = np.arange(1, len(trainer.episode_rewards) + 1)
    
    if len(trainer.episode_rewards) >= window:
        rolling_std = []
        rolling_mean = []
        rolling_episodes = []
        
        for i in range(window - 1, len(trainer.episode_rewards)):
            window_rewards = trainer.episode_rewards[i - window + 1:i + 1]
            rolling_std.append(np.std(window_rewards))
            rolling_mean.append(np.mean(window_rewards))
            rolling_episodes.append(episodes[i])
        
        ax1_twin = ax1.twinx()
        ax1.plot(rolling_episodes, rolling_mean, color='blue', linewidth=2, label='Mean Reward')
        ax1_twin.plot(rolling_episodes, rolling_std, color='red', linewidth=2, linestyle='--', label='Std Dev')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Mean Reward (%)', fontsize=12, color='blue')
        ax1_twin.set_ylabel('Standard Deviation', fontsize=12, color='red')
        ax1.set_title('Reward Stability Over Time', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)
        ax1_twin.legend(loc='upper right', fontsize=10)
    
    # 2. Reward distribution by training phase
    ax2 = axes[1]
    if len(trainer.episode_rewards) >= 1000:
        # Split into phases
        total = len(trainer.episode_rewards)
        early = trainer.episode_rewards[:total//3]
        mid = trainer.episode_rewards[total//3:2*total//3]
        late = trainer.episode_rewards[2*total//3:]
        
        ax2.hist(early, bins=50, alpha=0.5, label=f'Early ({len(early)} eps)', color='red')
        ax2.hist(mid, bins=50, alpha=0.5, label=f'Mid ({len(mid)} eps)', color='orange')
        ax2.hist(late, bins=50, alpha=0.5, label=f'Late ({len(late)} eps)', color='green')
        
        ax2.set_xlabel('Reward (%)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Reward Distribution by Training Phase', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / 'convergence_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_performance_metrics(trainer: DQNTrainer, save_dir: Path):
    """Plot performance metrics summary."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Metrics Summary', fontsize=18, fontweight='bold')
    
    if len(trainer.episode_rewards) == 0:
        print("No training data available")
        return
    
    # 1. Reward statistics
    ax1 = axes[0]
    rewards = np.array(trainer.episode_rewards)
    stats = {
        'Mean': np.mean(rewards),
        'Std': np.std(rewards),
        'Min': np.min(rewards),
        'Max': np.max(rewards),
        'Median': np.median(rewards)
    }
    
    bars = ax1.bar(stats.keys(), stats.values(), color=['blue', 'orange', 'red', 'green', 'purple'])
    ax1.set_ylabel('Reward (%)', fontsize=12)
    ax1.set_title('Reward Statistics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, (key, val) in zip(bars, stats.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. Learning progress (first 10% vs last 10%)
    ax2 = axes[1]
    if len(rewards) >= 100:
        n = len(rewards)
        first_10pct = rewards[:n//10]
        last_10pct = rewards[-n//10:]
        
        comparison = {
            'First 10%': np.mean(first_10pct),
            'Last 10%': np.mean(last_10pct),
            'Improvement': np.mean(last_10pct) - np.mean(first_10pct)
        }
        
        bars = ax2.bar(comparison.keys(), comparison.values(), 
                      color=['red', 'green', 'blue'])
        ax2.set_ylabel('Reward (%)', fontsize=12)
        ax2.set_title('Learning Progress', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, (key, val) in zip(bars, comparison.items()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. Episode length statistics
    ax3 = axes[2]
    if len(trainer.episode_lengths) > 0:
        lengths = np.array(trainer.episode_lengths)
        length_stats = {
            'Mean': np.mean(lengths),
            'Std': np.std(lengths),
            'Min': np.min(lengths),
            'Max': np.max(lengths)
        }
        
        bars = ax3.bar(length_stats.keys(), length_stats.values(), 
                      color=['blue', 'orange', 'red', 'green'])
        ax3.set_ylabel('Steps', fontsize=12)
        ax3.set_title('Episode Length Statistics', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, (key, val) in zip(bars, length_stats.items()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = save_dir / 'performance_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate RL paper graphs from training data")
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/dqn_ticket_pricing.pt'),
        help='Path to agent checkpoint'
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/demand_model_v12_anti_gouge.pkl'),
        help='Path to demand model'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('plots'),
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    learning_env_dir = Path(__file__).parent
    checkpoint_path = learning_env_dir / args.checkpoint if not args.checkpoint.is_absolute() else args.checkpoint
    model_path = learning_env_dir / args.model_path if not args.model_path.is_absolute() else args.model_path
    output_dir = learning_env_dir / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    print(f"Loading trainer from checkpoint: {checkpoint_path}")
    
    # Create trainer (this will load the checkpoint)
    trainer = DQNTrainer(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        load_checkpoint=True
    )
    
    # Try to load training metrics from file
    metrics_loaded = trainer.load_training_metrics()
    
    # Check if we have training data
    if len(trainer.episode_rewards) == 0:
        print("Warning: No training data found.")
        print("Make sure you've run training with train_headless.py which saves metrics.")
        print(f"Looking for metrics at: {trainer.plots_dir / 'training_metrics.json'}")
        return 1
    
    print(f"Found {len(trainer.episode_rewards)} episodes of training data")
    print(f"Generating plots in: {output_dir}")
    
    # Generate plots
    plot_learning_curves(trainer, output_dir)
    plot_convergence_analysis(trainer, output_dir)
    plot_performance_metrics(trainer, output_dir)
    
    print("\n✓ All plots generated successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

