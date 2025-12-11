"""
Generate additional graphs for research paper from training metrics.

Creates publication-quality plots from saved training_metrics.json file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import matplotlib.patches as mpatches

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_metrics(metrics_path: Path) -> Dict:
    """Load training metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_reward_distribution_by_phase(metrics: Dict, save_path: Path):
    """Plot reward distribution across training phases."""
    rewards = np.array(metrics['episode_rewards'])
    n = len(rewards)
    
    if n < 100:
        print("Not enough data for phase analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Reward Distribution by Training Phase', fontsize=16, fontweight='bold')
    
    # Split into phases
    early = rewards[:n//3]
    mid = rewards[n//3:2*n//3]
    late = rewards[2*n//3:]
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(early, bins=50, alpha=0.6, label=f'Early Phase ({len(early)} eps)', color='#e74c3c', density=True)
    ax1.hist(mid, bins=50, alpha=0.6, label=f'Mid Phase ({len(mid)} eps)', color='#f39c12', density=True)
    ax1.hist(late, bins=50, alpha=0.6, label=f'Late Phase ({len(late)} eps)', color='#27ae60', density=True)
    ax1.set_xlabel('Episode Reward (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Reward Distribution Histogram', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax2 = axes[1]
    data_to_plot = [early, mid, late]
    bp = ax2.boxplot(data_to_plot, labels=['Early', 'Mid', 'Late'], patch_artist=True)
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Episode Reward (%)', fontsize=12)
    ax2.set_title('Reward Distribution Box Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_sell_through_analysis(metrics: Dict, save_path: Path):
    """Analyze sell-through rate (episodes ending in sale vs expiration)."""
    rewards = np.array(metrics['episode_rewards'])
    
    # Episodes that sold (reward > -1) vs expired (reward == -1)
    sold = (rewards > -1.0).astype(int)
    expired = (rewards == -1.0).astype(int)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Sell-Through Analysis', fontsize=16, fontweight='bold')
    
    # Overall statistics
    ax1 = axes[0]
    sell_rate = np.mean(sold) * 100
    expire_rate = np.mean(expired) * 100
    bars = ax1.bar(['Sold', 'Expired'], [sell_rate, expire_rate], 
                   color=['#27ae60', '#e74c3c'], alpha=0.7)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title(f'Overall Sell-Through Rate: {sell_rate:.1f}%', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Sell-through over time (rolling window)
    ax2 = axes[1]
    window = 500
    if len(sold) >= window:
        rolling_sell_rate = []
        episodes = []
        for i in range(window-1, len(sold)):
            window_sold = np.mean(sold[i-window+1:i+1]) * 100
            rolling_sell_rate.append(window_sold)
            episodes.append(i+1)
        
        ax2.plot(episodes, rolling_sell_rate, color='#27ae60', linewidth=2)
        ax2.axhline(y=sell_rate, color='black', linestyle='--', alpha=0.5, label=f'Overall: {sell_rate:.1f}%')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Sell-Through Rate (%)', fontsize=12)
        ax2.set_title(f'Rolling Sell-Through Rate (window={window})', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Reward distribution: sold vs expired
    ax3 = axes[2]
    sold_rewards = rewards[sold == 1]
    expired_rewards = rewards[expired == 1]
    
    if len(sold_rewards) > 0 and len(expired_rewards) > 0:
        ax3.hist(sold_rewards, bins=50, alpha=0.6, label=f'Sold ({len(sold_rewards)})', 
                color='#27ae60', density=True)
        ax3.hist(expired_rewards, bins=50, alpha=0.6, label=f'Expired ({len(expired_rewards)})', 
                color='#e74c3c', density=True)
        ax3.set_xlabel('Episode Reward (%)', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Reward Distribution: Sold vs Expired', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_episode_length_analysis(metrics: Dict, save_path: Path):
    """Analyze episode length patterns."""
    lengths = np.array(metrics['episode_lengths'])
    rewards = np.array(metrics['episode_rewards'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Episode Length Analysis', fontsize=16, fontweight='bold')
    
    # Length distribution
    ax1 = axes[0, 0]
    ax1.hist(lengths, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(lengths):.1f}')
    ax1.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(lengths):.1f}')
    ax1.set_xlabel('Episode Length (steps)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Episode Length Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Length over time
    ax2 = axes[0, 1]
    window = 200
    if len(lengths) >= window:
        rolling_avg = []
        episodes = []
        for i in range(window-1, len(lengths)):
            rolling_avg.append(np.mean(lengths[i-window+1:i+1]))
            episodes.append(i+1)
        ax2.plot(episodes, rolling_avg, color='#3498db', linewidth=2, label=f'Rolling Avg (window={window})')
        ax2.axhline(np.mean(lengths), color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {np.mean(lengths):.1f}')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Average Length (steps)', fontsize=12)
        ax2.set_title('Episode Length Over Training', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Length vs Reward scatter
    ax3 = axes[1, 0]
    # Sample for visualization if too many points
    if len(lengths) > 5000:
        indices = np.random.choice(len(lengths), 5000, replace=False)
        sample_lengths = lengths[indices]
        sample_rewards = rewards[indices]
    else:
        sample_lengths = lengths
        sample_rewards = rewards
    
    scatter = ax3.scatter(sample_lengths, sample_rewards, alpha=0.3, s=10, c=sample_rewards, 
                         cmap='RdYlGn', vmin=-1, vmax=2)
    ax3.set_xlabel('Episode Length (steps)', fontsize=12)
    ax3.set_ylabel('Episode Reward (%)', fontsize=12)
    ax3.set_title('Length vs Reward Relationship', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Reward (%)')
    
    # Length by outcome (sold vs expired)
    ax4 = axes[1, 1]
    sold_mask = rewards > -1.0
    sold_lengths = lengths[sold_mask]
    expired_lengths = lengths[~sold_mask]
    
    if len(sold_lengths) > 0 and len(expired_lengths) > 0:
        ax4.hist(sold_lengths, bins=30, alpha=0.6, label=f'Sold (mean={np.mean(sold_lengths):.1f})', 
                color='#27ae60', density=True)
        ax4.hist(expired_lengths, bins=30, alpha=0.6, label=f'Expired (mean={np.mean(expired_lengths):.1f})', 
                color='#e74c3c', density=True)
        ax4.set_xlabel('Episode Length (steps)', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Length Distribution: Sold vs Expired', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_loss_analysis(metrics: Dict, save_path: Path):
    """Analyze training loss patterns."""
    losses = [l for l in metrics['episode_losses'] if l is not None]
    
    if len(losses) == 0:
        print("No loss data available")
        return
    
    losses = np.array(losses)
    loss_episodes = np.arange(len(losses))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Analysis', fontsize=16, fontweight='bold')
    
    # Loss over time
    ax1 = axes[0, 0]
    ax1.plot(loss_episodes, losses, alpha=0.5, color='orange', linewidth=0.5, label='Episode Loss')
    window = min(100, len(losses) // 10)
    if len(losses) >= window:
        rolling_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        rolling_episodes = loss_episodes[window-1:]
        ax1.plot(rolling_episodes, rolling_avg, color='red', linewidth=2, 
                label=f'Rolling Avg (window={window})')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss distribution
    ax2 = axes[0, 1]
    ax2.hist(losses, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(losses):.4f}')
    ax2.axvline(np.median(losses), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(losses):.4f}')
    ax2.set_xlabel('Loss', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Loss Distribution', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Loss by training phase
    ax3 = axes[1, 0]
    n = len(losses)
    if n >= 3:
        early_losses = losses[:n//3]
        mid_losses = losses[n//3:2*n//3]
        late_losses = losses[2*n//3:]
        
        ax3.hist(early_losses, bins=30, alpha=0.6, label=f'Early (mean={np.mean(early_losses):.4f})', 
                color='#e74c3c', density=True)
        ax3.hist(mid_losses, bins=30, alpha=0.6, label=f'Mid (mean={np.mean(mid_losses):.4f})', 
                color='#f39c12', density=True)
        ax3.hist(late_losses, bins=30, alpha=0.6, label=f'Late (mean={np.mean(late_losses):.4f})', 
                color='#27ae60', density=True)
        ax3.set_xlabel('Loss', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Loss Distribution by Training Phase', fontsize=13, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Loss statistics
    ax4 = axes[1, 1]
    stats = {
        'Mean': np.mean(losses),
        'Std': np.std(losses),
        'Min': np.min(losses),
        'Max': np.max(losses),
        'Median': np.median(losses)
    }
    bars = ax4.bar(stats.keys(), stats.values(), color=['blue', 'orange', 'red', 'green', 'purple'], alpha=0.7)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Loss Statistics', fontsize=13, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, (key, val) in zip(bars, stats.items()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_epsilon_analysis(metrics: Dict, save_path: Path):
    """Analyze epsilon decay and exploration."""
    epsilons = np.array(metrics['epsilon_values'])
    episodes = np.arange(1, len(epsilons) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Exploration-Exploitation Analysis', fontsize=16, fontweight='bold')
    
    # Epsilon decay
    ax1 = axes[0]
    ax1.plot(episodes, epsilons, color='#9b59b6', linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax1.set_title('Epsilon Decay Schedule', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3)
    
    # Exploration vs exploitation phases
    ax2 = axes[1]
    exploration_threshold = 0.5
    exploration_phase = epsilons > exploration_threshold
    exploitation_phase = epsilons <= exploration_threshold
    
    ax2.fill_between(episodes, 0, epsilons, where=exploration_phase, 
                     alpha=0.3, color='red', label='Exploration Phase')
    ax2.fill_between(episodes, 0, epsilons, where=exploitation_phase, 
                     alpha=0.3, color='green', label='Exploitation Phase')
    ax2.plot(episodes, epsilons, color='#9b59b6', linewidth=2)
    ax2.axhline(exploration_threshold, color='black', linestyle='--', alpha=0.5, 
                label=f'Threshold: {exploration_threshold}')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Epsilon', fontsize=12)
    ax2.set_title('Exploration vs Exploitation Phases', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_metrics(metrics: Dict, save_path: Path):
    """Plot convergence metrics."""
    rewards = np.array(metrics['episode_rewards'])
    episodes = np.arange(1, len(rewards) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Rolling mean and std
    ax1 = axes[0, 0]
    window = 500
    if len(rewards) >= window:
        rolling_mean = []
        rolling_std = []
        rolling_episodes = []
        for i in range(window-1, len(rewards)):
            window_rewards = rewards[i-window+1:i+1]
            rolling_mean.append(np.mean(window_rewards))
            rolling_std.append(np.std(window_rewards))
            rolling_episodes.append(episodes[i])
        
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(rolling_episodes, rolling_mean, color='blue', linewidth=2, label='Mean Reward')
        line2 = ax1_twin.plot(rolling_episodes, rolling_std, color='red', linewidth=2, 
                             linestyle='--', label='Std Dev')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Mean Reward (%)', fontsize=12, color='blue')
        ax1_twin.set_ylabel('Standard Deviation', fontsize=12, color='red')
        ax1.set_title('Reward Stability Over Time', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.grid(True, alpha=0.3)
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    # Convergence rate (improvement over time)
    ax2 = axes[0, 1]
    if len(rewards) >= 100:
        window = 1000
        improvement = []
        improvement_episodes = []
        for i in range(window, len(rewards)):
            recent = np.mean(rewards[i-window:i])
            previous = np.mean(rewards[i-2*window:i-window]) if i >= 2*window else np.mean(rewards[:i-window])
            improvement.append(recent - previous)
            improvement_episodes.append(episodes[i])
        
        if len(improvement) > 0:
            ax2.plot(improvement_episodes, improvement, color='green', linewidth=2)
            ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_ylabel('Reward Improvement (%)', fontsize=12)
            ax2.set_title('Learning Rate (Reward Improvement)', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
    
    # Reward percentiles
    ax3 = axes[1, 0]
    if len(rewards) >= 100:
        window = 500
        percentiles = []
        percentile_episodes = []
        for i in range(window-1, len(rewards)):
            window_rewards = rewards[i-window+1:i+1]
            percentiles.append({
                'p25': np.percentile(window_rewards, 25),
                'p50': np.percentile(window_rewards, 50),
                'p75': np.percentile(window_rewards, 75)
            })
            percentile_episodes.append(episodes[i])
        
        if len(percentiles) > 0:
            p25 = [p['p25'] for p in percentiles]
            p50 = [p['p50'] for p in percentiles]
            p75 = [p['p75'] for p in percentiles]
            
            ax3.fill_between(percentile_episodes, p25, p75, alpha=0.3, color='blue', label='IQR')
            ax3.plot(percentile_episodes, p50, color='red', linewidth=2, label='Median')
            ax3.set_xlabel('Episode', fontsize=12)
            ax3.set_ylabel('Reward (%)', fontsize=12)
            ax3.set_title('Reward Percentiles Over Time', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
    
    # Final performance metrics
    ax4 = axes[1, 1]
    if len(rewards) >= 1000:
        n = len(rewards)
        first_10pct = rewards[:n//10]
        last_10pct = rewards[-n//10:]
        
        comparison_data = {
            'First 10%': [np.mean(first_10pct), np.std(first_10pct)],
            'Last 10%': [np.mean(last_10pct), np.std(last_10pct)]
        }
        
        x = np.arange(len(comparison_data))
        means = [comparison_data[k][0] for k in comparison_data.keys()]
        stds = [comparison_data[k][1] for k in comparison_data.keys()]
        
        bars = ax4.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                      color=['#e74c3c', '#27ae60'], edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels(comparison_data.keys())
        ax4.set_ylabel('Reward (%)', fontsize=12)
        ax4.set_title('Early vs Late Training Performance', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate paper graphs from training metrics")
    parser.add_argument(
        '--metrics-path',
        type=Path,
        default=None,
        help='Path to training_metrics.json file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save generated graphs (default: plots/paper_graphs)'
    )
    
    args = parser.parse_args()
    
    # Get the script's directory and construct paths relative to it
    script_dir = Path(__file__).parent
    
    # Find metrics file
    possible_paths = [
        script_dir / 'plots' / 'training_metrics.json',
        Path('learning_environment/plots/training_metrics.json'),
        Path('plots/training_metrics.json'),
        script_dir.parent / 'learning_environment' / 'plots' / 'training_metrics.json',
    ]
    
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
    else:
        metrics_path = None
        for path in possible_paths:
            if path.exists():
                metrics_path = path
                break
    
    if metrics_path is None or not metrics_path.exists():
        print(f"Error: Metrics file not found. Tried:")
        for path in possible_paths:
            print(f"  - {path}")
        print(f"\nCurrent working directory: {Path.cwd()}")
        print(f"Script directory: {script_dir}")
        print(f"\nPlease specify the path using --metrics-path")
        return 1
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = metrics_path.parent / 'paper_graphs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)
    
    print(f"Found {len(metrics['episode_rewards'])} episodes")
    print(f"Generating graphs in: {output_dir}")
    
    # Generate all graphs
    plot_reward_distribution_by_phase(metrics, output_dir / 'reward_distribution_by_phase.png')
    plot_sell_through_analysis(metrics, output_dir / 'sell_through_analysis.png')
    plot_episode_length_analysis(metrics, output_dir / 'episode_length_analysis.png')
    plot_loss_analysis(metrics, output_dir / 'loss_analysis.png')
    plot_epsilon_analysis(metrics, output_dir / 'epsilon_analysis.png')
    plot_convergence_metrics(metrics, output_dir / 'convergence_metrics.png')
    
    print("\n✓ All graphs generated successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

