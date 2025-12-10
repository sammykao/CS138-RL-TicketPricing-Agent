"""
DQN Training Class for Ticket Pricing Agent

Provides a class-based interface for training DQN agents with metrics tracking,
checkpointing, and visualization.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.dqn_agent import DQNAgent
from env.ticket_pricing_env import TicketPricingEnv


class DQNTrainer:
    """
    Trainer class for DQN agent on ticket pricing environment.
    
    Handles training loop, metrics tracking, checkpointing, and visualization.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        plots_dir: Optional[Path] = None,
        env_config: Optional[Dict] = None,
        agent_config: Optional[Dict] = None,
        load_checkpoint: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model_path: Path to demand model .pkl file
            checkpoint_path: Path to save/load agent checkpoints
            plots_dir: Directory to save training plots
            env_config: Environment configuration dict
            agent_config: Agent configuration dict
            load_checkpoint: Whether to load existing checkpoint if available
        """
        # Get paths relative to learning_environment directory
        learning_env_dir = Path(__file__).parent.parent.parent
        
        # Set default paths
        if model_path is None:
            model_path = learning_env_dir / 'models' / 'demand_model_v12_anti_gouge.pkl'
        if checkpoint_path is None:
            checkpoint_path = learning_env_dir / 'checkpoints' / 'dqn_ticket_pricing.pt'
        if plots_dir is None:
            plots_dir = learning_env_dir / 'plots'
        
        self.model_path = Path(model_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Default environment config
        default_env_config = {
            'demand_model_path': self.model_path,
            'initial_price_range': (100.0, 500.0),
            'quality_range': (0.0, 1.0),
            'time_horizon': 2000.0,
            'time_step': 6.0,
            'price_bounds': (0.3, 3.0),
            'demand_scale': 0.5,
            'max_probability': 0.95,
            'random_seed': 42
        }
        if env_config:
            default_env_config.update(env_config)
        env_config = default_env_config
        
        # Create environment
        self.env = TicketPricingEnv(**env_config)
        
        # Default agent config
        default_agent_config = {
            'hidden_dim': 128,
            'gamma': 0.99,
            'lr': 1e-3,
            'batch_size': 64,
            'buffer_size': 50_000,
            'min_buffer_size': 1_000,
            'target_update_freq': 1000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 50_000,
        }
        if agent_config:
            default_agent_config.update(agent_config)
        agent_config = default_agent_config
        
        # Create or load agent
        if load_checkpoint and self.checkpoint_path.exists():
            self.agent = DQNAgent.load(self.env, self.checkpoint_path)
        else:
            self.agent = DQNAgent(env=self.env, **agent_config)
        
        # Initialize tracking variables
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[Optional[float]] = []
        self.epsilon_values: List[float] = []
        self.steps_history: List[int] = []
        
        # Rolling averages
        self.reward_window = deque(maxlen=100)
        self.loss_window = deque(maxlen=1000)
        
        # Training state
        self.current_episode = 0
        self.is_training = False
        self.is_paused = False
    
    def train(
        self,
        n_episodes: int,
        print_freq: int = 50,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        Train the agent for specified number of episodes.
        
        Args:
            n_episodes: Number of episodes to train
            print_freq: Print metrics every N episodes (0 to disable)
            callback: Optional callback function called after each episode
                     with signature: callback(episode, metrics_dict)
        
        Returns:
            Dictionary with training metrics
        """
        self.is_training = True
        self.is_paused = False
        
        # Ensure environment is in a clean state
        self.env.reset()
        
        start_episode = self.current_episode
        end_episode = start_episode + n_episodes
        
        for episode in range(start_episode, end_episode):
            # Check for pause
            while self.is_paused and self.is_training:
                import time
                time.sleep(0.1)
            
            if not self.is_training:
                break
            
            self.current_episode = episode
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            episode_losses_list = []
            self.agent.last_loss = None  # Reset loss tracking for new episode
            
            while not done:
                action = self.agent.select_action(np.asarray(obs, dtype=float))
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Observe and potentially train
                self.agent.observe(
                    np.asarray(obs, dtype=float),
                    action,
                    float(reward),
                    np.asarray(next_obs, dtype=float),
                    bool(terminated),
                    bool(truncated),
                    info,
                )
                
                # Track loss if training occurred
                if self.agent.last_loss is not None:
                    episode_losses_list.append(self.agent.last_loss)
                    self.loss_window.append(self.agent.last_loss)
                
                total_reward += float(reward)
                steps += 1
                obs = next_obs
                done = terminated or truncated
            
            # Record episode metrics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.reward_window.append(total_reward)
            self.epsilon_values.append(self.agent.epsilon())
            self.steps_history.append(self.agent.total_steps)
            
            avg_loss = np.mean(episode_losses_list) if episode_losses_list else None
            if avg_loss is not None:
                self.episode_losses.append(avg_loss)
            else:
                self.episode_losses.append(None)
            
            # Print metrics periodically
            if print_freq > 0 and ((episode + 1) % print_freq == 0 or episode == start_episode):
                self._print_metrics(episode + 1, end_episode, total_reward, steps, info)
            
            # Call callback if provided
            if callback:
                metrics = self.get_current_metrics()
                callback(episode + 1, metrics)
        
        self.is_training = False
        
        # Return final metrics
        return self.get_current_metrics()
    
    def _print_metrics(
        self,
        episode: int,
        total_episodes: int,
        total_reward: float,
        steps: int,
        info: Dict
    ) -> None:
        """Print training metrics."""
        avg_reward = np.mean(list(self.reward_window)) if self.reward_window else 0.0
        avg_loss_val = np.mean(list(self.loss_window)) if self.loss_window else None
        current_epsilon = self.agent.epsilon()
        
        print(f"\nEpisode {episode}/{total_episodes}")
        print(f"  Reward: {total_reward:7.2%} | Avg (last 100): {avg_reward:7.2%}")
        print(f"  Steps: {steps:4d} | Total steps: {self.agent.total_steps:6d}")
        print(f"  Epsilon: {current_epsilon:.4f} | Buffer: {len(self.agent.replay_buffer):5d}/{self.agent.buffer_size}")
        if avg_loss_val is not None:
            print(f"  Avg Loss: {avg_loss_val:.6f}")
        if info.get('sold'):
            price_pct = ((info.get('current_price', 0) - info.get('initial_price', 0)) / info.get('initial_price', 1)) * 100
            print(f"  Ticket sold at ${info.get('current_price', 0):.2f} ({price_pct:+.1f}% from initial ${info.get('initial_price', 0):.2f})")
        else:
            print(f"  Ticket not sold (time expired)")
    
    def get_current_metrics(self) -> Dict:
        """
        Get current training metrics.
        
        Returns:
            Dictionary with current metrics
        """
        avg_reward = np.mean(list(self.reward_window)) if self.reward_window else 0.0
        avg_loss = np.mean(list(self.loss_window)) if self.loss_window else None
        
        return {
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'episode_losses': self.episode_losses.copy(),
            'epsilon_values': self.epsilon_values.copy(),
            'steps_history': self.steps_history.copy(),
            'current_episode': self.current_episode,
            'total_steps': self.agent.total_steps,
            'current_epsilon': self.agent.epsilon(),
            'avg_reward_last_100': avg_reward,
            'avg_loss': avg_loss,
            'buffer_size': len(self.agent.replay_buffer),
            'buffer_capacity': self.agent.buffer_size,
        }
    
    def evaluate(self, n_episodes: int = 100) -> Dict:
        """
        Evaluate agent performance.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics dictionary
        """
        return self.agent.evaluate(n_episodes)
    
    def save_checkpoint(self) -> None:
        """Save agent checkpoint."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(self.checkpoint_path)
    
    def save_training_metrics(self, filepath: Optional[Path] = None) -> None:
        """
        Save training metrics to file for later analysis.
        
        Args:
            filepath: Path to save metrics (default: plots_dir/training_metrics.json)
        """
        if filepath is None:
            filepath = self.plots_dir / 'training_metrics.json'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': [l for l in self.episode_losses if l is not None],
            'epsilon_values': self.epsilon_values,
            'steps_history': self.steps_history,
            'current_episode': self.current_episode,
            'total_steps': self.agent.total_steps,
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Training metrics saved to {filepath}")
    
    def load_training_metrics(self, filepath: Optional[Path] = None) -> bool:
        """
        Load training metrics from file.
        
        Args:
            filepath: Path to load metrics from (default: plots_dir/training_metrics.json)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if filepath is None:
            filepath = self.plots_dir / 'training_metrics.json'
        
        filepath = Path(filepath)
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                metrics = json.load(f)
            
            self.episode_rewards = metrics.get('episode_rewards', [])
            self.episode_lengths = metrics.get('episode_lengths', [])
            self.epsilon_values = metrics.get('epsilon_values', [])
            self.steps_history = metrics.get('steps_history', [])
            self.current_episode = metrics.get('current_episode', 0)
            
            # Reconstruct loss list (may have None values)
            episode_losses = metrics.get('episode_losses', [])
            # Note: This is a simplified version - full reconstruction would need episode indices
            self.episode_losses = episode_losses
            
            print(f"Training metrics loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Warning: Could not load training metrics: {e}")
            return False
    
    def plot_metrics(self, save_path: Optional[Path] = None) -> None:
        """
        Generate and save training plots.
        
        Args:
            save_path: Path to save plot (default: plots_dir/dqn_training_metrics.png)
        """
        if save_path is None:
            save_path = self.plots_dir / 'dqn_training_metrics.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        if len(self.episode_rewards) > 0:
            ax1.plot(self.episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
            if len(self.reward_window) > 0:
                window_size = min(100, len(self.episode_rewards))
                rolling_avg = [np.mean(self.episode_rewards[max(0, i-window_size+1):i+1]) 
                             for i in range(len(self.episode_rewards))]
                ax1.plot(rolling_avg, color='red', linewidth=2, label=f'Rolling Avg ({window_size})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (%)')
        ax1.set_title('Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Epsilon Decay
        ax2 = axes[0, 1]
        if len(self.epsilon_values) > 0:
            ax2.plot(self.epsilon_values, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Epsilon Decay (Exploration Rate)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training Loss
        ax3 = axes[1, 0]
        valid_losses = [(i, loss) for i, loss in enumerate(self.episode_losses) if loss is not None]
        if valid_losses:
            loss_episodes, losses = zip(*valid_losses)
            ax3.plot(loss_episodes, losses, alpha=0.5, color='orange', label='Episode Avg Loss')
            if len(self.loss_window) > 0:
                window_size = min(100, len(valid_losses))
                loss_rolling = []
                for i in range(len(valid_losses)):
                    window_losses = [l for _, l in valid_losses[max(0, i-window_size+1):i+1]]
                    if window_losses:
                        loss_rolling.append(np.mean(window_losses))
                if loss_rolling:
                    ax3.plot([e for e, _ in valid_losses[:len(loss_rolling)]], 
                            loss_rolling, color='red', linewidth=2, label=f'Rolling Avg ({window_size})')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'No loss data yet\n(Waiting for buffer to fill)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Training Loss')
        
        # Plot 4: Episode Lengths
        ax4 = axes[1, 1]
        if len(self.episode_lengths) > 0:
            ax4.plot(self.episode_lengths, alpha=0.5, color='purple', label='Episode Length')
            window_size = min(100, len(self.episode_lengths))
            rolling_length = [np.mean(self.episode_lengths[max(0, i-window_size+1):i+1]) 
                             for i in range(len(self.episode_lengths))]
            ax4.plot(rolling_length, color='red', linewidth=2, label=f'Rolling Avg ({window_size})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps')
        ax4.set_title('Episode Lengths')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def pause(self) -> None:
        """Pause training."""
        self.is_paused = True
    
    def resume(self) -> None:
        """Resume training."""
        self.is_paused = False
    
    def stop(self) -> None:
        """Stop training."""
        self.is_training = False
        self.is_paused = False

