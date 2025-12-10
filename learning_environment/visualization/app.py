"""
Main visualization app with event loop and mode switching.

Handles episode view mode and training view mode, connecting to
TicketPricingEnv and DQNAgent.
"""

import pygame
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import sys
import json
from collections import deque

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.ticket_pricing_env import TicketPricingEnv
from agents.dqn_agent import DQNAgent
from agents.base_agent import BaseAgent

from .ui_style import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, COLORS
from .renderer import Renderer
from .data_helpers import compute_demand_curve, BehaviorTracker


class VisualizationApp:
    """
    Main visualization application for RL ticket pricing agent.
    
    Training mode only: Shows cumulative metrics and graphs as agent learns.
    Graphs accumulate data across episodes to show learning trends.
    """
    
    def __init__(
        self,
        env: TicketPricingEnv,
        agent: BaseAgent,
        step_delay_ms: int = 100,
        checkpoint_path: Optional[Path] = None,
        save_interval_episodes: int = 5,
        target_episodes: int = 200_000,
        episode_metadata_path: Optional[Path] = None
    ):
        """
        Initialize visualization app (training mode only).
        
        Args:
            env: TicketPricingEnv instance
            agent: DQNAgent or other BaseAgent instance
            step_delay_ms: Delay between steps (milliseconds) - lower = faster training
            checkpoint_path: Path to save checkpoints (optional, saves automatically if provided)
            save_interval_episodes: Save checkpoint every N episodes (default: 5)
            target_episodes: Target number of episodes to train (default: 200,000)
            episode_metadata_path: Path to save episode metadata (auto-generated if None)
        """
        self.env = env
        self.agent = agent
        self.step_delay_ms = step_delay_ms
        self.checkpoint_path = checkpoint_path
        self.save_interval_episodes = save_interval_episodes
        self.target_episodes = target_episodes
        
        # Set up episode metadata path
        if episode_metadata_path is None:
            if checkpoint_path:
                self.episode_metadata_path = checkpoint_path.parent / 'episode_metadata.json'
            else:
                self.episode_metadata_path = Path(__file__).parent.parent / 'checkpoints' / 'episode_metadata.json'
        else:
            self.episode_metadata_path = Path(episode_metadata_path)
        
        # Load episode count from metadata if exists
        self.current_episode = self._load_episode_count()
        self.last_saved_episode = self.current_episode
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RL Ticket Pricing Agent - Live Visualization")
        self.clock = pygame.time.Clock()
        
        # Initialize renderer
        self.renderer = Renderer(self.screen)
        
        # Initialize behavior tracker
        self.behavior_tracker = BehaviorTracker(max_time_steps=200)
        
        # Episode state
        self.current_episode = 0
        self.current_step = 0
        self.obs: Optional[np.ndarray] = None
        self.info: Dict[str, Any] = {}
        self.episode_done = False
        
        # Training metrics (cumulative)
        self.episode_rewards_pct: deque = deque(maxlen=100)  # Reward as percentage
        self.episode_rewards_dollar: deque = deque(maxlen=100)  # Reward in dollars
        self.episode_sold: deque = deque(maxlen=100)  # Track if ticket was sold (1) or not (0)
        self.avg_reward_pct: float = 0.0
        self.avg_reward_dollar: float = 0.0
        self.sellout_rate: float = 0.0  # % of tickets sold
        
        # Current step data
        self.current_price: float = 0.0
        self.previous_price: float = 0.0
        self.initial_price: float = 0.0
        self.sale_price: Optional[float] = None
        self.reward: float = 0.0
        self.reward_pct: float = 0.0
        self.last_action: int = 0
        self.p_sale: float = 0.0
        self.sold: bool = False
        
        # Demand curve cache
        self.demand_prices: Optional[np.ndarray] = None
        self.demand_probs: Optional[np.ndarray] = None
        
        # Step timer
        self.last_step_time = 0
        
        # Running flag
        self.running = True
    
    def reset_episode(self):
        """Reset environment for new episode."""
        self.obs, self.info = self.env.reset()
        self.episode_done = False
        self.current_step = 0
        
        # Extract initial state
        self.initial_price = self.env.initial_price
        self.current_price = self.env.current_price
        self.previous_price = self.env.current_price  # Track for price change calculation
        self.sale_price = None
        self.reward = 0.0
        self.reward_pct = 0.0
        self.sold = False
        self.last_action = 0  # Default to no action
        self.p_sale = 0.0
        
        # Note: Don't reset behavior tracker - we want cumulative data
        
        # Compute initial demand curve
        self._update_demand_curve()
        
        # Update initial probability
        try:
            self.p_sale = self.env._compute_sale_probability()
        except:
            self.p_sale = 0.0
    
    def _update_demand_curve(self):
        """Update demand curve data."""
        try:
            prices, probs = compute_demand_curve(self.env)
            self.demand_prices = prices
            self.demand_probs = probs
            self.renderer.update_demand_curve(prices, probs)
        except Exception as e:
            print(f"Warning: Failed to compute demand curve: {e}")
            self.demand_prices = None
            self.demand_probs = None
    
    def step_episode(self, train: bool = True):
        """
        Execute one step in the episode.
        
        Args:
            train: Whether agent should learn from this step
        """
        if self.episode_done:
            return
        
        # Select action
        action = self.agent.select_action(self.obs)
        self.last_action = action
        
        # Step environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update state
        self.current_price = info['current_price']
        self.p_sale = info['p_sale']
        self.sold = info['sold']
        self.reward = reward
        
        # Compute price change % from previous step
        # At step 0, compare to initial_price; otherwise compare to previous_price
        if self.current_step == 0:
            if self.initial_price > 0:
                price_change_pct = ((self.current_price - self.initial_price) / self.initial_price) * 100.0
            else:
                price_change_pct = 0.0
        elif self.previous_price > 0:
            price_change_pct = ((self.current_price - self.previous_price) / self.previous_price) * 100.0
        else:
            price_change_pct = 0.0
        
        # Compute reward percentage
        if self.sold:
            self.sale_price = self.current_price
            self.reward_pct = (self.current_price - self.initial_price) / self.initial_price * 100.0
        elif terminated or truncated:
            if truncated:
                self.reward_pct = -100.0  # Lost entire ticket
            else:
                self.reward_pct = 0.0
        else:
            self.reward_pct = 0.0
        
        # Agent observes transition
        if train:
            self.agent.observe(
                self.obs,
                action,
                reward,
                next_obs,
                terminated,
                truncated,
                info
            )
        
        # Update behavior tracker (cumulative across episodes)
        self.behavior_tracker.add_step(
            price_change_pct=price_change_pct,
            reward=reward,
            step=self.current_step
        )
        
        # Update previous price for next step
        self.previous_price = self.current_price
        
        # Update observation
        self.obs = next_obs
        self.info = info
        self.episode_done = terminated or truncated
        self.current_step += 1
        
        # Trigger animations
        self.renderer.animation_state.start_action_pulse()
        if self.sold:
            self.renderer.animation_state.start_sale_celebration()
        elif self.episode_done and truncated:
            self.renderer.animation_state.start_failure_flash()
        
        if abs(self.reward_pct) > 1.0:  # Significant reward change
            self.renderer.animation_state.start_reward_flash(self.reward_pct)
        
        # Update demand curve periodically (every 5 steps or on sale)
        if self.current_step % 5 == 0 or self.sold:
            self._update_demand_curve()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset episode
                    self.current_episode += 1
                    self.reset_episode()
                
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def _load_episode_count(self) -> int:
        """Load episode count from metadata file."""
        if self.episode_metadata_path.exists():
            try:
                with open(self.episode_metadata_path, 'r') as f:
                    metadata = json.load(f)
                    episode_count = metadata.get('episode_count', 0)
                    print(f"Resuming from episode {episode_count:,} / {self.target_episodes:,}")
                    return episode_count
            except Exception as e:
                print(f"Warning: Could not load episode metadata: {e}")
        return 0
    
    def _save_episode_metadata(self):
        """Save episode count to metadata file."""
        try:
            self.episode_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                'episode_count': self.current_episode,
                'target_episodes': self.target_episodes,
                'last_saved': str(self.checkpoint_path) if self.checkpoint_path else None
            }
            with open(self.episode_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save episode metadata: {e}")
    
    def run_training_step(self):
        """Run one training step."""
        # Check if target reached
        if self.current_episode >= self.target_episodes:
            print(f"\n{'='*60}")
            print(f"TARGET REACHED: {self.target_episodes:,} episodes completed!")
            print(f"{'='*60}")
            # Save final checkpoint
            if self.checkpoint_path and hasattr(self.agent, 'save'):
                try:
                    self.agent.save(self.checkpoint_path)
                    self._save_episode_metadata()
                    print(f"Final checkpoint saved at episode {self.current_episode:,}")
                except Exception as e:
                    print(f"Warning: Failed to save final checkpoint: {e}")
            self.running = False
            return
        
        if self.episode_done:
            # Episode finished - record metrics
            # Calculate reward in both % and dollars
            episode_reward_pct = self.reward_pct
            
            # Calculate dollar reward: if sold, it's (sale_price - initial_price), else -initial_price
            if self.sold and self.sale_price is not None:
                episode_reward_dollar = self.sale_price - self.initial_price
            elif self.episode_done:  # Time expired without sale
                episode_reward_dollar = -self.initial_price  # Lost entire ticket value
            else:
                episode_reward_dollar = 0.0
            
            self.episode_rewards_pct.append(episode_reward_pct)
            self.episode_rewards_dollar.append(episode_reward_dollar)
            self.episode_sold.append(1.0 if self.sold else 0.0)
            
            # Update average rewards and sellout rate
            if len(self.episode_rewards_pct) > 0:
                self.avg_reward_pct = np.mean(list(self.episode_rewards_pct))
            if len(self.episode_rewards_dollar) > 0:
                self.avg_reward_dollar = np.mean(list(self.episode_rewards_dollar))
            if len(self.episode_sold) > 0:
                self.sellout_rate = np.mean(list(self.episode_sold)) * 100.0  # Convert to percentage
            
            # Mark episode complete in tracker
            self.behavior_tracker.finish_episode()
            
            # Save checkpoint periodically
            if self.checkpoint_path and hasattr(self.agent, 'save'):
                episodes_since_save = self.current_episode - self.last_saved_episode
                if episodes_since_save >= self.save_interval_episodes:
                    try:
                        self.agent.save(self.checkpoint_path)
                        self._save_episode_metadata()
                        self.last_saved_episode = self.current_episode
                        progress_pct = (self.current_episode / self.target_episodes) * 100.0
                        print(f"Checkpoint saved at episode {self.current_episode:,} / {self.target_episodes:,} ({progress_pct:.1f}%)")
                    except Exception as e:
                        print(f"Warning: Failed to save checkpoint: {e}")
            
            # Start new episode
            self.current_episode += 1
            self.reset_episode()
        
        # Step episode
        self.step_episode(train=True)
    
    def render(self):
        """Render all UI elements."""
        self.renderer.clear()
        
        # Draw time bar
        self.renderer.draw_time_bar(
            time_remaining=self.env.time_remaining,
            time_horizon=self.env.time_horizon
        )
        
        # Draw left sidebar with metrics
        self.renderer.draw_metrics_sidebar(
            episode=self.current_episode,
            target_episodes=self.target_episodes,
            step=self.current_step,
            avg_reward_dollar=self.avg_reward_dollar,
            avg_reward_pct=self.avg_reward_pct,
            sellout_rate=self.sellout_rate,
            current_price=self.current_price,
            initial_price=self.initial_price,
            quality_score=self.env.quality_score,
            is_weekend=self.env.event_context['is_weekend'],
            is_playoff=self.env.event_context['is_playoff'],
            p_sale=self.p_sale
        )
        
        # Draw center action indicator with revolving circle around price
        self.renderer.draw_action_indicator_with_price(
            action=self.last_action,
            action_map=self.env.action_map,
            current_price=self.current_price,
            initial_price=self.initial_price
        )
        
        # Draw right side: price change graph only
        steps, avg_price_changes = self.behavior_tracker.get_avg_price_change_by_step()
        if len(steps) > 0:
            self.renderer.draw_price_change_graph(steps, avg_price_changes)
        
        # Draw animations
        self.renderer.draw_sale_celebration()
        self.renderer.draw_failure_flash()
        
        pygame.display.flip()
    
    def run(self):
        """Main event loop."""
        # Initialize first episode
        self.reset_episode()
        self.last_step_time = pygame.time.get_ticks()
        
        while self.running:
            self.handle_events()
            
            # Run training steps with delay
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time >= self.step_delay_ms:
                # Run multiple steps per frame for speed (adjustable)
                steps_per_frame = max(1, int(100 / self.step_delay_ms))  # More steps if delay is lower
                for _ in range(steps_per_frame):
                    self.run_training_step()
                self.last_step_time = current_time
            
            # Update animations
            self.renderer.update_animations()
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(FPS)
        
        pygame.quit()
    
    def close(self):
        """Close the visualization."""
        self.running = False
        pygame.quit()


def create_app_from_checkpoint(
    model_path: Path,
    checkpoint_path: Optional[Path] = None,
    env_config: Optional[Dict] = None,
    agent_config: Optional[Dict] = None,
    step_delay_ms: int = 100,
    save_interval_episodes: int = 5,
    target_episodes: int = 200_000
) -> VisualizationApp:
    """
    Convenience function to create app from checkpoint.
    
    Args:
        model_path: Path to demand model .pkl file
        checkpoint_path: Path to agent checkpoint (optional, also used for auto-saving)
        env_config: Environment configuration dict
        agent_config: Agent configuration dict
        step_delay_ms: Delay between steps (milliseconds) - lower = faster training
        save_interval_episodes: Save checkpoint every N episodes (default: 10)
        target_episodes: Target number of episodes to train (default: 200,000)
    
    Returns:
        Initialized VisualizationApp instance
    """
    # Default environment config
    default_env_config = {
        'demand_model_path': model_path,
        'initial_price_range': (100.0, 500.0),
        'quality_range': (0.0, 1.0),
        'time_horizon': 2000.0,
        'time_step': 6.0,
        'price_bounds': (0.3, 3.0),
        'demand_scale': 0.5,
        'max_probability': 1.0,  # No cap - model already calibrated
        'random_seed': 42
    }
    if env_config:
        default_env_config.update(env_config)
    
    # Create environment
    env = TicketPricingEnv(**default_env_config)
    
    # Create or load agent
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
    
    if checkpoint_path and Path(checkpoint_path).exists():
        agent = DQNAgent.load(env, checkpoint_path)
    else:
        agent = DQNAgent(env=env, **default_agent_config)
    
    # Use checkpoint_path for auto-saving if provided, or create default path
    save_path = checkpoint_path
    if save_path is None:
        # Default checkpoint path
        default_checkpoint = Path(__file__).parent.parent / 'checkpoints' / 'dqn_ticket_pricing.pt'
        save_path = default_checkpoint
    
    # Create app with auto-saving enabled
    app = VisualizationApp(
        env=env,
        agent=agent,
        step_delay_ms=step_delay_ms,
        checkpoint_path=save_path,
        save_interval_episodes=save_interval_episodes,
        target_episodes=target_episodes
    )
    return app

