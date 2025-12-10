"""
Headless DQN Training Script (Fast Training Without Visualization)

Trains the DQN agent without pygame rendering for maximum speed.
Saves checkpoints and episode metadata compatible with visualization.

Usage:
    uv run python train_headless.py --target-episodes 200000 --save-interval 100
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.train_dqn.trainer import DQNTrainer


def load_episode_metadata(metadata_path: Path) -> int:
    """Load episode count from metadata file."""
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('episode_count', 0)
        except Exception as e:
            print(f"Warning: Could not load episode metadata: {e}")
    return 0


def save_episode_metadata(metadata_path: Path, episode_count: int, target_episodes: int, checkpoint_path: Path):
    """Save episode count to metadata file."""
    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            'episode_count': episode_count,
            'target_episodes': target_episodes,
            'last_saved': str(checkpoint_path)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save episode metadata: {e}")


def main():
    parser = argparse.ArgumentParser(description="Headless DQN training (fast, no visualization)")
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/demand_model_v12_anti_gouge.pkl'),
        help='Path to demand model .pkl file'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/dqn_ticket_pricing.pt'),
        help='Path to agent checkpoint'
    )
    parser.add_argument(
        '--target-episodes',
        type=int,
        default=200_000,
        help='Target number of episodes to train (default: 200,000)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Save checkpoint every N episodes (default: 100)'
    )
    parser.add_argument(
        '--print-freq',
        type=int,
        default=100,
        help='Print metrics every N episodes (default: 100, use 0 to disable)'
    )
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=0.5,
        help='Demand scale factor (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to learning_environment directory
    learning_env_dir = Path(__file__).parent
    model_path = learning_env_dir / args.model_path if not args.model_path.is_absolute() else args.model_path
    checkpoint_path = learning_env_dir / args.checkpoint if not args.checkpoint.is_absolute() else args.checkpoint
    metadata_path = checkpoint_path.parent / 'episode_metadata.json'
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    # Load existing episode count
    start_episode = load_episode_metadata(metadata_path)
    remaining_episodes = args.target_episodes - start_episode
    
    if remaining_episodes <= 0:
        print(f"Training complete! Already at {start_episode:,} / {args.target_episodes:,} episodes")
        return 0
    
    print(f"{'='*60}")
    print(f"HEADLESS DQN TRAINING")
    print(f"{'='*60}")
    print(f"Target Episodes: {args.target_episodes:,}")
    print(f"Starting from: {start_episode:,} episodes")
    print(f"Remaining: {remaining_episodes:,} episodes")
    print(f"Save Interval: Every {args.save_interval:,} episodes")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Create trainer
    trainer = DQNTrainer(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        env_config={
            'demand_scale': args.demand_scale
        },
        load_checkpoint=True
    )
    
    # Set current episode from metadata
    trainer.current_episode = start_episode
    
    # Training callback to save checkpoints and metadata
    def training_callback(episode: int, metrics: Dict):
        """Callback to save checkpoints periodically."""
        if episode % args.save_interval == 0 or episode == args.target_episodes:
            try:
                trainer.save_checkpoint()
                save_episode_metadata(metadata_path, episode, args.target_episodes, checkpoint_path)
                progress_pct = (episode / args.target_episodes) * 100.0
                print(f"\n✓ Checkpoint saved at episode {episode:,} / {args.target_episodes:,} ({progress_pct:.1f}%)")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
    
    # Train in chunks to allow periodic checkpointing
    chunk_size = args.save_interval
    episodes_trained = 0
    
    while episodes_trained < remaining_episodes:
        chunk_episodes = min(chunk_size, remaining_episodes - episodes_trained)
        
        print(f"\nTraining chunk: {chunk_episodes:,} episodes (Total: {start_episode + episodes_trained:,} / {args.target_episodes:,})")
        
        metrics = trainer.train(
            n_episodes=chunk_episodes,
            print_freq=args.print_freq,
            callback=training_callback
        )
        
        episodes_trained += chunk_episodes
        
            # Save checkpoint after chunk
        try:
            trainer.save_checkpoint()
            trainer.save_training_metrics()  # Save metrics for plotting
            current_episode = start_episode + episodes_trained
            save_episode_metadata(metadata_path, current_episode, args.target_episodes, checkpoint_path)
            progress_pct = (current_episode / args.target_episodes) * 100.0
            print(f"\n✓ Checkpoint saved at episode {current_episode:,} / {args.target_episodes:,} ({progress_pct:.1f}%)")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
    
    # Final checkpoint
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {args.target_episodes:,} episodes")
    print(f"{'='*60}")
    trainer.save_checkpoint()
    trainer.save_training_metrics()  # Save final metrics
    save_episode_metadata(metadata_path, args.target_episodes, args.target_episodes, checkpoint_path)
    print(f"Final checkpoint saved: {checkpoint_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Training metrics saved: {trainer.plots_dir / 'training_metrics.json'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

