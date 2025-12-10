"""
Entry point script for running the RL ticket pricing visualization.

Usage:
    python visualization/run_visualization.py [--checkpoint PATH] [--mode episode|training] [--step-delay MS]
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.app import create_app_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run RL ticket pricing visualization")
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path(__file__).parent.parent / 'models' / 'demand_model_v12_anti_gouge.pkl',
        help='Path to demand model .pkl file'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Path to agent checkpoint (optional)'
    )
    parser.add_argument(
        '--step-delay',
        type=int,
        default=100,
        help='Delay between steps (milliseconds) - lower = faster training'
    )
    parser.add_argument(
        '--demand-scale',
        type=float,
        default=0.5,
        help='Demand scale factor (lower = harder, higher = easier)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Save checkpoint every N episodes (default: 100)'
    )
    parser.add_argument(
        '--target-episodes',
        type=int,
        default=200_000,
        help='Target number of episodes to train (default: 200,000)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not args.model_path.exists():
        print(f"Error: Model file not found: {args.model_path}")
        print("Please train the demand model first or specify a valid path with --model-path")
        return 1
    
    # Check if checkpoint exists (if provided)
    if args.checkpoint and not args.checkpoint.exists():
        print(f"Warning: Checkpoint not found: {args.checkpoint}")
        print("Starting with untrained agent...")
        args.checkpoint = None
    
    # Default checkpoint path
    if args.checkpoint is None:
        default_checkpoint = Path(__file__).parent.parent / 'checkpoints' / 'dqn_ticket_pricing.pt'
        if default_checkpoint.exists():
            args.checkpoint = default_checkpoint
            print(f"Using checkpoint: {args.checkpoint}")
    
    # Create app
    try:
        app = create_app_from_checkpoint(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint,
            env_config={
                'demand_scale': args.demand_scale
            },
            step_delay_ms=args.step_delay,
            save_interval_episodes=args.save_interval,
            target_episodes=args.target_episodes
        )
        
        print(f"Starting visualization (training mode)...")
        print(f"Target: {args.target_episodes:,} episodes")
        print(f"Checkpoint interval: Every {args.save_interval:,} episodes")
        print("Controls:")
        print("  R: Reset episode")
        print("  ESC: Quit")
        
        app.run()
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        return 0
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

