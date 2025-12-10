# Training Guide: Fast Headless Training + Visualization

## Quick Start

### 1. Fast Headless Training (No Visualization)

Train quickly without pygame rendering:

```bash
cd learning_environment

# Basic: Train to 200,000 episodes, saves every 100
uv run python train_headless.py

# Custom settings
uv run python train_headless.py \
    --target-episodes 200000 \
    --save-interval 100 \
    --print-freq 100 \
    --demand-scale 0.5
```

**Features:**
- ✅ **Much faster** - No pygame rendering overhead
- ✅ **Automatic checkpointing** - Saves every N episodes
- ✅ **Resume capability** - Automatically resumes from last episode
- ✅ **Progress tracking** - Shows progress percentage
- ✅ **Compatible with visualization** - Same checkpoint format

### 2. Resume with Visualization (When Close to Target)

Once you're close to your target (e.g., 195,000 / 200,000), switch to visualization:

```bash
# Windows
.\run_visualization.ps1 -TargetEpisodes 200000

# Linux/Mac
./run_visualization.sh --target-episodes 200000

# Or directly
cd learning_environment
uv run python visualization/run_visualization.py \
    --target-episodes 200000 \
    --checkpoint checkpoints/dqn_ticket_pricing.pt
```

The visualization will:
- Load the checkpoint from headless training
- Resume from the exact episode count
- Show progress bar and metrics
- Complete the remaining episodes with visual feedback

## Workflow Example

```bash
# Step 1: Fast training (headless) - trains most episodes quickly
uv run python train_headless.py --target-episodes 200000 --save-interval 100

# Step 2: When close to target (e.g., 195k), switch to visualization
# This will resume from 195k and complete to 200k with visual feedback
uv run python visualization/run_visualization.py --target-episodes 200000

# Step 3: Generate paper graphs
uv run python plot_training_curves.py --checkpoint checkpoints/dqn_ticket_pricing.pt
```

## Generating RL Paper Graphs

After training, generate publication-quality plots:

```bash
cd learning_environment
uv run python plot_training_curves.py \
    --checkpoint checkpoints/dqn_ticket_pricing.pt \
    --output-dir plots
```

**Generated Plots:**

1. **`training_curves.png`** - Main learning curves:
   - Episode rewards (raw + rolling averages)
   - Training loss
   - Epsilon decay (exploration rate)
   - Episode lengths

2. **`convergence_analysis.png`** - Convergence metrics:
   - Reward stability over time (mean + std dev)
   - Reward distribution by training phase (early/mid/late)

3. **`performance_metrics.png`** - Performance summary:
   - Reward statistics (mean, std, min, max, median)
   - Learning progress (first 10% vs last 10%)
   - Episode length statistics

## Files Created

**During Training:**
- `checkpoints/dqn_ticket_pricing.pt` - Agent weights
- `checkpoints/episode_metadata.json` - Episode count tracking
- `plots/training_metrics.json` - Full training history

**After Plotting:**
- `plots/training_curves.png`
- `plots/convergence_analysis.png`
- `plots/performance_metrics.png`

## Tips

1. **Start headless** - Train most episodes without visualization for speed
2. **Switch to visualization** - When close to target (e.g., 95% done) for final episodes
3. **Save frequently** - Use `--save-interval 100` to avoid losing progress
4. **Generate plots** - After training completes, create paper-ready graphs

## Performance Comparison

- **Headless training**: ~10-50x faster (no rendering overhead)
- **Visualization**: Slower but provides real-time feedback
- **Best practice**: Use headless for bulk training, visualization for final episodes

