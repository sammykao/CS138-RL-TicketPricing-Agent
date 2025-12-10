# Dynamic Ticket Pricing with Reinforcement Learning  
Tufts COMP 138 – Final Project

This project implements a reinforcement learning (RL) agent that dynamically sets ticket prices for live events.  
The agent interacts with a data-driven simulator built from SeatData.io historical ticket sales, learning pricing strategies that maximize revenue and sell out inventory before the event.

The system includes the full RL workflow: data ingestion, environment modeling, agent training, baseline comparison, and visualization.

---

## 📌 1. Overview

Traditional pricing assumes stable demand curves. Real markets fluctuate based on time, remaining inventory, and prior pricing decisions.  
We frame pricing as a **sequential decision-making problem**, where the RL agent:

- observes: time remaining, inventory, last price, demand rate  
- chooses: raise price, lower price, or keep it  
- receives reward: revenue + bonuses for selling out + penalty for unstable pricing  
- learns a policy that adapts to market behavior  

Two agents are implemented:

1. **Tabular Q-Learning** (discretized state space)  
2. **Deep Q-Network (DQN)** (neural function approximation)

Both are evaluated against heuristic baselines.

---

## 📁 2. Project Structure

project/
│
├── data/
│ ├── raw/ # SeatData.io files
│ ├── processed/ # cleaned data
│ └── loader.py
│
├── env/
│ ├── ticket_env.py # Gym-like environment
│ ├── demand_model.py # elasticity-based demand simulator
│ └── utils.py
│
├── agent/
│ ├── q_learning.py
│ ├── dqn.py
│ └── networks.py
│
├── train/
│ ├── train_qlearning.py
│ ├── train_dqn.py
│ └── replay_buffer.py
│
├── eval/
│ ├── baselines.py
│ ├── metrics.py
│ ├── experiment_runner.py
│ └── plots.py
│
└── notebooks/
└── EDA.ipynb


## ⚙️ 3. Setup & Dependencies

Project dependencies are tracked with **[uv](https://docs.astral.sh/uv/)** via `pyproject.toml`.  
This keeps installs fast, reproducible, and isolated per project.

### Quick Start

1. **Install uv (one time)**
   ```bash
   pip install uv
   ```
   _Windows users can also grab the standalone installer from Astral if preferred._

2. **Sync the project environment**
   ```bash
   cd RL-TicketPricing-Agent
   uv sync
   ```
   This creates (or updates) a `.venv/` folder with all core dependencies:
   `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `tqdm`, `seaborn`, `gymnasium`, `pygame`, etc.

### Running the Visualization

**Windows (PowerShell):**
```powershell
.\run_visualization.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x run_visualization.sh
./run_visualization.sh
```

Both scripts will:
- Check for `uv` and install if missing
- Sync dependencies (creates/updates `.venv/`)
- Activate the virtual environment
- Run the visualization

**Advanced Options:**
```powershell
# Training mode with checkpoint
.\run_visualization.ps1 -Checkpoint "learning_environment\checkpoints\dqn_ticket_pricing.pt" -StepDelay 10

# Custom demand scale
.\run_visualization.ps1 -DemandScale 0.5
```

```bash
# Training mode with checkpoint
./run_visualization.sh --checkpoint learning_environment/checkpoints/dqn_ticket_pricing.pt --step-delay 10

# Custom demand scale
./run_visualization.sh --demand-scale 0.5
```

### Running Other Scripts

**Run any script through uv:**
```bash
uv run python learning_environment/data_generation/import_data.py
uv run python learning_environment/demand_modeling/train_model.py
uv run python learning_environment/agents/train_dqn/trainer.py
```

`uv run` automatically uses the synced environment - no manual venv activation needed.

---

## 🧹 4. Dataset & Pipeline (SeatData.io)

### Source  
SeatData.io historical event ticket sales.

### Steps  
1. Place raw files in `data/raw/`.  
2. Run:
python data/loader.py

This produces cleaned sequences in `data/processed/`, with:

- aligned timestamps  
- inventory levels  
- historical prices  
- demand rates  
- per-interval ticket sales  

These sequences feed the RL environment.

### Artificial dataset option
We could also generate our own datasets in the event of SeatData.io getting too 
expensive, where we could not only simulate the different events, but also the 
environment where people buy tickets at different times.

---

## 🎮 5. Environment

Implemented in `env/ticket_env.py`.

### **State**

[time_remaining, last_price, demand_rate]

### **Actions**
-Δp, 0, +Δp

(or discrete price levels)

### **Demand Model**
q_t = α_t * exp(-β * price_t) + noise

### **Reward**
revenue + sellout_bonus - price_jump_penalty

Episode ends when time runs out or all tickets are sold.

---

## 🤖 6. Agents

### **Tabular Q-Learning**  
File: `agent/q_learning.py`  
- discretized state bins  
- ε-greedy exploration  
- simple debugging baseline  

### **Deep Q-Network (DQN)**  
File: `agent/dqn.py`  
- two fully connected layers (64 units)  
- replay buffer  
- target network  
- ε-decay for exploration  
- handles continuous state space  

---

## 🏋️ 7. Training Instructions

### Fast Headless Training (Recommended)

Train quickly without visualization overhead:

```bash
cd learning_environment

# Train to 200,000 episodes (saves every 100)
uv run python train_headless.py --target-episodes 200000 --save-interval 100

# Custom settings
uv run python train_headless.py \
    --target-episodes 200000 \
    --save-interval 100 \
    --print-freq 100 \
    --demand-scale 0.5
```

**Benefits:**
- ⚡ **10-50x faster** than visualization mode
- ✅ Automatic checkpointing and resume
- ✅ Compatible with visualization (same checkpoint format)

### Resume with Visualization (When Close to Target)

Once you're close to your target (e.g., 195k / 200k), switch to visualization:

```bash
# Will resume from checkpoint and complete remaining episodes
uv run python visualization/run_visualization.py --target-episodes 200000
```

### Tabular Q-Learning:
```bash
python train/train_qlearning.py
```

### DQN (Legacy):
```bash
python train/train_dqn.py
```

**Checkpoint Files:**
- `checkpoints/dqn_ticket_pricing.pt` - Agent weights
- `checkpoints/episode_metadata.json` - Episode count
- `plots/training_metrics.json` - Training history

---

## 📊 8. Evaluation & Baselines

Run experiments and performance comparisons:
python eval/experiment_runner.py

### Metrics:
- total revenue  
- sell-through rate  
- price volatility  
- convergence stability  
- regret vs. optimal static price  
- inventory trajectory  

### Baselines:
1. Constant price  
2. Linear price decay  
3. Elasticity-based heuristic  

---

## 📈 9. Plotting & Visualization

### Interactive Visualization

Run the interactive visualization (recommended):
```bash
# Windows - Basic (200,000 episodes, saves every 100)
.\run_visualization.ps1

# Windows - Custom settings
.\run_visualization.ps1 -TargetEpisodes 200000 -SaveInterval 100 -StepDelay 10

# Linux/Mac - Basic
./run_visualization.sh

# Linux/Mac - Custom settings
./run_visualization.sh --target-episodes 200000 --save-interval 100 --step-delay 10
```

**Or use Python directly:**
```bash
cd learning_environment
uv run python visualization/run_visualization.py \
    --target-episodes 200000 \
    --save-interval 100 \
    --step-delay 10 \
    --checkpoint checkpoints/dqn_ticket_pricing.pt
```

This launches a Pygame-based visualization showing:
- Real-time agent pricing decisions
- Demand curves and sale probabilities
- Episode statistics and rewards
- Learning progress over time
- **Episode progress tracking** (current / target with progress bar)

**Features:**
- **Automatic checkpointing**: Saves agent state every N episodes (default: 100)
- **Episode tracking**: Resumes from last saved episode if restarted
- **Progress visualization**: Shows "Episode: X / 200,000" with progress bar
- **Auto-stop**: Automatically stops when target episodes reached

**Controls:**
- `R` = Reset current episode
- `ESC` = Quit (checkpoint saved automatically)

**Checkpoint Files:**
- `checkpoints/dqn_ticket_pricing.pt` - Agent weights and training state
- `checkpoints/episode_metadata.json` - Episode count and training metadata

**Resuming Training:**
The visualization automatically resumes from the last saved episode. Just run the same command again - it will load the checkpoint and continue from where it left off.

### RL Paper Graphs

Generate publication-quality plots for papers:

```bash
cd learning_environment

# Generate all training curves and analysis plots
uv run python plot_training_curves.py \
    --checkpoint checkpoints/dqn_ticket_pricing.pt \
    --output-dir plots
```

**Generated Plots:**
1. **`training_curves.png`** - Learning curves (rewards, loss, epsilon, episode lengths)
2. **`convergence_analysis.png`** - Convergence metrics and reward stability
3. **`performance_metrics.png`** - Performance statistics and learning progress

### Static Plots

Generate demand model plots:
```bash
# 3D demand curves
uv run python learning_environment/plot_3d_demand_curves.py

# Average sale price over time
uv run python learning_environment/plot_avg_sale_price_by_time.py

# 3D price-probability-time surface
uv run python learning_environment/demand_modeling/plot_3d_price_prob_time.py
```

Plots are saved to `learning_environment/plots/`.

---

## 🔄 10. Reproducibility

Set seeds in `config.py`:

Or pass via CLI:
python train/train_dqn.py --seed 42


## 👥 11. Team Responsibilities

- **Sammy:** Data pipeline, environment  
- **Javier:** Q-Learning + DQN implementation  
- **Reed:** Evaluation, experiments, visualizations  

---

## 🎯 12. Expected Outcomes

The RL agents are expected to:

- beat heuristic pricing on revenue  
- consistently sell out inventory  
- adapt policies to demand shifts  
- reduce price volatility compared to naïve strategies  
- generalize across different event types  

---

If you'd like, I can also:

- export a `requirements.txt` via `uv pip compile pyproject.toml -o requirements.txt`  
- ship a polished version with badges, logos, section icons  
- add a mermaid diagram of the full system architecture  
