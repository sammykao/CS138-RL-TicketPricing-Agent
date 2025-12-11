# Visualization

This folder contains the interactive web-based visualization tool for exploring the RL agent's behavior.

## Contents

- **`app.py`**: Main Streamlit application. Provides interactive interface to:
  - Visualize agent pricing decisions in real-time
  - Adjust environment parameters (demand scale, initial price, quality)
  - Load trained agent checkpoints
  - View demand curves and sale probabilities

- **`run_visualization.py`**: Command-line interface to launch the visualization app.

- **`renderer.py`**: Renders the environment state and agent actions visually.

- **`data_helpers.py`**: Helper functions for computing demand curves and other visualizations.

- **`ui_style.py`**: UI styling and theme configuration for the Streamlit app.

## Usage

To launch the visualization:
```bash
# Windows
.\run_visualization.ps1

# Linux/Mac
./run_visualization.sh

# Or directly
cd learning_environment/visualization
uv run streamlit run app.py
```

The visualization allows you to:
- Watch the agent make pricing decisions step-by-step
- Adjust parameters and see how it affects behavior
- Compare different trained models
- Explore demand curves interactively

