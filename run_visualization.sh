#!/bin/bash
# Setup and run RL Ticket Pricing Visualization
# This script:
#   1. Checks for uv and installs if missing
#   2. Syncs dependencies (creates/updates .venv)
#   3. Activates virtual environment
#   4. Runs the visualization
#
# Usage: 
#   Basic:  ./run_visualization.sh
#   With checkpoint: ./run_visualization.sh --checkpoint path/to/checkpoint.pt
#   Custom settings: ./run_visualization.sh --step-delay 50 --demand-scale 0.5 --target-episodes 200000

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up RL Ticket Pricing Visualization...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Warning: uv is not installed. Installing uv...${NC}"
    pip install uv
fi

# Get script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Check if learning_environment directory exists
if [ ! -d "learning_environment" ]; then
    echo -e "${RED}Error: learning_environment directory not found!${NC}"
    exit 1
fi

# Sync dependencies (creates/updates .venv)
echo -e "${GREEN}Syncing dependencies with uv...${NC}"
uv sync

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to sync dependencies!${NC}"
    exit 1
fi

# Activate virtual environment (uv creates .venv in project root)
VENV_PATH="$SCRIPT_DIR/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$VENV_PATH/bin/activate"
fi

# Run visualization with all passed arguments
echo -e "${GREEN}Starting visualization...${NC}"
echo -e "${YELLOW}Controls: R=Reset | ESC=Quit${NC}"
echo ""

cd learning_environment
# uv run automatically uses .venv if activation didn't work
uv run python visualization/run_visualization.py "$@"

