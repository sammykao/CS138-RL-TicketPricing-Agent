# Setup and run RL Ticket Pricing Visualization
# This script:
#   1. Checks for uv and installs if missing
#   2. Syncs dependencies (creates/updates .venv)
#   3. Activates virtual environment
#   4. Runs the visualization
#
# Usage: 
#   Basic:  .\run_visualization.ps1
#   With checkpoint: .\run_visualization.ps1 -Checkpoint "path\to\checkpoint.pt"
#   Custom settings: .\run_visualization.ps1 -StepDelay 50 -DemandScale 0.5

param(
    [string]$Checkpoint = $null,
    [int]$StepDelay = 10,
    [float]$DemandScale = 1.0,
    [int]$TargetEpisodes = 200000,
    [int]$SaveInterval = 100
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "Setting up RL Ticket Pricing Visualization..." -ForegroundColor Green

# Check if uv is installed
try {
    $null = Get-Command uv -ErrorAction Stop
} catch {
    Write-Host "Warning: uv is not installed. Installing uv..." -ForegroundColor Yellow
    pip install uv
}

# Get script directory and navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if learning_environment directory exists
$learningEnvPath = Join-Path $scriptPath "learning_environment"
if (-not (Test-Path $learningEnvPath)) {
    Write-Host "Error: learning_environment directory not found!" -ForegroundColor Red
    exit 1
}

# Sync dependencies (creates/updates .venv)
Write-Host "Syncing dependencies with uv..." -ForegroundColor Green
uv sync

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to sync dependencies!" -ForegroundColor Red
    exit 1
}

# Activate virtual environment (uv creates .venv in project root)
$venvPath = Join-Path $scriptPath ".venv"
if (Test-Path $venvPath) {
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Activating virtual environment..." -ForegroundColor Green
        & $activateScript
    }
}

# Build argument list for visualization script
$visualizationArgs = @()

if ($Checkpoint) {
    $visualizationArgs += "--checkpoint", $Checkpoint
}

$visualizationArgs += "--step-delay", $StepDelay
$visualizationArgs += "--demand-scale", $DemandScale
$visualizationArgs += "--target-episodes", $TargetEpisodes
$visualizationArgs += "--save-interval", $SaveInterval

# Run visualization (uv run automatically uses .venv if activation fails)
Write-Host "Starting visualization..." -ForegroundColor Green
Write-Host "Target Episodes: $TargetEpisodes" -ForegroundColor Cyan
Write-Host "Save Interval: Every $SaveInterval episodes" -ForegroundColor Cyan
Write-Host "Controls: R=Reset | ESC=Quit" -ForegroundColor Yellow
Write-Host ""

Set-Location $learningEnvPath
uv run python visualization/run_visualization.py $visualizationArgs

