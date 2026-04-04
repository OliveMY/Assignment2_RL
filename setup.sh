#!/bin/bash
# ============================================================
# Setup script for Assignment 2 - Reinforcement Learning
# with Unity ML-Agents environments
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "==========================================="
echo " Assignment 2 - RL Setup"
echo "==========================================="

# ----------------------------------------------------------
# 1. Check Python version (require 3.9 or 3.10)
# ----------------------------------------------------------
PYTHON_CMD=""
for cmd in python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3 not found. Please install Python 3.9 or 3.10."
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

echo "Found $PYTHON_CMD ($PY_VERSION)"

if [ "$PY_MAJOR" -ne 3 ] || [ "$PY_MINOR" -lt 9 ] || [ "$PY_MINOR" -gt 12 ]; then
    echo "WARNING: Python 3.9-3.12 is recommended. You have $PY_VERSION."
    echo "         mlagents-envs may not work with other versions."
fi

# ----------------------------------------------------------
# 2. Create virtual environment
# ----------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment in $VENV_DIR ..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo ""
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "Activated virtual environment ($(python --version))"

# ----------------------------------------------------------
# 3. Upgrade pip
# ----------------------------------------------------------
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------
# 4. Install dependencies
# ----------------------------------------------------------
echo ""
echo "Installing project dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"

# ----------------------------------------------------------
# 5. Make Unity game executables runnable
# ----------------------------------------------------------
GAMES_DIR="$PROJECT_DIR/Games"
if [ -d "$GAMES_DIR" ]; then
    echo ""
    echo "Setting execute permissions on Unity builds..."
    for exe in "$GAMES_DIR"/*.x86_64; do
        if [ -f "$exe" ]; then
            chmod +x "$exe"
            echo "  +x $(basename "$exe")"
        fi
    done
else
    echo ""
    echo "WARNING: Games/ directory not found. Place Unity builds there before training."
fi

# ----------------------------------------------------------
# 6. Verify installation
# ----------------------------------------------------------
echo ""
echo "Verifying installation..."
python -c "
import stable_baselines3
import gymnasium
import mlagents_envs
import tensorboard
import numpy
print(f'  stable-baselines3 : {stable_baselines3.__version__}')
print(f'  gymnasium          : {gymnasium.__version__}')
print(f'  mlagents-envs      : {mlagents_envs.__version__}')
print(f'  numpy              : {numpy.__version__}')
print()
print('All packages installed successfully!')
"

# ----------------------------------------------------------
# 7. Print usage instructions
# ----------------------------------------------------------
echo ""
echo "==========================================="
echo " Setup complete!"
echo "==========================================="
echo ""
echo "To activate the environment in a new shell:"
echo "  source venv/bin/activate"
echo ""
echo "To start training:"
echo "  python src/train.py --env simple"
echo "  python src/train.py --env medium"
echo "  python src/train.py --env hard"
echo ""
echo "To monitor training with TensorBoard:"
echo "  tensorboard --logdir results/"
echo ""
