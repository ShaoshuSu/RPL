#!/bin/bash
# Setup hsmujoco-rpl environment for holosoma-extension-rpl.
#
# This script:
#   1. Runs the holosoma submodule's setup_mujoco.sh to create the conda env
#   2. Pip-installs holosoma_rpl into the same env
#
# Usage:
#   bash scripts/setup_mujoco.sh            # with GPU (default)
#   bash scripts/setup_mujoco.sh --no-warp  # CPU-only
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
REPO_ROOT=$(dirname "$SCRIPT_DIR")
HOLOSOMA_DIR="$REPO_ROOT/thirdparty/holosoma"

# Verify submodule is present
if [[ ! -f "$HOLOSOMA_DIR/scripts/setup_mujoco.sh" ]]; then
    echo "Error: holosoma submodule not found at $HOLOSOMA_DIR"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Step 1: Run holosoma's setup_mujoco.sh (creates conda env + installs holosoma)
echo "=========================================="
echo "Step 1: Setting up base holosoma environment"
echo "=========================================="
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsmujoco-rpl} \
    bash "$HOLOSOMA_DIR/scripts/setup_mujoco.sh" "$@"

# Step 2: Activate the env and install extension package
echo ""
echo "=========================================="
echo "Step 2: Installing holosoma_rpl"
echo "=========================================="

WORKSPACE_DIR=$HOME/.holosoma_deps
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsmujoco-rpl}

source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"

pip install -e "$REPO_ROOT/src/holosoma_rpl"

echo ""
echo "=========================================="
echo "holosoma-extension-rpl mujoco setup completed!"
echo "=========================================="
echo ""
echo "Activate with: source $REPO_ROOT/scripts/source_mujoco_setup.sh"
echo ""
echo "Run simulator:"
echo "  python -m holosoma_rpl.run_sim robot:g1-29dof terrain:terrain_locomotion_stairs_and_slope_eval"
