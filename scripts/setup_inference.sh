#!/bin/bash
# Setup hsinference-rpl environment for holosoma-extension-rpl.
#
# This script:
#   1. Runs the holosoma submodule's setup_inference.sh to create the conda env
#   2. Pip-installs holosoma_rpl_inference into the same env
#
# Usage:
#   bash scripts/setup_inference.sh
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(dirname "$SCRIPT_DIR")
HOLOSOMA_DIR="$REPO_ROOT/thirdparty/holosoma"

# Verify submodule is present
if [[ ! -f "$HOLOSOMA_DIR/scripts/setup_inference.sh" ]]; then
    echo "Error: holosoma submodule not found at $HOLOSOMA_DIR"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Step 1: Run holosoma's setup_inference.sh (creates conda env + installs holosoma_inference)
echo "=========================================="
echo "Step 1: Setting up base holosoma inference environment"
echo "=========================================="
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsinference-rpl} \
    bash "$HOLOSOMA_DIR/scripts/setup_inference.sh"

# Step 2: Activate the env and install extension package
echo ""
echo "=========================================="
echo "Step 2: Installing extension packages"
echo "=========================================="

WORKSPACE_DIR=$HOME/.holosoma_deps
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsinference-rpl}

source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"

# Install extension inference package in editable mode
pip install -e "$REPO_ROOT/src/holosoma_rpl_inference"

echo ""
echo "=========================================="
echo "holosoma-extension-rpl inference setup completed!"
echo "=========================================="
echo ""
echo "Activate with: source $REPO_ROOT/scripts/source_inference_setup.sh"
echo ""
echo "Run policy:"
echo "  python -m holosoma_rpl_inference.run_policy inference:g1-29dof-loco-manip-stand-height-waist"
