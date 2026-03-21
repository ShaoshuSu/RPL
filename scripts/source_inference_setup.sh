#!/bin/bash
# Activate hsinference environment for holosoma-extension-rpl.
#
# This script sources holosoma's inference environment activation.
# Extension packages are pip-installed, so no PYTHONPATH needed.
#
# Usage: source scripts/source_inference_setup.sh

# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi

REPO_ROOT=$(dirname "$SCRIPT_DIR")
HOLOSOMA_DIR="$REPO_ROOT/thirdparty/holosoma"

# Source holosoma's inference environment
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-hsinference-rpl}
source "$HOLOSOMA_DIR/scripts/source_inference_setup.sh"

echo ""
echo "holosoma-extension-rpl inference environment ready."
echo "  Run policy: python -m holosoma_rpl_inference.run_policy ..."
