#!/usr/bin/env bash
# Install project dependencies and the mbrats package in editable mode.
#
# PyTorch is installed separately first so the CUDA variant can be chosen.
# nnUNet is installed as editable from the local nnUNet/ clone.
# Everything else comes from pyproject.toml.
#
# Usage:
#   bash scripts/install.sh          # CUDA 12.4 (default)
#   TORCH_INDEX=cpu bash scripts/install.sh

set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "=== Installing PyTorch from $TORCH_INDEX ==="
pip install torch torchvision --index-url "$TORCH_INDEX"

echo "=== Installing nnunetv2 (editable) ==="
pip install -e nnUNet/

echo "=== Installing mbrats and all dependencies ==="
pip install -e .

echo "=== Done. Activate env and source scripts/setup_env.sh before running. ==="
