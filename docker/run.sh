#!/usr/bin/env bash
# Run a command inside the brats container with data dirs mounted.
#
# Usage:
#   bash docker/run.sh nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerBraTS --npz
#   bash docker/run.sh bash   # interactive shell

set -euo pipefail

DATA_DIR="${BRATS_DATA_DIR:-/media/mkurtys/T7/datasets/brats2026}"
IMAGE="${BRATS_IMAGE:-brats2026}"

docker run --rm -it \
    --gpus all \
    --shm-size=8g \
    -v "$DATA_DIR":/data/source:ro \
    -v "$(pwd)/nnunet_raw":/data/nnunet_raw \
    -v "$(pwd)/nnunet_preprocessed":/data/nnunet_preprocessed \
    -v "$(pwd)/nnunet_results":/data/nnunet_results \
    -v "$(pwd)/predictions":/data/predictions \
    -w /opt/brats \
    "$IMAGE" "$@"
