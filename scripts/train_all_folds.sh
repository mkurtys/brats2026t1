#!/usr/bin/env bash
set -e
source "$(dirname "$0")/setup_env.sh"

DATASET_ID=1
CONFIG=3d_fullres

for fold in 0 1 2 3 4; do
    echo "=== Training fold $fold ==="
    nnUNetv2_train $DATASET_ID $CONFIG $fold --npz
done

echo "=== All folds complete. Finding best configuration ==="
nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG
