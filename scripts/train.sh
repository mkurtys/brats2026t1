#!/usr/bin/env bash
# Usage:
#   bash scripts/train.sh <plans> [fold]
#   bash scripts/train.sh nnUNetResEncUNetMPlans      # all 5 folds
#   bash scripts/train.sh nnUNetResEncUNetMPlans 0    # fold 0 only
set -e
source "$(dirname "$0")/setup_env.sh"

PLANS="${1:?'Usage: train.sh <plans> [fold]'}"
DATASET_ID=1
CONFIG=3d_fullres
TRAINER=nnUNetTrainerCheckpoint250

if [[ -n "${2:-}" ]]; then
    folds=("$2")
else
    folds=(0 1 2 3 4)
fi

for fold in "${folds[@]}"; do
    echo "=== Training fold $fold ==="
    nnUNetv2_train $DATASET_ID $CONFIG $fold -p $PLANS -tr $TRAINER --npz
done

if [[ ${#folds[@]} -eq 5 ]]; then
    echo "=== All folds complete. Finding best configuration ==="
    nnUNetv2_find_best_configuration $DATASET_ID -c $CONFIG -p $PLANS -tr $TRAINER
fi
