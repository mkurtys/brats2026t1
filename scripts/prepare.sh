#!/usr/bin/env bash
# Data preparation: convert → preprocess → instance locations
#
# Usage:
#   bash scripts/prepare.sh <dataset_dir>            # run all steps
#   bash scripts/prepare.sh <dataset_dir> --from 2  # resume from step 2
#
# Steps:
#   1  convert            — symlink raw data into nnunet_raw, add subtraction + ratio channels
#   2  preprocess_default — plan + preprocess for nnUNetPlans (6-channel)
#   3  preprocess_resenc  — plan + preprocess for nnUNetResEncUNetLPlans (6-channel)
#   4  instance_locs      — precompute TC instance locations for both plan folders

set -euo pipefail
source "$(dirname "$0")/setup_env.sh"

DATASET_ID=1
DATASET_DIR="${1:?'Usage: prepare.sh <dataset_dir> [--from N]'}"; shift
TRAIN_DIR="$DATASET_DIR/MICCAI-LH-BraTS2025-MET-Challenge-Training"
VAL_DIR="$DATASET_DIR/Validation"
CORRECTED="$DATASET_DIR/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"

FROM_STEP=1
if [[ "${1:-}" == "--from" ]]; then FROM_STEP="${2:?'--from requires a step number'}"; fi

step() {
    local n=$1; local name=$2
    if (( n < FROM_STEP )); then
        echo "=== Step $n [$name] SKIPPED (--from $FROM_STEP) ==="; return 1
    fi
    echo ""; echo "======================================================"; echo "=== Step $n: $name ==="; echo "======================================================"
    return 0
}

# ── Step 1: convert to nnU-Net format (6 channels) ────────────────────────────
if step 1 "convert"; then
    python -m mbrats.preprocessing.convert_to_nnunet \
        --train_dir "$TRAIN_DIR" \
        --output_dir "$nnUNet_raw/Dataset001_BraTSMETS" \
        --corrected_labels "$CORRECTED" \
        --val_dir "$VAL_DIR" \
        --add_subtraction \
        --add_ratio
fi

# ── Step 2: preprocess — default plans ────────────────────────────────────────
if step 2 "preprocess (nnUNetPlans, 6-channel)"; then
    nnUNetv2_plan_and_preprocess \
        -d $DATASET_ID \
        --verify_dataset_integrity \
        -np 8
fi

# ── Step 3: preprocess — ResEncUNetL plans ────────────────────────────────────
if step 3 "preprocess (nnUNetResEncUNetLPlans, 6-channel)"; then
    nnUNetv2_plan_and_preprocess \
        -d $DATASET_ID \
        -pl nnUNetPlannerResEncL \
        --verify_dataset_integrity \
        -np 8
fi

# ── Step 4: precompute instance locations (both plan folders) ─────────────────
if step 4 "instance locations (nnUNetPlans + ResEncUNetL)"; then
    python -m mbrats.preprocessing.precompute_instance_locations \
        --folder "$nnUNet_preprocessed/Dataset001_BraTSMETS/nnUNetResEncUNetLPlans_3d_fullres" \
        --force
fi

echo ""; echo "=== Preparation complete ==="
