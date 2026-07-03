#!/usr/bin/env bash
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export nnUNet_raw="$PROJECT/nnunet_raw"
export nnUNet_preprocessed="$PROJECT/nnunet_preprocessed"
export nnUNet_results="$PROJECT/nnunet_results"
export nnUNet_extTrainer="$PROJECT/mbrats/training"

export PYENV_VERSION=brats

echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
