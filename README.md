# BraTS 2026 MET Task 1 — mwro team

Brain metastases segmentation using nnU-Net v2. Four labels: NETC (1), SNFH (2), ET (3), RC (4).

## Stack

- **nnU-Net v2** (`nnunetv2>=2.8.0`) with custom trainers in `mbrats/training/`
- **mbrats** — project Python package (`preprocessing/`, `training/`, `evaluation/`, `analysis/`)
- Python env: pyenv virtualenv `brats` (Python 3.10+)

## Custom Trainer (`nnUNetTrainerBraTS`)

Changes over default `nnUNetTrainer`:

1. **Focal loss** (γ=2) replaces CE in the Dice+CE compound loss — downweights easy background voxels, upweights hard small-lesion boundaries.
2. **Instance-uniform patch sampling** — foreground patches are centred on a randomly chosen TC/RC connected component (√size-weighted), so 27 mm³ lesions get the same sampling frequency as 27 000 mm³ lesions. Requires precomputed instance locations (prepare.sh step 4).
3. **Class-frequency-balanced case sampling** — per-case selection probability is proportional to inverse class frequency, so RC/NETC-containing cases are drawn more often.


## Source Data

Download from https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097&__forum_threadId=13631#Data%20Files
Three files:
* training - data+segmentation masks
* corrected labels - corrected labels for BraTS-MET-01094-003-seg.nii.gz  BraTS-MET-01184-002-seg.nii.gz
* validation - data


## Data Pre-Preprocessing

Applying corrected labels.
Converting to nnunet directories structure.
Adding T1c−T1n subtraction + T1c/T1n ratio =  6-channel per case (T1n, T1c, T2w, FLAIR + T1c−T1n subtraction + T1c/T1n ratio). 

## NNUnet Plan and preprocess

Observation is that regardless of planner used, dataset is processed same way (normalization/cropping/resampling), so we can reuse preprocessed dataset

Plan
```bash 
nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)
```

Plan and preprocess
```bash
  nnUNetv2_plan_and_preprocess \
      -d 1 \
      -pl nnUNetPlannerResEncL \
      --verify_dataset_integrity \
      -np 8
```

After preprocessing, the dataset folder in nnUNet_preprocessed contains:

* dataset_fingerprint.json
* nnUNetPlans.json
* preprocessed data folders for the created configurations



## Installation

### 1. Clone

```bash
git clone <repo-url>
cd brats2026t1
```

### 2. Create Python environment

```bash
# pyenv + virtualenv (matches the project convention)
pyenv install 3.13.4          # skip if already installed
pyenv virtualenv 3.13.4 brats
pyenv activate brats
```

Any Python 3.10+ environment works; pyenv is not required.

### 3. Install dependencies

```bash
bash scripts/install.sh            # PyTorch with CUDA 12.4 (default)
TORCH_INDEX=cpu bash scripts/install.sh   # CPU-only
TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/install.sh  # CUDA 12.1
```

This runs `pip install torch torchvision --index-url <TORCH_INDEX>` then `pip install -e .`, which installs `mbrats` (editable) and all remaining dependencies (`nnunetv2`, `nibabel`, `panoptica`, `BraTS_evaluation`, `SimpleITK`, …).

### 4. Set environment variables

```bash
source scripts/setup_env.sh
```

Sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, and `nnUNet_extTrainer` (points nnUNet's trainer discovery at `mbrats/training/`). Run once per shell session.

## Docker

Useful for reproducible environments and offline machines.

```bash
# Build (once, requires internet)
docker build -t brats2026 .

# Run any command with data dirs mounted from the current working directory
bash docker/run.sh nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerBraTS --npz

# Interactive shell
bash docker/run.sh bash
```

Data directories (`nnunet_raw/`, `nnunet_preprocessed/`, `nnunet_results/`, `predictions/`) are mounted from the host working directory. Override the data source or image name via env vars:

```bash
BRATS_DATA_DIR=/path/to/data BRATS_IMAGE=brats2026 bash docker/run.sh <cmd>
```

## Quickstart

```bash
# Set env variables once per shell session
source scripts/setup_env.sh

# Data preparation: convert → preprocess → instance locations
bash scripts/prepare.sh /media/mkurtys/T7/datasets/brats2026

# Training: all 5 folds × both plan configs
bash scripts/train.sh

# Resume from a specific step
bash scripts/prepare.sh /media/mkurtys/T7/datasets/brats2026 --from 2
bash scripts/train.sh --from 2
```

### prepare.sh steps

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python -m mbrats.preprocessing.convert_to_nnunet ...` | Symlink raw data + apply corrected labels + add derived channels |
| 2 | `nnUNetv2_plan_and_preprocess -d 1` | Plan + preprocess (nnUNetPlans) |
| 3 | `nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL` | Plan + preprocess (ResEncUNetLPlans) |
| 4 | `python -m mbrats.preprocessing.precompute_instance_locations` | TC/RC instance centroids for instance-uniform sampling |

## Training

```bash
# Single fold
nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerBraTS --npz

# All 5 folds × both plan configs
bash scripts/train.sh
```

## Distributed Training

nnU-Net uses `torch.multiprocessing.spawn` internally — no `torchrun` needed. Pass `-num_gpus N` to `nnUNetv2_train`:

### Device selection

Use `-device` to choose `cpu`, `cuda`, or `mps`.

For multi-GPU systems, select the GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0 --npz
```

### Recommended multi-GPU usage

If you have multiple GPUs, the preferred strategy is usually one training per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 --npz
```

Distributed training is also available:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz -num_gpus X
```

**Notes:**
- Batch size scales with GPU count — effective batch size = `batch_size × num_gpus`. With 2 GPUs and batch=2 per GPU, effective batch is 4.
- `is_ddp=True` is passed automatically to the loss; the `DC_and_Focal_loss` already handles it via `batch_dice`.
- Instance-uniform sampling runs independently per GPU worker — no special DDP handling needed.

## Inference

```bash
# Predict on validation set
nnUNetv2_predict \
  -d 1 -c 3d_fullres -f 0 -tr nnUNetTrainerBraTS \
  -i nnunet_raw/Dataset001_BraTSMETS/imagesVal \
  -o predictions/fold0_val
```

## Evaluation

**Option 1 — project evaluation script** (uses GT from `labelsTr`, panoptica-based):

```bash
python -m mbrats.evaluation.evaluate \
  --pred nnunet_results/Dataset001_BraTSMETS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
  --gt nnunet_raw/Dataset001_BraTSMETS/labelsTr \
  --out results/fold0_cv_eval.json
```

**Option 2 — official BraTS-evaluation package:**

```bash
# Run on a split where GT exists
brats-evaluate \
  --config mets \
  --ref_path /path/to/reference/niftis/ \
  --pred_path /path/to/prediction/niftis/ \
  --summary_json ./panoptica_evaluation_summary.json

# Parse summary into per-label CSV
brats-parse-metrics mets \
  --json_path ./panoptica_evaluation_summary.json \
  --vol_threshold 27.0 \
  --overlap_threshold 0.2 \
  --output_csv_path ./parsed_panoptica_mets_stats.csv
```

## Baseline Results (fold 0 CV, 260 cases)

will be in other md

## Repository Layout

```
mbrats/
  preprocessing/
    convert_to_nnunet.py        — BraTS → nnU-Net format, applies corrected labels, adds derived channels
    precompute_instance_locations.py — TC/RC CC centroids for instance-uniform sampling
  training/
    nnUNetTrainerBraTS.py       — focal loss + instance-uniform sampling + class-balanced case sampling
    nnUNetTrainerBraTSBlobLoss.py — adds per-instance blob region loss on top of BraTS trainer
  evaluation/
    evaluate.py                 — panoptica-based DSC/NSD/F1 per label
  analysis/                     — lesion instance stats, component extraction
  met_labels.py                 — label constants (NETC=1, SNFH=2, ET=3, RC=4)

pyproject.toml                  — mbrats package + all Python dependencies

scripts/
  install.sh                    — install PyTorch + pip install -e .
  setup_env.sh                  — export nnUNet_raw/preprocessed/results/extTrainer + PYENV_VERSION
  prepare.sh                    — data prep pipeline (steps 1–4), --from N to resume
  train.sh                      — training pipeline (both plan configs), --from N to resume
  panoptica_evaluate.sh         — wrapper for brats-evaluate + brats-parse-metrics
```
