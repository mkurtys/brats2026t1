# BraTS 2026 MET Task 1 — Baseline Report

## Solution Overview

Standard nnU-Net v2 3d_fullres pipeline, trained on fold 0 of 5, using all 1296 available training cases (650 standard BraTS 2025 MET + 646 UCSD longitudinal). No architectural changes, no custom augmentation, no postprocessing. This serves as a lower-bound baseline before ensembling, postprocessing, and additional folds.

---

## Data

| | |
|---|---|
| Training cases | 1296 (650 standard + 646 UCSD longitudinal) |
| Validation cases (no GT) | 179 |
| Modalities | T1n, T1c, T2w, T2-FLAIR |
| Image space | Mixed: SRI24 atlas (328 cases) and native space (1268 cases) |
| Voxel size | ~1mm³ isotropic, 240×240×155 |

**Labels (challenge spec):**
| Label | Name | Description |
|---|---|---|
| 1 | NETC | Necrotic tumor core |
| 2 | SNFH | Surrounding non-enhancing FLAIR hyperintensity (edema) |
| 3 | ET | Enhancing tumor |
| 4 | RC | Resection cavity (post-treatment only) |

**Known data issues fixed:**
- `BraTS-MET-01094-002`: label 6 in source (129 voxels) → remapped to 0
- `BraTS-MET-01094-003`: replaced with corrected label from official corrected-labels bundle
- macOS `._` hidden files filtered throughout

**Note on dataset.json:** The labels section has ET and SNFH semantic names swapped (`"ET": 2, "SNFH": 3`) relative to the challenge spec (which is `SNFH=2, ET=3`). Since no voxel remapping was performed during conversion, the integer values in all prediction files are correct for submission. The mislabeling is only in the metadata.

---

## nnU-Net Configuration

| Parameter | Value |
|---|---|
| Version | nnU-Net v2.8.0 (dev install) |
| Configuration | 3d_fullres |
| Patch size | 112 × 160 × 128 |
| Batch size | 2 |
| Spacing | [1.0, 0.898, 0.859] mm |
| Architecture | 6-stage PlainConvUNet, features 32→64→128→256→320→320 |
| Normalization | Z-score with brain mask, all 4 channels |
| Epochs | 1000 |
| Avg epoch time | 72 s (RTX 3090) |
| Total training time | ~20 h (fold 0 only) |
| 3d_lowres | Dropped (image size identical to fullres) |

---

## Commands Used

```bash
# Environment setup (run before any nnUNet command)
source scripts/setup_env.sh
# sets: nnUNet_raw, nnUNet_preprocessed, nnUNet_results, PYENV_VERSION=brats

# 1. Fix bad label (label 6 → 0) in one case
python -m mbrats.preprocessing.fix_labels

# 2. Convert raw BraTS data to nnU-Net format (symlinks)
python -m mbrats.preprocessing.convert_to_nnunet \
  --train_dir /media/mkurtys/T7/datasets/brats2026/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  --output_dir nnunet_raw/Dataset001_BraTSMETS \
  --corrected_labels \
      /media/mkurtys/T7/datasets/brats2026/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels \
      /media/mkurtys/T7/datasets/brats2026/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels \
  --val_dir /media/mkurtys/T7/datasets/brats2026/Validation

# 3. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -np 8

# 4. Train fold 0
nnUNetv2_train 1 3d_fullres 0 --npz

# 5. Inference on validation set (no GT)
nnUNetv2_predict \
  -d 1 -c 3d_fullres -f 0 \
  -i nnunet_raw/Dataset001_BraTSMETS/imagesVal \
  -o predictions/fold0_val

# 6. Evaluate fold 0 CV predictions (have GT)
python -m mbrats.evaluation.evaluate \
  --pred nnunet_results/Dataset001_BraTSMETS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
  --gt nnunet_raw/Dataset001_BraTSMETS/labelsTr \
  --out results/fold0_cv_eval.json
```

---

## Results — Fold 0 CV Validation (260 cases)

### Overall

| Label | DSC | NSD (2mm) | n |
|---|---|---|---|
| NETC | 0.581 | 0.714 | 118 |
| SNFH | 0.771 | 0.887 | 220 |
| ET | 0.760 | 0.866 | 237 |
| RC | 0.480 | 0.572 | 36 |
| **Mean** | **0.648** | **0.760** | |

Cases with GT volume < 27 mm³ are excluded per challenge spec.

### By GT Volume (DSC / NSD)

**NETC**
| Bin | DSC | NSD | n |
|---|---|---|---|
| S (27–500 mm³) | 0.479 | 0.638 | 54 |
| M (500–5k mm³) | 0.661 | 0.813 | 38 |
| L (5k–20k mm³) | 0.845 | 0.891 | 16 |
| XL (>20k mm³) | 0.913 | 0.906 | 2 |

**SNFH**
| Bin | DSC | NSD | n |
|---|---|---|---|
| S (27–500 mm³) | 0.562 | 0.825 | 35 |
| M (500–5k mm³) | 0.712 | 0.873 | 68 |
| L (5k–20k mm³) | 0.860 | 0.927 | 44 |
| XL (>20k mm³) | 0.902 | 0.930 | 70 |

**ET**
| Bin | DSC | NSD | n |
|---|---|---|---|
| S (27–500 mm³) | 0.700 | 0.838 | 71 |
| M (500–5k mm³) | 0.806 | 0.912 | 99 |
| L (5k–20k mm³) | 0.799 | 0.875 | 51 |
| XL (>20k mm³) | 0.824 | 0.875 | 10 |

**RC**
| Bin | DSC | NSD | n |
|---|---|---|---|
| S (27–500 mm³) | 0.267 | 0.357 | 11 |
| M (500–5k mm³) | 0.487 | 0.618 | 16 |
| L (5k–20k mm³) | 0.714 | 0.744 | 8 |
| XL (>20k mm³) | 0.850 | 0.825 | 1 |

---

## Key Observations

1. **ET is the easiest label** — least size-dependent (S bin still gets 0.70 DSC). High T1c contrast makes it reliably detectable regardless of size.

2. **NETC small-lesion gap is the biggest weakness** — 0.48 DSC for S bin vs 0.91 for XL. Small necrotic cores are hard to distinguish from artifact and are easily missed.

3. **SNFH NSD >> DSC for small lesions** (0.83 vs 0.56) — the shape/boundary is approximately right but the volume is off. A size-based calibration or postprocessing could help.

4. **RC is severely underrepresented** — only 36 GT cases with RC in fold 0 val split, and no RC cases in the standard training set (all RC is in UCSD longitudinal subset). The model barely sees this class.

5. **NSD consistently ~0.1 above DSC** across all classes — indicates reasonable boundary quality relative to volumetric overlap.

6. **dataset.json label names are swapped** (ET↔SNFH). Does not affect predictions but should be fixed in `convert_to_nnunet.py` to avoid confusion.

---

## Suggested Improvements

### High impact, low effort
- **Train all 5 folds + ensemble** — the single biggest expected gain. Fold ensemble typically adds +2–4 DSC points in nnUNet.
- **Fix dataset.json label names** — correct `"ET": 2, "SNFH": 3` → `"SNFH": 2, "ET": 3` in `convert_to_nnunet.py`.

### Postprocessing
- **Connected-component filtering** — remove predicted components below a voxel threshold (e.g., <10 voxels). Grid-search threshold on CV folds. Most useful for NETC and RC false positives.
- **RC suppression for likely treatment-naive cases** — if no RC in GT split, the model may hallucinate small RC regions. Could zero out RC predictions below a confidence or size threshold.

### Training improvements
- **Oversample small-lesion cases** — nnUNet already does foreground oversampling but doesn't weight by class size. A custom sampler that oversamples cases with small NETC could help the S-bin gap.
- **Larger patch size** — current patch (112×160×128) covers ~78% of median image. Increasing to full-image patches (144×192×160) would give more context at the cost of smaller batch size (1 instead of 2). Worth testing.
- **Region-based training** — train on hierarchical regions (TC = NETC+ET, WT = NETC+ET+SNFH) as additional targets, as in BraTS glioma solutions.

### Data
- **Register all cases to a common space** — the mix of native-space (1268) and SRI24-space (328) cases may be hurting consistency. Registering everything to SRI24 could improve generalization.
- **Additional public data** — BraTS 2024 MET and earlier years are public and could be appended (check challenge rules: current rules say BraTS 2026 dataset only).

### Evaluation
- **Add lesion-wise F1** — the challenge also scores detection. Implement connected-component matching with IoU threshold to compute per-case precision/recall.
- **Confidence intervals** — bootstrap the per-case metric vectors to report 95% CIs.
