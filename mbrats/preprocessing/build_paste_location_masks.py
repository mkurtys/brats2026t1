"""
Precompute, per training case, a boolean mask of voxels where it is safe to
paste a copy-paste lesion instance (see build_lesion_library.py).

A voxel is valid if all of:
  1. Inside the brain — not skull-stripped background. Detected via local
     variance (real tissue has texture; background is a flat constant), not
     a zero check — this data comes from nnU-Net's already z-normalised
     preprocessed folder, where background is shifted to some non-zero
     constant by normalisation, so `image != 0` never actually excludes it.
  2. Not too close to the brain surface — the brain mask is eroded by a few
     voxels so pasted content doesn't sit right at the boundary.
  3. Not on or near an existing tumor core (NETC/ET) or RC — dilated and
     excluded so a paste never overlaps/corrupts a real discrete lesion
     annotation. Existing SNFH (surrounding edema) is *not* excluded — it's
     a diffuse structure, and a new lesion growing within/near existing
     edematous tissue is anatomically plausible, so it's left available.
  4. Not inside a *large* CSF space — ventricles, major fissures, cisterns.
     CSF is detected by a crude intensity heuristic: dark on T1, bright on T2,
     and *suppressed* (dark) on FLAIR. FLAIR attenuates CSF, so it's the
     T2-bright + FLAIR-dark combination that distinguishes CSF from edema
     (bright on both). Thresholds are brain-relative z-scores (the channels'
     global z-norm centres brain tissue well above 0, so absolute thresholds
     would select air). The csf mask is then morphologically *opened* to keep
     only large CSF and drop thin sulci: a paste may liberally cross a small
     gyrus, but never a ventricle/fissure. Deliberately simple (no atlas) —
     good enough to avoid the worst offenders, not perfect anatomy.

Channel order (see convert_to_nnunet.py DATASET_JSON): 0=T1, 1=T1CE, 2=T2,
3=FLAIR, 4=T1c-T1n, 5=T1c/T1n.

Writes '<case_id>_pastemask.b2nd' (bool) next to each case's data.
Safe to re-run: skips cases that already have a mask file.

Usage:
    python -m mbrats.preprocessing.build_paste_location_masks
    python -m mbrats.preprocessing.build_paste_location_masks --folder nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres
"""

import argparse
from pathlib import Path

import blosc2
import numpy as np
from mbrats import met_labels
from scipy import ndimage
from tqdm import tqdm

BRAIN_EROSION = 4     # voxels to erode the brain mask away from the skull-strip boundary
LESION_DILATION = 4   # voxels to dilate existing tumor core/RC by, to keep pastes clearly separate
CSF_OPENING = 2       # morphological opening of the csf-like mask: keep only *large* CSF
                       # (ventricles, major fissures, cisterns) and drop thin sulci, so a paste
                       # may liberally cross a small gyrus but never a big CSF space. 0 disables.
CORE_LABELS = (met_labels.NONENHANCING_TUMOR_CORE, met_labels.ENHANCING_TUMOR, met_labels.RESECTION_CAVITY)
T1_CHANNEL = 0
T2_CHANNEL = 2
FLAIR_CHANNEL = 3
N_STRUCTURAL_CHANNELS = 4  # 0=T1,1=T1CE,2=T2,3=FLAIR; 4,5 are derived ratios, unreliable in air
FOREGROUND_TAU = 0.0       # structural-channel mean above this = tissue, below = background/air.
                            # Air z-normalises to ~-0.3 across structural channels, brain to ~+2.5.
# CSF thresholds are in *brain-relative* z units (standardised within the brain
# mask), NOT the raw channel values. nnU-Net z-normalises over the whole volume
# incl. air, so brain tissue sits around +2.5 and air near -0.3 — absolute
# thresholds like "T1 < -0.5" select air, never CSF. Re-standardising within the
# brain puts white matter near 0 so these signs mean what they say.
CSF_T1_Z = -0.5        # CSF is dark on T1 (below brain mean)...
CSF_T2_Z = 0.5         # ...bright on T2 (above brain mean)...
CSF_FLAIR_Z = -0.5     # ...and *suppressed* (dark) on FLAIR. FLAIR attenuates CSF,
                        # so FLAIR-bright is edema/lesion, not CSF — the T2-bright +
                        # FLAIR-dark combo is what separates the two.
LOCAL_VARIANCE_WINDOW = 3   # voxels, side length of the local-variance window for brain detection
LOCAL_VARIANCE_THRESHOLD = 1e-6  # background is exactly flat; any real texture clears this easily


def _brain_z(channel: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Standardise a channel using mean/std of its in-brain voxels only."""
    vals = channel[brain_mask]
    return (channel - vals.mean()) / (vals.std() + 1e-6)


def brain_mask_of(image: np.ndarray) -> np.ndarray:
    """Eroded in-brain mask. Two criteria, because neither alone excludes air:
    local-variance texture drops the flat interior of the background, but its
    window straddles the brain/air boundary and leaks a rim of air voxels; an
    intensity floor on the structural channels drops that rim (air sits well
    below FOREGROUND_TAU). Then erode away from the skull-strip boundary."""
    t1 = image[T1_CHANNEL]
    local_mean = ndimage.uniform_filter(t1, size=LOCAL_VARIANCE_WINDOW)
    local_var = ndimage.uniform_filter(t1 ** 2, size=LOCAL_VARIANCE_WINDOW) - local_mean ** 2
    textured = local_var > LOCAL_VARIANCE_THRESHOLD
    foreground = image[:N_STRUCTURAL_CHANNELS].mean(axis=0) > FOREGROUND_TAU
    return ndimage.binary_erosion(textured & foreground, iterations=BRAIN_EROSION)


def csf_like_mask(image: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """CSF/ventricle-like: dark T1, bright T2, suppressed FLAIR — all judged
    relative to the case's brain tissue (see CSF_*_Z)."""
    return ((_brain_z(image[T1_CHANNEL], brain_mask) < CSF_T1_Z)
            & (_brain_z(image[T2_CHANNEL], brain_mask) > CSF_T2_Z)
            & (_brain_z(image[FLAIR_CHANNEL], brain_mask) < CSF_FLAIR_Z))


def csf_exclusion_mask(image: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """csf_like_mask, opened to keep only large CSF (see CSF_OPENING) — thin
    sulci are dropped so small gyri stay crossable."""
    csf = csf_like_mask(image, brain_mask)
    return ndimage.binary_opening(csf, iterations=CSF_OPENING) if CSF_OPENING else csf


def compute_valid_mask(image: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """image: (C, H, W, D) float32, seg: (H, W, D) int. Returns (H, W, D) bool."""
    brain_mask = brain_mask_of(image)

    core_mask = np.isin(seg, CORE_LABELS)
    core_mask = ndimage.binary_dilation(core_mask, iterations=LESION_DILATION)

    csf_excl = csf_exclusion_mask(image, brain_mask)

    return brain_mask & ~core_mask & ~csf_excl


def build_masks(folder: Path, force: bool = False):
    seg_files = sorted(folder.glob("*_seg.b2nd"))
    if not seg_files:
        raise SystemExit(f"No *_seg.b2nd files found in {folder}")

    # Write pastemasks to a sibling folder, NOT into the nnU-Net data folder:
    # nnUNetv2_train -f all enumerates cases by scanning the data folder, and
    # co-located *_pastemask.b2nd files get mistaken for cases (looked-up seg
    # then missing). Trainer reads them from '<data_folder>_pastemasks' too.
    out_folder = Path(str(folder) + "_pastemasks")
    out_folder.mkdir(parents=True, exist_ok=True)

    already_done = 0
    fractions = []
    for seg_path in tqdm(seg_files, desc="Computing paste location masks"):
        case_id = seg_path.name[:-len("_seg.b2nd")]
        out_path = out_folder / f"{case_id}_pastemask.b2nd"
        if out_path.exists() and not force:
            already_done += 1
            continue

        image_path = folder / f"{case_id}.b2nd"
        if not image_path.exists():
            print(f"  SKIP {case_id}: no image file found")
            continue

        image = blosc2.open(str(image_path))[:]
        seg = blosc2.open(str(seg_path))[:][0]  # (1,H,W,D) -> (H,W,D)

        valid_mask = compute_valid_mask(image, seg)
        fractions.append(valid_mask.mean())

        if out_path.exists():
            out_path.unlink()
        blosc2.asarray(valid_mask, urlpath=str(out_path), mode='w')

    total = len(seg_files)
    processed = total - already_done
    print(f"\nDone. Processed {processed}/{total} cases ({already_done} already had a mask).")
    if fractions:
        fractions = np.array(fractions)
        print(f"Valid fraction of brain volume: mean {fractions.mean():.1%}, "
              f"min {fractions.min():.1%}, max {fractions.max():.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path,
                        default=Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres"))
    parser.add_argument("--force", action="store_true", help="Recompute even if a mask already exists")
    args = parser.parse_args()
    build_masks(args.folder, force=args.force)


if __name__ == "__main__":
    main()
