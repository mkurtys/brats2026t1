"""
Connected-component size filtering for BraTS 2026 predictions.

Motivation (see memory scoring-and-fp-diagnosis): the ranking DSC/NSD is computed
per *region* (ET/TC/WT/RC), and every false-positive lesion — of ANY size — adds a
zero to the region's lesion-wise DSC average. Removing small spurious components
therefore lifts DSC directly. Filtering is applied per evaluation region, matching
how panoptica scores.

Regions (label-based, per config_mets.yaml):
  et = {3}         rc = {4}
  tc = {1,3}       wt = {1,2,3}

Filtering removes, per region, every connected component (26-connectivity) whose
voxel count is below that region's threshold, by zeroing the underlying labels.
Voxel spacing in this dataset is 1 mm isotropic, so voxel count == volume in mm^3.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# region name -> set of labels whose union defines the region mask
REGION_LABELS: dict[str, tuple[int, ...]] = {
    "et": (3,),
    "tc": (1, 3),
    "wt": (1, 2, 3),
    "rc": (4,),
}

# 26-connectivity in 3D (a lesion touching only at a corner is one component)
_STRUCT = np.ones((3, 3, 3), dtype=np.uint8)


def _region_mask(seg: np.ndarray, region: str) -> np.ndarray:
    labels = REGION_LABELS[region]
    m = np.isin(seg, labels)
    return m


def small_component_voxels(mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Boolean array (mask's shape) that is True on voxels belonging to a connected
    component smaller than `min_size`. `min_size <= 1` is a no-op (nothing removed).
    """
    if min_size is None or min_size <= 1 or not mask.any():
        return np.zeros_like(mask, dtype=bool)
    cc, n = ndimage.label(mask, structure=_STRUCT)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    # component sizes indexed 0..n (0 = background); bincount is O(voxels)
    sizes = np.bincount(cc.ravel())
    small_ids = np.nonzero(sizes < min_size)[0]
    small_ids = small_ids[small_ids != 0]  # never flag background
    if len(small_ids) == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(cc, small_ids)


def filter_small_components(seg: np.ndarray, thresholds: dict[str, int]) -> np.ndarray:
    """
    Return a filtered copy of `seg`. For each region in `thresholds`, flag voxels in
    sub-threshold components of that region's mask; then zero the union of all flagged
    voxels. Flags are computed independently from the ORIGINAL seg (so region order
    does not matter), which matches panoptica's independent per-region scoring.
    """
    remove = np.zeros(seg.shape, dtype=bool)
    for region, k in thresholds.items():
        if k is None or k <= 1:
            continue
        remove |= small_component_voxels(_region_mask(seg, region), k)
    if not remove.any():
        return seg.copy()
    out = seg.copy()
    out[remove] = 0
    return out


# ── CLI: apply given thresholds to a directory of predictions ──────────────────

def _apply_to_dir(pred_dir, out_dir, thresholds):
    import nibabel as nib
    from pathlib import Path

    pred_dir, out_dir = Path(pred_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pred_dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No .nii.gz in {pred_dir}")
    n_removed = 0
    for f in files:
        im = nib.load(str(f))
        seg = np.asarray(im.dataobj).astype(np.int16)
        filt = filter_small_components(seg, thresholds)
        n_removed += int((seg != filt).sum() > 0)
        nib.save(nib.Nifti1Image(filt, im.affine, im.header), str(out_dir / f.name))
    print(f"Filtered {len(files)} cases ({n_removed} changed) -> {out_dir}")
    print(f"Thresholds: {thresholds}")


def main():
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, help="Input prediction directory")
    p.add_argument("--out", required=True, help="Output directory for filtered masks")
    p.add_argument("--thresholds", required=True,
                   help='JSON dict of per-region min voxel size, e.g. \'{"et":10,"tc":10,"wt":20,"rc":50}\' '
                        'or path to a JSON file')
    args = p.parse_args()

    t = args.thresholds
    try:
        thresholds = json.loads(t)
    except json.JSONDecodeError:
        with open(t) as fh:
            thresholds = json.load(fh)
    thresholds = {k: int(v) for k, v in thresholds.items()}
    _apply_to_dir(args.pred, args.out, thresholds)


if __name__ == "__main__":
    main()
