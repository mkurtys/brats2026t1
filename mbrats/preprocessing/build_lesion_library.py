"""
Build a library of individual lesion instances (image + label crops) from the
preprocessed training set, for copy-paste augmentation.

Algorithm, per case:
  1. Binarize on tumor core | RC (NETC=1, ET=3, RC=4) and run connected-
     component analysis on that union. Seeding on TC|RC (not SNFH) avoids
     the classic failure mode where diffuse edema merges genuinely separate
     lesions into one giant blob — same reason precompute_instance_locations
     excludes SNFH from its seed mask.
  2. For each core component, compute its NETC/ET/RC voxel composition.
  3. Search for surrounding edema: dilate the core mask slightly and collect
     whichever SNFH connected components it touches (i.e. edema that
     actually belongs to this lesion, not unrelated edema elsewhere).
  4. Store two crops per instance: a tight "tumor core" bbox (NETC|ET|RC
     only) and an expanded "tumor core + edema" bbox (core plus the
     matched surrounding SNFH). Paste-time code picks whichever
     composition it wants.

If a neighboring lesion's bbox overlaps, voxels belonging to a different
component are zeroed out so a paste never drags in an unrelated lesion.

Usage:
    python -m mbrats.preprocessing.build_lesion_library
    python -m mbrats.preprocessing.build_lesion_library --folder nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres --out nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl
"""

import argparse
from pathlib import Path

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_pickle
from mbrats import met_labels
from scipy import ndimage
from tqdm import tqdm

MARGIN = 4          # voxels of context padding around a bbox
EDEMA_DILATION = 2  # voxels to dilate the core mask when searching for touching SNFH

TC_RC_LABELS = (met_labels.NONENHANCING_TUMOR_CORE, met_labels.ENHANCING_TUMOR, met_labels.RESECTION_CAVITY)


def _make_crop(image, seg, valid_mask, coords, shape):
    lo = np.clip(coords.min(axis=0) - MARGIN, 0, None)
    hi = np.minimum(coords.max(axis=0) + MARGIN + 1, shape)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))

    seg_crop = seg[sl].copy()
    valid_crop = valid_mask[sl]
    seg_crop[~valid_crop] = 0  # drop anything not belonging to this instance (incl. neighboring lesions)

    image_crop = image[(slice(None),) + sl].copy()
    return image_crop, seg_crop


def extract_instances(case_id: str, image: np.ndarray, seg: np.ndarray) -> list:
    """image: (C, H, W, D) float32, seg: (H, W, D) int16/int. Returns list of instance dicts."""
    instances = []
    shape = seg.shape

    core_seed_mask = np.isin(seg, TC_RC_LABELS)
    if not core_seed_mask.any():
        return instances

    core_labeled, n_core = ndimage.label(core_seed_mask)
    snfh_mask = seg == met_labels.FLAIR_HYPERINTENSITY
    snfh_labeled, n_snfh = ndimage.label(snfh_mask)

    for i in range(1, n_core + 1):
        comp = core_labeled == i
        coords = np.argwhere(comp)

        # find SNFH components this lesion's core actually touches (its own surrounding edema)
        edema_mask = np.zeros_like(comp)
        if n_snfh > 0:
            dilated = ndimage.binary_dilation(comp, iterations=EDEMA_DILATION)
            touching_ids = set(np.unique(snfh_labeled[dilated & snfh_mask])) - {0}
            if touching_ids:
                edema_mask = np.isin(snfh_labeled, list(touching_ids))

        core_image_crop, core_seg_crop = _make_crop(image, seg, comp, coords, shape)

        wt_valid = comp | edema_mask
        wt_coords = np.argwhere(wt_valid)
        wt_image_crop, wt_seg_crop = _make_crop(image, seg, wt_valid, wt_coords, shape)

        instances.append({
            'case_id': case_id,
            'n_voxels_netc': int((comp & (seg == met_labels.NONENHANCING_TUMOR_CORE)).sum()),
            'n_voxels_et': int((comp & (seg == met_labels.ENHANCING_TUMOR)).sum()),
            'n_voxels_rc': int((comp & (seg == met_labels.RESECTION_CAVITY)).sum()),
            'n_voxels_snfh': int(edema_mask.sum()),
            'core_image_crop': core_image_crop,
            'core_seg_crop': core_seg_crop,
            'wt_image_crop': wt_image_crop,
            'wt_seg_crop': wt_seg_crop,
        })

    return instances


def build_library(folder: Path, out_path: Path):
    seg_files = sorted(folder.glob("*_seg.b2nd"))
    if not seg_files:
        raise SystemExit(f"No *_seg.b2nd files found in {folder}")

    library = []
    for seg_path in tqdm(seg_files, desc="Extracting lesion instances"):
        case_id = seg_path.name[:-len("_seg.b2nd")]
        image_path = folder / f"{case_id}.b2nd"
        if not image_path.exists():
            print(f"  SKIP {case_id}: no image file found")
            continue

        image = blosc2.open(str(image_path))[:]
        seg = blosc2.open(str(seg_path))[:][0]  # (1,H,W,D) -> (H,W,D)

        library.extend(extract_instances(case_id, image, seg))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(library, str(out_path))

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\nSaved {len(library)} instances to {out_path} ({size_mb:.0f} MB)")
    for name, key in [('NETC', 'n_voxels_netc'), ('ET', 'n_voxels_et'),
                       ('RC', 'n_voxels_rc'), ('SNFH (matched edema)', 'n_voxels_snfh')]:
        counts = np.array([inst[key] for inst in library])
        present = counts > 0
        print(f"  {name}: present in {present.sum()}/{len(library)} instances, "
              f"size range {counts[present].min() if present.any() else 0}-{counts.max()} voxels")
    with_edema = sum(1 for inst in library if inst['n_voxels_snfh'] > 0)
    with_rc = sum(1 for inst in library if inst['n_voxels_rc'] > 0)
    rc_and_tumor = sum(1 for inst in library
                        if inst['n_voxels_rc'] > 0 and (inst['n_voxels_netc'] + inst['n_voxels_et']) > 0)
    print(f"  instances with matched surrounding edema: {with_edema}")
    print(f"  instances containing RC: {with_rc} (of which fused with adjacent tumor: {rc_and_tumor})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path,
                        default=Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres"))
    parser.add_argument("--out", type=Path,
                        default=Path("nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl"))
    args = parser.parse_args()
    build_library(args.folder, args.out)


if __name__ == "__main__":
    main()
