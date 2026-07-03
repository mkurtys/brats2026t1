"""
Precompute connected-component instance locations for nnUNet training.

Reads each preprocessed .pkl + _seg.b2nd, computes CCs of the tumor core
(ET | NETC), and writes 'tc_rc_instances' back into the .pkl as a flat list:

  tc_rc_instances = [array_cc1, array_cc2, ...]

Each array has shape (N, 4) matching nnUNet's class_locations format:
  col 0 = channel (always 0), cols 1-3 = x, y, z in preprocessed space.

Safe to re-run: skips cases already processed.

Usage:
    python -m mbrats.preprocessing.precompute_tc_rc_instances
    python -m mbrats.preprocessing.precompute_tc_rc_instances --folder nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetResEncUNetLPlans_3d_fullres
"""

import argparse
from pathlib import Path

import blosc2
import numpy as np
from mbrats import met_labels
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
from scipy import ndimage
from tqdm import tqdm


def compute_tc_rc_instances(seg_3d: np.ndarray) -> list:
    """
    Returns a flat list of CC arrays for TC | RC (ET | NETC | RC).
    Each array is (N, 4): [channel=0, x, y, z] matching nnUNet class_locations format.
    """
    mask = (
        (seg_3d == met_labels.ENHANCING_TUMOR) |
        (seg_3d == met_labels.NONENHANCING_TUMOR_CORE) |
        (seg_3d == met_labels.RESECTION_CAVITY)
    )
    if not mask.any():
        return []
    labeled, n = ndimage.label(mask)
    ccs = []
    for i in range(1, n + 1):
        voxels_xyz = np.argwhere(labeled == i)
        channel_col = np.zeros((len(voxels_xyz), 1), dtype=np.int16)
        voxels = np.concatenate([channel_col, voxels_xyz], axis=1).astype(np.int16)
        ccs.append(voxels)
    return ccs


def process_folder(folder: Path, force: bool = False):
    pkl_files = sorted(folder.glob("*.pkl"))
    if not pkl_files:
        raise SystemExit(f"No .pkl files found in {folder}")

    already_done = 0
    for pkl_path in tqdm(pkl_files, desc="Computing instance locations"):
        props = load_pickle(str(pkl_path))
        if 'tc_rc_instances' in props and not force:
            already_done += 1
            continue

        seg_path = pkl_path.with_name(pkl_path.stem + "_seg.b2nd")
        if not seg_path.exists():
            print(f"  SKIP {pkl_path.name}: no seg file found")
            continue

        seg = blosc2.open(str(seg_path))[:]  # (1, H, W, D)
        props['tc_rc_instances'] = compute_tc_rc_instances(seg[0])
        save_pickle(props, str(pkl_path))

    total = len(pkl_files)
    processed = total - already_done
    print(f"\nDone. Processed {processed}/{total} cases ({already_done} already had tc_rc_instances).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path,
                        default=Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres"))
    parser.add_argument("--force", action="store_true", help="Recompute even if already present")
    args = parser.parse_args()
    process_folder(args.folder, force=args.force)


if __name__ == "__main__":
    main()
