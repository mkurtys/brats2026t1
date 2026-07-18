"""
Generate side-by-side example pastes contrasting degenerate (near
single-voxel) library instances against genuinely-shaped tiny lesions, to
visually confirm the compute_instance_weights over-concentration bug found
via eda_copy_paste_weights.py.

Usage:
    python -m mbrats.analysis.eda_copy_paste_examples BraTS-MET-00002-000 --n 3
"""

import argparse
import sys
from pathlib import Path

import blosc2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mbrats" / "training"))
from copy_paste import find_valid_offset, paste_instance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_copy_paste import render_pair  # noqa: E402

DATA_FOLDER = Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres")
LIBRARY_PATH = Path("nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl")


def instance_size(inst):
    return inst['n_voxels_netc'] + inst['n_voxels_et'] + inst['n_voxels_rc'] + inst['n_voxels_snfh']


def run_examples(case_id: str, library: list, size_lo: int, size_hi: int, tag: str, n: int,
                  valid_mask_full, image, seg, out_dir: Path, seed: int):
    candidates = [inst for inst in library if size_lo <= instance_size(inst) <= size_hi]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    made = 0
    for inst in candidates:
        if made >= n:
            break
        source_image, source_seg = inst['wt_image_crop'], inst['wt_seg_crop']
        offset = find_valid_offset(valid_mask_full, source_seg.shape, rng=rng)
        if offset is None:
            continue

        before_image = image.copy()
        before_seg = seg.copy()
        after_image = image.copy()
        after_seg = seg.copy()
        carved_mask, lam = paste_instance(after_image, after_seg, source_image, source_seg, offset, rng=rng)
        if not carved_mask.any():
            continue

        title = (f"[{tag}] {case_id}  src={inst['case_id']}  source_size={instance_size(inst)}  "
                 f"lam={lam:+.2f}  carved_voxels={int(carved_mask.sum())}")
        out_path = out_dir / f"{tag}_{case_id}_{made}.png"
        render_pair(before_image, before_seg, after_image, after_seg, offset, carved_mask.shape, out_path, title)
        made += 1
    print(f"[{tag}] generated {made}/{n} examples from {len(candidates)} candidates in size range "
          f"[{size_lo}, {size_hi}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_id", help="Target case to paste into, e.g. BraTS-MET-00002-000")
    parser.add_argument("--n", type=int, default=3, help="Number of examples per category")
    parser.add_argument("--out", type=Path, default=Path("eda"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image = blosc2.open(str(DATA_FOLDER / f"{args.case_id}.b2nd"))[:]
    seg = blosc2.open(str(DATA_FOLDER / f"{args.case_id}_seg.b2nd"))[:][0]
    valid_mask_full = blosc2.open(str(DATA_FOLDER / f"{args.case_id}_pastemask.b2nd"))[:]
    library = load_pickle(str(LIBRARY_PATH))

    run_examples(args.case_id, library, 1, 1, "degenerate_size1", args.n,
                 valid_mask_full, image, seg, args.out, args.seed)
    run_examples(args.case_id, library, 15, 27, "healthy_tiny_15to27", args.n,
                 valid_mask_full, image, seg, args.out, args.seed)


if __name__ == "__main__":
    main()
