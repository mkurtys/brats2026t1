"""
Visual QA for the copy-paste augmentation: paste a handful of library
instances into a real case and render before/after slices, so paste
plausibility can be checked by eye rather than only by aggregate metrics.

Usage:
    python -m mbrats.analysis.visualize_copy_paste BraTS-MET-00002-000 --n 5
"""

import argparse
import sys
from pathlib import Path

import blosc2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mbrats" / "training"))
from copy_paste import compute_instance_weights, find_valid_offset, paste_instance  # noqa: E402

from mbrats.view_case import LABEL_COLORS, LABEL_NAMES, apply_overlay  # noqa: E402

DATA_FOLDER = Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres")
PASTEMASK_FOLDER = Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres_pastemasks")
LIBRARY_PATH = Path("nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl")
# channel order: 0=T1, 1=T1CE (contrast), 2=T2, 3=FLAIR
MODALITIES = [(1, 'T1CE'), (3, 'FLAIR')]
CONTEXT = 2  # slices either side of the paste's centre


def render_pair(before_image, before_seg, after_image, after_seg, offset, extent, out_path, title,
                 modalities=MODALITIES, show_masks=True):
    axis = 0  # axial: preprocessed arrays are (Z, Y, X); slice axis is Z (1.0mm), in-plane is near-isotropic
    centre = offset[axis] + extent[axis] // 2
    max_idx = before_image.shape[axis + 1] - 1
    indices = [max(0, min(max_idx, centre + o)) for o in range(-CONTEXT, CONTEXT + 1)]
    indices = list(dict.fromkeys(indices))

    # Window each modality once from the *before* (target) volume and reuse it
    # for the after row. Re-normalising per slice would rescale the whole brain
    # whenever a bright paste shifts the percentiles, masquerading as a global
    # intensity mismatch between before and after.
    windows = {}
    for ch, _ in modalities:
        fg = before_image[ch][before_image[ch] > 0]
        windows[ch] = np.percentile(fg, [1, 99]) if fg.size else (0.0, 1.0)

    def norm_win(x, ch):
        lo, hi = windows[ch]
        return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)

    rows = [(f'{name} before', ch, before_image[ch], before_seg) for ch, name in modalities]
    rows += [(f'{name} after', ch, after_image[ch], after_seg) for ch, name in modalities]
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(3 * len(indices), 3 * len(rows)), squeeze=False)

    labels_present = sorted(set(np.unique(before_seg).tolist() + np.unique(after_seg).tolist()) - {0})

    for col, idx in enumerate(indices):
        for row, (name, ch, bg, seg) in enumerate(rows):
            bg_sl = norm_win(np.take(bg, idx, axis=axis), ch)
            if show_masks:
                img = apply_overlay(bg_sl, np.take(seg, idx, axis=axis), labels_present)
            else:
                img = np.stack([bg_sl] * 3, axis=-1)
            ax = axes[row, col]
            ax.imshow(img, interpolation='bilinear', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(f"z={idx}", fontsize=8)

    patches = [mpatches.Patch(color=LABEL_COLORS[l], label=LABEL_NAMES.get(l, str(l))) for l in labels_present]
    if show_masks and patches:
        fig.legend(handles=patches, loc='lower center', ncol=len(patches), fontsize=9, bbox_to_anchor=(0.5, 0.0))
    for row, (name, *_rest) in enumerate(rows):
        fig.text(0.01, 1 - (row + 0.5) / len(rows), name, fontsize=9, fontweight='bold', va='center', rotation=90)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0.05, 0.04, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_id", help="Target case to paste into, e.g. BraTS-MET-00002-000")
    parser.add_argument("--n", type=int, default=5, help="Number of example pastes to generate")
    parser.add_argument("--wt_fraction", type=float, default=0.5, help="P(use core+edema crop vs core-only)")
    parser.add_argument("--min_wt_voxels", type=int, default=0,
                        help="Only paste instances with at least this many core+edema voxels (0 = no filter). "
                             "Overrides the default small-lesion sampling bias to inspect big lesions.")
    parser.add_argument("--out", type=Path, default=Path("results/copy_paste_diagnostics"))
    parser.add_argument("--no_masks", action="store_true", help="Render raw modality slices only, no label overlay")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image = blosc2.open(str(DATA_FOLDER / f"{args.case_id}.b2nd"))[:]
    seg = blosc2.open(str(DATA_FOLDER / f"{args.case_id}_seg.b2nd"))[:][0]
    valid_mask_full = blosc2.open(str(PASTEMASK_FOLDER / f"{args.case_id}_pastemask.b2nd"))[:]

    library = load_pickle(str(LIBRARY_PATH))
    weights = compute_instance_weights(library)
    if args.min_wt_voxels > 0:
        wt_sizes = np.array([inst['n_voxels_netc'] + inst['n_voxels_et'] + inst['n_voxels_rc']
                             + inst['n_voxels_snfh'] for inst in library])
        weights = weights * (wt_sizes >= args.min_wt_voxels)
        if weights.sum() == 0:
            raise SystemExit(f"No library instances with >= {args.min_wt_voxels} core+edema voxels")
        weights = weights / weights.sum()
    rng = np.random.default_rng(args.seed)

    made = 0
    attempts = 0
    carved_masks = []
    while made < args.n and attempts < args.n * 20:
        attempts += 1
        idx = rng.choice(len(library), p=weights)
        inst = library[idx]
        use_wt = rng.random() < args.wt_fraction
        source_image = inst['wt_image_crop'] if use_wt else inst['core_image_crop']
        source_seg = inst['wt_seg_crop'] if use_wt else inst['core_seg_crop']

        offset = find_valid_offset(valid_mask_full, source_seg.shape, rng=rng)
        print(f"valid offsef {offset}")
        if offset is None:
            continue

        before_image = image.copy()
        before_seg = seg.copy()

        after_image = image.copy()
        after_seg = seg.copy()
        carved_mask, lam = paste_instance(after_image, after_seg, source_image, source_seg, offset, rng=rng)
        carved_masks.append(carved_mask)
        if not carved_mask.any():
            continue

        composition = 'core+edema' if use_wt else 'core-only'
        title = (f"{args.case_id}  src={inst['case_id']}  {composition}  "
                 f"lam={lam:+.2f}  carved_voxels={int(carved_mask.sum())}")
        out_path = args.out / f"{args.case_id}_paste{made}.png"
        render_pair(before_image, before_seg, after_image, after_seg, offset, carved_mask.shape, out_path, title,
                    show_masks=not args.no_masks)
        made += 1

    print(f"\nGenerated {made}/{args.n} examples ({attempts} attempts)")


if __name__ == "__main__":
    main()
