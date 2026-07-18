"""
Visual QA for the *carve* step of the copy-paste augmentation, in isolation
from the paste. For a handful of library instances this sweeps the CarveMix
threshold `lam` from maximum shrink (deep interior) through the original
lesion to maximum grow (band into surrounding context), and colour-codes each
carved region against the original mask so the shape-following carve can be
checked by eye:

    green = kept (original & carved)
    red   = removed (original & ~carved)  -> shrinking into the interior
    blue  = added   (~original & carved)  -> growing outward into context

The sweep is deterministic (`signed_distance` thresholded at evenly spaced
lam), spanning the same [min_dist/2, -min_dist] range carve_mask samples from,
rather than random draws — so a single figure shows the full carve range.

Usage:
    python -m mbrats.analysis.visualize_carve --n 6
    python -m mbrats.analysis.visualize_carve --case_id BraTS-MET-00002-000
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mbrats" / "training"))
from copy_paste import compute_instance_weights, foreground_mask, signed_distance  # noqa: E402

from mbrats.view_case import normalise  # noqa: E402

LIBRARY_PATH = Path("nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl")
CONTRAST_CHANNEL = 1  # T1CE

KEPT_COLOR = [0.2, 0.8, 0.2]
REMOVED_COLOR = [0.9, 0.2, 0.2]
ADDED_COLOR = [0.2, 0.5, 1.0]
ORIG_COLOR = [1.0, 0.5, 0.0]


def overlay_masks(bg_norm, colored_masks, alpha=0.55):
    """colored_masks: list of (bool 2D mask, rgb). Later entries win overlaps."""
    rgb = np.stack([bg_norm] * 3, axis=-1)
    for mask, color in colored_masks:
        if mask.any():
            rgb[mask] = rgb[mask] * (1 - alpha) + np.array(color) * alpha
    return rgb


def spotlight(bg_norm, region, dim=0.2):
    """Full intensity inside `region`, dimmed outside — shows the actual
    image content the carve selects (what paste_instance would composite),
    since carve itself changes only the label mask, not the pixels."""
    return np.stack([np.where(region, bg_norm, bg_norm * dim)] * 3, axis=-1)


def best_axial_slice(mask):
    """Axial slice holding the most mask voxels. Preprocessed arrays are
    (Z, Y, X) with Z (axis 0) the slice axis, so the axial plane is (Y, X)."""
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def lam_sweep(dist, n):
    """Evenly spaced lam over carve_mask's [min_dist/2, -min_dist] range
    (shrink -> original -> grow). min_dist is the most-interior distance (<0)."""
    min_dist = dist.min()
    return np.linspace(min_dist / 2, -min_dist, n)


def render_instance(axes_row, inst, source_image, source_seg, n_carves, show_masks=True):
    lesion_mask = source_seg != 0
    z_ref = best_axial_slice(lesion_mask)

    # same foreground gate paste_instance applies: the carve footprint must
    # exclude out-of-brain air the grow can balloon into, so counts and shapes
    # here match what actually gets pasted.
    brain = foreground_mask(source_image) | lesion_mask

    dist = signed_distance(lesion_mask)
    lams = lam_sweep(dist, n_carves)
    orig_n = int(lesion_mask.sum())

    # reference column: pristine original mask on its best slice
    bg_ref = normalise(source_image[CONTRAST_CHANNEL, z_ref])
    orig_sl_ref = lesion_mask[z_ref]
    ref_img = overlay_masks(bg_ref, [(orig_sl_ref, ORIG_COLOR)]) if show_masks else spotlight(bg_ref, orig_sl_ref)
    ax = axes_row[0]
    ax.imshow(ref_img, interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(f"original\nz={z_ref}  {orig_n} vox", fontsize=8)

    for col, lam in enumerate(lams, start=1):
        carved = (dist < lam) & brain
        # slice showing the most change (added|removed vs original); fall back
        # to the reference slice when this lam leaves the mask untouched.
        change = carved != lesion_mask
        z = int(np.argmax(change.sum(axis=(1, 2)))) if change.any() else z_ref

        bg = normalise(source_image[CONTRAST_CHANNEL, z])
        orig_sl = lesion_mask[z]
        carved_sl = carved[z]
        if show_masks:
            kept = orig_sl & carved_sl
            removed = orig_sl & ~carved_sl
            added = ~orig_sl & carved_sl
            img = overlay_masks(bg, [(kept, KEPT_COLOR), (removed, REMOVED_COLOR), (added, ADDED_COLOR)])
        else:
            img = spotlight(bg, carved_sl)

        carved_n = int(carved.sum())
        pct = 100.0 * (carved_n - orig_n) / max(orig_n, 1)
        ax = axes_row[col]
        ax.imshow(img, interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"lam={lam:+.1f}  z={z}\n{carved_n} vox ({pct:+.0f}%)", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case_id", default=None, help="Only use instances carved from this source case")
    parser.add_argument("--n", type=int, default=6, help="Number of library instances to render (rows)")
    parser.add_argument("--carves", type=int, default=6, help="Number of lam steps per instance (sweep columns)")
    parser.add_argument("--wt_fraction", type=float, default=0.5, help="P(use core+edema crop vs core-only)")
    parser.add_argument("--out", type=Path, default=Path("results/copy_paste_diagnostics/carve_sweep.png"))
    parser.add_argument("--no_masks", action="store_true", help="Render raw T1CE slices only, no carve overlay")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    library = load_pickle(str(LIBRARY_PATH))
    weights = compute_instance_weights(library)
    rng = np.random.default_rng(args.seed)

    candidates = [i for i in range(len(library))
                  if args.case_id is None or library[i]['case_id'] == args.case_id]
    if not candidates:
        raise SystemExit(f"No library instances for case_id={args.case_id}")

    cand_weights = weights[candidates]
    cand_weights = cand_weights / cand_weights.sum()
    n = min(args.n, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False, p=cand_weights)

    n_cols = args.carves + 1  # + original reference column
    fig, axes = plt.subplots(n, n_cols, figsize=(2.4 * n_cols, 2.6 * n), squeeze=False)

    for row, idx in enumerate(chosen):
        inst = library[int(idx)]
        use_wt = rng.random() < args.wt_fraction
        source_image = inst['wt_image_crop'] if use_wt else inst['core_image_crop']
        source_seg = inst['wt_seg_crop'] if use_wt else inst['core_seg_crop']
        render_instance(axes[row], inst, source_image, source_seg, args.carves, show_masks=not args.no_masks)

        composition = 'core+edema' if use_wt else 'core-only'
        fig.text(0.005, 1 - (row + 0.5) / n, f"{inst['case_id']}\n{composition}",
                 fontsize=7, fontweight='bold', va='center', rotation=90)

    if not args.no_masks:
        patches = [mpatches.Patch(color=ORIG_COLOR, label='original'),
                   mpatches.Patch(color=KEPT_COLOR, label='kept'),
                   mpatches.Patch(color=REMOVED_COLOR, label='removed (shrink)'),
                   mpatches.Patch(color=ADDED_COLOR, label='added (grow)')]
        fig.legend(handles=patches, loc='lower center', ncol=len(patches), fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("CarveMix carve sweep: shrink (left) -> original -> grow (right)", fontsize=11)
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.96])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
