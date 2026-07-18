"""
Visual QA for the paste-location mask (build_paste_location_masks.py): render
every input channel of a case as its own row, with the component masks
overlaid, so the CSF/brain/core heuristics can be checked by eye across
modalities — CSF, for instance, is dark on T1, bright on T2, dark on FLAIR,
so it should look different in each row while the red csf-like overlay stays
put.

    green  = final valid paste region
    red    = csf-like (excluded)
    blue   = dilated tumor core / RC (excluded)

Usage:
    python -m mbrats.analysis.visualize_pastemask BraTS-MET-00002-000
"""

import argparse
from pathlib import Path

import blosc2
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from mbrats.preprocessing.build_paste_location_masks import (
    CORE_LABELS, LESION_DILATION, brain_mask_of, compute_valid_mask, csf_exclusion_mask)
from mbrats.view_case import normalise

DATA_FOLDER = Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres")
CHANNEL_NAMES = ['T1', 'T1CE', 'T2', 'FLAIR', 'T1c-T1n', 'T1c/T1n']

VALID_COLOR = [0.2, 0.8, 0.2]
CSF_COLOR = [0.9, 0.2, 0.2]
CORE_COLOR = [0.2, 0.5, 1.0]


def component_masks(image, seg):
    """The pieces of compute_valid_mask, for visualization."""
    brain = brain_mask_of(image)
    core = ndimage.binary_dilation(np.isin(seg, CORE_LABELS), iterations=LESION_DILATION)
    csf = csf_exclusion_mask(image, brain)
    valid = compute_valid_mask(image, seg)
    return brain, core, csf, valid


def overlay(bg_norm, layers, alpha=0.5):
    rgb = np.stack([bg_norm] * 3, axis=-1)
    for mask, color in layers:
        if mask.any():
            rgb[mask] = rgb[mask] * (1 - alpha) + np.array(color) * alpha
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_id")
    parser.add_argument("--n_slices", type=int, default=5)
    parser.add_argument("--slices", type=int, nargs='+', default=None,
                        help="Explicit axial (axis-0) slice indices; default picks the most csf-like")
    parser.add_argument("--out", type=Path, default=Path("results/copy_paste_diagnostics/pastemask"))
    args = parser.parse_args()

    image = blosc2.open(str(DATA_FOLDER / f"{args.case_id}.b2nd"))[:]
    seg = blosc2.open(str(DATA_FOLDER / f"{args.case_id}_seg.b2nd"))[:][0]
    brain, core, csf, valid = component_masks(image, seg)

    print(f"{args.case_id}: brain {int(brain.sum())}  csf-like {int(csf.sum())} "
          f"({100 * csf.sum() / max(brain.sum(), 1):.2f}% of brain)  "
          f"core-excl {int(core.sum())}  valid {int(valid.sum())} ({100 * valid.mean():.1f}% of volume)")

    if args.slices:
        indices = args.slices
    else:
        # pick axial slices (axis 0) where csf-like is most present, else spread through the brain
        key = csf if csf.any() else brain
        counts = key.sum(axis=(1, 2))
        order = np.argsort(counts)[::-1]
        indices = sorted(order[:args.n_slices].tolist())

    n_ch = image.shape[0]
    fig, axes = plt.subplots(n_ch, len(indices), figsize=(3 * len(indices), 3 * n_ch), squeeze=False)
    for col, z in enumerate(indices):
        layers = [(valid[z], VALID_COLOR), (core[z], CORE_COLOR), (csf[z], CSF_COLOR)]
        for ch in range(n_ch):
            img = overlay(normalise(image[ch, z]), layers)
            ax = axes[ch, col]
            ax.imshow(img, interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}",
                              fontsize=10, fontweight='bold', rotation=90)
                ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
            if ch == 0:
                ax.set_title(f"z={z}", fontsize=8)

    handles = [mpatches.Patch(color=VALID_COLOR, label='valid paste region'),
               mpatches.Patch(color=CSF_COLOR, label='csf-like (excluded)'),
               mpatches.Patch(color=CORE_COLOR, label='core/RC dilated (excluded)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{args.case_id}  paste-location mask components per channel", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.97])

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.case_id}_pastemask_channels.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
