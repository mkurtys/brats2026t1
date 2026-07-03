"""
Generate a PNG montage for a BraTS case: unlabelled / GT / prediction rows.

Each row shows multiple slices along the chosen axis. The best slice (most
foreground voxels in GT ∪ pred) is selected automatically, or override with --slice.

Usage:
    python src/view_case.py BraTS-MET-01152-003
    python src/view_case.py BraTS-MET-01152-003 --label 4 --axis 2
    python src/view_case.py BraTS-MET-01152-003 --label 4 --slice 120
    python src/view_case.py BraTS-MET-01152-003 --pred predictions/fold0_val --out /tmp/review

Axes: 0=sagittal, 1=coronal, 2=axial (default)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

LABEL_NAMES  = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
LABEL_COLORS = {1: [1.0, 0.4, 0.0], 2: [0.2, 0.6, 1.0], 3: [1.0, 0.9, 0.0], 4: [0.9, 0.2, 0.9]}
AXIS_NAMES   = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}

IMAGES_DIR = Path('nnunet_raw/Dataset001_BraTSMETS/imagesTr')
GT_DIR     = Path('nnunet_raw/Dataset001_BraTSMETS/labelsTr')
MODALITY_SUFFIXES = {'t1n': '_0000', 't1c': '_0001', 't2w': '_0002', 't2f': '_0003'}


def load_nifti(path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.header.get_zooms()[:3]


def best_slice(union_mask, axis):
    counts = union_mask.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(counts))


def normalise(bg):
    lo, hi = np.percentile(bg[bg > 0], [1, 99]) if bg.max() > 0 else (0, 1)
    return np.clip((bg - lo) / (hi - lo + 1e-6), 0, 1)


def apply_overlay(bg_norm, seg, labels, alpha=0.5):
    rgb = np.stack([bg_norm] * 3, axis=-1)
    for lbl in labels:
        mask = seg == lbl
        if not mask.any():
            continue
        rgb[mask] = rgb[mask] * (1 - alpha) + np.array(LABEL_COLORS[lbl]) * alpha
    return rgb


def get_slice(vol, idx, axis):
    sl = np.take(vol, idx, axis=axis)
    # orient consistently: sagittal/coronal need rotation to appear upright
    if axis < 2:
        sl = np.rot90(sl)
    return sl


def make_montage(case_id, modality, pred_dir, labels, axis, slice_idx, n_context, out_dir):
    mod_suffix = MODALITY_SUFFIXES[modality]
    bg_path   = IMAGES_DIR / f'{case_id}{mod_suffix}.nii.gz'
    gt_path   = GT_DIR     / f'{case_id}.nii.gz'
    pred_path = pred_dir   / f'{case_id}.nii.gz'

    if not bg_path.exists():
        raise FileNotFoundError(f'Image not found: {bg_path}')
    if not gt_path.exists():
        raise FileNotFoundError(f'GT not found: {gt_path}')

    bg_vol, _   = load_nifti(bg_path)
    gt_seg, _   = load_nifti(gt_path)
    gt_seg      = gt_seg.astype(np.int16)
    has_pred    = pred_path.exists()
    pred_seg    = load_nifti(pred_path)[0].astype(np.int16) if has_pred else np.zeros_like(gt_seg)

    # determine centre slice
    union = np.zeros(gt_seg.shape, dtype=bool)
    for lbl in labels:
        union |= (gt_seg == lbl) | (pred_seg == lbl)
    centre = slice_idx if slice_idx is not None else (
        best_slice(union, axis) if union.any() else gt_seg.shape[axis] // 2
    )

    # collect slices: centre ± context offsets
    max_idx = gt_seg.shape[axis] - 1
    offsets = list(range(-n_context, n_context + 1))
    indices = [max(0, min(max_idx, centre + o)) for o in offsets]
    indices = list(dict.fromkeys(indices))  # deduplicate while preserving order

    n_cols = len(indices)
    rows   = [('No label', None)] + [('GT', gt_seg)] + ([('Pred', pred_seg)] if has_pred else [])
    n_rows = len(rows)

    bg_norm = normalise(bg_vol)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False)

    for col, idx in enumerate(indices):
        bg_sl = get_slice(bg_norm, idx, axis)
        for row, (row_label, seg) in enumerate(rows):
            ax = axes[row, col]
            if seg is None:
                img = np.stack([bg_sl] * 3, axis=-1)
            else:
                seg_sl = get_slice(seg, idx, axis)
                img = apply_overlay(bg_sl, seg_sl, labels)
            ax.imshow(img, interpolation='nearest', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                marker = ' ✦' if idx == centre else ''
                ax.set_title(f'{AXIS_NAMES[axis][0]}{idx}{marker}', fontsize=8)

    # voxel counts
    for lbl in labels:
        gt_n  = int((gt_seg  == lbl).sum())
        pr_n  = int((pred_seg == lbl).sum())
        print(f'  {LABEL_NAMES.get(lbl, lbl)}: GT={gt_n} vox  Pred={pr_n} vox')

    # legend
    patches = [mpatches.Patch(color=LABEL_COLORS[l], label=LABEL_NAMES.get(l, str(l)))
               for l in labels if l in LABEL_COLORS]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches), fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    label_tag = '+'.join(LABEL_NAMES.get(l, str(l)) for l in labels)
    title = f'{case_id}  |  {modality.upper()}  |  {AXIS_NAMES[axis]}  |  labels: {label_tag}'
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0.08, 0.04, 1, 0.97])

    # row labels in figure coordinates (after tight_layout fixes axes positions)
    for row, (row_label, _) in enumerate(rows):
        # y centre of this row in figure coords
        axs_in_row = axes[row]
        y0 = min(ax.get_position().y0 for ax in axs_in_row)
        y1 = max(ax.get_position().y1 for ax in axs_in_row)
        fig.text(0.01, (y0 + y1) / 2, row_label, fontsize=10, fontweight='bold',
                 va='center', ha='left', rotation=90)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{case_id}_{modality}_{label_tag}_{AXIS_NAMES[axis].lower()}.png'
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case_id', help='e.g. BraTS-MET-01152-003')
    parser.add_argument('--label', type=int, action='append', dest='labels',
                        help='Label to highlight (repeatable). Default: all. 1=NETC 2=SNFH 3=ET 4=RC')
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2],
                        help='Slice axis: 0=sagittal 1=coronal 2=axial (default: 2)')
    parser.add_argument('--slice', type=int, default=None, dest='slice_idx',
                        help='Override slice index (default: auto best slice)')
    parser.add_argument('--context', type=int, default=2,
                        help='Number of slices either side of centre (default: 2 → 5 cols)')
    parser.add_argument('--modality', default='t1c', choices=list(MODALITY_SUFFIXES),
                        help='Background modality (default: t1c)')
    parser.add_argument('--pred', type=Path,
                        default=Path('nnunet_results/Dataset001_BraTSMETS/'
                                     'nnUNetTrainerBraTS_250epochs__nnUNetPlans__3d_fullres/'
                                     'fold_0/validation'),
                        help='Directory with predicted .nii.gz files')
    parser.add_argument('--out', type=Path, default=Path('results/montages'),
                        help='Output directory (default: results/montages)')
    args = parser.parse_args()

    labels = args.labels if args.labels else list(LABEL_NAMES.keys())
    make_montage(args.case_id, args.modality, args.pred, labels,
                 args.axis, args.slice_idx, args.context, args.out)


if __name__ == '__main__':
    main()
