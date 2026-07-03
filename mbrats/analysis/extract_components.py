"""
Extract connected components from a segmentation file or directory.

Specify one or more label values to union before CC extraction.

Single-file mode — prints per-component table, optionally saves CC map:
    python src/extract_components.py --input input.nii.gz --labels 1 3 4
    python src/extract_components.py --input input.nii.gz --labels 1 3 4 --out cc.nii.gz

Directory mode — aggregates CC statistics across all cases:
    python src/extract_components.py --dir labelsTr/ --labels 1 3 4
    python src/extract_components.py --dir labelsTr/ --labels 1 3 4 --out results/cc_stats.json

Labels: 1=NETC  2=SNFH  3=ET  4=RC
"""

import argparse
import json
from pathlib import Path
import cc3d

import nibabel as nib
import numpy as np
from scipy import ndimage

LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
MIN_VOXELS = 27  # mm³ threshold for evaluable lesions


def extract_components(mask: np.ndarray, voxel_vol: float):
    """Label CCs sorted by size descending (largest = ID 1). Returns (labeled, sizes_mm3)."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return labeled, []

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    order = np.argsort(sizes)[::-1]

    new_labeled = np.zeros_like(labeled)
    for new_id, old_idx in enumerate(order, start=1):
        new_labeled[labeled == (old_idx + 1)] = new_id

    sizes_sorted = [int(sizes[i]) * voxel_vol for i in order]
    return new_labeled, sizes_sorted


def label_str(labels):
    return "+".join(f"{l}={LABEL_NAMES[l]}" for l in sorted(labels) if l in LABEL_NAMES)


def print_single(sizes_mm3, title):
    print(f"\nComponents for {title}:  n={len(sizes_mm3)}")
    if not sizes_mm3:
        return
    print(f"  {'ID':>4}  {'Vol (mm³)':>12}  {'Vol (cm³)':>10}  >=27mm³")
    print("  " + "-" * 46)
    for i, vol in enumerate(sizes_mm3, start=1):
        flag = "yes" if vol >= MIN_VOXELS else "no (ignored)"
        print(f"  {i:>4}  {vol:>12.1f}  {vol/1000:>10.3f}  {flag}")
    evaluable = sum(1 for v in sizes_mm3 if v >= MIN_VOXELS)
    print(f"\n  Total: {len(sizes_mm3)}  |  Evaluable (>=27mm³): {evaluable}")


def print_aggregate(all_sizes, n_cases, title):
    all_flat = [v for sizes in all_sizes for v in sizes]
    evaluable = [v for v in all_flat if v >= MIN_VOXELS]
    cases_with_any = sum(1 for s in all_sizes if s)
    components_per_case = [len(s) for s in all_sizes]
    evaluable_per_case = [sum(1 for v in s if v >= MIN_VOXELS) for s in all_sizes]

    print(f"\n{'='*60}")
    print(f"Directory summary — {title}")
    print(f"{'='*60}")
    print(f"  Cases processed:         {n_cases}")
    print(f"  Cases with any label:    {cases_with_any}")
    print(f"  Total components:        {len(all_flat)}")
    print(f"  Evaluable (>=27mm³):     {len(evaluable)}  ({100*len(evaluable)/max(len(all_flat),1):.1f}%)")
    print(f"  Avg components/case:     {np.mean(components_per_case):.1f}")
    print(f"  Avg evaluable/case:      {np.mean(evaluable_per_case):.1f}")

    if evaluable:
        p = np.percentile(evaluable, [25, 50, 75, 95, 99])
        print(f"\n  Volume of evaluable components (mm³):")
        print(f"    min={min(evaluable):.0f}  p25={p[0]:.0f}  p50={p[1]:.0f}  "
              f"p75={p[2]:.0f}  p95={p[3]:.0f}  p99={p[4]:.0f}  max={max(evaluable):.0f}")
        print(f"    mean={np.mean(evaluable):.0f}  std={np.std(evaluable):.0f}")

    # size bin histogram
    bins = [0, 27, 500, 5000, 20000, float("inf")]
    bin_labels = ["<27 (ignored)", "S 27-500", "M 500-5k", "L 5k-20k", "XL >20k"]
    total = len(all_flat)
    print(f"\n  Size distribution:")
    print(f"    {'Bin':<16} {'Count':>7}  {'%':>6}  bar")
    print("    " + "-" * 48)
    for i, bl in enumerate(bin_labels):
        c = sum(1 for v in all_flat if bins[i] <= v < bins[i+1])
        bar = "█" * int(25 * c / total) if total else ""
        print(f"    {bl:<16} {c:>7}  {100*c/total:>5.1f}%  {bar}")


def run_single(args):
    img = nib.load(args.input)
    seg = np.round(img.get_fdata()).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    voxel_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])

    mask = np.isin(seg, args.labels)
    title = label_str(args.labels)
    labeled, sizes_mm3 = extract_components(mask, voxel_vol)
    print_single(sizes_mm3, title)

    if args.out:
        out_img = nib.Nifti1Image(labeled.astype(np.int32), img.affine, img.header)
        out_img.header.set_data_dtype(np.int32)
        nib.save(out_img, args.out)
        print(f"\nSaved CC map → {args.out}")


def run_dir(args):
    files = sorted(args.dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No .nii.gz files in {args.dir}")

    title = label_str(args.labels)
    all_sizes = []

    for f in files:
        if f.name.startswith("._"):
            continue
        img = nib.load(f)
        seg = np.round(img.get_fdata()).astype(np.uint8)
        zooms = img.header.get_zooms()[:3]
        voxel_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])
        mask = np.isin(seg, args.labels)
        _, sizes_mm3 = extract_components(mask, voxel_vol)
        all_sizes.append(sizes_mm3)

    print_aggregate(all_sizes, len(files), title)

    if args.out:
        all_flat = [v for s in all_sizes for v in s]
        evaluable = [v for v in all_flat if v >= MIN_VOXELS]
        data = {
            "labels": args.labels,
            "n_cases": len(files),
            "n_components_total": len(all_flat),
            "n_evaluable": len(evaluable),
            "avg_components_per_case": float(np.mean([len(s) for s in all_sizes])),
            "volume_percentiles_mm3": {
                f"p{p}": float(np.percentile(evaluable, p)) for p in [25, 50, 75, 95, 99]
            } if evaluable else {},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved → {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single segmentation .nii.gz")
    group.add_argument("--dir", type=Path, help="Directory of .nii.gz files (aggregate mode)")
    parser.add_argument("--labels", type=int, nargs="+", required=True, choices=[1, 2, 3, 4],
                        help="Label values to union (1=NETC 2=SNFH 3=ET 4=RC)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output: CC map .nii.gz (single) or stats .json (dir)")
    args = parser.parse_args()

    if args.dir:
        run_dir(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
