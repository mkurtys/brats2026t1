"""
Per-lesion volume distribution and label composition analysis.

For each GT case:
  - finds individual lesions via connected components on the whole-tumor mask (labels 1+2+3+4)
  - records each lesion's volume (mm³) and which labels it contains

Usage:
    python src/lesion_analysis.py --gt nnunet_raw/Dataset001_BraTSMETS/labelsTr
"""

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

LABEL_NAMES = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
TUMOR_LABELS = {1, 2, 3, 4}
SIZE_BINS = [0, 27, 500, 5000, 20000, float("inf")]
SIZE_BIN_LABELS = ["<27 (ignored)", "S 27-500", "M 500-5k", "L 5k-20k", "XL >20k"]


def composition_name(label_set):
    return "+".join(LABEL_NAMES[l] for l in sorted(label_set) if l in LABEL_NAMES)


def analyze_case(gt_path: Path):
    img = nib.load(gt_path)
    gt = np.round(img.get_fdata()).astype(np.uint8)
    zooms = img.header.get_zooms()[:3]
    voxel_vol = float(zooms[0]) * float(zooms[1]) * float(zooms[2])

    tumor_mask = np.isin(gt, list(TUMOR_LABELS))
    labeled, n_lesions = ndimage.label(tumor_mask)

    lesions = []
    for i in range(1, n_lesions + 1):
        comp = labeled == i
        vol_mm3 = comp.sum() * voxel_vol
        labels_present = {l for l in TUMOR_LABELS if (gt[comp] == l).any()}
        lesions.append({"vol_mm3": vol_mm3, "labels": labels_present})

    return lesions


def bin_index(vol):
    for i in range(len(SIZE_BINS) - 1):
        if SIZE_BINS[i] <= vol < SIZE_BINS[i + 1]:
            return i
    return len(SIZE_BINS) - 2


def print_histogram(values, bins, bin_labels, title):
    counts = Counter(bin_index(v) for v in values)
    total = len(values)
    print(f"\n{title}  (n={total})")
    print(f"  {'Bin':<16} {'Count':>7}  {'%':>6}  bar")
    print("  " + "-" * 50)
    for i, label in enumerate(bin_labels):
        c = counts.get(i, 0)
        bar = "█" * int(30 * c / total) if total else ""
        print(f"  {label:<16} {c:>7}  {100*c/total:>5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gt_files = sorted(args.gt.glob("*.nii.gz"))
    if not gt_files:
        raise SystemExit(f"No .nii.gz files in {args.gt}")

    all_lesions = []
    n_cases = 0
    for gt_path in gt_files:
        if gt_path.name.startswith("._"):
            continue
        lesions = analyze_case(gt_path)
        all_lesions.extend(lesions)
        n_cases += 1

    print(f"\nCases: {n_cases}  |  Total lesions: {len(all_lesions)}")
    print(f"Avg lesions/case: {len(all_lesions)/n_cases:.1f}")

    # volume distribution (all lesions)
    vols = [l["vol_mm3"] for l in all_lesions]
    print_histogram(vols, SIZE_BINS, SIZE_BIN_LABELS, "Lesion volume distribution (all lesions)")

    # volume distribution excluding sub-threshold
    evaluable = [l for l in all_lesions if l["vol_mm3"] >= 27]
    vols_eval = [l["vol_mm3"] for l in evaluable]
    print(f"\nEvaluable lesions (>=27mm³): {len(evaluable)}  ({100*len(evaluable)/len(all_lesions):.1f}%)")
    p = np.percentile(vols_eval, [25, 50, 75, 95, 99])
    print(f"  Volume percentiles (mm³):  p25={p[0]:.0f}  p50={p[1]:.0f}  p75={p[2]:.0f}  p95={p[3]:.0f}  p99={p[4]:.0f}")
    print(f"  Mean: {np.mean(vols_eval):.0f}  Std: {np.std(vols_eval):.0f}  Min: {min(vols_eval):.0f}  Max: {max(vols_eval):.0f}")

    # composition breakdown
    comp_counts = Counter(composition_name(l["labels"]) for l in evaluable)
    total_eval = len(evaluable)
    print(f"\nLabel composition of evaluable lesions:")
    print(f"  {'Composition':<24} {'Count':>7}  {'%':>6}")
    print("  " + "-" * 42)
    for comp, count in sorted(comp_counts.items(), key=lambda x: -x[1]):
        print(f"  {comp:<24} {count:>7}  {100*count/total_eval:>5.1f}%")

    # composition × size bin
    print(f"\nComposition by size bin (evaluable lesions):")
    # get unique compositions sorted by frequency
    top_comps = [c for c, _ in comp_counts.most_common()]
    bin_labels_eval = SIZE_BIN_LABELS[1:]  # skip <27
    header = f"  {'Composition':<24}" + "".join(f"  {b:>12}" for b in bin_labels_eval)
    print(header)
    print("  " + "-" * (24 + 14 * len(bin_labels_eval)))
    for comp in top_comps:
        comp_lesions = [l for l in evaluable if composition_name(l["labels"]) == comp]
        row = f"  {comp:<24}"
        for i in range(1, len(SIZE_BIN_LABELS)):
            c = sum(1 for l in comp_lesions if bin_index(l["vol_mm3"]) == i)
            row += f"  {c:>12}"
        print(row)

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n_cases": n_cases,
            "n_lesions": len(all_lesions),
            "n_evaluable": len(evaluable),
            "composition_counts": dict(comp_counts),
            "volume_percentiles": {"p25": p[0], "p50": p[1], "p75": p[2], "p95": p[3], "p99": p[4]},
        }
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
