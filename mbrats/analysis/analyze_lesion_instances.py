"""
Analyze connected-component instances of NETC+ET+RC (labels 1, 3, 4).

SNFH (label 2) is treated as background — lesion instances are defined by
the tumor core / cavity mask only, so distant lesions aren't incorrectly
merged through SNFH bridges.

Reads from preprocessed blosc2 seg files (nnUNetPlans_3d_fullres folder).

Usage:
    python src/analyze_lesion_instances.py
    python src/analyze_lesion_instances.py --folder nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres
    python src/analyze_lesion_instances.py --out results/lesion_instance_stats.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import blosc2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

LESION_LABELS = [1, 3, 4]  # NETC, ET, RC — SNFH excluded
LABEL_NAMES = {1: "NETC", 3: "ET", 4: "RC"}
SIZE_BINS = [0, 27, 500, 5000, 20000, float("inf")]
SIZE_BIN_NAMES = ["<27 (tiny)", "S 27-500", "M 500-5k", "L 5k-20k", "XL >20k"]


def composition_key(label_set):
    return "+".join(LABEL_NAMES[l] for l in sorted(label_set))


def analyze_seg(seg_3d: np.ndarray, voxel_vol: float) -> list[dict]:
    """
    Find CCs of NETC+ET+RC, return list of per-component dicts:
      vol_mm3, voxels_{NETC,ET,RC}, labels_present
    """
    mask = np.isin(seg_3d, LESION_LABELS)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []

    components = []
    for cc_id in range(1, n + 1):
        cc = labeled == cc_id
        vol = int(cc.sum()) * voxel_vol
        vox = {l: int((seg_3d[cc] == l).sum()) for l in LESION_LABELS}
        present = frozenset(l for l, v in vox.items() if v > 0)
        components.append({"vol_mm3": vol, "voxels": vox, "labels": present})

    return components


def bin_idx(vol):
    for i in range(len(SIZE_BINS) - 1):
        if SIZE_BINS[i] <= vol < SIZE_BINS[i + 1]:
            return i
    return len(SIZE_BINS) - 2


def print_report(all_components: list[dict], per_case_counts: list[int], n_cases: int):
    total = len(all_components)
    evaluable = [c for c in all_components if c["vol_mm3"] >= 27]
    n_eval = len(evaluable)

    print(f"\n{'='*60}")
    print(f"Lesion instance analysis  (NETC+ET+RC, SNFH excluded)")
    print(f"{'='*60}")
    print(f"  Cases:                {n_cases}")
    print(f"  Total instances:      {total}")
    print(f"  Evaluable (>=27mm³):  {n_eval}  ({100*n_eval/max(total,1):.1f}%)")
    print(f"  Avg instances/case:   {np.mean(per_case_counts):.1f}  "
          f"(p50={np.median(per_case_counts):.0f}  max={max(per_case_counts)})")

    vols = [c["vol_mm3"] for c in evaluable]
    if vols:
        p = np.percentile(vols, [25, 50, 75, 95, 99])
        print(f"\n  Evaluable volume (mm³):")
        print(f"    p25={p[0]:.0f}  p50={p[1]:.0f}  p75={p[2]:.0f}  p95={p[3]:.0f}  p99={p[4]:.0f}")
        print(f"    mean={np.mean(vols):.0f}  min={min(vols):.0f}  max={max(vols):.0f}")

    # size bin histogram
    all_vols = [c["vol_mm3"] for c in all_components]
    print(f"\n  Size distribution (all instances):")
    print(f"    {'Bin':<16} {'Count':>7}  {'%':>6}  bar")
    print("    " + "-" * 50)
    for i, name in enumerate(SIZE_BIN_NAMES):
        c = sum(1 for v in all_vols if bin_idx(v) == i)
        bar = "█" * int(30 * c / total) if total else ""
        print(f"    {name:<16} {c:>7}  {100*c/total:>5.1f}%  {bar}")

    # composition breakdown (evaluable only)
    comp_counts = Counter(composition_key(c["labels"]) for c in evaluable)
    print(f"\n  Composition of evaluable instances:")
    print(f"    {'Labels':<20} {'Count':>7}  {'%':>6}")
    print("    " + "-" * 38)
    for comp, cnt in sorted(comp_counts.items(), key=lambda x: -x[1]):
        print(f"    {comp:<20} {cnt:>7}  {100*cnt/n_eval:>5.1f}%")

    # composition × size bin
    top_comps = [c for c, _ in comp_counts.most_common()]
    bin_names_eval = SIZE_BIN_NAMES[1:]  # skip tiny
    print(f"\n  Composition × size bin (evaluable):")
    header = f"    {'Composition':<20}" + "".join(f"  {b:>12}" for b in bin_names_eval)
    print(header)
    print("    " + "-" * (20 + 14 * len(bin_names_eval)))
    for comp in top_comps:
        row_cs = [c for c in evaluable if composition_key(c["labels"]) == comp]
        row = f"    {comp:<20}"
        for i in range(1, len(SIZE_BIN_NAMES)):
            cnt = sum(1 for c in row_cs if bin_idx(c["vol_mm3"]) == i)
            row += f"  {cnt:>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path,
                        default=Path("nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    seg_files = sorted(args.folder.glob("*_seg.b2nd"))
    if not seg_files:
        raise SystemExit(f"No *_seg.b2nd files in {args.folder}")

    all_components = []
    per_case_counts = []

    for seg_path in tqdm(seg_files, desc="Analyzing"):
        seg = blosc2.open(str(seg_path))[:]  # (1, H, W, D)
        # voxel volume: assume 1mm³ (preprocessed to median spacing ~1mm)
        components = analyze_seg(seg[0], voxel_vol=1.0)
        all_components.extend(components)
        per_case_counts.append(len(components))

    n_cases = len(seg_files)
    print_report(all_components, per_case_counts, n_cases)

    if args.out:
        evaluable = [c for c in all_components if c["vol_mm3"] >= 27]
        vols = [c["vol_mm3"] for c in evaluable]
        comp_counts = Counter(composition_key(c["labels"]) for c in evaluable)
        data = {
            "n_cases": n_cases,
            "n_instances_total": len(all_components),
            "n_evaluable": len(evaluable),
            "avg_instances_per_case": float(np.mean(per_case_counts)),
            "composition_counts": dict(comp_counts),
            "volume_percentiles_mm3": {
                f"p{p}": float(np.percentile(vols, p)) for p in [25, 50, 75, 95, 99]
            } if vols else {},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
