"""
Break down lesion-wise detection recall and matched-lesion Dice by GT lesion size.

Reads the raw panoptica JSON produced by evaluate.py (--out path, not the parsed
CSV) and bins every reference (GT) lesion instance across all subjects/classes
by volume (mm^3), reporting how many were detected (matched to a prediction)
and the mean sq_dsc among the ones that were. Also reports the mean global
(whole-volume, non-lesion-wise) segmentation Dice per class, and an over/under-
segmentation breakdown: instance-level FP vs FN counts (spurious vs missed
lesions) and predicted-vs-reference volume ratio (boundary-level bias on
lesions that were at least partially found).

Usage:
    python -m mbrats.evaluation.lesion_size_stats --json results/checkpoint500_fold0_cv_eval.json
"""

import argparse
import json

import numpy as np

BINS = [
    ("tiny   (<27mm3)", 0, 27),
    ("small  (27-100)", 27, 100),
    ("medium (100-500)", 100, 500),
    ("large  (500-2000)", 500, 2000),
    ("huge   (>2000)", 2000, float("inf")),
]

CLASSES = ["et", "tc", "wt", "rc"]


def bin_of(volume: float) -> str:
    for name, lo, hi in BINS:
        if lo <= volume < hi:
            return name
    return BINS[-1][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", required=True, help="Raw panoptica JSON from evaluate.py (--out)")
    args = parser.parse_args()

    data = json.load(open(args.json))

    stats = {c: {b[0]: {"n": 0, "matched": 0, "dsc": []} for b in BINS} for c in CLASSES}
    global_dsc = {c: [] for c in CLASSES}
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES}
    volume_pred = {c: 0.0 for c in CLASSES}
    volume_ref = {c: 0.0 for c in CLASSES}

    for subject in data["metrics"]:
        if "error" in subject:
            continue
        for c in CLASSES:
            region = subject.get(c)
            if not region:
                continue
            gdsc = region.get("global_bin_dsc")
            if gdsc is not None and not np.isnan(gdsc):
                global_dsc[c].append(gdsc)
            for k in ("tp", "fp", "fn"):
                counts[c][k] += region.get(k, 0) or 0
            vp = region.get("global_bin_volume_pred")
            vr = region.get("global_bin_volume_ref")
            if vp is not None and vr is not None:
                volume_pred[c] += vp
                volume_ref[c] += vr
            for inst in region.get("reference_instances", []):
                volume = inst.get("volume")
                if volume is None:
                    continue
                b = bin_of(volume)
                stats[c][b]["n"] += 1
                if inst.get("is_matched") == 1:
                    stats[c][b]["matched"] += 1
                    if inst.get("sq_dsc") is not None:
                        stats[c][b]["dsc"].append(inst["sq_dsc"])

    print(f"{'class':6s} {'n_subj':>7s} {'mean_global_dsc':>16s}")
    for c in CLASSES:
        vals = global_dsc[c]
        mean_dsc = np.mean(vals) if vals else float("nan")
        print(f"{c.upper():6s} {len(vals):7d} {mean_dsc:16.4f}")

    print(f"\n{'class':6s} {'tp':>6s} {'fp':>6s} {'fn':>6s} {'fp/fn':>7s} {'vol_pred/vol_ref':>17s}")
    for c in CLASSES:
        cnt = counts[c]
        fp_fn_ratio = cnt["fp"] / cnt["fn"] if cnt["fn"] else float("inf")
        vol_ratio = volume_pred[c] / volume_ref[c] if volume_ref[c] else float("nan")
        print(f"{c.upper():6s} {cnt['tp']:6d} {cnt['fp']:6d} {cnt['fn']:6d} {fp_fn_ratio:7.2f} {vol_ratio:17.3f}")
    print("(fp/fn > 1: more spurious extra lesions than missed ones -> over-detection."
          " vol_pred/vol_ref > 1: predicted regions bigger than GT on average -> over-segmentation.)")

    for c in CLASSES:
        print(f"\n=== {c.upper()} ===")
        print(f"{'bin':20s} {'n_gt':>6s} {'detected':>9s} {'recall':>7s} {'mean_dsc(matched)':>18s}")
        for name, _, _ in BINS:
            s = stats[c][name]
            recall = s["matched"] / s["n"] if s["n"] else float("nan")
            mean_dsc = np.mean(s["dsc"]) if s["dsc"] else float("nan")
            print(f"{name:20s} {s['n']:6d} {s['matched']:9d} {recall:7.2%} {mean_dsc:18.4f}")


if __name__ == "__main__":
    main()
