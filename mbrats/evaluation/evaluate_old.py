"""
Evaluate segmentation predictions against ground truth.

Computes per-class DSC and NSD (2mm tolerance) matching the BraTS 2026 spec.
Only evaluates structures with GT volume > 27 mm³.

Usage:
    python src/evaluate.py --pred predictions/fold0_val --gt nnunet_raw/Dataset001_BraTSMETS/labelsTr

Labels (challenge spec):
    1=NETC, 2=SNFH, 3=ET, 4=RC
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import surface_distance
from scipy import ndimage

LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
# Hierarchical regions (label sets): ET, TC=NETC+ET, WT=NETC+SNFH+ET
REGIONS = {"ET": {3}, "TC": {1, 3}, "WT": {1, 2, 3}}
NSD_TOLERANCE_MM = 2.0
MIN_GT_VOXELS = 27  # 27 mm³ at 1mm³/voxel

# Size bins (mm³): edges defining Small / Medium / Large / XL
SIZE_BINS = [27, 500, 5000, 20000, float("inf")]
SIZE_LABELS = ["S (27-500)", "M (500-5k)", "L (5k-20k)", "XL (>20k)"]


def dice(pred_mask, gt_mask):
    tp = (pred_mask & gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    return 2 * tp / denom if denom > 0 else float("nan")


def nsd(pred_mask, gt_mask, spacing_mm):
    if not gt_mask.any() and not pred_mask.any():
        return float("nan")
    if not gt_mask.any() or not pred_mask.any():
        return 0.0
    sd = surface_distance.compute_surface_distances(gt_mask, pred_mask, spacing_mm)
    return surface_distance.compute_surface_dice_at_tolerance(sd, NSD_TOLERANCE_MM)


def lesion_detection(pred_mask, gt_mask, voxel_vol, iou_threshold=0.1):
    """
    Returns (TP, FP, FN) for lesion-wise detection.
    GT lesions < MIN_GT_VOXELS are ignored.
    A predicted component matches a GT component if IoU >= iou_threshold.
    Each GT lesion can be matched at most once (greedy, largest-IoU first).
    """
    gt_labeled, n_gt = ndimage.label(gt_mask)
    pred_labeled, n_pred = ndimage.label(pred_mask)

    # filter GT lesions below size threshold
    gt_ids = [i for i in range(1, n_gt + 1) if (gt_labeled == i).sum() >= MIN_GT_VOXELS]
    pred_ids = list(range(1, n_pred + 1))

    matched_gt = set()
    matched_pred = set()

    # build IoU matrix
    candidates = []
    for gid in gt_ids:
        g = gt_labeled == gid
        for pid in pred_ids:
            p = pred_labeled == pid
            inter = (g & p).sum()
            if inter == 0:
                continue
            union = (g | p).sum()
            iou = inter / union
            if iou >= iou_threshold:
                candidates.append((iou, gid, pid))

    # greedy match highest-IoU first
    for iou, gid, pid in sorted(candidates, reverse=True):
        if gid not in matched_gt and pid not in matched_pred:
            matched_gt.add(gid)
            matched_pred.add(pid)

    tp = len(matched_gt)
    fn = len(gt_ids) - tp
    fp = len(pred_ids) - len(matched_pred)
    return tp, fp, fn


def evaluate_case(pred_path: Path, gt_path: Path):
    pred_img = nib.load(pred_path)
    gt_img = nib.load(gt_path)
    pred = np.round(pred_img.get_fdata()).astype(np.uint8)
    gt = np.round(gt_img.get_fdata()).astype(np.uint8)

    zooms = tuple(float(z) for z in gt_img.header.get_zooms()[:3])
    voxel_vol = zooms[0] * zooms[1] * zooms[2]

    results = {}
    for label, name in LABELS.items():
        gt_mask = gt == label
        gt_voxels = int(gt_mask.sum())
        gt_vol_mm3 = gt_voxels * voxel_vol
        if gt_voxels < MIN_GT_VOXELS:
            results[name] = {"dsc": float("nan"), "nsd": float("nan"), "gt_vol_mm3": gt_vol_mm3, "skipped": True}
            continue
        pred_mask = pred == label
        results[name] = {
            "dsc": dice(pred_mask, gt_mask),
            "nsd": nsd(pred_mask, gt_mask, zooms),
            "gt_vol_mm3": gt_vol_mm3,
            "skipped": False,
        }

    for label, name in LABELS.items():
        gt_mask = gt == label
        pred_mask = pred == label
        tp, fp, fn = lesion_detection(pred_mask, gt_mask, voxel_vol)
        results[name].setdefault("detection", {})
        results[name]["detection"] = {"tp": tp, "fp": fp, "fn": fn}

    for region_name, label_set in REGIONS.items():
        gt_mask = np.isin(gt, list(label_set))
        gt_voxels = int(gt_mask.sum())
        gt_vol_mm3 = gt_voxels * voxel_vol
        if gt_voxels < MIN_GT_VOXELS:
            results[f"region_{region_name}"] = {"dsc": float("nan"), "gt_vol_mm3": gt_vol_mm3, "skipped": True}
            continue
        pred_mask = np.isin(pred, list(label_set))
        results[f"region_{region_name}"] = {
            "dsc": dice(pred_mask, gt_mask),
            "gt_vol_mm3": gt_vol_mm3,
            "skipped": False,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Prediction directory")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth label directory")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    pred_files = sorted(args.pred.glob("*.nii.gz"))
    if not pred_files:
        raise SystemExit(f"No .nii.gz files found in {args.pred}")

    per_case = {}
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = args.gt / f"{case_id}.nii.gz"
        if not gt_path.exists():
            print(f"  SKIP {case_id}: no GT found")
            continue
        per_case[case_id] = evaluate_case(pred_path, gt_path)

    # aggregate (nanmean = only cases where GT exists and > 27mm³)
    print(f"\nEvaluated {len(per_case)} cases\n")
    print(f"{'Label':<8} {'DSC':>8} {'NSD':>8}  (n evaluated)")
    print("-" * 38)
    summary = {}
    for label, name in LABELS.items():
        dscs = [v[name]["dsc"] for v in per_case.values() if not np.isnan(v[name]["dsc"])]
        nsds = [v[name]["nsd"] for v in per_case.values() if not np.isnan(v[name]["nsd"])]
        mean_dsc = float(np.mean(dscs)) if dscs else float("nan")
        mean_nsd = float(np.mean(nsds)) if nsds else float("nan")
        print(f"{name:<8} {mean_dsc:>8.4f} {mean_nsd:>8.4f}  (n={len(dscs)})")
        summary[name] = {"dsc": mean_dsc, "nsd": mean_nsd, "n": len(dscs)}

    fg_dscs = [v for s in summary.values() for v in [s["dsc"]] if not np.isnan(v)]
    fg_nsds = [v for s in summary.values() for v in [s["nsd"]] if not np.isnan(v)]
    print("-" * 38)
    print(f"{'Mean':<8} {np.mean(fg_dscs):>8.4f} {np.mean(fg_nsds):>8.4f}")

    # hierarchical region Dice
    print(f"\n{'Region':<8} {'Dice':>8}  (n evaluated)")
    print("-" * 28)
    region_summary = {}
    for region_name in REGIONS:
        key = f"region_{region_name}"
        dscs = [v[key]["dsc"] for v in per_case.values() if not np.isnan(v[key]["dsc"])]
        mean_dsc = float(np.mean(dscs)) if dscs else float("nan")
        print(f"{region_name:<8} {mean_dsc:>8.4f}  (n={len(dscs)})")
        region_summary[region_name] = {"dsc": mean_dsc, "n": len(dscs)}
    summary["regions"] = region_summary

    # lesion-wise detection F1 (global TP/FP/FN aggregated across cases)
    print(f"\n{'Label':<8} {'F1':>8} {'Prec':>8} {'Recall':>8}  TP   FP   FN")
    print("-" * 58)
    detection_summary = {}
    for label, name in LABELS.items():
        tp = sum(v[name]["detection"]["tp"] for v in per_case.values())
        fp = sum(v[name]["detection"]["fp"] for v in per_case.values())
        fn = sum(v[name]["detection"]["fn"] for v in per_case.values())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        print(f"{name:<8} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f}  {tp:>4} {fp:>4} {fn:>4}")
        detection_summary[name] = {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}
    summary["detection"] = detection_summary

    # size-stratified breakdown per class
    print()
    for label, name in LABELS.items():
        bins = [[] for _ in SIZE_LABELS]
        for v in per_case.values():
            entry = v[name]
            if entry["skipped"] or np.isnan(entry["dsc"]):
                continue
            vol = entry["gt_vol_mm3"]
            for i in range(len(SIZE_BINS) - 1):
                if SIZE_BINS[i] <= vol < SIZE_BINS[i + 1]:
                    bins[i].append(entry)
                    break

        print(f"{name} by GT size:")
        print(f"  {'Bin':<14} {'DSC':>8} {'NSD':>8}  n")
        for bin_label, entries in zip(SIZE_LABELS, bins):
            if not entries:
                print(f"  {bin_label:<14} {'—':>8} {'—':>8}  0")
                continue
            mean_dsc = float(np.mean([e["dsc"] for e in entries]))
            mean_nsd = float(np.mean([e["nsd"] for e in entries]))
            print(f"  {bin_label:<14} {mean_dsc:>8.4f} {mean_nsd:>8.4f}  {len(entries)}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_case": per_case}, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
