"""
Grid-search per-region connected-component size thresholds on fold-0 CV predictions.

Approach: for each case/region, compute predicted + GT connected components, match
them (greedy by Dice, match iff Dice>=0.2 — the BraTS detection criterion), and cache
lightweight records (component sizes, matched?, matched Dice). Then the effect of any
threshold vector is evaluated analytically in memory — no panoptica in the loop — by
reproducing the official `parse_mets_results` lesion-wise DSC (see cc_filter.py header
and memory scoring-and-fp-diagnosis):

  ranking DSC (per region, per subject with >=1 large GT lesion) = mean of
    [ Dice of each matched large GT lesion,  0 per missed large GT lesion,
      0 per surviving false-positive lesion of ANY size ]

Filtering removes predicted components below the threshold: a removed FP disappears
(good); a removed TP turns its GT lesion into a miss (bad). The grid search finds, per
region independently (panoptica scores regions independently), the threshold that
maximises DSC + F1.

Self-check: at threshold 0 the reproduced DSC must match the panoptica eval numbers.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

from mbrats.postprocessing.cc_filter import REGION_LABELS

VOL_THRESHOLD = 27.0   # >= is "large" (segmentation-scored); < is small (detection-only)
OVERLAP_THRESHOLD = 0.2
_STRUCT = np.ones((3, 3, 3), dtype=np.uint8)   # 26-connectivity


# ── per-case record extraction ────────────────────────────────────────────────

def _match_region(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Returns (gt_records, pred_records) for one region of one case.
      gt_records:   list of dicts {size, matched(bool), dice, pred_idx}
      pred_records: list of dicts {size, matched(bool)}
    Matching: greedy by Dice, a pair matches iff Dice >= OVERLAP_THRESHOLD, one-to-one.
    """
    gt_cc, n_gt = ndimage.label(gt_mask, structure=_STRUCT)
    pr_cc, n_pr = ndimage.label(pred_mask, structure=_STRUCT)
    gt_sizes = np.bincount(gt_cc.ravel())      # 0..n_gt
    pr_sizes = np.bincount(pr_cc.ravel())      # 0..n_pr

    # intersections between overlapping (pred_id, gt_id) pairs — vectorised:
    # encode each overlap voxel as pid*(n_gt+1)+gid, then bincount (no Python loop
    # over voxels; the WT overlaps can be >1e5 voxels/case so Counter was the bottleneck)
    both = (gt_cc > 0) & (pr_cc > 0)
    cands = []
    if both.any():
        stride = n_gt + 1
        keys = pr_cc[both].astype(np.int64) * stride + gt_cc[both].astype(np.int64)
        counts = np.bincount(keys)
        nz = np.nonzero(counts)[0]
        for key in nz:
            pid = int(key // stride); gid = int(key % stride)
            ic = int(counts[key])
            dice = 2.0 * ic / (pr_sizes[pid] + gt_sizes[gid])
            if dice >= OVERLAP_THRESHOLD:
                cands.append((dice, pid, gid))
    cands.sort(reverse=True)   # greedy: highest Dice first

    gt_match: dict[int, tuple] = {}   # gid -> (pid, dice)
    pred_used: set[int] = set()
    gt_used: set[int] = set()
    for dice, pid, gid in cands:
        if pid in pred_used or gid in gt_used:
            continue
        pred_used.add(pid); gt_used.add(gid)
        gt_match[gid] = (pid, dice)

    gt_records = []
    for gid in range(1, n_gt + 1):
        if gid in gt_match:
            pid, dice = gt_match[gid]
            gt_records.append({"size": int(gt_sizes[gid]), "matched": True,
                               "dice": float(dice), "pred_idx": int(pid)})
        else:
            gt_records.append({"size": int(gt_sizes[gid]), "matched": False,
                               "dice": 0.0, "pred_idx": -1})
    pred_records = []
    for pid in range(1, n_pr + 1):
        pred_records.append({"size": int(pr_sizes[pid]), "matched": pid in pred_used})
    return gt_records, pred_records


def build_records(pred_dir: Path, gt_dir: Path, limit=None):
    import nibabel as nib
    files = sorted(pred_dir.glob("*.nii.gz"))
    if limit:
        files = files[:limit]
    records = []   # list of {case, region -> (gt_records, pred_records)}
    for i, f in enumerate(files):
        gt_f = gt_dir / f.name
        if not gt_f.exists():
            continue
        pred = np.asarray(nib.load(str(f)).dataobj).astype(np.int16)
        gt = np.asarray(nib.load(str(gt_f)).dataobj).astype(np.int16)
        case = {"case": f.name}
        for region, labels in REGION_LABELS.items():
            pm = np.isin(pred, labels)
            gm = np.isin(gt, labels)
            case[region] = _match_region(pm, gm)
        records.append(case)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(files)}")
    return records


# ── analytic scoring under a threshold ────────────────────────────────────────

def score_region(records, region: str, k: int):
    """Reproduce official per-region DSC + an all-instance F1, under threshold k."""
    dsc_subjects = []
    tp = fp = fn = 0
    for case in records:
        gt_recs, pred_recs = case[region]
        # which pred components survive the filter
        survive = [p["size"] >= k for p in pred_recs]

        large_dsc = []
        large_present = False
        n_ref = len(gt_recs)
        s_tp = s_fp = s_fn = 0   # per-subject detection counts (all sizes)

        for g in gt_recs:
            large = g["size"] >= VOL_THRESHOLD
            large_present |= large
            matched = g["matched"] and survive[g["pred_idx"] - 1] if g["matched"] else False
            if matched:
                s_tp += 1
                if large:
                    large_dsc.append(g["dice"])
            else:  # missed (never matched, or its pred was filtered away)
                s_fn += 1
                if large:
                    large_dsc.append(0.0)
        # surviving false positives (unmatched surviving pred components)
        n_fp = sum(1 for p, sv in zip(pred_recs, survive) if sv and not p["matched"])
        s_fp += n_fp
        large_dsc.extend([0.0] * n_fp)   # every FP zeros the DSC (any size)

        tp += s_tp; fp += s_fp; fn += s_fn
        if n_ref > 0 and large_present:
            dsc_subjects.append(np.mean(large_dsc) if large_dsc else 0.0)

    dsc = float(np.mean(dsc_subjects)) if dsc_subjects else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {"dsc": dsc, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def grid_search_region(records, region, grid, objective):
    best = None
    curve = []
    for k in grid:
        s = score_region(records, region, k)
        val = objective(s)
        curve.append((k, s, val))
        if best is None or val > best[2]:
            best = (k, s, val)
    return best, curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=None, help="pickle of records to reuse")
    p.add_argument("--out", type=Path, default=None, help="write best thresholds JSON")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--objective", choices=["dsc", "f1", "dsc+f1"], default="dsc+f1")
    args = p.parse_args()

    if args.cache and args.cache.exists():
        print(f"Loading cached records from {args.cache}")
        records = pickle.load(open(args.cache, "rb"))
    else:
        print("Building records (CC + matching per case)...")
        records = build_records(args.pred, args.gt, limit=args.limit)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(records, open(args.cache, "wb"))
            print(f"Cached {len(records)} cases to {args.cache}")

    obj_fn = {"dsc": lambda s: s["dsc"],
              "f1": lambda s: s["f1"],
              "dsc+f1": lambda s: s["dsc"] + s["f1"]}[args.objective]

    grid = [1, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]

    print(f"\n{'region':6} {'k*':>4} | {'DSC0':>6} {'DSCk':>6} {'dDSC':>6} | "
          f"{'F1_0':>6} {'F1_k':>6} | {'FP0':>5} {'FPk':>5} {'TPk':>5}")
    best_thresholds = {}
    for region in REGION_LABELS:
        base = score_region(records, region, 1)   # k=1 == no-op (self-check baseline)
        best, curve = grid_search_region(records, region, grid, obj_fn)
        k, s, _ = best
        best_thresholds[region] = k
        print(f"{region:6} {k:>4} | {base['dsc']:.4f} {s['dsc']:.4f} "
              f"{s['dsc']-base['dsc']:+.4f} | {base['f1']:.4f} {s['f1']:.4f} | "
              f"{base['fp']:>5} {s['fp']:>5} {s['tp']:>5}")

    print(f"\nBest thresholds ({args.objective}): {best_thresholds}")
    print("Self-check: DSC0 column above must match the panoptica no-filter numbers "
          "(blobloss ET/TC/WT/RC ~ 0.618/0.639/0.583/0.332).")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(best_thresholds, open(args.out, "w"), indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
