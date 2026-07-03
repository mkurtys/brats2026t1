"""
Evaluate segmentation predictions against ground truth using the official
BraTS evaluation package (panoptica-based).

Groups evaluated (per config_mets.yaml):
  et  = label 3
  rc  = label 4
  tc  = labels 1+3  (tumor core)
  wt  = labels 1+2+3  (whole tumor)

Usage:
    python src/evaluate.py --pred predictions/fold0_val --gt nnunet_raw/Dataset001_BraTSMETS/labelsTr
    python src/evaluate.py --pred predictions/fold0_val --gt ... --out results/eval.json --csv results/eval.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
from panoptica import Panoptica_Evaluator

from brats_evaluation import evaluate_single_exam, parse_mets_results, config_path


# Note: For BraTS 2026 METs challenge, lesions which have volume below 27mm^3 will be considered for the detection evaluation
#  and the overlapping threshold for detection criteria is DSC=0.2
# source: https://github.com/BraTS/BraTS_evaluation/blob/main/example/brats_mets.ipynb
METS_CONFIG = config_path("mets")
VOL_THRESHOLD_MM3 = 27.0
OVERLAP_THRESHOLD = 0.2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Prediction directory (.nii.gz)")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth directory (.nii.gz)")
    parser.add_argument("--out", type=Path, default=None, help="Raw JSON output path")
    parser.add_argument("--csv", type=Path, default=None, help="Summary CSV output path")
    parser.add_argument("--n", type=int, default=None, help="Limit number of cases (for quick checks)")
    args = parser.parse_args()

    evaluator = Panoptica_Evaluator.load_from_config(METS_CONFIG)

    pred_files = sorted(args.pred.glob("*.nii.gz"))
    if not pred_files:
        raise SystemExit(f"No .nii.gz files found in {args.pred}")
    if args.n:
        pred_files = pred_files[: args.n]

    metrics = []
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = args.gt / f"{case_id}.nii.gz"
        if not gt_path.exists():
            print(f"  SKIP {case_id}: no GT")
            continue
        result = evaluate_single_exam(str(pred_path), str(gt_path), case_id, evaluator)
        metrics.append(result)
        if "error" not in result:
            groups = [k for k in result if k != "subject_name"]
            parts = []
            for g in groups:
                dsc = result[g].get("global_dsc", float("nan"))
                parts.append(f"{g}={dsc:.3f}" if not np.isnan(dsc) else f"{g}=nan")
            print(f"  {case_id}: {', '.join(parts)}")
        else:
            print(f"  {case_id}: ERROR — {result['error']}")

    payload = {"metrics": metrics}

    # resolve output paths
    json_out = args.out or Path("panoptica_eval.json")
    csv_out = args.csv or json_out.with_suffix(".csv")

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRaw results saved to {json_out}")

    parse_mets_results(str(json_out), VOL_THRESHOLD_MM3, OVERLAP_THRESHOLD, str(csv_out))
    print(f"Summary CSV saved to {csv_out}")


if __name__ == "__main__":
    main()
