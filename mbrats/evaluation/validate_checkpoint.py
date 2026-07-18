"""
Run nnU-Net's internal fold validation using an arbitrary checkpoint, then
evaluate the result against ground truth with the official BraTS metrics.

nnUNetv2_train --val hardcodes loading 'checkpoint_final.pth' and refuses to
run if that file doesn't exist yet (see maybe_load_checkpoint in
nnunetv2/run/run_training.py). -pretrained_weights doesn't help either: it is
only consulted when neither --c nor --val is set, i.e. it seeds a *new*
training run rather than pointing validation at a specific checkpoint.

--val itself is just get_trainer_from_args() + nnunet_trainer.load_checkpoint(path)
+ nnunet_trainer.perform_actual_validation(). load_checkpoint takes any path, so
we call it with the real checkpoint directly and skip the checkpoint_final.pth
naming requirement entirely. Writes predictions to <fold_dir>/validation/,
matching the layout the rest of this project's evaluation tooling expects.

Usage:
    python -m mbrats.evaluation.validate_checkpoint \\
        -d 1 -c 3d_fullres -f 0 \\
        -tr nnUNetTrainerCheckpoint250 -p nnUNetResEncUNetMPlans \\
        -chk checkpoint_0250.pth \\
        --gt nnunet_raw/Dataset001_BraTSMETS/labelsTr \\
        --out results/checkpoint250_fold0_cv_eval.json
"""

import argparse
import os
import subprocess
import sys

from nnunetv2.run.run_training import get_trainer_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", required=True, help="Dataset name or ID")
    parser.add_argument("-c", required=True, help="nnU-Net configuration (e.g. 3d_fullres)")
    parser.add_argument("-f", type=int, required=True, help="Fold to validate")
    parser.add_argument("-tr", default="nnUNetTrainer", help="Trainer class used for training")
    parser.add_argument("-p", default="nnUNetPlans", help="Plans identifier")
    parser.add_argument("-chk", default="checkpoint_final.pth", help="Checkpoint file name to validate")
    parser.add_argument("--tile-step-size", type=float, default=None,
                        help="Sliding-window step size (fraction of patch). nnU-Net default is 0.5 "
                             "(50%% overlap); larger is faster with less overlap, e.g. 0.7.")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground-truth labelsTr dir. If set, runs mbrats.evaluation.evaluate afterwards")
    parser.add_argument("--out", type=str, default=None, help="Raw JSON output path (passed to evaluate.py)")
    parser.add_argument("--csv", type=str, default=None, help="Summary CSV output path (passed to evaluate.py)")
    args = parser.parse_args()

    if args.tile_step_size is not None:
        # perform_actual_validation() hardcodes nnUNetPredictor(tile_step_size=0.5).
        # Override that kwarg by wrapping the class in the trainer's module namespace,
        # avoiding an edit to the vendored nnU-Net source.
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_mod
        _orig_predictor = trainer_mod.nnUNetPredictor

        def _predictor_with_step(*a, **kw):
            kw["tile_step_size"] = args.tile_step_size
            return _orig_predictor(*a, **kw)

        trainer_mod.nnUNetPredictor = _predictor_with_step

    nnunet_trainer = get_trainer_from_args(args.d, args.c, args.f, args.tr, args.p)

    fold_dir = nnunet_trainer.output_folder
    checkpoint_path = os.path.join(fold_dir, args.chk)
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    nnunet_trainer.load_checkpoint(checkpoint_path)
    nnunet_trainer.perform_actual_validation(save_probabilities=False)

    validation_dir = os.path.join(fold_dir, "validation")
    print(f"Validation predictions written to {validation_dir}")

    if args.gt:
        cmd = [sys.executable, "-m", "mbrats.evaluation.evaluate", "--pred", validation_dir, "--gt", args.gt]
        if args.out:
            cmd += ["--out", args.out]
        if args.csv:
            cmd += ["--csv", args.csv]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
