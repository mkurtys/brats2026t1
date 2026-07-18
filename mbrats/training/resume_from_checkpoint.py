"""
Continue training a (possibly different) trainer/plans combo from an existing
checkpoint, preserving epoch count, optimizer state, and LR schedule position.

nnUNetv2_train's --c only resumes from a checkpoint inside that exact trainer's
own output folder (checkpoint_final/latest/best.pth). There's no CLI option to
seed one trainer's run from a checkpoint saved by a *different* trainer class
or plans identifier (e.g. warm-starting nnUNetTrainerBraTS from a checkpoint
produced by nnUNetTrainerCheckpoint250). -pretrained_weights doesn't fit either:
it only loads network weights and starts epoch 0 with a freshly reset LR
schedule, discarding the optimizer/epoch state we want to keep.

nnUNetTrainer.load_checkpoint(path) accepts any path and restores
current_epoch, optimizer state, and grad scaler state alongside the network
weights; nnUNetTrainer.run_training() then resumes its epoch loop from
self.current_epoch. This script builds the target trainer directly and calls
those two methods with an explicit source checkpoint path, then runs the
final internal validation exactly like nnUNetv2_train does after training.

Usage:
    python -m mbrats.training.resume_from_checkpoint \\
        -d 1 -c 3d_fullres -f 0 \\
        -tr nnUNetTrainerBraTS -p nnUNetResEncUNetMPlans \\
        --init_checkpoint nnunet_results/Dataset001_BraTSMETS/nnUNetTrainerCheckpoint250__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_0250.pth \\
        --npz
"""

import argparse

import torch
from torch.backends import cudnn

from nnunetv2.run.run_training import get_trainer_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", required=True, help="Dataset name or ID")
    parser.add_argument("-c", required=True, help="nnU-Net configuration (e.g. 3d_fullres)")
    parser.add_argument("-f", type=int, required=True, help="Fold to train")
    parser.add_argument("-tr", default="nnUNetTrainer", help="Trainer class to train with")
    parser.add_argument("-p", default="nnUNetPlans", help="Plans identifier to train with")
    parser.add_argument("--init_checkpoint", required=True,
                        help="Path to the checkpoint to resume from (network + optimizer + epoch state)")
    parser.add_argument("--npz", action="store_true",
                        help="Save softmax probabilities from final validation as npz files")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cudnn.deterministic = False
    cudnn.benchmark = True

    nnunet_trainer = get_trainer_from_args(args.d, args.c, args.f, args.tr, args.p)
    nnunet_trainer.load_checkpoint(args.init_checkpoint)
    print(f"Resuming '{args.tr}' / '{args.p}' fold {args.f} from epoch {nnunet_trainer.current_epoch} "
         f"(seeded from {args.init_checkpoint})")

    nnunet_trainer.run_training()
    nnunet_trainer.perform_actual_validation(save_probabilities=args.npz)


if __name__ == "__main__":
    main()
