"""
LR schedule helpers for warm-started fine-tuning.

nnU-Net's -pretrained_weights only loads network weights, not optimizer/LR
schedule state, so every warm-started run restarts the LR schedule from
scratch at whatever self.initial_lr is set to. nnU-Net's own default (1e-2)
is tuned for training from scratch; for fine-tuning from a decent checkpoint
it's too aggressive and can disrupt already-learned, fragile representations
(observed directly: the copypaste_250 run's RC pseudo dice stayed at exactly
0.0 for epochs 0-183 of 250 before recovering — consistent with a high-LR
restart knocking out RC's fragile learned features and costing most of the
epoch budget recovering rather than improving).

nnU-Net's own fine-tuning docs (documentation/pretraining_and_finetuning.md)
recommend writing a custom trainer with LR ramp-up for this; this module
provides that scheduler. Confirmed via literature search: an order-of-
magnitude LR reduction (1e-2 -> 1e-3) is commonly used for fine-tuning
ResEnc-family nnU-Net models specifically, and linear warmup over the first
few epochs is reported as measurably beneficial (not just a low flat LR).
"""

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class WarmupPolyLRScheduler(PolyLRScheduler):
    """
    PolyLRScheduler with a linear warmup over the first `warmup_steps` epochs,
    ramping 0 -> initial_lr, before the usual polynomial decay takes over
    (decaying to 0 by max_steps, same total epoch budget as a plain
    PolyLRScheduler would).
    """
    def __init__(self, optimizer, initial_lr: float, max_steps: int, warmup_steps: int = 10,
                 exponent: float = 0.9, current_step: int = None):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, initial_lr, max_steps, exponent, current_step)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            new_lr = self.initial_lr * (current_step + 1) / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            new_lr = self.initial_lr * (1 - progress) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class ResumeWarmupPolyLRScheduler(PolyLRScheduler):
    """
    Same idea as WarmupPolyLRScheduler, but the warmup window is anchored to
    `warmup_start_step` instead of step 0. WarmupPolyLRScheduler's warmup is a
    no-op once `current_step` is already past `warmup_steps` — exactly the
    case when resuming training at some absolute epoch > warmup_steps (e.g.
    epoch 200) rather than starting fresh. Here the ramp 0 -> initial_lr runs
    over [warmup_start_step, warmup_start_step + warmup_steps), then
    polynomial decay takes over for the remainder, still reaching 0 by
    max_steps.
    """
    def __init__(self, optimizer, initial_lr: float, max_steps: int, warmup_start_step: int,
                 warmup_steps: int = 10, exponent: float = 0.9, current_step: int = None):
        self.warmup_start_step = warmup_start_step
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, initial_lr, max_steps, exponent, current_step)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        warmup_end_step = self.warmup_start_step + self.warmup_steps
        steps_into_warmup = current_step - self.warmup_start_step

        if 0 <= steps_into_warmup < self.warmup_steps:
            new_lr = self.initial_lr * (steps_into_warmup + 1) / self.warmup_steps
        else:
            progress = (current_step - warmup_end_step) / max(1, self.max_steps - warmup_end_step)
            new_lr = self.initial_lr * (1 - progress) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
