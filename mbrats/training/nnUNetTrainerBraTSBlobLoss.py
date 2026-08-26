"""
Custom nnUNet trainer for BraTS 2026 MET Task 1.

Changes vs default nnUNetTrainer:
  1. Focal loss  — replaces CE in the Dice+CE compound loss (gamma=2)
  2. Instance-uniform patch sampling — when centering on foreground, picks a
     connected component uniformly (not a voxel), so small lesions get equal
     sampling probability regardless of size.

Note: nnUNet's default class-uniform sampling (picking a class uniformly before
picking a voxel) is already implemented in the base DataLoader and is preserved here.
"""

import functools
import warnings
from typing import Union, Tuple, List

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from torch import nn

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from mbrats.training.lr_schedules import WarmupPolyLRScheduler


# ──────────────────────────────────────────────────────────────────────────────
# Focal loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.  FL = -(1 - p_t)^gamma * log(p_t)
    Input: logits (B, C, spatial...), target: long (B, spatial...)
    """
    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        ce = F.cross_entropy(input, target, weight=self.weight,
                             ignore_index=self.ignore_index, reduction='none')

        # p_t: probability assigned to the correct class
        with torch.no_grad():
            probs = F.softmax(input, dim=1)
            target_clamped = target.clamp(min=0)
            pt = probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
            if self.ignore_index >= 0:
                pt = torch.where(target == self.ignore_index, torch.ones_like(pt), pt)
            focal_weight = (1.0 - pt) ** self.gamma

        return (focal_weight * ce).mean()


class DC_and_Focal_loss(DC_and_CE_loss):
    """Dice + Focal loss (drops-in for DC_and_CE_loss)."""
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        # initialise parent with dummy CE kwargs; we'll replace self.ce below
        super().__init__(soft_dice_kwargs, {}, weight_ce=weight_focal,
                         weight_dice=weight_dice, ignore_label=ignore_label,
                         dice_class=dice_class)
        if ignore_label is not None:
            focal_kwargs['ignore_index'] = ignore_label
        self.ce = FocalLoss(**focal_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Blob region loss (TC | RC)
# ──────────────────────────────────────────────────────────────────────────────

def tversky(p, g, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
    # p, g: (B, *spatial) probabilities and binary GT for ONE region
    tp = (p * g).sum(dim=tuple(range(1, p.ndim)))                 # (B,)
    fp = (p * (1 - g)).sum(dim=tuple(range(1, p.ndim)))           # (B,)
    fn = ((1 - p) * g).sum(dim=tuple(range(1, p.ndim)))           # (B,)
    # ti: Tversky index, a generalized Dice (Dice = this formula with alpha=beta=0.5).
    # alpha=0.3, beta=0.7 weights FN (missed lesion voxels) > 2x harder than FP in the
    # denominator, so a high ti requires recall more than it requires precision.
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)         # (B,)
    # loss = (1-ti)^gamma is linear in ti at gamma=1 (dL/dti = -1, constant slope regardless
    # of how good ti already is). gamma=1.33 curves it: dL/dti = -gamma*(1-ti)^(gamma-1),
    # which -> 0 as ti->1 (vanishing gradient on already-good instances) and stays near its
    # max as ti->0 (full gradient on still-bad instances) — a focal-style hardness focus,
    # per instance rather than per voxel. 1.33 = 4/3, the default gamma from the Focal
    # Tversky Loss paper (Abraham & Khan 2019).
    return ((1 - ti) ** gamma).mean()                             # scalar


_BLOB_CHUNK = 8   # components per chunk — bounds memory to ~8 × patch × 4 intermediate tensors
_BLOB_MAX   = 64  # skip cases with more CCs (elastic deformation fragments → noise anyway)

def blob_term(p, comp_labels, n_comp, alpha=0.3, beta=0.7):
    # p, comp_labels: (*spatial) — single batch element (called per-b in _tc_rc_blob_loss)
    if n_comp == 0 or n_comp > _BLOB_MAX:
        return p.new_tensor(0.0)                    # scalar
    all_fg = (comp_labels > 0).float()               # (*spatial)
    cl_exp = comp_labels.unsqueeze(0)                # (1, *spatial)
    ndim   = p.ndim                                  # spatial rank, e.g. 3
    total  = p.new_tensor(0.0)                        # scalar accumulator
    for start in range(1, n_comp + 1, _BLOB_CHUNK):
        ids     = torch.arange(start, min(start + _BLOB_CHUNK, n_comp + 1), device=p.device)  # (chunk,)
        chunk   = len(ids)
        ids_    = ids.view(chunk, *([1] * ndim))     # (chunk, 1, 1, 1)
        # gi: stack of one-hot GT masks, one per instance in this chunk. gi[k] == 1 at every
        # voxel belonging to instance ids[k], 0 elsewhere (including other instances' voxels).
        gi      = (cl_exp == ids_).float()          # (chunk, *spatial)
        # pi: predicted prob map p, broadcast per instance, with prediction mass on OTHER
        # instances' voxels zeroed out ((all_fg - gi) is 1 only on other instances' territory).
        # This stops a prediction spilling onto a neighboring lesion from counting as this
        # instance's false positive — each instance is scored only against its own voxels
        # plus true background.
        pi      = p.unsqueeze(0) * (1 - (all_fg - gi).clamp(0, 1))  # (chunk, *spatial)
        tp      = (pi * gi).sum(dim=tuple(range(1, ndim + 1)))       # (chunk,)
        fp      = (pi * (1 - gi)).sum(dim=tuple(range(1, ndim + 1)))  # (chunk,)
        fn      = ((1 - pi) * gi).sum(dim=tuple(range(1, ndim + 1)))  # (chunk,)
        ti      = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)  # (chunk,) per-instance Tversky index, see tversky() above
        total   = total + ((1 - ti) ** 1.33).sum()   # scalar; same focal-style exponent as tversky()'s gamma, applied per instance here
    return total / n_comp                            # scalar, mean over instances (dividing by n_comp, not voxel count)
    return total / n_comp                            # scalar, mean over instances


_SMALL_INSTANCE_VOXELS = 27  # proxy for BraTS's "<27mm3 = small lesion" cutoff, assumes ~1mm3 voxels
_SATURATION_MARGIN = 0.35    # plain-Dice safety margin above the real detection bar (DSC>=0.2)

def saturating_blob_term(p, comp_labels, n_comp, alpha=0.3, beta=0.7,
                          small_voxel_thresh=_SMALL_INSTANCE_VOXELS, margin=_SATURATION_MARGIN):
    """
    Same per-instance isolation/chunking as blob_term, but stops pushing gradient on small
    instances once they clear a safety margin above the real BraTS lesion-wise detection bar.

    Large instances (>= small_voxel_thresh voxels) are untouched — they're scored on real
    DSC for the segmentation leaderboard, so they still need to be pushed toward ti=1.

    The saturation gate uses plain Dice (alpha=beta=0.5), not this function's own
    recall-biased `ti` (alpha=0.3, beta=0.7) — the real eval match criterion is DSC>=0.2,
    and `ti` is a different number than DSC for the same TP/FP/FN (harsher on FN-heavy
    errors, since beta>alpha), so gating on `ti` directly would drift from what "detected"
    actually means. Below the margin, the loss shape is identical to blob_term's.
    """
    if n_comp == 0 or n_comp > _BLOB_MAX:
        return p.new_tensor(0.0)                    # scalar
    all_fg = (comp_labels > 0).float()               # (*spatial)
    cl_exp = comp_labels.unsqueeze(0)                # (1, *spatial)
    ndim   = p.ndim                                  # spatial rank, e.g. 3
    total  = p.new_tensor(0.0)                        # scalar accumulator
    for start in range(1, n_comp + 1, _BLOB_CHUNK):
        ids     = torch.arange(start, min(start + _BLOB_CHUNK, n_comp + 1), device=p.device)  # (chunk,)
        chunk   = len(ids)
        ids_    = ids.view(chunk, *([1] * ndim))     # (chunk, 1, 1, 1)
        gi      = (cl_exp == ids_).float()          # (chunk, *spatial) — this instance's GT mask
        pi      = p.unsqueeze(0) * (1 - (all_fg - gi).clamp(0, 1))  # (chunk, *spatial) — pred, isolated per instance
        tp      = (pi * gi).sum(dim=tuple(range(1, ndim + 1)))       # (chunk,)
        fp      = (pi * (1 - gi)).sum(dim=tuple(range(1, ndim + 1)))  # (chunk,)
        fn      = ((1 - pi) * gi).sum(dim=tuple(range(1, ndim + 1)))  # (chunk,)
        # instance voxel count == tp+fn == gi.sum(), computed directly from gi (independent of pi)
        instance_size = gi.sum(dim=tuple(range(1, ndim + 1)))         # (chunk,)

        ti   = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)    # (chunk,) — same recall-biased index as blob_term
        dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)           # (chunk,) — plain Dice, matches the real match criterion

        saturate = (instance_size < small_voxel_thresh) & (dice > margin)  # (chunk,) bool
        # where saturated: use ti.detach() (same value, zero gradient). where not: normal ti,
        # full gradient flows as in blob_term. torch.where selects per-element grad-vs-no-grad.
        ti_for_loss = torch.where(saturate, ti.detach(), ti)          # (chunk,)

        total = total + ((1 - ti_for_loss) ** 1.33).sum()   # scalar, accumulated across chunks
    return total / n_comp                            # scalar, mean over instances


def saturating_region_loss(p, g, comp_labels, n_comp, lam=1.5, alpha=0.3, beta=0.7,
                            small_voxel_thresh=_SMALL_INSTANCE_VOXELS, margin=_SATURATION_MARGIN):
    # p, g, comp_labels: (*spatial) — single batch element, whole TC|RC region (not per-instance)
    p_c = p.clamp(1e-6, 1 - 1e-6)                    # (*spatial)
    bce = -(g * p_c.log() + (1 - g) * (1 - p_c).log()).mean()  # scalar
    blob = saturating_blob_term(p, comp_labels, n_comp, alpha=alpha, beta=beta,
                                 small_voxel_thresh=small_voxel_thresh, margin=margin)
    return tversky(p, g, alpha=alpha, beta=beta) + bce + lam * blob  # scalar


def region_loss(p, g, comp_labels, n_comp, lam=1.5):
    # p, g, comp_labels: (*spatial) — single batch element, whole TC|RC region (not per-instance)
    p_c = p.clamp(1e-6, 1 - 1e-6)                    # (*spatial)
    bce = -(g * p_c.log() + (1 - g) * (1 - p_c).log()).mean()  # scalar
    # NB: tversky() expects (B, *spatial) and reduces dims 1..ndim-1, keeping dim 0 as "batch".
    # p, g here have no batch dim, so dim 0 (depth) is what gets kept -> this computes a
    # per-depth-slice Tversky index, then .mean() averages over depth (not one whole-volume TI).
    return tversky(p, g) + bce + lam * blob_term(p, comp_labels, n_comp)  # scalar


class BlobRegionWrapper(nn.Module):
    """
    Wraps a deep-supervision loss and adds a TC|RC blob region loss at full resolution.
    TC|RC = NETC (ch 1) + ET (ch 3) + RC (ch 4).
    CC computation runs on CPU per batch element (scipy); acceptable at batch_size=2.

    Two modes, selected by whether rc_region_loss_fn is given:
      - combined (default, rc_region_loss_fn=None): single CC pass over NETC|ET|RC
        together via region_loss_fn — the original behaviour. Since resection cavities
        routinely touch residual tumor, this frequently MERGES an RC instance into the
        same connected component as adjacent ET/NETC, so RC voxels end up scored under
        whatever alpha/beta the merged blob gets, not an RC-specific one.
      - split (rc_region_loss_fn given): TWO independent CC passes computed from scratch
        each step — TC (NETC|ET) via region_loss_fn, RC (RC only) via rc_region_loss_fn —
        so RC gets its own alpha/beta/saturation gate, decoupled from TC's. Costs one
        extra scipy.ndimage.label call per batch element per step (not offloaded to the
        dataloader workers, unlike the combined path's precomputed-comp_labels fast path).
    """

    def __init__(self, ds_loss, blob_weight: float = 0.5, ignore_label=None,
                 region_loss_fn=region_loss, rc_region_loss_fn=None,
                 tc_weight: float = 1.0, rc_weight: float = 1.0):
        super().__init__()
        self.ds_loss = ds_loss
        self.blob_weight = blob_weight
        self.ignore_label = ignore_label if ignore_label is not None else -100
        self.region_loss_fn = region_loss_fn
        self.rc_region_loss_fn = rc_region_loss_fn
        self.tc_weight = tc_weight
        self.rc_weight = rc_weight

    def forward(self, net_output, target):
        # net_output: list of (B, C, *spatial) logits, one per DS scale, or a single (B, C, *spatial)
        # target: matching list/single of (B, 1 or 2, *spatial) — 2 channels when comp_labels is attached
        # strip comp_labels channel before passing to DS loss (expects only seg labels)
        if isinstance(target, (list, tuple)):
            target_seg = [t[:, :1] if t.shape[1] > 1 else t for t in target]
        else:
            target_seg = target[:, :1] if target.shape[1] > 1 else target

        main_loss = self.ds_loss(net_output, target_seg)         # scalar
        full_logits = net_output[0] if isinstance(net_output, (list, tuple)) else net_output  # (B, C, *spatial), highest-res DS head only
        full_target = target[0] if isinstance(target, (list, tuple)) else target              # (B, 1 or 2, *spatial)
        # disable autocast: binary_cross_entropy is unsafe in AMP context
        with torch.amp.autocast(full_logits.device.type, enabled=False):
            blob_loss = self._tc_rc_blob_loss(full_logits.float(), full_target)  # scalar
        return main_loss + self.blob_weight * blob_loss

    def _tc_rc_blob_loss(self, logits, seg):
        if self.rc_region_loss_fn is None:
            return self._combined_blob_loss(logits, seg)
        return self._split_blob_loss(logits, seg)

    def _combined_blob_loss(self, logits, seg):
        # logits: (B, C, *spatial); seg: (B, 1 or 2, *spatial) — channel 1, if present, is comp_labels
        probs = F.softmax(logits, dim=1)          # (B, C, *spatial)
        # sums per-class probs (NETC + ET + RC channels) into one region prob map, channel dim gone
        tc_rc_prob = (probs[:, 1] + probs[:, 3] + probs[:, 4]).clamp(0.0, 1.0)  # (B, *spatial)

        seg_3d = seg[:, 0].long()                 # (B, *spatial) — the real segmentation labels
        valid = seg_3d != self.ignore_label        # (B, *spatial)
        tc_rc_gt = ((seg_3d == 1) | (seg_3d == 3) | (seg_3d == 4)).float()  # (B, *spatial)
        tc_rc_gt = tc_rc_gt * valid.float()        # (B, *spatial)

        # comp_labels precomputed in data loader and passed as seg channel 1
        has_precomputed = seg.shape[1] >= 2
        loss = logits.new_tensor(0.0)              # scalar accumulator
        for b in range(logits.shape[0]):
            if has_precomputed:
                comp_labels = seg[b, 1].long().to(logits.device)   # (*spatial)
                n_comp = int(comp_labels.max().item())
            else:
                import scipy.ndimage
                gt_np = tc_rc_gt[b].cpu().numpy()                  # (*spatial), numpy
                comp_np, n_comp = scipy.ndimage.label(gt_np)        # (*spatial), numpy
                comp_labels = torch.from_numpy(comp_np).to(logits.device)  # (*spatial)
            # tc_rc_prob[b], tc_rc_gt[b], comp_labels: (*spatial) — single batch element, fed to region_loss_fn
            loss = loss + self.region_loss_fn(tc_rc_prob[b], tc_rc_gt[b], comp_labels, n_comp)
        return loss / logits.shape[0]              # scalar, mean over batch

    def _split_blob_loss(self, logits, seg):
        # TC (NETC|ET) and RC scored independently: own CCs, own alpha/beta, so an RC
        # instance touching adjacent tumor is no longer merged into TC's blob (see class
        # docstring). comp_labels precomputed for the combined mask (seg channel 1) is NOT
        # reusable here — a merged id in it may mix TC and RC voxels — so both regions'
        # CCs are computed fresh from the seg labels every step.
        import scipy.ndimage
        probs = F.softmax(logits, dim=1)          # (B, C, *spatial)
        seg_3d = seg[:, 0].long()                 # (B, *spatial)
        valid = (seg_3d != self.ignore_label).float()  # (B, *spatial)

        tc_prob = (probs[:, 1] + probs[:, 3]).clamp(0.0, 1.0)          # (B, *spatial) — NETC + ET
        tc_gt = ((seg_3d == 1) | (seg_3d == 3)).float() * valid        # (B, *spatial)
        rc_prob = probs[:, 4].clamp(0.0, 1.0)                          # (B, *spatial) — RC only
        rc_gt = (seg_3d == 4).float() * valid                          # (B, *spatial)

        loss = logits.new_tensor(0.0)              # scalar accumulator
        for b in range(logits.shape[0]):
            tc_np = tc_gt[b].detach().cpu().numpy()
            tc_comp_np, tc_n = scipy.ndimage.label(tc_np)
            tc_comp = torch.from_numpy(tc_comp_np).to(logits.device)

            rc_np = rc_gt[b].detach().cpu().numpy()
            rc_comp_np, rc_n = scipy.ndimage.label(rc_np)
            rc_comp = torch.from_numpy(rc_comp_np).to(logits.device)

            tc_loss = self.region_loss_fn(tc_prob[b], tc_gt[b], tc_comp, tc_n)
            rc_loss = self.rc_region_loss_fn(rc_prob[b], rc_gt[b], rc_comp, rc_n)
            loss = loss + self.tc_weight * tc_loss + self.rc_weight * rc_loss
        return loss / logits.shape[0]              # scalar, mean over batch


# ──────────────────────────────────────────────────────────────────────────────
# Instance-uniform data loader
# ──────────────────────────────────────────────────────────────────────────────

def _build_comp_labels(instances: list, bbox_lbs: list, patch_size) -> np.ndarray:
    """
    Build a comp_labels int16 array (shape = patch_size) from precomputed
    tc_rc_instances voxel lists. Voxels outside the patch bbox are discarded.
    Instance IDs are 1-based; 0 = background.
    """
    comp = np.zeros(patch_size, dtype=np.int16)
    lbs = np.array(bbox_lbs, dtype=np.int64)
    sz  = np.array(patch_size, dtype=np.int64)
    for idx, inst in enumerate(instances, 1):
        coords = inst[:, 1:].astype(np.int64)       # (N, 3) xyz in full-volume space
        shifted = coords - lbs                       # patch-relative coordinates
        valid = np.all((shifted >= 0) & (shifted < sz), axis=1)
        shifted = shifted[valid]
        if len(shifted):
            comp[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = idx
    return comp

class nnUNetDataLoaderInstanceUniform(nnUNetDataLoader):
    """
    Foreground patch sampling strategy: class-uniform per case, then instance-uniform
    within class.

    When force_fg is True and tc_rc_instances is present in the case properties:
      1. Pick a foreground class uniformly at random (each class gets equal probability
         regardless of how many instances or voxels it has).
      2. Pick a connected component (instance) uniformly from that class.
      3. Pick a random voxel within that instance as the patch centre.

    This prevents large classes (SNFH with thousands of voxels) from drowning out small
    ones (NETC/RC with tens of voxels). A 27-voxel lesion gets the same within-class
    sampling probability as a 27 000-voxel one.

    Requires precomputed tc_rc_instances in each case's .pkl file.
    Run scripts/precompute_tc_rc_instances.py before training.
    Falls back to voxel-uniform (parent behaviour) if tc_rc_instances is absent.
    """

    # probs ∝ size**exponent. 1.0 = size-proportional (large lesions dominate),
    # 0.5 = sqrt compromise (default), 0.0 = fully instance-uniform (every instance,
    # regardless of size, gets equal selection probability — maximum small-lesion weight).
    _instance_size_exponent = 0.5

    def _instance_uniform_bbox(self, shape: tuple, properties: dict):
        """
        Returns (bbox_lbs, bbox_ubs) centered on a size**_instance_size_exponent-weighted
        TC instance.

        tc_rc_instances is a flat list of CC arrays (ET | NETC combined mask).
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(shape)
        for d in range(dim):
            if need_to_pad[d] + shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
               for i in range(dim)]

        instances = properties['tc_rc_instances']  # flat list of CC arrays

        if not instances:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
            return bbox_lbs, bbox_ubs

        # pick instance with size**exponent weighting, then a random voxel within it
        sizes = np.array([len(inst) for inst in instances], dtype=np.float64)
        probs = sizes ** self._instance_size_exponent; probs /= probs.sum()
        chosen_instance = instances[np.random.choice(len(instances), p=probs)]  # (N, 4)
        sv = chosen_instance[np.random.randint(len(chosen_instance))]            # [ch, x, y, z]
        center = [int(sv[i + 1]) for i in range(dim)]  # skip channel col

        bbox_lbs = [max(lbs[i], min(ubs[i], center[i] - self.patch_size[i] // 2))
                    for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]

                    if force_fg and 'tc_rc_instances' in properties:
                        bbox_lbs, bbox_ubs = self._instance_uniform_bbox(shape, properties)
                    else:
                        # fallback: parent voxel-uniform behaviour
                        bbox_lbs, bbox_ubs = self.get_bbox(
                            shape, force_fg, properties['class_locations'])

                    bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(
                        crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)

                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(
                            **{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    # attach comp_labels built from augmented seg (runs in worker → CPU)
                    if not self.patch_size_was_2d:
                        seg_0 = seg_sample[0].numpy() if isinstance(seg_sample, torch.Tensor) else seg_sample[0][0].numpy()
                        tc_rc_mask = ((seg_0 == 1) | (seg_0 == 3) | (seg_0 == 4)).astype(np.uint8)
                        comp_np, _ = ndimage.label(tc_rc_mask)
                        comp_t = torch.from_numpy(comp_np[None].astype(np.int16))
                        if isinstance(seg_sample, torch.Tensor):
                            seg_sample = torch.cat((seg_sample, comp_t), dim=0)
                        else:
                            seg_sample[0] = torch.cat((seg_sample[0], comp_t), dim=0)

                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty(
                                (self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


class nnUNetDataLoaderInstanceUniformSmallBoost(nnUNetDataLoaderInstanceUniform):
    """
    Same as nnUNetDataLoaderInstanceUniform, but exponent=0.25: TC|RC connected components
    are weighted by size**0.25, between the base class's sqrt(size) [exponent=0.5, still
    tilts toward larger instances] and fully instance-uniform [exponent=0.0, every instance
    equal regardless of size]. exponent=0.0 was tried first and over-corrected — it favored
    small lesions so heavily that large-lesion (DSC) sampling density suffered too much;
    0.25 keeps a real boost toward small-lesion signal (recall/F1, incl. the 2026 detection
    leaderboard) without abandoning large-instance exposure entirely.
    """
    _instance_size_exponent = 0.25


# ──────────────────────────────────────────────────────────────────────────────
# Custom trainer
# ──────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerBraTSBlobLoss(nnUNetTrainer):
    """
    BraTS 2026 MET custom trainer.
    - Dice + Focal loss (gamma=2)
    - Instance-uniform foreground patch sampling
    """

    _region_loss_fn = staticmethod(region_loss)  # override in subclasses to swap the blob-loss shape
    _focal_gamma = 2.0                           # override in subclasses to soften the FP-inflating focal tilt
    _dataloader_cls = nnUNetDataLoaderInstanceUniform  # override to swap instance-sampling weighting
    _rc_region_loss_fn = None                    # override to decouple RC from TC (split blob loss)
    _tc_region_weight = 1.0                      # only used when _rc_region_loss_fn is set
    _rc_region_weight = 1.0                      # only used when _rc_region_loss_fn is set
    _focal_class_weights = None                  # override to per-class-weight the focal/CE term (len = num classes incl. background)

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerBraTS expects label-based (not region-based) training"

        focal_kwargs = {'gamma': self._focal_gamma}
        if self._focal_class_weights is not None:
            focal_kwargs['weight'] = torch.tensor(self._focal_class_weights, device=self.device, dtype=torch.float32)

        loss = DC_and_Focal_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                              'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs=focal_kwargs,
            weight_focal=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = self._get_ds_loss_weights(deep_supervision_scales)
            loss = DeepSupervisionWrapper(loss, weights)

        return BlobRegionWrapper(loss, blob_weight=0.5,
                                 ignore_label=self.label_manager.ignore_label,
                                 region_loss_fn=self._region_loss_fn,
                                 rc_region_loss_fn=self._rc_region_loss_fn,
                                 tc_weight=self._tc_region_weight,
                                 rc_weight=self._rc_region_weight)

    def _get_ds_loss_weights(self, deep_supervision_scales) -> np.ndarray:
        # halve the weight per scale (finest=1, then 1/2, 1/4, ...), coarsest scale zeroed
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        return weights / weights.sum()

    def _class_balanced_sampling_weights(self, dataset_tr) -> np.ndarray:
        """
        Per-case sampling probabilities based on inverse class frequency.

        For each case, weight = max over its present foreground classes of
        (n_cases / n_cases_with_class) — i.e. the rarity of its rarest class.
        Cases containing rare classes (RC, NETC) get proportionally higher
        selection probability than common-class-only cases, regardless of how
        many other (common) classes also happen to be present.
        """
        from batchgenerators.utilities.file_and_folder_operations import load_pickle
        import os

        # pass 1: collect which classes each case has
        case_classes = []
        for key in dataset_tr.identifiers:
            pkl_path = os.path.join(self.preprocessed_dataset_folder, key + '.pkl')
            props = load_pickle(pkl_path)
            locs = props.get('class_locations', {})
            present = frozenset(l for l, voxels in locs.items() if len(voxels) > 0)
            case_classes.append(present)

        n_cases = len(case_classes)
        class_counts = {}
        for present in case_classes:
            for l in present:
                class_counts[l] = class_counts.get(l, 0) + 1

        # inverse-frequency weight per class
        inv_freq = {l: n_cases / cnt for l, cnt in class_counts.items()}

        self.print_to_log_file('Case sampling — class inverse frequencies:')
        for l, w in sorted(inv_freq.items()):
            self.print_to_log_file(f'  label {l}: {class_counts[l]} cases  weight {w:.2f}x')

        # pass 2: per-case weight = max inv_freq over its classes (rarest class wins)
        weights = np.array(
            [max(inv_freq[l] for l in present) if present else 1.0 for present in case_classes],
            dtype=np.float64,
        )
        return weights / weights.sum()

    def get_dataloaders(self):
        # identical to parent except dl_tr uses nnUNetDataLoaderInstanceUniform
        # with class-frequency-balanced case sampling probabilities
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from batchgeneratorsv2.helpers.scalar_type import RandomScalar
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (rotation_for_DA, do_dummy_2d_data_aug,
         initial_patch_size, mirror_axes) = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        sampling_weights = self._class_balanced_sampling_weights(dataset_tr)

        dl_tr = self._dataloader_cls(
            dataset_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=sampling_weights, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None, pin_memory=self.device.type == 'cuda', wait_time=0.002)

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerBraTSBlobLoss_2epochs(nnUNetTrainerBraTSBlobLoss):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2


class nnUNetTrainerBraTSBlobLoss_250epochs(nnUNetTrainerBraTSBlobLoss):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturating(nnUNetTrainerBraTSBlobLoss):
    """
    Same as nnUNetTrainerBraTSBlobLoss, but small (<27 voxel) TC|RC instances stop
    accumulating blob-loss gradient once their plain Dice clears a safety margin (0.35)
    above the real BraTS lesion-wise detection bar (DSC>=0.2). Large instances are
    unaffected. See saturating_blob_term / saturating_region_loss.
    """
    _region_loss_fn = staticmethod(saturating_region_loss)


class nnUNetTrainerBraTSBlobLossSaturating_2epochs(nnUNetTrainerBraTSBlobLossSaturating):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2


class nnUNetTrainerBraTSBlobLossSaturating_250epochs(nnUNetTrainerBraTSBlobLossSaturating):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingWarmStart(nnUNetTrainerBraTSBlobLossSaturating):
    """
    For warm-starting from a DIFFERENT trainer/config's checkpoint via -pretrained_weights
    (e.g. coarse-res nnUNetTrainerBraTS_750epochs -> this highres+blob-loss trainer).
    -pretrained_weights only loads network weights; epoch/optimizer/LR schedule all reset
    to 0, so nnU-Net's default initial_lr=1e-2 would restart at full strength on already-
    converged weights. That previously stalled RC's fragile learned features for ~180
    epochs in an earlier warm start (see memory/current-baseline.md). Uses a 10x lower
    initial_lr with a short linear warmup (WarmupPolyLRScheduler) instead — this is the
    first use case where the warmup phase actually activates (a genuine epoch-0 start),
    unlike a load_checkpoint-based resume mid-schedule where it's a no-op past epoch ~10.
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = WarmupPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, warmup_steps=10)
        return optimizer, lr_scheduler


class nnUNetTrainerBraTSBlobLossSaturatingWarmStart_250epochs(nnUNetTrainerBraTSBlobLossSaturatingWarmStart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingFewDS(nnUNetTrainerBraTSBlobLossSaturating):
    """
    Same as nnUNetTrainerBraTSBlobLossSaturating, but zeros deep-supervision loss weight
    for every scale coarser than 1/2 (not just the coarsest, which the base class already
    zeros). At 3d_fullres (patch 96x160x160, 5 DS scales: 1, 1/2, 1/4, 1/8, 1/16), a small
    (~27 voxel, i.e. ~3-voxel-cube) lesion is already sub-voxel by the 1/4 scale — the DS
    ground truth there has effectively erased it, so that DS head is trained to predict
    background exactly where a small lesion exists in the full-res label. The base class's
    weights ([0.53, 0.27, 0.13, 0.07, 0] after zeroing only the last) still gave that
    1/4-scale head 13% of the DS loss and the 1/8-scale head 7% — both already past the
    point where small lesions survive downsampling. This zeros scales 2 and up, keeping
    only full-res (weight 2/3) and 1/2 (weight 1/3).
    """
    def _get_ds_loss_weights(self, deep_supervision_scales) -> np.ndarray:
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[2:] = 0
        return weights / weights.sum()


class nnUNetTrainerBraTSBlobLossSaturatingFewDS_250epochs(nnUNetTrainerBraTSBlobLossSaturatingFewDS):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingFewDSWarmStart(
    nnUNetTrainerBraTSBlobLossSaturatingWarmStart,
    nnUNetTrainerBraTSBlobLossSaturatingFewDS,
):
    """Combines the reduced-LR warm-start fix with the reduced deep-supervision scales."""


class nnUNetTrainerBraTSBlobLossSaturatingFewDSWarmStart_250epochs(nnUNetTrainerBraTSBlobLossSaturatingFewDSWarmStart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingHighres(nnUNetTrainerBraTSBlobLossSaturating):
    """
    Same as nnUNetTrainerBraTSBlobLossSaturating, but recalibrated for the 3d_highres config's
    finer in-plane spacing ([1, 0.5, 0.5]mm, voxel volume 0.25mm3 vs fullres's 0.77mm3). The
    base class's small_voxel_thresh=27 assumes ~1mm3 voxels (already an approximation at
    fullres, where 27mm3 is actually ~35 voxels); at highres a real 27mm3 lesion is ~108
    voxels, not 27, so the base default would misclassify genuinely-small lesions as "large"
    and never saturate them. Uses small_voxel_thresh=108 instead; margin (plain-Dice-based,
    dimensionless, not spacing-dependent) is unchanged.
    """
    _region_loss_fn = staticmethod(functools.partial(saturating_region_loss, small_voxel_thresh=108))


class nnUNetTrainerBraTSBlobLossSaturatingFewDSHighres(nnUNetTrainerBraTSBlobLossSaturatingHighres):
    """
    Same recalibration idea as FewDS, but for highres spacing: a 27mm3 lesion is ~108 voxels
    (~4.76-voxel cube) at highres vs ~35 voxels (~3.27-voxel cube) at fullres, so it survives
    one more downsampling level before going sub-voxel — at 1/4 scale it's still ~1.19 voxels
    (borderline visible), only vanishing (~0.6 voxels) at 1/8. Zeros DS weight from scale index
    3 (1/8) onward instead of index 2 (1/4, what FewDS uses for fullres), keeping full-res,
    1/2, and 1/4 all contributing.
    """
    def _get_ds_loss_weights(self, deep_supervision_scales) -> np.ndarray:
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[3:] = 0
        return weights / weights.sum()


class nnUNetTrainerBraTSBlobLossSaturatingFewDSHighresWarmStart(
    nnUNetTrainerBraTSBlobLossSaturatingWarmStart,
    nnUNetTrainerBraTSBlobLossSaturatingFewDSHighres,
):
    """Combines the reduced-LR warm-start fix with the highres-recalibrated saturation + DS scales."""


class nnUNetTrainerBraTSBlobLossSaturatingFewDSHighresWarmStart_250epochs(
        nnUNetTrainerBraTSBlobLossSaturatingFewDSHighresWarmStart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingBalanced(nnUNetTrainerBraTSBlobLossSaturating):
    """
    Reduces the model's positive/FP bias and increases small-lesion sampling weight, per
    the 2026-07-22 leaderboard gap analysis: the ET/TC/WT DSC gap is substantially a
    false-positive problem (the model is deliberately recall-tilted, see loss-vs-metric-audit
    memory), and RC small-lesion recall is 0 in every run to date. Three changes, all
    independent of the FewDS/WarmStart mixins:

      1. Tversky alpha/beta 0.3/0.7 -> 0.4/0.6 (both the region-level Tversky and the
         per-instance blob term) — FN penalized 1.5x FP instead of 2.33x. Still recall-
         leaning (BraTS boundaries are noisy), just less so; softer than the audit's
         "don't touch beta" caution because this run is deliberately testing that tradeoff.
      2. Focal gamma 2.0 -> 1.0 — down-weights easy background less aggressively; the
         audit's cheapest, least recall-coupled FP lever.
      3. Instance-sampling exponent sqrt(size) [0.5] -> instance-uniform [0.0]
         (nnUNetDataLoaderInstanceUniformSmallBoost) — every TC|RC connected component
         gets equal selection probability regardless of size, maximizing small-lesion
         patch exposure. This is the OPPOSITE direction from the audit's DSC-only lever
         (0.5->0.75, which tilts toward large lesions); here we're trading some large-
         lesion/DSC sampling density for small-lesion recall, aimed at the 2026 detection
         leaderboard as well as DSC.
    """
    _region_loss_fn = staticmethod(functools.partial(saturating_region_loss, alpha=0.4, beta=0.6))
    _focal_gamma = 1.0
    _dataloader_cls = nnUNetDataLoaderInstanceUniformSmallBoost


class nnUNetTrainerBraTSBlobLossSaturatingBalancedFewDSWarmStart(
    nnUNetTrainerBraTSBlobLossSaturatingWarmStart,
    nnUNetTrainerBraTSBlobLossSaturatingFewDS,
    nnUNetTrainerBraTSBlobLossSaturatingBalanced,
):
    """Combines the reduced-LR warm-start fix, reduced DS scales, and the lower-positive-
    bias/higher-small-lesion-weight changes (Balanced). Warm-startable from the current
    reference checkpoint (nnUNetTrainerBraTSBlobLossSaturatingFewDSWarmStart_250epochs)."""


class nnUNetTrainerBraTSBlobLossSaturatingBalancedFewDSWarmStart_250epochs(
        nnUNetTrainerBraTSBlobLossSaturatingBalancedFewDSWarmStart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRC(nnUNetTrainerBraTSBlobLossSaturatingBalanced):
    """
    Same as Balanced, but decouples RC from TC in the blob loss instead of letting RC ride
    on TC's softened settings. Balanced's combined blob loss (base class) runs ONE
    connected-components pass over NETC|ET|RC together — since resection cavities
    routinely touch residual tumor, this frequently MERGES an RC instance into the same
    blob as adjacent ET/NETC, so RC voxels end up scored under whatever alpha/beta the
    merged blob gets, not an RC-specific one. Balanced softens alpha/beta (0.3/0.7 ->
    0.4/0.6) and focal gamma (2.0 -> 1.0) to cut ET/TC/WT's false-positive problem —
    reasonable there, but RC's small-lesion recall is already 0 in every run to date (see
    current-baseline memory), so silently inheriting that softening risks making RC's
    already-worst region worse.

    Uses BlobRegionWrapper's split mode (see its docstring): TC (NETC|ET) and RC get
    independent connected-components passes each step, so RC keeps its ORIGINAL, more
    recall-aggressive alpha/beta (0.3/0.7, unchanged from the pre-Balanced baseline) while
    TC still gets Balanced's softened 0.4/0.6. Bonus: RC's small-instance saturation gate
    (<27vox, see saturating_blob_term) is now computed on RC's own instance size, not a
    merged TC+RC blob's size — a more faithful gate than the combined pass ever gave RC.

    Caveat: summing two independent region losses (instead of one pooled combined call)
    roughly doubles the region-loss-family's magnitude relative to the un-split baseline;
    the warm-start's low initial_lr + short warmup should absorb this, but
    _tc_region_weight/_rc_region_weight are exposed on the base class if retuning is
    needed. Also costs one extra scipy.ndimage.label call per batch element per step (RC's
    CCs are typically small so shouldn't dominate runtime, but unlike the combined path's
    precomputed-comp_labels fast path, this one is NOT offloaded to the dataloader workers).
    """
    _rc_region_loss_fn = staticmethod(functools.partial(saturating_region_loss, alpha=0.3, beta=0.7))


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSWarmStart(
    nnUNetTrainerBraTSBlobLossSaturatingWarmStart,
    nnUNetTrainerBraTSBlobLossSaturatingFewDS,
    nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRC,
):
    """Combines the reduced-LR warm-start fix, reduced DS scales, and Balanced's TC/ET/WT
    de-biasing with RC decoupled (kept at the original alpha/beta 0.3/0.7, unaffected by
    the TC softening). Warm-startable from the current reference checkpoint
    (nnUNetTrainerBraTSBlobLossSaturatingFewDSWarmStart_250epochs)."""


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSWarmStart_250epochs(
        nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSWarmStart):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDS(
    nnUNetTrainerBraTSBlobLossSaturatingFewDS,
    nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRC,
):
    """Combines reduced DS scales with Balanced's TC/ET/WT de-biasing and RC decoupling,
    WITHOUT the WarmStart mixin — for training from scratch (fresh init) on a new fold,
    using nnU-Net's default optimizer/LR schedule (SGD, initial_lr=1e-2, poly decay)
    instead of the WarmStart mixin's low initial_lr=1e-3 (which assumes a converged
    starting point and would badly undertrain a random init)."""


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDS_750epochs(
        nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDS):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFocalWeighted(nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRC):
    """
    Same as SplitRC, but adds per-class weighting to the focal/CE term — SplitRC's blob
    loss gives RC its own recall-aggressive alpha/beta, but the per-voxel focal/CE term
    still weights every voxel equally regardless of class, and RC is 0.0047% of all
    training voxels (44.8x rarer than SNFH, 8.2x rarer than ET, 21328x rarer than
    background — measured over all 1037 fold_1 training cases' full-resolution segs).
    Across 358 logged epochs of the un-weighted SplitRC run, RC's per-voxel pseudo-dice
    never moved off exactly 0.0 (softmax prob ~1e-10 at true RC voxels even on large,
    >10k-voxel cavities) — the model defaults to background/SNFH at every RC voxel and the
    unweighted focal/CE mean never pushes back hard enough to change that, regardless of
    how much RC-centered sampling exposure it gets (case+instance-level sampling was
    verified separately and is not the bottleneck).

    Weight vector started from median-frequency balancing (Eigen & Fergus), computed over
    voxel frequency across the 4 foreground classes only; background is left at 1.0
    (unchanged) since focal's (1-p_t)^gamma term already suppresses its easy, correctly-
    classified voxels — reweighting it further risked destabilizing the already-solid
    background/foreground boundary for no clear benefit.

    First attempt used the literal median-frequency value for SNFH (0.117, 44.6x below
    RC's 5.221 — exactly cancelling their voxel-count ratio, since SNFH was identified via
    logit/intensity analysis as RC's actual closest competitor, not background). A 750-
    epoch run with that vector fixed RC (non-zero and climbing from epoch ~5 on) but broke
    SNFH instead: SNFH's pseudo-dice went to exactly 0.0 for 40+ straight epochs (same
    zero-true-positive signature RC had before) and showed no sign of recovering the way it
    did in an earlier, shorter smoke test — background was now out-competing SNFH the same
    way it previously out-competed RC. Run was stopped at epoch ~55.

    Current vector softens SNFH's weight to 0.35 (from 0.117) — still meaningfully down-
    weighted vs background, but not enough to lose the argmax competition outright:
        background=1.0, NETC=2.317, SNFH=0.35, ET=0.638, RC=5.221
    RC/SNFH ratio drops from 44.6x to 14.9x. Not derived from voxel counts anymore (that
    formula is what caused the SNFH collapse) — chosen empirically as a middle ground
    between the unweighted baseline (1.0, RC dead) and the literal median-frequency value
    (0.117, SNFH dead). May still need further tuning.

    This remains a starting point, not a fully validated fix — RC's near-total logit
    collapse may also require the network to actually learn to use FLAIR/T1 as RC-positive
    evidence (channel-ablation showed the model currently leans on T2, RC's WEAKEST
    discriminator from SNFH, while FLAIR/T1 — RC's STRONGEST discriminators from SNFH — are
    used against RC).
    """
    _focal_class_weights = (1.0, 2.317, 0.35, 0.638, 5.221)


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSFocalWeighted(
    nnUNetTrainerBraTSBlobLossSaturatingFewDS,
    nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFocalWeighted,
):
    """Combines reduced DS scales with SplitRC + the per-class focal/CE weighting above,
    WITHOUT the WarmStart mixin — matches
    nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDS_750epochs (the run that
    crashed at epoch 358 with RC pseudo-dice still at 0.0) except for the added focal
    weight, so the two runs are directly comparable."""


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSFocalWeighted_750epochs(
        nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSFocalWeighted):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFocalWeightedWarmStart(
    nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDSFocalWeighted
):
    """
    Warm-starts (via -pretrained_weights) from the crashed
    nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFewDS_750epochs run's checkpoint —
    same architecture/config, only the loss changes (adds the per-class focal weight) — to
    smoke-test whether RC's pseudo-dice can move off 0.0 at all.

    Deliberately uses a HIGH initial_lr (asserted below), the OPPOSITE of this codebase's
    usual WarmStart mixin (nnUNetTrainerBraTSBlobLossSaturatingWarmStart), which drops to
    1e-3 to protect already-good transferred features when warm-starting across a
    DIFFERENT architecture/resolution. That concern doesn't apply here: RC's features are
    dead (softmax prob ~1e-10 after 350 epochs), so the entire point of this run is to find
    out whether the new focal weight can move them AT ALL. A low LR would risk a false
    negative -- RC staying at 0.0 simply because the gradient was too small to matter within
    a short smoke-test, not because the loss reweighting doesn't work.
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-2
        assert self.initial_lr >= 1e-2, (
            f'this smoke test requires a high LR to find out whether the focal weight can '
            f'move RC at all -- got initial_lr={self.initial_lr}, expected >= 1e-2'
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = WarmupPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, warmup_steps=10)
        return optimizer, lr_scheduler


class nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFocalWeightedWarmStart_30epochs(
        nnUNetTrainerBraTSBlobLossSaturatingBalancedSplitRCFocalWeightedWarmStart):
    """Short smoke-test length: enough epochs (~1h at this config's ~120s/epoch) to see
    whether RC's pseudo-dice moves off 0.0, without committing to a full-length run."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 30
