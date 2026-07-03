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
    # p, g: (B, ...) probabilities and binary GT for ONE region
    tp = (p * g).sum(dim=tuple(range(1, p.ndim)))
    fp = (p * (1 - g)).sum(dim=tuple(range(1, p.ndim)))
    fn = ((1 - p) * g).sum(dim=tuple(range(1, p.ndim)))
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - ti) ** gamma).mean()


_BLOB_CHUNK = 8   # components per chunk — bounds memory to ~8 × patch × 4 intermediate tensors
_BLOB_MAX   = 64  # skip cases with more CCs (elastic deformation fragments → noise anyway)

def blob_term(p, comp_labels, n_comp, alpha=0.3, beta=0.7):
    if n_comp == 0 or n_comp > _BLOB_MAX:
        return p.new_tensor(0.0)
    all_fg = (comp_labels > 0).float()
    cl_exp = comp_labels.unsqueeze(0)
    ndim   = p.ndim
    total  = p.new_tensor(0.0)
    for start in range(1, n_comp + 1, _BLOB_CHUNK):
        ids     = torch.arange(start, min(start + _BLOB_CHUNK, n_comp + 1), device=p.device)
        chunk   = len(ids)
        ids_    = ids.view(chunk, *([1] * ndim))
        gi      = (cl_exp == ids_).float()          # (chunk, *spatial)
        pi      = p.unsqueeze(0) * (1 - (all_fg - gi).clamp(0, 1))
        tp      = (pi * gi).sum(dim=tuple(range(1, ndim + 1)))
        fp      = (pi * (1 - gi)).sum(dim=tuple(range(1, ndim + 1)))
        fn      = ((1 - pi) * gi).sum(dim=tuple(range(1, ndim + 1)))
        ti      = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        total   = total + ((1 - ti) ** 1.33).sum()
    return total / n_comp


def region_loss(p, g, comp_labels, n_comp, lam=1.5):
    p_c = p.clamp(1e-6, 1 - 1e-6)
    bce = -(g * p_c.log() + (1 - g) * (1 - p_c).log()).mean()
    return tversky(p, g) + bce + lam * blob_term(p, comp_labels, n_comp)


class BlobRegionWrapper(nn.Module):
    """
    Wraps a deep-supervision loss and adds a TC|RC blob region loss at full resolution.
    TC|RC = NETC (ch 1) + ET (ch 3) + RC (ch 4).
    CC computation runs on CPU per batch element (scipy); acceptable at batch_size=2.
    """

    def __init__(self, ds_loss, blob_weight: float = 0.5, ignore_label=None):
        super().__init__()
        self.ds_loss = ds_loss
        self.blob_weight = blob_weight
        self.ignore_label = ignore_label if ignore_label is not None else -100

    def forward(self, net_output, target):
        # strip comp_labels channel before passing to DS loss (expects only seg labels)
        if isinstance(target, (list, tuple)):
            target_seg = [t[:, :1] if t.shape[1] > 1 else t for t in target]
        else:
            target_seg = target[:, :1] if target.shape[1] > 1 else target

        main_loss = self.ds_loss(net_output, target_seg)
        full_logits = net_output[0] if isinstance(net_output, (list, tuple)) else net_output
        full_target = target[0] if isinstance(target, (list, tuple)) else target
        # disable autocast: binary_cross_entropy is unsafe in AMP context
        with torch.amp.autocast(full_logits.device.type, enabled=False):
            blob_loss = self._tc_rc_blob_loss(full_logits.float(), full_target)
        return main_loss + self.blob_weight * blob_loss

    def _tc_rc_blob_loss(self, logits, seg):
        probs = F.softmax(logits, dim=1)          # (B, C, *spatial)
        tc_rc_prob = (probs[:, 1] + probs[:, 3] + probs[:, 4]).clamp(0.0, 1.0)  # NETC + ET + RC

        seg_3d = seg[:, 0].long()                 # (B, *spatial)
        valid = seg_3d != self.ignore_label
        tc_rc_gt = ((seg_3d == 1) | (seg_3d == 3) | (seg_3d == 4)).float()
        tc_rc_gt = tc_rc_gt * valid.float()

        # comp_labels precomputed in data loader and passed as seg channel 1
        has_precomputed = seg.shape[1] >= 2
        loss = logits.new_tensor(0.0)
        for b in range(logits.shape[0]):
            if has_precomputed:
                comp_labels = seg[b, 1].long().to(logits.device)
                n_comp = int(comp_labels.max().item())
            else:
                import scipy.ndimage
                gt_np = tc_rc_gt[b].cpu().numpy()
                comp_np, n_comp = scipy.ndimage.label(gt_np)
                comp_labels = torch.from_numpy(comp_np).to(logits.device)
            loss = loss + region_loss(tc_rc_prob[b], tc_rc_gt[b], comp_labels, n_comp)
        return loss / logits.shape[0]


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

    def _instance_uniform_bbox(self, shape: tuple, properties: dict):
        """
        Returns (bbox_lbs, bbox_ubs) centered on a sqrt-size-weighted TC instance.

        tc_rc_instances is a flat list of CC arrays (ET | NETC combined mask).
        Instances are selected with sqrt(n_voxels) weighting so large lesions get
        proportionally more pulls without completely starving small ones.
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

        # pick instance with sqrt-size weighting, then a random voxel within it
        sizes = np.array([len(inst) for inst in instances], dtype=np.float64)
        probs = np.sqrt(sizes); probs /= probs.sum()
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


# ──────────────────────────────────────────────────────────────────────────────
# Custom trainer
# ──────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerBraTSBlobLoss(nnUNetTrainer):
    """
    BraTS 2026 MET custom trainer.
    - Dice + Focal loss (gamma=2)
    - Instance-uniform foreground patch sampling
    """

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerBraTS expects label-based (not region-based) training"

        loss = DC_and_Focal_loss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                              'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            focal_kwargs={'gamma': 2.0},
            weight_focal=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return BlobRegionWrapper(loss, blob_weight=0.5,
                                 ignore_label=self.label_manager.ignore_label)

    def _class_balanced_sampling_weights(self, dataset_tr) -> np.ndarray:
        """
        Per-case sampling probabilities based on inverse class frequency.

        For each case, weight = sum of (n_cases / n_cases_with_class) for every
        foreground class present. Cases containing rare classes (RC, NETC) get
        proportionally higher selection probability than common-class-only cases.
        """
        from batchgenerators.utilities.file_and_folder_operations import load_pickle
        import os

        # pass 1: collect which classes each case has
        case_classes = []
        for key in dataset_tr.identifiers:
            pkl_path = os.path.join(self.preprocessed_dataset_folder, key + '.pkl')
            try:
                props = load_pickle(pkl_path)
            except Exception:
                case_classes.append(frozenset())
                continue
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

        # pass 2: per-case weight = sum of inv_freq for its classes
        weights = np.array(
            [sum(inv_freq[l] for l in present) if present else 1.0 for present in case_classes],
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

        dl_tr = nnUNetDataLoaderInstanceUniform(
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
