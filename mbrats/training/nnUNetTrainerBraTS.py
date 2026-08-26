"""
Custom nnUNet trainer for BraTS 2026 MET Task 1.

Changes vs default nnUNetTrainer:
  1. Focal loss  — replaces CE in the Dice+CE compound loss (gamma=2)
  2. Instance-uniform patch sampling — when centering on foreground, picks a
     connected component uniformly (not a voxel), so small lesions get equal
     sampling probability regardless of size.
  3. Class-frequency-balanced case sampling — rare classes (RC, NETC) get
     proportionally higher selection probability.
"""

from os.path import join
from typing import Union, Tuple, List

import numpy as np
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
from mbrats.training.lr_schedules import ResumeWarmupPolyLRScheduler
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform, ImageOnlyTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform


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
        super().__init__(soft_dice_kwargs, {}, weight_ce=weight_focal,
                         weight_dice=weight_dice, ignore_label=ignore_label,
                         dice_class=dice_class)
        if ignore_label is not None:
            focal_kwargs['ignore_index'] = ignore_label
        self.ce = FocalLoss(**focal_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Instance-uniform data loader
# ──────────────────────────────────────────────────────────────────────────────

class nnUNetDataLoaderInstanceUniform(nnUNetDataLoader):
    """
    Foreground patch sampling: instance-uniform within TC|RC connected components.

    When force_fg is True and tc_rc_instances is present in the case properties,
    picks a random instance (sqrt-size-weighted) and centres the patch on it.
    Falls back to voxel-uniform (parent behaviour) if tc_rc_instances is absent.

    Requires precomputed tc_rc_instances — run precompute_instance_locations.py first.
    """

    def _instance_uniform_bbox(self, shape: tuple, properties: dict):
        need_to_pad = self.need_to_pad.copy()
        dim = len(shape)
        for d in range(dim):
            if need_to_pad[d] + shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
               for i in range(dim)]

        instances = properties['tc_rc_instances']

        if not instances:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
            return bbox_lbs, bbox_ubs

        sizes = np.array([len(inst) for inst in instances], dtype=np.float64)
        probs = np.sqrt(sizes); probs /= probs.sum()
        chosen = instances[np.random.choice(len(instances), p=probs)]
        sv = chosen[np.random.randint(len(chosen))]
        center = [int(sv[i + 1]) for i in range(dim)]

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


class nnUNetDataLoaderInstanceUniformWeighted(nnUNetDataLoaderInstanceUniform):
    """
    Same as nnUNetDataLoaderInstanceUniform, but the instance-picking
    probability is size ** INSTANCE_SIZE_EXPONENT instead of the fixed
    sqrt(size) (exponent 0.5) used there. A lower exponent shifts exposure
    toward small lesions relative to the base class's sqrt weighting.
    """

    INSTANCE_SIZE_EXPONENT = 0.25

    def _instance_uniform_bbox(self, shape: tuple, properties: dict):
        need_to_pad = self.need_to_pad.copy()
        dim = len(shape)
        for d in range(dim):
            if need_to_pad[d] + shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
               for i in range(dim)]

        instances = properties['tc_rc_instances']

        if not instances:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
            return bbox_lbs, bbox_ubs

        sizes = np.array([len(inst) for inst in instances], dtype=np.float64)
        probs = sizes ** self.INSTANCE_SIZE_EXPONENT
        probs /= probs.sum()
        chosen = instances[np.random.choice(len(instances), p=probs)]
        sv = chosen[np.random.randint(len(chosen))]
        center = [int(sv[i + 1]) for i in range(dim)]

        bbox_lbs = [max(lbs[i], min(ubs[i], center[i] - self.patch_size[i] // 2))
                    for i in range(dim)]
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs


# ──────────────────────────────────────────────────────────────────────────────
# Custom trainer
# ──────────────────────────────────────────────────────────────────────────────

class nnUNetTrainerBraTS(nnUNetTrainer):
    """
    BraTS 2026 MET custom trainer.
    - Dice + Focal loss (gamma=2)
    - Instance-uniform foreground patch sampling on TC|RC components
    - Class-frequency-balanced case sampling
    """

    DATALOADER_TR_CLASS = nnUNetDataLoaderInstanceUniform

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

        return loss

    def _class_balanced_sampling_weights(self, dataset_tr) -> np.ndarray:
        """
        Per-case sampling probabilities based on inverse class frequency.
        Cases containing rare classes (RC, NETC) get higher selection probability.
        """
        from batchgenerators.utilities.file_and_folder_operations import load_pickle
        import os

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

        inv_freq = {l: n_cases / cnt for l, cnt in class_counts.items()}

        self.print_to_log_file('Case sampling — class inverse frequencies:')
        for l, w in sorted(inv_freq.items()):
            self.print_to_log_file(f'  label {l}: {class_counts[l]} cases  weight {w:.2f}x')

        weights = np.array(
            [sum(inv_freq[l] for l in present) if present else 1.0 for present in case_classes],
            dtype=np.float64,
        )
        return weights / weights.sum()

    def get_dataloaders(self):
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
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

        dl_tr = self.DATALOADER_TR_CLASS(
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


class nnUNetTrainerCheckpoint250(nnUNetTrainer):
    """Standard nnUNetTrainer with checkpoints saved every 250 epochs."""
    def on_epoch_end(self):
        current_epoch = self.current_epoch
        super().on_epoch_end()
        if (current_epoch + 1) % 250 == 0 and (current_epoch + 1) != self.num_epochs:
            self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch + 1:04d}.pth'))


class nnUNetTrainerBraTS_2epochs(nnUNetTrainerBraTS):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2


class nnUNetTrainerBraTS_250epochs(nnUNetTrainerBraTS):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTS_500epochs(nnUNetTrainerBraTS):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerBraTS_1000epochs(nnUNetTrainerBraTS):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000


class nnUNetTrainerBraTS_750epochs(nnUNetTrainerBraTS):
    """
    For resuming nnUNetTrainerBraTS_500epochs to 750 epochs total.

    PolyLRScheduler.step(current_epoch) recomputes LR from self.initial_lr and
    self.num_epochs every epoch (see on_train_epoch_start), overwriting whatever
    LR was restored from the checkpoint's optimizer state. Resuming at epoch 500
    with the default initial_lr=1e-2 and num_epochs=750 would jump the LR back
    up to ~5e-3 (progress 500/750=0.67 on a fresh 1e-2 schedule) even though it
    had already decayed to ~0 by the end of the 500-epoch run. Lowering
    initial_lr an order of magnitude keeps the post-resume LR small
    (~4e-4 at epoch 500) instead of re-inflating it.
    """
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750
        self.initial_lr = 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# T1/T2 modality-dropout trainer
#
# Ablation testing whether T1CE+FLAIR are ~sufficient for BraTS-METS segmentation:
# T1 and T2 are independently zeroed out during training. Channels 4/5
# (T1c-T1n, T1c/T1n) are derived FROM T1, so whenever T1 is dropped they're
# dropped too — otherwise the network could reconstruct T1 from T1CE minus the
# still-present subtraction/ratio channel, defeating the point of dropping it.
# ──────────────────────────────────────────────────────────────────────────────

class ModalityDropoutTransform(ImageOnlyTransform):
    """
    Independently zeroes the T1 and/or T2 channels (post-normalisation zero is
    the z-scored mean, i.e. a flat/uninformative channel), each with its own
    probability. Dropping T1 also zeroes the channels derived from it
    (`derived_from_t1_channels`), since those otherwise leak T1 back in.
    """

    def __init__(self, t1_channel: int, t2_channel: int, derived_from_t1_channels: Tuple[int, ...],
                 p_t1: float, p_t2: float):
        super().__init__()
        self.t1_channel = t1_channel
        self.t2_channel = t2_channel
        self.derived_from_t1_channels = derived_from_t1_channels
        self.p_t1 = p_t1
        self.p_t2 = p_t2

    def get_parameters(self, **data_dict) -> dict:
        return {
            'drop_t1': np.random.uniform() < self.p_t1,
            'drop_t2': np.random.uniform() < self.p_t2,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if params['drop_t1']:
            img[self.t1_channel] = 0
            for ch in self.derived_from_t1_channels:
                img[ch] = 0
        if params['drop_t2']:
            img[self.t2_channel] = 0
        return img


class nnUNetTrainerModDropout(nnUNetTrainer):
    """
    Standard nnUNetTrainer (default Dice+CE loss, default augmentation) plus
    three minimal additions:
      - T1 and T2 are independently zeroed out during training (see
        ModalityDropoutTransform), to push the network toward not depending on
        them being present. Dropping T1 also zeroes the two channels derived
        from it (T1c-T1n, T1c/T1n), so they can't be used to reconstruct T1.
      - Inverse-SQRT class-frequency case sampling (softer than plain inverse
        frequency — rare classes are still favored, just less aggressively).
      - Instance-uniform patch sampling lightly favoring bigger lesions
        (size ** 0.25, see nnUNetDataLoaderInstanceUniformWeighted) instead of
        plain nnU-Net's voxel-level foreground sampling (fully size-proportional).
      - Softer random-scaling augmentation (0.7-1.4 -> 0.85-1.15).
    """

    T1_CHANNEL = 0
    T2_CHANNEL = 2
    DERIVED_FROM_T1_CHANNELS = (4, 5)  # T1c-T1n, T1c/T1n
    MOD_DROPOUT_P_T1 = 0.2
    MOD_DROPOUT_P_T2 = 0.2
    SCALING_RANGE = (0.85, 1.15)

    DATALOADER_TR_CLASS = nnUNetDataLoaderInstanceUniformWeighted

    def _inv_sqrt_class_sampling_weights(self, dataset_tr) -> np.ndarray:
        """Per-case sampling probabilities, weighted by sqrt(inverse class frequency)."""
        from batchgenerators.utilities.file_and_folder_operations import load_pickle
        import os

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

        inv_sqrt_freq = {l: np.sqrt(n_cases / cnt) for l, cnt in class_counts.items()}

        self.print_to_log_file('Case sampling — inverse-sqrt class frequencies:')
        for l, w in sorted(inv_sqrt_freq.items()):
            self.print_to_log_file(f'  label {l}: {class_counts[l]} cases  weight {w:.2f}x')

        weights = np.array(
            [sum(inv_sqrt_freq[l] for l in present) if present else 1.0 for present in case_classes],
            dtype=np.float64,
        )
        return weights / weights.sum()

    def get_dataloaders(self):
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
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

        sampling_weights = self._inv_sqrt_class_sampling_weights(dataset_tr)

        dl_tr = self.DATALOADER_TR_CLASS(
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

    def get_training_transforms(
            self, patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=None, is_cascaded=False,
            foreground_labels=None, regions=None, ignore_label=None) -> BasicTransform:
        composed = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded,
            foreground_labels=foreground_labels, regions=regions, ignore_label=ignore_label)

        for t in composed.transforms:
            if isinstance(t, SpatialTransform):
                t.scaling = self.SCALING_RANGE

        # appended last so it runs after intensity augmentations (noise/gamma/etc.
        # renormalise per-channel using that channel's own stats — running before
        # them would make a zeroed channel non-zero again, and gamma's retain-stats
        # rescaling divides by a zero-variance channel's std)
        composed.transforms.append(ModalityDropoutTransform(
            t1_channel=self.T1_CHANNEL, t2_channel=self.T2_CHANNEL,
            derived_from_t1_channels=self.DERIVED_FROM_T1_CHANNELS,
            p_t1=self.MOD_DROPOUT_P_T1, p_t2=self.MOD_DROPOUT_P_T2,
        ))
        return composed


class nnUNetTrainerModDropout_250epochs(nnUNetTrainerModDropout):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerModDropout_500epochs(nnUNetTrainerModDropout):
    """
    Extends nnUNetTrainerModDropout_250epochs's poly-decay budget from 250 to
    500 epochs. That run was stopped at the epoch-199 checkpoint_latest.pth
    (200 epochs done) because EMA pseudo-dice was still climbing — just
    slower, tracking the LR decay — right up to the stop point, suggesting
    the 250-epoch schedule was cutting the LR to ~0 before convergence.

    Uses the *native* 500-epoch schedule value (initial_lr=1e-2, nnU-Net's
    default) rather than matching the already-decayed ~2.35e-3 the old
    250-epoch schedule had reached at epoch 200 — a genuine from-scratch
    500-epoch run would still be at ~6.3e-3 at that point (0.01 * 0.6**0.9),
    and matching the decayed value would just lock in the under-scheduling
    this class exists to fix. That's a real (2.7x) LR increase on
    already-trained weights, though nowhere near the jump that caused this
    project's one documented resume failure (RC dice stuck at 0 for 183/250
    epochs — see lr_schedules.py) — that was a restart from a fully-decayed,
    near-zero checkpoint straight back to 1e-2, an unbounded relative jump,
    on far more fine-tuned/fragile weights than 200 epochs-in.
    ResumeWarmupPolyLRScheduler still ramps into it gently over 10 epochs
    (200->210) rather than applying it in one step, as a cheap hedge against
    that risk; see configure_optimizers().

    Launch via resume_from_checkpoint.py (preserves optimizer/epoch state),
    not nnUNetv2_train directly:
        python -m mbrats.training.resume_from_checkpoint \\
            -d 1 -c 3d_fullres -f 0 \\
            -tr nnUNetTrainerModDropout_500epochs -p nnUNetResEncUNetMPlans \\
            --init_checkpoint nnunet_results/Dataset001_BraTSMETS/nnUNetTrainerModDropout_250epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_latest.pth
    """

    RESUME_START_EPOCH = 200  # current_epoch restored from checkpoint_latest.pth
    WARMUP_EPOCHS = 10

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.initial_lr = 0.01

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = ResumeWarmupPolyLRScheduler(
            optimizer, self.initial_lr, self.num_epochs,
            warmup_start_step=self.RESUME_START_EPOCH, warmup_steps=self.WARMUP_EPOCHS,
        )
        return optimizer, lr_scheduler
