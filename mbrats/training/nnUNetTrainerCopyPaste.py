"""
Copy-paste-augmented variants of nnUNetTrainerBraTS / nnUNetDataLoaderInstanceUniform.

Deliberately kept as new classes in a new file rather than editing
nnUNetTrainerBraTS.py, so the existing trainers (and everything already
trained with them, e.g. nnUNetTrainerBraTS_500epochs) stay untouched and
reproducible.

Requires the lesion library and paste-location masks to exist first:
    python -m mbrats.preprocessing.build_lesion_library
    python -m mbrats.preprocessing.build_paste_location_masks
"""

import os
from pathlib import Path

import blosc2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from threadpoolctl import threadpool_limits

from copy_paste import compute_instance_weights, find_valid_offset, paste_instance
from nnUNetTrainerBraTS import nnUNetDataLoaderInstanceUniform, nnUNetTrainerBraTS
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetDataLoaderCopyPaste(nnUNetDataLoaderInstanceUniform):
    """
    Same as nnUNetDataLoaderInstanceUniform, but after cropping (and before
    nnU-Net's own spatial/intensity augmentation) pastes 0-`max_pastes_per_sample`
    carved lesion instances from `lesion_library` into each sample, at
    locations drawn from the case's precomputed paste-location mask. Pasting
    before self.transforms means pasted content gets the same downstream
    augmentation as everything else in the patch.
    """

    def __init__(self, *args, lesion_library: list, paste_location_folder: str,
                 paste_probability: float = 0.5, max_pastes_per_sample: int = 2,
                 wt_fraction: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lesion_library = lesion_library
        self.instance_weights = compute_instance_weights(lesion_library) if lesion_library else None
        self.paste_location_folder = Path(paste_location_folder)
        self.paste_probability = paste_probability
        self.max_pastes_per_sample = max_pastes_per_sample
        self.wt_fraction = wt_fraction  # fraction of pastes using core+edema crop vs core-only
        self._paste_rng = np.random.default_rng()

    def _load_paste_mask(self, case_id: str):
        path = self.paste_location_folder / f"{case_id}_pastemask.b2nd"
        if not path.exists():
            return None
        return blosc2.open(str(path))[:]

    def _paste_into_sample(self, data_cropped: np.ndarray, seg_cropped: np.ndarray, case_id: str, bbox):
        if not self.lesion_library or self._paste_rng.random() > self.paste_probability:
            return

        full_valid_mask = self._load_paste_mask(case_id)
        if full_valid_mask is None:
            return
        valid_crop = crop_and_pad_nd(full_valid_mask[None], bbox, 0)[0].astype(bool)

        n_pastes = self._paste_rng.integers(1, self.max_pastes_per_sample + 1)
        for _ in range(n_pastes):
            idx = self._paste_rng.choice(len(self.lesion_library), p=self.instance_weights)
            inst = self.lesion_library[idx]
            use_wt = self._paste_rng.random() < self.wt_fraction
            source_image = inst['wt_image_crop'] if use_wt else inst['core_image_crop']
            source_seg = inst['wt_seg_crop'] if use_wt else inst['core_seg_crop']

            offset = find_valid_offset(valid_crop, source_seg.shape, rng=self._paste_rng)
            if offset is None:
                continue

            carved_mask, _ = paste_instance(data_cropped, seg_cropped, source_image, source_seg,
                                             offset, rng=self._paste_rng)
            # keep subsequent pastes in this sample from overlapping this one
            sl = tuple(slice(o, o + s) for o, s in zip(offset, carved_mask.shape))
            valid_crop[sl] &= ~carved_mask

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

                    data_cropped = crop_and_pad_nd(data, bbox, 0)
                    seg_cropped_np = crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)

                    self._paste_into_sample(data_cropped, seg_cropped_np[0], i, bbox)

                    data_cropped = torch.from_numpy(data_cropped).float()
                    seg_cropped = torch.from_numpy(seg_cropped_np).to(torch.int16)

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


class nnUNetTrainerBraTSCopyPaste(nnUNetTrainerBraTS):
    """
    nnUNetTrainerBraTS + copy-paste augmentation via nnUNetDataLoaderCopyPaste.
    Only get_dataloaders() differs from the parent (to swap in the new
    dataloader class) — everything else (loss, case sampling) is inherited
    unchanged.
    """

    LESION_LIBRARY_NAME = 'lesion_library.pkl'
    PASTE_PROBABILITY = 0.5
    MAX_PASTES_PER_SAMPLE = 2

    def get_dataloaders(self):
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

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

        library_path = os.path.join(os.path.dirname(self.preprocessed_dataset_folder), self.LESION_LIBRARY_NAME)
        lesion_library = load_pickle(library_path)
        self.print_to_log_file(f'Copy-paste: loaded {len(lesion_library)} instances from {library_path}')

        dl_tr = nnUNetDataLoaderCopyPaste(
            dataset_tr, self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=sampling_weights, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            lesion_library=lesion_library,
            paste_location_folder=self.preprocessed_dataset_folder,
            paste_probability=self.PASTE_PROBABILITY,
            max_pastes_per_sample=self.MAX_PASTES_PER_SAMPLE,
        )
        dl_val = nnUNetDataLoaderInstanceUniform(
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


class nnUNetTrainerBraTSCopyPaste_250epochs(nnUNetTrainerBraTSCopyPaste):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerBraTSCopyPaste_20epochs(nnUNetTrainerBraTSCopyPaste):
    """Throwaway smoke-test variant — enough epochs to exercise the real training loop."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20
