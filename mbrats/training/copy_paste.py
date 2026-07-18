"""
Copy-paste augmentation: carve a random sub-region of a lesion instance
(from build_lesion_library.py) and hard-composite it into a training patch.

Carving follows CarveMix (Zhang et al., "CarveMix: A Simple Data Augmentation
Method for Brain Lesion Segmentation", MICCAI 2021 — formulation taken from
their reference implementation, Task100_ATLASwithCarveMix/Simple_CarveMix.py):
a signed distance transform of the lesion mask is thresholded at a randomly
sampled level, so the carved region either shrinks into the lesion's interior
or grows outward into surrounding context, following the lesion's own shape
rather than a fixed bounding box. CarveMix's own ablations found hard
replacement (no alpha blending) sufficient once the mask boundary follows
real anatomy instead of a rectangle — mirrored here for the same reason.

CarveMix itself mixes two images at the *same* voxel position (their dataset
is in a common registered space). We paste to a *new* location instead, so
placement is handled separately (see build_paste_location_masks.py); this
module only implements the carve + hard-composite step.
"""

import numpy as np
from scipy import ndimage

# nnU-Net's global z-scoring maps out-of-brain background to a negative
# constant (~-0.3 across every modality), not 0, so "!= 0" doesn't find it.
# Brain tissue sits well above 0 in the channel mean; air well below. A
# threshold at 0 separates them with a large margin (see the 00675-000 crop:
# air mean -0.28, brain mean +3.1, lesion 0% below 0).
FOREGROUND_TAU = 0.0
# The brain/air boundary carries a 1-voxel partial-volume band that's dim in
# every modality but still passes the threshold; pasting it leaves a dark rim.
# Eroding the mask by 1 voxel stops the paste in clean brain (drops the dim
# shell entirely, keeps ~99.6% of lesion voxels — see the 00675-000 crop).
BRAIN_ERODE = 1


def foreground_mask(image_crop: np.ndarray) -> np.ndarray:
    """In-brain mask for a (C, ...) crop: channel-mean above FOREGROUND_TAU,
    eroded by BRAIN_ERODE to shed the dim brain/air partial-volume band."""
    brain = image_crop.mean(axis=0) > FOREGROUND_TAU
    if BRAIN_ERODE:
        brain = ndimage.binary_erosion(brain, iterations=BRAIN_ERODE)
    return brain


def find_valid_offset(valid_mask: np.ndarray, instance_shape, rng: np.random.Generator = None, max_tries: int = 50):
    """
    Find a random offset (lower corner) such that a box of `instance_shape`
    placed at that offset lies entirely within `valid_mask`. Cheap rejection
    sampling rather than a full erosion — fine given valid_mask is usually
    ~90% True. Returns None if no valid offset found (instance doesn't fit,
    or unlucky draws) after `max_tries`; callers should just skip that paste.
    """
    if rng is None:
        rng = np.random.default_rng()
    max_offset = [s - i for s, i in zip(valid_mask.shape, instance_shape)]
    if any(m < 0 for m in max_offset):
        return None

    for _ in range(max_tries):
        offset = tuple(int(rng.integers(0, m + 1)) for m in max_offset)
        sl = tuple(slice(o, o + s) for o, s in zip(offset, instance_shape))
        if valid_mask[sl].all():
            return offset
    return None


def compute_instance_weights(library: list, size_key: str = 'wt', bias_power: float = 1.0,
                              min_size: int = 10) -> np.ndarray:
    """
    Sampling weights favoring small instances (1 / size ** bias_power),
    further divided by how many instances came from the same case, so a
    handful of cases with many small satellite lesions don't dominate the
    paste distribution. `size_key`: 'wt' (core+edema) or 'tc' (core only)
    voxel count used for the size term.

    Instances smaller than `min_size` get zero weight — excluded from
    selection entirely, not just discounted. A 1-2 voxel fragment isn't a
    lesion shape to teach from regardless of how it's weighted: an unbounded
    1/size term puts ~49% of all mass on single-voxel instances (see
    eda/size_weight_distribution.png), which paste as invisible single-pixel
    label noise and measurably hurt small-lesion recall in practice (see
    results/copypaste250_fold0_cv_eval.* vs brats_500 baseline).
    """
    if size_key == 'tc':
        sizes = np.array([inst['n_voxels_netc'] + inst['n_voxels_et'] + inst['n_voxels_rc'] for inst in library])
    else:
        sizes = np.array([inst['n_voxels_netc'] + inst['n_voxels_et'] + inst['n_voxels_rc'] + inst['n_voxels_snfh']
                           for inst in library])
    eligible = sizes >= min_size
    sizes = np.maximum(sizes, 1)

    case_counts = {}
    for inst in library:
        case_counts[inst['case_id']] = case_counts.get(inst['case_id'], 0) + 1
    per_case_weight = np.array([1.0 / case_counts[inst['case_id']] for inst in library])

    weights = (1.0 / sizes ** bias_power) * per_case_weight * eligible
    return weights / weights.sum()


def signed_distance(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Negative inside mask (more negative toward the interior), positive outside."""
    edt = ndimage.distance_transform_edt
    return np.where(mask, -edt(mask, sampling=spacing), edt(~mask, sampling=spacing))


def carve_mask(mask: np.ndarray, spacing=(1.0, 1.0, 1.0), rng: np.random.Generator = None):
    """
    Randomly carve a sub-region of `mask`: shrink into the interior, or grow
    outward into surrounding context, by an amount sampled per CarveMix.

    Returns (carved_mask, lam). If mask is all-False, returns mask unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not mask.any():
        return mask.copy(), 0.0

    dist = signed_distance(mask, spacing)
    c = rng.beta(1, 1)
    c = (c - 0.5) * 2  # U(-1, 1)
    min_dist = dist.min()
    lam = (c * min_dist / 2) if c > 0 else (c * min_dist)
    return dist < lam, float(lam)


def paste_instance(target_image: np.ndarray, target_seg: np.ndarray,
                    source_image_crop: np.ndarray, source_seg_crop: np.ndarray,
                    offset, spacing=(1.0, 1.0, 1.0), feather: float = 1.5,
                    rng: np.random.Generator = None):
    """
    Carve a random sub-region of the source instance and composite it into
    target_image/target_seg at `offset` (lower corner, in target-patch voxel
    coords). Modifies target_image/target_seg in place.

    target_image: (C, H, W, D), target_seg: (H, W, D)
    source_image_crop: (C, h, w, d), source_seg_crop: (h, w, d) — one
    instance's stored crop from the lesion library (e.g. 'core_*' or 'wt_*').

    Two adjustments over a plain hard paste, both to hide the cross-location
    seam CarveMix's same-position mixing never had to deal with:
      * out-of-brain source voxels are excluded, so an edge lesion whose
        grow-carve balloons into the surrounding air never stamps a black
        rim into the target interior (see FOREGROUND_TAU);
      * the *image* is alpha-blended over a `feather`-voxel ramp at the paste
        boundary instead of hard-replaced. The *segmentation* is still a hard
        integer paste — labels must not blend.

    Returns (paste_mask, lam) — the effective (in-brain) label footprint — for
    diagnostics/visualization.
    """
    lesion_mask = source_seg_crop != 0
    carved_mask, lam = carve_mask(lesion_mask, spacing=spacing, rng=rng)

    # gate out air / brain-surface dim band, but never erode away actual lesion
    brain = foreground_mask(source_image_crop)
    paste_mask = carved_mask & (brain | lesion_mask)

    if feather > 0 and paste_mask.any():
        inside_dist = ndimage.distance_transform_edt(paste_mask, sampling=spacing)
        alpha = np.clip(inside_dist / feather, 0.0, 1.0)
    else:
        alpha = paste_mask.astype(source_image_crop.dtype)

    sl = tuple(slice(o, o + s) for o, s in zip(offset, carved_mask.shape))

    target_view = target_image[(slice(None),) + sl]
    target_image[(slice(None),) + sl] = alpha[None] * source_image_crop + (1.0 - alpha[None]) * target_view
    target_seg[sl] = np.where(paste_mask, source_seg_crop, target_seg[sl])

    return paste_mask, lam
