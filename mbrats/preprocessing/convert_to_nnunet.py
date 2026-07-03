"""
Convert BraTS 2025/2026 MET training data to nnU-Net Dataset format.

Source layout (one or two levels deep):
  <train_dir>/BraTS-MET-{id}/{id}-{t1n,t1c,t2w,t2f,seg}.nii.gz
  <train_dir>/UCSD - Training/BraTS-MET-{id}/{id}-{t1n,t1c,t2w,t2f,seg}.nii.gz

Output layout (nnU-Net Dataset001):
  imagesTr/BraTS-MET-{id}_0000.nii.gz  (T1)
  imagesTr/BraTS-MET-{id}_0001.nii.gz  (T1CE)
  imagesTr/BraTS-MET-{id}_0002.nii.gz  (T2)
  imagesTr/BraTS-MET-{id}_0003.nii.gz  (T2-FLAIR)
  labelsTr/BraTS-MET-{id}.nii.gz
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

MODALITY_MAP = {
    "t1n": "_0000",
    "t1c": "_0001",
    "t2w": "_0002",
    "t2f": "_0003",
}

DATASET_JSON = {
    "channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"},
    "labels": {"background": 0, "NETC": 1, "ET": 2, "SNFH": 3, "RC": 4},
    "regions_class_order": [1, 2, 3, 4],
    "file_ending": ".nii.gz",
    "dataset_name": "BraTSMETS",
    "reference": "https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097",
    "licence": "see challenge page",
    "release": "BraTS2025/2026",
    "overwrite_image_reader_writer": "SimpleITKIO",
}


def find_cases(train_dir: Path) -> dict[str, Path]:
    """Return {case_id: case_dir} for all valid BraTS-MET cases, two levels deep."""
    cases = {}
    for entry in sorted(train_dir.iterdir()):
        if entry.name.startswith("._") or not entry.is_dir():
            continue
        if entry.name.startswith("BraTS-MET-"):
            cases[entry.name] = entry
        elif entry.is_dir():
            # subdirectory (e.g. "UCSD - Training")
            for sub in sorted(entry.iterdir()):
                if sub.name.startswith("._") or not sub.is_dir():
                    continue
                if sub.name.startswith("BraTS-MET-"):
                    cases[sub.name] = sub
    return cases


def apply_corrected_labels(labels_dir: Path) -> dict[str, Path]:
    """Return {case_id: corrected_seg_path} from corrected-labels directories."""
    overrides = {}
    if not labels_dir.exists():
        return overrides
    for f in labels_dir.rglob("*-seg.nii.gz"):
        if f.name.startswith("._"):
            continue
        case_id = f.name.replace("-seg.nii.gz", "")
        overrides[case_id] = f
    return overrides


def compute_subtraction(t1c_path: Path, t1n_path: Path, dst: Path) -> None:
    """Save T1c - T1n as a new NIfTI file (float32)."""
    t1c_img = nib.load(str(t1c_path))
    t1n = nib.load(str(t1n_path)).get_fdata(dtype=np.float32)
    diff = t1c_img.get_fdata(dtype=np.float32) - t1n
    nib.save(nib.Nifti1Image(diff, t1c_img.affine, t1c_img.header), str(dst))


def compute_ratio(t1c_path: Path, t1n_path: Path, dst: Path) -> None:
    """Save T1c / (T1n + eps) as a new NIfTI file (float32).

    eps is set to 1% of the mean foreground T1n signal, making it scale-invariant
    across scanners. Background (T1n == 0 and T1c == 0) is set to 1.0 (no enhancement).
    """
    t1c_img = nib.load(str(t1c_path))
    t1c = t1c_img.get_fdata(dtype=np.float32)
    t1n = nib.load(str(t1n_path)).get_fdata(dtype=np.float32)
    brain_mask = t1n > 0
    eps = float(t1n[brain_mask].mean()) * 0.01 if brain_mask.any() else 1.0
    ratio = t1c / (t1n + eps)
    # background voxels → 1.0 (neutral, no enhancement)
    ratio[~brain_mask] = 1.0
    nib.save(nib.Nifti1Image(ratio, t1c_img.affine, t1c_img.header), str(dst))


def convert(
    train_dir: Path,
    output_dir: Path,
    corrected_labels_dirs: list[Path],
    val_dir: Path | None,
    use_symlinks: bool,
    add_subtraction: bool,
    add_ratio: bool,
) -> None:
    images_tr = output_dir / "imagesTr"
    labels_tr = output_dir / "labelsTr"
    images_val = output_dir / "imagesVal"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)

    # collect corrected label overrides
    overrides: dict[str, Path] = {}
    for d in corrected_labels_dirs:
        overrides.update(apply_corrected_labels(d))
    if overrides:
        print(f"Corrected labels for {len(overrides)} cases: {list(overrides)}")

    cases = find_cases(train_dir)
    print(f"Found {len(cases)} training cases")

    link_or_copy = os.symlink if use_symlinks else shutil.copy2
    converted = 0
    skipped = 0

    for case_id, case_dir in sorted(cases.items()):
        files = {f.stem.replace(".nii", "").split("-")[-1]: f
                 for f in case_dir.iterdir()
                 if not f.name.startswith("._") and f.suffix in (".gz", ".nii")}

        missing_mods = [m for m in MODALITY_MAP if m not in files]
        if missing_mods:
            print(f"  SKIP {case_id}: missing {missing_mods}")
            skipped += 1
            continue

        for mod, suffix in MODALITY_MAP.items():
            src = files[mod].resolve()
            dst = images_tr / f"{case_id}{suffix}.nii.gz"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if use_symlinks:
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)

        next_ch = 4
        if add_subtraction:
            dst = images_tr / f"{case_id}_{next_ch:04d}.nii.gz"
            if dst.exists():
                dst.unlink()
            compute_subtraction(files["t1c"], files["t1n"], dst)
            next_ch += 1
        if add_ratio:
            dst = images_tr / f"{case_id}_{next_ch:04d}.nii.gz"
            if dst.exists():
                dst.unlink()
            compute_ratio(files["t1c"], files["t1n"], dst)

        seg_src = overrides.get(case_id, files.get("seg"))
        if seg_src is None:
            print(f"  SKIP {case_id}: no segmentation")
            skipped += 1
            continue
        seg_dst = labels_tr / f"{case_id}.nii.gz"
        if seg_dst.exists() or seg_dst.is_symlink():
            seg_dst.unlink()
        if use_symlinks:
            os.symlink(seg_src.resolve(), seg_dst)
        else:
            shutil.copy2(seg_src, seg_dst)

        converted += 1

    print(f"Training: {converted} converted, {skipped} skipped")

    # validation images (no labels)
    if val_dir and val_dir.exists():
        val_cases = find_cases(val_dir)
        print(f"Found {len(val_cases)} validation cases")
        for case_id, case_dir in sorted(val_cases.items()):
            files = {f.stem.replace(".nii", "").split("-")[-1]: f
                     for f in case_dir.iterdir()
                     if not f.name.startswith("._") and f.suffix in (".gz", ".nii")}
            for mod, suffix in MODALITY_MAP.items():
                if mod not in files:
                    continue
                src = files[mod].resolve()
                dst = images_val / f"{case_id}{suffix}.nii.gz"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if use_symlinks:
                    os.symlink(src, dst)
                else:
                    shutil.copy2(src, dst)
            if "t1c" in files and "t1n" in files:
                next_ch = 4
                if add_subtraction:
                    dst = images_val / f"{case_id}_{next_ch:04d}.nii.gz"
                    if dst.exists():
                        dst.unlink()
                    compute_subtraction(files["t1c"], files["t1n"], dst)
                    next_ch += 1
                if add_ratio:
                    dst = images_val / f"{case_id}_{next_ch:04d}.nii.gz"
                    if dst.exists():
                        dst.unlink()
                    compute_ratio(files["t1c"], files["t1n"], dst)

    dataset_json = dict(DATASET_JSON)
    dataset_json["numTraining"] = converted
    next_ch = 4
    if add_subtraction:
        dataset_json["channel_names"][str(next_ch)] = "T1c-T1n"
        next_ch += 1
    if add_ratio:
        dataset_json["channel_names"][str(next_ch)] = "T1c/T1n"
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Wrote dataset.json ({converted} training cases)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--corrected_labels", nargs="*", type=Path, default=[])
    parser.add_argument("--val_dir", type=Path, default=None)
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking (slower but standalone)")
    parser.add_argument("--add_subtraction", action="store_true",
                        help="Add T1c-T1n subtraction channel")
    parser.add_argument("--add_ratio", action="store_true",
                        help="Add T1c/(T1n+eps) ratio channel")
    args = parser.parse_args()
    convert(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        corrected_labels_dirs=args.corrected_labels or [],
        val_dir=args.val_dir,
        use_symlinks=not args.copy,
        add_subtraction=args.add_subtraction,
        add_ratio=args.add_ratio,
    )


if __name__ == "__main__":
    main()
