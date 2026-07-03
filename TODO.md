# TODO

## Switch nnUNet install from editable to PyPI

Currently `nnUNet/` is a local git clone installed as `pip install -e nnUNet/`.

**Option:** install from PyPI instead (`nnunetv2==2.8.0` is confirmed to include
`nnUNet_extTrainer` support), remove the `nnUNet/` directory entirely, and add
`nnunetv2>=2.8.0` back to `pyproject.toml` dependencies.

Steps when ready:
1. Add `"nnunetv2>=2.8.0"` to `pyproject.toml` dependencies
2. Remove `pip install -e nnUNet/` from `scripts/install.sh`
3. `pip install nnunetv2==2.8.0 && rm -rf nnUNet/`
4. Remove the `nnUNet/` comment from `.gitignore`
