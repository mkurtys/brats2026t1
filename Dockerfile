FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /opt/brats

# Install nnunetv2 from the vendored clone
COPY nnUNet/ nnUNet/
RUN pip install --no-cache-dir -e nnUNet/

# Install mbrats and all dependencies
COPY pyproject.toml .
COPY mbrats/ mbrats/
RUN pip install --no-cache-dir -e .

COPY scripts/ scripts/

# nnUNet data dirs are mounted at runtime under /data
ENV nnUNet_raw=/data/nnunet_raw \
    nnUNet_preprocessed=/data/nnunet_preprocessed \
    nnUNet_results=/data/nnunet_results \
    nnUNet_extTrainer=/opt/brats/mbrats/training
