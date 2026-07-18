"""
Plot how compute_instance_weights' size bias distributes paste-selection
probability across the lesion library — specifically to check how much
mass concentrates on near-empty (1-2 voxel) instances.

Usage:
    python -m mbrats.analysis.eda_copy_paste_weights
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mbrats" / "training"))
from copy_paste import compute_instance_weights  # noqa: E402

LIBRARY_PATH = Path("nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl")
OUT_PATH = Path("eda/size_weight_distribution.png")


def main():
    library = load_pickle(str(LIBRARY_PATH))
    sizes = np.array([inst['n_voxels_netc'] + inst['n_voxels_et'] + inst['n_voxels_rc'] + inst['n_voxels_snfh']
                       for inst in library])
    weights = compute_instance_weights(library)

    order = np.argsort(sizes)
    sizes_sorted = sizes[order]
    cum_weight = np.cumsum(weights[order])
    mass_at_1 = weights[sizes == 1].sum()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(sizes, bins=np.logspace(0, np.log10(sizes.max()), 60))
    axes[0].set_xscale('log')
    axes[0].set_xlabel('instance size (voxels)')
    axes[0].set_ylabel('count')
    axes[0].set_title('library size distribution (log-x)')
    axes[0].axvline(27, color='red', linestyle='--', label='tiny threshold (27)')
    axes[0].legend(fontsize=8)

    axes[1].plot(sizes_sorted, cum_weight)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('instance size (voxels)')
    axes[1].set_ylabel('cumulative sampling probability')
    axes[1].set_title('cumulative paste-selection probability vs size')
    axes[1].axhline(mass_at_1, color='red', linestyle='--', label=f'{mass_at_1:.1%} mass at size==1')
    axes[1].axvline(1, color='red', linestyle=':')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    bins = [1, 2, 5, 10, 27, 100, 500, sizes.max() + 1]
    labels = ['=1', '2-4', '5-9', '10-26', '27-99', '100-499', '500+']
    mass = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ((sizes >= lo) & (sizes < hi)).astype(float) * weights
        mass.append(m.sum())
    axes[2].bar(labels, mass, color=['crimson' if i == 0 else 'steelblue' for i in range(len(labels))])
    axes[2].set_ylabel('total sampling probability')
    axes[2].set_title('probability mass by size bucket')
    axes[2].tick_params(axis='x', rotation=30)
    for i, v in enumerate(mass):
        axes[2].text(i, v + 0.01, f"{v:.1%}", ha='center', fontsize=8)

    fig.suptitle('compute_instance_weights: sampling mass collapses onto near-empty instances', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches='tight')
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
