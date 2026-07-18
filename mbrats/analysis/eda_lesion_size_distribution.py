"""
Detailed EDA of the lesion library's size distribution — specifically how
many instances are just a handful of voxels, whether they're genuine tiny
lesions or likely segmentation noise (by label composition), and whether
they're spread across many cases or concentrated in a few.

Usage:
    python -m mbrats.analysis.eda_lesion_size_distribution
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle

LIBRARY_PATH = Path("nnunet_preprocessed/Dataset001_BraTSMETS/lesion_library.pkl")
OUT_PATH = Path("eda/lesion_size_distribution.png")


def instance_size(inst):
    return inst['n_voxels_netc'] + inst['n_voxels_et'] + inst['n_voxels_rc'] + inst['n_voxels_snfh']


def main():
    library = load_pickle(str(LIBRARY_PATH))
    sizes = np.array([instance_size(inst) for inst in library])
    n = len(sizes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # exact counts for size 1-30
    ax = axes[0, 0]
    small_sizes = np.arange(1, 31)
    counts = [(sizes == s).sum() for s in small_sizes]
    ax.bar(small_sizes, counts, color='steelblue')
    ax.set_xlabel('exact instance size (voxels)')
    ax.set_ylabel('count')
    ax.set_title('exact instance counts, size 1-30')
    ax.axvline(27.5, color='red', linestyle='--', alpha=0.6)

    # cumulative fraction of library by size threshold
    ax = axes[0, 1]
    thresholds = np.arange(1, 101)
    frac = [(sizes <= t).sum() / n for t in thresholds]
    ax.plot(thresholds, frac)
    ax.set_xlabel('size threshold (voxels)')
    ax.set_ylabel('fraction of library <= threshold')
    ax.set_title('cumulative fraction of instances by size')
    ax.axvline(27, color='red', linestyle='--', label='tiny threshold (27)')
    for t in [1, 5, 10, 27]:
        f = (sizes <= t).sum() / n
        ax.annotate(f"{f:.1%}", (t, f), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # label composition of size==1 instances
    ax = axes[1, 0]
    size1 = [inst for inst, s in zip(library, sizes) if s == 1]
    comp = Counter()
    for inst in size1:
        labels = [name for name, key in [('NETC', 'n_voxels_netc'), ('ET', 'n_voxels_et'), ('RC', 'n_voxels_rc')]
                  if inst[key] > 0]
        comp['+'.join(labels)] += 1
    items = sorted(comp.items(), key=lambda kv: -kv[1])
    ax.bar([k for k, v in items], [v for k, v in items],
           color=['gray' if k == 'ET' else 'crimson' for k, v in items])
    ax.set_title(f'label composition of {len(size1)} single-voxel instances\n'
                 f'(non-ET-only cases are anatomically implausible in isolation)')
    ax.set_ylabel('count')
    for i, (k, v) in enumerate(items):
        ax.text(i, v + 3, str(v), ha='center', fontsize=8)

    # per-case concentration of size==1 instances
    ax = axes[1, 1]
    case_counts = Counter(inst['case_id'] for inst in size1)
    counts_sorted = sorted(case_counts.values(), reverse=True)
    ax.bar(range(len(counts_sorted)), counts_sorted, color='steelblue')
    ax.set_xlabel(f'case rank (of {len(case_counts)} cases contributing size==1 instances)')
    ax.set_ylabel('count of size==1 instances from that case')
    top5_frac = sum(counts_sorted[:5]) / len(size1)
    ax.set_title(f'per-case concentration of single-voxel instances\n'
                 f'(top 5 cases = {top5_frac:.0%} of all {len(size1)} single-voxel instances)')

    fig.suptitle(f'Lesion library size distribution ({n} instances total)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130, bbox_inches='tight')
    print(f"saved {OUT_PATH}")

    print(f"\ntotal instances: {n}")
    for lo, hi in [(1, 1), (1, 3), (1, 5), (1, 10), (1, 27)]:
        c = ((sizes >= lo) & (sizes <= hi)).sum()
        print(f"  size in [{lo},{hi}]: {c} ({c/n:.1%})")
    print(f"\nsize==1 composition: {dict(comp)}")
    print(f"size==1 spread across {len(case_counts)} cases, top 5 cases = {top5_frac:.0%} of them")


if __name__ == "__main__":
    main()
