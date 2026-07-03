# Hierarchical Tumor Regions — BraTS MET

## Labels (per-voxel, challenge spec)

| Label | Name | Description |
|---|---|---|
| 1 | NETC | Necrotic Tumor Core — the non-enhancing, dead center of the tumor. Appears dark on T1c (no contrast uptake). Enclosed by enhancing tumor. |
| 2 | SNFH | Surrounding Non-enhancing FLAIR Hyperintensity — peritumoral edema and infiltrated tissue. Visible as bright signal on T2-FLAIR beyond the tumor proper. |
| 3 | ET | Enhancing Tumor — the actively growing tumor rim with contrast uptake. Bright on T1c post-contrast. The primary surgical and radiosurgery target. |
| 4 | RC | Resection Cavity — hollow region left after surgical resection. Only present in post-treatment cases. |

---

## Hierarchical Regions (combinations of labels)

Regions are derived by merging labels. They capture clinically meaningful composite structures and are the standard way BraTS reports summary scores across all challenges.

### ET — Enhancing Tumor
**Labels: {3}**

The enhancing rim only. Identical to the ET label. Included as a region for consistency with the TC/WT reporting convention.

- Clinical relevance: primary radiosurgery target; its volume drives treatment planning (single vs. multi-fraction SRS).
- Segmentation difficulty: easiest — high T1c contrast makes it visible regardless of lesion size.

### TC — Tumor Core
**Labels: {1, 3} = NETC + ET**

The solid tumor mass, excluding surrounding edema. Represents everything that would be excised in surgery.

- Clinical relevance: total resectable/treatable volume. Necrosis + enhancing rim together define the "bulk" of the metastasis.
- Why TC > ET in Dice: merging NETC into ET fills in the necrotic interior, making the combined mask more compact and easier to match even when NETC boundaries are imprecise. A model that slightly misplaces the NETC/ET boundary internally still gets a good TC score.

### WT — Whole Tumor
**Labels: {1, 2, 3} = NETC + SNFH + ET**

The full extent of tumor-related abnormality, including peritumoral edema. RC (label 4) is excluded — it is a post-treatment void, not tumor tissue.

- Clinical relevance: radiation planning target volume (PTV margin). SNFH/edema is often included in whole-brain radiotherapy planning.
- Why WT ≈ TC in Dice: SNFH is a large diffuse region that envelops TC. Adding it makes the union mask bigger, but since SNFH is well-predicted (Dice 0.77) the gain in true positives roughly offsets the increase in denominator.

---

## Relationship Between Regions

```
┌─────────────────────────────────┐
│  SNFH (edema)                   │
│  ┌───────────────────────────┐  │
│  │  ET (enhancing rim)       │  │
│  │  ┌─────────────────────┐  │  │
│  │  │  NETC (necrosis)    │  │  │
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘

ET   = {3}         — enhancing rim only
TC   = {1,3}       = NETC + ET       (tumor core)
WT   = {1,2,3}     = TC + SNFH       (whole tumor, excl. RC)
```

RC sits outside this hierarchy — it is spatially where TC used to be before surgery, and is scored independently.

---

## Baseline Results (fold 0 CV, 260 cases)

### Per-label

| Label | DSC | NSD (2mm) | n |
|---|---|---|---|
| NETC | 0.581 | 0.714 | 118 |
| SNFH | 0.771 | 0.887 | 220 |
| ET | 0.760 | 0.866 | 237 |
| RC | 0.480 | 0.572 | 36 |
| Mean | 0.648 | 0.760 | |

### Per-region (Dice only)

| Region | Dice | n |
|---|---|---|
| ET | 0.760 | 237 |
| TC | 0.783 | 237 |
| WT | 0.782 | 243 |

TC > ET (+0.023): necrosis fills in the core, improving combined mask overlap even when NETC boundaries are imprecise.

WT ≈ TC (+0.001 only): SNFH is already well-predicted and large, so adding it barely moves the combined score.
