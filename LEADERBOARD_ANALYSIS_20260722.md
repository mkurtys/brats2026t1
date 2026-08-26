# BraTS 2026 Task 1 — Leaderboard Analysis & Recommendations

_Analysis date: 2026-07-22. Based on the first val-leaderboard result (`leaderboard.csv`),
the verified official scoring code, the fold-0 CC-filter tuning records, and the blob-loss
trainer's loss + sampler. Supersedes the pre-leaderboard `RECOMMENDATIONS.md` (2026-07-06)
where they conflict._

## Where we stand

We are team **`mwro`** (fullres blob-loss fold-all + CC-filter `{et:8,tc:8,wt:5,rc:300}`).
Mean lesion-wise DSC **0.607** vs top (Junho) **0.712**; 2nd (MicroBT) 0.702.

| Region | Us DSC / NSD | Junho DSC / NSD | DSC gap | share of 0.105 mean gap |
|--------|--------------|-----------------|---------|-------------------------|
| RC | 0.410 / 0.284 | 0.601 / 0.480 | **0.191** | ~45% |
| WT | 0.642 / 0.661 | 0.722 / 0.738 | 0.080 | ~19% |
| ET | 0.674 / 0.737 | 0.752 / 0.817 | 0.078 | ~19% |
| TC | 0.700 / 0.747 | 0.772 / 0.823 | 0.072 | ~17% |

## Main conclusions

1. **The ET/TC/WT gap is not a boundary problem.** Our DSC↔NSD offset moves in lockstep
   with Junho's on all three regions, so it is *not* surface/boundary quality (highres won't
   specifically help). It's a flat ~0.07 deficit = **capacity/ensemble + false positives**.

2. **We are false-positive heavy.** We emit ~2.5–3× more total FP/case than the leaders
   (ET 1.25 vs 0.45, etc.) while small-lesion **TP is on par** — a precision problem, not recall.
   The leaders beat us on **both** detection F1 (ET small-F1 0.512 vs our 0.422) **and** FP count,
   i.e. our recall bias is *not* buying a detection edge — it is mostly just costing DSC via FP.

3. **FP suppression is a real ranked-DSC lever — VERIFIED at the source**
   (`config_mets.yaml` + `metrics_parser.py::parse_mets_results`):
   - Panoptica applies **no min-size filter** (`ConnectedComponentsInstanceApproximator`,
     `matching_threshold 1e-6`). Every FP of **any size** adds a `0` term to the lesion-wise DSC.
   - **But** a region's DSC is NaN-gated to cases with ≥1 large (≥27 mm³) GT lesion of that region
     (`metrics_parser.py:187`). So an FP only hurts DSC when it **co-occurs with a genuine large
     lesion** of the same class; FPs on clean/small-only cases are invisible to DSC (F1 only).
     → the FP lever pays off exactly where ensemble/TTA are strongest.
   - (The leaderboard `small_instance_fp` and `large_instance_fp` columns are both just the total
     `num_fp`, not size-split — so "our small FP 1.25 vs 0.45" is really *total* FP/case.)

4. **RC is ~half the total gap and is a modeling deficit**, not something postproc can close.
   RC NSD (0.284) ≪ DSC (0.410) → cavity boundaries genuinely bad; RC is rare in train (~13%).

5. **The model is deliberately positive/recall-biased** (this is by design, per the "buy recall in
   training, remove FP in postproc" strategy), across three mechanisms:
   - **Loss:** Tversky α=0.3/β=0.7 (region + per-instance blob term) penalizes a missed voxel 2.33×
     a false one; focal (γ=2) mildly recall-tilts; the per-instance blob term only scores GT instances
     so gives weak FP penalty. BCE (background term) is the lone precision counterweight.
   - **Sampler (bigger lever), measured over all 1296 fold-all cases:**
     - fg-forced patch rate is **identical** (`oversample_foreground_percent=0.33`) — bias is in the
       *composition*, not the rate.
     - **100% of the fg budget centers on tumor-core/RC (vs 64% standard); 0% on edema (vs 34%).**
     - **RC-containing cases drawn 2.62×** more than natural (inverse-freq case weighting); NETC 1.17×.
     - sqrt-size instance weighting boosts small-instance center-share 1.7× alone (6.1%→10.5%), netted
       to 7.0% after RC-case weighting pulls in large cavities.
   - **fg-voxel fraction per patch** (measured on real segs): modified is **higher** than standard
     on every metric (overall 1.24% vs 1.07%, +16% rel) — tumor-core centers grab mass + surrounding
     edema, and inv-freq draws fuller cases. So the voxel level reinforces the bias, modestly. Note
     both are tiny — even fg-forced patches are ~98.4% background.

## Recommendations (ranked)

**0. Lower the RC CC-filter `300 → 150` (DONE in config, not yet run) — near-free.**
   RC-DSC is tuned in-sample on just 36 cases. The sweep shows `rc:300` sits on a cliff: the smallest
   matched true-cavity prediction is **339 voxels — only 13% above 300**, so any genuine val cavity
   predicted at 250–330 vox gets deleted (→ a large-lesion 0-DSC miss). This matches the CV→val RC drop
   (val 0.410 vs CV ~0.44). `rc:150` keeps 79% of the in-sample gain with a 2.3× TP margin. The k300→150
   difference is only +0.024 DSC from 2 FP on 36 cases — noise-sized, high-risk. Files already updated:
   `postprocessing_out/cc_thresholds_fold0_dscf1.json`, `logs/blobloss_foldall_submission.sh`.
   ET/TC/WT {8,8,5} unchanged.

**1. 5-fold ensemble (folds 1–4) + same filter — top move, ~30h, reliable.**
   We submitted a *single* fold-all model. Ensembling attacks **both** remaining problems at once: it
   lifts segmentation quality uniformly (the capacity gap) **and** suppresses fold-idiosyncratic FP
   (the precision gap), which is exactly the FP that co-occurs with real lesions where DSC counts.

**2. RC-focused training — biggest single modeling gap.**
   Oversample / copy-paste RC cavities; RC is 45% of the gap and postproc can't recover 0.19 DSC.
   (Note: the sampler already draws RC cases 2.6× and centers on them — the remaining RC deficit is
   recall + overlap + surface, i.e. genuine model capacity on a rare class, not exposure.)

**3. Cheap precision pre-checks before committing 30h to the ensemble:**
   - **TTA** on the single model — should cut the FP rate directly; free-ish, no retrain.
   - **Inference decision-threshold / prior-shift sweep** — since the model is positive-biased, raising
     the foreground bar post-hoc directly "undoes" the bias; sweep on fold-0 the same analytic way as
     the RC filter. Tests empirically whether the recall bias is over-tuned before any α=0.5 retrain.

**4. ResEnc-L / longer training — uniform-gap capacity lever, expensive; lowest priority.**

## Provenance / reproduce

- Scoring facts: `brats_evaluation/{configs/config_mets.yaml,metrics_parser.py}` (pip pkg).
- RC sweep + matched-TP sizes: `mbrats/postprocessing/tune_cc_filter.py` on
  `postprocessing_out/records_blobloss_fold0.pkl`.
- Sampler exposure + fg-voxel fraction: scratch scripts `sampler_compare.py`, `fg_voxel_fraction.py`
  over `nnunet_preprocessed/Dataset001_BraTSMETS/nnUNetPlans_3d_fullres/*` (class_locations,
  tc_rc_instances, `_seg.b2nd`).
- Memory: `leaderboard-first-result`, `scoring-and-fp-diagnosis`, `postprocessing-cc-filter-status`,
  `sampler-foreground-exposure`.
