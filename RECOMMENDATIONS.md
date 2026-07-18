# BraTS 2026 Task 1 — Recommended Solutions

_Analysis date: 2026-07-06. Based on CHALLANGE.md, README.md, evaluate.py, nnU-Net fingerprint, IDEAS.md._

## The scoring reality (read this first)

There are **two leaderboards that pull in opposite directions**, and every design
choice should be traced back to which one it serves.

| | Segmentation LB | Detection LB |
|---|---|---|
| Metric | DSC + NSD (subject-wise) | Lesion-wise **F1** |
| Scope | **Only lesions > 27 mm³** | All lesions, incl. tiny (<27 mm³) |
| Match criterion | boundary quality | **DSC ≥ 0.2** overlap = "detected" |
| Rewards | clean boundaries on big lesions | **sensitivity** to small lesions, FP control |

Three consequences drive the strategy:

1. **Detection positive at DSC=0.2 is a very low bar.** You don't need to segment a
   3×3×3-voxel met well — you just need to *land on it*. Recall of small lesions is
   the entire game on the detection LB, bounded by false-positive control (precision).
2. **Segmentation ignores everything ≤27 mm³.** Aggressive small-lesion recall costs
   you *nothing* on the segmentation LB (those voxels aren't scored) but can *hurt*
   precision on the detection LB. → the optimal operating point differs per LB.
3. **"Failed" cases aren't set to worst (no 0 penalty).** Producing a bad-but-present
   mask can score worse than the case being un-scorable. Don't chase every case.

**Headline recommendation: decouple the two submissions.** Train one strong base model,
then derive *two* post-processed outputs from the same probabilities:
- **Segmentation submission:** higher confidence threshold + CC filtering → clean big lesions.
- **Detection submission:** lower threshold / TTA-smoothed probs, keep small blobs,
  tune the FP filter for F1 (not DSC).

Same weights, two operating points. This is the highest-leverage insight and is nearly free.

---

## Ranking mechanics → where marginal effort pays

Ranking = sum of ranks across averaged metrics, with permutation significance testing.
Implication: **broad, consistent gains across all 4 regions (ET/TC/WT/RC) beat a spike
in one.** RC and NETC are rare and currently starved — lifting their weak metrics moves
your *rank* more than squeezing another 0.01 DSC out of already-good WT. Prioritize the
laggards.

---

## Current state (what's already built / running)

- 6-channel input: T1, T1c, T2, FLAIR, **T1c−T1n subtraction**, **T1c/T1n ratio** — done.
- Custom trainer `nnUNetTrainerBraTS`: **focal loss + instance-uniform sampling +
  class-balanced case sampling** — done.
- Preprocessing to median spacing `[1.0, 0.898, 0.859]` mm; 1296 cases (1268 native / 328 SRI24).
- Trained so far (fold 0 only): ResEncM `96×160×160`, ResEncL `128×128×128` and `128×224×256`,
  checkpoint-250 trainer variant. **No 5-fold ensemble yet.**

---

## Recommended solutions, in priority order

### Tier 1 — do these first (high ROI, low/known effort)

1. **Finish the 5-fold ensemble + TTA (mirroring).** Currently fold 0 only. This is the
   single biggest free win, especially for small-lesion detection: more votes smooth the
   probability map near tiny mets → higher recall at fixed FP. nnU-Net gives it out of the box.
   *Verify:* fold-averaged F1 on held-out CV > best single fold.

2. **Two-operating-point post-processing (the decoupling above).** Grid-search on fold-0 CV:
   - foreground probability threshold,
   - **per-class CC size threshold** (the CC-filtering idea — already scoped),
   separately optimizing **DSC** (seg submission) and **F1** (detection submission).
   *Verify:* pick thresholds on CV, confirm each metric improves vs argmax+no-filter.

3. **Lock the evaluation loop before more training.** Wire `evaluate.py` / `panoptica`
   to report DSC, NSD **and lesion-wise F1 split by lesion size (≤27 vs >27 mm³)** on
   fold-0 CV. You cannot tune (1)–(2) or compare loss variants without this. This is the
   prerequisite for every experiment below.

### Tier 2 — targeted model improvements (medium effort)

4. **Auxiliary TC + WT region Dice losses** on top of per-label loss. Directly targets the
   scored regions (ET/TC/WT), cheap to add, expected +0.01–0.02 DSC. Keep per-label outputs
   (don't switch to region-based training — NETC absent in ~1/3 of cases makes TC≈ET unstable).

5. **Copy-paste augmentation for small ET lesions.** The small-lesion failure is a *data
   scarcity* problem (p50 lesion ≈ 27 voxels), not capacity. Pre-extract a bank of small
   instance patches, paste 0–k into each case inside the brain mask (avoid ventricles/overlap).
   This is the most direct lever on detection-LB recall. Higher effort but high ceiling.

6. **Joint (not per-channel) intensity augmentation for the derived channels.** nnU-Net
   augments brightness/contrast *per channel* by default, which corrupts the T1c−T1n and
   T1c/T1n signals (shifting T1c and T1n independently destroys the difference/ratio).
   Apply intensity aug jointly to raw modalities *before* recomputing derived channels, or
   exclude channels 4–5 from per-channel intensity aug. Low effort, prevents a silent bug.

### Tier 3 — bigger bets (do only if small-lesion F1 still lags after Tier 1–2)

7. **High-resolution / sub-mm handling.** At 1 mm a 27 mm³ met is ~27 voxels and a single
   voxel error is catastrophic. Options, cheapest first: (a) already covered by TTA+ensemble;
   (b) a **sub-mm specialist** model trained only on native sub-mm cases to re-segment small
   lesions flagged by the base model — more targeted than a full 0.5 mm retrain. Only worth
   it if detection F1 on small lesions is still the bottleneck.

8. **Registration to a common space (SRI24).** The native/SRI24 mix may hurt consistency,
   but re-registering risks interpolating away the very small lesions detection rewards, and
   contradicts the challenge's own note that native space preserves small lesions. **Low
   priority / risky** — measure per-subset CV metrics first to confirm there's even a problem.

### Explicitly deprioritize

- **Topology-aware loss** — high complexity, uncertain payoff; revisit only if F1 plateaus.
- **Deformation-field / longitudinal augmentation** — high effort, nnU-Net elastic aug likely
  covers most of it; the UCSD longitudinal idea depends on recovering patient groupings (unknown).
- **Per-class inverse-frequency loss weighting** — partly redundant now that focal loss +
  class-balanced sampling are in; test as an ablation, not a headline.

---

## Suggested execution path

```
1. Wire size-split DSC/NSD/F1 eval on fold-0 CV        → verify: metrics reproduce per LB
2. Finish 5-fold + TTA                                  → verify: ensemble > single fold
3. Grid-search two operating points (thresh + CC size)  → verify: DSC↑ and F1↑ on CV
4. Add TC+WT aux loss, retrain fold 0                    → verify: +DSC vs baseline
5. Copy-paste small-lesion aug, retrain                  → verify: small-lesion F1↑
6. (Gate) sub-mm specialist only if small-lesion F1 lags → verify: F1↑ on sub-mm subset
```

Steps 1–3 need no retraining and likely capture most of the achievable gain from what's
already trained. Everything after is a retraining experiment gated on the CV numbers.
