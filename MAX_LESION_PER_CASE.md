# Max Lesion per Case Analysis

For each training case, the largest NETC+ET+RC connected component is identified.
SNFH excluded from instance definition (same as `analyze_lesion_instances.py`).

Generated from: 1296 training cases, preprocessed blosc2 segs at 1mm³.

---

## Size Distribution of Largest Instance

| Bin | Count | % cases |
|---|---|---|
| no tumor (SNFH-only) | 29 | 2.2% |
| S 27–500mm³ | 367 | 28.3% |
| M 500–5k mm³ | 459 | 35.4% |
| L 5k–20k mm³ | 310 | 23.9% |
| XL >20k mm³ | 93 | 7.2% |

Percentiles of max instance vol (mm³):

| p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean | max |
|---|---|---|---|---|---|---|---|---|
| 61 | 257 | 1 637 | 6 907 | 16 952 | 24 330 | 60 551 | 6 006 | 125 585 |

**Median case has a 1 637mm³ dominant lesion — solidly M-bin.** Only 28% of cases are
small-tumor-dominated (max ≤500mm³).

---

## Composition of Largest Instance per Case

| Composition | Count | % cases |
|---|---|---|
| NETC+ET | 608 | 46.9% |
| ET | 518 | 40.0% |
| ET+RC | 106 | 8.2% |
| no tumor | 29 | 2.2% |
| RC | 26 | 2.0% |
| NETC+ET+RC | 8 | 0.6% |
| NETC | 1 | 0.1% |

## Composition × Size of Largest Instance

| Composition | S 27–500 | M 500–5k | L 5k–20k | XL >20k |
|---|---|---|---|---|
| NETC+ET | 74 | 251 | 206 | 76 |
| ET | 281 | 151 | 42 | 8 |
| ET+RC | 5 | 49 | 45 | 6 |
| RC | 7 | 4 | 14 | 1 |
| NETC+ET+RC | 0 | 4 | 2 | 2 |
| NETC | 0 | 0 | 1 | 0 |

---

## Key Observations

1. **NETC+ET dominates at case level (47%) vs globally (27%).** The many small satellite
   lesions are predominantly pure ET; the dominant tumor mass usually carries a necrotic
   core. The global 70% pure-ET figure is driven by small instances.

2. **Small-lesion problem is almost entirely a pure-ET problem.** 281/367 (77%) of cases
   whose largest lesion is S-bin are pure ET — small enhancing mets with no necrotic core.
   NETC is almost always paired with a large tumor (M/L/XL).

3. **Size-based case weighting would favour large-tumor cases.** Since 65% of cases have
   a dominant lesion ≥500mm³, weighting by 1/max_size would shift training signal toward
   the minority of small-tumor cases — the opposite of the case-frequency argument.
   Class-balanced weighting (upweighting RC/NETC cases) is more targeted.

4. **RC as largest instance: 26 cases (2%).** These are the hardest RC cases — no
   enhancing tumor present, just an isolated cavity. Enhancement channels (T1c−T1n)
   are the primary tool to help the model distinguish RC from CSF.
