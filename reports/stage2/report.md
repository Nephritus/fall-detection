# Second-stage confirmation

Seeds: 3. Leave-one-fall-dataset-out (erciyes, umafall, upfall, weda); stage 1 = accel-only augmented CNN trained with Capture-24 negatives (as `reports/negatives`), stage 2 = gradient boosting on context features (1 s before to 4 s after a candidate impact, |a| ≥ 2 g, so **alarms are delayed by 4 s**). Free-living cost is measured on the 48 held-out Capture-24 people (1226 h) that no model trained on; alarm events merge with a 30 s refractory period. Sensitivity is **per fall recording** (simulated falls; not comparable with the per-window figures of earlier reports). Recordings without the full context are excluded from every sensitivity; recordings whose impact is below the trigger count as misses.

## 1. Coverage

| held-out dataset | fall recordings | with full context | impact ≥ trigger | stage-2 training positives / negatives (per seed) | stage-2 validation AUROC |
|---|---|---|---|---|---|
| erciyes | 1834 | 1833 | 1820 | 599 / 41396 | 0.993 |
| umafall | 201 | 200 | 200 | 1922 / 41408 | 0.988 |
| upfall | 180 | 173 | 171 | 1909 / 41417 | 0.990 |
| weda | 350 | 334 | 333 | 1776 / 41387 | 0.990 |

## 2. A practical, non-oracle operating point

Stage 1 at its validation-slice threshold (95% of validation fall windows). Stage 2 threshold keeps 98% of validation fall candidates (a held-out slice of training subjects; no held-out data used). `stage 1 alone` counts fired windows as before; `stage 1 + trigger` only alarms on candidates (|a| peak ≥ 2 g); `two-stage` also needs stage 2. Macro over held-out datasets, mean ± std over seeds.

| system | alarm events / 24 h | fall-recording sensitivity |
|---|---|---|
| stage 1 alone | 35.4 ± 14.4 | 0.893 ± 0.031 |
| stage 1 + trigger | 33.1 ± 12.4 | 0.889 ± 0.031 |
| two-stage | 16.9 ± 6.7 | 0.845 ± 0.040 |

Per held-out dataset (mean over seeds):

| held-out | stage 1 alone: alarms/day / sens | two-stage: alarms/day / sens |
|---|---|---|
| erciyes | 36.6 / 0.901 | 14.0 / 0.781 |
| umafall | 32.3 / 0.928 | 20.7 / 0.927 |
| upfall | 34.8 / 0.869 | 17.1 / 0.830 |
| weda | 37.8 / 0.873 | 16.0 / 0.842 |

## 3. Operating curves (threshold-free; thresholds chosen on the held-out data)

Lowest free-living alarms/day that reaches a per-recording sensitivity. `stage 1` = window CNN alone, `stage 2` = trigger + stage 2 alone, `both` = best (t1, t2) pair. `both` searches a two-dimensional grid, so it has more freedom than the one-dimensional curves: read it as an upper bound on what the combination can do, and use section 2 for a non-oracle figure.

| sensitivity | stage 1 | stage 2 | both | `both` vs stage 1 |
|---|---|---|---|---|
| 0.80 | 17.8 | 20.6 | 8.5 | 0.48× |
| 0.90 | 35.8 | 52.0 | 21.9 | 0.61× |
| 0.95 | 77.7 | 146.8 | 49.8 | 0.64× |

Sensitivity reachable within an alarm budget:

| alarms/day budget | stage 1 | stage 2 | both |
|---|---|---|---|
| 10 | 0.286 | 0.739 | 0.829 |
| 25 | 0.791 | 0.840 | 0.909 |
| 50 | 0.924 | 0.900 | 0.950 |
| 100 | 0.962 | 0.950 | 0.980 |

Per held-out dataset: lowest alarms/day at sensitivity 0.90 (mean over seeds):

| held-out | stage 1 | stage 2 | both |
|---|---|---|---|
| erciyes | 33.0 | 110.4 | 31.8 |
| umafall | 20.5 | 12.9 | 7.3 |
| upfall | 45.4 | 48.5 | 30.8 |
| weda | 44.4 | 36.2 | 17.8 |

## 4. Where the remaining alarms come from (practical rule)

Events per hour of each Capture-24 activity, pooled over the held-out datasets and seeds (the 48 held-out people are scored once per fold and seed; an alarm event belongs to the activity of the candidate that opened it).

| activity | hours | stage 1 + trigger | two-stage |
|---|---|---|---|
| unlabelled | 417 | 1.71 | 0.82 |
| household-chores | 61 | 2.91 | 1.80 |
| walking | 59 | 3.43 | 1.75 |
| sitting | 271 | 0.64 | 0.35 |
| mixed-activity | 44 | 2.48 | 1.28 |
| sports | 7 | 16.48 | 6.83 |
| standing | 25 | 2.43 | 1.53 |
| manual-work | 14 | 4.23 | 2.10 |
| sleep | 292 | 0.10 | 0.06 |
| vehicle | 28 | 1.17 | 0.59 |
| bicycling | 7 | 2.29 | 1.37 |

Stage 2 alone, held-out falls vs held-out Capture-24 candidates: AUROC 0.981 (mean over folds and seeds).
