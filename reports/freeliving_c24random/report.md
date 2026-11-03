# Free-living false alarms (Capture-24) with Capture-24 negatives in training

Seeds: 3. 48 of 51 participants used (1226 h of wrist accelerometer, about a day each, including sleep); each 4 s window (2 s stride) is scored by the final augmented accel-only CNN trained on all four fall datasets plus random Capture-24 windows of people P001-P100 as negatives (only the held-out people P101-P151 are scored here, so no one in this report was trained on). Alarm events merge fired windows with a 30 s refractory period. **Rates are alarm events, not windows. Nothing here is assumed: the exposure is measured.** Rule `val-slice` = 95% sensitivity on a validation slice of the training data; `lodo-pooled` = nested leave-one-dataset-out (see `threshold.py`). The threshold belongs to one operating point each; sensitivity on real falls is unknown (the fall datasets are simulated; see section 5 and `reports/threshold_augmented/report.md` for fall sensitivity of each rule).

**Limits.** (1) The people are 151 adults in Oxford wearing an Axivity AX3 on the *dominant* wrist, not a smartwatch; the side is not recorded, so I assumed right and mirrored it, and the mounting is aligned per person from walking (direction only, norm ~0.7 g because of arm swing). (2) Only about 66% of the hours are labelled; the rest is shown as `unlabelled`. (3) A fall or near-fall that really happened would count as a false alarm here, since nothing marks it. (4) The AX3 samples at 100 Hz and was low-passed with the same anti-alias filter as the real datasets and decimated to 25 Hz. (5) 151 people × one day: day-to-day variation within a person is not measured.

Excluded participants: 3 (P112: walking mean norm 0.20 g; P135: walking mean norm 0.25 g; P136: walking mean norm 0.22 g).

## 1. Alarm events per day of wear

Events per 24 h of wear, pooled over all participants and hours (labelled and unlabelled). Interval: 95% bootstrap over participants (seed-averaged). `per-participant` is the distribution over participants of their own events/24 h.

| rule | events / 24 h (95% CI) | median participant | p90 participant | participants with ≥1 alarm per day | participants with 0 alarms |
|---|---|---|---|---|---|
| val-slice | 35.1 (29.0–41.4) | 30.2 | 68.6 | 100% | 0% |
| lodo-pooled | 112.5 (97.4–128.2) | 102.3 | 192.4 | 100% | 0% |

## 2. Where the alarms come from

Rate per hour of each labelled activity (pooled over participants, seed-averaged) and its share of all alarm events.

### Rule: val-slice

| activity | hours | share of wear time | events/h | share of alarm events | windows/h (before merging) |
|---|---|---|---|---|---|
| unlabelled | 417 | 34% | 1.81 | 42% | 3.67 |
| walking | 59 | 5% | 3.63 | 12% | 6.02 |
| household-chores | 61 | 5% | 3.19 | 11% | 5.59 |
| sitting | 271 | 22% | 0.65 | 10% | 1.03 |
| sports | 7 | 1% | 19.22 | 8% | 51.88 |
| mixed-activity | 44 | 4% | 2.56 | 6% | 4.69 |
| manual-work | 14 | 1% | 4.66 | 4% | 6.96 |
| standing | 25 | 2% | 2.61 | 4% | 4.18 |
| vehicle | 28 | 2% | 1.09 | 2% | 1.61 |
| sleep | 292 | 24% | 0.10 | 2% | 0.13 |
| bicycling | 7 | 1% | 1.89 | 1% | 3.23 |

### Rule: lodo-pooled

| activity | hours | share of wear time | events/h | share of alarm events | windows/h (before merging) |
|---|---|---|---|---|---|
| unlabelled | 417 | 34% | 5.67 | 41% | 12.98 |
| sitting | 271 | 22% | 2.77 | 13% | 4.76 |
| walking | 59 | 5% | 11.41 | 12% | 22.03 |
| household-chores | 61 | 5% | 9.32 | 10% | 17.95 |
| mixed-activity | 44 | 4% | 8.33 | 6% | 16.58 |
| sports | 7 | 1% | 39.78 | 5% | 144.24 |
| manual-work | 14 | 1% | 16.72 | 4% | 31.56 |
| standing | 25 | 2% | 8.09 | 4% | 16.18 |
| sleep | 292 | 24% | 0.47 | 2% | 0.69 |
| vehicle | 28 | 2% | 4.06 | 2% | 7.34 |
| bicycling | 7 | 1% | 6.22 | 1% | 11.04 |

## 3. Day versus night

| rule | awake labelled hours: events/h | sleep: events/h | unlabelled: events/h |
|---|---|---|---|
| val-slice | 1.96 | 0.098 | 1.81 |
| lodo-pooled | 6.28 | 0.467 | 5.67 |

## 4. How sensitive is this to my mounting assumptions?

The same participants (about 40 of them) scored again after rotating the aligned signal about the forearm axis, or treating the wrist as the left one. If the false-alarm rate barely moves, the numbers do not hinge on the alignment guess; roll offsets the model never saw during training (augmentation covers ±30°) would raise it.

| mounting | val-slice: events/24 h | lodo-pooled: events/24 h |
|---|---|---|
| as assumed | 35.1 | 112.5 |
| roll +30° | 36.0 | 117.8 |
| roll -30° | 35.1 | 110.9 |
| roll +60° | 37.9 | 124.2 |
| roll -60° | 36.5 | 114.7 |
| roll 90° | 39.4 | 131.1 |
| roll 180° | 39.4 | 133.7 |
| left wrist (no mirror) | 34.4 | 113.3 |

## 5. What each threshold costs and buys

Same threshold applied to two things: alarm events per 24 h of Capture-24 wear (final model, the scored participants) and the share of **out-of-dataset fall windows** it detects (nested models trained on three of the four fall datasets, each dataset weighted equally). The two models differ, so read this as an approximate trade-off curve, not an exact operating point. Fall windows are simulated falls, so real sensitivity is lower. Per-window sensitivity is not per-fall: several windows can contain one fall, so per-fall detection is higher.

| threshold | alarm events / 24 h | fall-window sensitivity |
|---|---|---|
| 0.5 | 105.3 ± 4.6 | 0.946 ± 0.004 |
| 0.7 | 78.7 ± 1.6 | 0.927 ± 0.003 |
| 0.8 | 65.2 ± 0.4 | 0.911 ± 0.002 |
| 0.9 | 49.3 ± 0.7 | 0.889 ± 0.004 |
| 0.95 | 38.4 ± 1.5 | 0.865 ± 0.006 |
| 0.98 | 28.6 ± 1.3 | 0.830 ± 0.011 |
| 0.99 | 22.7 ± 1.8 | 0.800 ± 0.016 |
| 0.995 | 18.1 ± 1.7 | 0.764 ± 0.023 |
| 0.999 | 11.4 ± 1.0 | 0.669 ± 0.032 |

## 6. Comparison with the assumed-budget estimate

`reports/alarms/report.md` estimated alarms per day from lab clips and assumed activity time budgets (`budget_estimate.py`). The measured rates above replace that assumption.

Threshold levels per seed are in `data/processed/capture24/c24random/thresholds.csv`. Runtime of the last report step: 0.0 min (fits and scoring are cached).
