# Free-living false alarms (Capture-24)

Seeds: 5. 140 of 151 participants used (3603 h of wrist accelerometer, about a day each, including sleep); each 4 s window (2 s stride) is scored by the final augmented accel-only CNN trained on all four fall datasets. Alarm events merge fired windows with a 30 s refractory period. **Rates are alarm events, not windows. Nothing here is assumed: the exposure is measured.** Rule `val-slice` = 95% sensitivity on a validation slice of the training data; `lodo-pooled` = nested leave-one-dataset-out (see `threshold.py`). The threshold belongs to one operating point each; sensitivity on real falls is unknown (the fall datasets are simulated, and their leave-one-dataset-out detection was 0.89 and 0.96 for these rules).

**Limits.** (1) The people are 151 adults in Oxford wearing an Axivity AX3 on the *dominant* wrist, not a smartwatch; the side is not recorded, so I assumed right and mirrored it, and the mounting is aligned per person from walking (direction only, norm ~0.7 g because of arm swing). (2) Only about 66% of the hours are labelled; the rest is shown as `unlabelled`. (3) A fall or near-fall that really happened would count as a false alarm here, since nothing marks it. (4) The AX3 samples at 100 Hz and was low-passed and decimated to 25 Hz by the same code path as the real datasets. (5) 151 people × one day: day-to-day variation within a person is not measured.

Excluded participants: 11 (P018: walking mean norm 0.47 g; P019: walking mean norm 0.43 g; P025: walking mean norm 0.37 g; P042: walking 140 s; P056: walking mean norm 0.39 g; P061: walking mean norm 0.37 g; P075: walking mean norm 0.16 g; P091: walking mean norm 0.38 g; P112: walking mean norm 0.20 g; P135: walking mean norm 0.25 g; P136: walking mean norm 0.22 g).

## 1. Alarm events per day of wear

Events per 24 h of wear, pooled over all participants and hours (labelled and unlabelled). Interval: 95% bootstrap over participants (seed-averaged). `per-participant` is the distribution over participants of their own events/24 h.

| rule | events / 24 h (95% CI) | median participant | p90 participant | participants with ≥1 alarm per day | participants with 0 alarms |
|---|---|---|---|---|---|
| val-slice | 70.9 (63.3–78.8) | 58.3 | 142.1 | 100% | 0% |
| lodo-pooled | 193.2 (176.8–210.2) | 174.6 | 334.3 | 100% | 0% |

## 2. Where the alarms come from

Rate per hour of each labelled activity (pooled over participants, seed-averaged) and its share of all alarm events.

### Rule: val-slice

| activity | hours | share of wear time | events/h | share of alarm events | windows/h (before merging) |
|---|---|---|---|---|---|
| unlabelled | 1212 | 34% | 3.75 | 43% | 8.16 |
| household-chores | 169 | 5% | 9.48 | 15% | 19.95 |
| walking | 161 | 4% | 7.97 | 12% | 16.03 |
| sitting | 850 | 24% | 1.21 | 10% | 2.18 |
| mixed-activity | 102 | 3% | 5.10 | 5% | 9.95 |
| manual-work | 28 | 1% | 16.33 | 4% | 47.27 |
| standing | 78 | 2% | 5.09 | 4% | 10.23 |
| sports | 13 | 0% | 24.40 | 3% | 106.18 |
| vehicle | 91 | 3% | 2.73 | 2% | 4.67 |
| sleep | 876 | 24% | 0.18 | 1% | 0.27 |
| bicycling | 23 | 1% | 4.25 | 1% | 7.91 |

### Rule: lodo-pooled

| activity | hours | share of wear time | events/h | share of alarm events | windows/h (before merging) |
|---|---|---|---|---|---|
| unlabelled | 1212 | 34% | 10.02 | 42% | 28.12 |
| household-chores | 169 | 5% | 23.59 | 14% | 65.85 |
| walking | 161 | 4% | 21.92 | 12% | 60.62 |
| sitting | 850 | 24% | 4.11 | 12% | 8.83 |
| mixed-activity | 102 | 3% | 15.85 | 6% | 45.52 |
| standing | 78 | 2% | 13.61 | 4% | 33.64 |
| manual-work | 28 | 1% | 35.47 | 3% | 122.85 |
| vehicle | 91 | 3% | 8.39 | 3% | 19.51 |
| sleep | 876 | 24% | 0.74 | 2% | 1.35 |
| sports | 13 | 0% | 38.63 | 2% | 195.74 |
| bicycling | 23 | 1% | 12.00 | 1% | 26.03 |

## 3. Day versus night

| rule | awake labelled hours: events/h | sleep: events/h | unlabelled: events/h |
|---|---|---|---|
| val-slice | 3.93 | 0.177 | 3.75 |
| lodo-pooled | 10.71 | 0.735 | 10.02 |

## 4. How sensitive is this to my mounting assumptions?

The same participants (about 40 of them) scored again after rotating the aligned signal about the forearm axis, or treating the wrist as the left one. If the false-alarm rate barely moves, the numbers do not hinge on the alignment guess; roll offsets the model never saw during training (augmentation covers ±30°) would raise it.

| mounting | val-slice: events/24 h | lodo-pooled: events/24 h |
|---|---|---|
| as assumed | 70.1 | 185.9 |
| roll +30° | 72.7 | 201.8 |
| roll -30° | 72.9 | 187.8 |
| roll +60° | 79.8 | 231.2 |
| roll -60° | 80.7 | 206.8 |
| roll 90° | 88.8 | 263.9 |
| roll 180° | 99.4 | 290.2 |
| left wrist (no mirror) | 75.1 | 200.5 |

## 5. What each threshold costs and buys

Same threshold applied to two things: alarm events per 24 h of Capture-24 wear (final model, all participants) and the share of **out-of-dataset fall windows** it detects (nested models trained on three of the four fall datasets, each dataset weighted equally). The two models differ, so read this as an approximate trade-off curve, not an exact operating point. Fall windows are simulated falls, so real sensitivity is lower. Per-window sensitivity is not per-fall: several windows can contain one fall, so per-fall detection is higher.

| threshold | alarm events / 24 h | fall-window sensitivity |
|---|---|---|
| 0.5 | 166.8 ± 53.3 | 0.937 ± 0.016 |
| 0.7 | 127.4 ± 45.0 | 0.908 ± 0.020 |
| 0.8 | 107.5 ± 40.2 | 0.887 ± 0.025 |
| 0.9 | 83.5 ± 33.7 | 0.847 ± 0.032 |
| 0.95 | 66.3 ± 28.4 | 0.807 ± 0.036 |
| 0.98 | 50.1 ± 23.0 | 0.750 ± 0.042 |
| 0.99 | 40.9 ± 19.5 | 0.699 ± 0.044 |
| 0.995 | 33.6 ± 16.6 | 0.652 ± 0.049 |
| 0.999 | 21.7 ± 11.1 | 0.540 ± 0.065 |

## 6. Comparison with the assumed-budget estimate (`alarms.py`)

`reports/alarms/report.md` estimated 26 (val-slice) and 155 (lodo-pooled) alarms for a typical day, and 55 / 210 for an active day, from lab clips and assumed time budgets. The measured rates above replace that assumption.

Weights: threshold levels per seed are in `data/processed/capture24/thresholds.csv`. Runtime of the last report step: 8.8 min (fits and scoring are cached).
