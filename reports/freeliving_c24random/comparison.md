# Final models with and without Capture-24 negatives

Seeds: 3 (same seed numbers, same fall-data splits). Both models are scored on the same 48 held-out Capture-24 people (P101-P151, never trained on), 1226 h. Alarm events merge fired windows with a 30 s refractory period. 'Fall sensitivity' is the share of out-of-dataset fall windows (nested models trained on three of the four fall datasets, each dataset weighted equally, plus the Capture-24 negatives for the c24 models) at the rule's threshold; the final model itself is trained on all four, so this is an approximation. Fall windows are simulated.

## 1. Threshold rules: free-living alarms against fall sensitivity

| model | rule | alarm events / 24 h (95% CI over people) | median person | p90 person | fall-window sensitivity at this threshold |
|---|---|---|---|---|---|
| without Capture-24 negatives | val-slice | 74.6 (62.9-86.6) | 67.2 | 138.2 | 0.787 ± 0.099 |
| without Capture-24 negatives | lodo-pooled | 212.4 (186.4-239.4) | 201.5 | 331.9 | 0.950 ± 0.000 |
| with Capture-24 negatives (c24-random) | val-slice | 35.1 (29.2-41.2) | 30.2 | 68.6 | 0.851 ± 0.030 |
| with Capture-24 negatives (c24-random) | lodo-pooled | 112.5 (97.5-128.4) | 102.3 | 192.4 | 0.950 ± 0.000 |

Per-seed alarms/day: without Capt… val-slice: 49, 112, 63; without Capt… lodo-pooled: 166, 157, 314; with Capture… val-slice: 27, 53, 26; with Capture… lodo-pooled: 121, 98, 119

## 2. What each threshold costs and buys

Same grid of thresholds applied to free-living alarms/day and to out-of-dataset fall windows.

| threshold | alarms/day (without) | alarms/day (c24-random) | fall sensitivity (without) | fall sensitivity (c24-random) |
|---|---|---|---|---|
| 0.5 | 166.6 | 105.3 | 0.926 | 0.946 |
| 0.7 | 128.4 | 78.7 | 0.894 | 0.927 |
| 0.8 | 109.6 | 65.2 | 0.869 | 0.911 |
| 0.9 | 86.3 | 49.3 | 0.826 | 0.889 |
| 0.95 | 69.3 | 38.4 | 0.784 | 0.865 |
| 0.98 | 52.8 | 28.6 | 0.724 | 0.830 |
| 0.99 | 43.3 | 22.7 | 0.674 | 0.800 |
| 0.995 | 35.8 | 18.1 | 0.627 | 0.764 |
| 0.999 | 23.0 | 11.4 | 0.511 | 0.669 |

The two models' scores are calibrated differently (extra negatives shift probabilities down), so compare at equal *alarms/day* or equal *sensitivity*, not at equal threshold.

## 3. Where the remaining alarms come from (val-slice threshold)

| activity | hours | events/h (without) | events/h (c24-random) |
|---|---|---|---|
| unlabelled | 417 | 3.70 | 1.81 |
| walking | 59 | 8.76 | 3.63 |
| household-chores | 61 | 7.64 | 3.19 |
| sitting | 271 | 1.30 | 0.65 |
| mixed-activity | 44 | 6.49 | 2.56 |
| sports | 7 | 28.20 | 19.22 |
| manual-work | 14 | 13.03 | 4.66 |
| standing | 25 | 4.70 | 2.61 |
| vehicle | 28 | 2.30 | 1.09 |
| sleep | 292 | 0.14 | 0.10 |
| bicycling | 7 | 4.48 | 1.89 |
