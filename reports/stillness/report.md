# Stillness check

An alarm is raised only if the wrist stays still for a wait period after the detector's decision (stage 1: window end; two-stage: impact + 4 s). Motion = per-second standard deviation of the acceleration vector (g), so a constant calibration offset is not motion but a rotating wrist is. `strict` = every second below the level, `most` = at least 80% of seconds, `mean` = the average. Each firing detection is checked separately, then the passing ones merge into alarm events (30 s refractory period). The alert comes **wait seconds later** than without the check.

Free-living cost: 48 held-out Capture-24 people (1226 h) that no model trained on. `stage 1` = window CNN with Capture-24 negatives (final models of `negatives --final`, validation-slice threshold); `two-stage` = `reports/stage2` at its practical thresholds, macro over the leave-one-fall-dataset-out folds. Mean ± std over seeds. A recording that ends during the wait counts as still (keeps the alarm).

## 1. Free-living alarms per day (`strict`)

| wait after decision | stage 1, 0.02 g | two-stage, 0.02 g | per week | two-stage, 0.05 g | per week |
|---|---|---|---|---|---|
| 0 s | 35.11 ± 15.07 | 16.94 ± 6.72 | 118.6 | 16.94 ± 6.72 | 118.6 |
| 2 s | 1.41 ± 0.71 | 1.26 ± 0.60 | 8.8 | 1.99 ± 0.93 | 13.9 |
| 5 s | 0.82 ± 0.40 | 0.75 ± 0.37 | 5.2 | 1.18 ± 0.57 | 8.3 |
| 10 s | 0.54 ± 0.26 | 0.51 ± 0.25 | 3.6 | 0.81 ± 0.37 | 5.6 |
| 20 s | 0.35 ± 0.18 | 0.35 ± 0.18 | 2.4 | 0.48 ± 0.25 | 3.4 |
| 30 s | 0.25 ± 0.13 | 0.29 ± 0.14 | 2.0 | 0.40 ± 0.20 | 2.8 |
| 60 s | 0.18 ± 0.11 | 0.20 ± 0.11 | 1.4 | 0.24 ± 0.13 | 1.7 |
| 120 s | 0.10 ± 0.08 | 0.14 ± 0.09 | 1.0 | 0.19 ± 0.11 | 1.3 |

## 2. How strict the check is

Alarms per day, by rule and motion level, after 10 s and 60 s waits.

| rule | level (g) | stage 1, 10 s | stage 1, 60 s | two-stage, 10 s | two-stage, 60 s |
|---|---|---|---|---|---|
| strict | 0.01 | 0.35 ± 0.14 | 0.11 ± 0.08 | 0.34 ± 0.16 | 0.13 ± 0.06 |
| strict | 0.02 | 0.54 ± 0.26 | 0.18 ± 0.11 | 0.51 ± 0.25 | 0.20 ± 0.11 |
| strict | 0.05 | 0.80 ± 0.36 | 0.20 ± 0.13 | 0.81 ± 0.37 | 0.24 ± 0.13 |
| most | 0.01 | 0.64 ± 0.25 | 0.75 ± 0.40 | 0.62 ± 0.27 | 0.47 ± 0.21 |
| most | 0.02 | 1.02 ± 0.47 | 0.89 ± 0.49 | 0.88 ± 0.41 | 0.64 ± 0.30 |
| most | 0.05 | 1.81 ± 0.96 | 1.59 ± 0.92 | 1.43 ± 0.66 | 1.05 ± 0.52 |
| mean | 0.01 | 0.54 ± 0.27 | 0.31 ± 0.16 | 0.57 ± 0.26 | 0.30 ± 0.17 |
| mean | 0.02 | 0.86 ± 0.38 | 0.60 ± 0.30 | 0.85 ± 0.40 | 0.50 ± 0.24 |
| mean | 0.05 | 1.70 ± 0.94 | 1.33 ± 0.77 | 1.40 ± 0.68 | 0.96 ± 0.47 |

## 3. Sensor noise: what "still" can mean per device

Motion when nothing moves. A level below a device's noise floor rejects everything recorded with it, falls included.

| source | measured on | motion (g) |
|---|---|---|
| erciyes | lab ADL, quietest second | 0.005 |
| umafall | lab ADL, quietest second | 0.029 |
| upfall | lab ADL, quietest second | 0.006 |
| weda | lab ADL, quietest second | 0.054 |
| capture24 | free-living sleep, median second | 0.004 |

## 4. Per person (`strict`, 0.02 g)

Each person's alarms per day averaged over seeds and folds; people are recorded for about a day each, so "no alarm" means none in their whole recording (mean over seeds and folds). 95% interval: bootstrap over people of the pooled rate.

| system | wait | pooled alarms/day (95% CI) | median person | 90th percentile person | worst person | people with no alarm |
|---|---|---|---|---|---|---|
| stage 1 | 0 s | 35.10 (29.39–40.93) | 30.33 | 69.55 | 91.1 | 0% |
| stage 1 | 10 s | 0.54 (0.35–0.75) | 0.30 | 1.47 | 4.1 | 62% |
| stage 1 | 60 s | 0.18 (0.07–0.31) | 0.00 | 0.60 | 3.0 | 85% |
| two-stage | 0 s | 16.97 (14.13–20.12) | 14.26 | 33.02 | 49.2 | 1% |
| two-stage | 10 s | 0.52 (0.35–0.70) | 0.30 | 1.12 | 3.9 | 63% |
| two-stage | 60 s | 0.21 (0.08–0.35) | 0.00 | 0.45 | 3.1 | 85% |

## 5. What the check costs on lab falls

`detected` = share of fall recordings (simulated falls) the system detects without any check. The check columns: of the detected ones, the share that also pass it, starting at impact + 4 s. Only recordings that continue long enough after the impact can be checked: `n` is how many (summed over folds, mean over seeds), so longer waits rest on few, unrepresentative recordings. In Erciyes 905, 906, 914 and 916 the volunteer gets up. Read each dataset against its noise floor (section 3).

`strict`, 0.02 g:

| system | falls | detected | time recorded after impact + 4 s (median) | 2 s: n / pass | 5 s: n / pass | 10 s: n / pass |
|---|---|---|---|---|---|---|
| stage 1 | erciyes other | 0.88 | 5.0 s | 1272 / 0.96 | 646 / 0.92 | 9 / 0.78 |
| stage 1 | erciyes recovery | 0.92 | 7.4 s | 334 / 0.23 | 294 / 0.20 | 28 / 0.00 |
| stage 1 | umafall | 0.92 | 7.9 s | 185 / 0.12 | 182 / 0.09 | 0 / – |
| stage 1 | upfall | 0.87 | 4.2 s | 154 / 0.68 | 6 / 0.33 | 0 / – |
| stage 1 | weda | 0.89 | 2.7 s | 205 / 0.14 | 38 / 0.01 | 0 / – |
| two-stage | erciyes other | 0.86 | 5.0 s | 1245 / 0.96 | 635 / 0.92 | 9 / 0.78 |
| two-stage | erciyes recovery | 0.46 | 7.4 s | 167 / 0.29 | 140 / 0.23 | 21 / 0.00 |
| two-stage | umafall | 0.93 | 7.9 s | 185 / 0.13 | 182 / 0.09 | 0 / – |
| two-stage | upfall | 0.83 | 4.2 s | 142 / 0.71 | 1 / 1.00 | 0 / – |
| two-stage | weda | 0.84 | 2.9 s | 194 / 0.14 | 36 / 0.02 | 0 / – |

`strict`, 0.05 g:

| system | falls | detected | time recorded after impact + 4 s (median) | 2 s: n / pass | 5 s: n / pass | 10 s: n / pass |
|---|---|---|---|---|---|---|
| stage 1 | erciyes other | 0.88 | 5.0 s | 1272 / 0.99 | 646 / 0.98 | 9 / 0.78 |
| stage 1 | erciyes recovery | 0.92 | 7.4 s | 334 / 0.45 | 294 / 0.39 | 28 / 0.04 |
| stage 1 | umafall | 0.92 | 7.9 s | 185 / 0.75 | 182 / 0.63 | 0 / – |
| stage 1 | upfall | 0.87 | 4.2 s | 154 / 0.84 | 6 / 0.50 | 0 / – |
| stage 1 | weda | 0.89 | 2.7 s | 205 / 0.20 | 38 / 0.01 | 0 / – |
| two-stage | erciyes other | 0.86 | 5.0 s | 1245 / 0.99 | 635 / 0.98 | 9 / 0.78 |
| two-stage | erciyes recovery | 0.46 | 7.4 s | 167 / 0.49 | 140 / 0.40 | 21 / 0.00 |
| two-stage | umafall | 0.93 | 7.9 s | 185 / 0.75 | 182 / 0.63 | 0 / – |
| two-stage | upfall | 0.83 | 4.2 s | 142 / 0.86 | 1 / 1.00 | 0 / – |
| two-stage | weda | 0.84 | 2.9 s | 194 / 0.20 | 36 / 0.02 | 0 / – |

## 6. Where the remaining alarms come from (two-stage, `strict`, 0.02 g)

Alarm events per hour of each Capture-24 activity (pooled over seeds and folds; an event belongs to the activity at its first detection).

| activity | hours | no check | 10 s | 60 s |
|---|---|---|---|---|
| unlabelled | 417 | 0.808 | 0.027 | 0.009 |
| sleep | 292 | 0.059 | 0.027 | 0.018 |
| sitting | 271 | 0.352 | 0.022 | 0.003 |
| standing | 25 | 1.517 | 0.023 | 0.000 |
| mixed-activity | 44 | 1.278 | 0.004 | 0.000 |
| bicycling | 7 | 1.344 | 0.000 | 0.000 |
| household-chores | 61 | 1.824 | 0.000 | 0.000 |
| manual-work | 14 | 2.172 | 0.000 | 0.000 |
| vehicle | 28 | 0.593 | 0.000 | 0.000 |
| walking | 59 | 1.775 | 0.000 | 0.000 |
| sports | 7 | 6.748 | 0.000 | 0.000 |

## Reading

* Without a check, two-stage halves stage 1's alarms (35.1 → 16.9 per day). With a `strict` 0.02 g check the two converge: 0.54 vs 0.51 after 10 s, 0.18 vs 0.20 after 60 s. Stage 2's own gain comes mostly from short post-impact stillness, which the check already requires.
* The check is a product decision, not a better classifier: it trades alert delay and falls after which the wearer moves (gets up, struggles, moves an arm) for fewer false alarms.
* The free-living side is measured; the fall side is not, beyond a few seconds. Lab falls are scripted and recordings stop early, so whether real fallers keep the wrist still for 10–60 s is unknown here. That needs fall recordings that continue for at least a minute after the impact, with and without getting up, on the target watch (its noise floor sets the level).
* The levels and rules are a grid, not tuned on falls.
