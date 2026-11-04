# False alarms in older adults (NHANES 2013-14)

93 of 100 sampled NHANES participants aged 65+ (age 73 median, 14 aged 80+, 52 women), ActiGraph GT3X+ on the non-dominant wrist (taken as the left), 16,689 h worn of 17,732 h recorded (22.6 h worn per 24 h). Models: the final stage-1 CNNs with Capture-24 negatives (`negatives --final`, 3 seeds) at their validation-slice thresholds; nobody here was trained on, and neither the device nor the age group was. Alarm events merge with a 30 s refractory period; only worn windows count, as exposure and as detections. Mean ± std over seeds.

Excluded at preparation: 73747: walking-like signal 280 s; 75495: walking mean norm 0.42 g; 76469: walking mean norm 0.46 g; 77729: walking mean norm 0.34 g; 81464: walking mean norm 0.44 g; 81593: walking mean norm 0.21 g; 82868: walking mean norm 0.35 g.

## 1. Alarms per day of wear, against Capture-24

Stillness check `strict`, 0.02 g (as `reports/stillness`). Capture-24 = the 48 held-out people, same models.

| wait after decision | NHANES 65+ | per week | Capture-24 | NHANES / Capture-24 |
|---|---|---|---|---|
| 0 s | 15.12 ± 6.57 | 105.8 | 35.11 | 0.43× |
| 2 s | 1.18 ± 0.58 | 8.3 | 1.41 | 0.84× |
| 5 s | 0.80 ± 0.40 | 5.6 | 0.82 | 0.98× |
| 10 s | 0.57 ± 0.28 | 4.0 | 0.54 | 1.07× |
| 20 s | 0.40 ± 0.19 | 2.8 | 0.35 | 1.13× |
| 30 s | 0.32 ± 0.15 | 2.3 | 0.25 | 1.30× |
| 60 s | 0.21 ± 0.09 | 1.5 | 0.18 | 1.18× |
| 120 s | 0.14 ± 0.06 | 1.0 | 0.10 | 1.32× |

Without removing non-wear time (exposure = all recorded time, detections anywhere): 14.2 alarms/day before any check.

Mounting check: 10 people needed more than 80° to bring their own walking direction onto the pooled one (an unusual or flipped strap position, or a right wrist; neither explains them fully). Without them: 13.44 / 0.48 / 0.18 alarms/day with no check / 10 s / 60 s.

## 2. Per person

Each person's rate averaged over seeds; people wore the monitor for about a week. 95% interval: bootstrap over people of the seed-averaged rates.

| wait | pooled (95% CI) | median person | 90th percentile | worst person | people with no alarm in the week |
|---|---|---|---|---|---|
| 0 s | 15.12 (12.64–18.11) | 11.20 | 34.60 | 59.9 | 0% |
| 10 s | 0.57 (0.44–0.72) | 0.30 | 1.65 | 5.0 | 18% |
| 60 s | 0.21 (0.16–0.27) | 0.13 | 0.44 | 1.7 | 35% |

## 3. By age

| age | people | no check | 10 s | 60 s |
|---|---|---|---|---|
| 65–69 | 28 | 22.79 | 0.78 | 0.24 |
| 70–74 | 31 | 14.68 | 0.64 | 0.25 |
| 75–79 | 20 | 10.43 | 0.45 | 0.18 |
| 80 (top-coded 80+) | 14 | 7.17 | 0.18 | 0.07 |

## 4. By time of day

Share of alarm events by clock hour of the detection that opened them (all seeds), and alarms per day from each part of the day.

| part of day | no check | 10 s | 60 s |
|---|---|---|---|
| night 23–07 | 1.52 | 0.13 | 0.05 |
| morning 07–12 | 4.68 | 0.12 | 0.05 |
| afternoon 12–18 | 5.68 | 0.16 | 0.06 |
| evening 18–23 | 3.24 | 0.17 | 0.05 |

## 5. Threshold trade-off (no check)

Alarm events per day of wear at fixed thresholds, against Capture-24 (`reports/freeliving_c24random/tradeoff.csv`, same models).

| threshold | NHANES 65+ | Capture-24 |
|---|---|---|
| 0.5 | 51.1 | 105.3 |
| 0.7 | 36.5 | 78.7 |
| 0.8 | 29.7 | 65.2 |
| 0.9 | 22.0 | 49.3 |
| 0.95 | 16.6 | 38.4 |
| 0.98 | 11.8 | 28.6 |
| 0.99 | 9.2 | 22.7 |
| 0.995 | 7.2 | 18.1 |
| 0.999 | 4.2 | 11.4 |

## 6. Data checks

* Mounting: walking-like signal per person 136 min median (min 8); its mean specific force 0.69 g. One device rotation for everyone (roll 272° about the hanging-arm direction, from pooled resting poses), then a per-person refinement of 23° median (90th percentile 88°).
* Wear: 184 h worn of 192 h recorded (median person).
* The GT3X+ reports 0.001 g steps and holds identical values while still, so a still wrist reads almost exactly 0 g of motion; the 0.02 g stillness level is well above that floor.

## Reading

* This measures false alarms only: NHANES has no fall annotations, so a real fall during the week would count as a false alarm.
* Device and wrist differ from a smartwatch: a research ActiGraph on the non-dominant wrist (assumed left, so not mirrored).
* Mounting comes from detected walking and pooled resting poses, not labels; on held-out Capture-24 people seen through random device rotations that gave 28.6 alarms/day against 29.7 with the labelled alignment.
