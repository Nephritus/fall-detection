# False alarms: from replayed clips to a daily estimate

Seeds: 5. Leave-one-dataset-out, accel-only augmented CNN; every window of the held-out dataset is scored. Alarm events merge fired windows of a recording with a 30 s refractory period. Categories map the four datasets' ADL codes to common activities (`alarms.py`, UP-Fall codes from the dataset's published list, not from the files). **The daily estimates multiply measured per-activity rates by an ASSUMED time budget; the budgets are not measurements, so treat the daily numbers as scenario arithmetic.** All data are scripted clips of simulated activity, not free-living wear.

## 1. What event counting changes

| rule | windows fired per replay hour | alarm events per replay hour (refractory sweep: 2 s / 10 s / 30 s / 60 s) | fall events detected |
|---|---|---|---|
| val-slice | 17.0 | 17.0 / 11.0 / 10.8 / 10.8 | 0.885 (2565 fall recordings) |
| lodo-pooled | 44.8 | 44.8 / 26.6 / 25.4 / 25.3 | 0.964 (2565 fall recordings) |

This is still the *replay* mix (lots of jogging, hopping and clapping), so it is not a daily rate; it only shows how much of the old number was overlapping windows.

## 2. Alarm rate per activity (refractory 30 s)

`events/clip` suits discrete activities, `events/h` continuous ones. In brackets: exact Poisson one-sided 95% upper bound. Counts are small and clustered by subject (a few volunteers repeating one motion), so the true uncertainty is larger than the bound shows.

### Rule: val-slice

| category | enters a day as | clips | hours | alarm events (mean per seed) | events/clip | events/h |
|---|---|---|---|---|---|---|
| hand impact | count | 184 | 0.46 | 35.0 | 0.190 (≤0.251) | 76.18 (≤100.66) |
| jumping/hopping | hours | 129 | 0.55 | 35.6 | 0.276 (≤0.365) | 64.34 (≤85.08) |
| lying down/rising | count | 244 | 1.18 | 31.6 | 0.130 (≤0.173) | 26.77 (≤35.83) |
| stairs | hours | 92 | 0.55 | 9.0 | 0.098 (≤0.169) | 16.25 (≤28.00) |
| jogging | hours | 162 | 0.83 | 10.0 | 0.062 (≤0.104) | 12.07 (≤20.43) |
| stumble | count | 227 | 1.23 | 9.0 | 0.040 (≤0.069) | 7.31 (≤12.75) |
| arm gesture | count | 296 | 1.21 | 8.4 | 0.028 (≤0.050) | 6.95 (≤12.30) |
| bending/crouching | count | 409 | 1.68 | 10.4 | 0.025 (≤0.042) | 6.20 (≤10.18) |
| sit-down/stand-up | count | 548 | 2.20 | 5.6 | 0.010 (≤0.020) | 2.55 (≤4.98) |
| walking | hours | 443 | 3.06 | 2.6 | 0.006 (≤0.016) | 0.85 (≤2.33) |
| stationary | hours | 108 | 1.75 | 1.4 | 0.013 (≤0.050) | 0.80 (≤3.07) |

### Rule: lodo-pooled

| category | enters a day as | clips | hours | alarm events (mean per seed) | events/clip | events/h |
|---|---|---|---|---|---|---|
| hand impact | count | 184 | 0.46 | 67.2 | 0.365 (≤0.447) | 146.26 (≤179.18) |
| jumping/hopping | hours | 129 | 0.55 | 48.8 | 0.378 (≤0.480) | 88.19 (≤111.94) |
| lying down/rising | count | 244 | 1.18 | 78.8 | 0.323 (≤0.389) | 66.75 (≤80.35) |
| stairs | hours | 92 | 0.55 | 24.4 | 0.265 (≤0.372) | 44.05 (≤61.74) |
| jogging | hours | 162 | 0.83 | 28.8 | 0.178 (≤0.242) | 34.77 (≤47.42) |
| stumble | count | 227 | 1.23 | 27.2 | 0.120 (≤0.164) | 22.10 (≤30.26) |
| arm gesture | count | 296 | 1.21 | 25.0 | 0.084 (≤0.118) | 20.67 (≤28.81) |
| bending/crouching | count | 409 | 1.68 | 29.4 | 0.072 (≤0.098) | 17.53 (≤23.84) |
| sit-down/stand-up | count | 548 | 2.20 | 24.0 | 0.044 (≤0.062) | 10.91 (≤15.34) |
| stationary | hours | 108 | 1.75 | 10.6 | 0.098 (≤0.161) | 6.06 (≤9.98) |
| walking | hours | 443 | 3.06 | 9.8 | 0.022 (≤0.038) | 3.20 (≤5.43) |

## 3. Daily estimate under assumed time budgets

Alarm events per day, mean ± std over seeds. *point* uses measured rates; *upper* uses every category's 95% upper bound (a conservative sum of bounds).

| budget | rule | point estimate | upper (sum of 95% bounds) |
|---|---|---|---|
| typical day (24 h worn incl. sleep) | val-slice | 26.1 ± 10.0 | 81.8 ± 14.2 |
| typical day (24 h worn incl. sleep) | lodo-pooled | 154.7 ± 84.2 | 249.1 ± 104.9 |
| active day (+1 h jogging, 10 min jumping, more impacts) | val-slice | 55.2 ± 14.9 | 122.6 ± 19.6 |
| active day (+1 h jogging, 10 min jumping, more impacts) | lodo-pooled | 210.0 ± 78.3 | 319.5 ± 97.3 |

### Where the daily alarms come from (point estimate)

- typical day (24 h worn incl. sleep), val-slice: stationary 16.8, stairs 3.2, arm gesture 1.7, walking 1.7, hand impact 1.0
- typical day (24 h worn incl. sleep), lodo-pooled: stationary 127.4, stairs 8.8, walking 6.4, arm gesture 5.1, bending/crouching 2.2
- active day (+1 h jogging, 10 min jumping, more impacts), val-slice: stationary 14.8, jogging 12.1, jumping/hopping 10.9, stairs 6.5, hand impact 3.8
- active day (+1 h jogging, 10 min jumping, more impacts), lodo-pooled: stationary 112.2, jogging 34.8, stairs 17.6, jumping/hopping 15.0, walking 9.6

`stationary` comes only from UP-Fall's 60 s sitting/standing/laying clips (no dataset has hours of rest or sleep); its alarm events come from very few clips, so its daily contribution is the least certain part of the estimate.

### Budgets used (assumptions)

| category | unit | typical day (24 h worn incl. sleep) | active day (+1 h jogging, 10 min jumping, more impacts) |
|---|---|---|---|
| arm gesture | times/day | 60 | 80 |
| bending/crouching | times/day | 30 | 50 |
| hand impact | times/day | 5 | 20 |
| jogging | h/day | 0 | 1 |
| jumping/hopping | h/day | 0 | 0.17 |
| lying down/rising | times/day | 4 | 4 |
| sit-down/stand-up | times/day | 40 | 40 |
| stairs | h/day | 0.2 | 0.4 |
| stationary | h/day | 21 | 18.5 |
| stumble | times/day | 0.5 | 1 |
| walking | h/day | 2 | 3 |

Not covered by any dataset, counted as stationary: sleep movement, driving, typing, cooking, cleaning, sport other than jogging/jumping. Vigorous ones among them would add alarms.
