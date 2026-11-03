# Capture-24 as training negatives

Seeds: 3. Leave-one-fall-dataset-out, accel-only augmented CNN (stage 1). Capture-24 participants P001-P100 (usable ones: 80% train / 20% validation negatives) are added to training; P101-P151 are held out and measure free-living alarm events per day (30 s merge). `c24-random` adds 1500 random windows per training person; `c24-mined` adds the same number, two thirds of them the highest-scoring windows under the control model (hard negatives). Cells: macro average over the four held-out fall datasets, mean ± std over seeds. The free-living columns use the **held-out** Capture-24 people, whose data the model never saw. **Sensitivity at a budget** = share of the held-out dataset's fall windows detected at the lowest threshold that keeps free-living alarms at or below B events/day (threshold-free: it reads the operating curve, so it is not affected by how the added negatives shift score calibration). Fall windows are simulated falls, so real sensitivity is lower, and per-window sensitivity is not per-fall detection.

## Macro over held-out fall datasets

| condition | AUROC | AUPRC | lab oracle FA/h | lab sens @val thr | lab FA/h @val thr | free-living alarms/day @val thr | sens @10/day | sens @25/day | sens @50/day | sens @100/day | free-living alarms/day @95% sens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 0.990 ± 0.002 | 0.939 ± 0.005 | 50.4 ± 8.6 | 0.857 ± 0.038 | 29.2 ± 11.5 | 92.8 ± 30.6 | 0.359 ± 0.033 | 0.596 ± 0.044 | 0.748 ± 0.041 | 0.862 ± 0.019 | 209.9 ± 9.9 |
| c24-random | 0.992 ± 0.001 | 0.957 ± 0.008 | 43.5 ± 4.9 | 0.862 ± 0.043 | 20.3 ± 11.5 | 39.1 ± 15.4 | 0.597 ± 0.046 | 0.798 ± 0.023 | 0.890 ± 0.009 | 0.943 ± 0.011 | 98.9 ± 10.4 |
| c24-mined | 0.980 ± 0.005 | 0.937 ± 0.016 | 96.6 ± 52.9 | 0.779 ± 0.043 | 15.5 ± 7.9 | 17.0 ± 6.9 | 0.716 ± 0.012 | 0.833 ± 0.018 | 0.895 ± 0.017 | 0.937 ± 0.013 | 410.0 ± 280.0 |

## Paired difference to the control

Δ = condition − `real`, per seed (macro over folds); mean ± standard error; wins = seeds in which the condition is better.

| condition | metric | Δ | wins |
|---|---|---|---|
| c24-random | AUROC | +0.002 ± 0.001 | 2/3 |
| c24-random | AUPRC | +0.018 ± 0.006 | 3/3 |
| c24-random | lab oracle FA/h | -6.930 ± 2.602 | 3/3 |
| c24-random | lab sens @val thr | +0.004 ± 0.005 | 2/3 |
| c24-random | lab FA/h @val thr | -8.920 ± 2.120 | 3/3 |
| c24-random | free-living alarms/day @val thr | -53.644 ± 8.968 | 3/3 |
| c24-random | sens @10/day | +0.238 ± 0.013 | 3/3 |
| c24-random | sens @25/day | +0.203 ± 0.014 | 3/3 |
| c24-random | sens @50/day | +0.142 ± 0.019 | 3/3 |
| c24-random | sens @100/day | +0.081 ± 0.006 | 3/3 |
| c24-random | free-living alarms/day @95% sens | -110.990 ± 11.193 | 3/3 |
| c24-mined | AUROC | -0.010 ± 0.003 | 0/3 |
| c24-mined | AUPRC | -0.002 ± 0.009 | 1/3 |
| c24-mined | lab oracle FA/h | +46.219 ± 25.683 | 0/3 |
| c24-mined | lab sens @val thr | -0.079 ± 0.003 | 0/3 |
| c24-mined | lab FA/h @val thr | -13.751 ± 2.375 | 3/3 |
| c24-mined | free-living alarms/day @val thr | -75.751 ± 13.745 | 3/3 |
| c24-mined | sens @10/day | +0.357 ± 0.016 | 3/3 |
| c24-mined | sens @25/day | +0.237 ± 0.021 | 3/3 |
| c24-mined | sens @50/day | +0.147 ± 0.023 | 3/3 |
| c24-mined | sens @100/day | +0.075 ± 0.012 | 3/3 |
| c24-mined | free-living alarms/day @95% sens | +200.133 ± 156.207 | 0/3 |

## Per held-out fall dataset: Sensitivity at 50 alarms/day

| condition | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| real | 0.782 | 0.865 | 0.641 | 0.706 |
| c24-random | 0.893 | 0.939 | 0.856 | 0.873 |
| c24-mined | 0.905 | 0.959 | 0.922 | 0.795 |

## Per held-out fall dataset: Free-living alarms/day at 95% sensitivity

| condition | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| real | 219.1 | 103.8 | 277.3 | 239.3 |
| c24-random | 121.6 | 57.6 | 114.5 | 101.8 |
| c24-mined | 474.7 | 35.9 | 81.0 | 1048.4 |
