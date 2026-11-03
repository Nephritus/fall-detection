# Threshold transfer (with augmentation)

Seeds: 5. Runtime 23.9 min. A held-out dataset stands in for a new device; rules are scored on its evaluation half (same split as `finetune`). All rules use the same trained model per run, so AUROC/AUPRC are identical across rules; only the threshold differs. Target: 95% sensitivity at as few false alarms as possible. Rules are described in `threshold.py`. `adl-budget-N` needs only ADL recordings of the new device; `falls-pool` needs labelled falls; `oracle` uses the evaluation labels and is not realisable. `sens ≥ 0.90` is the share of (seed, dataset) runs that got at least 0.90 sensitivity. All simulated falls: optimistic.

Training (final and nested models) uses augmentation: Augment(p=0.5, roll_deg=30.0, tilt_deg=10.0, gain=0.1, stretch=0.1, shift_s=0.2, jitter_acc_g=0.02, jitter_gyro=0.05, enabled=('rotation', 'mirror', 'gain', 'stretch', 'shift', 'jitter')).

## Accelerometer only

Model quality (same for all rules): AUROC 0.992, AUPRC 0.949 (macro).

### Macro average over target datasets

| rule | sensitivity | false alarms/h | sens ≥ 0.90 | \|sens − 0.95\| |
|---|---|---|---|---|
| val-slice | 0.867 | 26.6 | 30% | 0.085 |
| lodo-pooled | 0.958 | 75.5 | 85% | 0.034 |
| lodo-median | 0.938 | 71.2 | 80% | 0.040 |
| lodo-worst | 0.979 | 164.8 | 100% | 0.031 |
| adl-budget-10 | 0.719 | 11.7 | 30% | 0.242 |
| adl-budget-25 | 0.850 | 31.9 | 55% | 0.121 |
| adl-budget-50 | 0.937 | 55.1 | 75% | 0.046 |
| falls-pool | 0.949 | 55.0 | 100% | 0.018 |
| oracle | 0.952 | 46.4 | 100% | 0.002 |

### Target: erciyes

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.858 ± 0.040 | 8.3 ± 3.5 | 0.749 |
| lodo-pooled | 0.932 ± 0.036 | 19.8 ± 9.2 | 0.372 |
| lodo-median | 0.904 ± 0.064 | 16.4 ± 9.7 | 0.503 |
| lodo-worst | 0.962 ± 0.011 | 29.9 ± 12.1 | 0.185 |
| adl-budget-10 | 0.879 ± 0.032 | 9.8 ± 3.8 | 0.626 |
| adl-budget-25 | 0.951 ± 0.016 | 22.9 ± 7.8 | 0.300 |
| adl-budget-50 | 0.973 ± 0.009 | 50.2 ± 27.8 | 0.135 |
| falls-pool | 0.954 ± 0.014 | 26.6 ± 8.4 | 0.276 |
| oracle | 0.950 ± 0.000 | 19.8 ± 4.6 | 0.317 |

### Target: umafall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.932 ± 0.025 | 35.5 ± 9.9 | 0.928 |
| lodo-pooled | 0.981 ± 0.008 | 84.8 ± 19.1 | 0.354 |
| lodo-median | 0.975 ± 0.011 | 79.1 ± 12.5 | 0.400 |
| lodo-worst | 0.989 ± 0.010 | 124.9 ± 37.2 | 0.194 |
| adl-budget-10 | 0.565 ± 0.169 | 10.5 ± 6.7 | 0.999 |
| adl-budget-25 | 0.794 ± 0.116 | 21.5 ± 9.8 | 0.969 |
| adl-budget-50 | 0.936 ± 0.039 | 45.7 ± 14.6 | 0.765 |
| falls-pool | 0.944 ± 0.030 | 47.2 ± 10.6 | 0.796 |
| oracle | 0.952 ± 0.000 | 47.3 ± 15.0 | 0.764 |

### Target: upfall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.797 ± 0.070 | 2.3 ± 2.1 | 0.875 |
| lodo-pooled | 0.951 ± 0.041 | 16.0 ± 10.8 | 0.240 |
| lodo-median | 0.901 ± 0.061 | 5.6 ± 5.1 | 0.552 |
| lodo-worst | 0.979 ± 0.023 | 229.2 ± 209.2 | 0.098 |
| adl-budget-10 | 0.958 ± 0.039 | 18.2 ± 11.1 | 0.185 |
| adl-budget-25 | 0.985 ± 0.013 | 64.8 ± 36.2 | 0.077 |
| adl-budget-50 | 0.987 ± 0.013 | 80.5 ± 40.0 | 0.069 |
| falls-pool | 0.936 ± 0.006 | 8.9 ± 6.8 | 0.437 |
| oracle | 0.953 ± 0.001 | 10.6 ± 6.0 | 0.327 |

### Target: weda

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.880 ± 0.062 | 60.4 ± 28.4 | 0.844 |
| lodo-pooled | 0.968 ± 0.040 | 181.2 ± 39.6 | 0.397 |
| lodo-median | 0.972 ± 0.030 | 183.6 ± 41.7 | 0.362 |
| lodo-worst | 0.984 ± 0.024 | 275.1 ± 110.1 | 0.238 |
| adl-budget-10 | 0.473 ± 0.083 | 8.4 ± 2.6 | 0.999 |
| adl-budget-25 | 0.671 ± 0.070 | 18.3 ± 3.7 | 0.993 |
| adl-budget-50 | 0.852 ± 0.056 | 44.0 ± 5.4 | 0.939 |
| falls-pool | 0.960 ± 0.021 | 137.5 ± 51.5 | 0.574 |
| oracle | 0.951 ± 0.000 | 108.0 ± 48.0 | 0.666 |

## Accelerometer + gyroscope

Model quality (same for all rules): AUROC 0.990, AUPRC 0.932 (macro).

### Macro average over target datasets

| rule | sensitivity | false alarms/h | sens ≥ 0.90 | \|sens − 0.95\| |
|---|---|---|---|---|
| val-slice | 0.858 | 31.4 | 30% | 0.095 |
| lodo-pooled | 0.963 | 109.5 | 95% | 0.028 |
| lodo-median | 0.951 | 97.0 | 90% | 0.033 |
| lodo-worst | 0.981 | 183.5 | 100% | 0.032 |
| adl-budget-10 | 0.657 | 13.2 | 20% | 0.301 |
| adl-budget-25 | 0.817 | 36.1 | 50% | 0.149 |
| adl-budget-50 | 0.900 | 58.1 | 60% | 0.075 |
| falls-pool | 0.948 | 63.0 | 85% | 0.022 |
| oracle | 0.952 | 54.6 | 100% | 0.002 |

### Target: erciyes

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.874 ± 0.037 | 12.9 ± 4.3 | 0.666 |
| lodo-pooled | 0.954 ± 0.018 | 29.7 ± 5.2 | 0.279 |
| lodo-median | 0.934 ± 0.024 | 22.1 ± 2.6 | 0.407 |
| lodo-worst | 0.981 ± 0.007 | 72.2 ± 32.3 | 0.097 |
| adl-budget-10 | 0.812 ± 0.033 | 7.4 ± 2.3 | 0.830 |
| adl-budget-25 | 0.927 ± 0.026 | 21.3 ± 5.6 | 0.451 |
| adl-budget-50 | 0.970 ± 0.010 | 52.0 ± 22.1 | 0.169 |
| falls-pool | 0.948 ± 0.027 | 28.6 ± 8.9 | 0.322 |
| oracle | 0.950 ± 0.000 | 26.9 ± 6.2 | 0.345 |

### Target: umafall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.950 ± 0.017 | 66.5 ± 10.1 | 0.902 |
| lodo-pooled | 0.982 ± 0.011 | 126.2 ± 33.7 | 0.370 |
| lodo-median | 0.980 ± 0.012 | 123.2 ± 33.4 | 0.413 |
| lodo-worst | 0.989 ± 0.005 | 169.7 ± 37.1 | 0.190 |
| adl-budget-10 | 0.419 ± 0.231 | 19.1 ± 22.0 | 1.000 |
| adl-budget-25 | 0.648 ± 0.186 | 29.6 ± 31.8 | 0.993 |
| adl-budget-50 | 0.810 ± 0.121 | 47.4 ± 39.3 | 0.957 |
| falls-pool | 0.964 ± 0.012 | 86.8 ± 31.4 | 0.687 |
| oracle | 0.952 ± 0.000 | 65.7 ± 15.9 | 0.906 |

### Target: upfall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.752 ± 0.032 | 1.6 ± 0.9 | 0.890 |
| lodo-pooled | 0.932 ± 0.031 | 37.5 ± 55.0 | 0.263 |
| lodo-median | 0.917 ± 0.049 | 39.5 ± 58.7 | 0.339 |
| lodo-worst | 0.968 ± 0.019 | 117.8 ± 132.3 | 0.125 |
| adl-budget-10 | 0.943 ± 0.054 | 19.4 ± 15.3 | 0.275 |
| adl-budget-25 | 0.979 ± 0.017 | 73.4 ± 40.5 | 0.075 |
| adl-budget-50 | 0.979 ± 0.017 | 90.3 ± 38.9 | 0.066 |
| falls-pool | 0.920 ± 0.045 | 8.6 ± 4.0 | 0.367 |
| oracle | 0.953 ± 0.001 | 13.5 ± 9.2 | 0.253 |

### Target: weda

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.854 ± 0.021 | 44.5 ± 21.6 | 0.876 |
| lodo-pooled | 0.982 ± 0.016 | 244.4 ± 131.0 | 0.256 |
| lodo-median | 0.972 ± 0.029 | 203.1 ± 130.4 | 0.329 |
| lodo-worst | 0.987 ± 0.015 | 374.1 ± 228.6 | 0.136 |
| adl-budget-10 | 0.453 ± 0.099 | 7.1 ± 4.0 | 0.998 |
| adl-budget-25 | 0.717 ± 0.058 | 20.2 ± 5.8 | 0.979 |
| adl-budget-50 | 0.842 ± 0.046 | 42.6 ± 11.0 | 0.870 |
| falls-pool | 0.960 ± 0.008 | 128.1 ± 34.7 | 0.490 |
| oracle | 0.951 ± 0.000 | 112.2 ± 39.5 | 0.567 |
