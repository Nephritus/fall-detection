# Threshold transfer

Seeds: 5. Runtime 11.2 min. A held-out dataset stands in for a new device; rules are scored on its evaluation half (same split as `finetune`). All rules use the same trained model per run, so AUROC/AUPRC are identical across rules; only the threshold differs. Target: 95% sensitivity at as few false alarms as possible. Rules are described in `threshold.py`. `adl-budget-N` needs only ADL recordings of the new device; `falls-pool` needs labelled falls; `oracle` uses the evaluation labels and is not realisable. `sens ≥ 0.90` is the share of (seed, dataset) runs that got at least 0.90 sensitivity. All simulated falls: optimistic.

## Accelerometer only

Model quality (same for all rules): AUROC 0.982, AUPRC 0.862 (macro).

### Macro average over target datasets

| rule | sensitivity | false alarms/h | sens ≥ 0.90 | \|sens − 0.95\| |
|---|---|---|---|---|
| val-slice | 0.851 | 51.6 | 40% | 0.104 |
| lodo-pooled | 0.973 | 372.2 | 95% | 0.035 |
| lodo-median | 0.940 | 144.4 | 80% | 0.042 |
| lodo-worst | 0.986 | 731.9 | 100% | 0.041 |
| adl-budget-10 | 0.542 | 16.1 | 15% | 0.410 |
| adl-budget-25 | 0.717 | 34.2 | 45% | 0.237 |
| adl-budget-50 | 0.840 | 57.6 | 55% | 0.123 |
| falls-pool | 0.954 | 123.4 | 95% | 0.022 |
| oracle | 0.952 | 92.5 | 100% | 0.002 |

### Target: erciyes

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.872 ± 0.016 | 11.7 ± 6.8 | 0.546 |
| lodo-pooled | 0.990 ± 0.011 | 737.0 ± 612.4 | 0.040 |
| lodo-median | 0.935 ± 0.025 | 32.8 ± 15.8 | 0.310 |
| lodo-worst | 0.997 ± 0.005 | 1324.1 ± 554.0 | 0.008 |
| adl-budget-10 | 0.861 ± 0.018 | 9.7 ± 5.2 | 0.585 |
| adl-budget-25 | 0.945 ± 0.014 | 28.4 ± 10.1 | 0.249 |
| adl-budget-50 | 0.965 ± 0.015 | 53.9 ± 25.6 | 0.167 |
| falls-pool | 0.960 ± 0.011 | 46.2 ± 23.1 | 0.204 |
| oracle | 0.950 ± 0.000 | 35.5 ± 11.8 | 0.245 |

### Target: umafall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.954 ± 0.018 | 109.9 ± 28.6 | 0.787 |
| lodo-pooled | 0.989 ± 0.007 | 312.1 ± 66.1 | 0.171 |
| lodo-median | 0.988 ± 0.007 | 314.3 ± 114.0 | 0.181 |
| lodo-worst | 0.995 ± 0.005 | 622.5 ± 473.9 | 0.081 |
| adl-budget-10 | 0.241 ± 0.141 | 11.9 ± 10.8 | 1.000 |
| adl-budget-25 | 0.400 ± 0.139 | 23.9 ± 22.9 | 1.000 |
| adl-budget-50 | 0.626 ± 0.130 | 44.9 ± 31.5 | 0.998 |
| falls-pool | 0.955 ± 0.022 | 124.1 ± 49.0 | 0.762 |
| oracle | 0.952 ± 0.000 | 101.6 ± 36.1 | 0.837 |

### Target: upfall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.723 ± 0.071 | 14.8 ± 10.5 | 0.675 |
| lodo-pooled | 0.947 ± 0.054 | 189.2 ± 223.8 | 0.097 |
| lodo-median | 0.896 ± 0.064 | 25.0 ± 15.9 | 0.206 |
| lodo-worst | 0.969 ± 0.035 | 544.4 ± 447.3 | 0.044 |
| adl-budget-10 | 0.749 ± 0.225 | 32.7 ± 41.6 | 0.407 |
| adl-budget-25 | 0.928 ± 0.027 | 54.1 ± 57.4 | 0.146 |
| adl-budget-50 | 0.947 ± 0.023 | 75.7 ± 50.4 | 0.077 |
| falls-pool | 0.943 ± 0.036 | 152.2 ± 182.1 | 0.092 |
| oracle | 0.953 ± 0.001 | 90.0 ± 51.0 | 0.083 |

### Target: weda

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.854 ± 0.061 | 69.9 ± 31.0 | 0.793 |
| lodo-pooled | 0.966 ± 0.020 | 250.6 ± 196.0 | 0.123 |
| lodo-median | 0.940 ± 0.059 | 205.3 ± 195.0 | 0.206 |
| lodo-worst | 0.984 ± 0.009 | 436.5 ± 271.0 | 0.058 |
| adl-budget-10 | 0.319 ± 0.179 | 10.1 ± 6.5 | 0.999 |
| adl-budget-25 | 0.597 ± 0.153 | 30.4 ± 16.4 | 0.971 |
| adl-budget-50 | 0.824 ± 0.073 | 55.8 ± 21.4 | 0.788 |
| falls-pool | 0.959 ± 0.022 | 171.1 ± 50.6 | 0.163 |
| oracle | 0.951 ± 0.000 | 142.7 ± 63.6 | 0.276 |

## Accelerometer + gyroscope

Model quality (same for all rules): AUROC 0.978, AUPRC 0.859 (macro).

### Macro average over target datasets

| rule | sensitivity | false alarms/h | sens ≥ 0.90 | \|sens − 0.95\| |
|---|---|---|---|---|
| val-slice | 0.850 | 59.4 | 30% | 0.105 |
| lodo-pooled | 0.967 | 235.9 | 95% | 0.031 |
| lodo-median | 0.937 | 145.0 | 70% | 0.044 |
| lodo-worst | 0.987 | 605.6 | 100% | 0.037 |
| adl-budget-10 | 0.546 | 15.0 | 10% | 0.405 |
| adl-budget-25 | 0.678 | 28.6 | 20% | 0.274 |
| adl-budget-50 | 0.786 | 46.3 | 30% | 0.167 |
| falls-pool | 0.957 | 176.2 | 95% | 0.024 |
| oracle | 0.952 | 146.4 | 100% | 0.002 |

### Target: erciyes

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.865 ± 0.043 | 30.2 ± 25.3 | 0.457 |
| lodo-pooled | 0.969 ± 0.028 | 185.7 ± 125.7 | 0.063 |
| lodo-median | 0.897 ± 0.033 | 41.8 ± 30.5 | 0.315 |
| lodo-worst | 0.988 ± 0.011 | 854.1 ± 628.8 | 0.020 |
| adl-budget-10 | 0.737 ± 0.041 | 8.1 ± 2.3 | 0.809 |
| adl-budget-25 | 0.884 ± 0.023 | 25.3 ± 6.0 | 0.386 |
| adl-budget-50 | 0.933 ± 0.026 | 56.2 ± 16.8 | 0.182 |
| falls-pool | 0.953 ± 0.013 | 84.1 ± 38.5 | 0.104 |
| oracle | 0.950 ± 0.000 | 78.8 ± 44.1 | 0.119 |

### Target: umafall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.957 ± 0.011 | 143.4 ± 41.8 | 0.804 |
| lodo-pooled | 0.985 ± 0.010 | 321.9 ± 76.5 | 0.107 |
| lodo-median | 0.982 ± 0.009 | 269.6 ± 70.7 | 0.167 |
| lodo-worst | 0.994 ± 0.008 | 581.1 ± 194.9 | 0.030 |
| adl-budget-10 | 0.228 ± 0.168 | 25.4 ± 20.8 | 1.000 |
| adl-budget-25 | 0.307 ± 0.217 | 29.9 ± 24.4 | 1.000 |
| adl-budget-50 | 0.501 ± 0.184 | 43.1 ± 29.7 | 0.998 |
| falls-pool | 0.978 ± 0.011 | 215.1 ± 65.4 | 0.378 |
| oracle | 0.952 ± 0.000 | 137.3 ± 30.8 | 0.810 |

### Target: upfall

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.782 ± 0.088 | 9.0 ± 6.3 | 0.481 |
| lodo-pooled | 0.953 ± 0.037 | 219.1 ± 118.8 | 0.046 |
| lodo-median | 0.921 ± 0.066 | 100.2 ± 97.9 | 0.086 |
| lodo-worst | 0.985 ± 0.015 | 587.8 ± 340.3 | 0.019 |
| adl-budget-10 | 0.875 ± 0.059 | 18.2 ± 10.9 | 0.180 |
| adl-budget-25 | 0.892 ± 0.058 | 35.6 ± 28.9 | 0.128 |
| adl-budget-50 | 0.892 ± 0.058 | 37.2 ± 29.4 | 0.121 |
| falls-pool | 0.940 ± 0.042 | 194.3 ± 174.1 | 0.056 |
| oracle | 0.953 ± 0.001 | 186.1 ± 150.9 | 0.153 |

### Target: weda

| rule | sensitivity | false alarms/h | threshold |
|---|---|---|---|
| val-slice | 0.797 ± 0.060 | 55.0 ± 45.7 | 0.750 |
| lodo-pooled | 0.961 ± 0.026 | 216.8 ± 98.0 | 0.117 |
| lodo-median | 0.949 ± 0.044 | 168.4 ± 85.1 | 0.160 |
| lodo-worst | 0.982 ± 0.013 | 399.4 ± 207.8 | 0.062 |
| adl-budget-10 | 0.345 ± 0.071 | 8.3 ± 6.0 | 0.988 |
| adl-budget-25 | 0.629 ± 0.052 | 23.7 ± 18.3 | 0.895 |
| adl-budget-50 | 0.818 ± 0.034 | 48.6 ± 26.0 | 0.727 |
| falls-pool | 0.957 ± 0.022 | 211.3 ± 108.4 | 0.250 |
| oracle | 0.951 ± 0.000 | 183.4 ± 117.7 | 0.274 |
