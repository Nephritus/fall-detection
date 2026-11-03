# More Capture-24 negatives: dose-response

Seeds: 3 ([0, 1, 2]). Leave-one-fall-dataset-out, accel-only augmented CNN; random Capture-24 windows per training person (about 47,000 windows exist per person; training people: 74, validation people: 18; held-out people P101-P151, never trained on). K=0 and K=1500 are the control and `c24-random` from `reports/negatives/report.md` (same seeds and splits; windows drawn with a different random stream). Cells: macro over the four held-out fall datasets, mean ± std over seeds. Free-living columns use the held-out people. Sensitivity at a budget is threshold-free (see `negatives.py`). Fall windows are simulated.

## Macro over held-out fall datasets

| windows per person | total negatives added | AUROC | AUPRC | lab oracle FA/h | lab sens @val thr | lab FA/h @val thr | free-living alarms/day @val thr | sens @10/day | sens @25/day | sens @50/day | sens @100/day | free-living alarms/day @95% sens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.990 ± 0.002 | 0.939 ± 0.005 | 50.4 ± 8.6 | 0.857 ± 0.038 | 29.2 ± 11.5 | 92.8 ± 30.6 | 0.359 ± 0.033 | 0.596 ± 0.044 | 0.748 ± 0.041 | 0.862 ± 0.019 | 209.9 ± 9.9 |
| 1500 | 111,000 | 0.992 ± 0.001 | 0.957 ± 0.008 | 43.5 ± 4.9 | 0.862 ± 0.043 | 20.3 ± 11.5 | 39.1 ± 15.4 | 0.597 ± 0.046 | 0.798 ± 0.023 | 0.890 ± 0.009 | 0.943 ± 0.011 | 98.9 ± 10.4 |
| 5000 | 370,000 | 0.991 ± 0.001 | 0.949 ± 0.004 | 52.7 ± 2.4 | 0.849 ± 0.045 | 22.3 ± 8.3 | 36.8 ± 16.0 | 0.612 ± 0.019 | 0.789 ± 0.017 | 0.885 ± 0.014 | 0.938 ± 0.012 | 107.6 ± 14.4 |
| 15000 | 1,110,000 | 0.991 ± 0.001 | 0.944 ± 0.005 | 56.1 ± 4.4 | 0.864 ± 0.036 | 27.9 ± 9.7 | 44.2 ± 12.8 | 0.568 ± 0.032 | 0.779 ± 0.018 | 0.876 ± 0.013 | 0.937 ± 0.006 | 113.8 ± 8.6 |

## Paired difference to K=1500 (the previous setting)

| windows per person | metric | Δ | wins |
|---|---|---|---|
| 5000 | AUROC | -0.001 ± 0.001 | 1/3 |
| 5000 | AUPRC | -0.008 ± 0.005 | 1/3 |
| 5000 | lab oracle FA/h | +9.199 ± 1.740 | 0/3 |
| 5000 | lab sens @val thr | -0.013 ± 0.002 | 0/3 |
| 5000 | lab FA/h @val thr | +1.927 ± 2.463 | 1/3 |
| 5000 | free-living alarms/day @val thr | -2.360 ± 1.496 | 3/3 |
| 5000 | sens @10/day | +0.015 ± 0.023 | 2/3 |
| 5000 | sens @25/day | -0.009 ± 0.005 | 1/3 |
| 5000 | sens @50/day | -0.006 ± 0.003 | 0/3 |
| 5000 | sens @100/day | -0.005 ± 0.000 | 0/3 |
| 5000 | free-living alarms/day @95% sens | +8.669 ± 6.362 | 1/3 |
| 15000 | AUROC | -0.001 ± 0.000 | 0/3 |
| 15000 | AUPRC | -0.012 ± 0.005 | 0/3 |
| 15000 | lab oracle FA/h | +12.629 ± 0.651 | 0/3 |
| 15000 | lab sens @val thr | +0.002 ± 0.004 | 2/3 |
| 15000 | lab FA/h @val thr | +7.615 ± 2.931 | 0/3 |
| 15000 | free-living alarms/day @val thr | +5.062 ± 1.578 | 0/3 |
| 15000 | sens @10/day | -0.029 ± 0.008 | 0/3 |
| 15000 | sens @25/day | -0.019 ± 0.005 | 0/3 |
| 15000 | sens @50/day | -0.015 ± 0.002 | 0/3 |
| 15000 | sens @100/day | -0.006 ± 0.004 | 1/3 |
| 15000 | free-living alarms/day @95% sens | +14.963 ± 7.146 | 0/3 |

## Per held-out fall dataset: Free-living alarms/day at 95% sensitivity

| windows per person | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| 0 | 219.1 | 103.8 | 277.3 | 239.3 |
| 1500 | 121.6 | 57.6 | 114.5 | 101.8 |
| 5000 | 124.1 | 52.3 | 88.7 | 165.1 |
| 15000 | 110.2 | 71.3 | 136.9 | 137.0 |

## Per held-out fall dataset: Sensitivity at 25 alarms/day

| windows per person | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| 0 | 0.634 | 0.755 | 0.464 | 0.529 |
| 1500 | 0.817 | 0.899 | 0.732 | 0.745 |
| 5000 | 0.771 | 0.895 | 0.783 | 0.709 |
| 15000 | 0.785 | 0.866 | 0.746 | 0.719 |
