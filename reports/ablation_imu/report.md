# Augmentation ablation

Seeds: 5, input: `imu`, stage-1 CNN, leave-one-dataset-out. Runtime 52.5 min. Each cell is the macro average over the held-out datasets, then mean (± std) over seeds. Sensitivity and false alarms/h use the standard validation-slice threshold; AUROC, AUPRC and the oracle FA/h are threshold-free ranking measures, so judge transforms mainly by those. `none` and `all` correspond to the baseline and baseline_augmented stage-1 rows (agreeing only within seed noise: individual folds differ by up to ~2x between runs with the same seed; probably early stopping amplifying GPU nondeterminism, not verified). `only-X` turns on one transform; `without-X` turns on all but one. `wins vs none/all` is the share of seeds whose oracle FA/h is lower than that reference. Transform strengths are untuned (`augment.py`); all falls simulated.

## Macro average over held-out datasets

| config | AUROC | AUPRC | oracle FA/h at 95% sens | sensitivity | false alarms/h | wins vs none | wins vs all |
|---|---|---|---|---|---|---|---|
| none | 0.976 ± 0.006 | 0.849 ± 0.038 | 153.8 ± 54.7 | 0.825 ± 0.034 | 52.3 ± 17.8 | 0% | 0% |
| all | 0.989 ± 0.002 | 0.921 ± 0.016 | 65.1 ± 16.8 | 0.849 ± 0.011 | 35.3 ± 10.0 | 100% | 0% |
| only-rotation | 0.981 ± 0.005 | 0.873 ± 0.037 | 141.4 ± 51.1 | 0.843 ± 0.021 | 47.4 ± 16.6 | 60% | 0% |
| only-mirror | 0.985 ± 0.002 | 0.888 ± 0.017 | 90.4 ± 21.1 | 0.854 ± 0.019 | 46.5 ± 10.4 | 100% | 0% |
| only-gain | 0.979 ± 0.005 | 0.849 ± 0.044 | 130.8 ± 50.5 | 0.842 ± 0.025 | 53.1 ± 14.4 | 80% | 0% |
| only-stretch | 0.979 ± 0.005 | 0.866 ± 0.029 | 169.7 ± 78.8 | 0.834 ± 0.022 | 48.8 ± 14.0 | 60% | 0% |
| only-shift | 0.980 ± 0.003 | 0.861 ± 0.023 | 123.7 ± 31.3 | 0.838 ± 0.037 | 48.5 ± 9.2 | 60% | 0% |
| only-jitter | 0.976 ± 0.004 | 0.841 ± 0.033 | 173.9 ± 54.5 | 0.836 ± 0.023 | 61.3 ± 15.6 | 20% | 0% |
| without-rotation | 0.986 ± 0.003 | 0.908 ± 0.013 | 90.0 ± 35.4 | 0.851 ± 0.021 | 42.6 ± 14.2 | 80% | 20% |
| without-mirror | 0.985 ± 0.002 | 0.902 ± 0.013 | 83.0 ± 14.5 | 0.835 ± 0.026 | 36.1 ± 10.8 | 100% | 40% |
| without-gain | 0.988 ± 0.001 | 0.922 ± 0.004 | 70.0 ± 9.0 | 0.867 ± 0.024 | 36.2 ± 8.1 | 100% | 40% |
| without-stretch | 0.988 ± 0.001 | 0.923 ± 0.010 | 71.9 ± 15.0 | 0.849 ± 0.023 | 33.8 ± 5.1 | 100% | 40% |
| without-shift | 0.988 ± 0.001 | 0.921 ± 0.009 | 76.6 ± 19.9 | 0.841 ± 0.026 | 30.3 ± 4.8 | 100% | 0% |
| without-jitter | 0.987 ± 0.003 | 0.917 ± 0.005 | 96.7 ± 47.5 | 0.845 ± 0.020 | 34.2 ± 10.4 | 80% | 0% |

## Contribution of each transform (oracle FA/h at 95% sens, lower is better)

Both columns are changes in oracle FA/h; positive means the transform helps. `alone` = none − only-X (false alarms it removes by itself); `if removed` = without-X − all (false alarms added when it is dropped from the full set). Seed std of a macro value is 10–30 FA/h, so differences below about 15 are noise.

| transform | alone | if removed |
|---|---|---|
| rotation | +12.4 | +24.8 |
| mirror | +63.3 | +17.9 |
| gain | +23.0 | +4.9 |
| stretch | -16.0 | +6.8 |
| shift | +30.1 | +11.5 |
| jitter | -20.1 | +31.6 |

## Per held-out dataset: oracle FA/h at 95% sens (mean over seeds)

| config | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| all | 29.7 | 84.5 | 15.5 | 130.8 |
| none | 67.1 | 168.3 | 122.7 | 256.9 |
| only-gain | 90.5 | 165.9 | 81.3 | 185.3 |
| only-jitter | 115.6 | 166.7 | 194.7 | 218.5 |
| only-mirror | 37.6 | 109.5 | 82.3 | 132.3 |
| only-rotation | 53.6 | 150.5 | 188.0 | 173.4 |
| only-shift | 75.0 | 158.8 | 111.2 | 149.8 |
| only-stretch | 72.8 | 178.3 | 257.4 | 170.5 |
| without-gain | 27.7 | 77.8 | 31.1 | 143.4 |
| without-jitter | 27.3 | 73.8 | 100.2 | 185.5 |
| without-mirror | 34.6 | 121.2 | 17.5 | 158.7 |
| without-rotation | 32.6 | 96.2 | 83.6 | 147.4 |
| without-shift | 29.3 | 85.3 | 89.2 | 102.6 |
| without-stretch | 28.0 | 77.8 | 53.9 | 127.9 |
