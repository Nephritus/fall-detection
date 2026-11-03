# Augmentation ablation

Seeds: 5, input: `acc`, stage-1 CNN, leave-one-dataset-out. Runtime 56.1 min. Each cell is the macro average over the held-out datasets, then mean (± std) over seeds. Sensitivity and false alarms/h use the standard validation-slice threshold; AUROC, AUPRC and the oracle FA/h are threshold-free ranking measures, so judge transforms mainly by those. `none` and `all` correspond to the baseline and baseline_augmented stage-1 rows (agreeing only within seed noise: individual folds differ by up to ~2x between runs with the same seed; probably early stopping amplifying GPU nondeterminism, not verified). `only-X` turns on one transform; `without-X` turns on all but one. `wins vs none/all` is the share of seeds whose oracle FA/h is lower than that reference. Transform strengths are untuned (`augment.py`); all falls simulated.

## Macro average over held-out datasets

| config | AUROC | AUPRC | oracle FA/h at 95% sens | sensitivity | false alarms/h | wins vs none | wins vs all |
|---|---|---|---|---|---|---|---|
| none | 0.979 ± 0.003 | 0.846 ± 0.021 | 131.9 ± 29.9 | 0.841 ± 0.028 | 58.2 ± 17.4 | 0% | 0% |
| all | 0.992 ± 0.002 | 0.947 ± 0.009 | 49.5 ± 16.2 | 0.860 ± 0.032 | 27.8 ± 12.4 | 100% | 0% |
| only-rotation | 0.984 ± 0.001 | 0.892 ± 0.014 | 86.3 ± 16.2 | 0.836 ± 0.023 | 40.0 ± 13.5 | 100% | 0% |
| only-mirror | 0.986 ± 0.004 | 0.914 ± 0.017 | 72.3 ± 26.9 | 0.859 ± 0.027 | 36.4 ± 10.7 | 80% | 40% |
| only-gain | 0.981 ± 0.002 | 0.873 ± 0.015 | 144.3 ± 78.9 | 0.845 ± 0.028 | 59.7 ± 18.3 | 80% | 0% |
| only-stretch | 0.983 ± 0.003 | 0.886 ± 0.037 | 96.9 ± 32.2 | 0.840 ± 0.041 | 47.6 ± 24.2 | 80% | 0% |
| only-shift | 0.982 ± 0.005 | 0.889 ± 0.015 | 118.5 ± 47.9 | 0.825 ± 0.044 | 40.0 ± 8.5 | 60% | 0% |
| only-jitter | 0.979 ± 0.003 | 0.873 ± 0.017 | 144.7 ± 68.0 | 0.825 ± 0.025 | 41.9 ± 9.3 | 60% | 0% |
| without-rotation | 0.990 ± 0.002 | 0.935 ± 0.010 | 57.0 ± 14.9 | 0.862 ± 0.029 | 34.0 ± 10.5 | 100% | 20% |
| without-mirror | 0.987 ± 0.002 | 0.919 ± 0.015 | 74.9 ± 19.0 | 0.838 ± 0.021 | 35.3 ± 8.6 | 100% | 0% |
| without-gain | 0.988 ± 0.003 | 0.926 ± 0.006 | 64.8 ± 23.9 | 0.860 ± 0.017 | 32.8 ± 6.1 | 100% | 0% |
| without-stretch | 0.989 ± 0.002 | 0.929 ± 0.008 | 65.6 ± 17.7 | 0.850 ± 0.029 | 32.5 ± 6.2 | 100% | 20% |
| without-shift | 0.990 ± 0.002 | 0.933 ± 0.010 | 54.8 ± 10.1 | 0.862 ± 0.023 | 33.2 ± 10.6 | 100% | 20% |
| without-jitter | 0.991 ± 0.001 | 0.941 ± 0.005 | 48.8 ± 5.6 | 0.861 ± 0.028 | 29.4 ± 5.8 | 100% | 40% |

## Contribution of each transform (oracle FA/h at 95% sens, lower is better)

Both columns are changes in oracle FA/h; positive means the transform helps. `alone` = none − only-X (false alarms it removes by itself); `if removed` = without-X − all (false alarms added when it is dropped from the full set). Seed std of a macro value is 10–30 FA/h, so differences below about 15 are noise.

| transform | alone | if removed |
|---|---|---|
| rotation | +45.6 | +7.5 |
| mirror | +59.6 | +25.4 |
| gain | -12.3 | +15.3 |
| stretch | +35.0 | +16.1 |
| shift | +13.4 | +5.3 |
| jitter | -12.8 | -0.7 |

## Per held-out dataset: oracle FA/h at 95% sens (mean over seeds)

| config | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| all | 23.9 | 49.1 | 7.2 | 118.0 |
| none | 76.4 | 144.3 | 111.3 | 195.7 |
| only-gain | 81.7 | 134.4 | 191.4 | 169.6 |
| only-jitter | 82.8 | 111.9 | 191.5 | 192.8 |
| only-mirror | 25.5 | 77.3 | 60.3 | 126.1 |
| only-rotation | 36.7 | 107.4 | 56.7 | 144.5 |
| only-shift | 41.4 | 103.7 | 108.2 | 220.7 |
| only-stretch | 42.2 | 113.8 | 56.7 | 174.8 |
| without-gain | 21.0 | 63.8 | 39.4 | 135.0 |
| without-jitter | 20.2 | 51.1 | 8.0 | 116.0 |
| without-mirror | 24.2 | 97.9 | 25.3 | 152.0 |
| without-rotation | 21.7 | 68.6 | 7.1 | 130.6 |
| without-shift | 23.1 | 55.3 | 7.8 | 133.0 |
| without-stretch | 22.5 | 63.7 | 5.8 | 170.4 |
