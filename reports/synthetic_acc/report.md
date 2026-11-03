# Does synthetic data help?

Seeds: 8, input: `acc`, stage-1 CNN, leave-one-dataset-out over the real datasets, runtime 73.0 min. Synthetic data: LAFAN1 mocap through a numpy virtual wrist IMU (`synth.py`, ADR 0002): 693 fall windows from 99 fall events ({'forward': 63, 'backward': 20, 'lateral': 16}) and 7320 ADL windows, 5 subjects. Each fall event contributes 7 windows with the impact at different positions, so the fall windows are far fewer independent samples than their count. Synthetic data is only ever in the training set; validation, threshold and test are real. Conditions are described in `synthetic_exp.py`. Cells are the macro average over held-out datasets, mean ± std over seeds. Sensitivity and false alarms/h use the validation-slice threshold; AUROC, AUPRC and oracle FA/h are threshold-free. Δ is the paired difference to the reference condition (`real`, or `real-noaug` for the `-noaug` rows) over seeds, with its standard error; wins = seeds in which the condition beats the reference on oracle FA/h. All real falls are simulated.

## Macro average over held-out datasets

| condition | AUROC | AUPRC | oracle FA/h at 95% sens | Δ oracle FA/h (± SE) | wins | sensitivity | false alarms/h |
|---|---|---|---|---|---|---|---|
| real | 0.991 ± 0.001 | 0.942 ± 0.010 | 53.9 ± 9.3 | - | - | 0.867 ± 0.023 | 28.9 ± 8.5 |
| mix-25 | 0.992 ± 0.002 | 0.951 ± 0.009 | 43.2 ± 10.3 | -10.7 ± 5.3 | 6/8 | 0.865 ± 0.037 | 26.6 ± 10.6 |
| mix-100 | 0.989 ± 0.002 | 0.935 ± 0.011 | 54.4 ± 10.6 | +0.5 ± 5.4 | 4/8 | 0.866 ± 0.035 | 34.9 ± 13.8 |
| mix-adl | 0.987 ± 0.002 | 0.940 ± 0.012 | 63.4 ± 14.3 | +9.5 ± 7.8 | 3/8 | 0.854 ± 0.053 | 27.7 ± 11.0 |
| mix-fall | 0.989 ± 0.002 | 0.930 ± 0.012 | 64.0 ± 8.2 | +10.1 ± 4.5 | 2/8 | 0.873 ± 0.020 | 40.3 ± 10.9 |
| synth-only | 0.915 ± 0.029 | 0.665 ± 0.064 | 687.0 ± 195.9 | +633.1 ± 70.6 | 0/8 | 0.908 ± 0.034 | 577.1 ± 205.0 |
| pretrain-ft | 0.989 ± 0.002 | 0.937 ± 0.010 | 55.3 ± 15.2 | +1.4 ± 7.5 | 4/8 | 0.867 ± 0.022 | 35.0 ± 7.4 |
| real-noaug | 0.981 ± 0.002 | 0.857 ± 0.022 | 115.7 ± 37.3 | - | - | 0.843 ± 0.028 | 52.2 ± 12.2 |
| mix-100-noaug | 0.982 ± 0.002 | 0.899 ± 0.013 | 132.6 ± 31.7 | +17.0 ± 21.4 | 3/8 | 0.833 ± 0.025 | 38.8 ± 7.9 |

## Per held-out dataset: oracle FA/h at 95% sens (mean over seeds)

| condition | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| real | 22.9 | 58.9 | 8.5 | 125.3 |
| mix-25 | 22.1 | 56.4 | 7.0 | 87.3 |
| mix-100 | 30.9 | 62.8 | 16.3 | 107.5 |
| mix-adl | 18.0 | 77.3 | 20.1 | 138.3 |
| mix-fall | 35.2 | 57.9 | 13.5 | 149.3 |
| synth-only | 476.2 | 443.1 | 1158.8 | 669.7 |
| pretrain-ft | 21.7 | 64.2 | 7.6 | 127.6 |
| real-noaug | 53.4 | 132.8 | 118.1 | 158.4 |
| mix-100-noaug | 67.0 | 116.5 | 226.9 | 120.1 |

## Per held-out dataset: AUPRC (mean over seeds)

| condition | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| real | 0.983 | 0.902 | 0.968 | 0.914 |
| mix-25 | 0.984 | 0.896 | 0.978 | 0.947 |
| mix-100 | 0.979 | 0.865 | 0.957 | 0.940 |
| mix-adl | 0.981 | 0.875 | 0.971 | 0.932 |
| mix-fall | 0.978 | 0.882 | 0.960 | 0.901 |
| synth-only | 0.881 | 0.566 | 0.487 | 0.726 |
| pretrain-ft | 0.982 | 0.894 | 0.965 | 0.906 |
| real-noaug | 0.975 | 0.747 | 0.830 | 0.877 |
| mix-100-noaug | 0.973 | 0.784 | 0.902 | 0.936 |

## Synthetic data quality

Peak magnitude per window, p10 / p50 / p90. Real datasets (range over the four):

| statistic | synthetic | real, per dataset |
|---|---|---|
| fall windows, accel (g) | [2.33, 4.23, 5.96] | erciyes [2.8, 4.5, 6.1]; umafall [3.1, 4.8, 6.2]; upfall [2.5, 3.6, 4.9]; weda [3.2, 4.3, 5.5] |
| ADL windows, accel (g) | [1.45, 2.5, 5.27] | erciyes [1.0, 1.1, 1.9]; umafall [1.1, 1.5, 4.2]; upfall [1.0, 1.1, 2.8]; weda [1.3, 1.6, 3.5] |
| fall windows, gyro (rad/s), synthetic p50 only | 11.3 | erciyes [4.6, 8.3, 13.0]; umafall [5.4, 6.6, 7.5]; upfall [4.4, 7.9, 11.9]; weda [6.2, 9.4, 15.7] |
