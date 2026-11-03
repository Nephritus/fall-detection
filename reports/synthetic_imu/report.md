# Does synthetic data help?

Seeds: 8, input: `imu`, stage-1 CNN, leave-one-dataset-out over the real datasets, runtime 59.5 min. Synthetic data: LAFAN1 mocap through a numpy virtual wrist IMU (`synth.py`, ADR 0002): 693 fall windows from 99 fall events ({'forward': 63, 'backward': 20, 'lateral': 16}) and 7320 ADL windows, 5 subjects. Each fall event contributes 7 windows with the impact at different positions, so the fall windows are far fewer independent samples than their count. Synthetic data is only ever in the training set; validation, threshold and test are real. Conditions are described in `synthetic_exp.py`. Cells are the macro average over held-out datasets, mean ± std over seeds. Sensitivity and false alarms/h use the validation-slice threshold; AUROC, AUPRC and oracle FA/h are threshold-free. Δ is the paired difference to the reference condition (`real`, or `real-noaug` for the `-noaug` rows) over seeds, with its standard error; wins = seeds in which the condition beats the reference on oracle FA/h. All real falls are simulated.

## Macro average over held-out datasets

| condition | AUROC | AUPRC | oracle FA/h at 95% sens | Δ oracle FA/h (± SE) | wins | sensitivity | false alarms/h |
|---|---|---|---|---|---|---|---|
| real | 0.989 ± 0.002 | 0.922 ± 0.010 | 59.4 ± 10.7 | - | - | 0.855 ± 0.017 | 34.5 ± 6.2 |
| mix-25 | 0.990 ± 0.003 | 0.938 ± 0.012 | 65.0 ± 19.3 | +5.5 ± 7.9 | 2/8 | 0.847 ± 0.018 | 28.9 ± 8.0 |
| mix-100 | 0.988 ± 0.002 | 0.929 ± 0.006 | 64.5 ± 16.3 | +5.1 ± 6.9 | 3/8 | 0.849 ± 0.032 | 28.7 ± 6.9 |
| mix-adl | 0.987 ± 0.003 | 0.937 ± 0.012 | 83.7 ± 33.7 | +24.2 ± 12.7 | 2/8 | 0.817 ± 0.036 | 24.0 ± 7.1 |
| mix-fall | 0.984 ± 0.004 | 0.891 ± 0.018 | 108.5 ± 50.2 | +49.1 ± 18.4 | 1/8 | 0.876 ± 0.018 | 55.3 ± 15.2 |
| synth-only | 0.885 ± 0.017 | 0.576 ± 0.035 | 931.3 ± 112.4 | +871.9 ± 41.8 | 0/8 | 0.928 ± 0.030 | 877.4 ± 211.9 |
| pretrain-ft | 0.987 ± 0.004 | 0.913 ± 0.015 | 87.5 ± 54.6 | +28.1 ± 18.2 | 2/8 | 0.855 ± 0.015 | 36.0 ± 5.7 |
| real-noaug | 0.975 ± 0.006 | 0.841 ± 0.024 | 190.6 ± 68.9 | - | - | 0.838 ± 0.031 | 60.7 ± 23.3 |
| mix-100-noaug | 0.974 ± 0.005 | 0.861 ± 0.012 | 244.0 ± 110.8 | +53.4 ± 50.2 | 2/8 | 0.818 ± 0.023 | 47.1 ± 9.1 |

## Per held-out dataset: oracle FA/h at 95% sens (mean over seeds)

| condition | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| real | 32.5 | 77.3 | 13.1 | 114.9 |
| mix-25 | 25.4 | 62.2 | 29.5 | 142.7 |
| mix-100 | 34.3 | 68.7 | 23.1 | 131.7 |
| mix-adl | 15.7 | 75.6 | 54.1 | 189.3 |
| mix-fall | 60.6 | 76.8 | 79.1 | 217.4 |
| synth-only | 866.0 | 748.2 | 1218.0 | 892.9 |
| pretrain-ft | 29.5 | 82.2 | 114.0 | 124.2 |
| real-noaug | 95.1 | 181.8 | 271.5 | 213.8 |
| mix-100-noaug | 262.6 | 146.8 | 361.1 | 205.4 |

## Per held-out dataset: AUPRC (mean over seeds)

| condition | erciyes | umafall | upfall | weda |
|---|---|---|---|---|
| real | 0.980 | 0.829 | 0.960 | 0.922 |
| mix-25 | 0.984 | 0.873 | 0.969 | 0.926 |
| mix-100 | 0.979 | 0.848 | 0.963 | 0.928 |
| mix-adl | 0.986 | 0.879 | 0.954 | 0.929 |
| mix-fall | 0.963 | 0.798 | 0.930 | 0.872 |
| synth-only | 0.809 | 0.437 | 0.423 | 0.636 |
| pretrain-ft | 0.982 | 0.803 | 0.950 | 0.917 |
| real-noaug | 0.960 | 0.686 | 0.838 | 0.882 |
| mix-100-noaug | 0.955 | 0.747 | 0.835 | 0.909 |

## Synthetic data quality

Peak magnitude per window, p10 / p50 / p90. Real datasets (range over the four):

| statistic | synthetic | real, per dataset |
|---|---|---|
| fall windows, accel (g) | [2.33, 4.23, 5.96] | erciyes [2.8, 4.5, 6.1]; umafall [3.1, 4.8, 6.2]; upfall [2.5, 3.6, 4.9]; weda [3.2, 4.3, 5.5] |
| ADL windows, accel (g) | [1.45, 2.5, 5.27] | erciyes [1.0, 1.1, 1.9]; umafall [1.1, 1.5, 4.2]; upfall [1.0, 1.1, 2.8]; weda [1.3, 1.6, 3.5] |
| fall windows, gyro (rad/s), synthetic p50 only | 11.3 | erciyes [4.6, 8.3, 13.0]; umafall [5.4, 6.6, 7.5]; upfall [4.4, 7.9, 11.9]; weda [6.2, 9.4, 15.7] |
