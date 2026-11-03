# Baseline results (with augmentation)

Training windows are augmented on the fly (`augment.py`): Augment(p=0.5, roll_deg=30.0, tilt_deg=10.0, gain=0.1, stretch=0.1, shift_s=0.2, jitter_acc_g=0.02, jitter_gyro=0.05, enabled=('rotation', 'mirror', 'gain', 'stretch', 'shift', 'jitter')). Validation and test windows are untouched.

Seeds: 3. Runtime not recorded (results converted from an earlier run). Values are mean ± std over seeds (and over folds where noted). Sensitivity is at the threshold that gives ≥95% sensitivity on a subject-wise validation slice of the training data; the threshold reference has one fixed operating point. False alarms/h counts false-alarm windows while replaying held-out ADL windows (one stride = 2 s of replay), so overlapping windows of one event count more than once.

**Read these numbers as optimistic.** Every fall is a simulated fall (mattress, mostly young volunteers); the published literature reports far worse results on real falls.

**FA/h at 95% sens (oracle)** sets the threshold on the held-out scores themselves, so it measures ranking quality independent of threshold transfer across datasets; it is a diagnostic, never used to choose a model. The threshold reference has no score, so it is compared through its single operating point.

## Data

| dataset | subjects | fall recordings | ADL recordings | fall windows | ADL windows | impact source |
|---|---|---|---|---|---|---|
| weda | 25 | 350 | 619 | 615 | 3122 | {'annotation': 350} |
| upfall | 12 | 180 | 216 | 209 | 4631 | {'annotation': 180} |
| erciyes | 17 | 1836 | 1481 | 3203 | 12667 | {'peak': 1836} |
| umafall | 18 | 202 | 533 | 367 | 3198 | {'peak': 202} |

Excluded subjects (arm not still when standing, so mounting could not be aligned): {'upfall': {'upfall:S02': 0.64, 'upfall:S08': 0.63, 'upfall:S09': 0.68, 'upfall:S10': 0.71, 'upfall:S14': 0.67}, 'umafall': {'umafall:16': 0.66}}.

**Magnitude-peak fallback** was used for the impact moment in 2038 of 2568 fall recordings (all of Erciyes: it has no fall annotations). Fall-recording windows excluded: 12326 (+599 edge-ambiguous).

## Leave-one-dataset-out (headline)

### Held-out: erciyes (AUPRC of a no-skill model = fall share = 0.202)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.936 ± 0.000 | 0.944 ± 0.000 | 100.324 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.881 ± 0.055 | 0.991 ± 0.003 | 16.768 ± 5.931 | 0.994 ± 0.001 | 0.982 ± 0.002 | 29.557 ± 7.584 |
| CNN imu stage2 | 0.890 ± 0.021 | 0.991 ± 0.001 | 16.673 ± 2.221 | 0.995 ± 0.000 | 0.983 ± 0.002 | 26.526 ± 5.370 |
| CNN acc stage1 | 0.827 ± 0.080 | 0.996 ± 0.002 | 7.389 ± 3.647 | 0.992 ± 0.003 | 0.980 ± 0.002 | 21.126 ± 3.079 |
| CNN acc stage2 | 0.839 ± 0.023 | 0.996 ± 0.002 | 6.631 ± 3.651 | 0.986 ± 0.004 | 0.974 ± 0.002 | 28.894 ± 2.647 |

### Held-out: umafall (AUPRC of a no-skill model = fall share = 0.103)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.997 ± 0.000 | 0.726 ± 0.000 | 492.495 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.939 ± 0.016 | 0.961 ± 0.004 | 70.356 ± 7.223 | 0.985 ± 0.001 | 0.843 ± 0.027 | 78.424 ± 9.752 |
| CNN imu stage2 | 0.933 ± 0.023 | 0.953 ± 0.004 | 84.428 ± 7.164 | 0.982 ± 0.004 | 0.808 ± 0.033 | 91.182 ± 12.594 |
| CNN acc stage1 | 0.932 ± 0.017 | 0.974 ± 0.007 | 47.467 ± 12.334 | 0.990 ± 0.001 | 0.902 ± 0.015 | 54.034 ± 7.396 |
| CNN acc stage2 | 0.930 ± 0.016 | 0.975 ± 0.006 | 45.216 ± 10.573 | 0.990 ± 0.000 | 0.904 ± 0.011 | 57.411 ± 11.599 |

### Held-out: upfall (AUPRC of a no-skill model = fall share = 0.043)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 1.000 ± 0.000 | 0.880 ± 0.000 | 216.886 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.778 ± 0.016 | 0.998 ± 0.001 | 2.721 ± 1.144 | 0.991 ± 0.001 | 0.946 ± 0.008 | 30.447 ± 8.716 |
| CNN imu stage2 | 0.783 ± 0.031 | 0.998 ± 0.002 | 3.887 ± 2.767 | 0.990 ± 0.007 | 0.944 ± 0.024 | 77.867 ± 93.426 |
| CNN acc stage1 | 0.786 ± 0.069 | 0.999 ± 0.000 | 1.943 ± 0.317 | 0.995 ± 0.001 | 0.962 ± 0.006 | 11.142 ± 4.492 |
| CNN acc stage2 | 0.778 ± 0.043 | 0.999 ± 0.000 | 2.073 ± 0.485 | 0.996 ± 0.001 | 0.967 ± 0.006 | 8.551 ± 2.598 |

### Held-out: weda (AUPRC of a no-skill model = fall share = 0.165)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.997 ± 0.000 | 0.702 ± 0.000 | 535.618 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.856 ± 0.029 | 0.963 ± 0.006 | 66.688 ± 11.389 | 0.980 ± 0.003 | 0.891 ± 0.018 | 155.285 ± 18.619 |
| CNN imu stage2 | 0.840 ± 0.039 | 0.975 ± 0.006 | 44.587 ± 11.233 | 0.985 ± 0.002 | 0.929 ± 0.011 | 117.617 ± 8.474 |
| CNN acc stage1 | 0.862 ± 0.043 | 0.968 ± 0.012 | 57.463 ± 22.112 | 0.983 ± 0.003 | 0.906 ± 0.017 | 118.578 ± 16.458 |
| CNN acc stage2 | 0.849 ± 0.057 | 0.959 ± 0.027 | 73.030 ± 47.953 | 0.976 ± 0.012 | 0.899 ± 0.039 | 171.621 ± 80.200 |

### Macro average over held-out datasets

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.982 | 0.813 | 336.331 | nan | nan | nan |
| CNN imu stage1 | 0.864 | 0.978 | 39.133 | 0.988 | 0.915 | 73.428 |
| CNN imu stage2 | 0.861 | 0.979 | 37.394 | 0.988 | 0.916 | 78.298 |
| CNN acc stage1 | 0.852 | 0.984 | 28.566 | 0.990 | 0.938 | 51.220 |
| CNN acc stage2 | 0.849 | 0.982 | 31.738 | 0.987 | 0.936 | 66.619 |

## Subject-wise 5-fold CV (pooled datasets)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.982 ± 0.033 | 0.855 ± 0.042 | 261.543 ± 75.574 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.942 ± 0.016 | 0.992 ± 0.004 | 14.475 ± 7.227 | 0.997 ± 0.001 | 0.985 ± 0.009 | 16.783 ± 8.738 |
| CNN imu stage2 | 0.941 ± 0.018 | 0.992 ± 0.003 | 14.232 ± 5.762 | 0.997 ± 0.001 | 0.987 ± 0.007 | 17.063 ± 9.980 |
| CNN acc stage1 | 0.941 ± 0.029 | 0.991 ± 0.005 | 16.322 ± 9.832 | 0.997 ± 0.002 | 0.985 ± 0.009 | 17.717 ± 11.234 |
| CNN acc stage2 | 0.945 ± 0.024 | 0.991 ± 0.005 | 16.131 ± 8.420 | 0.997 ± 0.002 | 0.985 ± 0.008 | 15.958 ± 10.136 |

Threshold reference levels chosen on training windows: dip/peak (g) pairs seen [(1.5, 2.0), (1.5, 2.5)]. The chosen dip is ≥1 g, so the free-fall condition is not selective at 25 Hz wrist data and the rule reduces to an impact-magnitude test.

Hard-negative mining: after stage 1, training ADL windows scoring above the decision threshold (capped at 10% of ADL windows) are re-weighted ×3 and the model is retrained from scratch (`stage2`). Windows mined per fold: median 152, range 85–243.

Model: 3 conv blocks (32/64/128), 53281 parameters for 8 input channels (the accel-only variant is smaller).

## Hard negatives: ADL activities that fire most (leave-one-dataset-out)

**cnn-stage2** (activity code, share of that activity's windows flagged as fall):

| dataset | activity | false-alarm rate | windows |
|---|---|---|---|
| weda | D08 | 0.34 | 243 |
| umafall | Aplausing | 0.29 | 756 |
| umafall | Jogging | 0.15 | 504 |
| umafall | Hopping | 0.10 | 954 |
| weda | D09 | 0.08 | 378 |
| erciyes | 815 | 0.07 | 2202 |
| erciyes | 816 | 0.05 | 2415 |
| weda | D03 | 0.03 | 1995 |
| weda | D06 | 0.03 | 732 |
| umafall | HandsUp | 0.03 | 774 |

**threshold** (activity code, share of that activity's windows flagged as fall):

| dataset | activity | false-alarm rate | windows |
|---|---|---|---|
| upfall | A10 | 1.00 | 1404 |
| weda | D02 | 0.99 | 957 |
| umafall | Jogging | 0.94 | 504 |
| umafall | Hopping | 0.93 | 954 |
| weda | D08 | 0.80 | 243 |
| umafall | Aplausing | 0.65 | 756 |
| weda | D07 | 0.63 | 411 |
| weda | D09 | 0.50 | 378 |
| erciyes | 803 | 0.47 | 2526 |
| umafall | GoDownstairs | 0.42 | 360 |

## Missed falls by fall type (leave-one-dataset-out)

**cnn-stage2** (lowest detection rate):

| dataset | activity | detected | windows |
|---|---|---|---|
| erciyes | 903 | 0.47 | 474 |
| upfall | A2 | 0.55 | 123 |
| erciyes | 917 | 0.63 | 510 |
| weda | F08 | 0.68 | 282 |
| erciyes | 920 | 0.69 | 468 |
| upfall | A4 | 0.73 | 117 |
| upfall | A5 | 0.78 | 129 |
| weda | F05 | 0.79 | 237 |

**threshold** (lowest detection rate):

| dataset | activity | detected | windows |
|---|---|---|---|
| erciyes | 917 | 0.68 | 510 |
| erciyes | 920 | 0.78 | 468 |
| erciyes | 919 | 0.84 | 492 |
| erciyes | 916 | 0.88 | 486 |
| erciyes | 915 | 0.89 | 483 |
| erciyes | 909 | 0.89 | 471 |
| erciyes | 912 | 0.94 | 468 |
| erciyes | 910 | 0.94 | 477 |
