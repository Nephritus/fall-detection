# Baseline results

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
| CNN imu stage1 | 0.840 ± 0.034 | 0.990 ± 0.002 | 18.284 ± 3.293 | 0.989 ± 0.001 | 0.966 ± 0.003 | 63.188 ± 8.786 |
| CNN imu stage2 | 0.851 ± 0.047 | 0.983 ± 0.013 | 29.841 ± 23.195 | 0.987 ± 0.003 | 0.964 ± 0.007 | 98.050 ± 46.361 |
| CNN acc stage1 | 0.831 ± 0.040 | 0.994 ± 0.003 | 10.326 ± 6.096 | 0.991 ± 0.001 | 0.974 ± 0.002 | 52.720 ± 8.595 |
| CNN acc stage2 | 0.834 ± 0.046 | 0.991 ± 0.007 | 16.152 ± 13.097 | 0.990 ± 0.001 | 0.972 ± 0.003 | 76.972 ± 25.372 |

### Held-out: umafall (AUPRC of a no-skill model = fall share = 0.103)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.997 ± 0.000 | 0.726 ± 0.000 | 492.495 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.916 ± 0.005 | 0.929 ± 0.007 | 127.955 ± 12.547 | 0.966 ± 0.002 | 0.676 ± 0.018 | 175.235 ± 25.110 |
| CNN imu stage2 | 0.915 ± 0.016 | 0.926 ± 0.007 | 132.458 ± 13.074 | 0.964 ± 0.002 | 0.656 ± 0.009 | 182.364 ± 36.616 |
| CNN acc stage1 | 0.947 ± 0.029 | 0.936 ± 0.024 | 114.634 ± 42.644 | 0.977 ± 0.003 | 0.768 ± 0.045 | 100.188 ± 8.756 |
| CNN acc stage2 | 0.942 ± 0.036 | 0.951 ± 0.011 | 88.743 ± 19.061 | 0.980 ± 0.007 | 0.800 ± 0.056 | 105.629 ± 45.787 |

### Held-out: upfall (AUPRC of a no-skill model = fall share = 0.043)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 1.000 ± 0.000 | 0.880 ± 0.000 | 216.886 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.780 ± 0.071 | 0.996 ± 0.002 | 6.608 ± 3.977 | 0.975 ± 0.006 | 0.876 ± 0.009 | 293.975 ± 83.639 |
| CNN imu stage2 | 0.797 ± 0.026 | 0.991 ± 0.009 | 16.325 ± 17.043 | 0.971 ± 0.003 | 0.815 ± 0.122 | 420.428 ± 120.908 |
| CNN acc stage1 | 0.748 ± 0.052 | 0.997 ± 0.001 | 5.830 ± 1.144 | 0.981 ± 0.001 | 0.885 ± 0.021 | 142.906 ± 30.850 |
| CNN acc stage2 | 0.750 ± 0.060 | 0.993 ± 0.003 | 11.920 ± 4.603 | 0.984 ± 0.006 | 0.822 ± 0.028 | 132.412 ± 124.934 |

### Held-out: weda (AUPRC of a no-skill model = fall share = 0.165)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.997 ± 0.000 | 0.702 ± 0.000 | 535.618 ± 0.000 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.787 ± 0.064 | 0.966 ± 0.023 | 61.691 ± 41.799 | 0.971 ± 0.013 | 0.861 ± 0.037 | 237.540 ± 133.667 |
| CNN imu stage2 | 0.805 ± 0.094 | 0.957 ± 0.025 | 76.682 ± 44.692 | 0.969 ± 0.007 | 0.857 ± 0.041 | 218.898 ± 38.717 |
| CNN acc stage1 | 0.846 ± 0.043 | 0.963 ± 0.014 | 66.880 ± 25.694 | 0.977 ± 0.001 | 0.873 ± 0.008 | 158.744 ± 4.762 |
| CNN acc stage2 | 0.772 ± 0.049 | 0.970 ± 0.009 | 53.235 ± 15.827 | 0.965 ± 0.009 | 0.855 ± 0.016 | 297.502 ± 106.989 |

### Macro average over held-out datasets

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.982 | 0.813 | 336.331 | nan | nan | nan |
| CNN imu stage1 | 0.831 | 0.970 | 53.634 | 0.975 | 0.844 | 192.484 |
| CNN imu stage2 | 0.842 | 0.965 | 63.826 | 0.972 | 0.823 | 229.935 |
| CNN acc stage1 | 0.843 | 0.973 | 49.418 | 0.981 | 0.875 | 113.640 |
| CNN acc stage2 | 0.824 | 0.976 | 42.512 | 0.980 | 0.862 | 153.128 |

## Subject-wise 5-fold CV (pooled datasets)

| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |
|---|---|---|---|---|---|---|
| threshold reference | 0.982 ± 0.033 | 0.855 ± 0.042 | 261.543 ± 75.574 | nan ± nan | nan ± nan | nan ± nan |
| CNN imu stage1 | 0.939 ± 0.027 | 0.990 ± 0.006 | 18.843 ± 11.437 | 0.993 ± 0.004 | 0.971 ± 0.016 | 22.699 ± 14.961 |
| CNN imu stage2 | 0.936 ± 0.026 | 0.988 ± 0.011 | 21.938 ± 19.655 | 0.994 ± 0.004 | 0.971 ± 0.021 | 28.400 ± 25.628 |
| CNN acc stage1 | 0.935 ± 0.028 | 0.990 ± 0.005 | 17.569 ± 9.087 | 0.995 ± 0.002 | 0.975 ± 0.015 | 19.435 ± 9.560 |
| CNN acc stage2 | 0.936 ± 0.026 | 0.990 ± 0.005 | 17.360 ± 9.609 | 0.995 ± 0.003 | 0.977 ± 0.015 | 21.288 ± 14.704 |

Threshold reference levels chosen on training windows: dip/peak (g) pairs seen [(1.5, 2.0), (1.5, 2.5)]. The chosen dip is ≥1 g, so the free-fall condition is not selective at 25 Hz wrist data and the rule reduces to an impact-magnitude test.

Hard-negative mining: after stage 1, training ADL windows scoring above the decision threshold (capped at 10% of ADL windows) are re-weighted ×3 and the model is retrained from scratch (`stage2`). Windows mined per fold: median 301, range 141–448.

Model: 3 conv blocks (32/64/128), 53281 parameters for 8 input channels (the accel-only variant is smaller).

## Hard negatives: ADL activities that fire most (leave-one-dataset-out)

**cnn-stage2** (activity code, share of that activity's windows flagged as fall):

| dataset | activity | false-alarm rate | windows |
|---|---|---|---|
| umafall | Aplausing | 0.38 | 756 |
| weda | D08 | 0.38 | 243 |
| umafall | Jogging | 0.32 | 504 |
| weda | D09 | 0.17 | 378 |
| umafall | Hopping | 0.14 | 954 |
| erciyes | 815 | 0.12 | 2202 |
| weda | D02 | 0.11 | 957 |
| weda | D07 | 0.07 | 411 |
| erciyes | 812 | 0.05 | 2046 |
| weda | D06 | 0.04 | 732 |

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
| erciyes | 903 | 0.28 | 474 |
| upfall | A2 | 0.51 | 123 |
| weda | F08 | 0.62 | 282 |
| erciyes | 916 | 0.65 | 486 |
| erciyes | 920 | 0.71 | 468 |
| erciyes | 917 | 0.72 | 510 |
| weda | F05 | 0.78 | 237 |
| weda | F03 | 0.80 | 222 |

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
