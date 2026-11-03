# Comparison: baseline → baseline_augmented

Paired by protocol, fold and seed (same splits, same seeds). Δ = augmented − baseline. `wins` is the share of paired (fold, seed) runs where the augmented model is better (higher for sensitivity/specificity/AUROC/AUPRC, lower for false alarms). `ap` is AUPRC (average precision).

## Leave-one-dataset-out, macro over held-out datasets

| variant | metric | baseline | augmented | Δ | wins |
|---|---|---|---|---|---|
| acc stage1 | sensitivity | 0.843 | 0.852 | +0.009 | 50% |
| acc stage1 | specificity | 0.973 | 0.984 | +0.012 | 83% |
| acc stage1 | false_alarms_per_h | 49.418 | 28.566 | -20.852 | 83% |
| acc stage1 | auroc | 0.981 | 0.990 | +0.009 | 92% |
| acc stage1 | ap | 0.875 | 0.938 | +0.063 | 100% |
| acc stage1 | fa_h_at_95sens_oracle | 113.640 | 51.220 | -62.420 | 100% |
| acc stage2 | sensitivity | 0.824 | 0.849 | +0.025 | 67% |
| acc stage2 | specificity | 0.976 | 0.982 | +0.006 | 83% |
| acc stage2 | false_alarms_per_h | 42.512 | 31.738 | -10.775 | 83% |
| acc stage2 | auroc | 0.980 | 0.987 | +0.007 | 75% |
| acc stage2 | ap | 0.862 | 0.936 | +0.074 | 83% |
| acc stage2 | fa_h_at_95sens_oracle | 153.128 | 66.619 | -86.509 | 83% |
| imu stage1 | sensitivity | 0.831 | 0.864 | +0.033 | 67% |
| imu stage1 | specificity | 0.970 | 0.978 | +0.008 | 67% |
| imu stage1 | false_alarms_per_h | 53.634 | 39.133 | -14.501 | 67% |
| imu stage1 | auroc | 0.975 | 0.988 | +0.012 | 92% |
| imu stage1 | ap | 0.844 | 0.915 | +0.071 | 83% |
| imu stage1 | fa_h_at_95sens_oracle | 192.484 | 73.428 | -119.056 | 92% |
| imu stage2 | sensitivity | 0.842 | 0.861 | +0.019 | 67% |
| imu stage2 | specificity | 0.965 | 0.979 | +0.015 | 67% |
| imu stage2 | false_alarms_per_h | 63.826 | 37.394 | -26.433 | 67% |
| imu stage2 | auroc | 0.972 | 0.988 | +0.016 | 100% |
| imu stage2 | ap | 0.823 | 0.916 | +0.093 | 100% |
| imu stage2 | fa_h_at_95sens_oracle | 229.935 | 78.298 | -151.637 | 100% |

## Held-out: erciyes

| variant | metric | baseline | augmented | Δ | wins |
|---|---|---|---|---|---|
| acc stage1 | sensitivity | 0.831 | 0.827 | -0.005 | 67% |
| acc stage1 | specificity | 0.994 | 0.996 | +0.002 | 67% |
| acc stage1 | false_alarms_per_h | 10.326 | 7.389 | -2.937 | 67% |
| acc stage1 | auroc | 0.991 | 0.992 | +0.001 | 67% |
| acc stage1 | ap | 0.974 | 0.980 | +0.007 | 100% |
| acc stage1 | fa_h_at_95sens_oracle | 52.720 | 21.126 | -31.594 | 100% |
| acc stage2 | sensitivity | 0.834 | 0.839 | +0.005 | 67% |
| acc stage2 | specificity | 0.991 | 0.996 | +0.005 | 100% |
| acc stage2 | false_alarms_per_h | 16.152 | 6.631 | -9.521 | 100% |
| acc stage2 | auroc | 0.990 | 0.986 | -0.004 | 33% |
| acc stage2 | ap | 0.972 | 0.974 | +0.003 | 67% |
| acc stage2 | fa_h_at_95sens_oracle | 76.972 | 28.894 | -48.078 | 100% |
| imu stage1 | sensitivity | 0.840 | 0.881 | +0.041 | 67% |
| imu stage1 | specificity | 0.990 | 0.991 | +0.001 | 67% |
| imu stage1 | false_alarms_per_h | 18.284 | 16.768 | -1.516 | 67% |
| imu stage1 | auroc | 0.989 | 0.994 | +0.005 | 100% |
| imu stage1 | ap | 0.966 | 0.982 | +0.016 | 100% |
| imu stage1 | fa_h_at_95sens_oracle | 63.188 | 29.557 | -33.631 | 100% |
| imu stage2 | sensitivity | 0.851 | 0.890 | +0.039 | 67% |
| imu stage2 | specificity | 0.983 | 0.991 | +0.007 | 33% |
| imu stage2 | false_alarms_per_h | 29.841 | 16.673 | -13.168 | 33% |
| imu stage2 | auroc | 0.987 | 0.995 | +0.009 | 100% |
| imu stage2 | ap | 0.964 | 0.983 | +0.019 | 100% |
| imu stage2 | fa_h_at_95sens_oracle | 98.050 | 26.526 | -71.524 | 100% |

## Held-out: umafall

| variant | metric | baseline | augmented | Δ | wins |
|---|---|---|---|---|---|
| acc stage1 | sensitivity | 0.947 | 0.932 | -0.015 | 0% |
| acc stage1 | specificity | 0.936 | 0.974 | +0.037 | 100% |
| acc stage1 | false_alarms_per_h | 114.634 | 47.467 | -67.167 | 100% |
| acc stage1 | auroc | 0.977 | 0.990 | +0.013 | 100% |
| acc stage1 | ap | 0.768 | 0.902 | +0.134 | 100% |
| acc stage1 | fa_h_at_95sens_oracle | 100.188 | 54.034 | -46.154 | 100% |
| acc stage2 | sensitivity | 0.942 | 0.930 | -0.012 | 33% |
| acc stage2 | specificity | 0.951 | 0.975 | +0.024 | 100% |
| acc stage2 | false_alarms_per_h | 88.743 | 45.216 | -43.527 | 100% |
| acc stage2 | auroc | 0.980 | 0.990 | +0.010 | 100% |
| acc stage2 | ap | 0.800 | 0.904 | +0.104 | 100% |
| acc stage2 | fa_h_at_95sens_oracle | 105.629 | 57.411 | -48.218 | 67% |
| imu stage1 | sensitivity | 0.916 | 0.939 | +0.023 | 67% |
| imu stage1 | specificity | 0.929 | 0.961 | +0.032 | 100% |
| imu stage1 | false_alarms_per_h | 127.955 | 70.356 | -57.598 | 100% |
| imu stage1 | auroc | 0.966 | 0.985 | +0.019 | 100% |
| imu stage1 | ap | 0.676 | 0.843 | +0.167 | 100% |
| imu stage1 | fa_h_at_95sens_oracle | 175.235 | 78.424 | -96.811 | 100% |
| imu stage2 | sensitivity | 0.915 | 0.933 | +0.018 | 100% |
| imu stage2 | specificity | 0.926 | 0.953 | +0.027 | 100% |
| imu stage2 | false_alarms_per_h | 132.458 | 84.428 | -48.030 | 100% |
| imu stage2 | auroc | 0.964 | 0.982 | +0.018 | 100% |
| imu stage2 | ap | 0.656 | 0.808 | +0.152 | 100% |
| imu stage2 | fa_h_at_95sens_oracle | 182.364 | 91.182 | -91.182 | 100% |

## Held-out: upfall

| variant | metric | baseline | augmented | Δ | wins |
|---|---|---|---|---|---|
| acc stage1 | sensitivity | 0.748 | 0.786 | +0.038 | 67% |
| acc stage1 | specificity | 0.997 | 0.999 | +0.002 | 100% |
| acc stage1 | false_alarms_per_h | 5.830 | 1.943 | -3.887 | 100% |
| acc stage1 | auroc | 0.981 | 0.995 | +0.014 | 100% |
| acc stage1 | ap | 0.885 | 0.962 | +0.078 | 100% |
| acc stage1 | fa_h_at_95sens_oracle | 142.906 | 11.142 | -131.764 | 100% |
| acc stage2 | sensitivity | 0.750 | 0.778 | +0.029 | 67% |
| acc stage2 | specificity | 0.993 | 0.999 | +0.005 | 100% |
| acc stage2 | false_alarms_per_h | 11.920 | 2.073 | -9.847 | 100% |
| acc stage2 | auroc | 0.984 | 0.996 | +0.012 | 100% |
| acc stage2 | ap | 0.822 | 0.967 | +0.145 | 100% |
| acc stage2 | fa_h_at_95sens_oracle | 132.412 | 8.551 | -123.861 | 100% |
| imu stage1 | sensitivity | 0.780 | 0.778 | -0.002 | 33% |
| imu stage1 | specificity | 0.996 | 0.998 | +0.002 | 67% |
| imu stage1 | false_alarms_per_h | 6.608 | 2.721 | -3.887 | 67% |
| imu stage1 | auroc | 0.975 | 0.991 | +0.016 | 100% |
| imu stage1 | ap | 0.876 | 0.946 | +0.070 | 100% |
| imu stage1 | fa_h_at_95sens_oracle | 293.975 | 30.447 | -263.528 | 100% |
| imu stage2 | sensitivity | 0.797 | 0.783 | -0.014 | 33% |
| imu stage2 | specificity | 0.991 | 0.998 | +0.007 | 67% |
| imu stage2 | false_alarms_per_h | 16.325 | 3.887 | -12.438 | 67% |
| imu stage2 | auroc | 0.971 | 0.990 | +0.019 | 100% |
| imu stage2 | ap | 0.815 | 0.944 | +0.129 | 100% |
| imu stage2 | fa_h_at_95sens_oracle | 420.428 | 77.867 | -342.561 | 100% |

## Held-out: weda

| variant | metric | baseline | augmented | Δ | wins |
|---|---|---|---|---|---|
| acc stage1 | sensitivity | 0.846 | 0.862 | +0.016 | 67% |
| acc stage1 | specificity | 0.963 | 0.968 | +0.005 | 67% |
| acc stage1 | false_alarms_per_h | 66.880 | 57.463 | -9.417 | 67% |
| acc stage1 | auroc | 0.977 | 0.983 | +0.006 | 100% |
| acc stage1 | ap | 0.873 | 0.906 | +0.034 | 100% |
| acc stage1 | fa_h_at_95sens_oracle | 158.744 | 118.578 | -40.167 | 100% |
| acc stage2 | sensitivity | 0.772 | 0.849 | +0.077 | 100% |
| acc stage2 | specificity | 0.970 | 0.959 | -0.011 | 33% |
| acc stage2 | false_alarms_per_h | 53.235 | 73.030 | +19.795 | 33% |
| acc stage2 | auroc | 0.965 | 0.976 | +0.011 | 67% |
| acc stage2 | ap | 0.855 | 0.899 | +0.044 | 67% |
| acc stage2 | fa_h_at_95sens_oracle | 297.502 | 171.621 | -125.881 | 67% |
| imu stage1 | sensitivity | 0.787 | 0.856 | +0.069 | 100% |
| imu stage1 | specificity | 0.966 | 0.963 | -0.003 | 33% |
| imu stage1 | false_alarms_per_h | 61.691 | 66.688 | +4.997 | 33% |
| imu stage1 | auroc | 0.971 | 0.980 | +0.009 | 67% |
| imu stage1 | ap | 0.861 | 0.891 | +0.031 | 33% |
| imu stage1 | fa_h_at_95sens_oracle | 237.540 | 155.285 | -82.255 | 67% |
| imu stage2 | sensitivity | 0.805 | 0.840 | +0.035 | 67% |
| imu stage2 | specificity | 0.957 | 0.975 | +0.018 | 67% |
| imu stage2 | false_alarms_per_h | 76.682 | 44.587 | -32.095 | 67% |
| imu stage2 | auroc | 0.969 | 0.985 | +0.016 | 100% |
| imu stage2 | ap | 0.857 | 0.929 | +0.072 | 100% |
| imu stage2 | fa_h_at_95sens_oracle | 218.898 | 117.617 | -101.281 | 100% |

## Subject-wise 5-fold CV (pooled)

| variant | metric | baseline | augmented | Δ | wins |
|---|---|---|---|---|---|
| acc stage1 | sensitivity | 0.935 | 0.941 | +0.006 | 60% |
| acc stage1 | specificity | 0.990 | 0.991 | +0.001 | 60% |
| acc stage1 | false_alarms_per_h | 17.569 | 16.322 | -1.247 | 60% |
| acc stage1 | auroc | 0.995 | 0.997 | +0.002 | 87% |
| acc stage1 | ap | 0.975 | 0.985 | +0.010 | 93% |
| acc stage1 | fa_h_at_95sens_oracle | 19.435 | 17.717 | -1.718 | 80% |
| acc stage2 | sensitivity | 0.936 | 0.945 | +0.009 | 73% |
| acc stage2 | specificity | 0.990 | 0.991 | +0.001 | 60% |
| acc stage2 | false_alarms_per_h | 17.360 | 16.131 | -1.229 | 60% |
| acc stage2 | auroc | 0.995 | 0.997 | +0.002 | 87% |
| acc stage2 | ap | 0.977 | 0.985 | +0.008 | 93% |
| acc stage2 | fa_h_at_95sens_oracle | 21.288 | 15.958 | -5.330 | 80% |
| imu stage1 | sensitivity | 0.939 | 0.942 | +0.003 | 53% |
| imu stage1 | specificity | 0.990 | 0.992 | +0.002 | 67% |
| imu stage1 | false_alarms_per_h | 18.843 | 14.475 | -4.368 | 67% |
| imu stage1 | auroc | 0.993 | 0.997 | +0.004 | 93% |
| imu stage1 | ap | 0.971 | 0.985 | +0.014 | 93% |
| imu stage1 | fa_h_at_95sens_oracle | 22.699 | 16.783 | -5.916 | 73% |
| imu stage2 | sensitivity | 0.936 | 0.941 | +0.006 | 60% |
| imu stage2 | specificity | 0.988 | 0.992 | +0.004 | 60% |
| imu stage2 | false_alarms_per_h | 21.938 | 14.232 | -7.706 | 60% |
| imu stage2 | auroc | 0.994 | 0.997 | +0.003 | 73% |
| imu stage2 | ap | 0.971 | 0.987 | +0.015 | 87% |
| imu stage2 | fa_h_at_95sens_oracle | 28.400 | 17.063 | -11.337 | 73% |
