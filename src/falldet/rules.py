"""Decision threshold rules: scores in, threshold out (numpy only, no training).

* ``keep_falls``: the largest threshold that keeps a share of the given fall windows. On a subject-wise
  validation slice of the training data this is the ``val-slice`` rule; on target-device volunteers it is
  ``falls-pool``; on the test windows themselves it is the ``oracle`` (a diagnostic, not realisable).
* ``lodo_pooled`` / ``lodo_median`` / ``lodo_worst``: from out-of-dataset fall scores (each training
  dataset scored by a model that never saw it, ``runner.Fold.out_of_dataset_fall_scores``): pooled with
  every dataset weighted equally, or the median / lowest of the per-dataset thresholds.
* ``adl_budget``: the threshold whose false-alarm rate on ADL windows stays within a budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .windows import STRIDE_S

TARGET_SENSITIVITY = 0.95
RULE_NAMES = ("val-slice", "lodo-pooled", "lodo-median", "lodo-worst", "adl-budget", "falls-pool", "oracle")


def keep_falls(scores: np.ndarray, labels: np.ndarray, share: float = TARGET_SENSITIVITY) -> float:
    """Largest threshold at which at least ``share`` of the fall windows (``labels == 1``) score at or above it."""
    falls = np.asarray(scores)[np.asarray(labels) == 1]
    if len(falls) == 0:
        raise ValueError("no fall windows to keep")
    return float(np.quantile(falls, 1 - share, method="lower"))


def _groups(fall_scores: Mapping[str, np.ndarray] | Sequence[np.ndarray]) -> list[np.ndarray]:
    groups = list(fall_scores.values()) if isinstance(fall_scores, Mapping) else list(fall_scores)
    if not groups or any(len(g) == 0 for g in groups):
        raise ValueError("every dataset needs fall scores")
    return [np.asarray(g) for g in groups]


def lodo_pooled(fall_scores: Mapping[str, np.ndarray] | Sequence[np.ndarray], share: float = TARGET_SENSITIVITY) -> float:
    """``share`` of the out-of-dataset fall windows kept, every dataset carrying equal total weight."""
    groups = _groups(fall_scores)
    s = np.concatenate(groups)
    w = np.concatenate([np.full(len(g), 1 / len(g)) for g in groups])
    order = np.argsort(-s)
    cum = np.cumsum(w[order]) / w.sum()
    return float(s[order][np.argmax(cum >= share)])


def lodo_median(fall_scores: Mapping[str, np.ndarray] | Sequence[np.ndarray], share: float = TARGET_SENSITIVITY) -> float:
    """Median over datasets of each dataset's own ``keep_falls`` threshold."""
    return float(np.median([keep_falls(g, np.ones(len(g)), share) for g in _groups(fall_scores)]))


def lodo_worst(fall_scores: Mapping[str, np.ndarray] | Sequence[np.ndarray], share: float = TARGET_SENSITIVITY) -> float:
    """Lowest per-dataset threshold, so every training dataset keeps at least ``share`` of its falls."""
    return float(np.min([keep_falls(g, np.ones(len(g)), share) for g in _groups(fall_scores)]))


def adl_budget(adl_scores: np.ndarray, false_alarms_per_hour: float, stride_s: float = STRIDE_S) -> float:
    """Smallest threshold whose rate of firing ADL windows (one stride of exposure per window) stays
    within ``false_alarms_per_hour``."""
    rate = false_alarms_per_hour * stride_s / 3600
    return float(np.quantile(adl_scores, 1 - rate, method="higher"))
