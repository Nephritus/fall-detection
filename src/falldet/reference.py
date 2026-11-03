"""Threshold reference (GLOSSARY): free-fall dip followed by an impact peak on |accelerometer|."""

from __future__ import annotations

import itertools

import numpy as np

from .canonical import FS

DIPS_G = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)
PEAKS_G = (1.5, 2.0, 2.5, 3.0, 3.5, 3.9)
GAP_S = 1.0  # impact must follow the dip within this time


def _forward_max(mag: np.ndarray, gap: int) -> np.ndarray:
    """fwd[i] = max(mag[i : i + gap]) for every window (N, T)."""
    n, t = mag.shape
    pad = np.concatenate([mag, np.full((n, gap - 1), -np.inf)], axis=1)
    out = pad[:, :t].copy()
    for k in range(1, gap):
        np.maximum(out, pad[:, k : k + t], out=out)
    return out


class ThresholdReference:
    """Fires when |a| drops below ``dip`` and, within GAP_S, rises above ``peak``.

    The two levels are tuned on the training windows (maximising sensitivity + specificity - 1);
    the rule has a single operating point, so it has no score to re-threshold.
    """

    def __init__(self) -> None:
        self.dip: float | None = None
        self.peak: float | None = None

    @staticmethod
    def _prepare(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mag = np.linalg.norm(x[:, :, :3], axis=2)
        return mag, _forward_max(mag, round(GAP_S * FS))

    @staticmethod
    def _fire(mag, fwd, dip, peak) -> np.ndarray:
        return ((mag < dip) & (fwd > peak)).any(axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ThresholdReference":
        mag, fwd = self._prepare(x)
        best = (-2.0, None, None)
        for dip, peak in itertools.product(DIPS_G, PEAKS_G):
            pred = self._fire(mag, fwd, dip, peak)
            sens = pred[y == 1].mean()
            spec = 1 - pred[y == 0].mean()
            if sens + spec - 1 > best[0]:
                best = (sens + spec - 1, dip, peak)
        _, self.dip, self.peak = best
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        mag, fwd = self._prepare(x)
        return self._fire(mag, fwd, self.dip, self.peak).astype(int)
