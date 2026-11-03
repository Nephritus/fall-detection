"""Canonical format: the vendor-independent representation all signals are converted to.

Units: accelerometer in g (gravity included), gyroscope in rad/s.
Rate: FS Hz uniform grid (ADR 0001). Axes: the wrist frame of GLOSSARY.md, left-wrist convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 25  # Hz, ADR 0001
G = 9.80665  # m/s^2
ACC_CLIP_G = 4.0  # narrowest source range (Fitbit/WEDA-FALL); removes a dataset signature
LOWPASS_HZ = 0.4 * FS  # anti-alias cutoff before decimating to FS

# Specific-force direction (g) while the arm hangs at the side and the wearer walks, left wrist,
# measured on WEDA-FALL (native axes already match the wrist frame). Datasets whose mounting
# differs are rotated so their own walking/standing mean lands on this direction.
REFERENCE_HANGING_G = np.array([-0.78, -0.28, 0.25])
MIN_HANGING_NORM_G = 0.85  # a still, hanging arm reads ~1 g; lower means the arm was not hanging


@dataclass
class Recording:
    dataset: str
    subject: str  # globally unique, e.g. "weda:U01"
    trial: str
    activity: str  # the dataset's activity code
    is_fall: bool
    acc: np.ndarray  # (n, 3) g, wrist frame, FS grid
    gyro: np.ndarray  # (n, 3) rad/s, wrist frame, FS grid
    impact_s: float | None  # seconds from recording start (fall recordings only)
    impact_source: str  # "annotation" | "peak" | ""


def rot_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking direction ``a`` onto direction ``b`` (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        raise ValueError("antiparallel vectors: rotation is ambiguous")
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def mirror_right_to_left(acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a right-wrist recording (already in the right-wrist wrist frame) to the left wrist.

    Reflecting the body across the sagittal plane negates y for polar vectors; the gyroscope is an
    axial vector, so its x and z components flip instead.
    """
    return acc * _MIRROR_ACC, gyro * _MIRROR_GYRO


_MIRROR_ACC = np.array([1.0, -1.0, 1.0])  # polar vector: lateral axis flips
_MIRROR_GYRO = np.array([-1.0, 1.0, -1.0])  # axial vector: the other two flip


@dataclass(frozen=True)
class MountingAlignment:
    """How one wearer's signal maps into the wrist frame (GLOSSARY: Mounting alignment)."""

    wrist: str  # "left" | "right"
    norm_g: float  # magnitude of the reference mean; a hanging, still arm reads about 1 g
    rotation: np.ndarray | None  # None when the wearer is excluded

    @property
    def excluded(self) -> bool:
        return self.rotation is None

    def apply(self, acc: np.ndarray, gyro: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Rotate into the wrist frame and, for a right wrist, mirror to the left-wrist convention."""
        if self.rotation is None:
            raise ValueError(f"excluded wearer (reference mean {self.norm_g:.2f} g) has no alignment")
        acc = acc @ self.rotation.T
        gyro = None if gyro is None else gyro @ self.rotation.T
        if self.wrist == "right":
            acc = acc * _MIRROR_ACC
            gyro = None if gyro is None else gyro * _MIRROR_GYRO
        return acc, gyro


def align_mounting(reference_acc: np.ndarray, wrist: str, min_norm_g: float = MIN_HANGING_NORM_G) -> MountingAlignment:
    """Mounting alignment of one wearer from accelerometer samples (g) of a reference activity during which
    the arm hangs (standing or walking): the mean specific force is rotated onto REFERENCE_HANGING_G (its
    right-wrist mirror for a right wrist). A mean weaker than ``min_norm_g`` shows no reliable direction,
    so the wearer is excluded instead."""
    if wrist not in ("left", "right"):
        raise ValueError(f"wrist must be 'left' or 'right', not {wrist!r}")
    if len(reference_acc) == 0:
        raise ValueError("no reference samples")
    mean = np.asarray(reference_acc, float).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < min_norm_g:
        return MountingAlignment(wrist, norm, None)
    target = REFERENCE_HANGING_G * (_MIRROR_ACC if wrist == "right" else 1.0)
    return MountingAlignment(wrist, norm, rot_align(mean, target))


def to_grid(t: np.ndarray, x: np.ndarray, fs_native: float, grid: np.ndarray) -> np.ndarray:
    """Resample ``x`` (n, 3) sampled at times ``t`` onto ``grid``.

    When the source is faster than FS the signal is low-passed first (zero-phase) so decimation
    does not alias. ``t`` must be near-uniform in that case.
    """
    if fs_native > FS * 1.01:
        sos = butter(4, LOWPASS_HZ, btype="low", fs=fs_native, output="sos")
        x = sosfiltfilt(sos, x, axis=0)
    return np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], axis=1)


def find_impact(acc_g: np.ndarray, interval_s: tuple[float, float] | None) -> tuple[float, str]:
    """Impact moment (seconds from start): accelerometer-magnitude peak.

    Searches inside ``interval_s`` when the dataset annotates the fall ("annotation"), else over the
    whole recording ("peak", the fallback that is counted in the report).
    """
    mag = np.linalg.norm(acc_g, axis=1)
    if interval_s is not None:
        lo = max(int(np.floor(interval_s[0] * FS)), 0)
        hi = min(int(np.ceil(interval_s[1] * FS)) + 1, len(mag))
        if hi - lo >= 2:
            return (lo + int(np.argmax(mag[lo:hi]))) / FS, "annotation"
    return int(np.argmax(mag)) / FS, "peak"
