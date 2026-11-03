"""Motion capture to virtual wrist IMU (ADR 0002): BVH forward kinematics + finite differences.

Specific force is ``f = R^T (a - g)`` for the sensor's world acceleration ``a`` and gravity ``g``;
angular rate comes from the relative rotation of consecutive frames. Positions are low-passed at the
canonical anti-alias cutoff before differentiation so mocap jitter does not become fake impacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from .canonical import FS, G, LOWPASS_HZ

_AXES = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


@dataclass
class Bvh:
    names: list[str]
    parent: list[int]
    offset: np.ndarray  # (J, 3) rest offsets from the parent joint, in file units
    end_offset: dict[str, np.ndarray]  # first End Site offset of each leaf-parent joint
    pos: np.ndarray  # (F, 3) root translation, file units
    rot: np.ndarray  # (F, J, 3, 3) local rotation matrices
    fps: float


def parse_bvh(path: Path) -> Bvh:
    toks = path.read_text().split()
    names: list[str] = []
    parent: list[int] = []
    offsets: list[list[float]] = []
    order: list[str] = []  # rotation channel order per joint (e.g. "zyx")
    end_offset: dict[str, np.ndarray] = {}
    stack: list[int] = []
    i, cur, pending_end = 0, -1, False
    while toks[i] != "MOTION":
        t = toks[i]
        if t in ("ROOT", "JOINT"):
            names.append(toks[i + 1])
            parent.append(stack[-1] if stack else -1)
            cur = len(names) - 1
            i += 2
        elif t == "End":
            pending_end = True
            i += 2  # "End Site"
        elif t == "OFFSET":
            v = [float(x) for x in toks[i + 1 : i + 4]]
            if pending_end:
                end_offset[names[stack[-1]]] = np.array(v)
                pending_end = False
            else:
                offsets.append(v)
            i += 4
        elif t == "CHANNELS":
            n = int(toks[i + 1])
            ch = toks[i + 2 : i + 2 + n]
            order.append("".join(_AXES[c] for c in ch if c in _AXES))
            i += 2 + n
        elif t == "{":
            stack.append(cur)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        else:
            i += 1
    i += 1  # MOTION
    assert toks[i] == "Frames:"
    nf = int(toks[i + 1])
    dt = float(toks[i + 4])
    data = np.array(toks[i + 5 :], dtype=np.float64).reshape(nf, -1)
    pos = data[:, :3]
    rot = np.empty((nf, len(names), 3, 3))
    c = 3  # root: 3 position channels first, then its rotations like every other joint
    for j, seq in enumerate(order):
        ang = data[:, c : c + 3]
        c += 3
        # BVH applies channels in file order: R = R_first @ R_second @ R_third (intrinsic)
        rot[:, j] = Rotation.from_euler(seq.upper(), ang, degrees=True).as_matrix()
    return Bvh(names, parent, np.array(offsets), end_offset, pos, rot, 1.0 / dt)


def forward_kinematics(b: Bvh) -> tuple[np.ndarray, np.ndarray]:
    """World joint positions (F, J, 3) in file units and world rotations (F, J, 3, 3)."""
    nf, nj = len(b.pos), len(b.names)
    wp = np.empty((nf, nj, 3))
    wr = np.empty((nf, nj, 3, 3))
    for j in range(nj):
        p = b.parent[j]
        if p < 0:
            wr[:, j] = b.rot[:, j]
            wp[:, j] = b.pos
        else:
            wr[:, j] = wr[:, p] @ b.rot[:, j]
            wp[:, j] = wp[:, p] + np.einsum("fij,j->fi", wr[:, p], b.offset[j])
    return wp, wr


def virtual_wrist_imu(
    p_m: np.ndarray, r_sw: np.ndarray, fps: float, up: np.ndarray = np.array([0.0, 1.0, 0.0])
) -> tuple[np.ndarray, np.ndarray]:
    """Virtual IMU from sensor position ``p_m`` (F, 3) in metres and sensor->world rotations ``r_sw``.

    Returns accel (g) and gyro (rad/s) in the sensor frame, on the mocap frame grid.
    """
    sos = butter(4, LOWPASS_HZ, btype="low", fs=fps, output="sos")
    p = sosfiltfilt(sos, p_m, axis=0)
    a = np.zeros_like(p)
    a[1:-1] = (p[2:] + p[:-2] - 2 * p[1:-1]) * fps**2
    a[0], a[-1] = a[1], a[-2]
    f_world = a + up * G  # a - g with g pointing down
    acc = np.einsum("fji,fj->fi", r_sw, f_world) / G
    rv = Rotation.from_matrix(r_sw)
    w = np.zeros((len(r_sw), 3))
    w[1:-1] = (rv[:-2].inv() * rv[2:]).as_rotvec() * fps / 2  # body-frame rate, centred difference
    w[0], w[-1] = w[1], w[-2]
    return acc, w


def resample_to_canonical(x: np.ndarray, fps: float) -> np.ndarray:
    """Uniform ``fps`` signal -> FS grid (zero-phase low-pass first, as for real sources)."""
    from .canonical import to_grid

    t = np.arange(len(x)) / fps
    grid = np.arange(int((len(x) - 1) / fps * FS) + 1) / FS
    return to_grid(t, x, fps, grid)
