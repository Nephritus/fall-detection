"""Synthetic training windows from LAFAN1 motion capture (ADR 0002).

The left-hand joint of every BVH becomes a virtual wrist IMU in the canonical wrist frame. LAFAN1 has
no fall annotations, so falls are detected from the skeleton: an *event* is an upright stance
(hips > 0.75 m, trunk < 35 deg from vertical) followed within 2 s by a lying one (hips < 0.45 m,
trunk > 55 deg). Events in the ``fallAndGetUp`` and ``pushAndFall`` themes are falls; events in any
other theme are ambiguous (stumbles, ground work) and their surroundings are left out entirely.
Every window follows the real-data rules of ``windows.py``: a fall window contains the impact
moment, other windows of a fall take are excluded, other themes give ADL windows.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import ACC_CLIP_G, FS, Recording, align_mounting, find_impact
from .mocap import forward_kinematics, parse_bvh, resample_to_canonical, virtual_wrist_imu
from .windows import WIN_S, build_windows

FALL_THEMES = {"fallAndGetUp", "pushAndFall"}
# hand-local axes -> wrist frame: x along the bone to the fingers, z = dorsal (body-left when the arm
# hangs, palm to the thigh), y = z x x. Found from the mean hanging orientation across subjects.
HAND_TO_SENSOR = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
UP_H, UP_TILT, LOW_H, LOW_TILT = 0.75, 35.0, 0.45, 55.0
MAX_DESCENT_S, MIN_GAP_S = 2.0, 3.0
AMBIGUOUS_MARGIN_S = 3.0  # seconds dropped either side of an ambiguous event
IMPACT_AFTER_LOW_S = 0.75  # the impact is searched up to this long after the lying posture starts
IMPACT_OFFSETS_S = tuple(np.arange(0.5, WIN_S - 0.49, 0.5))  # where the impact sits in each fall window
TRIM_S = 0.5  # virtual IMU edge artefacts


def detect_events(wp: np.ndarray, names: list[str], fps: float) -> list[tuple[int, int]]:
    """(last upright frame, first lying frame) pairs."""
    j = names.index
    h = wp[:, 0, 1] / 100
    trunk = wp[:, j("Neck")] - wp[:, 0]
    trunk /= np.linalg.norm(trunk, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(trunk[:, 1], -1, 1)))
    up, low = (h > UP_H) & (tilt < UP_TILT), (h < LOW_H) & (tilt > LOW_TILT)
    out, last = [], -(10**9)
    for i in range(1, len(h)):
        if low[i] and not low[i - 1]:
            lo = max(0, i - int(MAX_DESCENT_S * fps))
            k = np.where(up[lo:i])[0]
            if len(k) and i - last > MIN_GAP_S * fps:
                out.append((lo + int(k[-1]), i))
                last = i
    return out


def fall_kind(wp: np.ndarray, names: list[str], u: int, i: int) -> str:
    """forward / backward / lateral from the trunk direction at the lying frame, in the body frame at u."""
    j = names.index
    left = wp[u, j("LeftUpLeg")] - wp[u, j("RightUpLeg")]
    left[1] = 0
    left /= np.linalg.norm(left)
    fwd = np.cross(left, [0, 1.0, 0])
    trunk = wp[i, j("Neck")] - wp[i, 0]
    trunk[1] = 0
    a = np.degrees(np.arctan2(trunk @ left, trunk @ fwd))
    return "forward" if abs(a) < 45 else "backward" if abs(a) > 135 else "lateral"


def build_synthetic(raw: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    base = raw / "lafan1"
    files = sorted(base.glob("*.bvh"))
    if not files:
        raise FileNotFoundError(base)
    pat = re.compile(r"([a-zA-Z]+?)\d*_subject(\d)\.bvh")
    takes = []  # (theme, subject, acc, gyro, events [(u_s, i_s, kind)])
    for f in files:
        m = pat.match(f.name)
        theme, subj = m.group(1), m.group(2)
        b = parse_bvh(f)
        wp, wr = forward_kinematics(b)
        hand = b.names.index("LeftHand")
        acc, gyro = virtual_wrist_imu(wp[:, hand] / 100, wr[:, hand] @ HAND_TO_SENSOR, b.fps)
        acc, gyro = resample_to_canonical(acc, b.fps), resample_to_canonical(gyro, b.fps)
        n = round(TRIM_S * FS)
        ev = [(u / b.fps, i / b.fps, fall_kind(wp, b.names, u, i)) for u, i in detect_events(wp, b.names, b.fps)]
        takes.append((theme, subj, acc[n:-n], gyro[n:-n], [(u - TRIM_S, i - TRIM_S, k) for u, i, k in ev]))

    # per-subject mounting alignment from walking, exactly as for the real datasets
    aligns, info_align = {}, {}
    for subj in sorted({t[1] for t in takes}):
        walk = [t[2] for t in takes if t[1] == subj and t[0] == "walk"]
        aligns[subj] = align_mounting(np.concatenate(walk), "left")
        info_align[f"lafan1:S{subj}"] = round(aligns[subj].norm_g, 3)

    recs: list[Recording] = []
    n_fall = {"forward": 0, "backward": 0, "lateral": 0}
    n_ambiguous = 0
    for theme, subj, acc, gyro, events in takes:
        if aligns[subj].excluded:
            continue
        acc, gyro = aligns[subj].apply(acc, gyro)
        acc = np.clip(acc, -ACC_CLIP_G, ACC_CLIP_G).astype(np.float32)
        gyro = gyro.astype(np.float32)
        sid, tid = f"lafan1:S{subj}", f"{theme}"
        if theme in FALL_THEMES:
            for k, (u, i, kind) in enumerate(events):
                imp, _ = find_impact(acc, (u, i + IMPACT_AFTER_LOW_S))
                n_fall[kind] += 1
                for off in IMPACT_OFFSETS_S:
                    s0 = round((imp - off) * FS)
                    s1 = s0 + round(WIN_S * FS)
                    if s0 < 0 or s1 > len(acc):
                        continue
                    recs.append(Recording("lafan1", sid, f"{tid}-e{k}-o{off:g}", f"fall-{kind}", True,
                                          acc[s0:s1], gyro[s0:s1], off, "interval"))
        else:
            keep = np.ones(len(acc), bool)
            for u, i, _ in events:
                n_ambiguous += 1
                keep[max(0, round((u - AMBIGUOUS_MARGIN_S) * FS)) : round((i + AMBIGUOUS_MARGIN_S) * FS) + 1] = False
            edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
            for k, (a, z) in enumerate(zip(edges[::2], edges[1::2])):
                recs.append(Recording("lafan1", sid, f"{tid}-s{k}", theme, False, acc[a:z], gyro[a:z], None, ""))
    X, meta, counts = build_windows(recs)
    fall, adl = X[meta.label == 1], X[meta.label == 0]
    pk = lambda a: np.percentile(np.linalg.norm(a[:, :, :3], axis=2).max(1), [10, 50, 90]).round(2).tolist()  # noqa: E731
    info = {
        "takes": len(takes), "subjects": len({t[1] for t in takes}),
        "fall_events": n_fall, "ambiguous_events_excluded": n_ambiguous,
        "walking_mean_norm_g": info_align,
        "excluded_subjects": sorted(f"lafan1:S{s}" for s, a in aligns.items() if a.excluded),
        "windows": {"fall": int(len(fall)), "adl": int(len(adl))},
        "fall_peak_acc_g_p10_50_90": pk(fall), "adl_peak_acc_g_p10_50_90": pk(adl),
        "fall_peak_gyro_rad_s_p50": float(np.median(np.linalg.norm(fall[:, :, 3:], axis=2).max(1))),
        **{f"build_{k}": v for k, v in counts.items()},
    }
    return X, meta, info
