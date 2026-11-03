"""Cut canonical recordings into labelled windows (GLOSSARY: Window, Fall window, Impact moment)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .canonical import FS, Recording

WIN_S = 4.0
STRIDE_S = 2.0  # 50% overlap
EDGE_S = 0.25  # an impact this close to a window edge is ambiguous: window is excluded


def build_windows(
    recs: list[Recording], win_s: float = WIN_S, stride_s: float = STRIDE_S
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Return X (N, win, 6) [acc g, gyro rad/s], per-window metadata, and exclusion counts.

    ADL recordings: every window is ADL. Fall recordings: a window is a fall window only when the
    impact moment lies inside it (not within EDGE_S of its edges); every other window is excluded.
    """
    win, stride, edge = round(win_s * FS), round(stride_s * FS), round(EDGE_S * FS)
    xs, rows = [], []
    counts = {"fall_recording_windows_excluded": 0, "fall_recording_edge_ambiguous": 0,
              "recordings_shorter_than_window": 0}
    for ri, r in enumerate(recs):
        n = len(r.acc)
        if n < win:
            counts["recordings_shorter_than_window"] += 1
            continue
        sig = np.concatenate([r.acc, r.gyro], axis=1)
        imp = round(r.impact_s * FS) if r.impact_s is not None else None
        for start in range(0, n - win + 1, stride):
            end = start + win
            if r.is_fall:
                if start + edge <= imp < end - edge:
                    label = 1
                elif start <= imp < end:
                    counts["fall_recording_edge_ambiguous"] += 1
                    continue
                else:
                    counts["fall_recording_windows_excluded"] += 1
                    continue
            else:
                label = 0
            xs.append(sig[start:end])
            rows.append((r.dataset, r.subject, r.trial, r.activity, label, ri, start / FS))
    meta = pd.DataFrame(
        rows, columns=["dataset", "subject", "trial", "activity", "label", "rec", "start_s"]
    )
    return np.stack(xs).astype(np.float32), meta, counts
