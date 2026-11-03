"""False-alarm counting (GLOSSARY: Alarm event, Refractory period, False alarm, Exposure).

A ``FalseAlarms`` holds, for a group of wearers, every detection opportunity: its alarm time (when the
device would alert), its score and optionally an activity label, plus each wearer's exposure. Given a
threshold it merges the detections that fire into alarm events with a refractory period, separately per
wearer, and reports events, rates per day, a bootstrap interval over wearers, a per-activity breakdown
and the alarms-versus-threshold curve. Numpy only: no model, no files.

Stage-1 detections are windows: ``FalseAlarms.from_window_scores`` sets each alarm time to the window's
end and the exposure to one stride per window. Callers with other detections (stage-2 candidates) pass
alarm times and exposure explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .windows import STRIDE_S, WIN_S

REFRACTORY_S = 30.0
FIRE_FRACTIONS = np.logspace(-5, -0.8, 40)  # default curve: thresholds at these shares of detections firing


def _merge(times: np.ndarray, refractory: float) -> np.ndarray:
    """Indices (into sorted ``times``) of detections that open an alarm event: a detection opens one when
    at least ``refractory`` seconds have passed since the last opening detection."""
    out, i, n = [], 0, len(times)
    while i < n:
        out.append(i)
        j = int(np.searchsorted(times, times[i] + refractory, side="left"))
        while j > i + 1 and times[j - 1] - times[i] >= refractory:  # guard float rounding at the edge
            j -= 1
        while j < n and times[j] - times[i] < refractory:
            j += 1
        i = j
    return np.asarray(out, dtype=np.int64)


def per_day_ci(events: np.ndarray, hours: np.ndarray, rng: np.random.Generator, n: int = 2000,
               level: float = 0.95) -> tuple[float, float]:
    """Bootstrap interval over wearers for the pooled false alarms per day."""
    events, hours = np.asarray(events, float), np.asarray(hours, float)
    idx = rng.integers(0, len(events), (n, len(events)))
    r = events[idx].sum(1) / hours[idx].sum(1) * 24
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(r, [a, 100 - a])
    return float(lo), float(hi)


def sensitivity_at_budget(thresholds: np.ndarray, per_day: np.ndarray, fall_scores: np.ndarray,
                          budget: float) -> float:
    """Share of ``fall_scores`` detected at the lowest threshold whose false alarms stay within ``budget``
    per day. ``thresholds`` descend; alarms can only grow as the threshold drops, so the curve is made
    monotone first. 0 if even the highest threshold exceeds the budget."""
    per_day = np.maximum.accumulate(np.asarray(per_day, float))
    ok = np.where(per_day <= budget)[0]
    return float((np.asarray(fall_scores) >= thresholds[ok[-1]]).mean()) if len(ok) else 0.0


class FalseAlarms:
    """Detections of a group of wearers, countable as alarm events at any threshold."""

    def __init__(self, times: Sequence[np.ndarray], scores: Sequence[np.ndarray], hours: Sequence[float],
                 labels: Sequence[np.ndarray] | None = None,
                 label_hours: Sequence[dict] | None = None, refractory: float = REFRACTORY_S) -> None:
        if not (len(times) == len(scores) == len(hours)):
            raise ValueError("times, scores and hours need one entry per wearer")
        self.refractory = refractory
        self.hours = np.asarray(hours, float)
        self._t, self._s, self._l = [], [], []
        for k, (t, s) in enumerate(zip(times, scores)):
            t, s = np.asarray(t, float), np.asarray(s, float)
            if t.shape != s.shape:
                raise ValueError(f"wearer {k}: times and scores differ in length")
            o = np.argsort(t, kind="stable")
            self._t.append(t[o])
            self._s.append(s[o])
            self._l.append(None if labels is None else np.asarray(labels[k])[o])
        self._label_hours = list(label_hours) if label_hours is not None else None

    @classmethod
    def from_window_scores(cls, scores: Sequence[np.ndarray], labels: Sequence[np.ndarray] | None = None,
                           refractory: float = REFRACTORY_S) -> "FalseAlarms":
        """Window detections (4 s windows, 2 s stride, from the start of each wearer's signal): alarm time
        = window end, exposure = one stride per window, label exposure from the window labels."""
        times = [np.arange(len(s)) * STRIDE_S + WIN_S for s in scores]
        hours = [len(s) * STRIDE_S / 3600 for s in scores]
        lh = None
        if labels is not None:
            lh = []
            for lab in labels:
                u, c = np.unique(np.asarray(lab), return_counts=True)
                lh.append({k: n * STRIDE_S / 3600 for k, n in zip(u.tolist(), c)})
        return cls(times, scores, hours, labels, lh, refractory)

    @property
    def n_wearers(self) -> int:
        return len(self._t)

    def events(self, threshold: float) -> np.ndarray:
        """Alarm events per wearer."""
        return np.array([len(_merge(t[s >= threshold], self.refractory)) for t, s in zip(self._t, self._s)])

    def per_day(self, threshold: float) -> float:
        """False alarms per 24 h of exposure, pooled over wearers."""
        return float(self.events(threshold).sum() / self.hours.sum() * 24)

    def per_day_by_wearer(self, threshold: float) -> np.ndarray:
        return self.events(threshold) / self.hours * 24

    def per_day_ci(self, threshold: float, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
        return per_day_ci(self.events(threshold), self.hours, rng, n)

    def by_label(self, threshold: float) -> pd.DataFrame:
        """One row per (wearer, label): exposure hours, alarm events (an event belongs to the label of the
        detection that opened it, so labels sum to the wearer's events) and firing detections."""
        if self._label_hours is None or any(l is None for l in self._l):
            raise ValueError("by_label needs labels and label exposure")
        rows = []
        for k, (t, s, lab, lh) in enumerate(zip(self._t, self._s, self._l, self._label_hours)):
            fire = s >= threshold
            opened = lab[fire][_merge(t[fire], self.refractory)]
            ev = dict(zip(*np.unique(opened, return_counts=True))) if len(opened) else {}
            det = dict(zip(*np.unique(lab[fire], return_counts=True))) if fire.any() else {}
            for name in set(lh) | set(ev):
                rows.append({"wearer": k, "label": name, "hours": lh.get(name, 0.0),
                             "events": int(ev.get(name, 0)), "detections": int(det.get(name, 0))})
        return pd.DataFrame(rows, columns=["wearer", "label", "hours", "events", "detections"])

    def curve(self, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(thresholds, false alarms per day). By default the thresholds are the score quantiles at which
        ``FIRE_FRACTIONS`` of all detections fire, in descending order."""
        if thresholds is None:
            thresholds = np.quantile(np.concatenate(self._s), 1 - FIRE_FRACTIONS)
        thresholds = np.asarray(thresholds, float)
        return thresholds, np.array([self.per_day(t) for t in thresholds])
