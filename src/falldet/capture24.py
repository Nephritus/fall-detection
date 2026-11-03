"""Capture-24 free-living recordings (GLOSSARY: Free-living recording) as a source of ADL windows.

The ``Capture24`` source owns everything about the prepared recordings: building them from the raw
download (mounting alignment, 25 Hz canonical accelerometer, one activity label per sample), the split
of people into training, validation and held-out groups, and window draws. A draw is keyed by
(seed, purpose, fold, person), so it does not depend on what else was drawn before or in which order:
an interrupted and resumed experiment gets exactly the windows an uninterrupted one gets.

Windows are canonical accelerometer windows, (N, 4 s at 25 Hz, 3) in g; encoding them for a model is
the model's job.
"""

from __future__ import annotations

import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import ACC_CLIP_G, FS, align_mounting, to_grid
from .windows import STRIDE_S, WIN_S

SRC_FS = 100  # Axivity AX3 rate
LABEL_COL = "label:WillettsSpecific2018"
LABELS = ["sleep", "sitting", "standing", "walking", "household-chores", "manual-work",
          "mixed-activity", "sports", "vehicle", "bicycling"]
UNLABELLED = 255
LABEL_NAMES = {**dict(enumerate(LABELS)), UNLABELLED: "unlabelled"}
_NAME_OF_CODE = np.array([LABEL_NAMES.get(c, f"code-{c}") for c in range(256)], dtype=object)
# Free-living walking includes arm swing: its mean specific force is about 0.7 g (lab walking 0.9-1.0 g),
# so only its direction is used and the exclusion threshold is lower than the lab default.
MIN_WALK_S, MIN_WALK_NORM_G = 300, 0.5
POOL_MAX_ID, VALIDATION_SHARE = 100, 0.2  # P001-P100 may be trained on; P101-P151 are held out
WIN, STRIDE = round(WIN_S * FS), round(STRIDE_S * FS)


@dataclass(frozen=True)
class PeopleSplit:
    train: list[str]
    validation: list[str]
    held_out: list[str]


def _key(*parts) -> list[int]:
    return [p if isinstance(p, int) else zlib.crc32(str(p).encode()) for p in parts]


def _cut(acc: np.ndarray, index: np.ndarray) -> np.ndarray:
    starts = np.asarray(index, dtype=np.int64) * STRIDE
    return acc[starts[:, None] + np.arange(WIN)]


class Capture24:
    """Prepared Capture-24 recordings under ``root`` (one ``<pid>.npz`` per usable person and
    ``prepare_log.csv`` for everyone, with the reason a person was excluded)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------------------- people
    def log(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "prepare_log.csv")

    def people(self) -> list[str]:
        """Usable people (not excluded at preparation), sorted."""
        log = self.log()
        return sorted(log[log.excluded.isna()].pid) if "excluded" in log else sorted(log.pid)

    def split(self, seed: int) -> PeopleSplit:
        """P001-P100 split into training and validation (a seed-dependent 20%); P101-P151 held out."""
        pool = [p for p in self.people() if int(p[1:]) <= POOL_MAX_ID]
        rng = np.random.default_rng(seed + 321)
        val = sorted(rng.choice(pool, size=max(1, round(VALIDATION_SHARE * len(pool))), replace=False))
        held = [p for p in self.people() if int(p[1:]) > POOL_MAX_ID]
        return PeopleSplit([p for p in pool if p not in val], list(val), held)

    # ------------------------------------------------------------------------------ signals
    def signal(self, pid: str) -> np.ndarray:
        """Canonical accelerometer signal (n, 3) in g at 25 Hz."""
        return np.load(self.root / f"{pid}.npz")["acc"]

    def labels(self, pid: str) -> np.ndarray:
        """Activity name per sample of ``signal(pid)`` ("unlabelled" where the annotation is missing)."""
        return _NAME_OF_CODE[np.load(self.root / f"{pid}.npz")["lab"]]

    def window_labels(self, pid: str) -> np.ndarray:
        """Activity at the centre of each 4 s window (2 s stride) of the signal."""
        lab = self.labels(pid)
        return lab[np.arange((len(lab) - WIN) // STRIDE + 1) * STRIDE + WIN // 2]

    def n_windows(self, pid: str) -> int:
        return (len(self.signal(pid)) - WIN) // STRIDE + 1

    def hours(self, pid: str) -> float:
        """Exposure of the person's windows (one stride per window)."""
        return self.n_windows(pid) * STRIDE_S / 3600

    # ------------------------------------------------------------------------------ windows
    def windows_at(self, pid: str, index: np.ndarray) -> np.ndarray:
        """Windows of ``pid`` by window index (0 = the first 4 s), (N, WIN, 3)."""
        return _cut(self.signal(pid), index)

    def windows(self, pids: list[str], k: int, seed: int, purpose: str, fold: str = "") -> np.ndarray:
        """Up to ``k`` random windows per person, drawn without replacement; every person's draw is keyed
        by (seed, purpose, fold, person), so the result does not depend on call order."""
        out = []
        for pid in pids:
            acc = self.signal(pid)
            n = (len(acc) - WIN) // STRIDE + 1
            rng = np.random.default_rng(_key(seed, purpose, fold, pid))
            out.append(_cut(acc, rng.choice(n, size=min(k, n), replace=False)))
        return np.concatenate(out) if out else np.zeros((0, WIN, 3), np.float32)

    # ------------------------------------------------------------------------------ prepare
    def prepare(self, raw_dir: Path, workers: int = 3) -> pd.DataFrame:
        """Build the prepared recordings from the raw download (``capture24/capture24/`` under ``raw_dir``);
        a no-op when they exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        log = self.root / "prepare_log.csv"
        if log.exists() and all((self.root / f"{p}.npz").exists() for p in self.people()):
            return self.log()
        base = Path(raw_dir) / "capture24" / "capture24"
        if not base.is_dir():
            raise FileNotFoundError(base)
        mapping = pd.read_csv(base / "annotation-label-dictionary.csv", index_col=0, dtype=str)[LABEL_COL].to_dict()
        files = sorted(base.glob("P[0-9][0-9][0-9].csv.gz"))
        t0, rows = time.time(), []
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(_prepare_one, [(f, mapping, self.root) for f in files])):
                rows.append(r)
                if i % 10 == 0:
                    print(f"[prepare] {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(log, index=False)
        return df


def _resample(x: np.ndarray) -> np.ndarray:
    """100 Hz samples -> the canonical 25 Hz grid (anti-alias filter included), as for every source."""
    t = np.arange(len(x)) / SRC_FS
    return to_grid(t, x, SRC_FS, np.arange(int(t[-1] * FS) + 1) / FS)


def _prepare_one(args) -> dict:
    path, mapping, root = args
    pid = path.name.split(".")[0]
    t = pd.read_csv(path, usecols=["x", "y", "z", "annotation"],
                    dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": "category"})
    cats = t.annotation.cat.categories
    code_of = np.array([LABELS.index(mapping[c]) if c in mapping and mapping[c] in LABELS else UNLABELLED
                        for c in cats], dtype=np.uint8)
    codes = t.annotation.cat.codes.to_numpy()
    lab = np.where(codes >= 0, code_of[np.maximum(codes, 0)], UNLABELLED).astype(np.uint8)
    xyz = t[["x", "y", "z"]].to_numpy(np.float64)
    del t
    info = {"pid": pid, "hours": len(xyz) / SRC_FS / 3600}
    walk = xyz[lab == LABELS.index("walking")]
    info["walk_s"] = len(walk) / SRC_FS
    if info["walk_s"] < MIN_WALK_S:
        return {**info, "excluded": f"walking {info['walk_s']:.0f} s"}
    # the dominant wrist is recorded but its side is not: assumed right [UNVERIFIED]
    al = align_mounting(walk, "right", min_norm_g=MIN_WALK_NORM_G)
    info["walk_norm"] = al.norm_g
    if al.excluded:
        return {**info, "excluded": f"walking mean norm {info['walk_norm']:.2f} g"}
    acc = np.clip(al.apply(_resample(xyz))[0], -ACC_CLIP_G, ACC_CLIP_G).astype(np.float32)
    np.savez(Path(root) / f"{pid}.npz", acc=acc, lab=lab[:: SRC_FS // FS][: len(acc)])
    info["walk_axis_std"] = al.apply(_resample(walk))[0].std(axis=0).round(3).tolist()
    return info
