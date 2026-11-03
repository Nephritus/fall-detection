"""The evaluation protocol: folds of the fall datasets, training on a fold, and resumable results.

* ``FallData``: the canonical lab windows (``data/processed``), loaded once.
* ``lodo_folds`` / ``cv_folds`` / ``final_fold``: a ``Fold`` holds out whole subjects. Leave-one-dataset-out
  holds out one dataset (GLOSSARY: Leave-one-dataset-out); subject-wise CV holds out a group of subjects;
  the final fold trains on everything. Every fold carries a subject-wise validation slice of its
  training subjects (GLOSSARY: Subject-wise split) for early stopping and the decision threshold.
* ``Fold.train_model``: train on the fold's training windows plus optional extra windows; returns the
  model and its validation-slice threshold (``rules.keep_falls``). ``Fold.with_training`` narrows the
  training windows (out-of-fold mining). ``Fold.out_of_dataset_fall_scores``: each training dataset's
  fall windows scored by a model trained without that dataset, the input of the nested rules.
* ``Results``: one stored result per (condition, seed, fold), so an interrupted experiment resumes and
  its report can be rebuilt from what is stored.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from .model import FallModel
from .paths import PROCESSED
from .rules import keep_falls

VAL_FRACTION = 0.2
_SIGNAL_CHANNELS = {"acc": 3, "imu": 6}


@dataclass(frozen=True)
class FallData:
    X: np.ndarray  # (N, T, 6) canonical windows: accel g, gyro rad/s
    meta: pd.DataFrame  # one row per window: dataset, subject, trial, activity, label, rec, start_s
    y: np.ndarray  # 1 = fall window

    @classmethod
    def load(cls, root: Path = PROCESSED) -> "FallData":
        meta = pd.read_csv(root / "meta.csv")
        return cls(np.load(root / "X.npy"), meta, meta.label.to_numpy().astype(np.int64))

    @property
    def datasets(self) -> list[str]:
        return sorted(self.meta.dataset.unique())

    def windows(self, idx: np.ndarray, channels: str) -> np.ndarray:
        return self.X[idx][:, :, : _SIGNAL_CHANNELS[channels]]


def split_validation(meta: pd.DataFrame, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ~20% of the training subjects of every training dataset (at least 1 if it has >= 3)."""
    rng = np.random.default_rng(seed)
    sub = meta.iloc[train_idx][["dataset", "subject"]].drop_duplicates()
    val_subjects = []
    for _, g in sub.groupby("dataset"):
        names = sorted(g.subject)
        k = max(1, round(VAL_FRACTION * len(names))) if len(names) >= 3 else 0
        val_subjects += list(rng.choice(names, size=k, replace=False))
    is_val = meta.subject.iloc[train_idx].isin(val_subjects).to_numpy()
    return train_idx[~is_val], train_idx[is_val]


@dataclass(frozen=True)
class Trained:
    model: FallModel
    threshold: float  # validation-slice threshold: keeps 95% of the fold's validation falls


@dataclass(frozen=True)
class Fold:
    data: FallData
    protocol: str  # "lodo" | "cv" | "final"
    name: str  # held-out dataset, "fold<k>", or "all"
    seed: int
    train: np.ndarray  # window indices
    validation: np.ndarray
    test: np.ndarray  # empty for the final fold

    @property
    def key(self) -> str:
        return self.name if self.protocol == "lodo" else f"{self.protocol}-{self.name}"

    @property
    def training_datasets(self) -> list[str]:
        return sorted(self.data.meta.dataset.iloc[np.r_[self.train, self.validation]].unique())

    def with_training(self, idx: np.ndarray) -> "Fold":
        """The same fold trained on a subset ``idx`` of its training windows (validation and test unchanged)."""
        if not np.isin(idx, self.train).all():
            raise ValueError("with_training takes a subset of the fold's training windows")
        return replace(self, train=np.asarray(idx))

    def train_model(self, channels: str, *, augment=None, extra: tuple[np.ndarray, np.ndarray] | None = None,
                    extra_validation: np.ndarray | None = None, weights: np.ndarray | None = None,
                    own_windows: bool = True, **train_kw) -> Trained:
        """Train on the fold's training windows (unless ``own_windows`` is False) plus ``extra`` = (canonical
        windows, labels). ``extra_validation`` (canonical ADL windows) joins the validation set for early
        stopping only; the threshold is set on the fold's own validation windows. ``weights`` apply to the
        fold's training windows (extra windows weigh 1)."""
        d, c = self.data, _SIGNAL_CHANNELS[channels]
        parts_x, parts_y, parts_w = [], [], []
        if own_windows:
            parts_x.append(d.windows(self.train, channels))
            parts_y.append(d.y[self.train])
            parts_w.append(np.ones(len(self.train), np.float32) if weights is None else np.asarray(weights, np.float32))
        if extra is not None:
            parts_x.append(np.asarray(extra[0])[:, :, :c])
            parts_y.append(np.asarray(extra[1], np.int64))
            parts_w.append(np.ones(len(extra[1]), np.float32))
        if not parts_x:
            raise ValueError("nothing to train on")
        xv, yv = d.windows(self.validation, channels), d.y[self.validation]
        if extra_validation is not None:
            xv = np.concatenate([xv, np.asarray(extra_validation)[:, :, :c]])
            yv = np.concatenate([yv, np.zeros(len(extra_validation), np.int64)])
        w = np.concatenate(parts_w)
        model = FallModel.train(np.concatenate(parts_x), np.concatenate(parts_y), xv, yv, channels=channels,
                                seed=self.seed, augment=augment, weights=None if np.all(w == 1) else w, **train_kw)
        threshold = keep_falls(model.scores(d.windows(self.validation, channels)), d.y[self.validation])
        return Trained(model, threshold)

    def out_of_dataset_fall_scores(self, channels: str, **train_kw) -> dict[str, np.ndarray]:
        """For every training dataset: its fall windows scored by a model trained on the fold's other training
        datasets (nested leave-one-dataset-out; same seed and training options as ``train_model``)."""
        d, out = self.data, {}
        held_out = set(d.datasets) - set(self.training_datasets)
        for s in self.training_datasets:
            pool = np.where(~d.meta.dataset.isin(held_out | {s}))[0]
            tr, va = split_validation(d.meta, pool, self.seed)
            inner = Fold(d, "nested", f"{self.key}-without-{s}", self.seed, tr, va, np.zeros(0, np.int64))
            trained = inner.train_model(channels, **train_kw)
            falls = np.where((d.meta.dataset == s).to_numpy() & (d.y == 1))[0]
            out[s] = trained.model.scores(d.windows(falls, channels))
            del trained
            torch.cuda.empty_cache()
        return out


def lodo_folds(data: FallData, seed: int) -> list[Fold]:
    out = []
    for ds in data.datasets:
        is_test = (data.meta.dataset == ds).to_numpy()
        tr, va = split_validation(data.meta, np.where(~is_test)[0], seed)
        out.append(Fold(data, "lodo", ds, seed, tr, va, np.where(is_test)[0]))
    return out


def cv_folds(data: FallData, seed: int, k: int = 5) -> list[Fold]:
    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for i, (train, test) in enumerate(gkf.split(data.X, data.y, groups=data.meta.subject)):
        tr, va = split_validation(data.meta, train, seed)
        out.append(Fold(data, "cv", f"fold{i}", seed, tr, va, test))
    return out


def final_fold(data: FallData, seed: int) -> Fold:
    tr, va = split_validation(data.meta, np.arange(len(data.meta)), seed)
    return Fold(data, "final", "all", seed, tr, va, np.zeros(0, np.int64))


# ------------------------------------------------------------------------------------ results
class Results:
    """Stored results under ``root``: per (condition, seed, fold) a table of rows and optional arrays.
    A result is complete once its table exists (arrays are written first), so a crash leaves no partial key."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _base(self, condition: str, seed: int, fold: str) -> Path:
        return self.root / condition / f"seed{seed}" / fold

    def has(self, condition: str, seed: int, fold: str) -> bool:
        return self._base(condition, seed, fold).with_suffix(".csv").exists()

    def save(self, condition: str, seed: int, fold: str, rows, arrays: dict | None = None) -> None:
        base = self._base(condition, seed, fold)
        base.parent.mkdir(parents=True, exist_ok=True)
        if arrays:
            tmp = base.with_name(base.name + ".tmp.npz")
            np.savez(tmp, **arrays)
            os.replace(tmp, base.with_suffix(".npz"))
        tmp = base.with_name(base.name + ".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, base.with_suffix(".csv"))

    def keys(self) -> Iterator[tuple[str, int, str]]:
        for f in sorted(self.root.glob("*/seed*/*.csv")):
            if ".tmp." not in f.name:
                yield f.parent.parent.name, int(f.parent.name[4:]), f.stem

    def table(self) -> pd.DataFrame:
        """All stored rows with ``condition``, ``seed`` and ``fold`` columns."""
        frames = [pd.read_csv(self._base(c, s, f).with_suffix(".csv")).assign(condition=c, seed=s, fold=f)
                  for c, s, f in self.keys()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["condition", "seed", "fold"])

    def arrays(self, condition: str, seed: int, fold: str) -> dict[str, np.ndarray]:
        with np.load(self._base(condition, seed, fold).with_suffix(".npz"), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
