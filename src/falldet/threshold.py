"""Threshold transfer experiment: how should the decision threshold be chosen for an unseen dataset?

The model ranks windows well on unseen datasets, but the threshold chosen for >=95% sensitivity on a
validation slice of the *training* datasets gives only ~0.85 sensitivity on a new one. Here a held-out
dataset stands in for a new device (same target split as ``finetune``: evaluation half, recording-pool
half) and several threshold rules are compared on the same trained models:

* ``val-slice``: current rule, 95% sensitivity on a subject-wise slice of the training datasets
* ``lodo-pooled``: nested leave-one-dataset-out. For every training dataset a model is trained on the
  remaining training datasets and scores the dataset it never saw; the threshold gives 95% sensitivity
  over those out-of-dataset scores (every dataset's falls weigh equally)
* ``lodo-median`` / ``lodo-worst``: the median / minimum of the per-dataset 95% thresholds from the
  same nested scores (worst = conservative: every training dataset reaches 95%)
* ``adl-budget-N``: needs only ADL recordings of the new device: the threshold that gives N false
  alarms/h on the ADL windows of the recording pool (no falls needed)
* ``falls-pool``: 95% sensitivity on the labelled pool windows (needs fall recordings: upper bound for
  device calibration with the data used in ``finetune``)
* ``oracle``: 95% sensitivity on the evaluation half itself (not realisable; the target)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .augment import Augment
from .baseline import metrics
from .finetune import split_target
from .paths import REPORTS
from .rules import TARGET_SENSITIVITY, adl_budget, keep_falls, lodo_median, lodo_pooled, lodo_worst
from .runner import FallData, Fold, Results, lodo_folds

BUDGETS = (10, 25, 50)  # false alarms per hour
RULES = ["val-slice", "lodo-pooled", "lodo-median", "lodo-worst",
         *[f"adl-budget-{b}" for b in BUDGETS], "falls-pool", "oracle"]


def compare_rules(fold: Fold, channels: str, aug=None) -> list[dict]:
    """Every rule on one held-out dataset (``fold`` from ``lodo_folds``): one row per rule."""
    data, y = fold.data, fold.data.y
    ev_idx, pool = split_target(data.meta, fold.name, fold.seed)
    pool_idx = np.where(data.meta.subject.isin(pool) & (data.meta.dataset == fold.name))[0]
    trained = fold.train_model(channels, augment=aug)
    p_ev = trained.model.scores(data.windows(ev_idx, channels))
    p_pool = trained.model.scores(data.windows(pool_idx, channels))
    thr = {"val-slice": trained.threshold}
    del trained
    torch.cuda.empty_cache()
    fall_scores = fold.out_of_dataset_fall_scores(channels, augment=aug)
    thr["lodo-pooled"] = lodo_pooled(fall_scores)
    thr["lodo-median"] = lodo_median(fall_scores)
    thr["lodo-worst"] = lodo_worst(fall_scores)
    adl_pool = p_pool[y[pool_idx] == 0]
    for b in BUDGETS:
        thr[f"adl-budget-{b}"] = adl_budget(adl_pool, b)
    thr["falls-pool"] = keep_falls(p_pool, y[pool_idx])
    thr["oracle"] = keep_falls(p_ev, y[ev_idx])
    return [{"rule": rule, "threshold": t, **metrics(y[ev_idx], (p_ev >= t).astype(int), p_ev)}
            for rule, t in thr.items()]


def run(seeds: int = 3, augment: bool = False) -> None:
    t0 = time.time()
    data = FallData.load()
    aug = Augment() if augment else None
    out = REPORTS / ("threshold_augmented" if augment else "threshold")
    results = Results(out / "runs")
    for seed in range(seeds):
        for fold in lodo_folds(data, seed):
            for ch in ("imu", "acc"):
                if not results.has(ch, seed, fold.key):
                    results.save(ch, seed, fold.key, compare_rules(fold, ch, aug))
            print(f"[seed {seed}] {fold.key} done ({time.time() - t0:.0f}s)", flush=True)
    df = results.table().rename(columns={"condition": "channels", "fold": "dataset"})
    write_report(df[df.seed < seeds], out, seeds, time.time() - t0, aug)
    print(f"report: {out / 'report.md'}")


def write_report(df: pd.DataFrame, out, seeds: int, seconds: float, aug=None) -> None:
    df = df.assign(hit=(df.sensitivity >= 0.90).astype(float), gap=(df.sensitivity - TARGET_SENSITIVITY).abs())
    L = ["# Threshold transfer" + (" (with augmentation)" if aug else "") + "\n"]
    L.append(f"Seeds: {seeds}. Runtime {seconds / 60:.1f} min. A held-out dataset stands in for a new device; "
             "rules are scored on its evaluation half (same split as `finetune`). All rules use the same "
             "trained model per run, so AUROC/AUPRC are identical across rules; only the threshold differs. "
             "Target: 95% sensitivity at as few false alarms as possible. Rules are described in "
             "`threshold.py`. `adl-budget-N` needs only ADL recordings of the new device; `falls-pool` "
             "needs labelled falls; `oracle` uses the evaluation labels and is not realisable. "
             "`sens ≥ 0.90` is the share of (seed, dataset) runs that got at least 0.90 sensitivity. "
             "All simulated falls: optimistic.\n")
    if aug:
        L.append(f"Training (final and nested models) uses augmentation: {aug}.\n")
    cols = ["sensitivity", "false_alarms_per_h", "hit", "gap"]
    hdr = "| rule | sensitivity | false alarms/h | sens ≥ 0.90 | \\|sens − 0.95\\| |"
    sep = "|---|---|---|---|---|"
    for ch, title in (("acc", "Accelerometer only"), ("imu", "Accelerometer + gyroscope")):
        sub = df[df.channels == ch]
        a = sub.groupby("dataset")[["auroc", "ap"]].mean().mean()
        L.append(f"## {title}\n")
        L.append(f"Model quality (same for all rules): AUROC {a.auroc:.3f}, AUPRC {a.ap:.3f} (macro).\n")
        L.append("### Macro average over target datasets\n")
        L.append(hdr); L.append(sep)
        macro = sub.groupby(["rule", "dataset"])[cols].mean().groupby("rule").mean()
        for r in RULES:
            if r in macro.index:
                m = macro.loc[r]
                L.append(f"| {r} | {m.sensitivity:.3f} | {m.false_alarms_per_h:.1f} | {m.hit:.0%} | {m.gap:.3f} |")
        L.append("")
        for ds in sorted(sub.dataset.unique()):
            L.append(f"### Target: {ds}\n")
            L.append("| rule | sensitivity | false alarms/h | threshold |")
            L.append("|---|---|---|---|")
            s = sub[sub.dataset == ds]
            for r in RULES:
                g = s[s.rule == r]
                if len(g):
                    L.append(f"| {r} | {g.sensitivity.mean():.3f} ± {g.sensitivity.std(ddof=0):.3f} | "
                             f"{g.false_alarms_per_h.mean():.1f} ± {g.false_alarms_per_h.std(ddof=0):.1f} | "
                             f"{g.threshold.mean():.3f} |")
            L.append("")
    (out / "report.md").write_text("\n".join(L))
