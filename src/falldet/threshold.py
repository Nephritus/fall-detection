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
from .baseline import TARGET_SENSITIVITY, choose_threshold, metrics, split_validation
from .cli import PROCESSED, REPORTS
from .finetune import split_target
from .model import FallModel
from .windows import STRIDE_S

BUDGETS = (10, 25, 50)  # false alarms per hour
RULES = ["val-slice", "lodo-pooled", "lodo-median", "lodo-worst",
         *[f"adl-budget-{b}" for b in BUDGETS], "falls-pool", "oracle"]


def balanced_threshold(scores: list[np.ndarray]) -> float:
    """95% sensitivity where every group of fall scores carries equal total weight."""
    s = np.concatenate(scores)
    w = np.concatenate([np.full(len(a), 1 / len(a)) for a in scores])
    order = np.argsort(-s)
    cum = np.cumsum(w[order]) / w.sum()
    return float(s[order][np.argmax(cum >= TARGET_SENSITIVITY)])


def budget_threshold(adl_scores: np.ndarray, fa_per_h: float) -> float:
    """Smallest threshold whose false-alarm rate on ``adl_scores`` is <= ``fa_per_h``."""
    rate = fa_per_h * STRIDE_S / 3600
    return float(np.quantile(adl_scores, 1 - rate, method="higher"))


def run(seeds: int = 3, augment: bool = False) -> None:
    t0 = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    aug = Augment() if augment else None
    datasets = sorted(meta.dataset.unique())
    rows = []
    for seed in range(seeds):
        for ds in datasets:
            ev_idx, pool = split_target(meta, ds, seed)
            pool_idx = np.where(meta.subject.isin(pool) & (meta.dataset == ds))[0]
            src_sets = [d for d in datasets if d != ds]
            src = np.where(meta.dataset != ds)[0]
            tr, va = split_validation(meta, src, seed)
            for ch in ("imu", "acc"):
                net = FallModel.train(X[tr], y[tr], X[va], y[va], channels=ch, seed=seed, augment=aug)
                p_ev, p_pool = net.scores(X[ev_idx]), net.scores(X[pool_idx])
                thr = {"val-slice": choose_threshold(net.scores(X[va]), y[va])}
                del net
                # nested leave-one-dataset-out over the training datasets
                per_ds_fall_scores, per_ds_thr = [], []
                for s in src_sets:
                    s_idx = np.where(meta.dataset == s)[0]
                    inner = np.where(~meta.dataset.isin([ds, s]))[0]
                    itr, iva = split_validation(meta, inner, seed)
                    inet = FallModel.train(X[itr], y[itr], X[iva], y[iva], channels=ch, seed=seed, augment=aug)
                    ps = inet.scores(X[s_idx])
                    fall = ps[y[s_idx] == 1]
                    per_ds_fall_scores.append(fall)
                    per_ds_thr.append(choose_threshold(ps, y[s_idx]))
                    del inet
                    torch.cuda.empty_cache()
                thr["lodo-pooled"] = balanced_threshold(per_ds_fall_scores)
                thr["lodo-median"] = float(np.median(per_ds_thr))
                thr["lodo-worst"] = float(np.min(per_ds_thr))
                adl_pool = p_pool[y[pool_idx] == 0]
                for b in BUDGETS:
                    thr[f"adl-budget-{b}"] = budget_threshold(adl_pool, b)
                thr["falls-pool"] = choose_threshold(p_pool, y[pool_idx])
                thr["oracle"] = choose_threshold(p_ev, y[ev_idx])
                for rule, t in thr.items():
                    rows.append({"seed": seed, "dataset": ds, "channels": ch, "rule": rule,
                                 "threshold": t, **metrics(y[ev_idx], (p_ev >= t).astype(int), p_ev)})
                torch.cuda.empty_cache()
            print(f"[seed {seed}] {ds} done ({time.time() - t0:.0f}s)", flush=True)
    out = REPORTS / ("threshold_augmented" if augment else "threshold")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    write_report(df, out, seeds, time.time() - t0, aug)
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
