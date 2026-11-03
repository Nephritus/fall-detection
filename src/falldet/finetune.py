"""Per-device adaptation experiment: how much target-device data does a pretrained model need?

A held-out dataset stands in for "a new watch". Its subjects are split in half: a fixed evaluation
half that is never used for adaptation, and a recording pool of volunteers we could capture on the
device. The base model is trained on the other datasets. From the pool, k volunteers adapt the model
and a few more (calibration) pick the early-stopping point and the decision threshold.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .augment import Augment
from .baseline import choose_threshold, metrics, split_validation
from .cli import PROCESSED, REPORTS
from .model import FallModel

K_VALUES = (1, 2, 4)
METHODS = ("zero-shot", "recalibrated", "adabn", "head-only", "full", "scratch")


def split_target(meta: pd.DataFrame, dataset: str, seed: int):
    """Return (eval_idx, pool_subjects). Subjects that have fall windows are split separately from
    ADL-only subjects so both halves keep falls; the pool lists fall-recording subjects first
    (a new-device session records scripted falls and ADLs)."""
    rng = np.random.default_rng(seed + 7)
    d = meta[meta.dataset == dataset]
    has_fall = d.groupby("subject").label.max()
    pool, ev = [], []
    for flag in (1, 0):
        subs = sorted(has_fall[has_fall == flag].index)
        rng.shuffle(subs)
        half = (len(subs) + 1) // 2  # evaluation gets the larger half
        ev += subs[:half]
        pool += subs[half:]
    eval_idx = np.where(meta.subject.isin(ev) & (meta.dataset == dataset))[0]
    pool_fall_first = [s for s in pool if has_fall[s] == 1] + [s for s in pool if has_fall[s] == 0]
    return eval_idx, pool_fall_first


def run(seeds: int = 3, augment: bool = False) -> None:
    t0 = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    aug = Augment() if augment else None
    rows = []
    for seed in range(seeds):
        for ds in sorted(meta.dataset.unique()):
            ev_idx, pool = split_target(meta, ds, seed)
            src = np.where(meta.dataset != ds)[0]
            tr, va = split_validation(meta, src, seed)
            for ch in ("imu", "acc"):
                base = FallModel.train(X[tr], y[tr], X[va], y[va], channels=ch, seed=seed, augment=aug)
                thr_src = choose_threshold(base.scores(X[va]), y[va])

                def record(method, k, net, thr, n_ft=0, n_ft_fall=0):
                    p = net.scores(X[ev_idx])
                    rows.append({
                        "seed": seed, "dataset": ds, "channels": ch, "k": k, "method": method,
                        "threshold": thr, "ft_windows": n_ft, "ft_falls": n_ft_fall,
                        **metrics(y[ev_idx], (p >= thr).astype(int), p),
                    })

                for k in K_VALUES:
                    c = max(1, k // 2)
                    ft_subs, cal_subs = pool[:k], pool[k : k + c]
                    if len(cal_subs) < c:
                        continue
                    ft = np.where(meta.subject.isin(ft_subs))[0]
                    cal = np.where(meta.subject.isin(cal_subs))[0]
                    pool_all = np.where(meta.subject.isin(pool[: k + c]))[0]
                    if y[ft].sum() == 0 or y[cal].sum() == 0:
                        continue
                    n_ft, n_f = len(ft), int(y[ft].sum())
                    record("zero-shot", k, base, thr_src)
                    thr = choose_threshold(base.scores(X[cal]), y[cal])
                    record("recalibrated", k, base, thr)
                    ada = base.adapt_batchnorm(X[pool_all])
                    record("adabn", k, ada, choose_threshold(ada.scores(X[cal]), y[cal]))
                    for name, freeze in (("head-only", True), ("full", False)):
                        net = base.continue_training(X[ft], y[ft], X[cal], y[cal], seed=seed, freeze_features=freeze,
                                                     augment=aug)
                        record(name, k, net, choose_threshold(net.scores(X[cal]), y[cal]), n_ft, n_f)
                    sc = FallModel.train(X[ft], y[ft], X[cal], y[cal], channels=ch, seed=seed, augment=aug)
                    record("scratch", k, sc, choose_threshold(sc.scores(X[cal]), y[cal]), n_ft, n_f)
                del base
                torch.cuda.empty_cache()
            print(f"[seed {seed}] {ds} done ({time.time() - t0:.0f}s)", flush=True)
    out = REPORTS / ("finetune_augmented" if augment else "finetune")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    write_report(df, out, seeds, time.time() - t0, aug)
    print(f"report: {out / 'report.md'}")


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}"


def write_report(df: pd.DataFrame, out, seeds: int, seconds: float, aug=None) -> None:
    cols = ["sensitivity", "specificity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle"]
    hdr = "| method | k | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |"
    sep = "|---|---|---|---|---|---|---|---|"
    L = ["# Fine-tuning / per-device adaptation" + (" (with augmentation)" if aug else "") + "\n"]
    if aug:
        L.append(f"Every training step (base model, head-only, full fine-tuning, scratch) uses on-the-fly "
                 f"augmentation: {aug}. Calibration, evaluation and early-stopping data are never "
                 "augmented.\n")
    L.append(f"Seeds: {seeds}. Runtime {seconds / 60:.1f} min. A held-out dataset stands in for a new "
             "watch; its subjects are split in half. **Evaluation** is the fixed half (never used "
             "for adaptation), so zero-shot here is not the same number as the baseline report (which "
             "used all subjects). **k** is the number of volunteers from the other half used to adapt; "
             "a further max(1, k/2) volunteers calibrate (early stopping and the ≥95%-sensitivity "
             "threshold). `recalibrated` changes only the threshold; `adabn` re-estimates BatchNorm "
             "statistics on the unlabelled pool windows; `scratch` trains on the k volunteers only "
             "(control for whether pretraining matters). All simulated falls: optimistic.\n")
    for ch, title in (("acc", "Accelerometer only"), ("imu", "Accelerometer + gyroscope")):
        sub = df[df.channels == ch]
        L.append(f"## {title}\n")
        L.append("### Macro average over the target datasets\n")
        L.append(hdr); L.append(sep)
        macro = sub.groupby(["method", "k", "dataset"])[cols].mean().groupby(["method", "k"]).mean()
        for k in sorted(sub.k.unique()):
            for m in METHODS:
                if (m, k) in macro.index:
                    r = macro.loc[(m, k)]
                    L.append(f"| {m} | {k} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
        for ds in sorted(sub.dataset.unique()):
            L.append(f"\n### Target: {ds}\n")
            L.append(hdr); L.append(sep)
            s = sub[sub.dataset == ds]
            for k in sorted(s.k.unique()):
                for m in METHODS:
                    g = s[(s.method == m) & (s.k == k)]
                    if len(g):
                        L.append(f"| {m} | {k} | " + " | ".join(_ms(g[c]) for c in cols) + " |")
            g = s[(s.k == s.k.max()) & (s.method == "full")]
            if len(g):
                L.append(f"\nAdaptation set at k={int(s.k.max())}: ~{int(g.ft_windows.mean())} windows, "
                         f"{int(g.ft_falls.mean())} of them falls.")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
