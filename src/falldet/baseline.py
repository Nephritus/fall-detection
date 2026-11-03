"""Baseline experiment: 1D-CNN (+ one hard-negative mining round) vs the threshold reference.

Protocols (GLOSSARY): leave-one-dataset-out (headline) and subject-wise 5-fold CV. Every fold holds
out whole subjects; a subject-wise slice of the training subjects is used for early stopping and for
choosing the decision threshold (>=95% sensitivity on that validation slice).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from . import model as M
from .augment import Augment
from .canonical import FS
from .cli import PROCESSED, REPORTS
from .reference import ThresholdReference
from .windows import STRIDE_S

TARGET_SENSITIVITY = 0.95
VAL_FRACTION = 0.2
MINE_THRESHOLD_CAP = 0.10  # at most this share of training ADL windows is mined
MINE_WEIGHT = 3.0


# ------------------------------------------------------------------------------- helpers
def choose_threshold(p_val: np.ndarray, y_val: np.ndarray) -> float:
    """Largest threshold that keeps >= TARGET_SENSITIVITY of validation falls."""
    return float(np.quantile(p_val[y_val == 1], 1 - TARGET_SENSITIVITY, method="lower"))


def split_validation(meta: pd.DataFrame, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ~20% of the training subjects of every training dataset (at least 1 if >=3)."""
    rng = np.random.default_rng(seed)
    sub = meta.iloc[train_idx][["dataset", "subject"]].drop_duplicates()
    val_subjects = []
    for _, g in sub.groupby("dataset"):
        names = sorted(g.subject)
        k = max(1, round(VAL_FRACTION * len(names))) if len(names) >= 3 else 0
        val_subjects += list(rng.choice(names, size=k, replace=False))
    is_val = meta.subject.iloc[train_idx].isin(val_subjects).to_numpy()
    return train_idx[~is_val], train_idx[is_val]


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    hours = (tn + fp) * STRIDE_S / 3600  # each ADL window advances the replay by one stride
    fa95 = float("nan")
    if score is not None and len(set(y)) == 2:
        # diagnostic only: threshold set on the *test* scores, so it separates ranking quality from
        # threshold transfer; never used for model selection
        fpr, tpr, _ = roc_curve(y, score)
        fa95 = float(fpr[np.argmax(tpr >= TARGET_SENSITIVITY)] * 3600 / STRIDE_S)
    return {
        "n_fall": tp + fn, "n_adl": tn + fp,
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "false_alarms_per_h": fp / hours if hours else float("nan"),
        "fa_h_at_95sens_oracle": fa95,
        "auroc": roc_auc_score(y, score) if score is not None and len(set(y)) == 2 else float("nan"),
        "ap": average_precision_score(y, score) if score is not None and len(set(y)) == 2 else float("nan"),
    }


# --------------------------------------------------------------------------------- folds
def run_fold(X, meta, y, train_idx, test_idx, channels, seed, xin_cache, augment=None):
    """Train CNN stage 1 and stage 2 (after mining) on one split; return metric rows + predictions."""
    tr, va = split_validation(meta, train_idx, seed)
    xin = xin_cache[channels]
    out, preds = [], {}
    weights = np.ones(len(tr), dtype=np.float32)
    mined = 0
    for stage in ("stage1", "stage2"):
        net = M.fit(xin[tr], y[tr], weights, xin[va], y[va], seed, augment=augment)
        thr = choose_threshold(M.predict_proba(net, xin[va]), y[va])
        p_test = M.predict_proba(net, xin[test_idx])
        out.append({
            "model": "cnn", "channels": channels, "stage": stage, "threshold": thr, "mined": mined,
            "params": M.n_params(net), **metrics(y[test_idx], (p_test >= thr).astype(int), p_test),
        })
        preds[stage] = (p_test, thr)
        if stage == "stage1":
            weights, mined = mine_hard_negatives(xin, y, meta, tr, va, seed, augment=augment)
        del net
        torch.cuda.empty_cache()
    return out, preds


def mine_hard_negatives(xin, y, meta, tr, va, seed, folds: int = 3, augment=None):
    """Out-of-fold hard negative mining. Training subjects are split into ``folds`` groups; a model
    trained on the other groups scores the held-out group, and ADL windows that fire at that
    model's own >=95%-sensitivity threshold are hard negatives (the in-sample model would rarely
    fire on windows it has already fit). Returns per-window loss weights for ``tr`` and the count.
    """
    subs = meta.subject.iloc[tr]
    names = np.array(sorted(subs.unique()))
    np.random.default_rng(seed + 1000).shuffle(names)
    hard_local, hard_scores = [], []
    ones = lambda n: np.ones(n, dtype=np.float32)  # noqa: E731
    for part in np.array_split(names, folds):
        held = subs.isin(part).to_numpy()
        net = M.fit(xin[tr[~held]], y[tr[~held]], ones((~held).sum()), xin[va], y[va], seed, augment=augment)
        thr = choose_threshold(M.predict_proba(net, xin[va]), y[va])
        p = M.predict_proba(net, xin[tr[held]])
        idx = np.where(held)[0]
        fire = (y[tr[idx]] == 0) & (p >= thr)
        hard_local += list(idx[fire])
        hard_scores += list(p[fire])
        del net
        torch.cuda.empty_cache()
    hard = np.array(hard_local, dtype=int)
    cap = int(MINE_THRESHOLD_CAP * (y[tr] == 0).sum())
    hard = hard[np.argsort(-np.array(hard_scores))][:cap] if len(hard) else hard
    weights = ones(len(tr))
    weights[hard] = MINE_WEIGHT
    return weights, len(hard)


def run_reference(X, meta, y, train_idx, test_idx):
    ref = ThresholdReference().fit(X[train_idx], y[train_idx])
    pred = ref.predict(X[test_idx])
    row = {"model": "threshold", "channels": "acc", "stage": "-", "threshold": float("nan"),
           "mined": 0, "params": 0, "dip_g": ref.dip, "peak_g": ref.peak,
           **metrics(y[test_idx], pred, None)}
    return row, pred


# --------------------------------------------------------------------------------- driver
def run(seeds: int = 3, augment: bool = False) -> None:
    t_start = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    prep = json.loads((PROCESSED / "prepare_log.json").read_text())
    xin_cache = {c: M.make_inputs(X, c) for c in ("imu", "acc")}
    aug = Augment() if augment else None
    datasets = sorted(meta.dataset.unique())
    print(f"{len(X)} windows, {int(y.sum())} fall; datasets {datasets}; device {M.device()}")

    rows, act_rows = [], []

    def add(protocol, fold, seed, results):
        for r in results:
            rows.append({"protocol": protocol, "fold": fold, "seed": seed, **r})

    for seed in range(seeds):
        # ---- leave-one-dataset-out
        for d in datasets:
            test = np.where(meta.dataset == d)[0]
            train = np.where(meta.dataset != d)[0]
            for ch in ("imu", "acc"):
                res, preds = run_fold(X, meta, y, train, test, ch, seed, xin_cache, aug)
                add("lodo", d, seed, res)
                if ch == "imu":
                    for stage, (p, thr) in preds.items():
                        t = meta.iloc[test][["dataset", "activity", "label"]].copy()
                        t["fired"] = (p >= thr).astype(int)
                        t["model"] = f"cnn-{stage}"
                        act_rows.append(t.assign(seed=seed))
            row, pred = run_reference(X, meta, y, train, test)
            add("lodo", d, seed, [row])
            t = meta.iloc[test][["dataset", "activity", "label"]].copy()
            t["fired"], t["model"] = pred, "threshold"
            act_rows.append(t.assign(seed=seed))
            print(f"[seed {seed}] lodo {d} done ({time.time() - t_start:.0f}s)", flush=True)
        # ---- subject-wise 5-fold CV over the pooled data
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for k, (train, test) in enumerate(gkf.split(X, y, groups=meta.subject)):
            for ch in ("imu", "acc"):
                res, _ = run_fold(X, meta, y, train, test, ch, seed, xin_cache, aug)
                add("cv", f"fold{k}", seed, res)
            row, _ = run_reference(X, meta, y, train, test)
            add("cv", f"fold{k}", seed, [row])
            print(f"[seed {seed}] cv fold {k} done ({time.time() - t_start:.0f}s)", flush=True)

    out = REPORTS / ("baseline_augmented" if augment else "baseline")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    acts = pd.concat(act_rows)
    acts.to_csv(out / "per_activity_windows.csv.gz", index=False)
    write_report(df, acts, prep, out, seeds, time.time() - t_start, aug)
    print(f"report: {out / 'report.md'}")


# ---------------------------------------------------------------------------------- report
HDR = "| model | sensitivity | specificity | false alarms/h | AUROC | AUPRC (AP) | FA/h at 95% sens (oracle) |"
SEP = "|---|---|---|---|---|---|---|"


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3f} ± {s.std(ddof=0):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def _label(r) -> str:
    if r["model"] == "threshold":
        return "threshold reference"
    return f"CNN {r['channels']} {r['stage']}"


def write_report(df, acts, prep, out, seeds, seconds, aug=None) -> None:
    df = df.assign(variant=df.apply(_label, axis=1))
    order = ["threshold reference", "CNN imu stage1", "CNN imu stage2", "CNN acc stage1", "CNN acc stage2"]
    cols = ["sensitivity", "specificity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle"]
    L = []
    L.append("# Baseline results" + (" (with augmentation)" if aug else "") + "\n")
    if aug:
        L.append(f"Training windows are augmented on the fly (`augment.py`): {aug}. Validation and test "
                 "windows are untouched.\n")
    L.append(f"Seeds: {seeds}. Runtime {seconds / 60:.1f} min. Values are mean ± std over seeds "
             "(and over folds where noted). Sensitivity is at the threshold that gives ≥95% "
             "sensitivity on a subject-wise validation slice of the training data; the threshold "
             "reference has one fixed operating point. False alarms/h counts false-alarm windows "
             "while replaying held-out ADL windows (one stride = 2 s of replay), so overlapping "
             "windows of one event count more than once.\n")
    L.append("**Read these numbers as optimistic.** Every fall is a simulated fall (mattress, mostly "
             "young volunteers); the published literature reports far worse results on real falls.\n")

    L.append("**FA/h at 95% sens (oracle)** sets the threshold on the held-out scores themselves, so it "
             "measures ranking quality independent of threshold transfer across datasets; it is a "
             "diagnostic, never used to choose a model. The threshold reference has no score, so it is "
             "compared through its single operating point.\n")
    w = prep["windows"]
    L.append("## Data\n")
    L.append("| dataset | subjects | fall recordings | ADL recordings | fall windows | ADL windows | impact source |")
    L.append("|---|---|---|---|---|---|---|")
    for d, info in prep["datasets"].items():
        bw = w["by_dataset"][d]
        L.append(f"| {d} | {info['subjects']} | {info['fall_recordings']} | {info['adl_recordings']} | "
                 f"{bw['fall']} | {bw['adl']} | {info['impact_source']} |")
    ex = {d: i.get("excluded_subjects") for d, i in prep["datasets"].items() if i.get("excluded_subjects")}
    L.append(f"\nExcluded subjects (arm not still when standing, so mounting could not be aligned): {ex or 'none'}.")
    n_peak = sum(i["impact_source"].get("peak", 0) for i in prep["datasets"].values())
    n_fall_rec = sum(i["fall_recordings"] for i in prep["datasets"].values())
    L.append(f"\n**Magnitude-peak fallback** was used for the impact moment in {n_peak} of {n_fall_rec} "
             "fall recordings (all of Erciyes: it has no fall annotations). "
             f"Fall-recording windows excluded: {w['fall_recording_windows_excluded']} "
             f"(+{w['fall_recording_edge_ambiguous']} edge-ambiguous).\n")

    for proto, title in (("lodo", "Leave-one-dataset-out (headline)"), ("cv", "Subject-wise 5-fold CV (pooled datasets)")):
        sub = df[df.protocol == proto]
        L.append(f"## {title}\n")
        if proto == "lodo":
            for d in sorted(sub.fold.unique()):
                bw = prep["windows"]["by_dataset"][d]
                prev = bw["fall"] / (bw["fall"] + bw["adl"])
                L.append(f"### Held-out: {d} (AUPRC of a no-skill model = fall share = {prev:.3f})\n")
                L.append(HDR)
                L.append(SEP)
                s = sub[sub.fold == d]
                for v in order:
                    g = s[s.variant == v]
                    if len(g):
                        L.append(f"| {v} | " + " | ".join(_ms(g[c]) for c in cols) + " |")
                L.append("")
            L.append("### Macro average over held-out datasets\n")
            per = sub.groupby(["variant", "fold"])[cols].mean().groupby("variant").mean()
        else:
            L.append(HDR)
            L.append(SEP)
            per = None
        if per is not None:
            L.append(HDR)
            L.append(SEP)
            for v in order:
                if v in per.index:
                    L.append(f"| {v} | " + " | ".join(f"{per.loc[v, c]:.3f}" for c in cols) + " |")
        else:
            for v in order:
                g = sub[sub.variant == v]
                if len(g):
                    L.append(f"| {v} | " + " | ".join(_ms(g[c]) for c in cols) + " |")
        L.append("")

    th = df[df.model == "threshold"]
    pairs = sorted({(a, b) for a, b in zip(th.dip_g, th.peak_g)})
    degenerate = " The chosen dip is ≥1 g, so the free-fall condition is not selective at 25 Hz wrist data and the rule reduces to an impact-magnitude test." if min(a for a, _ in pairs) >= 1.0 else ""
    L.append(f"Threshold reference levels chosen on training windows: dip/peak (g) pairs seen {pairs}.{degenerate}\n")
    m2 = df[(df.stage == "stage2") & (df.channels == "imu")]
    L.append(f"Hard-negative mining: after stage 1, training ADL windows scoring above the decision "
             f"threshold (capped at {MINE_THRESHOLD_CAP:.0%} of ADL windows) are re-weighted ×{MINE_WEIGHT:g} "
             f"and the model is retrained from scratch (`stage2`). Windows mined per fold: median "
             f"{int(m2.mined.median())}, range {int(m2.mined.min())}–{int(m2.mined.max())}.\n")
    L.append(f"Model: 3 conv blocks (32/64/128), {int(df[df.model == 'cnn'].params.iloc[0])} parameters "
             "for 8 input channels (the accel-only variant is smaller).\n")

    # ---- per-activity
    a = acts[acts.model.isin(["cnn-stage2", "threshold"])]
    L.append("## Hard negatives: ADL activities that fire most (leave-one-dataset-out)\n")
    adl = a[a.label == 0].groupby(["model", "dataset", "activity"]).fired.agg(["mean", "size"]).reset_index()
    for m in ("cnn-stage2", "threshold"):
        top = adl[(adl.model == m) & (adl["size"] >= 20)].sort_values("mean", ascending=False).head(10)
        L.append(f"**{m}** (activity code, share of that activity's windows flagged as fall):\n")
        L.append("| dataset | activity | false-alarm rate | windows |")
        L.append("|---|---|---|---|")
        for r in top.itertuples():
            L.append(f"| {r.dataset} | {r.activity} | {r.mean:.2f} | {r.size} |")
        L.append("")
    L.append("## Missed falls by fall type (leave-one-dataset-out)\n")
    fl = a[a.label == 1].groupby(["model", "dataset", "activity"]).fired.agg(["mean", "size"]).reset_index()
    for m in ("cnn-stage2", "threshold"):
        low = fl[(fl.model == m) & (fl["size"] >= 20)].sort_values("mean").head(8)
        L.append(f"**{m}** (lowest detection rate):\n")
        L.append("| dataset | activity | detected | windows |")
        L.append("|---|---|---|---|")
        for r in low.itertuples():
            L.append(f"| {r.dataset} | {r.activity} | {r.mean:.2f} | {r.size} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
