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

from . import model as M
from .augment import Augment
from .canonical import FS
from .paths import PROCESSED, REPORTS
from .reference import ThresholdReference
from .rules import TARGET_SENSITIVITY
from .runner import FallData, Fold, Results, cv_folds, lodo_folds
from .windows import STRIDE_S

MINE_THRESHOLD_CAP = 0.10  # at most this share of training ADL windows is mined
MINE_WEIGHT = 3.0
CONDITIONS = ("imu", "acc", "reference")  # CNN imu, CNN accel-only, threshold reference


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
def run_fold(fold: Fold, channels: str, augment=None) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Train CNN stage 1 and stage 2 (after mining) on one fold; return metric rows + which test windows fired."""
    d = fold.data
    x_test, y_test = d.windows(fold.test, channels), d.y[fold.test]
    out, fired = [], {}
    weights = None
    mined = 0
    for stage in ("stage1", "stage2"):
        trained = fold.train_model(channels, augment=augment, weights=weights)
        thr = trained.threshold
        p_test = trained.model.scores(x_test)
        out.append({
            "model": "cnn", "channels": channels, "stage": stage, "threshold": thr, "mined": mined,
            "params": trained.model.n_params, **metrics(y_test, (p_test >= thr).astype(int), p_test),
        })
        fired[f"fired_{stage}"] = (p_test >= thr).astype(np.int8)
        if stage == "stage1":
            weights, mined = mine_hard_negatives(fold, channels, augment=augment)
        del trained
        torch.cuda.empty_cache()
    return out, fired


def mine_hard_negatives(fold: Fold, channels: str, folds: int = 3, augment=None) -> tuple[np.ndarray, int]:
    """Out-of-fold hard negative mining. Training subjects are split into ``folds`` groups; a model
    trained on the other groups scores the held-out group, and ADL windows that fire at that
    model's own >=95%-sensitivity threshold are hard negatives (the in-sample model would rarely
    fire on windows it has already fit). Returns per-window loss weights for ``fold.train`` and the count.
    """
    d, tr = fold.data, fold.train
    subs = d.meta.subject.iloc[tr]
    names = np.array(sorted(subs.unique()))
    np.random.default_rng(fold.seed + 1000).shuffle(names)
    hard_local, hard_scores = [], []
    for part in np.array_split(names, folds):
        held = subs.isin(part).to_numpy()
        trained = fold.with_training(tr[~held]).train_model(channels, augment=augment)
        p = trained.model.scores(d.windows(tr[held], channels))
        idx = np.where(held)[0]
        fire = (d.y[tr[idx]] == 0) & (p >= trained.threshold)
        hard_local += list(idx[fire])
        hard_scores += list(p[fire])
        del trained
        torch.cuda.empty_cache()
    hard = np.array(hard_local, dtype=int)
    cap = int(MINE_THRESHOLD_CAP * (d.y[tr] == 0).sum())
    hard = hard[np.argsort(-np.array(hard_scores))][:cap] if len(hard) else hard
    weights = np.ones(len(tr), dtype=np.float32)
    weights[hard] = MINE_WEIGHT
    return weights, len(hard)


def run_reference(fold: Fold) -> tuple[list[dict], dict[str, np.ndarray]]:
    """The threshold reference, its levels tuned on all of the fold's training subjects (training + validation)."""
    d = fold.data
    fit_idx = np.sort(np.r_[fold.train, fold.validation])
    ref = ThresholdReference().fit(d.windows(fit_idx, "acc"), d.y[fit_idx])
    pred = ref.predict(d.windows(fold.test, "acc"))
    row = {"model": "threshold", "channels": "acc", "stage": "-", "threshold": float("nan"),
           "mined": 0, "params": 0, "dip_g": ref.dip, "peak_g": ref.peak,
           **metrics(d.y[fold.test], pred, None)}
    return [row], {"fired": pred.astype(np.int8)}


# --------------------------------------------------------------------------------- driver
def _out(augment: bool):
    return REPORTS / ("baseline_augmented" if augment else "baseline")


def run(seeds: int = 3, augment: bool = False) -> None:
    """Every (condition, seed, fold) not yet in ``<report dir>/runs``, then the report."""
    t_start = time.time()
    data = FallData.load()
    aug = Augment() if augment else None
    results = Results(_out(augment) / "runs")
    print(f"{len(data.X)} windows, {int(data.y.sum())} fall; datasets {data.datasets}; device {M.device()}")
    for seed in range(seeds):
        for fold in lodo_folds(data, seed) + cv_folds(data, seed):
            for cond in CONDITIONS:
                if results.has(cond, seed, fold.key):
                    continue
                t0 = time.time()
                rows, fired = run_reference(fold) if cond == "reference" else run_fold(fold, cond, aug)
                rows = [{"protocol": fold.protocol, **r, "seconds": time.time() - t0} for r in rows]
                # the per-activity tables need which held-out windows fired (CNN imu and the reference)
                arrays = {"test": fold.test, **fired} if fold.protocol == "lodo" and cond != "acc" else None
                results.save(cond, seed, fold.key, rows, arrays)
            print(f"[seed {seed}] {fold.protocol} {fold.name} done ({time.time() - t_start:.0f}s)", flush=True)
    report(augment, seeds)


def report(augment: bool = False, seeds: int | None = None) -> None:
    """Rebuild ``per_activity_windows.csv.gz`` and ``report.md`` from the stored results (seeds < ``seeds``)."""
    out = _out(augment)
    results = Results(out / "runs")
    df = results.table()
    if seeds is not None:
        df = df[df.seed < seeds]
    meta = pd.read_csv(PROCESSED / "meta.csv")
    act_rows = []
    lodo = df[df.protocol == "lodo"]
    for seed in sorted(lodo.seed.unique()):
        for d in sorted(lodo[lodo.seed == seed].fold.unique()):
            for cond, models in (("imu", {"fired_stage1": "cnn-stage1", "fired_stage2": "cnn-stage2"}),
                                 ("reference", {"fired": "threshold"})):
                a = results.arrays(cond, int(seed), d)
                for k, m in models.items():
                    t = meta.iloc[a["test"]][["dataset", "activity", "label"]].copy()
                    t["fired"], t["model"] = a[k].astype(int), m
                    act_rows.append(t.assign(seed=seed))
    acts = pd.concat(act_rows)
    acts.to_csv(out / "per_activity_windows.csv.gz", index=False)
    timed = df.groupby(["condition", "seed", "fold"]).seconds.first() if "seconds" in df else pd.Series(dtype=float)
    seconds = timed.sum() if len(timed) and timed.notna().all() else None
    prep = json.loads((PROCESSED / "prepare_log.json").read_text())
    write_report(df, acts, prep, out, df.seed.nunique(), seconds, Augment() if augment else None)
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
    runtime = f"{seconds / 60:.1f} min" if seconds is not None else "not recorded (results converted from an earlier run)"
    L.append(f"Seeds: {seeds}. Runtime {runtime}. Values are mean ± std over seeds "
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
    L.append(f"Model: 3 conv blocks (32/64/128), {int(df[(df.model == 'cnn') & (df.channels == 'imu')].params.iloc[0])} parameters "
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
