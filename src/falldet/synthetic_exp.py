"""Does synthetic data (LAFAN1 virtual IMU, ``synth.py``) improve the model on real unseen datasets?

Leave-one-dataset-out over the real datasets, stage-1 CNN, one input variant. Synthetic windows are
only ever added to the *training* set; validation (early stopping, threshold) and test windows are
real. Conditions (``aug`` = on-the-fly augmentation, ``augment.py``):

* ``real``            real training data only (the control)
* ``mix-25`` / ``mix-100``  real + 25% / all synthetic windows
* ``mix-adl``         real + synthetic ADL windows only (negatives)
* ``mix-fall``        real + synthetic fall windows only
* ``synth-only``      synthetic windows only (how far synthetic alone transfers)
* ``pretrain-ft``     train on synthetic, then continue on real
* ``real-noaug`` / ``mix-100-noaug``  the control and the full mix without augmentation
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .augment import Augment
from .baseline import metrics
from .paths import PROCESSED, REPORTS
from .rules import keep_falls
from .runner import FallData, Fold, Results, Trained, lodo_folds

COLS = ["sensitivity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle"]
LOWER = {"false_alarms_per_h", "fa_h_at_95sens_oracle"}
CONDITIONS = ["real", "mix-25", "mix-100", "mix-adl", "mix-fall", "synth-only", "pretrain-ft",
              "real-noaug", "mix-100-noaug"]
REFERENCE = {c: ("real-noaug" if c.endswith("noaug") else "real") for c in CONDITIONS}


def train(cond: str, fold: Fold, xs, ys, aug, channels: str) -> Trained:
    """Train one model for ``cond`` on ``fold``; ``xs``/``ys`` are the synthetic windows and labels."""
    rng = np.random.default_rng(fold.seed + 500)  # the mix-25 subset depends on the seed only
    a = None if cond.endswith("noaug") else aug
    if cond == "synth-only":
        return fold.train_model(channels, augment=a, extra=(xs, ys), own_windows=False)
    if cond == "pretrain-ft":
        base = fold.train_model(channels, augment=a, extra=(xs, ys), own_windows=False).model
        d = fold.data
        xv, yv = d.windows(fold.validation, channels), d.y[fold.validation]
        model = base.continue_training(d.windows(fold.train, channels), d.y[fold.train], xv, yv, seed=fold.seed,
                                       freeze_features=False, lr=1e-3, batch=256, max_epochs=40, patience=6, augment=a)
        return Trained(model, keep_falls(model.scores(xv), yv))
    if not cond.startswith("mix"):
        return fold.train_model(channels, augment=a)
    keep = np.ones(len(xs), bool)
    if cond == "mix-25":
        keep = rng.random(len(xs)) < 0.25
    elif cond == "mix-adl":
        keep = ys == 0
    elif cond == "mix-fall":
        keep = ys == 1
    return fold.train_model(channels, augment=a, extra=(xs[keep], ys[keep]))


def evaluate(cond: str, fold: Fold, xs, ys, aug, channels: str) -> dict:
    """Metrics on the fold's held-out dataset of the model trained for ``cond``."""
    trained = train(cond, fold, xs, ys, aug, channels)
    p = trained.model.scores(fold.data.windows(fold.test, channels))
    y = fold.data.y[fold.test]
    row = metrics(y, (p >= trained.threshold).astype(int), p)
    del trained
    torch.cuda.empty_cache()
    return row


def run(seeds: int = 8, channels: str = "acc") -> None:
    t0 = time.time()
    data = FallData.load()
    xs = np.load(PROCESSED / "synthetic_X.npy")
    ys = pd.read_csv(PROCESSED / "synthetic_meta.csv").label.to_numpy().astype(np.int64)
    aug = Augment()
    out = REPORTS / f"synthetic_{channels}"
    results = Results(out / "runs")  # resume: stored (condition, seed, fold) results are kept
    for cond in CONDITIONS:
        for seed in range(seeds):
            for fold in lodo_folds(data, seed):
                if not results.has(cond, seed, fold.key):
                    results.save(cond, seed, fold.key, [evaluate(cond, fold, xs, ys, aug, channels)])
        print(f"[{cond}] done ({time.time() - t0:.0f}s)", flush=True)
    df = results.table()
    write_report(df[df.seed < seeds], out, seeds, channels, time.time() - t0)
    print(f"report: {out / 'report.md'}")


def write_report(df: pd.DataFrame, out, seeds: int, channels: str, seconds: float) -> None:
    import json

    syn = json.loads((PROCESSED / "synthetic_log.json").read_text())
    per_seed = df.groupby(["condition", "seed"])[COLS].mean()  # macro over held-out datasets
    mean = per_seed.groupby("condition").mean()
    std = per_seed.groupby("condition").std()
    L = ["# Does synthetic data help?\n"]
    L.append(f"Seeds: {seeds}, input: `{channels}`, stage-1 CNN, leave-one-dataset-out over the real datasets, "
             f"runtime {seconds / 60:.1f} min. Synthetic data: LAFAN1 mocap through a numpy virtual wrist IMU "
             f"(`synth.py`, ADR 0002): {syn['windows']['fall']} fall windows from {sum(syn['fall_events'].values())} "
             f"fall events ({syn['fall_events']}) and {syn['windows']['adl']} ADL windows, 5 subjects. Each fall "
             "event contributes 7 windows with the impact at different positions, so the fall windows are far "
             "fewer independent samples than their count. Synthetic data is only ever in the training set; "
             "validation, threshold and test are real. Conditions are described in `synthetic_exp.py`. "
             "Cells are the macro average over held-out datasets, mean ± std over seeds. Sensitivity and "
             "false alarms/h use the validation-slice threshold; AUROC, AUPRC and oracle FA/h are threshold-free. "
             "Δ is the paired difference to the reference condition (`real`, or `real-noaug` for the `-noaug` "
             "rows) over seeds, with its standard error; wins = seeds in which the condition beats the "
             "reference on oracle FA/h. All real falls are simulated.\n")
    L.append("## Macro average over held-out datasets\n")
    L.append("| condition | AUROC | AUPRC | oracle FA/h at 95% sens | Δ oracle FA/h (± SE) | wins | sensitivity | false alarms/h |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        if c not in mean.index:
            continue
        ref = REFERENCE[c]
        d = per_seed.loc[c].fa_h_at_95sens_oracle - per_seed.loc[ref].fa_h_at_95sens_oracle
        delta = "-" if c == ref else f"{d.mean():+.1f} ± {d.std() / np.sqrt(len(d)):.1f}"
        wins = "-" if c == ref else f"{(d < 0).sum()}/{len(d)}"
        f = lambda k, p=3: f"{mean.loc[c, k]:.{p}f} ± {std.loc[c, k]:.{p}f}"  # noqa: E731
        L.append(f"| {c} | {f('auroc')} | {f('ap')} | {f('fa_h_at_95sens_oracle', 1)} | {delta} | {wins} | "
                 f"{f('sensitivity')} | {f('false_alarms_per_h', 1)} |")
    L.append("")
    for metric, title in (("fa_h_at_95sens_oracle", "oracle FA/h at 95% sens"), ("ap", "AUPRC")):
        L.append(f"## Per held-out dataset: {title} (mean over seeds)\n")
        pv = df.groupby(["condition", "fold"])[metric].mean().unstack()
        L.append("| condition | " + " | ".join(pv.columns) + " |")
        L.append("|---|" + "---|" * len(pv.columns))
        for c in CONDITIONS:
            if c in pv.index:
                L.append(f"| {c} | " + " | ".join(f"{v:.3f}" if metric == "ap" else f"{v:.1f}" for v in pv.loc[c]) + " |")
        L.append("")
    Xr, mr = np.load(PROCESSED / "X.npy"), pd.read_csv(PROCESSED / "meta.csv")
    pk = lambda sel, sl: np.percentile(np.linalg.norm(Xr[sel][:, :, sl], axis=2).max(1), [10, 50, 90]).round(1).tolist()  # noqa: E731
    L.append("## Synthetic data quality\n")
    L.append("Peak magnitude per window, p10 / p50 / p90. Real datasets (range over the four):\n")
    L.append("| statistic | synthetic | real, per dataset |")
    L.append("|---|---|---|")
    for lab, key, sl, unit in ((1, "fall_peak_acc_g_p10_50_90", slice(0, 3), "g"), (0, "adl_peak_acc_g_p10_50_90", slice(0, 3), "g")):
        real = "; ".join(f"{d} {pk(((mr.dataset == d) & (mr.label == lab)).to_numpy(), sl)}" for d in sorted(mr.dataset.unique()))
        L.append(f"| {'fall' if lab else 'ADL'} windows, accel ({unit}) | {syn[key]} | {real} |")
    realg = "; ".join(f"{d} {pk(((mr.dataset == d) & (mr.label == 1)).to_numpy(), slice(3, 6))}" for d in sorted(mr.dataset.unique()))
    L.append(f"| fall windows, gyro (rad/s), synthetic p50 only | {syn['fall_peak_gyro_rad_s_p50']:.1f} | {realg} |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
