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

from . import model as M
from .augment import Augment
from .baseline import choose_threshold, metrics, split_validation
from .cli import PROCESSED, REPORTS

COLS = ["sensitivity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle"]
LOWER = {"false_alarms_per_h", "fa_h_at_95sens_oracle"}
CONDITIONS = ["real", "mix-25", "mix-100", "mix-adl", "mix-fall", "synth-only", "pretrain-ft",
              "real-noaug", "mix-100-noaug"]
REFERENCE = {c: ("real-noaug" if c.endswith("noaug") else "real") for c in CONDITIONS}


def train(cond: str, seed: int, xr, yr, tr, va, xs, ys, aug) -> M.FallCNN:
    """Train one model for ``cond``; ``tr``/``va`` index the real arrays ``xr``/``yr``."""
    rng = np.random.default_rng(seed + 500)
    a = None if cond.endswith("noaug") else aug
    ones = lambda n: np.ones(n, np.float32)  # noqa: E731
    if cond == "synth-only":
        return M.fit(xs, ys, ones(len(xs)), xr[va], yr[va], seed, augment=a)
    if cond == "pretrain-ft":
        base = M.fit(xs, ys, ones(len(xs)), xr[va], yr[va], seed, augment=a)
        return M.finetune(base, xr[tr], yr[tr], xr[va], yr[va], seed, freeze_features=False,
                          lr=1e-3, batch=256, max_epochs=40, patience=6, augment=a)
    x, y = xr[tr], yr[tr]
    if cond.startswith("mix"):
        keep = np.ones(len(xs), bool)
        if cond == "mix-25":
            keep = rng.random(len(xs)) < 0.25
        elif cond == "mix-adl":
            keep = ys == 0
        elif cond == "mix-fall":
            keep = ys == 1
        x, y = np.concatenate([x, xs[keep]]), np.concatenate([y, ys[keep]])
    return M.fit(x, y, ones(len(x)), xr[va], yr[va], seed, augment=a)


def run(seeds: int = 8, channels: str = "acc") -> None:
    t0 = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    xr = M.make_inputs(X, channels)
    xs = M.make_inputs(np.load(PROCESSED / "synthetic_X.npy"), channels)
    ys = pd.read_csv(PROCESSED / "synthetic_meta.csv").label.to_numpy().astype(np.int64)
    datasets = sorted(meta.dataset.unique())
    aug = Augment()
    out = REPORTS / f"synthetic_{channels}"
    out.mkdir(parents=True, exist_ok=True)
    prior = out / "metrics.csv"
    rows = pd.read_csv(prior).to_dict("records") if prior.exists() else []
    done = {r["cond"] for r in rows}  # resume: conditions finished by an interrupted run are kept
    for cond in CONDITIONS:
        if cond in done:
            continue
        for seed in range(seeds):
            for d in datasets:
                test = np.where(meta.dataset == d)[0]
                tr, va = split_validation(meta, np.where(meta.dataset != d)[0], seed)
                net = train(cond, seed, xr, y, tr, va, xs, ys, aug)
                thr = choose_threshold(M.predict_proba(net, xr[va]), y[va])
                p = M.predict_proba(net, xr[test])
                rows.append({"cond": cond, "seed": seed, "fold": d,
                             **metrics(y[test], (p >= thr).astype(int), p)})
                del net
                torch.cuda.empty_cache()
        print(f"[{cond}] done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    write_report(pd.DataFrame(rows), out, seeds, channels, time.time() - t0)
    print(f"report: {out / 'report.md'}")


def write_report(df: pd.DataFrame, out, seeds: int, channels: str, seconds: float) -> None:
    import json

    syn = json.loads((PROCESSED / "synthetic_log.json").read_text())
    per_seed = df.groupby(["cond", "seed"])[COLS].mean()  # macro over held-out datasets
    mean = per_seed.groupby("cond").mean()
    std = per_seed.groupby("cond").std()
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
        pv = df.groupby(["cond", "fold"])[metric].mean().unstack()
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
