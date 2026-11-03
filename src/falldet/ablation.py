"""Per-transform augmentation ablation (leave-one-dataset-out, stage-1 CNN, one input variant).

Configurations: ``none``; ``all``; ``only-<t>`` (one transform on); ``without-<t>`` (all but one), for
the six transforms in ``augment.TRANSFORMS``. Splits, seeds and training match the baseline, so ``none``
and ``all`` should agree with its stage-1 rows within seed noise (GPU training is not bit-reproducible).
"""

from __future__ import annotations

import time

import pandas as pd
import torch

from .augment import TRANSFORMS, Augment
from .baseline import metrics
from .paths import REPORTS
from .runner import FallData, Results, lodo_folds

COLS = ["sensitivity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle"]
LOWER = {"false_alarms_per_h", "fa_h_at_95sens_oracle"}


def configs() -> dict[str, Augment | None]:
    cfg: dict[str, Augment | None] = {"none": None, "all": Augment()}
    for t in TRANSFORMS:
        cfg[f"only-{t}"] = Augment(enabled=(t,))
    for t in TRANSFORMS:
        cfg[f"without-{t}"] = Augment(enabled=tuple(x for x in TRANSFORMS if x != t))
    return cfg


def run(seeds: int = 5, channels: str = "acc") -> None:
    t0 = time.time()
    data = FallData.load()
    out = REPORTS / f"ablation_{channels}"
    results = Results(out / "runs")
    for name, aug in configs().items():
        for seed in range(seeds):
            for fold in lodo_folds(data, seed):
                if results.has(name, seed, fold.key):
                    continue
                trained = fold.train_model(channels, augment=aug)
                p = trained.model.scores(data.windows(fold.test, channels))
                y = data.y[fold.test]
                results.save(name, seed, fold.key, [metrics(y, (p >= trained.threshold).astype(int), p)])
                del trained
                torch.cuda.empty_cache()
        print(f"[{name}] done ({time.time() - t0:.0f}s)", flush=True)
    df = results.table().rename(columns={"condition": "config"})
    write_report(df[df.seed < seeds], out, seeds, channels, time.time() - t0)
    print(f"report: {out / 'report.md'}")


def _macro(df: pd.DataFrame) -> pd.DataFrame:
    """Per config: mean over folds then seeds."""
    return df.groupby(["config", "seed"])[COLS].mean().groupby("config").agg(["mean", "std"])


def _wins(df: pd.DataFrame, a: str, b: str, metric: str) -> float:
    """Share of seeds (macro over folds) where config ``a`` beats ``b``."""
    m = df.groupby(["config", "seed"])[metric].mean().unstack(0)
    return float((m[a] < m[b]).mean() if metric in LOWER else (m[a] > m[b]).mean())


def write_report(df: pd.DataFrame, out, seeds: int, channels: str, seconds: float) -> None:
    mac = _macro(df)
    L = ["# Augmentation ablation\n"]
    L.append(f"Seeds: {seeds}, input: `{channels}`, stage-1 CNN, leave-one-dataset-out. Runtime "
             f"{seconds / 60:.1f} min. Each cell is the macro average over the held-out datasets, then mean "
             "(± std) over seeds. Sensitivity and false alarms/h use the standard validation-slice "
             "threshold; AUROC, AUPRC and the oracle FA/h are threshold-free ranking measures, so judge "
             "transforms mainly by those. `none` and `all` correspond to the baseline and baseline_augmented "
             "stage-1 rows (agreeing only within seed noise: individual folds differ by up to ~2x between "
             "runs with the same seed; probably early stopping amplifying GPU nondeterminism, not verified). "
             "`only-X` turns on one transform; `without-X` turns on all but one. "
             "`wins vs none/all` is the share of seeds whose oracle FA/h is lower than that reference. "
             "Transform strengths are untuned (`augment.py`); all falls simulated.\n")
    hdr = "| config | AUROC | AUPRC | oracle FA/h at 95% sens | sensitivity | false alarms/h | wins vs none | wins vs all |"
    sep = "|---|---|---|---|---|---|---|---|"

    def line(c: str) -> str:
        r = mac.loc[c]
        f = lambda k, p=3: f"{r[(k, 'mean')]:.{p}f} ± {r[(k, 'std')]:.{p}f}"  # noqa: E731
        return (f"| {c} | {f('auroc')} | {f('ap')} | {f('fa_h_at_95sens_oracle', 1)} | {f('sensitivity')} | "
                f"{f('false_alarms_per_h', 1)} | {_wins(df, c, 'none', 'fa_h_at_95sens_oracle'):.0%} | "
                f"{_wins(df, c, 'all', 'fa_h_at_95sens_oracle'):.0%} |")

    L.append("## Macro average over held-out datasets\n")
    L.append(hdr); L.append(sep)
    for c in ["none", "all"] + [f"only-{t}" for t in TRANSFORMS] + [f"without-{t}" for t in TRANSFORMS]:
        if c in mac.index:
            L.append(line(c))
    L.append("")
    ref = mac.loc["all", ("fa_h_at_95sens_oracle", "mean")]
    non = mac.loc["none", ("fa_h_at_95sens_oracle", "mean")]
    L.append("## Contribution of each transform (oracle FA/h at 95% sens, lower is better)\n")
    L.append("Both columns are changes in oracle FA/h; positive means the transform helps. "
             "`alone` = none − only-X (false alarms it removes by itself); `if removed` = "
             "without-X − all (false alarms added when it is dropped from the full set). "
             "Seed std of a macro value is 10–30 FA/h, so differences below about 15 are noise.\n")
    L.append("| transform | alone | if removed |")
    L.append("|---|---|---|")
    for t in TRANSFORMS:
        if f"only-{t}" in mac.index and f"without-{t}" in mac.index:
            a = non - mac.loc[f"only-{t}", ("fa_h_at_95sens_oracle", "mean")]
            b = mac.loc[f"without-{t}", ("fa_h_at_95sens_oracle", "mean")] - ref
            L.append(f"| {t} | {a:+.1f} | {b:+.1f} |")
    L.append("")
    L.append("## Per held-out dataset: oracle FA/h at 95% sens (mean over seeds)\n")
    pv = df.groupby(["config", "fold"]).fa_h_at_95sens_oracle.mean().unstack()
    L.append("| config | " + " | ".join(pv.columns) + " |")
    L.append("|---|" + "---|" * len(pv.columns))
    for c in pv.index:
        L.append(f"| {c} | " + " | ".join(f"{v:.1f}" for v in pv.loc[c]) + " |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
