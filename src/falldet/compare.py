"""Compare two baseline runs (e.g. without and with augmentation), paired by protocol, fold, seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .runner import Results

METRICS = ["sensitivity", "specificity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle"]
KEY = ["protocol", "fold", "seed", "model", "channels", "stage"]


def _variant(df: pd.DataFrame) -> pd.Series:
    return df.channels + " " + df.stage


def run(base_dir: Path, other_dir: Path) -> None:
    a = Results(base_dir / "runs").table()
    b = Results(other_dir / "runs").table()
    a, b = (d[d.model == "cnn"].copy() for d in (a, b))
    j = a.merge(b, on=KEY, suffixes=("_base", "_aug"))
    j["variant"] = _variant(j)
    L = [f"# Comparison: {base_dir.name} → {other_dir.name}\n",
         "Paired by protocol, fold and seed (same splits, same seeds). Δ = augmented − baseline. "
         "`wins` is the share of paired (fold, seed) runs where the augmented model is better "
         "(higher for sensitivity/specificity/AUROC/AUPRC, lower for false alarms). `ap` is AUPRC "
         "(average precision).\n"]
    lower_better = {"false_alarms_per_h", "fa_h_at_95sens_oracle"}

    def table(g: pd.DataFrame, title: str, macro_over: str | None) -> None:
        L.append(f"## {title}\n")
        L.append("| variant | metric | baseline | augmented | Δ | wins |")
        L.append("|---|---|---|---|---|---|")
        for v, gv in g.groupby("variant"):
            for m in METRICS:
                x, y = gv[f"{m}_base"], gv[f"{m}_aug"]
                if macro_over:  # macro average over folds, per seed
                    xs = gv.groupby("seed")[[f"{m}_base"]].mean()[f"{m}_base"]
                    ys = gv.groupby("seed")[[f"{m}_aug"]].mean()[f"{m}_aug"]
                    base_v, aug_v = xs.mean(), ys.mean()
                else:
                    base_v, aug_v = x.mean(), y.mean()
                d = y - x
                wins = (d < 0).mean() if m in lower_better else (d > 0).mean()
                L.append(f"| {v} | {m} | {base_v:.3f} | {aug_v:.3f} | {aug_v - base_v:+.3f} | {wins:.0%} |")
        L.append("")

    lodo = j[j.protocol == "lodo"]
    table(lodo, "Leave-one-dataset-out, macro over held-out datasets", "fold")
    for d in sorted(lodo.fold.unique()):
        table(lodo[lodo.fold == d], f"Held-out: {d}", None)
    table(j[j.protocol == "cv"], "Subject-wise 5-fold CV (pooled)", None)
    out = other_dir / "comparison.md"
    out.write_text("\n".join(L))
    print(f"comparison: {out}")
