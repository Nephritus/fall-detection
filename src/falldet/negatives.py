"""Does real free-living data as extra *negatives* (Capture-24) cut false alarms without losing falls?

Leave-one-fall-dataset-out (as in ``baseline.py``), accel-only augmented CNN, plus Capture-24 windows
in training. Capture-24 has no falls, so its windows are all negatives. Participants are split by
person: P001-P100 (usable ones) give training and validation negatives, P101-P151 are never trained on
and measure free-living alarm events per day.

Conditions (every one sees the same fall data and the same fold/seed splits):

* ``real``         the control: fall datasets only
* ``c24-random``   + ``RANDOM_K`` uniformly random Capture-24 windows per training participant
* ``c24-mined``    + 1/3 of that as random windows and 2/3 as the windows the control model scores
                   highest (hard negatives; non-maximum suppression so neighbours are not duplicated)

Both Capture-24 conditions add the same number of windows. Validation (early stopping, threshold) is
the fall datasets' validation slice, plus random windows of held-out Capture-24 training participants
for the Capture-24 conditions. Metrics, for the held-out fall dataset and the held-out Capture-24 people:

* lab ranking: AUROC, AUPRC, oracle FA/h at 95% sensitivity (the earlier lab replay metric)
* free-living: alarm events/day at the validation-slice threshold, and
  - sensitivity (fall windows of the held-out dataset) at a budget of B alarm events/day, threshold-free
  - alarm events/day at 95% fall sensitivity (oracle threshold on the held-out falls)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .model import FallModel
from .alarms import REFRACTORY_S, FalseAlarms, per_day_ci, sensitivity_at_budget
from .augment import Augment
from .baseline import metrics
from .canonical import FS
from .paths import REPORTS
from .freeliving import OUT, SOURCE
from .runner import FallData, Fold, Results, Trained, lodo_folds

RANDOM_K = 1500  # Capture-24 windows added per training participant
MINED_FRACTION = 2 / 3
VAL_K = 800  # random windows per Capture-24 validation participant
BUDGETS = (10, 25, 50, 100)  # alarm events per day
CONDITIONS = ("real", "c24-random", "c24-mined")
RUNS = REPORTS / "negatives" / "runs"  # negatives.run's conditions and the scale conditions c24-random-<K>


# --------------------------------------------------------------------------------- windows
def mined_windows(net: FallModel, pids: list[str], k: int, sep: int = 3) -> np.ndarray:
    """Top-scoring windows per participant under ``net`` (greedy, ``sep`` windows apart)."""
    out = []
    for pid in pids:
        sc = net.score_signal(SOURCE.signal(pid))
        taken = np.zeros(len(sc), bool)
        pick = []
        for i in np.argsort(-sc)[: k * 10]:
            if not taken[max(0, i - sep) : i + sep + 1].any():
                taken[i] = True
                pick.append(i)
                if len(pick) == k:
                    break
        out.append(SOURCE.windows_at(pid, np.array(pick)))
    return np.concatenate(out)


# ------------------------------------------------------------------------------ free-living
def free_living(scores: list[np.ndarray], fall_scores: np.ndarray, val_thr: float):
    """Row of free-living numbers plus the alarm curve (thresholds descending, events/day ascending)."""
    fa = FalseAlarms.from_window_scores(scores)
    grid, ev = fa.curve()
    t95 = np.quantile(fall_scores, 0.05, method="lower")
    row = {"fl_events_per_day_at_val_thr": fa.per_day(val_thr), "fl_events_per_day_at_95sens": fa.per_day(t95)}
    return row, {"grid": grid, "ev": ev, "falls": fall_scores}


# --------------------------------------------------------------------------------- driver
def evaluate(trained: Trained, fold: Fold, test_acc: list[np.ndarray]) -> tuple[dict, dict]:
    """Lab metrics on the fold's held-out fall dataset, free-living numbers on the held-out Capture-24 people."""
    net, thr, y = trained.model, trained.threshold, fold.data.y[fold.test]
    p = net.scores(fold.data.windows(fold.test, "acc"))
    lab = metrics(y, (p >= thr).astype(int), p)
    scores = [net.score_signal(a) for a in test_acc]
    fl, curve = free_living(scores, p[y == 1], thr)
    row = {"threshold": thr, **{k: lab[k] for k in
           ("sensitivity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle")}, **fl}
    return row, curve


def run(seed: int) -> None:
    t0 = time.time()
    data, results, aug = FallData.load(), Results(RUNS), Augment()
    people = SOURCE.split(seed)
    test_acc = [SOURCE.signal(p) for p in people.held_out]
    print(f"seed {seed}: C24 train {len(people.train)}, val {len(people.validation)}, test {len(people.held_out)} "
          f"participants ({sum(len(a) for a in test_acc) / FS / 3600:.0f} h)", flush=True)
    x_c24_val = SOURCE.windows(people.validation, VAL_K, seed, "validation")
    for fold in lodo_folds(data, seed):
        run_fold(fold, results, people, test_acc, x_c24_val, aug)
        print(f"[seed {seed}] {fold.name} done ({time.time() - t0:.0f}s)", flush=True)


def run_fold(fold: Fold, results: Results, people, test_acc: list[np.ndarray], x_c24_val: np.ndarray, aug) -> None:
    """The conditions of one fold that are not stored yet. ``c24-mined`` mines with the fold's ``real`` model,
    retrained (same seed) when ``real`` itself is already stored."""
    todo = [c for c in CONDITIONS if not results.has(c, fold.seed, fold.key)]
    real = None
    if "real" in todo:
        real = fold.train_model("acc", augment=aug)
        row, curve = evaluate(real, fold, test_acc)
        results.save("real", fold.seed, fold.key, [row], curve)
    added = {}
    if "c24-random" in todo:
        added["c24-random"] = SOURCE.windows(people.train, RANDOM_K, fold.seed, "train", fold.name)
    if "c24-mined" in todo:
        if real is None:
            real = fold.train_model("acc", augment=aug)
        n_mined = round(RANDOM_K * MINED_FRACTION)
        added["c24-mined"] = np.concatenate([SOURCE.windows(people.train, RANDOM_K - n_mined, fold.seed, "mined-random", fold.name),
                                             mined_windows(real.model, people.train, n_mined)])
    del real
    torch.cuda.empty_cache()
    for cond, xc in added.items():
        trained = fold.train_model("acc", augment=aug, extra=(xc, np.zeros(len(xc), np.int64)), extra_validation=x_c24_val)
        row, curve = evaluate(trained, fold, test_acc)
        results.save(cond, fold.seed, fold.key, [row], curve)
        del trained
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------- report
COLS = ["auroc", "ap", "fa_h_at_95sens_oracle", "sensitivity", "false_alarms_per_h",
        "fl_events_per_day_at_val_thr", *[f"sens_at_{b}_per_day" for b in BUDGETS], "fl_events_per_day_at_95sens"]
LOWER = {"fa_h_at_95sens_oracle", "false_alarms_per_h", "fl_events_per_day_at_val_thr", "fl_events_per_day_at_95sens"}


def _table(results: Results) -> pd.DataFrame:
    """Stored rows plus the sensitivity at each alarm budget, read off every key's free-living curve."""
    df = results.table()
    sens = {f"sens_at_{b}_per_day": [] for b in BUDGETS}
    for r in df.itertuples():
        cv = results.arrays(r.condition, r.seed, r.fold)
        for b in BUDGETS:
            sens[f"sens_at_{b}_per_day"].append(sensitivity_at_budget(cv["grid"], cv["ev"], cv["falls"], b))
    return df.assign(**sens)


def report() -> None:
    out = REPORTS / "negatives"
    df = _table(Results(RUNS))
    df = df[df.condition.isin(CONDITIONS)]
    seeds = df.seed.nunique()
    per = df.groupby(["condition", "seed"])[COLS].mean()  # macro over held-out fall datasets
    mean, std = per.groupby("condition").mean(), per.groupby("condition").std()
    L = ["# Capture-24 as training negatives\n"]
    L.append(f"Seeds: {seeds}. Leave-one-fall-dataset-out, accel-only augmented CNN (stage 1). Capture-24 "
             f"participants P001-P100 (usable ones: 80% train / 20% validation negatives) are added to training; "
             f"P101-P151 are held out and measure free-living alarm events per day (30 s merge). `c24-random` adds "
             f"{RANDOM_K} random windows per training person; `c24-mined` adds the same number, two thirds of "
             "them the highest-scoring windows under the control model (hard negatives). Cells: macro average over "
             "the four held-out fall datasets, mean ± std over seeds. The free-living columns use the **held-out** "
             "Capture-24 people, whose data the model never saw. **Sensitivity at a budget** = share of the "
             "held-out dataset's fall windows detected at the lowest threshold that keeps free-living alarms at or "
             "below B events/day (threshold-free: it reads the operating curve, so it is not affected by how the "
             "added negatives shift score calibration). Fall windows are simulated falls, so real sensitivity is "
             "lower, and per-window sensitivity is not per-fall detection.\n")
    L.append("## Macro over held-out fall datasets\n")
    names = {"auroc": "AUROC", "ap": "AUPRC", "fa_h_at_95sens_oracle": "lab oracle FA/h", "sensitivity": "lab sens @val thr",
             "false_alarms_per_h": "lab FA/h @val thr", "fl_events_per_day_at_val_thr": "free-living alarms/day @val thr",
             "fl_events_per_day_at_95sens": "free-living alarms/day @95% sens",
             **{f"sens_at_{b}_per_day": f"sens @{b}/day" for b in BUDGETS}}
    L.append("| condition | " + " | ".join(names[c] for c in COLS) + " |")
    L.append("|---|" + "---|" * len(COLS))
    for c in CONDITIONS:
        L.append(f"| {c} | " + " | ".join(f"{mean.loc[c, k]:.3f} ± {std.loc[c, k]:.3f}" if k in ("auroc", "ap", "sensitivity") or k.startswith("sens_at")
                                           else f"{mean.loc[c, k]:.1f} ± {std.loc[c, k]:.1f}" for k in COLS) + " |")
    L.append("")
    L.append("## Paired difference to the control\n")
    L.append("Δ = condition − `real`, per seed (macro over folds); mean ± standard error; wins = seeds in which the "
             "condition is better.\n")
    L.append("| condition | metric | Δ | wins |")
    L.append("|---|---|---|---|")
    for c in CONDITIONS[1:]:
        for k in COLS:
            d = per.loc[c][k] - per.loc["real"][k]
            better = (d < 0) if k in LOWER else (d > 0)
            L.append(f"| {c} | {names[k]} | {d.mean():+.3f} ± {d.std() / np.sqrt(len(d)):.3f} | {int(better.sum())}/{len(d)} |")
    L.append("")
    for k, title in (("sens_at_50_per_day", "Sensitivity at 50 alarms/day"), ("fl_events_per_day_at_95sens", "Free-living alarms/day at 95% sensitivity")):
        L.append(f"## Per held-out fall dataset: {title}\n")
        pv = df.groupby(["condition", "fold"])[k].mean().unstack()
        L.append("| condition | " + " | ".join(pv.columns) + " |")
        L.append("|---|" + "---|" * len(pv.columns))
        for c in CONDITIONS:
            L.append(f"| {c} | " + " | ".join(f"{v:.3f}" if k.startswith("sens") else f"{v:.1f}" for v in pv.loc[c]) + " |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
    print(f"report: {out / 'report.md'}")


# ------------------------------------------------------------------------ final-model analysis
def c24_negatives(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(training, validation) random Capture-24 windows for the final models (people P001-P100)."""
    people = SOURCE.split(seed)
    return (SOURCE.windows(people.train, RANDOM_K, seed, "train", "final"),
            SOURCE.windows(people.validation, VAL_K, seed, "validation"))


def final(seeds: int = 3) -> None:
    """Final models with Capture-24 negatives: free-living report plus a comparison with the models
    trained without them, on the same held-out people."""
    from . import freeliving as F
    from .freeliving import RULES

    F.run(seeds, negatives=True)
    prep = SOURCE.log()
    test = SOURCE.split(0).held_out
    roots = {"without Capture-24 negatives": OUT, "with Capture-24 negatives (c24-random)": OUT / "c24random"}
    out = REPORTS / "freeliving_c24random"
    rng = np.random.default_rng(0)
    L = ["# Final models with and without Capture-24 negatives\n"]
    L.append(f"Seeds: {seeds} (same seed numbers, same fall-data splits). Both models are scored on the same "
             f"{len(test)} held-out Capture-24 people (P101-P151, never trained on), {sum(prep[prep.pid.isin(test)].hours):.0f} h. "
             "Alarm events merge fired windows with a 30 s refractory period. 'Fall sensitivity' is the share of "
             "out-of-dataset fall windows (nested models trained on three of the four fall datasets, each "
             "dataset weighted equally, plus the Capture-24 negatives for the c24 models) at the rule's "
             "threshold; the final model itself is trained on all four, so this is an approximation. Fall windows "
             "are simulated.\n")
    tabs, thrs = {}, {}
    for name, root in roots.items():
        thr = pd.read_csv(root / "thresholds.csv")
        thrs[name] = thr
        tabs[name] = F.alarm_table(test, thr, seeds, REFRACTORY_S, root)
    L.append("## 1. Threshold rules: free-living alarms against fall sensitivity\n")
    L.append("| model | rule | alarm events / 24 h (95% CI over people) | median person | p90 person | fall-window sensitivity at this threshold |")
    L.append("|---|---|---|---|---|---|")
    for name, root in roots.items():
        for rule in RULES:
            g = tabs[name][tabs[name].rule == rule]
            pp = g.groupby(["seed", "pid"])[["events", "hours"]].sum()
            pp["per_day"] = pp.events / pp.hours * 24
            rates, cis, sens = [], [], []
            for s, gg in pp.groupby("seed"):
                rates.append(gg.events.sum() / gg.hours.sum() * 24)
                cis.append(per_day_ci(gg.events.to_numpy(), gg.hours.to_numpy(), rng))
                t = thrs[name][(thrs[name].seed == s) & (thrs[name].rule == rule)].threshold.iloc[0]
                fs = np.load(root / "fall_scores" / f"seed{s}.npz")
                sens.append(np.mean([(fs[k] >= t).mean() for k in fs.files]))
            lo, hi = np.mean(cis, axis=0)
            L.append(f"| {name} | {rule} | {np.mean(rates):.1f} ({lo:.1f}-{hi:.1f}) | {pp.groupby('seed').per_day.median().mean():.1f} | "
                     f"{pp.groupby('seed').per_day.quantile(0.9).mean():.1f} | {np.mean(sens):.3f} ± {np.std(sens):.3f} |")
    L.append(f"\nPer-seed alarms/day: " + "; ".join(
        f"{n[:12]}… {r}: " + ", ".join(f"{v:.0f}" for v in (tabs[n][tabs[n].rule == r].groupby('seed')[['events', 'hours']].sum().eval('events/hours*24')))
        for n in roots for r in RULES) + "\n")
    L.append("## 2. What each threshold costs and buys\n")
    L.append("Same grid of thresholds applied to free-living alarms/day and to out-of-dataset fall windows.\n")
    L.append("| threshold | alarms/day (without) | alarms/day (c24-random) | fall sensitivity (without) | fall sensitivity (c24-random) |")
    L.append("|---|---|---|---|---|")
    tr = {n: (F.tradeoff(test, seeds, REFRACTORY_S, r) if r == OUT else pd.read_csv(out / "tradeoff.csv")) for n, r in roots.items()}
    a, b = [tr[n].groupby("threshold").mean(numeric_only=True) for n in roots]
    for t in a.index:
        L.append(f"| {t:g} | {a.loc[t, 'events_per_24h']:.1f} | {b.loc[t, 'events_per_24h']:.1f} | "
                 f"{a.loc[t, 'fall_window_sensitivity']:.3f} | {b.loc[t, 'fall_window_sensitivity']:.3f} |")
    L.append("\nThe two models' scores are calibrated differently (extra negatives shift probabilities down), "
             "so compare at equal *alarms/day* or equal *sensitivity*, not at equal threshold.\n")
    L.append("## 3. Where the remaining alarms come from (val-slice threshold)\n")
    L.append("| activity | hours | events/h (without) | events/h (c24-random) |")
    L.append("|---|---|---|---|")
    per = {}
    for name in roots:
        g = tabs[name][tabs[name].rule == "val-slice"].groupby(["seed", "label"])[["hours", "events"]].sum().groupby("label").mean()
        per[name] = g
    names = list(roots)
    for lab in per[names[0]].sort_values("events", ascending=False).index:
        h = per[names[0]].loc[lab, "hours"]
        L.append(f"| {lab} | {h:.0f} | {per[names[0]].loc[lab, 'events'] / h:.2f} | {per[names[1]].loc[lab, 'events'] / per[names[1]].loc[lab, 'hours']:.2f} |")
    (out / "comparison.md").write_text("\n".join(L) + "\n")
    print(f"comparison: {out / 'comparison.md'}")


# ------------------------------------------------------------------------------ dose-response
SCALE_KS = (5000, 15000)  # random Capture-24 windows per training person (negatives.run used RANDOM_K)


def scale(seed: int, ks: tuple[int, ...] = SCALE_KS) -> None:
    """More random Capture-24 negatives per training person, same folds/seeds/held-out people as ``run``.
    Every (K, seed, fold) result is stored on its own (condition ``c24-random-<K>``), so an interrupted run resumes."""
    t0 = time.time()
    data, results, aug = FallData.load(), Results(RUNS), Augment()
    people = SOURCE.split(seed)
    test_acc = [SOURCE.signal(p) for p in people.held_out]
    x_c24_val = SOURCE.windows(people.validation, VAL_K, seed, "validation")
    folds = lodo_folds(data, seed)
    for k in ks:
        for fold in folds:
            cond = f"c24-random-{k}"
            if results.has(cond, seed, fold.key):
                continue
            xc = SOURCE.windows(people.train, k, seed, "train", fold.name)
            trained = fold.train_model("acc", augment=aug, extra=(xc, np.zeros(len(xc), np.int64)), extra_validation=x_c24_val)
            del xc
            row, curve = evaluate(trained, fold, test_acc)
            results.save(cond, seed, fold.key, [row], curve)
            del trained
            torch.cuda.empty_cache()
            print(f"[seed {seed}] K={k} {fold.name} done ({time.time() - t0:.0f}s)", flush=True)


def _dose(condition: str) -> int | None:
    """Random Capture-24 windows per training person of a stored condition (None: not a random-negatives dose)."""
    if condition == "real":
        return 0
    if condition == "c24-random":
        return RANDOM_K
    return int(condition.removeprefix("c24-random-")) if condition.startswith("c24-random-") else None


def scale_report() -> None:
    out = REPORTS / "negatives"
    df = _table(Results(RUNS))
    df = df.assign(K=df.condition.map(_dose)).dropna(subset=["K"]).astype({"K": int})
    seeds = sorted(int(s) for s in df[df.K > RANDOM_K].seed.unique())
    df = df[df.seed.isin(seeds)]
    per = df.groupby(["K", "seed"])[COLS].mean()
    mean, std = per.groupby("K").mean(), per.groupby("K").std()
    names = {"auroc": "AUROC", "ap": "AUPRC", "fa_h_at_95sens_oracle": "lab oracle FA/h", "sensitivity": "lab sens @val thr",
             "false_alarms_per_h": "lab FA/h @val thr", "fl_events_per_day_at_val_thr": "free-living alarms/day @val thr",
             "fl_events_per_day_at_95sens": "free-living alarms/day @95% sens", **{f"sens_at_{b}_per_day": f"sens @{b}/day" for b in BUDGETS}}
    split = SOURCE.split(0)
    n_fit = len(split.train), len(split.validation)
    L = ["# More Capture-24 negatives: dose-response\n"]
    L.append(f"Seeds: {len(seeds)} ({seeds}). Leave-one-fall-dataset-out, accel-only augmented CNN; random Capture-24 windows "
             f"per training person (about 47,000 windows exist per person; training people: {n_fit[0]}, validation people: {n_fit[1]}; "
             "held-out people P101-P151, never trained on). K=0 and K=1500 are the control and `c24-random` from "
             "`reports/negatives/report.md` (same seeds and splits). "
             "Cells: macro over the four held-out fall datasets, mean ± std over seeds. Free-living columns use the "
             "held-out people. Sensitivity at a budget is threshold-free (see `negatives.py`). Fall windows are simulated.\n")
    L.append("## Macro over held-out fall datasets\n")
    L.append("| windows per person | total negatives added | " + " | ".join(names[c] for c in COLS) + " |")
    L.append("|---|---|" + "---|" * len(COLS))
    for k in mean.index:
        cells = [f"{mean.loc[k, c]:.3f} ± {std.loc[k, c]:.3f}" if c in ("auroc", "ap", "sensitivity") or c.startswith("sens_at")
                 else f"{mean.loc[k, c]:.1f} ± {std.loc[k, c]:.1f}" for c in COLS]
        L.append(f"| {k} | {k * n_fit[0]:,} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Paired difference to K=1500 (the previous setting)\n")
    L.append("| windows per person | metric | Δ | wins |")
    L.append("|---|---|---|---|")
    for k in [k for k in mean.index if k > RANDOM_K]:
        for c in COLS:
            d = per.loc[k][c] - per.loc[RANDOM_K][c]
            better = (d < 0) if c in LOWER else (d > 0)
            L.append(f"| {k} | {names[c]} | {d.mean():+.3f} ± {d.std() / np.sqrt(len(d)):.3f} | {int(better.sum())}/{len(d)} |")
    L.append("")
    for c, title in (("fl_events_per_day_at_95sens", "Free-living alarms/day at 95% sensitivity"), ("sens_at_25_per_day", "Sensitivity at 25 alarms/day")):
        L.append(f"## Per held-out fall dataset: {title}\n")
        pv = df.groupby(["K", "fold"])[c].mean().unstack()
        L.append("| windows per person | " + " | ".join(pv.columns) + " |")
        L.append("|---|" + "---|" * len(pv.columns))
        for k in pv.index:
            L.append(f"| {k} | " + " | ".join(f"{v:.3f}" if c.startswith("sens") else f"{v:.1f}" for v in pv.loc[k]) + " |")
        L.append("")
    (out / "scale_report.md").write_text("\n".join(L))
    print(f"report: {out / 'scale_report.md'}")
