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

from . import model as M
from .alarms import REFRACTORY_S, FalseAlarms, per_day_ci, sensitivity_at_budget
from .augment import Augment
from .baseline import choose_threshold, metrics, split_validation
from .canonical import FS
from .cli import PROCESSED, REPORTS
from .freeliving import OUT, score_signal
from .windows import STRIDE_S, WIN_S

WIN, STRIDE = round(WIN_S * FS), round(STRIDE_S * FS)
RANDOM_K = 1500  # Capture-24 windows added per training participant
MINED_FRACTION = 2 / 3
VAL_K = 800  # random windows per Capture-24 validation participant
BUDGETS = (10, 25, 50, 100)  # alarm events per day
CONDITIONS = ("real", "c24-random", "c24-mined")


# --------------------------------------------------------------------------------- windows
def to_inputs(w: np.ndarray) -> np.ndarray:
    """(N, WIN, 3) accel windows in g -> model inputs (N, 4, WIN), as ``model.make_inputs(.., 'acc')``."""
    mag = np.linalg.norm(w, axis=2, keepdims=True)
    return (np.concatenate([w, mag], axis=2) / M.ACC_SCALE).transpose(0, 2, 1).astype(np.float32)


def gather(acc: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return acc[starts[:, None] + np.arange(WIN)]


def n_windows(acc: np.ndarray) -> int:
    return (len(acc) - WIN) // STRIDE + 1


def load_acc(pid: str) -> np.ndarray:
    return np.load(OUT / f"{pid}.npz")["acc"]


def random_windows(pids: list[str], k: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    for pid in pids:
        acc = load_acc(pid)
        s = rng.choice(n_windows(acc), size=min(k, n_windows(acc)), replace=False) * STRIDE
        out.append(to_inputs(gather(acc, s)))
    return np.concatenate(out)


def mined_windows(net: M.FallCNN, pids: list[str], k: int, sep: int = 3) -> np.ndarray:
    """Top-scoring windows per participant under ``net`` (greedy, ``sep`` windows apart)."""
    out = []
    for pid in pids:
        acc = load_acc(pid)
        sc = score_signal(net, acc)
        taken = np.zeros(len(sc), bool)
        pick = []
        for i in np.argsort(-sc)[: k * 10]:
            if not taken[max(0, i - sep) : i + sep + 1].any():
                taken[i] = True
                pick.append(i)
                if len(pick) == k:
                    break
        out.append(to_inputs(gather(acc, np.array(pick) * STRIDE)))
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
def evaluate(net, cond, x, y, test_idx, va, val_thr, test_acc):
    p = M.predict_proba(net, x[test_idx])
    lab = metrics(y[test_idx], (p >= val_thr).astype(int), p)
    scores = [score_signal(net, a) for a in test_acc]
    fl, curve = free_living(scores, p[y[test_idx] == 1], val_thr)
    row = {"cond": cond, "threshold": val_thr, **{k: lab[k] for k in
           ("sensitivity", "false_alarms_per_h", "auroc", "ap", "fa_h_at_95sens_oracle")}, **fl}
    return row, curve


def run(seed: int) -> None:
    t0 = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    x = M.make_inputs(X, "acc")
    aug = Augment()
    prep = pd.read_csv(OUT / "prepare_log.csv")
    pids = sorted(prep[prep.excluded.isna()].pid)
    train_pool = [p for p in pids if int(p[1:]) <= 100]
    test_pool = [p for p in pids if int(p[1:]) > 100]
    rng = np.random.default_rng(seed + 321)
    val_p = sorted(rng.choice(train_pool, size=max(1, round(0.2 * len(train_pool))), replace=False))
    fit_p = [p for p in train_pool if p not in val_p]
    test_acc = [load_acc(p) for p in test_pool]
    print(f"seed {seed}: C24 train {len(fit_p)}, val {len(val_p)}, test {len(test_pool)} participants "
          f"({sum(len(a) for a in test_acc) / FS / 3600:.0f} h)", flush=True)
    x_c24_val = random_windows(val_p, VAL_K, rng)
    ones = lambda n: np.ones(n, np.float32)  # noqa: E731
    rows, curves = [], {}
    out = REPORTS / "negatives"
    out.mkdir(parents=True, exist_ok=True)
    for ds in sorted(meta.dataset.unique()):
        test = np.where(meta.dataset == ds)[0]
        tr, va = split_validation(meta, np.where(meta.dataset != ds)[0], seed)
        xv, yv = x[va], y[va]
        net = M.fit(x[tr], y[tr], ones(len(tr)), xv, yv, seed, augment=aug)
        thr = choose_threshold(M.predict_proba(net, xv), yv)
        row, cv = evaluate(net, "real", x, y, test, va, thr, test_acc)
        rows.append({"seed": seed, "fold": ds, **row})
        curves.update({f"{ds}|real|{k}": v for k, v in cv.items()})
        xc_rand = random_windows(fit_p, RANDOM_K, rng)
        n_mined = round(RANDOM_K * MINED_FRACTION)
        xc_mix = np.concatenate([random_windows(fit_p, RANDOM_K - n_mined, rng), mined_windows(net, fit_p, n_mined)])
        del net
        xvc = np.concatenate([xv, x_c24_val])
        yvc = np.concatenate([yv, np.zeros(len(x_c24_val), np.int64)])
        for cond, xc in (("c24-random", xc_rand), ("c24-mined", xc_mix)):
            xt = np.concatenate([x[tr], xc])
            yt = np.concatenate([y[tr], np.zeros(len(xc), np.int64)])
            net = M.fit(xt, yt, ones(len(xt)), xvc, yvc, seed, augment=aug)
            thr = choose_threshold(M.predict_proba(net, xv), yv)  # falls only matter for the threshold
            row, cv = evaluate(net, cond, x, y, test, va, thr, test_acc)
            rows.append({"seed": seed, "fold": ds, **row})
            curves.update({f"{ds}|{cond}|{k}": v for k, v in cv.items()})
            del net
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out / f"metrics_seed{seed}.csv", index=False)
        np.savez(out / f"curves_seed{seed}.npz", **curves)
        print(f"[seed {seed}] {ds} done ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------- report
COLS = ["auroc", "ap", "fa_h_at_95sens_oracle", "sensitivity", "false_alarms_per_h",
        "fl_events_per_day_at_val_thr", *[f"sens_at_{b}_per_day" for b in BUDGETS], "fl_events_per_day_at_95sens"]
LOWER = {"fa_h_at_95sens_oracle", "false_alarms_per_h", "fl_events_per_day_at_val_thr", "fl_events_per_day_at_95sens"}


def report() -> None:
    out = REPORTS / "negatives"
    df = pd.concat([pd.read_csv(f) for f in sorted(out.glob("metrics_seed*.csv"))])
    sens = {f"sens_at_{b}_per_day": [] for b in BUDGETS}
    for r in df.itertuples():
        cv = np.load(out / f"curves_seed{r.seed}.npz")
        k = f"{r.fold}|{r.cond}|"
        for b in BUDGETS:
            sens[f"sens_at_{b}_per_day"].append(sensitivity_at_budget(cv[k + "grid"], cv[k + "ev"], cv[k + "falls"], b))
    df = df.assign(**sens)
    seeds = df.seed.nunique()
    per = df.groupby(["cond", "seed"])[COLS].mean()  # macro over held-out fall datasets
    mean, std = per.groupby("cond").mean(), per.groupby("cond").std()
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
        pv = df.groupby(["cond", "fold"])[k].mean().unstack()
        L.append("| condition | " + " | ".join(pv.columns) + " |")
        L.append("|---|" + "---|" * len(pv.columns))
        for c in CONDITIONS:
            L.append(f"| {c} | " + " | ".join(f"{v:.3f}" if k.startswith("sens") else f"{v:.1f}" for v in pv.loc[c]) + " |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
    print(f"report: {out / 'report.md'}")


# ------------------------------------------------------------------------ final-model analysis
def test_pids(prep: pd.DataFrame) -> list[str]:
    """Held-out Capture-24 people (never trained on), usable ones only."""
    return sorted(p for p in prep[prep.excluded.isna()].pid if int(p[1:]) > 100)


def c24_negatives(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(training, validation) random Capture-24 windows for the final models (people P001-P100)."""
    prep = pd.read_csv(OUT / "prepare_log.csv")
    train_pool = sorted(p for p in prep[prep.excluded.isna()].pid if int(p[1:]) <= 100)
    rng = np.random.default_rng(seed + 321)
    val_p = sorted(rng.choice(train_pool, size=max(1, round(0.2 * len(train_pool))), replace=False))
    fit_p = [p for p in train_pool if p not in val_p]
    return random_windows(fit_p, RANDOM_K, rng), random_windows(val_p, VAL_K, rng)


def final(seeds: int = 3) -> None:
    """Final models with Capture-24 negatives: free-living report plus a comparison with the models
    trained without them, on the same held-out people."""
    from . import freeliving as F
    from .freeliving import RULES

    F.run(seeds, negatives=True)
    prep = pd.read_csv(OUT / "prepare_log.csv")
    test = test_pids(prep)
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
    Every (seed, K, fold) result is saved on its own, so an interrupted run resumes."""
    t0 = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    x = M.make_inputs(X, "acc")
    aug = Augment()
    prep = pd.read_csv(OUT / "prepare_log.csv")
    pids = sorted(prep[prep.excluded.isna()].pid)
    train_pool = [p for p in pids if int(p[1:]) <= 100]
    test_acc = [load_acc(p) for p in test_pids(prep)]
    rng = np.random.default_rng(seed + 321)  # same draw order as run(): identical validation people
    val_p = sorted(rng.choice(train_pool, size=max(1, round(0.2 * len(train_pool))), replace=False))
    fit_p = [p for p in train_pool if p not in val_p]
    x_c24_val = random_windows(val_p, VAL_K, rng)
    out = REPORTS / "negatives" / "scale"
    out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        for fi, ds in enumerate(sorted(meta.dataset.unique())):
            f = out / f"seed{seed}_K{k}_{ds}.csv"
            if f.exists():
                continue
            test = np.where(meta.dataset == ds)[0]
            tr, va = split_validation(meta, np.where(meta.dataset != ds)[0], seed)
            xv, yv = x[va], y[va]
            xc = random_windows(fit_p, k, np.random.default_rng(seed * 1000 + k + fi))
            xt = np.concatenate([x[tr], xc])
            yt = np.concatenate([y[tr], np.zeros(len(xc), np.int64)])
            del xc
            xvc = np.concatenate([xv, x_c24_val])
            yvc = np.concatenate([yv, np.zeros(len(x_c24_val), np.int64)])
            net = M.fit(xt, yt, np.ones(len(xt), np.float32), xvc, yvc, seed, augment=aug)
            del xt
            thr = choose_threshold(M.predict_proba(net, xv), yv)
            row, cv = evaluate(net, f"c24-random-{k}", x, y, test, va, thr, test_acc)
            np.savez(out / f"seed{seed}_K{k}_{ds}.npz", **cv)
            pd.DataFrame([{"seed": seed, "fold": ds, "K": k, **row}]).to_csv(f, index=False)
            del net
            torch.cuda.empty_cache()
            print(f"[seed {seed}] K={k} {ds} done ({time.time() - t0:.0f}s)", flush=True)


def scale_report() -> None:
    out = REPORTS / "negatives"
    rows = []
    for f in sorted((out / "scale").glob("seed*_K*_*.csv")):
        r = pd.read_csv(f).iloc[0]
        cv = np.load(f.with_suffix(".npz"))
        d = r.to_dict()
        for b in BUDGETS:
            d[f"sens_at_{b}_per_day"] = sensitivity_at_budget(cv["grid"], cv["ev"], cv["falls"], b)
        rows.append(d)
    new = pd.DataFrame(rows)
    old = pd.concat([pd.read_csv(f) for f in sorted(out.glob("metrics_seed*.csv"))])
    sens = {f"sens_at_{b}_per_day": [] for b in BUDGETS}
    for r in old.itertuples():
        cv = np.load(out / f"curves_seed{r.seed}.npz")
        k = f"{r.fold}|{r.cond}|"
        for b in BUDGETS:
            sens[f"sens_at_{b}_per_day"].append(sensitivity_at_budget(cv[k + "grid"], cv[k + "ev"], cv[k + "falls"], b))
    old = old.assign(**sens)
    old = old[old.cond.isin(["real", "c24-random"])].assign(K=lambda d: np.where(d.cond == "real", 0, RANDOM_K))
    df = pd.concat([old, new.assign(K=new.K.astype(int))])
    seeds = sorted(set(new.seed))
    df = df[df.seed.isin(seeds)]
    per = df.groupby(["K", "seed"])[COLS].mean()
    mean, std = per.groupby("K").mean(), per.groupby("K").std()
    names = {"auroc": "AUROC", "ap": "AUPRC", "fa_h_at_95sens_oracle": "lab oracle FA/h", "sensitivity": "lab sens @val thr",
             "false_alarms_per_h": "lab FA/h @val thr", "fl_events_per_day_at_val_thr": "free-living alarms/day @val thr",
             "fl_events_per_day_at_95sens": "free-living alarms/day @95% sens", **{f"sens_at_{b}_per_day": f"sens @{b}/day" for b in BUDGETS}}
    n_fit = train_pids_count()
    L = ["# More Capture-24 negatives: dose-response\n"]
    L.append(f"Seeds: {len(seeds)} ({seeds}). Leave-one-fall-dataset-out, accel-only augmented CNN; random Capture-24 windows "
             f"per training person (about 47,000 windows exist per person; training people: {n_fit[0]}, validation people: {n_fit[1]}; "
             "held-out people P101-P151, never trained on). K=0 and K=1500 are the control and `c24-random` from "
             "`reports/negatives/report.md` (same seeds and splits; windows drawn with a different random stream). "
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


def train_pids_count() -> tuple[int, int]:
    prep = pd.read_csv(OUT / "prepare_log.csv")
    n = sum(int(p[1:]) <= 100 for p in prep[prep.excluded.isna()].pid)
    v = max(1, round(0.2 * n))
    return n - v, v
