"""Free-living false-alarm rate on Capture-24 (wrist accelerometer worn ~24 h per person, labelled).

Replaces the assumed time budgets of ``budget_estimate.py`` with measured exposure: every 4 s window (2 s
stride) of every participant is scored by the final models and alarm events are counted with the same
refractory merge, per participant and per labelled activity (sleep included).

Steps (each cached): ``prepare`` Capture-24 CSVs -> 25 Hz canonical-frame signal + activity labels;
``fit`` final augmented accel-only CNNs on all four fall datasets with the two threshold rules;
``score`` every window; ``report``.

Mounting: each participant is aligned from walking and mirrored to the left-wrist convention by the
Capture-24 source (``capture24.py``), which also owns the person split and activity labels.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .model import FallModel
from .alarms import REFRACTORY_S, FalseAlarms, per_day_ci
from .augment import Augment
from .capture24 import Capture24
from .paths import PROCESSED, RAW, REPORTS
from .rules import lodo_pooled
from .runner import FallData, final_fold

OUT = PROCESSED / "capture24"
SOURCE = Capture24(OUT)
RULES = ("val-slice", "lodo-pooled")


# -------------------------------------------------------------------------------------- fit
def fit_models(seeds: int, root=OUT, extra=None) -> pd.DataFrame:
    """Final models (all four fall datasets) and thresholds per seed; weights saved to ``root``/models.

    ``extra(seed)`` optionally returns (training, validation) canonical accelerometer windows of ADL that
    are added to every fit, final and nested (see ``negatives.c24_negatives``)."""
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "fall_scores").mkdir(exist_ok=True)
    tp = root / "thresholds.csv"
    if tp.exists() and pd.read_csv(tp).seed.nunique() >= seeds:
        return pd.read_csv(tp)
    data, aug = FallData.load(), Augment()
    rows, t0 = [], time.time()
    for seed in range(seeds):
        fold = final_fold(data, seed)
        kw = {"augment": aug}
        if extra:
            xn, xnv = extra(seed)
            kw |= {"extra": (xn, np.zeros(len(xn), np.int64)), "extra_validation": xnv}
        trained = fold.train_model("acc", **kw)
        thr = {"val-slice": trained.threshold}
        trained.model.save(root / "models" / f"seed{seed}.pt")
        del trained
        fall_scores = fold.out_of_dataset_fall_scores("acc", **kw)  # nested leave-one-dataset-out
        thr["lodo-pooled"] = lodo_pooled(fall_scores)
        np.savez(root / "fall_scores" / f"seed{seed}.npz", **fall_scores)
        rows += [{"seed": seed, "rule": r, "threshold": t} for r, t in thr.items()]
        print(f"[fit] seed {seed} done ({time.time() - t0:.0f}s) {thr}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(tp, index=False)
    return df


# ------------------------------------------------------------------------------------ score
def score_all(seeds: int, pids: list[str], root=OUT) -> None:
    sp = root / "scores"
    sp.mkdir(exist_ok=True)
    nets = [FallModel.load(root / "models" / f"seed{s}.pt", "acc") for s in range(seeds)]
    t0 = time.time()
    for i, pid in enumerate(pids):
        f = sp / f"{pid}.npy"
        if f.exists() and np.load(f, mmap_mode="r").shape[0] >= seeds:
            continue
        acc = SOURCE.signal(pid)
        np.save(f, np.stack([n.score_signal(acc) for n in nets]).astype(np.float32))
        if i % 20 == 0:
            print(f"[score] {i + 1}/{len(pids)} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------- report
def _roll(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


MOUNTINGS = {"as assumed": (0, False), "roll +30°": (30, False), "roll -30°": (-30, False),
             "roll +60°": (60, False), "roll -60°": (-60, False), "roll 90°": (90, False),
             "roll 180°": (180, False), "left wrist (no mirror)": (0, True)}


def mounting_sensitivity(pids: list[str], thr: pd.DataFrame, seeds: int, refractory: float, root=OUT) -> pd.DataFrame:
    """Events/24 h when the assumed mounting is wrong: forearm-roll offsets and the wrong wrist side."""
    nets = [FallModel.load(root / "models" / f"seed{s}.pt", "acc") for s in range(seeds)]
    sub = pids[:: max(1, len(pids) // 40)]  # ~40 participants keep this affordable
    rows = []
    for pid in sub:
        acc0 = SOURCE.signal(pid)
        for name, (deg, unmirror) in MOUNTINGS.items():
            a = acc0 * np.array([1.0, -1.0, 1.0], np.float32) if unmirror else acc0
            a = np.ascontiguousarray(a @ _roll(deg).T)
            for seed in range(seeds):
                fa = FalseAlarms.from_window_scores([nets[seed].score_signal(a)], refractory=refractory)
                for rule in RULES:
                    t = thr[(thr.seed == seed) & (thr.rule == rule)].threshold.iloc[0]
                    rows.append({"mounting": name, "rule": rule, "seed": seed, "pid": pid,
                                 "hours": fa.hours[0], "events": int(fa.events(t)[0])})
    return pd.DataFrame(rows)


THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def tradeoff(pids: list[str], seeds: int, refractory: float, root=OUT) -> pd.DataFrame:
    """Alarm events/24 h on Capture-24 (final model) against sensitivity on out-of-dataset fall windows
    (nested models trained on three of the four datasets, equal weight per dataset) at the same threshold.
    The two models differ (3 vs 4 training datasets), so this is an approximation."""
    rows = []
    scores = {pid: np.load(root / "scores" / f"{pid}.npy") for pid in pids}
    for seed in range(seeds):
        fs = np.load(root / "fall_scores" / f"seed{seed}.npz")
        fa = FalseAlarms.from_window_scores([scores[p][seed] for p in pids], refractory=refractory)
        _, per_day = fa.curve(THRESHOLDS)
        for t, v in zip(THRESHOLDS, per_day):
            sens = np.mean([(fs[k] >= t).mean() for k in fs.files])
            rows.append({"seed": seed, "threshold": t, "events_per_24h": v, "fall_window_sensitivity": sens})
    return pd.DataFrame(rows)


def alarm_table(pids: list[str], thr: pd.DataFrame, seeds: int, refractory: float, root=OUT) -> pd.DataFrame:
    """One row per (seed, rule, participant, label): hours of exposure, alarm events (labelled by the
    window that opened the event) and fired windows."""
    sc = {p: np.load(root / "scores" / f"{p}.npy") for p in pids}
    labels = [SOURCE.window_labels(p)[: sc[p].shape[1]] for p in pids]
    rows = []
    for seed in range(seeds):
        fa = FalseAlarms.from_window_scores([sc[p][seed] for p in pids], labels, refractory)
        for rule in RULES:
            t = thr[(thr.seed == seed) & (thr.rule == rule)].threshold.iloc[0]
            d = fa.by_label(t)
            rows.append(d.assign(seed=seed, rule=rule, pid=[pids[k] for k in d.wearer]).rename(columns={"detections": "windows"}))
    return pd.concat(rows, ignore_index=True)[["seed", "rule", "pid", "label", "hours", "events", "windows"]]


def run(seeds: int = 5, negatives: bool = False) -> None:
    """Free-living report. ``negatives``: final models also train on Capture-24 negatives (people
    P001-P100), so only the held-out people P101-P151 are scored."""
    t0 = time.time()
    prep = SOURCE.prepare(RAW)
    pids, root, extra, name = SOURCE.people(), OUT, None, "freeliving"
    if negatives:
        from .negatives import c24_negatives

        pids, root, extra, name = SOURCE.split(0).held_out, OUT / "c24random", c24_negatives, "freeliving_c24random"
        prep = prep[prep.pid.map(lambda p: int(p[1:])) > 100]
    thr = fit_models(seeds, root, extra)
    score_all(seeds, pids, root)
    tab = alarm_table(pids, thr, seeds, REFRACTORY_S, root)
    out = REPORTS / name
    out.mkdir(parents=True, exist_ok=True)
    ms_path, tr_path = out / "mounting_sensitivity.csv", out / "tradeoff.csv"
    ms = pd.read_csv(ms_path) if ms_path.exists() else mounting_sensitivity(pids, thr, seeds, REFRACTORY_S, root)
    ms.to_csv(ms_path, index=False)
    tr = pd.read_csv(tr_path) if tr_path.exists() else tradeoff(pids, seeds, REFRACTORY_S, root)
    tr.to_csv(tr_path, index=False)
    tab.to_csv(out / "alarms_by_label.csv.gz", index=False)
    write_report(prep, tab, thr, out, seeds, time.time() - t0, ms, tr, negatives, root)
    print(f"report: {out / 'report.md'}")


def write_report(prep, tab, thr, out, seeds, seconds, ms, tr, negatives=False, root=OUT) -> None:
    rng = np.random.default_rng(0)
    ex = prep[prep.excluded.notna()] if "excluded" in prep else prep.iloc[0:0]
    ok = prep[prep.excluded.isna()] if "excluded" in prep else prep
    L = ["# Free-living false alarms (Capture-24)" + (" with Capture-24 negatives in training" if negatives else "") + "\n"]
    L.append(f"Seeds: {seeds}. {len(ok)} of {len(prep)} participants used ({ok.hours.sum():.0f} h of wrist "
             "accelerometer, about a day each, including sleep); each 4 s window (2 s stride) is scored by the "
             "final augmented accel-only CNN trained on all four fall datasets"
             + (" plus random Capture-24 windows of people P001-P100 as negatives (only the held-out people "
                "P101-P151 are scored here, so no one in this report was trained on)" if negatives else "")
             + f". Alarm events merge fired windows with a "
             f"{REFRACTORY_S:.0f} s refractory period. **Rates are alarm events, not windows. Nothing here "
             "is assumed: the exposure is measured.** Rule `val-slice` = 95% sensitivity on a validation slice of "
             "the training data; `lodo-pooled` = nested leave-one-dataset-out (see `threshold.py`). The threshold "
             "belongs to one operating point each; sensitivity on real falls is unknown (the fall datasets are "
             "simulated; see section 5 and `reports/threshold_augmented/report.md` for fall sensitivity of each rule).\n")
    L.append("**Limits.** (1) The people are 151 adults in Oxford wearing an Axivity AX3 on the *dominant* "
             "wrist, not a smartwatch; the side is not recorded, so I assumed right and mirrored it, and the "
             "mounting is aligned per person from walking (direction only, norm ~0.7 g because of arm swing). "
             "(2) Only about 66% of the hours are labelled; the rest is shown as `unlabelled`. (3) A fall or "
             "near-fall that really happened would count as a false alarm here, since nothing marks it. (4) The "
             "AX3 samples at 100 Hz and is resampled to 25 Hz by the same code path as the real datasets (`canonical.to_grid`). "
             "(5) 151 people × one day: day-to-day variation within a person is not measured.\n")
    if len(ex):
        L.append(f"Excluded participants: {len(ex)} ({'; '.join(f'{r.pid}: {r.excluded}' for r in ex.itertuples())}).\n")

    # per rule: overall
    L.append("## 1. Alarm events per day of wear\n")
    L.append("Events per 24 h of wear, pooled over all participants and hours (labelled and unlabelled). "
             "Interval: 95% bootstrap over participants (seed-averaged). `per-participant` is the distribution "
             "over participants of their own events/24 h.\n")
    L.append("| rule | events / 24 h (95% CI) | median participant | p90 participant | participants with ≥1 alarm per day | participants with 0 alarms |")
    L.append("|---|---|---|---|---|---|")
    per_part = {}
    for rule in RULES:
        g = tab[tab.rule == rule]
        pp = g.groupby(["seed", "pid"])[["events", "hours"]].sum()
        pp["per_day"] = pp.events / pp.hours * 24
        per_part[rule] = pp
        rates, lo_hi = [], []
        for s, gg in pp.groupby("seed"):
            rates.append(gg.events.sum() / gg.hours.sum() * 24)
            lo_hi.append(per_day_ci(gg.events.to_numpy(), gg.hours.to_numpy(), rng))
        lo, hi = np.mean(lo_hi, axis=0)
        med = pp.groupby("seed").per_day.median().mean()
        p90 = pp.groupby("seed").per_day.quantile(0.9).mean()
        ge1 = pp.groupby("seed").per_day.apply(lambda s: (s >= 1).mean()).mean()
        z = pp.groupby("seed").events.apply(lambda s: (s == 0).mean()).mean()
        L.append(f"| {rule} | {np.mean(rates):.1f} ({lo:.1f}–{hi:.1f}) | {med:.1f} | {p90:.1f} | {ge1:.0%} | {z:.0%} |")
    L.append("")

    # per label
    L.append("## 2. Where the alarms come from\n")
    L.append("Rate per hour of each labelled activity (pooled over participants, seed-averaged) and its share "
             "of all alarm events.\n")
    for rule in RULES:
        g = tab[tab.rule == rule].groupby(["seed", "label"])[["hours", "events", "windows"]].sum()
        tot = g.groupby("seed").events.sum()
        L.append(f"### Rule: {rule}\n")
        L.append("| activity | hours | share of wear time | events/h | share of alarm events | windows/h (before merging) |")
        L.append("|---|---|---|---|---|---|")
        m = g.groupby("label").mean()
        th = g.groupby("seed").hours.sum().mean()
        for lab_name in m.sort_values("events", ascending=False).index:
            r = m.loc[lab_name]
            L.append(f"| {lab_name} | {r.hours:.0f} | {r.hours / th:.0%} | {r.events / r.hours:.2f} | "
                     f"{r.events / tot.mean():.0%} | {r.windows / r.hours:.2f} |")
        L.append("")

    # sleep vs day, day-hours
    L.append("## 3. Day versus night\n")
    L.append("| rule | awake labelled hours: events/h | sleep: events/h | unlabelled: events/h |")
    L.append("|---|---|---|---|")
    for rule in RULES:
        g = tab[tab.rule == rule].groupby(["seed", "label"])[["hours", "events"]].sum()
        m = g.groupby("label").mean()
        awake = m.drop(index=[i for i in ("sleep", "unlabelled") if i in m.index])
        L.append(f"| {rule} | {awake.events.sum() / awake.hours.sum():.2f} | "
                 f"{m.loc['sleep'].events / m.loc['sleep'].hours:.3f} | "
                 f"{(m.loc['unlabelled'].events / m.loc['unlabelled'].hours) if 'unlabelled' in m.index else float('nan'):.2f} |")
    L.append("")

    # mounting robustness, threshold trade-off, comparison with the assumed budgets
    L.append("## 4. How sensitive is this to my mounting assumptions?\n")
    L.append("The same participants (about 40 of them) scored again after rotating the aligned signal about the "
             "forearm axis, or treating the wrist as the left one. If the false-alarm rate barely moves, "
             "the numbers do not hinge on the alignment guess; roll offsets the model never saw during "
             "training (augmentation covers ±30°) would raise it.\n")
    L.append("| mounting | " + " | ".join(f"{r}: events/24 h" for r in RULES) + " |")
    L.append("|---|" + "---|" * len(RULES))
    for name in MOUNTINGS:
        cells = []
        for rule in RULES:
            g = ms[(ms.mounting == name) & (ms.rule == rule)].groupby("seed")[["events", "hours"]].sum()
            cells.append(f"{(g.events / g.hours * 24).mean():.1f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## 5. What each threshold costs and buys\n")
    L.append("Same threshold applied to two things: alarm events per 24 h of Capture-24 wear (final model, "
             "the scored participants) and the share of **out-of-dataset fall windows** it detects (nested models trained "
             "on three of the four fall datasets, each dataset weighted equally). The two models differ, so read "
             "this as an approximate trade-off curve, not an exact operating point. Fall windows are simulated "
             "falls, so real sensitivity is lower. Per-window sensitivity is not per-fall: several windows can "
             "contain one fall, so per-fall detection is higher.\n")
    L.append("| threshold | alarm events / 24 h | fall-window sensitivity |")
    L.append("|---|---|---|")
    for t, g in tr.groupby("threshold"):
        L.append(f"| {t:g} | {g.events_per_24h.mean():.1f} ± {g.events_per_24h.std(ddof=0):.1f} | "
                 f"{g.fall_window_sensitivity.mean():.3f} ± {g.fall_window_sensitivity.std(ddof=0):.3f} |")
    L.append("")
    L.append("## 6. Comparison with the assumed-budget estimate\n")
    L.append("`reports/alarms/report.md` estimated alarms per day from lab clips and assumed activity time budgets "
             "(`budget_estimate.py`). The measured rates above replace that assumption.\n")
    L.append(f"Threshold levels per seed are in `{root.relative_to(PROCESSED.parent.parent)}/thresholds.csv`. "
             f"Runtime of the last report step: {seconds / 60:.1f} min (fits and scoring are cached).\n")
    (out / "report.md").write_text("\n".join(L))
