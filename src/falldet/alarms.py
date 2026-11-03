"""Turn "false alarms/h on replayed clips" into a false-alarms-per-day estimate.

The replay metric in the other reports counts every fired 4 s window of scripted ADL clips, so (a)
overlapping windows of one event count several times and (b) hard activities (jogging, hopping,
clapping) are over-represented compared with a day of wear. This module fixes both:

1. **Scores**: leave-one-dataset-out, accel-only, augmented CNN; the held-out dataset's windows are
   scored (all of them, with their order) and two threshold rules are applied: ``val-slice`` (95%
   sensitivity on a validation slice of the training datasets) and ``lodo-pooled`` (nested
   leave-one-dataset-out, see ``threshold.py``).
2. **Alarm events**: fired windows of one recording are merged by a refractory period (a real device
   shows one prompt, then ignores the user's arm for a while), giving alarm events per clip.
3. **Per-activity rates**: ADL codes of all datasets map to common categories (``CATEGORY``); each gets
   events per clip and per hour of that activity, with an exact Poisson 95% upper bound (counts are small, and
   clustered by subject, so even this bound is optimistic).
4. **Daily estimate**: rates are weighted by an assumed activity time budget (``BUDGETS``). The budgets
   are assumptions, not measurements; the report shows each category's contribution so they can be replaced.
"""

from __future__ import annotations

import time

from scipy.stats import gamma

import numpy as np
import pandas as pd
import torch

from . import model as M
from .augment import Augment
from .baseline import choose_threshold, split_validation
from .cli import PROCESSED, REPORTS
from .threshold import balanced_threshold
from .windows import STRIDE_S, WIN_S

REFRACTORY_S = 30.0
REFRACTORY_SWEEP = (2.0, 10.0, 30.0, 60.0)
RULES = ("val-slice", "lodo-pooled")

# (dataset, activity code) -> category. UP-Fall codes follow the dataset's published list
# (6 walking, 7 standing, 8 sitting, 9 picking up an object, 10 jumping, 11 laying) [not in the files].
CATEGORY = {
    **{("weda", c): k for c, k in {
        "D01": "walking", "D02": "jogging", "D03": "stairs", "D04": "sit-down/stand-up",
        "D05": "sit-down/stand-up", "D06": "bending/crouching", "D07": "stumble",
        "D08": "jumping/hopping", "D09": "hand impact", "D10": "hand impact", "D11": "arm gesture"}.items()},
    **{("erciyes", str(c)): k for c, k in {
        801: "walking", 802: "walking", 803: "jogging", 804: "bending/crouching", 805: "bending/crouching",
        806: "bending/crouching", 807: "walking", 808: "stumble", 809: "stumble", 810: "arm gesture",
        811: "sit-down/stand-up", 812: "sit-down/stand-up", 813: "sit-down/stand-up", 814: "sit-down/stand-up",
        815: "lying down/rising", 816: "lying down/rising"}.items()},
    **{("upfall", c): k for c, k in {
        "A6": "walking", "A7": "stationary", "A8": "stationary", "A9": "bending/crouching",
        "A10": "jumping/hopping", "A11": "stationary"}.items()},
    **{("umafall", c): k for c, k in {
        "Walking": "walking", "Jogging": "jogging", "GoUpstairs": "stairs", "GoDownstairs": "stairs",
        "Sitting_GettingUpOnAChair": "sit-down/stand-up", "LyingDown_OnABed": "lying down/rising",
        "Bending": "bending/crouching", "Hopping": "jumping/hopping", "Aplausing": "hand impact",
        "HandsUp": "arm gesture", "MakingACall": "arm gesture", "OpeningDoor": "arm gesture"}.items()},
}
# how each category enters a day: continuous activities by hours, discrete ones by occurrences
KIND = {"stationary": "hours", "walking": "hours", "stairs": "hours", "jogging": "hours",
        "sit-down/stand-up": "count", "lying down/rising": "count", "bending/crouching": "count",
        "arm gesture": "count", "hand impact": "count", "stumble": "count", "jumping/hopping": "hours"}
# ASSUMED time budgets for one worn day (hours for continuous, occurrences for discrete). Not measured:
# replace with your own. Activities the datasets do not cover (driving, typing, cooking, sport, sleep
# movement) are counted as "stationary" hours, which is optimistic for the vigorous ones.
BUDGETS = {
    "typical day (24 h worn incl. sleep)": {
        "stationary": 21.0, "walking": 2.0, "stairs": 0.2, "jogging": 0.0, "jumping/hopping": 0.0,
        "sit-down/stand-up": 40, "lying down/rising": 4, "bending/crouching": 30, "arm gesture": 60,
        "hand impact": 5, "stumble": 0.5},
    "active day (+1 h jogging, 10 min jumping, more impacts)": {
        "stationary": 18.5, "walking": 3.0, "stairs": 0.4, "jogging": 1.0, "jumping/hopping": 0.17,
        "sit-down/stand-up": 40, "lying down/rising": 4, "bending/crouching": 50, "arm gesture": 80,
        "hand impact": 20, "stumble": 1},
}


# ---------------------------------------------------------------------------------- scoring
def score(seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores of every window of each held-out dataset and the two thresholds, per seed."""
    t0 = time.time()
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    x = M.make_inputs(X, "acc")
    aug = Augment()
    datasets = sorted(meta.dataset.unique())
    ones = lambda n: np.ones(n, np.float32)  # noqa: E731
    sc, th = [], []
    for seed in range(seeds):
        for ds in datasets:
            test = np.where(meta.dataset == ds)[0]
            src = np.where(meta.dataset != ds)[0]
            tr, va = split_validation(meta, src, seed)
            net = M.fit(x[tr], y[tr], ones(len(tr)), x[va], y[va], seed, augment=aug)
            thr = {"val-slice": choose_threshold(M.predict_proba(net, x[va]), y[va])}
            p = M.predict_proba(net, x[test])
            del net
            falls = []
            for s in sorted(set(datasets) - {ds}):
                s_idx = np.where(meta.dataset == s)[0]
                itr, iva = split_validation(meta, np.where(~meta.dataset.isin([ds, s]))[0], seed)
                inet = M.fit(x[itr], y[itr], ones(len(itr)), x[iva], y[iva], seed, augment=aug)
                ps = M.predict_proba(inet, x[s_idx])
                falls.append(ps[y[s_idx] == 1])
                del inet
                torch.cuda.empty_cache()
            thr["lodo-pooled"] = balanced_threshold(falls)
            sc.append(pd.DataFrame({"seed": seed, "dataset": ds, "idx": test, "score": p}))
            th += [{"seed": seed, "dataset": ds, "rule": r, "threshold": t} for r, t in thr.items()]
            print(f"[seed {seed}] {ds} done ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(sc), pd.DataFrame(th)


# ---------------------------------------------------------------------------------- events
def count_events(times: np.ndarray, refractory: float) -> int:
    """Greedy refractory: an alarm at t is suppressed if the last counted alarm was < refractory ago."""
    n, last = 0, -np.inf
    for t in np.sort(times):
        if t - last >= refractory:
            n += 1
            last = t
    return n


def evaluate(meta: pd.DataFrame, scores: pd.DataFrame, thr: pd.DataFrame, refractory: float) -> pd.DataFrame:
    """Per (seed, dataset, rule, ADL category): clips, hours, alarm events, plus fall-event sensitivity."""
    thr_map = {(r.seed, r.dataset, r.rule): r.threshold for r in thr.itertuples()}
    m = meta.assign(cat=[CATEGORY.get((d, str(a))) for d, a in zip(meta.dataset, meta.activity)])
    unmapped = m[(m.label == 0) & m.cat.isna()]
    assert unmapped.empty, f"unmapped ADL activities: {sorted(set(zip(unmapped.dataset, unmapped.activity)))}"
    rows = []
    for (seed, ds), g in scores.groupby(["seed", "dataset"]):
        w = m.loc[g.idx].assign(score=g.score.to_numpy())
        adl, fall = w[w.label == 0], w[w.label == 1]
        dur = adl.groupby("rec").start_s.agg(lambda s: s.max() - s.min() + WIN_S)
        for rule in RULES:
            t = thr_map[(seed, ds, rule)]
            fired = adl[adl.score >= t]
            end_times = {rec: (gg.start_s.to_numpy() + WIN_S) for rec, gg in fired.groupby("rec")}
            ev = pd.Series({rec: count_events(end_times.get(rec, np.array([])), refractory) for rec in dur.index})
            win = fired.groupby("rec").size().reindex(dur.index, fill_value=0)
            per = pd.DataFrame({"cat": adl.groupby("rec").cat.first(), "dur_s": dur, "events": ev, "windows": win})
            for cat, gc in per.groupby("cat"):
                rows.append({"seed": seed, "dataset": ds, "rule": rule, "cat": cat, "clips": len(gc),
                             "hours": gc.dur_s.sum() / 3600, "events": int(gc.events.sum()),
                             "windows": int(gc.windows.sum()), "fall_events": np.nan, "falls_detected": np.nan})
            hit = (fall.assign(f=fall.score >= t).groupby("rec").f.any()) if len(fall) else pd.Series(dtype=bool)
            rows.append({"seed": seed, "dataset": ds, "rule": rule, "cat": "_falls", "clips": 0, "hours": 0.0,
                         "events": 0, "windows": 0, "fall_events": len(hit), "falls_detected": int(hit.sum())})
    return pd.DataFrame(rows)


def rates(ev: pd.DataFrame) -> pd.DataFrame:
    """Per (rule, category): seed-averaged events/clip and events/hour (pooled over datasets within a seed)."""
    a = ev[ev.cat != "_falls"].groupby(["rule", "cat", "seed"])[["clips", "hours", "events", "windows"]].sum()
    a["per_clip"] = a.events / a.clips
    a["per_hour"] = a.events / a.hours
    a["ub_per_hour"] = gamma.ppf(0.95, a.events + 1) / a.hours  # exact Poisson one-sided 95% bound
    a["ub_per_clip"] = gamma.ppf(0.95, a.events + 1) / a.clips
    return a


def daily(rt: pd.DataFrame, budget: dict, upper: bool) -> pd.DataFrame:
    """Alarm events per day per category for each (rule, seed); ``upper`` uses each category's 95% upper bound."""
    out = []
    for (rule, cat, seed), r in rt.iterrows():
        amount = budget.get(cat, 0.0)
        if KIND[cat] == "hours":
            v = r.ub_per_hour if upper else r.per_hour
        else:
            v = r.ub_per_clip if upper else r.per_clip
        out.append({"rule": rule, "cat": cat, "seed": seed, "per_day": v * amount})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------- driver
def run(seeds: int = 5) -> None:
    t0 = time.time()
    out = REPORTS / "alarms"
    out.mkdir(parents=True, exist_ok=True)
    sp, tp = out / "scores.csv.gz", out / "thresholds.csv"
    if sp.exists() and tp.exists() and pd.read_csv(tp).seed.nunique() >= seeds:
        scores, thr = pd.read_csv(sp), pd.read_csv(tp)  # reuse saved scores (delete to retrain)
    else:
        scores, thr = score(seeds)
        scores.to_csv(sp, index=False)
        thr.to_csv(tp, index=False)
    write_report(scores, thr, out, seeds, time.time() - t0)
    print(f"report: {out / 'report.md'}")


def write_report(scores: pd.DataFrame, thr: pd.DataFrame, out, seeds: int, seconds: float) -> None:
    meta = pd.read_csv(PROCESSED / "meta.csv")
    ev = evaluate(meta, scores, thr, REFRACTORY_S)
    rt = rates(ev)
    L = ["# False alarms: from replayed clips to a daily estimate\n"]
    L.append(f"Seeds: {seeds}. Leave-one-dataset-out, accel-only augmented CNN; every window of the held-out "
             f"dataset is scored. Alarm events merge fired windows of a recording with a {REFRACTORY_S:.0f} s "
             "refractory period. Categories map the four datasets' ADL codes to common activities "
             "(`alarms.py`, UP-Fall codes from the dataset's published list, not from the files). "
             "**The daily estimates multiply measured per-activity rates by an ASSUMED time budget; the budgets "
             "are not measurements, so treat the daily numbers as scenario arithmetic.** All data are scripted "
             "clips of simulated activity, not free-living wear.\n")

    # --- window replay vs event
    L.append("## 1. What event counting changes\n")
    L.append("| rule | windows fired per replay hour | alarm events per replay hour (refractory sweep: "
             + " / ".join(f"{r:g} s" for r in REFRACTORY_SWEEP) + ") | fall events detected |")
    L.append("|---|---|---|---|")
    sweeps = {r: evaluate(meta, scores, thr, r) for r in REFRACTORY_SWEEP}
    for rule in RULES:
        base = ev[(ev.rule == rule) & (ev.cat != "_falls")]
        hours = base.groupby("seed").hours.sum()
        winph = (base.groupby("seed").windows.sum() / hours).mean()
        evph = [(sweeps[r][(sweeps[r].rule == rule) & (sweeps[r].cat != "_falls")].groupby("seed").events.sum() / hours).mean()
                for r in REFRACTORY_SWEEP]
        f = ev[(ev.rule == rule) & (ev.cat == "_falls")].groupby("seed")[["falls_detected", "fall_events"]].sum()
        L.append(f"| {rule} | {winph:.1f} | " + " / ".join(f"{v:.1f}" for v in evph)
                 + f" | {(f.falls_detected / f.fall_events).mean():.3f} ({int(f.fall_events.mean())} fall recordings) |")
    L.append("\nThis is still the *replay* mix (lots of jogging, hopping and clapping), so it is not a daily rate; "
             "it only shows how much of the old number was overlapping windows.\n")

    # --- per activity
    L.append("## 2. Alarm rate per activity (refractory 30 s)\n")
    L.append("`events/clip` suits discrete activities, `events/h` continuous ones. In brackets: exact Poisson "
             "one-sided 95% upper bound. Counts are small and clustered by subject (a few volunteers repeating "
             "one motion), so the true uncertainty is larger than the bound shows.\n")
    for rule in RULES:
        L.append(f"### Rule: {rule}\n")
        L.append("| category | enters a day as | clips | hours | alarm events (mean per seed) | events/clip | events/h |")
        L.append("|---|---|---|---|---|---|---|")
        r = rt.loc[rule].groupby("cat").agg(clips=("clips", "mean"), hours=("hours", "mean"), events=("events", "mean"),
                                           per_clip=("per_clip", "mean"), per_hour=("per_hour", "mean"),
                                           ub_c=("ub_per_clip", "mean"), ub_h=("ub_per_hour", "mean"))
        for cat in r.sort_values("per_hour", ascending=False).index:
            x = r.loc[cat]
            L.append(f"| {cat} | {KIND[cat]} | {x.clips:.0f} | {x.hours:.2f} | {x.events:.1f} | "
                     f"{x.per_clip:.3f} (≤{x.ub_c:.3f}) | {x.per_hour:.2f} (≤{x.ub_h:.2f}) |")
        L.append("")

    # --- daily
    L.append("## 3. Daily estimate under assumed time budgets\n")
    L.append("Alarm events per day, mean ± std over seeds. *point* uses measured rates; *upper* uses every "
             "category's 95% upper bound (a conservative sum of bounds).\n")
    L.append("| budget | rule | point estimate | upper (sum of 95% bounds) |")
    L.append("|---|---|---|---|")
    contrib = {}
    for name, b in BUDGETS.items():
        for rule in RULES:
            vals = {}
            for upper in (False, True):
                d = daily(rt[rt.index.get_level_values(0) == rule], b, upper)
                tot = d.groupby("seed").per_day.sum()
                vals[upper] = (tot.mean(), tot.std(ddof=0))
                if not upper:
                    contrib[(name, rule)] = d.groupby("cat").per_day.mean().sort_values(ascending=False)
            L.append(f"| {name} | {rule} | {vals[False][0]:.1f} ± {vals[False][1]:.1f} | {vals[True][0]:.1f} ± {vals[True][1]:.1f} |")
    L.append("")
    L.append("### Where the daily alarms come from (point estimate)\n")
    for (name, rule), c in contrib.items():
        top = ", ".join(f"{k} {v:.1f}" for k, v in c.head(5).items() if v > 0)
        L.append(f"- {name}, {rule}: {top}")
    L.append("\n`stationary` comes only from UP-Fall's 60 s sitting/standing/laying clips (no dataset has hours "
             "of rest or sleep); its alarm events come from very few clips, so its daily contribution is the "
             "least certain part of the estimate.")
    L.append("")
    L.append("### Budgets used (assumptions)\n")
    cats = sorted(KIND)
    L.append("| category | unit | " + " | ".join(BUDGETS) + " |")
    L.append("|---|---|" + "---|" * len(BUDGETS))
    for c in cats:
        L.append(f"| {c} | {'h/day' if KIND[c] == 'hours' else 'times/day'} | " + " | ".join(f"{b.get(c, 0):g}" for b in BUDGETS.values()) + " |")
    L.append("\nNot covered by any dataset, counted as stationary: sleep movement, driving, typing, cooking, "
             "cleaning, sport other than jogging/jumping. Vigorous ones among them would add alarms.\n")
    (out / "report.md").write_text("\n".join(L))
