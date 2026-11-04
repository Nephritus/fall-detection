"""Stillness check: an alarm is only raised once the wrist has stayed still for a wait period after the
detector's decision (GLOSSARY: Stillness check). Apple Watch, for comparison, calls emergency services only
after about a minute of immobility (support.apple.com/guide/watch/apd34c409704).

Motion is the per-second standard deviation of the acceleration vector (square root of the summed per-axis
variances, g): a constant calibration offset does not count as motion, a rotating wrist does. The check is
applied to every firing detection at its decision time, then the detections that pass merge into alarm events
with the usual refractory period. Two systems:

* ``stage1``: the window CNN trained with Capture-24 negatives (final models of ``negatives --final``, the
  validation-slice threshold); a window's decision time is its end;
* ``two-stage``: the second-stage confirmation of ``stage2.py`` at its practical thresholds, per
  leave-one-fall-dataset-out fold; decision time = candidate impact + 4 s.

Free-living cost: the 48 held-out Capture-24 people. Fall cost: lab fall recordings that the system detects,
with the check starting at impact + 4 s for both systems. Lab recordings end a few seconds after the impact,
so only short waits can be checked on falls. Sensor noise differs between devices, so the motion level that
counts as still has to sit above the target watch's noise floor (stored as ``noise-floor``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import stage2
from .alarms import REFRACTORY_S, FalseAlarms, per_day_ci
from .canonical import FS
from .freeliving import OUT, SOURCE
from .paths import PROCESSED, RAW, REPORTS
from .runner import Results

STILL_G = 0.02
RULES = ("strict", "most", "mean")  # every second / >= MOST_SHARE of seconds / the average below the level
MOST_SHARE = 0.8
WAITS = (0, 2, 5, 10, 20, 30, 60, 120)
LEVELS = (0.01, 0.02, 0.05)
FALL_WAITS = (2, 5, 10)
HORIZON_S = max(WAITS)
RECOVERY = {"905", "906", "914", "916"}  # Erciyes falls with recovery
RUNS = REPORTS / "stillness" / "runs"
C24 = OUT / "c24random"


# ------------------------------------------------------------------------------------- the check
def post_motion(acc: np.ndarray, starts_s: np.ndarray, seconds: int) -> np.ndarray:
    """(n starts, ``seconds``) per-second standard deviation of the acceleration vector after each start
    (g); NaN past the signal's end."""
    x = np.asarray(acc, np.float64)
    c1 = np.concatenate([np.zeros((1, 3)), np.cumsum(x, axis=0)])
    c2 = np.concatenate([np.zeros((1, 3)), np.cumsum(x * x, axis=0)])
    a = np.round(np.asarray(starts_s, float) * FS).astype(np.int64)
    edges = a[:, None] + np.arange(seconds + 1) * FS
    e = np.minimum(edges, len(x))
    s1, s2 = (c1[e[:, 1:]] - c1[e[:, :-1]]) / FS, (c2[e[:, 1:]] - c2[e[:, :-1]]) / FS
    m = np.sqrt(np.maximum((s2 - s1 * s1).sum(axis=-1), 0.0))
    m[edges[:, 1:] > len(x)] = np.nan
    return m.astype(np.float32)


def stays_still(motion: np.ndarray, wait_s: int, level: float = STILL_G, rule: str = "strict") -> np.ndarray:
    """Whether each row of ``post_motion`` stays still for ``wait_s`` seconds; unknown motion never passes."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    m = np.asarray(motion)[:, :wait_s]
    if wait_s == 0:
        return np.ones(len(m), bool)
    known = ~np.isnan(m).any(axis=1)
    with np.errstate(invalid="ignore"):
        if rule == "strict":
            ok = (m < level).all(axis=1)
        elif rule == "most":
            ok = (m < level).mean(axis=1) >= MOST_SHARE
        else:
            ok = m.mean(axis=1) < level
    return ok & known


# ------------------------------------------------------------------------------------- systems
def _group(dataset: str, activity: str) -> str:
    if dataset == "erciyes":
        return "erciyes recovery" if activity in RECOVERY else "erciyes other"
    return dataset


def _two_stage(seed: int, fold: str, meta: pd.DataFrame, lab: pd.DataFrame) -> tuple[FalseAlarms, float, pd.DataFrame]:
    """Stored stage-2 result: Capture-24 candidates (score = stage 2 where stage 1 fires) and fall recordings."""
    z = Results(stage2.RUNS).arrays("two-stage", seed, fold)
    fl = {k: z[f"fl_{k}"] for k in ("t", "s1", "p2", "lab", "pid")}
    fa = stage2.c24_alarms(fl, np.where(fl["s1"] >= z["t1_val"], fl["p2"], -np.inf), stage2.wearer_hours(z))
    T = lab[(lab.dataset == fold) & lab.is_fall].set_index("rec")  # the fall recordings in stage2.run's order
    T = T.loc[T.index.isin(meta[(meta.dataset == fold) & (meta.label == 1)].rec)]
    if len(T) != len(z["fall_s1"]):
        raise RuntimeError(f"stage-2 fall recordings of {fold} do not match the lab cache")
    detected = z["fall_trig"] & (z["fall_s1"] >= z["t1_val"]) & (z["fall_p2"] >= z["t2_val"])
    falls = pd.DataFrame({"rec": T.index, "detected": detected, "counted": z["fall_ctx"]})
    return fa, float(z["t2_val"]), falls


def _stage1(seed: int, meta: pd.DataFrame, pids: list[str]) -> tuple[FalseAlarms, float, pd.DataFrame]:
    """Final model with Capture-24 negatives: window scores of the held-out people; nested out-of-dataset fall
    scores (models trained without the fall's dataset) give the detected fall recordings."""
    thr = pd.read_csv(C24 / "thresholds.csv")
    t = float(thr[(thr.seed == seed) & (thr.rule == "val-slice")].threshold.iloc[0])
    fa = FalseAlarms.from_window_scores([np.load(C24 / "scores" / f"{p}.npy")[seed] for p in pids])
    fs = np.load(C24 / "fall_scores" / f"seed{seed}.npz")
    parts = []
    for ds in fs.files:
        rec = meta[(meta.dataset == ds) & (meta.label == 1)].rec.to_numpy()
        if len(rec) != len(fs[ds]):
            raise RuntimeError(f"fall scores of {ds} do not match meta.csv")
        hit = pd.Series(fs[ds] >= t).groupby(rec).max()
        parts.append(pd.DataFrame({"rec": hit.index, "detected": hit.to_numpy(), "counted": True}))
    return fa, t, pd.concat(parts, ignore_index=True)


def _keys() -> list[tuple[str, int, str]]:
    two = [("two-stage", s, d) for _, s, d in Results(stage2.RUNS).keys()]
    n = np.load(C24 / "scores" / f"{SOURCE.split(0).held_out[0]}.npy", mmap_mode="r").shape[0]
    return [("noise-floor", 0, "all")] + [("stage1", s, "all") for s in range(n)] + two


def _seconds(acc: np.ndarray) -> np.ndarray:
    return post_motion(acc, np.array([0.0]), len(acc) // FS)[0]


def _noise_floor(recs: list, signals: dict, labels: dict) -> list[dict]:
    """Motion when nothing moves, per source: the quietest second of each lab ADL recording (median over
    recordings) and the median second of Capture-24 sleep (median over people)."""
    rows = []
    for ds in sorted({r.dataset for r in recs}):
        q = [_seconds(r.acc).min() for r in recs if r.dataset == ds and not r.is_fall and len(r.acc) >= FS]
        rows.append({"source": ds, "device_note": "lab ADL, quietest second", "motion_g": float(np.median(q))})
    sleep = []
    for p, acc in signals.items():
        m = _seconds(acc)
        lab = labels[p][: len(m) * FS: FS]
        if (lab == "sleep").any():
            sleep.append(np.median(m[lab == "sleep"]))
    rows.append({"source": "capture24", "device_note": "free-living sleep, median second", "motion_g": float(np.median(sleep))})
    return rows


def run() -> None:
    """Motion after every firing detection and every detected fall, stored per (system, seed, fold)."""
    from .loaders import LOADERS

    results, t0 = Results(RUNS), time.time()
    todo = [k for k in _keys() if not results.has(*k)]
    if todo:
        pids = SOURCE.split(0).held_out
        meta = pd.read_csv(PROCESSED / "meta.csv")
        lab = pd.read_pickle(stage2.CACHE / "lab.pkl")
        recs = []
        for loader in LOADERS.values():
            try:
                recs += loader(RAW)[0]
            except FileNotFoundError:
                pass
        signals = {p: SOURCE.signal(p) for p in pids}
        labels = {p: SOURCE.labels(p) for p in pids}
        print(f"[stillness] inputs loaded ({time.time() - t0:.0f}s)", flush=True)
    for system, seed, fold in todo:
        if system == "noise-floor":
            results.save(system, seed, fold, _noise_floor(recs, signals, labels))
            continue
        fa, thr, falls = _two_stage(seed, fold, meta, lab) if system == "two-stage" else _stage1(seed, meta, pids)
        det = fa.detections(thr)
        mot, labs, ts, ws = [], [], [], []
        for k, (p, t) in enumerate(zip(pids, det)):
            if not len(t):
                continue
            mot.append(post_motion(signals[p], t, HORIZON_S))
            labs.append(labels[p][np.minimum(np.round(t * FS).astype(int), len(labels[p]) - 1)])
            ts.append(t)
            ws.append(np.full(len(t), k))
        r = [recs[i] for i in falls.rec]
        start = np.array([x.impact_s + stage2.POST_S for x in r])
        fall_motion = np.stack([post_motion(x.acc, s[None], HORIZON_S)[0] for x, s in zip(r, start)])
        arrays = {"det_t": np.concatenate(ts), "det_w": np.concatenate(ws), "det_lab": np.concatenate(labs).astype(str),
                  "det_motion": np.concatenate(mot), "hours": fa.hours,
                  "fall_rec": falls.rec.to_numpy(), "fall_group": np.array([_group(x.dataset, str(x.activity)) for x in r]),
                  "fall_activity": np.array([str(x.activity) for x in r]), "fall_detected": falls.detected.to_numpy(bool),
                  "fall_counted": falls.counted.to_numpy(bool), "fall_motion": fall_motion,
                  "fall_post_s": np.array([len(x.acc) / FS for x in r]) - start}
        row = {"threshold": thr, "detections": len(arrays["det_t"]), "hours": fa.hours.sum(), "falls": len(r),
               "alarms_per_day_without_check": fa.per_day(thr)}
        results.save(system, seed, fold, [row], arrays)
        print(f"[stillness] {system} seed {seed} {fold} done ({time.time() - t0:.0f}s)", flush=True)
    report()


# ---------------------------------------------------------------------------------------- report
def _alarms(a: dict, wait: int, level: float = STILL_G, rule: str = "strict", labels: list | None = None) -> FalseAlarms:
    """Free-living alarm events after the check. A recording that ends during the wait counts as still
    (conservative: the alarm is kept). ``labels``: per-wearer label exposure, to break events down by activity."""
    ok = stays_still(np.nan_to_num(a["det_motion"], nan=0.0), wait, level, rule)
    sel = [(a["det_w"] == k) & ok for k in range(len(a["hours"]))]
    return FalseAlarms([a["det_t"][m] for m in sel], [np.ones(m.sum()) for m in sel], a["hours"],
                       [a["det_lab"][m] for m in sel] if labels is not None else None, labels)


def report() -> None:
    out = REPORTS / "stillness"
    results = Results(RUNS)
    A = {k: results.arrays(*k) for k in results.keys() if k[0] != "noise-floor"}
    noise = results.table()
    noise = noise[noise.condition == "noise-floor"]
    systems = ("stage1", "two-stage")
    names = {"stage1": "stage 1", "two-stage": "two-stage"}
    pids = SOURCE.split(0).held_out

    def per_seed(system: str, fn) -> pd.Series:
        """``fn(arrays)`` averaged over folds per seed."""
        v = pd.DataFrame([{"seed": s, "v": fn(a)} for (c, s, _), a in A.items() if c == system])
        return v.groupby("seed").v.mean()

    def ms(system: str, fn, fmt: str = ".2f") -> str:
        v = per_seed(system, fn)
        return f"{v.mean():{fmt}} ± {v.std():{fmt}}"

    a0 = next(a for (c, _, _), a in A.items() if c == "two-stage")
    L = ["# Stillness check\n"]
    L.append(f"An alarm is raised only if the wrist stays still for a wait period after the detector's decision (stage 1: window end; "
             f"two-stage: impact + {stage2.POST_S:g} s). Motion = per-second standard deviation of the acceleration vector (g), so a constant "
             f"calibration offset is not motion but a rotating wrist is. `strict` = every second below the level, `most` = at least "
             f"{MOST_SHARE:.0%} of seconds, `mean` = the average. Each firing detection is checked separately, then the passing ones merge "
             f"into alarm events ({REFRACTORY_S:.0f} s refractory period). The alert comes **wait seconds later** than without the check.\n")
    L.append(f"Free-living cost: {len(pids)} held-out Capture-24 people ({a0['hours'].sum():.0f} h) that no model trained on. `stage 1` = window CNN "
             "with Capture-24 negatives (final models of `negatives --final`, validation-slice threshold); `two-stage` = `reports/stage2` at its "
             "practical thresholds, macro over the leave-one-fall-dataset-out folds. Mean ± std over seeds. A recording that ends during the wait "
             "counts as still (keeps the alarm).\n")

    L.append("## 1. Free-living alarms per day (`strict`)\n")
    L.append("| wait after decision | stage 1, 0.02 g | two-stage, 0.02 g | per week | two-stage, 0.05 g | per week |")
    L.append("|---|---|---|---|---|---|")
    for w in WAITS:
        cells = [ms("stage1", lambda a: _alarms(a, w).per_day(0.5))]
        for lv in (0.02, 0.05):
            v = per_seed("two-stage", lambda a: _alarms(a, w, lv).per_day(0.5))
            cells += [f"{v.mean():.2f} ± {v.std():.2f}", f"{v.mean() * 7:.1f}"]
        L.append(f"| {w} s | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 2. How strict the check is\n")
    L.append("Alarms per day, by rule and motion level, after 10 s and 60 s waits.\n")
    L.append("| rule | level (g) | stage 1, 10 s | stage 1, 60 s | two-stage, 10 s | two-stage, 60 s |")
    L.append("|---|---|---|---|---|---|")
    for rule in RULES:
        for lv in LEVELS:
            cells = [ms(s, lambda a: _alarms(a, w, lv, rule).per_day(0.5)) for s in systems for w in (10, 60)]
            L.append(f"| {rule} | {lv:g} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 3. Sensor noise: what \"still\" can mean per device\n")
    L.append("Motion when nothing moves. A level below a device's noise floor rejects everything recorded with it, falls included.\n")
    L.append("| source | measured on | motion (g) |")
    L.append("|---|---|---|")
    for r in noise.itertuples():
        L.append(f"| {r.source} | {r.device_note} | {r.motion_g:.3f} |")
    L.append("")

    L.append(f"## 4. Per person (`strict`, {STILL_G:g} g)\n")
    L.append("Each person's alarms per day averaged over seeds and folds; people are recorded for about a day each, so "
             "\"no alarm\" means none in their whole recording (mean over seeds and folds). 95% interval: bootstrap over people of the pooled rate.\n")
    L.append("| system | wait | pooled alarms/day (95% CI) | median person | 90th percentile person | worst person | people with no alarm |")
    L.append("|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for s in systems:
        for w in (0, 10, 60):
            fas = [_alarms(a, w) for (c, _, _), a in A.items() if c == s]
            rate = np.mean([fa.per_day_by_wearer(0.5) for fa in fas], axis=0)
            ev = np.sum([fa.events(0.5) for fa in fas], axis=0)
            lo, hi = per_day_ci(ev, fas[0].hours * len(fas), rng)
            none = np.mean([(fa.events(0.5) == 0).mean() for fa in fas])
            L.append(f"| {names[s]} | {w} s | {rate.mean():.2f} ({lo:.2f}–{hi:.2f}) | {np.median(rate):.2f} | {np.quantile(rate, 0.9):.2f} | "
                     f"{rate.max():.1f} | {none:.0%} |")
    L.append("")

    L.append("## 5. What the check costs on lab falls\n")
    L.append(f"`detected` = share of fall recordings (simulated falls) the system detects without any check. The check columns: of the detected "
             f"ones, the share that also pass it, starting at impact + {stage2.POST_S:g} s. Only recordings that continue long enough after the "
             "impact can be checked: `n` is how many (summed over folds, mean over seeds), so longer waits rest on few, unrepresentative "
             "recordings. In Erciyes 905, 906, 914 and 916 the volunteer gets up. Read each dataset against its noise floor (section 3).\n")
    hdr = " | ".join(f"{w} s: n / pass" for w in FALL_WAITS)
    for lv in (0.02, 0.05):
        L.append(f"`strict`, {lv:g} g:\n")
        L.append(f"| system | falls | detected | time recorded after impact + {stage2.POST_S:g} s (median) | {hdr} |")
        L.append("|---|---|---|---|" + "---|" * len(FALL_WAITS))
        for s in systems:
            rows = []
            for (c, seed, fold), a in A.items():
                if c != s:
                    continue
                counted = a["fall_counted"]
                use = a["fall_detected"] & counted
                for g in np.unique(a["fall_group"][counted]):
                    m = use & (a["fall_group"] == g)
                    r = {"seed": seed, "group": g, "post": np.median(a["fall_post_s"][m]),
                         "falls": (counted & (a["fall_group"] == g)).sum(), "det": m.sum()}
                    for w in FALL_WAITS:
                        known = m & ~np.isnan(a["fall_motion"][:, :w]).any(axis=1)
                        r[f"n{w}"] = known.sum()
                        r[f"k{w}"] = stays_still(a["fall_motion"][known], w, lv).sum()
                    rows.append(r)
            D = pd.DataFrame(rows).groupby(["group", "seed"]).agg(
                {**{f"{p}{w}": "sum" for p in "nk" for w in FALL_WAITS}, "post": "mean", "falls": "sum", "det": "sum"}).reset_index()
            for g, d in D.groupby("group"):
                cells = " | ".join(f"{d[f'n{w}'].mean():.0f} / {d[f'k{w}'].sum() / d[f'n{w}'].sum():.2f}" if d[f"n{w}"].sum() else "0 / –"
                                   for w in FALL_WAITS)
                L.append(f"| {names[s]} | {g} | {d.det.sum() / d.falls.sum():.2f} | {d.post.mean():.1f} s | {cells} |")
        L.append("")

    L.append(f"## 6. Where the remaining alarms come from (two-stage, `strict`, {STILL_G:g} g)\n")
    label_hours = []
    for p in pids:
        u, c = np.unique(SOURCE.labels(p), return_counts=True)
        label_hours.append({k: n / FS / 3600 for k, n in zip(u, c)})
    hours = pd.DataFrame(label_hours).sum()
    L.append("Alarm events per hour of each Capture-24 activity (pooled over seeds and folds; an event belongs to the activity at its first detection).\n")
    L.append("| activity | hours | no check | 10 s | 60 s |")
    L.append("|---|---|---|---|---|")
    cnt, keys = {}, [k for k in A if k[0] == "two-stage"]
    for k in keys:
        for w in (0, 10, 60):
            for lbl, v in _alarms(A[k], w, labels=label_hours).by_label(0.5).groupby("label").events.sum().items():
                cnt[(w, lbl)] = cnt.get((w, lbl), 0) + int(v)
    for lbl in sorted(hours.index, key=lambda x: -cnt.get((10, x), 0)):
        cells = " | ".join(f"{cnt.get((w, lbl), 0) / (hours[lbl] * len(keys)):.3f}" for w in (0, 10, 60))
        L.append(f"| {lbl} | {hours[lbl]:.0f} | {cells} |")
    L.append("")

    L.append("## Reading\n")
    rate = {(s, w): per_seed(s, lambda a: _alarms(a, w).per_day(0.5)).mean() for s in systems for w in (0, 10, 60)}
    L.append(f"* Without a check, two-stage halves stage 1's alarms ({rate[('stage1', 0)]:.1f} → {rate[('two-stage', 0)]:.1f} per day). "
             f"With a `strict` {STILL_G:g} g check the two converge: {rate[('stage1', 10)]:.2f} vs {rate[('two-stage', 10)]:.2f} after 10 s, "
             f"{rate[('stage1', 60)]:.2f} vs {rate[('two-stage', 60)]:.2f} after 60 s. Stage 2's own gain comes mostly from short post-impact "
             "stillness, which the check already requires.")
    L.append("* The check is a product decision, not a better classifier: it trades alert delay and falls after which the wearer moves "
             "(gets up, struggles, moves an arm) for fewer false alarms.\n"
             "* The free-living side is measured; the fall side is not, beyond a few seconds. Lab falls are scripted and recordings stop "
             "early, so whether real fallers keep the wrist still for 10–60 s is unknown here. That needs fall recordings that continue "
             "for at least a minute after the impact, with and without getting up, on the target watch (its noise floor sets the level).\n"
             "* The levels and rules are a grid, not tuned on falls.\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    print(f"report: {out / 'report.md'}")
