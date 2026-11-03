"""Second-stage confirmation: does what happens *around* a candidate impact confirm the fall?

Stage 1 is the window CNN (trained with Capture-24 negatives, ``negatives.py``). Stage 2 looks at a
context segment around a candidate impact, 1 s before to 4 s after (so an alarm is delayed by 4 s),
through hand-built, rotation- and mirror-invariant features (|a| statistics before/after the impact,
stillness, posture change, repeated impacts) and a gradient-boosted classifier.

* candidates: local maxima of |a| >= TRIGGER_G, at least MIN_SEP_S apart, with the full context available
* stage-2 positives: fall recordings (the dataset's impact moment); negatives: candidates in lab ADL
  recordings and in Capture-24 people P001-P100 (<= NEG_PER_PERSON per person)
* alarm = candidate where stage 1 (max score of the windows containing it) >= t1 AND stage 2 >= t2

Evaluation (leave-one-fall-dataset-out, held-out Capture-24 people P101-P151 never trained on):
sensitivity is **per fall recording** (detected if its impact candidate is confirmed), free-living cost
is alarm events per day (30 s refractory). Fall recordings without the full context, or whose impact is
below the trigger, are reported separately (below-trigger ones count as misses).
"""

from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .alarms import REFRACTORY_S, FalseAlarms
from .augment import Augment
from .canonical import FS
from .paths import PROCESSED, RAW, REPORTS
from .capture24 import LABEL_NAMES
from .freeliving import OUT, SOURCE
from .negatives import RANDOM_K, VAL_K
from .rules import keep_falls
from .runner import FallData, Results, lodo_folds
from .windows import STRIDE_S, WIN_S

TRIGGER_G, MIN_SEP_S = 2.0, 2.0
PRE_S, POST_S = 1.0, 4.0
NEG_PER_PERSON = 600
SEGS = {"pre": (-1.0, -0.2), "peak": (-0.2, 0.5), "p1": (0.5, 2.0), "p2": (2.0, 4.0)}
GRID = np.r_[0.0, 1 / (1 + np.exp(-np.linspace(-6, 6.5, 26)))]
CACHE = OUT / "stage2"
RUNS = REPORTS / "stage2" / "runs"
WIN, STRIDE = round(WIN_S * FS), round(STRIDE_S * FS)
BUDGETS = (10, 25, 50, 100)
SENS_TARGETS = (0.80, 0.90, 0.95)


def wearer_hours(z) -> list[float]:
    """Exposure per held-out person for a saved fold (older files lack it: rebuilt from signal lengths)."""
    if "wearer_hours" in z:
        return [float(h) for h in z["wearer_hours"]]
    n = [int(np.load(CACHE / f"{p}.npz", allow_pickle=True)["n"]) for p in SOURCE.split(0).held_out]
    return [((k - WIN) // STRIDE + 1) * STRIDE_S / 3600 for k in n]


# --------------------------------------------------------------------------------- features
def _r(s: float) -> int:
    return int(round(s * FS))


def candidates(acc: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(acc, axis=1)
    pk, _ = find_peaks(mag, height=TRIGGER_G, distance=_r(MIN_SEP_S))
    return pk[(pk >= _r(PRE_S)) & (pk + _r(POST_S) <= len(acc))]


def features(acc: np.ndarray, idx: np.ndarray) -> pd.DataFrame:
    """Context features for candidate sample indices ``idx`` (all need the full context)."""
    mag = np.linalg.norm(acc, axis=1)

    def cs(a):
        return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0, dtype=np.float64)])

    c_dyn, c_mag, c_mag2, c_xyz = cs(np.abs(mag - 1)), cs(mag), cs(mag**2), cs(acc)
    c_hi, c_mid = cs((mag >= 2.0).astype(float)), cs((mag >= 1.5).astype(float))

    def mean(c, a, b):
        return (c[idx + _r(b)] - c[idx + _r(a)]) / (_r(b) - _r(a))

    out, vec = {"peak": mag[idx]}, {}
    for name, (a, b) in SEGS.items():
        mm = mean(c_mag, a, b)
        out[f"{name}_dyn"] = mean(c_dyn, a, b)
        out[f"{name}_std"] = np.sqrt(np.maximum(mean(c_mag2, a, b) - mm**2, 0))
        out[f"{name}_hi"] = mean(c_hi, a, b)
        v = mean(c_xyz, a, b)
        vec[name] = v
        out[f"{name}_still"] = np.linalg.norm(v, axis=1) / np.maximum(mm, 1e-6)
    out["peak_wide"] = mean(c_mid, -0.2, 0.5)

    def angle(u, v):
        c = (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    out["ang_pre_p1"], out["ang_pre_p2"] = angle(vec["pre"], vec["p1"]), angle(vec["pre"], vec["p2"])
    out["ang_p1_p2"] = angle(vec["p1"], vec["p2"])
    out["lr_p1"] = np.log((out["p1_dyn"] + 0.01) / (out["pre_dyn"] + 0.01))
    out["lr_p2"] = np.log((out["p2_dyn"] + 0.01) / (out["pre_dyn"] + 0.01))
    out["p1_max"] = mag[idx[:, None] + np.arange(_r(0.5), _r(2.0))].max(1)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------- caches
def build_cache() -> None:
    """Candidate features for every Capture-24 person and every lab recording (cached)."""
    from .loaders import LOADERS

    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for pid in SOURCE.people():
        f = CACHE / f"{pid}.npz"
        if f.exists():
            continue
        acc = SOURCE.signal(pid)
        idx = candidates(acc)
        feat = features(acc, idx)
        np.savez(f, idx=idx, lab=SOURCE.labels(pid)[idx].astype(str), n=len(acc), cols=np.array(feat.columns),
                 X=feat.to_numpy(np.float32))
        if int(pid[1:]) % 20 == 0:
            print(f"[cache] {pid} {len(idx)} candidates ({time.time() - t0:.0f}s)", flush=True)
    if (CACHE / "lab.pkl").exists():
        return
    recs = []
    for loader in LOADERS.values():
        try:
            recs += loader(RAW)[0]
        except FileNotFoundError:
            pass
    meta = pd.read_csv(PROCESSED / "meta.csv").drop_duplicates("rec").set_index("rec")
    for ri in np.random.default_rng(0).choice(meta.index, 50, replace=False):  # recording order matches prepare
        r, m = recs[ri], meta.loc[ri]
        assert (r.dataset, r.subject, str(r.activity)) == (m.dataset, m.subject, str(m.activity)), "recording order mismatch"
    rows = []
    for ri, r in enumerate(recs):
        acc = r.acc.astype(np.float64)
        if r.is_fall:
            i = round(r.impact_s * FS)
            ctx = i >= _r(PRE_S) and i + _r(POST_S) <= len(acc)
            idx = np.array([i]) if ctx else np.array([], int)
            trig = bool(np.linalg.norm(acc[i]) >= TRIGGER_G)
        else:
            idx, trig = candidates(acc), False
        base = pd.DataFrame({"rec": ri, "dataset": r.dataset, "subject": r.subject, "activity": str(r.activity),
                             "is_fall": r.is_fall, "idx": idx, "trig": trig, "ctx": len(idx) > 0 or not r.is_fall})
        if r.is_fall and not len(idx):
            base = pd.DataFrame([{"rec": ri, "dataset": r.dataset, "subject": r.subject, "activity": str(r.activity),
                                  "is_fall": True, "idx": -1, "trig": trig, "ctx": False}])
        elif len(idx):
            base = pd.concat([base, features(acc, idx)], axis=1)
        rows.append(base)
    pd.concat(rows, ignore_index=True).to_pickle(CACHE / "lab.pkl")
    print(f"[cache] lab recordings done ({time.time() - t0:.0f}s)", flush=True)


def load_people(pids: list[str]) -> dict:
    out = {}
    for p in pids:
        z = np.load(CACHE / f"{p}.npz", allow_pickle=True)
        out[p] = {"idx": z["idx"], "lab": z["lab"], "n": int(z["n"]), "X": z["X"], "cols": list(z["cols"])}
    return out


# ------------------------------------------------------------------------------------- stage 2
def stage1_at(sc: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Max stage-1 score of the windows (2 s stride) that contain each candidate sample."""
    hi = np.minimum(idx // STRIDE, len(sc) - 1)
    lo = np.clip(-((-(idx - WIN + 1)) // STRIDE), 0, len(sc) - 1)
    return np.maximum(sc[lo], sc[hi])


def fit_stage2(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4, l2_regularization=1.0, early_stopping=False,
        random_state=seed).fit(X, y, sample_weight=w)


def c24_alarms(fl: dict, score: np.ndarray, hours: list[float], label_hours: list[dict] | None = None) -> FalseAlarms:
    """Candidates of the held-out Capture-24 people as detections: alarm time = impact + POST_S (when the
    device would alert), one wearer per person. ``fl["t"]`` is impact time in seconds per person (files
    written before the counting refactor stored it offset by 1e6 s per person; the offset is removed)."""
    sel = [np.where(fl["pid"] == k)[0] for k in range(len(hours))]
    times = [np.round((fl["t"][i] - k * 1e6) * FS) / FS + POST_S for k, i in enumerate(sel)]
    lab = fl["lab"]
    if lab.dtype.kind in "iu":  # files written before the source refactor stored label codes
        lab = np.array([LABEL_NAMES[int(c)] for c in lab])
    labels = [lab[i] for i in sel] if label_hours is not None else None
    return FalseAlarms(times, [score[i] for i in sel], hours, labels, label_hours)


def eval_grid(fall: dict, fl: dict, hours: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-recording sensitivity and free-living alarms/day on the (t1, t2) grid."""
    n = len(GRID)
    sens, alarms = np.zeros((n, n)), np.zeros((n, n))
    for i, t1 in enumerate(GRID):
        alarms[i] = c24_alarms(fl, np.where(fl["s1"] >= t1, fl["p2"], -np.inf), hours).curve(GRID)[1]
        for j, t2 in enumerate(GRID):
            hit = fall["trig"] & (fall["s1"] >= t1) & (fall["p2"] >= t2)
            sens[i, j] = hit[fall["ctx"]].mean()
    return sens, alarms


def run(seed: int) -> None:
    t0 = time.time()
    build_cache()
    data, results, aug = FallData.load(), Results(RUNS), Augment()
    meta, y = data.meta, data.y
    split = SOURCE.split(seed)
    test_p, fit_p, val_p = split.held_out, split.train, split.validation
    x_c24_val = SOURCE.windows(val_p, VAL_K, seed, "validation")
    pids = SOURCE.people()
    people = load_people(pids)
    lab = pd.read_pickle(CACHE / "lab.pkl")
    cols = people[pids[0]]["cols"]
    feat_cols = [c for c in lab.columns if c in cols]
    assert feat_cols == cols, (feat_cols, cols)

    def people_rows(ps, cap, fold):
        """Up to ``cap`` candidates per person, keyed by (seed, fold, person) so resuming draws the same."""
        xs = []
        for p in ps:
            q = people[p]["X"]
            r = np.random.default_rng([seed, zlib.crc32(fold.encode()), zlib.crc32(p.encode())])
            xs.append(q[r.choice(len(q), min(cap, len(q)), replace=False)] if len(q) else q)
        return np.concatenate(xs)

    for fold in lodo_folds(data, seed):
        if results.has("two-stage", seed, fold.key):
            continue
        ds, test = fold.name, fold.test
        val_subj = set(meta.subject.iloc[fold.validation])
        # ---- stage 1: the same Capture-24 negatives as negatives.run's c24-random condition for this fold
        xc = SOURCE.windows(fit_p, RANDOM_K, seed, "train", ds)
        trained = fold.train_model("acc", augment=aug, extra=(xc, np.zeros(len(xc), np.int64)), extra_validation=x_c24_val)
        net, t1_val = trained.model, trained.threshold
        p_test = net.scores(data.windows(test, "acc"))
        sc = {p: net.score_signal(SOURCE.signal(p)) for p in test_p}
        del trained, net, xc
        torch.cuda.empty_cache()
        # ---- stage 2 training data
        L = lab[lab.dataset != ds]
        has = L[L.ctx & (L.idx >= 0)]
        is_tr = ~has.subject.isin(val_subj)
        parts = {"tr": has[is_tr], "va": has[~is_tr]}
        sets = {}
        for k, part in parts.items():
            ps = fit_p if k == "tr" else val_p
            neg = np.concatenate([part[~part.is_fall][feat_cols].to_numpy(np.float32), people_rows(ps, NEG_PER_PERSON, ds)])
            pos_df = part[part.is_fall]
            pos = pos_df[feat_cols].to_numpy(np.float32)
            w_pos = 1.0 / pos_df.groupby("dataset").dataset.transform("size").to_numpy()
            w_pos = w_pos / w_pos.sum() * len(neg)  # classes and datasets carry equal weight
            sets[k] = (np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                       np.r_[w_pos, np.ones(len(neg))])
        clf = fit_stage2(*sets["tr"], seed)
        pv = clf.predict_proba(sets["va"][0])[:, 1]
        yv = sets["va"][1]
        t2_val = keep_falls(pv, yv, share=0.98)
        # ---- held-out dataset: fall recordings
        T = lab[(lab.dataset == ds) & lab.is_fall].set_index("rec")
        wf = meta.iloc[test]
        wf = wf[wf.label == 1].assign(s=p_test[(y[test] == 1)]).groupby("rec").s.max()
        T = T.loc[T.index.isin(wf.index)]
        p2 = np.zeros(len(T))
        okm = (T.ctx & (T.idx >= 0)).to_numpy()
        if okm.any():
            p2[okm] = clf.predict_proba(T[okm][feat_cols].to_numpy(np.float32))[:, 1]
        fall = {"s1": wf.loc[T.index].to_numpy(), "p2": p2, "trig": T.trig.to_numpy() & okm, "ctx": T.ctx.to_numpy()}
        # ---- held-out Capture-24 people
        t_, s1_, p2_, lb_, pid_ = [], [], [], [], []
        for k, p in enumerate(test_p):
            d = people[p]
            if not len(d["idx"]):
                continue
            t_.append(d["idx"] / FS)
            s1_.append(stage1_at(sc[p], d["idx"]))
            p2_.append(clf.predict_proba(d["X"])[:, 1])
            lb_.append(d["lab"])
            pid_.append(np.full(len(d["idx"]), k))
        fl = {"t": np.concatenate(t_), "s1": np.concatenate(s1_), "p2": np.concatenate(p2_),
              "lab": np.concatenate(lb_), "pid": np.concatenate(pid_)}
        hours = [len(sc[p]) * STRIDE_S / 3600 for p in test_p]
        sens, alarms = eval_grid(fall, fl, hours)
        # stage 1 alone, window level (as in the earlier reports), per recording
        stage1 = FalseAlarms.from_window_scores([sc[p] for p in test_p])
        ev1 = [stage1.per_day(t1) for t1 in np.r_[GRID, t1_val]]
        sens1 = [(fall["s1"][fall["ctx"]] >= t1).mean() for t1 in np.r_[GRID, t1_val]]
        summary = dict(t1_val=t1_val, t2_val=t2_val, hours=sum(hours), stage2_val_auroc=roc_auc_score(yv, pv),
                       n_pos=int(sets["tr"][1].sum()), n_neg=int((1 - sets["tr"][1]).sum()), n_fall_recs=len(T),
                       n_fall_ctx=int(T.ctx.sum()), n_fall_trig=int((T.trig.to_numpy() & okm).sum()))
        results.save("two-stage", seed, fold.key, [summary],
                     dict(sens=sens, alarms=alarms, ev1=ev1, sens1=sens1, wearer_hours=hours, **summary,
                          **{f"fall_{k}": v for k, v in fall.items()}, **{f"fl_{k}": v for k, v in fl.items()}))
        print(f"[seed {seed}] {ds} done ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------------- report
def _frontier(sens: np.ndarray, alarms: np.ndarray, target: float) -> float:
    ok = sens >= target
    return float(alarms[ok].min()) if ok.any() else np.nan


def _sens_at(sens: np.ndarray, alarms: np.ndarray, budget: float) -> float:
    ok = alarms <= budget
    return float(sens[ok].max()) if ok.any() else 0.0


def report() -> None:
    out = REPORTS / "stage2"
    results = Results(RUNS)
    tab = results.table()
    Z = {(s, d): results.arrays(c, s, d) for c, s, d in tab[["condition", "seed", "fold"]].itertuples(index=False)}
    seeds = sorted({k[0] for k in Z})
    folds = sorted({k[1] for k in Z})
    held_out = SOURCE.split(0).held_out
    L = ["# Second-stage confirmation\n"]
    z0 = next(iter(Z.values()))
    L.append(f"Seeds: {len(seeds)}. Leave-one-fall-dataset-out ({', '.join(folds)}); stage 1 = accel-only augmented CNN trained with "
             f"Capture-24 negatives (as `reports/negatives`), stage 2 = gradient boosting on context features "
             f"({PRE_S:g} s before to {POST_S:g} s after a candidate impact, |a| ≥ {TRIGGER_G:g} g, so **alarms are delayed by "
             f"{POST_S:g} s**). Free-living cost is measured on the {len(held_out)} held-out Capture-24 people "
             f"({z0['hours']:.0f} h) that no model trained on; alarm events merge with a {REFRACTORY_S:.0f} s refractory period. "
             "Sensitivity is **per fall recording** (simulated falls; not comparable with the per-window figures of earlier reports). "
             "Recordings without the full context are excluded from every sensitivity; recordings whose impact is below the trigger count as misses.\n")
    stats = tab.rename(columns={"n_fall_recs": "recs", "n_fall_ctx": "ctx", "n_fall_trig": "trig",
                                "stage2_val_auroc": "auroc", "n_pos": "pos", "n_neg": "neg"})
    L.append("## 1. Coverage\n")
    L.append("| held-out dataset | fall recordings | with full context | impact ≥ trigger | stage-2 training positives / negatives (per seed) | stage-2 validation AUROC |")
    L.append("|---|---|---|---|---|---|")
    for d, g in stats.groupby("fold"):
        L.append(f"| {d} | {g.recs.mean():.0f} | {g.ctx.mean():.0f} | {g.trig.mean():.0f} | {g.pos.mean():.0f} / {g.neg.mean():.0f} | {g.auroc.mean():.3f} |")
    L.append("")

    def per(fold_seed):
        z = Z[fold_seed]
        return z["sens"], z["alarms"], z["ev1"], z["sens1"]

    rows, fr, sb = [], [], []
    for (s, d), z in Z.items():
        sens, alarms, ev1, sens1 = per((s, d))
        pract = dict(fold=d, seed=s)
        # practical rule: t1 = validation-slice threshold, t2 keeps 98% of stage-2 validation falls
        fall = {k: z[f"fall_{k}"] for k in ("s1", "p2", "trig", "ctx")}
        fl = {k: z[f"fl_{k}"] for k in ("t", "s1", "p2", "lab", "pid")}
        a = (fl["s1"] >= z["t1_val"])
        wh = wearer_hours(z)
        pract["s1_alarms"], pract["s1_sens"] = float(z["ev1"][-1]), float(z["sens1"][-1])
        pract["gate_alarms"] = c24_alarms(fl, fl["s1"], wh).per_day(float(z["t1_val"]))
        pract["two_alarms"] = c24_alarms(fl, np.where(a, fl["p2"], -np.inf), wh).per_day(float(z["t2_val"]))
        hit = lambda msk: (fall["trig"] & msk)[fall["ctx"]].mean()  # noqa: E731
        pract["gate_sens"] = hit(fall["s1"] >= z["t1_val"])
        pract["two_sens"] = hit((fall["s1"] >= z["t1_val"]) & (fall["p2"] >= z["t2_val"]))
        rows.append(pract)
        base = {"fold": d, "seed": s}
        for tg in SENS_TARGETS:
            fr.append({**base, "target": tg, "stage1": _frontier(np.array(z["sens1"][:-1])[:, None], np.array(z["ev1"][:-1])[:, None], tg),
                       "stage2": _frontier(sens[0:1, :], alarms[0:1, :], tg), "both": _frontier(sens, alarms, tg)})
        for b in BUDGETS:
            sb.append({**base, "budget": b, "stage1": _sens_at(np.array(z["sens1"][:-1]), np.array(z["ev1"][:-1]), b),
                       "stage2": _sens_at(sens[0:1, :], alarms[0:1, :], b), "both": _sens_at(sens, alarms, b)})
    P, F, S = pd.DataFrame(rows), pd.DataFrame(fr), pd.DataFrame(sb)

    L.append("## 2. A practical, non-oracle operating point\n")
    L.append("Stage 1 at its validation-slice threshold (95% of validation fall windows). Stage 2 threshold keeps 98% of validation "
             "fall candidates (a held-out slice of training subjects; no held-out data used). `stage 1 alone` counts fired windows as before; "
             "`stage 1 + trigger` only alarms on candidates (|a| peak ≥ 2 g); `two-stage` also needs stage 2. Macro over held-out datasets, mean ± std over seeds.\n")
    L.append("| system | alarm events / 24 h | fall-recording sensitivity |")
    L.append("|---|---|---|")
    for name, a, s_ in (("stage 1 alone", "s1_alarms", "s1_sens"), ("stage 1 + trigger", "gate_alarms", "gate_sens"), ("two-stage", "two_alarms", "two_sens")):
        m = P.groupby("seed")[[a, s_]].mean()
        L.append(f"| {name} | {m[a].mean():.1f} ± {m[a].std():.1f} | {m[s_].mean():.3f} ± {m[s_].std():.3f} |")
    L.append("\nPer held-out dataset (mean over seeds):\n")
    L.append("| held-out | stage 1 alone: alarms/day / sens | two-stage: alarms/day / sens |")
    L.append("|---|---|---|")
    for d, g in P.groupby("fold"):
        L.append(f"| {d} | {g.s1_alarms.mean():.1f} / {g.s1_sens.mean():.3f} | {g.two_alarms.mean():.1f} / {g.two_sens.mean():.3f} |")
    L.append("")

    L.append("## 3. Operating curves (threshold-free; thresholds chosen on the held-out data)\n")
    L.append("Lowest free-living alarms/day that reaches a per-recording sensitivity. `stage 1` = window CNN alone, `stage 2` = trigger + stage 2 alone, "
             "`both` = best (t1, t2) pair. `both` searches a two-dimensional grid, so it has more freedom than the one-dimensional curves: "
             "read it as an upper bound on what the combination can do, and use section 2 for a non-oracle figure.\n")
    L.append("| sensitivity | stage 1 | stage 2 | both | `both` vs stage 1 |")
    L.append("|---|---|---|---|---|")
    for tg in SENS_TARGETS:
        g = F[F.target == tg].groupby("seed")[["stage1", "stage2", "both"]].mean()
        L.append(f"| {tg:.2f} | {g.stage1.mean():.1f} | {g.stage2.mean():.1f} | {g.both.mean():.1f} | {g.both.mean() / g.stage1.mean():.2f}× |")
    L.append("")
    L.append("Sensitivity reachable within an alarm budget:\n")
    L.append("| alarms/day budget | stage 1 | stage 2 | both |")
    L.append("|---|---|---|---|")
    for b in BUDGETS:
        g = S[S.budget == b].groupby("seed")[["stage1", "stage2", "both"]].mean()
        L.append(f"| {b} | {g.stage1.mean():.3f} | {g.stage2.mean():.3f} | {g.both.mean():.3f} |")
    L.append("")
    L.append("Per held-out dataset: lowest alarms/day at sensitivity 0.90 (mean over seeds):\n")
    L.append("| held-out | stage 1 | stage 2 | both |")
    L.append("|---|---|---|---|")
    for d, g in F[F.target == 0.90].groupby("fold"):
        L.append(f"| {d} | {g.stage1.mean():.1f} | {g.stage2.mean():.1f} | {g.both.mean():.1f} |")
    L.append("")

    # where remaining alarms come from
    label_hours = []
    for p in held_out:
        u, c = np.unique(SOURCE.labels(p), return_counts=True)
        label_hours.append({k: n / FS / 3600 for k, n in zip(u, c)})
    hours = pd.DataFrame(label_hours).sum().to_dict()
    cnt = {}
    for (s, d), z in Z.items():
        fl = {k: z[f"fl_{k}"] for k in ("t", "s1", "p2", "lab", "pid")}
        a = fl["s1"] >= z["t1_val"]
        for key, score, thr in (("gate", fl["s1"], z["t1_val"]), ("two", np.where(a, fl["p2"], -np.inf), z["t2_val"])):
            ev = c24_alarms(fl, score, wearer_hours(z), label_hours).by_label(float(thr)).groupby("label").events.sum()
            for lbl, v in ev.items():
                cnt[(key, lbl)] = cnt.get((key, lbl), 0) + int(v)
    L.append("## 4. Where the remaining alarms come from (practical rule)\n")
    L.append(f"Events per hour of each Capture-24 activity, pooled over the held-out datasets and seeds (the {len(held_out)} "
             "held-out people are scored once per fold and seed; an alarm event belongs to the activity of the candidate that opened it).\n")
    L.append("| activity | hours | stage 1 + trigger | two-stage |")
    L.append("|---|---|---|---|")
    for lbl in sorted(hours, key=lambda k: -cnt.get(("two", k), 0)):
        L.append(f"| {lbl} | {hours[lbl]:.0f} | {cnt.get(('gate', lbl), 0) / (hours[lbl] * len(Z)):.2f} | {cnt.get(('two', lbl), 0) / (hours[lbl] * len(Z)):.2f} |")
    L.append("")
    # stage 2 standalone ranking quality
    aucs = []
    for (s, d), z in Z.items():
        ok = z["fall_trig"] & z["fall_ctx"]
        aucs.append(roc_auc_score(np.r_[np.ones(ok.sum()), np.zeros(len(z["fl_p2"]))], np.r_[z["fall_p2"][ok], z["fl_p2"]]))
    L.append(f"Stage 2 alone, held-out falls vs held-out Capture-24 candidates: AUROC {np.mean(aucs):.3f} (mean over folds and seeds).\n")
    (out / "report.md").write_text("\n".join(L))
    print(f"report: {out / 'report.md'}")
