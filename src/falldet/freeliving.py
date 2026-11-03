"""Free-living false-alarm rate on Capture-24 (wrist accelerometer worn ~24 h per person, labelled).

Replaces the assumed time budgets of ``alarms.py`` with measured exposure: every 4 s window (2 s
stride) of every participant is scored by the final models and alarm events are counted with the same
refractory merge, per participant and per labelled activity (sleep included).

Steps (each cached): ``prepare`` Capture-24 CSVs -> 25 Hz canonical-frame signal + activity labels;
``fit`` final augmented accel-only CNNs on all four fall datasets with the two threshold rules;
``score`` every window; ``report``.

Mounting: each participant is rotated so the mean specific force while walking points at the
right-wrist reference, then mirrored to the left-wrist convention (as for Erciyes/UMAFall). Capture-24
records the *dominant* wrist only, so the side is assumed right [UNVERIFIED]. Free-living walking
includes arm swing, so the walking mean has norm ~0.7 g (lab data: 0.9-1.0 g); only its direction is
used and participants below MIN_WALK_NORM_G are excluded.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

from . import model as M
from .alarms import REFRACTORY_S, count_events
from .augment import Augment
from .baseline import choose_threshold, split_validation
from .canonical import ACC_CLIP_G, FS, REFERENCE_HANGING_G, mirror_right_to_left, rot_align
from .cli import PROCESSED, RAW, REPORTS
from .threshold import balanced_threshold
from .windows import STRIDE_S, WIN_S

SRC_FS = 100
LABEL_COL = "label:WillettsSpecific2018"
LABELS = ["sleep", "sitting", "standing", "walking", "household-chores", "manual-work",
          "mixed-activity", "sports", "vehicle", "bicycling"]
UNLABELLED = 255
MIN_WALK_S, MIN_WALK_NORM_G = 300, 0.5
OUT = PROCESSED / "capture24"
RULES = ("val-slice", "lodo-pooled")


def _to_left(acc: np.ndarray) -> np.ndarray:
    """Accelerometer part of the right-to-left wrist mirror (``canonical.mirror_right_to_left``)."""
    return mirror_right_to_left(acc, np.zeros_like(acc))[0]

# --------------------------------------------------------------------------------- prepare
def _prepare_one(args) -> dict:
    path, mapping = args
    pid = path.name.split(".")[0]
    t = pd.read_csv(path, usecols=["x", "y", "z", "annotation"],
                    dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": "category"})
    cats = t.annotation.cat.categories
    code_of = np.array([LABELS.index(mapping[c]) if c in mapping and mapping[c] in LABELS else UNLABELLED
                        for c in cats], dtype=np.uint8)
    codes = t.annotation.cat.codes.to_numpy()
    lab = np.where(codes >= 0, code_of[np.maximum(codes, 0)], UNLABELLED).astype(np.uint8)
    xyz = t[["x", "y", "z"]].to_numpy(np.float64)
    del t
    info = {"pid": pid, "hours": len(xyz) / SRC_FS / 3600}
    walk = xyz[lab == LABELS.index("walking")]
    info["walk_s"] = len(walk) / SRC_FS
    if info["walk_s"] < MIN_WALK_S:
        return {**info, "excluded": f"walking {info['walk_s']:.0f} s"}
    mean = walk.mean(axis=0)
    info["walk_norm"] = float(np.linalg.norm(mean))
    if info["walk_norm"] < MIN_WALK_NORM_G:
        return {**info, "excluded": f"walking mean norm {info['walk_norm']:.2f} g"}
    sos = butter(4, 0.4 * FS, btype="low", fs=SRC_FS, output="sos")
    acc = sosfiltfilt(sos, xyz, axis=0)[:: SRC_FS // FS]
    rot = rot_align(mean, REFERENCE_HANGING_G * np.array([1.0, -1.0, 1.0]))
    acc = np.clip(_to_left(acc @ rot.T), -ACC_CLIP_G, ACC_CLIP_G).astype(np.float32)
    np.savez(OUT / f"{pid}.npz", acc=acc, lab=lab[:: SRC_FS // FS][: len(acc)])
    wa = _to_left(sosfiltfilt(sos, walk, axis=0)[:: SRC_FS // FS] @ rot.T)
    info["walk_axis_std"] = wa.std(axis=0).round(3).tolist()
    return info


def prepare(workers: int = 3) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "prepare_log.csv"
    if log.exists():
        return pd.read_csv(log)
    base = RAW / "capture24" / "capture24"
    if not base.is_dir():
        raise FileNotFoundError(base)
    d = pd.read_csv(base / "annotation-label-dictionary.csv", index_col=0, dtype=str)
    mapping = d[LABEL_COL].to_dict()
    files = sorted(base.glob("P[0-9][0-9][0-9].csv.gz"))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = []
        for i, r in enumerate(ex.map(_prepare_one, [(f, mapping) for f in files])):
            rows.append(r)
            if i % 10 == 0:
                print(f"[prepare] {i + 1}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(log, index=False)
    return df


# -------------------------------------------------------------------------------------- fit
def fit_models(seeds: int, root=OUT, extra=None) -> pd.DataFrame:
    """Final models (all four fall datasets) and thresholds per seed; weights saved to ``root``/models.

    ``extra(seed)`` optionally returns (training, validation) negative windows (model inputs) that are
    added to every fit, final and nested (see ``negatives.c24_negatives``)."""
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "fall_scores").mkdir(exist_ok=True)
    tp = root / "thresholds.csv"
    if tp.exists() and pd.read_csv(tp).seed.nunique() >= seeds:
        return pd.read_csv(tp)
    X = np.load(PROCESSED / "X.npy")
    meta = pd.read_csv(PROCESSED / "meta.csv")
    y = meta.label.to_numpy().astype(np.int64)
    x = M.make_inputs(X, "acc")
    aug = Augment()
    datasets = sorted(meta.dataset.unique())
    ones = lambda n: np.ones(n, np.float32)  # noqa: E731
    rows, t0 = [], time.time()
    for seed in range(seeds):
        xn, xnv = extra(seed) if extra else (x[:0], x[:0])

        def fit(tr_idx, va_idx):
            xt = np.concatenate([x[tr_idx], xn])
            yt = np.concatenate([y[tr_idx], np.zeros(len(xn), np.int64)])
            xv = np.concatenate([x[va_idx], xnv])
            yv = np.concatenate([y[va_idx], np.zeros(len(xnv), np.int64)])
            return M.fit(xt, yt, ones(len(xt)), xv, yv, seed, augment=aug)

        tr, va = split_validation(meta, np.arange(len(meta)), seed)
        net = fit(tr, va)
        thr = {"val-slice": choose_threshold(M.predict_proba(net, x[va]), y[va])}
        torch.save(net.state_dict(), root / "models" / f"seed{seed}.pt")
        del net
        falls, fall_scores = [], {}
        for s in datasets:  # nested leave-one-dataset-out, as in threshold.py
            s_idx = np.where(meta.dataset == s)[0]
            itr, iva = split_validation(meta, np.where(meta.dataset != s)[0], seed)
            inet = fit(itr, iva)
            fs_ = M.predict_proba(inet, x[s_idx])[y[s_idx] == 1]
            falls.append(fs_)
            fall_scores[s] = fs_
            del inet
            torch.cuda.empty_cache()
        thr["lodo-pooled"] = balanced_threshold(falls)
        np.savez(root / "fall_scores" / f"seed{seed}.npz", **fall_scores)
        rows += [{"seed": seed, "rule": r, "threshold": t} for r, t in thr.items()]
        print(f"[fit] seed {seed} done ({time.time() - t0:.0f}s) {thr}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(tp, index=False)
    return df


# ------------------------------------------------------------------------------------ score
@torch.no_grad()
def score_signal(net: M.FallCNN, acc: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """Probability for every 4 s window (2 s stride) of one participant's signal."""
    dev = M.device()
    win, stride = round(WIN_S * FS), round(STRIDE_S * FS)
    a = torch.from_numpy(acc).unfold(0, win, stride)  # (nw, 3, win), a view
    out = []
    for i in range(0, len(a), chunk):
        w = a[i : i + chunk].to(dev)
        mag = w.norm(dim=1, keepdim=True)
        inp = torch.cat([w / M.ACC_SCALE, mag / M.ACC_SCALE], dim=1)
        out.append(torch.sigmoid(net(inp)).cpu())
    return torch.cat(out).numpy()


def score_all(seeds: int, pids: list[str], root=OUT) -> None:
    sp = root / "scores"
    sp.mkdir(exist_ok=True)
    nets = []
    for s in range(seeds):
        n = M.FallCNN(4).to(M.device())
        n.load_state_dict(torch.load(root / "models" / f"seed{s}.pt"))
        nets.append(n.eval())
    t0 = time.time()
    for i, pid in enumerate(pids):
        f = sp / f"{pid}.npy"
        if f.exists() and np.load(f, mmap_mode="r").shape[0] >= seeds:
            continue
        acc = np.load(OUT / f"{pid}.npz")["acc"]
        np.save(f, np.stack([score_signal(n, acc) for n in nets]).astype(np.float32))
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
    nets = []
    for s in range(seeds):
        n = M.FallCNN(4).to(M.device())
        n.load_state_dict(torch.load(root / "models" / f"seed{s}.pt"))
        nets.append(n.eval())
    win, stride = round(WIN_S * FS), round(STRIDE_S * FS)
    sub = pids[:: max(1, len(pids) // 40)]  # ~40 participants keep this affordable
    rows = []
    for pid in sub:
        acc0 = np.load(OUT / f"{pid}.npz")["acc"]
        for name, (deg, unmirror) in MOUNTINGS.items():
            a = acc0 * np.array([1.0, -1.0, 1.0], np.float32) if unmirror else acc0
            a = np.ascontiguousarray(a @ _roll(deg).T)
            for seed in range(seeds):
                sc = score_signal(nets[seed], a)
                end_s = (np.arange(len(sc)) * stride + win) / FS
                for rule in RULES:
                    t = thr[(thr.seed == seed) & (thr.rule == rule)].threshold.iloc[0]
                    rows.append({"mounting": name, "rule": rule, "seed": seed, "pid": pid,
                                 "hours": len(sc) * STRIDE_S / 3600,
                                 "events": count_events(end_s[sc >= t], refractory)})
    return pd.DataFrame(rows)


THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def tradeoff(pids: list[str], seeds: int, refractory: float, root=OUT) -> pd.DataFrame:
    """Alarm events/24 h on Capture-24 (final model) against sensitivity on out-of-dataset fall windows
    (nested models trained on three of the four datasets, equal weight per dataset) at the same threshold.
    The two models differ (3 vs 4 training datasets), so this is an approximation."""
    win, stride = round(WIN_S * FS), round(STRIDE_S * FS)
    rows = []
    scores = {pid: np.load(root / "scores" / f"{pid}.npy") for pid in pids}
    for seed in range(seeds):
        fs = np.load(root / "fall_scores" / f"seed{seed}.npz")
        for t in THRESHOLDS:
            ev = hrs = 0
            for pid in pids:
                sc = scores[pid][seed]
                end_s = (np.arange(len(sc)) * stride + win) / FS
                ev += count_events(end_s[sc >= t], refractory)
                hrs += len(sc) * STRIDE_S / 3600
            sens = np.mean([(fs[k] >= t).mean() for k in fs.files])
            rows.append({"seed": seed, "threshold": t, "events_per_24h": ev / hrs * 24, "fall_window_sensitivity": sens})
    return pd.DataFrame(rows)


def alarm_table(pids: list[str], thr: pd.DataFrame, seeds: int, refractory: float, root=OUT) -> pd.DataFrame:
    """One row per (seed, rule, participant, label): hours of exposure and alarm events."""
    rows = []
    win, stride = round(WIN_S * FS), round(STRIDE_S * FS)
    for pid in pids:
        sc = np.load(root / "scores" / f"{pid}.npy")
        lab = np.load(OUT / f"{pid}.npz")["lab"]
        centre = lab[np.arange(sc.shape[1]) * stride + win // 2]
        end_s = (np.arange(sc.shape[1]) * stride + win) / FS
        for seed in range(seeds):
            for rule in RULES:
                t = thr[(thr.seed == seed) & (thr.rule == rule)].threshold.iloc[0]
                fired = sc[seed] >= t
                for code in np.unique(centre):
                    m = centre == code
                    name = LABELS[code] if code != UNLABELLED else "unlabelled"
                    ev = count_events(end_s[fired & m], refractory)
                    rows.append({"seed": seed, "rule": rule, "pid": pid, "label": name,
                                 "hours": m.sum() * STRIDE_S / 3600, "events": ev,
                                 "windows": int((fired & m).sum())})
    return pd.DataFrame(rows)


def run(seeds: int = 5, negatives: bool = False) -> None:
    """Free-living report. ``negatives``: final models also train on Capture-24 negatives (people
    P001-P100), so only the held-out people P101-P151 are scored."""
    t0 = time.time()
    prep = prepare()
    ok = prep[prep.excluded.isna()] if "excluded" in prep else prep
    pids, root, extra, name = list(ok.pid), OUT, None, "freeliving"
    if negatives:
        from .negatives import c24_negatives, test_pids

        pids, root, extra, name = test_pids(prep), OUT / "c24random", c24_negatives, "freeliving_c24random"
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
    write_report(prep, tab, thr, out, seeds, time.time() - t0, ms, tr, negatives)
    print(f"report: {out / 'report.md'}")


def _wilson_ci(x: np.ndarray, hours: np.ndarray, rng: np.random.Generator, n: int = 2000):
    """Bootstrap over participants of the pooled rate (events per hour)."""
    idx = rng.integers(0, len(x), (n, len(x)))
    r = x[idx].sum(1) / hours[idx].sum(1)
    return np.percentile(r, [2.5, 97.5])


def write_report(prep, tab, thr, out, seeds, seconds, ms, tr, negatives=False) -> None:
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
             "AX3 samples at 100 Hz and was low-passed and decimated to 25 Hz by the same code path as the real "
             "datasets. (5) 151 people × one day: day-to-day variation within a person is not measured.\n")
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
            lo_hi.append(_wilson_ci(gg.events.to_numpy(float), gg.hours.to_numpy(float), rng) * 24)
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
    L.append("## 6. Comparison with the assumed-budget estimate (`alarms.py`)\n")
    L.append("`reports/alarms/report.md` estimated 26 (val-slice) and 155 (lodo-pooled) alarms for a typical day, "
             "and 55 / 210 for an active day, from lab clips and assumed time budgets. The measured rates above "
             "replace that assumption.\n")
    L.append(f"Weights: threshold levels per seed are in `data/processed/capture24/thresholds.csv`. "
             f"Runtime of the last report step: {seconds / 60:.1f} min (fits and scoring are cached).\n")
    (out / "report.md").write_text("\n".join(L))
