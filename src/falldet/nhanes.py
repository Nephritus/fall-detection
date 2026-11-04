"""NHANES 2013-14 wrist accelerometry (PAX80_H) as a second free-living population: older adults, another
device, about eight days each. Measures false alarms of the current design (stage-1 CNN with Capture-24
negatives, then the stillness check) on people, a device and an age group the models never saw.

* data: ActiGraph GT3X+ at 80 Hz, ±6 g, worn on the non-dominant wrist, dorsal, around the clock for up to
  eight days; one ``<SEQN>.tar.bz2`` per participant on the CDC FTP server (PAX80_H documentation:
  wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAX80_H.htm). Public, NCHS data user agreement.
* selection: a seeded random sample of participants aged MIN_AGE+ with valid data (PAXSTS = 1), worn on the
  non-dominant hand (PAXHAND = 1) in the dorsal orientation (PAXORENT = 1). The wrist side is not coded; the
  non-dominant wrist is taken as the left one [ASSUMPTION: right-handed majority], so no mirroring.
* mounting: there are no activity labels, so walking is found from the signal (``gait_epochs``: a periodic
  |a| at a step rhythm). Every participant wore the same device the same way, so one device rotation is
  estimated from everyone pooled: their mean walking direction is turned onto the hanging-arm reference, and
  the roll about it is chosen so that the pooled resting orientations (still 10 s epochs) best match those
  of the Capture-24 training people. Each person is then refined by aligning their own walking direction
  (a small rotation, so well defined). Checked on 16 held-out Capture-24 people seen through random device
  rotations: the pooled roll came within about 16°, and alarms per day were 28.6 against 29.7 with the
  labelled alignment (seed 0, 2 rotations).
* non-wear: 60-min windows every 15 min, non-wear when at least two axes have a standard deviation below
  13 mg or a range below 50 mg (the GGIR rule, van Hees et al. 2013, doi:10.1371/journal.pone.0061691).
  Exposure counts only worn windows; a watch knows when it is off the wrist.
"""

from __future__ import annotations

import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import stillness
from .alarms import REFRACTORY_S, FalseAlarms, _merge, per_day_ci
from .canonical import ACC_CLIP_G, FS, REFERENCE_HANGING_G, rot_align, to_grid
from .freeliving import SOURCE as CAPTURE24
from .model import FallModel
from .paths import PROCESSED, RAW, REPORTS
from .runner import Results
from .windows import STRIDE_S, WIN_S

SRC_FS = 80
URL = "https://ftp.cdc.gov/pub/pax_h/{}.tar.bz2"
TABLES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/{}.xpt"
RAW_DIR, OUT, RUNS = RAW / "nhanes", PROCESSED / "nhanes", REPORTS / "nhanes" / "runs"
MIN_AGE, N_PEOPLE = 65, 100
GAIT_EPOCH_S, MIN_GAIT_S, MIN_GAIT_NORM_G = 10, 300, 0.5
STILL_MAX_STD_G, STILL_SAMPLE, REFERENCE_SAMPLE, ROLL_STEP_DEG, KAPPA = 0.05, 300, 4000, 2, 20.0
MAX_REFINE_DEG = 80.0  # reported: people whose own walking direction is this far from the pooled one
NONWEAR_WINDOW_S, NONWEAR_STEP_S, NONWEAR_SD_G, NONWEAR_RANGE_G = 3600, 900, 0.013, 0.050
THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)  # as freeliving.tradeoff
WIN, STRIDE = round(WIN_S * FS), round(STRIDE_S * FS)


# ---------------------------------------------------------------------------------- signal tools
def resample(x: np.ndarray, fs_src: float = SRC_FS) -> np.ndarray:
    """A long uniform recording -> the 25 Hz grid, one hour at a time with 10 s of overlap (same filter and
    interpolation as ``canonical.to_grid``, without holding a week of float64 in memory)."""
    n = len(x)
    n_out = int((n - 1) / fs_src * FS) + 1
    out = np.empty((n_out, x.shape[1]), np.float32)
    blk, pad = int(3600 * fs_src), int(10 * fs_src)
    for a in range(0, n, blk):
        lo, hi = max(0, a - pad), min(n, a + blk + pad)
        g0 = int(np.ceil(a / fs_src * FS))
        g1 = n_out if a + blk >= n else int(np.ceil((a + blk) / fs_src * FS))
        out[g0:g1] = to_grid(np.arange(lo, hi) / fs_src, x[lo:hi].astype(np.float64), fs_src, np.arange(g0, g1) / FS)
    return out


def gait_epochs(acc: np.ndarray, min_autocorr: float = 0.4, std_g: tuple[float, float] = (0.08, 0.6)) -> np.ndarray:
    """Walking-like 10 s epochs of a 25 Hz signal (rotation invariant): |a| varies (std within ``std_g``) and
    repeats at a step period of 0.4-1.0 s (normalised autocorrelation peak >= ``min_autocorr``)."""
    ep = GAIT_EPOCH_S * FS
    n = len(acc) // ep
    m = np.linalg.norm(acc[: n * ep], axis=1).reshape(n, ep)
    m = m - m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1)
    f = np.fft.rfft(m, 2 * ep, axis=1)
    ac = np.fft.irfft(f * f.conj(), axis=1)[:, :ep]
    ac = ac / np.maximum(ac[:, :1], 1e-12)
    lo, hi = int(0.4 * FS), int(1.0 * FS)
    k = lo + ac[:, lo: hi + 1].argmax(axis=1)  # a real peak inside the step range, not the edge of a slow sway
    peak = ac[np.arange(n), k]
    return (peak >= min_autocorr) & (k > lo) & (k < hi) & (sd >= std_g[0]) & (sd <= std_g[1])


def wear_seconds(acc: np.ndarray) -> np.ndarray:
    """Per second of a 25 Hz signal in device axes: worn (True) or not, by the GGIR rule (module docstring).
    A second is non-wear when any window covering it is."""
    n_s = len(acc) // FS
    w, step = NONWEAR_WINDOW_S, NONWEAR_STEP_S
    worn = np.ones(n_s, bool)
    x = acc[: n_s * FS].reshape(n_s, FS, 3)
    s1, s2 = x.sum(axis=1, dtype=np.float64), (x.astype(np.float64) ** 2).sum(axis=1)
    mx, mn = x.max(axis=1), x.min(axis=1)
    c1 = np.concatenate([np.zeros((1, 3)), np.cumsum(s1, axis=0)])
    c2 = np.concatenate([np.zeros((1, 3)), np.cumsum(s2, axis=0)])
    for a in range(0, max(n_s - w, 0) + 1, step):
        b = min(a + w, n_s)
        k = (b - a) * FS
        mean = (c1[b] - c1[a]) / k
        sd = np.sqrt(np.maximum((c2[b] - c2[a]) / k - mean**2, 0))
        rng = mx[a:b].max(axis=0) - mn[a:b].min(axis=0)
        if (sd < NONWEAR_SD_G).sum() >= 2 or (rng < NONWEAR_RANGE_G).sum() >= 2:
            worn[a:b] = False
    return worn


def worn_windows(wear: np.ndarray, n_windows: int) -> np.ndarray:
    """A window (4 s, 2 s stride) is worn when every second it covers is."""
    start = np.arange(n_windows) * int(STRIDE_S)
    c = np.concatenate([[0], np.cumsum(~wear)])
    end = np.minimum(start + int(WIN_S), len(wear))
    return (c[end] - c[np.minimum(start, len(wear))] == 0) & (start + int(WIN_S) <= len(wear))


# ---------------------------------------------------------------------------------------- source
def read_archive(path: Path) -> tuple[np.ndarray, float]:
    """One participant: the 80 Hz signal (n, 3) in g, device axes, and the clock time of its first sample (s)."""
    parts, start = [], None
    with tarfile.open(path, "r:bz2") as tf:
        members = sorted((m for m in tf if m.name.endswith(".sensor.csv")), key=lambda m: m.name)
        for m in members:
            d = pd.read_csv(tf.extractfile(m), usecols=["HEADER_TIMESTAMP", "X", "Y", "Z"], dtype={"X": "f4", "Y": "f4", "Z": "f4"})
            if start is None:
                t = pd.Timestamp(d.HEADER_TIMESTAMP.iloc[0])
                start = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            parts.append(d[["X", "Y", "Z"]].to_numpy())
    return np.concatenate(parts), float(start)


def still_directions(acc: np.ndarray, rng: np.random.Generator, k: int = STILL_SAMPLE,
                     worn: np.ndarray | None = None) -> np.ndarray:
    """Up to ``k`` unit gravity directions of still 10 s epochs (vector std < STILL_MAX_STD_G); ``worn``
    (per 10 s epoch) leaves out epochs off the wrist."""
    ep = GAIT_EPOCH_S * FS
    n = len(acc) // ep
    x = acc[: n * ep].reshape(n, ep, 3).astype(np.float64)
    m, sd = x.mean(axis=1), np.sqrt(x.var(axis=1).sum(axis=1))
    keep = sd < STILL_MAX_STD_G
    if worn is not None:
        keep &= worn[:n]
    u = m[keep]
    norm = np.linalg.norm(u, axis=1)
    u = u[norm > 0.5] / norm[norm > 0.5, None]  # a still wrist reads about 1 g; near 0 is a device fault
    return u[rng.choice(len(u), min(k, len(u)), replace=False)] if len(u) else u


def roll_about(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotation by ``deg`` about the unit ``axis`` (Rodrigues)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def device_rotation(walk_means: np.ndarray, stills: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """One rotation from device axes to the wrist frame for people who all wore the device the same way: the
    pooled walking direction goes onto the hanging-arm reference, and the roll about it maximises the likelihood
    of the pooled resting directions under a von Mises-Fisher kernel density of the ``reference`` directions."""
    r1 = rot_align(walk_means.mean(axis=0), np.asarray(REFERENCE_HANGING_G, float))
    axis = np.asarray(REFERENCE_HANGING_G, float)
    u = stills @ r1.T
    degs = np.arange(0, 360, ROLL_STEP_DEG)
    score = [np.log(np.exp(KAPPA * ((u @ roll_about(axis, d).T) @ reference.T - 1)).mean(axis=1) + 1e-12).mean() for d in degs]
    best = float(degs[int(np.argmax(score))])
    return roll_about(axis, best) @ r1, best


def reference_still_directions(root: Path = OUT) -> np.ndarray:
    """Resting directions of the Capture-24 training people (seed-0 split, every second person), cached."""
    f = Path(root) / "reference_still.npy"
    if f.exists():
        return np.load(f)
    rng = np.random.default_rng(0)
    u = np.concatenate([still_directions(CAPTURE24.signal(p), rng, 10_000) for p in CAPTURE24.split(0).train[::2]])
    u = u[rng.choice(len(u), REFERENCE_SAMPLE, replace=False)]
    np.save(f, u)
    return u


def _prepare_one(args) -> dict:
    """Device-frame 25 Hz signal, wear, and the person's walking direction and resting directions."""
    seqn, archive, root = args
    raw, start = read_archive(archive)
    acc = resample(raw)
    del raw
    wear = wear_seconds(acc)
    info = {"seqn": seqn, "hours": len(acc) / FS / 3600, "wear_hours": wear.sum() / 3600, "start_s": start}
    ep = GAIT_EPOCH_S * FS
    g = gait_epochs(acc)
    worn_ep = wear[: len(g) * GAIT_EPOCH_S].reshape(len(g), GAIT_EPOCH_S).all(axis=1)
    g &= worn_ep
    info["gait_s"] = int(g.sum()) * GAIT_EPOCH_S
    walk = acc[: len(g) * ep].reshape(len(g), ep, 3)[g].reshape(-1, 3).astype(np.float64).mean(axis=0) if g.any() else np.zeros(3)
    info["gait_norm"] = float(np.linalg.norm(walk))
    np.savez(Path(root) / f"{seqn}.npz", acc=acc.astype(np.float16), wear=wear, start_s=start, walk=walk,
             still=still_directions(acc, np.random.default_rng(seqn), worn=worn_ep))
    if info["gait_s"] < MIN_GAIT_S:
        info["excluded"] = f"walking-like signal {info['gait_s']} s"
    elif info["gait_norm"] < MIN_GAIT_NORM_G:
        info["excluded"] = f"walking mean norm {info['gait_norm']:.2f} g"
    return info


class Nhanes:
    """Prepared NHANES participants under ``root``: ``<SEQN>.npz`` (25 Hz signal in device axes as float16,
    per-second wear, clock time of the first sample, walking and resting directions), ``rotations.npz``
    (device axes -> wrist frame per person) and ``prepare_log.csv`` with age, sex and exclusions."""

    def __init__(self, root: Path = OUT) -> None:
        self.root = Path(root)

    def log(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "prepare_log.csv")

    def people(self) -> list[int]:
        log = self.log()
        return sorted(log[log.excluded.isna()].seqn) if "excluded" in log else sorted(log.seqn)

    def _z(self, seqn: int):
        return np.load(self.root / f"{seqn}.npz")

    def rotation(self, seqn: int) -> np.ndarray:
        return np.load(self.root / "rotations.npz")[str(seqn)]

    def signal(self, seqn: int) -> np.ndarray:
        """Canonical accelerometer signal (n, 3) in g at 25 Hz: rotated into the wrist frame and clipped."""
        acc = self._z(seqn)["acc"].astype(np.float32) @ self.rotation(seqn).T.astype(np.float32)
        return np.clip(acc, -ACC_CLIP_G, ACC_CLIP_G)

    def wear(self, seqn: int) -> np.ndarray:
        return self._z(seqn)["wear"]

    def start_s(self, seqn: int) -> float:
        return float(self._z(seqn)["start_s"])

    def prepare(self, raw_dir: Path = RAW_DIR, n: int = N_PEOPLE, workers: int = 3) -> pd.DataFrame:
        """Select, download (resumable) and convert ``n`` participants; a no-op when they exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        sel = select(raw_dir, n)
        logf = self.root / "prepare_log.csv"
        if logf.exists():
            done = self.log()
            if set(done.seqn) >= set(sel.SEQN):
                return done
        download(raw_dir, list(sel.SEQN))
        t0, rows = time.time(), []
        jobs = [(int(s), Path(raw_dir) / "pax_h" / f"{int(s)}.tar.bz2", self.root) for s in sel.SEQN]
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(_prepare_one, jobs)):
                rows.append(r)
                if i % 10 == 0:
                    print(f"[nhanes] prepared {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows).merge(sel.rename(columns={"SEQN": "seqn"}), on="seqn")
        df = self._align(df)
        df.to_csv(logf, index=False)
        return df

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pooled device rotation from the usable people, then each person's walking direction refines it."""
        ok = df[df.excluded.isna()] if "excluded" in df else df
        z = {s: self._z(s) for s in ok.seqn}
        walks = np.array([z[s]["walk"] / np.linalg.norm(z[s]["walk"]) for s in ok.seqn])
        stills = np.concatenate([z[s]["still"] for s in ok.seqn])
        r0, roll = device_rotation(walks, stills, reference_still_directions(self.root))
        rot, refine = {}, {}
        for s in ok.seqn:
            r = rot_align(r0 @ z[s]["walk"], np.asarray(REFERENCE_HANGING_G, float))
            rot[str(s)] = r @ r0
            refine[s] = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        np.savez(self.root / "rotations.npz", **rot)
        print(f"[nhanes] device rotation: roll {roll:.0f}° about the hanging axis; per-person refinement "
              f"{np.median(list(refine.values())):.0f}° median", flush=True)
        return df.assign(device_roll_deg=roll, refine_deg=df.seqn.map(refine))


def select(raw_dir: Path, n: int, seed: int = 0) -> pd.DataFrame:
    """Seeded random sample of eligible participants (module docstring); writes ``selection.csv``."""
    raw_dir = Path(raw_dir)
    f = raw_dir / "selection.csv"
    if f.exists() and len(pd.read_csv(f)) == n:
        return pd.read_csv(f)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for t in ("DEMO_H", "PAXHD_H"):
        if not (raw_dir / f"{t}.xpt").exists():
            subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(raw_dir / f"{t}.xpt"), TABLES.format(t)], check=True)
    demo = pd.read_sas(raw_dir / "DEMO_H.xpt", encoding="latin-1")
    hd = pd.read_sas(raw_dir / "PAXHD_H.xpt", encoding="latin-1")
    m = hd.merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR"]], on="SEQN")
    ok = m[(m.PAXSTS == 1) & (m.PAXHAND == "1") & (m.PAXORENT == "1") & (m.RIDAGEYR >= MIN_AGE)]
    sel = ok.sample(n, random_state=seed).sort_values("SEQN")
    sel = pd.DataFrame({"SEQN": sel.SEQN.astype(int), "age": sel.RIDAGEYR.astype(int),
                        "sex": sel.RIAGENDR.map({1.0: "male", 2.0: "female"})})
    sel.to_csv(f, index=False)
    return sel


def download(raw_dir: Path, seqns: list[int], workers: int = 4) -> None:
    """Participant archives from the CDC FTP server into ``raw_dir``/pax_h (skips complete files)."""
    d = Path(raw_dir) / "pax_h"
    d.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqns if not (d / f"{s}.tar.bz2").exists()]

    def one(s):
        part = d / f"{s}.tar.bz2.part"
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(part), URL.format(s)], check=True)
        part.rename(d / f"{s}.tar.bz2")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


# ------------------------------------------------------------------------------------ experiment
def _models(seeds: int):
    c24 = stillness.C24
    thr = pd.read_csv(c24 / "thresholds.csv")
    thr = thr[thr.rule == "val-slice"].set_index("seed").threshold
    return [FallModel.load(c24 / "models" / f"seed{s}.pt", "acc") for s in range(seeds)], thr


def run(seeds: int = 3, n: int = N_PEOPLE) -> None:
    """Score every prepared participant with the final models (Capture-24 negatives, validation-slice
    threshold); store per (seed, participant) the detections, the motion after them and an alarm curve."""
    src = Nhanes()
    src.prepare(RAW_DIR, n)
    results, t0 = Results(RUNS), time.time()
    todo = [(s, p) for p in src.people() for s in range(seeds) if not results.has("stage1", s, str(p))]
    if todo:
        nets, thr = _models(seeds)
    for i, (seed, p) in enumerate(todo):
        acc, wear, start = src.signal(p), src.wear(p), src.start_s(p)
        sc = nets[seed].score_signal(acc)
        worn = worn_windows(wear, len(sc))
        hours_worn = worn.sum() * STRIDE_S / 3600
        t_end = np.arange(len(sc)) * STRIDE_S + WIN_S
        fa_worn = FalseAlarms([t_end[worn]], [sc[worn]], [hours_worn])
        fa_all = FalseAlarms.from_window_scores([sc])
        t = float(thr[seed])
        det = fa_worn.detections(t)[0]
        arrays = {"det_t": det, "det_motion": stillness.post_motion(acc, det, stillness.HORIZON_S),
                  "det_clock_h": ((start + det) / 3600) % 24,
                  "curve_thresholds": np.array(THRESHOLDS), "curve_events": np.array([fa_worn.events(x)[0] for x in THRESHOLDS])}
        row = {"threshold": t, "hours": len(sc) * STRIDE_S / 3600, "hours_worn": hours_worn,
               "events_worn": int(fa_worn.events(t)[0]), "events_all_time": int(fa_all.events(t)[0])}
        results.save("stage1", seed, str(p), [row], arrays)
        if i % 25 == 0:
            print(f"[nhanes] scored {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    report()


def _per_day(arrays: list[dict], hours: list[float], wait: int, level: float = stillness.STILL_G) -> np.ndarray:
    """Alarm events per day of wear for each participant after a stillness check of ``wait`` seconds."""
    out = []
    for a, h in zip(arrays, hours):
        ok = stillness.stays_still(np.nan_to_num(a["det_motion"], nan=0.0), wait, level)
        fa = FalseAlarms([a["det_t"][ok]], [np.ones(ok.sum())], [h])
        out.append(fa.per_day(0.5))
    return np.array(out)


def report() -> None:
    out = REPORTS / "nhanes"
    src = Nhanes()
    log = src.log()
    results = Results(RUNS)
    tab = results.table()
    tab["seqn"] = tab.fold.astype(int)
    seeds = sorted(tab.seed.unique())
    people = sorted(tab.seqn.unique())
    A = {(s, p): results.arrays("stage1", s, str(p)) for s, p in zip(tab.seed, tab.seqn)}
    H = tab.set_index(["seed", "seqn"]).hours_worn
    used = log[log.seqn.isin(people)]

    # Capture-24 reference: same models and rule, held-out people, from the stillness store
    st = Results(stillness.RUNS)
    c24 = {w: np.mean([stillness._alarms(st.arrays("stage1", s, "all"), w).per_day(0.5) for s in seeds]) for w in stillness.WAITS}

    def rates(wait, level=stillness.STILL_G, subset=None):
        """Pooled alarms/day per seed and per-person rates (mean over seeds)."""
        ps = [p for p in people if subset is None or p in subset]
        pooled, per = [], []
        for s in seeds:
            r = _per_day([A[(s, p)] for p in ps], [H[(s, p)] for p in ps], wait, level)
            h = np.array([H[(s, p)] for p in ps])
            pooled.append((r * h / 24).sum() / h.sum() * 24)
            per.append(r)
        return np.array(pooled), np.mean(per, axis=0)

    L = ["# False alarms in older adults (NHANES 2013-14)\n"]
    excl = log[log.excluded.notna()]
    L.append(f"{len(used)} of {len(log)} sampled NHANES participants aged {MIN_AGE}+ (age {used.age.median():.0f} median, "
             f"{(used.age >= 80).sum()} aged 80+, {(used.sex == 'female').sum()} women), ActiGraph GT3X+ on the non-dominant wrist "
             f"(taken as the left), {used.wear_hours.sum():,.0f} h worn of {used.hours.sum():,.0f} h recorded "
             f"({used.wear_hours.sum() / len(used) / (used.hours.sum() / len(used)) * 24:.1f} h worn per 24 h). "
             f"Models: the final stage-1 CNNs with Capture-24 negatives (`negatives --final`, {len(seeds)} seeds) at their "
             "validation-slice thresholds; nobody here was trained on, and neither the device nor the age group was. Alarm events merge "
             f"with a {REFRACTORY_S:.0f} s refractory period; only worn windows count, as exposure and as detections. Mean ± std over seeds.\n")
    if len(excl):
        L.append("Excluded at preparation: " + "; ".join(f"{r.seqn}: {r.excluded}" for r in excl.itertuples()) + ".\n")

    L.append("## 1. Alarms per day of wear, against Capture-24\n")
    L.append(f"Stillness check `strict`, {stillness.STILL_G:g} g (as `reports/stillness`). Capture-24 = the 48 held-out people, same models.\n")
    L.append("| wait after decision | NHANES 65+ | per week | Capture-24 | NHANES / Capture-24 |")
    L.append("|---|---|---|---|---|")
    for w in stillness.WAITS:
        p, _ = rates(w)
        L.append(f"| {w} s | {p.mean():.2f} ± {p.std():.2f} | {p.mean() * 7:.1f} | {c24[w]:.2f} | {p.mean() / c24[w]:.2f}× |")
    L.append("")
    allt = tab.groupby("seed").apply(lambda d: d.events_all_time.sum() / d.hours.sum() * 24, include_groups=False)
    L.append(f"Without removing non-wear time (exposure = all recorded time, detections anywhere): {allt.mean():.1f} alarms/day before any check.\n")
    sure = set(used[used.refine_deg <= MAX_REFINE_DEG].seqn)
    cells = [f"{rates(w, subset=sure)[0].mean():.2f}" for w in (0, 10, 60)]
    L.append(f"Mounting check: {len(people) - len(sure)} people needed more than {MAX_REFINE_DEG:.0f}° to bring their own walking direction onto the "
             f"pooled one (an unusual or flipped strap position, or a right wrist; neither explains them fully). Without them: "
             f"{cells[0]} / {cells[1]} / {cells[2]} alarms/day with no check / 10 s / 60 s.\n")

    L.append("## 2. Per person\n")
    L.append("Each person's rate averaged over seeds; people wore the monitor for about a week. 95% interval: bootstrap over people "
             "of the seed-averaged rates.\n")
    L.append("| wait | pooled (95% CI) | median person | 90th percentile | worst person | people with no alarm in the week |")
    L.append("|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for w in (0, 10, 60):
        p, per = rates(w)
        ev = per * np.array([H[(seeds[0], q)] for q in people]) / 24
        lo, hi = per_day_ci(ev, np.array([H[(seeds[0], q)] for q in people]), rng)
        none = np.mean([(_per_day([A[(s, q)] for q in people], [H[(s, q)] for q in people], w) == 0).mean() for s in seeds])
        L.append(f"| {w} s | {p.mean():.2f} ({lo:.2f}–{hi:.2f}) | {np.median(per):.2f} | {np.quantile(per, 0.9):.2f} | {per.max():.1f} | {none:.0%} |")
    L.append("")

    L.append("## 3. By age\n")
    L.append("| age | people | no check | 10 s | 60 s |")
    L.append("|---|---|---|---|---|")
    for lo_, hi_, name in ((65, 69, "65–69"), (70, 74, "70–74"), (75, 79, "75–79"), (80, 200, "80 (top-coded 80+)")):
        sub = set(used[(used.age >= lo_) & (used.age <= hi_)].seqn)
        if not sub:
            continue
        cells = [f"{rates(w, subset=sub)[0].mean():.2f}" for w in (0, 10, 60)]
        L.append(f"| {name} | {len(sub)} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 4. By time of day\n")
    L.append("Share of alarm events by clock hour of the detection that opened them (all seeds), and alarms per day from each part of the day.\n")
    L.append("| part of day | no check | 10 s | 60 s |")
    L.append("|---|---|---|---|")
    parts = {"night 23–07": lambda h: (h >= 23) | (h < 7), "morning 07–12": lambda h: (h >= 7) & (h < 12),
             "afternoon 12–18": lambda h: (h >= 12) & (h < 18), "evening 18–23": lambda h: (h >= 18) & (h < 23)}
    tot_h = sum(H[(s, p)] for s in seeds for p in people)
    for name, f in parts.items():
        cells = []
        for w in (0, 10, 60):
            n_ev = 0
            for s in seeds:
                for p in people:
                    a = A[(s, p)]
                    ok = stillness.stays_still(np.nan_to_num(a["det_motion"], nan=0.0), w)
                    t = a["det_t"][ok]
                    if not len(t):
                        continue
                    opened = _merge(t, REFRACTORY_S)
                    n_ev += int(f(a["det_clock_h"][ok][opened]).sum())
            cells.append(f"{n_ev / tot_h * 24:.2f}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## 5. Threshold trade-off (no check)\n")
    L.append("Alarm events per day of wear at fixed thresholds, against Capture-24 (`reports/freeliving_c24random/tradeoff.csv`, same models).\n")
    tr = pd.read_csv(REPORTS / "freeliving_c24random" / "tradeoff.csv")
    L.append("| threshold | NHANES 65+ | Capture-24 |")
    L.append("|---|---|---|")
    for j, x in enumerate(THRESHOLDS):
        v = np.mean([sum(A[(s, p)]["curve_events"][j] for p in people) / sum(H[(s, p)] for p in people) * 24 for s in seeds])
        c = tr[(tr.threshold == x) & tr.seed.isin(seeds)].events_per_24h.mean()
        L.append(f"| {x:g} | {v:.1f} | {c:.1f} |")
    L.append("")

    L.append("## 6. Data checks\n")
    ref = used.refine_deg
    L.append(f"* Mounting: walking-like signal per person {used.gait_s.median() / 60:.0f} min median (min {used.gait_s.min() / 60:.0f}); its mean "
             f"specific force {used.gait_norm.median():.2f} g. One device rotation for everyone (roll {used.device_roll_deg.iloc[0]:.0f}° about "
             f"the hanging-arm direction, from pooled resting poses), then a per-person refinement of {ref.median():.0f}° median "
             f"(90th percentile {ref.quantile(0.9):.0f}°).\n"
             f"* Wear: {used.wear_hours.median():.0f} h worn of {used.hours.median():.0f} h recorded (median person).\n"
             "* The GT3X+ reports 0.001 g steps and holds identical values while still, so a still wrist reads almost exactly 0 g of motion; "
             "the 0.02 g stillness level is well above that floor.\n")
    L.append("## Reading\n")
    L.append("* This measures false alarms only: NHANES has no fall annotations, so a real fall during the week would count as a false alarm.\n"
             "* Device and wrist differ from a smartwatch: a research ActiGraph on the non-dominant wrist (assumed left, so not mirrored).\n"
             "* Mounting comes from detected walking and pooled resting poses, not labels; on held-out Capture-24 people seen through random "
             "device rotations that gave 28.6 alarms/day against 29.7 with the labelled alignment.\n")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L))
    print(f"report: {out / 'report.md'}")
