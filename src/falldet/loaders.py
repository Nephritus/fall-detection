"""Dataset loaders: raw files -> canonical Recordings (units g and rad/s, FS grid, wrist frame)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .canonical import (
    ACC_CLIP_G,
    FS,
    G,
    Recording,
    align_mounting,
    find_impact,
    to_grid,
)

DEG = np.pi / 180.0


def _finish(
    dataset, subject, trial, activity, is_fall, acc, gyro, interval_s
) -> Recording:
    acc = np.clip(acc, -ACC_CLIP_G, ACC_CLIP_G)
    impact_s, source = (None, "")
    if is_fall:
        impact_s, source = find_impact(acc, interval_s)
    return Recording(
        dataset, subject, trial, activity, is_fall,
        acc.astype(np.float32), gyro.astype(np.float32), impact_s, source,
    )


# --------------------------------------------------------------------------- WEDA-FALL
def load_weda(raw: Path) -> tuple[list[Recording], dict]:
    """WEDA-FALL (Fitbit Sense). Axes already match the wrist frame; accel m/s^2, gyro rad/s.

    Row timestamps are batch-arrival times (bursts ~3 ms apart, then a gap), so uniform times are
    rebuilt from the row index between each file's first and last timestamp.
    """
    base = raw / "WEDA-FALL" / "dataset" / "50Hz"
    stamps = pd.read_csv(raw / "WEDA-FALL" / "dataset" / "fall_timestamps.csv")
    stamps.columns = [c.strip("\ufeff") for c in stamps.columns]
    interval = {r.filename: (r.start_time, r.end_time) for r in stamps.itertuples()}
    pat = re.compile(r"(U\d+)_R(\d+)_accel\.csv")
    recs, skipped = [], 0
    for code_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        code = code_dir.name
        for af in sorted(code_dir.glob("U*_R*_accel.csv")):
            m = pat.fullmatch(af.name)
            if not m:
                continue
            a = pd.read_csv(af).to_numpy(dtype=float)
            g = pd.read_csv(af.with_name(af.name.replace("_accel", "_gyro"))).to_numpy(dtype=float)
            if len(a) < 10 or len(g) < 10:
                skipped += 1
                continue
            ta = np.linspace(a[0, 0], a[-1, 0], len(a))
            tg = np.linspace(g[0, 0], g[-1, 0], len(g))
            fs_native = min(len(a) / (ta[-1] - ta[0]), len(g) / (tg[-1] - tg[0]))
            t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
            grid = t0 + np.arange(int((t1 - t0) * FS) + 1) / FS
            acc = to_grid(ta, a[:, 1:4] / G, fs_native, grid)
            gyro = to_grid(tg, g[:, 1:4], fs_native, grid)
            key = f"{code}/{m.group(1)}_R{m.group(2)}"
            iv = interval.get(key)
            iv = (iv[0] - t0, iv[1] - t0) if iv else None
            recs.append(_finish(
                "weda", f"weda:{m.group(1)}", f"R{m.group(2)}", code, code.startswith("F"),
                acc, gyro, iv,
            ))
    return recs, {"skipped_short": skipped}


# ----------------------------------------------------------------------------- UP-Fall
def load_upfall(raw: Path) -> tuple[list[Recording], dict]:
    """UP-Fall wrist IMU (accel g, gyro deg/s, ~20 Hz irregular). Mounting varies per subject, so
    each subject is rotated so the standing-arm-hanging mean matches REFERENCE_HANGING_G; subjects
    whose standing arm was not still (|mean| < MIN_HANGING_NORM_G) are excluded.
    """
    df = pd.read_csv(
        raw / "upfall" / "upfall.csv", skiprows=2, header=None,
        usecols=[0, 29, 30, 31, 32, 33, 34, 43, 44, 45, 46],
    )
    df.columns = ["ts", "ax", "ay", "az", "gx", "gy", "gz", "subj", "act", "trial", "tag"]
    df["t"] = (pd.to_datetime(df["ts"]) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    recs, excluded = [], {}
    for subj, sdf in df.groupby("subj"):
        al = align_mounting(sdf[sdf.act == 7][["ax", "ay", "az"]].to_numpy(), "left")
        if al.excluded:
            excluded[f"upfall:S{subj:02d}"] = round(al.norm_g, 2)
            continue
        for (act, trial), tdf in sdf.groupby(["act", "trial"]):
            t = tdf["t"].to_numpy() - tdf["t"].iloc[0]
            keep = np.diff(t, prepend=-1.0) > 0
            tdf, t = tdf[keep], t[keep]
            if len(t) < 10:
                continue
            acc, gyro = al.apply(tdf[["ax", "ay", "az"]].to_numpy(), tdf[["gx", "gy", "gz"]].to_numpy() * DEG)
            grid = np.arange(int(t[-1] * FS) + 1) / FS
            fs_native = (len(t) - 1) / t[-1]
            is_fall = act <= 5
            iv = None
            if is_fall:
                tagged = t[(tdf["tag"] == act).to_numpy()]
                iv = (float(tagged[0]), float(tagged[-1])) if len(tagged) else None
            recs.append(_finish(
                "upfall", f"upfall:S{subj:02d}", f"T{trial}", f"A{act}", is_fall,
                to_grid(t, acc, fs_native, grid), to_grid(t, gyro, fs_native, grid), iv,
            ))
    return recs, {"excluded_subjects": excluded}


# ----------------------------------------------------------------------------- Erciyes
MAX_NAN_FRACTION = 0.05
_WRIST_SENSOR = "340537.txt"  # right wrist (Information.pdf)
_ERC_COLS = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]


def load_erciyes(raw: Path) -> tuple[list[Recording], dict]:
    """Erciyes (Xsens MTw, 25 Hz, accel m/s^2, gyro rad/s) right-wrist sensor.

    Each subject is rotated so the mean of walking-forward (801) matches the right-wrist mirror of
    REFERENCE_HANGING_G, then mirrored to the left-wrist convention. Fall labels are trial-level
    only, so every fall impact comes from the magnitude-peak fallback.
    """
    root = raw / "erciyes" / "Tests"
    recs, excluded, missing, empty, too_nan, repaired = [], {}, 0, 0, 0, 0
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        sid = sdir.name
        exports = list(sdir.glob("*/Testler Export"))
        if not exports:
            continue
        trials = []  # (code, test, acc_g, gyro)
        for adir in sorted(p for p in exports[0].iterdir() if p.is_dir()):
            code = adir.name[:3]
            for tdir in sorted(p for p in adir.iterdir() if p.is_dir()):
                f = tdir / _WRIST_SENSOR
                if not f.exists():
                    missing += 1
                    continue
                d = pd.read_csv(f, sep="\t", skiprows=4, usecols=_ERC_COLS).to_numpy(dtype=float)
                if len(d) < 10:
                    empty += 1
                    continue
                bad = np.isnan(d).any(axis=1)
                if bad.mean() > MAX_NAN_FRACTION:
                    too_nan += 1
                    continue
                if bad.any():  # rare isolated dropouts: interpolate across them
                    idx = np.arange(len(d))
                    for c in range(d.shape[1]):
                        d[bad, c] = np.interp(idx[bad], idx[~bad], d[~bad, c])
                    repaired += 1
                trials.append((code, tdir.name, d[:, :3] / G, d[:, 3:]))
        walk = [a for c, _, a, _ in trials if c == "801"]
        if not walk:
            excluded[f"erciyes:{sid}"] = "no walking trials"
            continue
        al = align_mounting(np.concatenate(walk), "right")
        if al.excluded:
            excluded[f"erciyes:{sid}"] = round(al.norm_g, 2)
            continue
        for code, test, acc, gyro in trials:
            acc, gyro = al.apply(acc, gyro)
            recs.append(_finish(
                "erciyes", f"erciyes:{sid}", test, code, int(code) >= 900, acc, gyro, None,
            ))
    return recs, {"excluded_subjects": excluded, "missing_wrist_files": missing,
                  "empty_trials": empty, "dropped_too_many_nan": too_nan, "nan_repaired": repaired}


# ----------------------------------------------------------------------------- UMAFall
_UMA_NAME = re.compile(r"UMAFall_Subject_(\d+)_(ADL|Fall)_(.+)_(\d+)_\d{4}")
_UMA_WRIST_ID = 3  # SensorTag on the wrist (file header); accel = type 0, gyro = type 1
UMA_GYRO_SATURATION_DPS = 256.0  # observed clipping of the wrist gyro


def load_umafall(raw: Path) -> tuple[list[Recording], dict]:
    """UMAFall (SensorTag MPU-9250 on the wrist, ~20 Hz, accel g, gyro deg/s).

    The wrist side is not stated in the files ("WRIST"); it is treated as the right wrist as in the
    research report [UNVERIFIED in the data]. Row timestamps are arrival bursts, so uniform times
    are rebuilt from the row index. Mounting differs per subject: each subject is rotated so the
    mean of their Walking trials matches the right-wrist mirror of REFERENCE_HANGING_G, then the
    data is mirrored to the left-wrist convention. Falls are annotated per trial only.
    """
    base = raw / "UMAFall_Dataset_corrected_version"
    if not base.is_dir():
        raise FileNotFoundError(base)
    by_subject: dict[str, list] = {}
    skipped = 0
    for f in sorted(base.glob("UMAFall_Subject_*.csv")):
        m = _UMA_NAME.match(f.name)
        if not m:
            continue
        d = pd.read_csv(f, comment="%", sep=";", header=None, usecols=range(7), skipinitialspace=True).dropna()
        d.columns = ["t", "n", "x", "y", "z", "type", "sid"]
        w = d[d.sid == _UMA_WRIST_ID]
        a, g = w[w.type == 0], w[w.type == 1]
        if len(a) < 10 or len(g) < 10:
            skipped += 1
            continue
        subj, kind, act, trial = m.groups()
        by_subject.setdefault(subj, []).append((kind, act, trial, a, g))
    recs, excluded = [], {}
    for subj, items in sorted(by_subject.items()):
        walk = [a[["x", "y", "z"]].to_numpy() for kind, act, _, a, _ in items if act == "Walking"]
        if not walk:
            excluded[f"umafall:{subj}"] = "no walking trials"
            continue
        al = align_mounting(np.concatenate(walk), "right")
        if al.excluded:
            excluded[f"umafall:{subj}"] = round(al.norm_g, 2)
            continue
        for kind, act, trial, a, g in items:
            ta = np.linspace(a.t.iloc[0], a.t.iloc[-1], len(a)) / 1000.0
            tg = np.linspace(g.t.iloc[0], g.t.iloc[-1], len(g)) / 1000.0
            t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
            grid = t0 + np.arange(int((t1 - t0) * FS) + 1) / FS
            acc = to_grid(ta, a[["x", "y", "z"]].to_numpy(), len(a) / (ta[-1] - ta[0]), grid)
            gyro = to_grid(tg, g[["x", "y", "z"]].to_numpy() * DEG, len(g) / (tg[-1] - tg[0]), grid)
            acc, gyro = al.apply(acc, gyro)
            recs.append(_finish(
                "umafall", f"umafall:{subj}", f"T{trial}", act, kind == "Fall", acc, gyro, None,
            ))
    return recs, {"excluded_subjects": excluded, "skipped_no_wrist": skipped,
                  "gyro_saturation_dps": UMA_GYRO_SATURATION_DPS}


LOADERS = {"weda": load_weda, "upfall": load_upfall, "erciyes": load_erciyes, "umafall": load_umafall}
