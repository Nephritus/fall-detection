"""Showcase of the current fall detector: one self-contained HTML page (``docs/showcase.html``) with the
architecture, the evaluation protocol, lab metrics, example signals and the false-alarm story.

Every number and curve is read from the stored results (``reports/negatives``, ``reports/stillness``) and
the prepared data, so the page rebuilds after new runs. Needs ``negatives`` and ``stillness`` results.
"""

from __future__ import annotations

import html
import json
import math
from datetime import date

import numpy as np
import pandas as pd

from . import stage2, stillness
from .canonical import FS
from .freeliving import SOURCE
from .model import _IN_CHANNELS, FallCNN
from .paths import PROCESSED, RAW, REPORTS, ROOT
from .runner import Results
from .windows import STRIDE_S, WIN_S

OUT = ROOT / "docs" / "showcase.html"
DATASETS = {"erciyes": "Erciyes", "umafall": "UMAFall", "upfall": "UP-Fall", "weda": "WEDA-FALL"}
BUDGETS = np.logspace(0, np.log10(400), 60)
LONG_WAIT = 60


def esc(s) -> str:
    return html.escape(str(s))


# ----------------------------------------------------------------------------------- svg plotting
class Axes:
    """A plot area inside an SVG; maps data to pixels (linear or log axes) and draws ticks and lines."""

    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _f(self, v, log):
        return math.log10(v) if log else v

    def X(self, v):
        a, b = (self._f(v, self.xlog) for v in self.xlim)
        return self.x0 + (self._f(v, self.xlog) - a) / (b - a) * self.w

    def Y(self, v):
        a, b = (self._f(v, self.ylog) for v in self.ylim)
        return self.y0 + self.h - (self._f(v, self.ylog) - a) / (b - a) * self.h

    def frame(self, xticks, yticks, xlabel="", ylabel="", xfmt="{:g}", yfmt="{:g}") -> str:
        s = [f'<rect class="plotbg" x="{self.x0}" y="{self.y0}" width="{self.w}" height="{self.h}"/>']
        for t in yticks:
            y = self.Y(t)
            s.append(f'<line class="gl" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{y:.1f}" y2="{y:.1f}"/>')
            s.append(f'<text x="{self.x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{esc(yfmt.format(t))}</text>')
        for t in xticks:
            x = self.X(t)
            s.append(f'<line class="gl" x1="{x:.1f}" x2="{x:.1f}" y1="{self.y0}" y2="{self.y0 + self.h}"/>')
            s.append(f'<text x="{x:.1f}" y="{self.y0 + self.h + 16}" text-anchor="middle">{esc(xfmt.format(t))}</text>')
        s.append(f'<rect class="ax" x="{self.x0}" y="{self.y0}" width="{self.w}" height="{self.h}"/>')
        if xlabel:
            s.append(f'<text x="{self.x0 + self.w / 2}" y="{self.y0 + self.h + 34}" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            cx, cy = self.x0 - 44, self.y0 + self.h / 2
            s.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" transform="rotate(-90 {cx} {cy})">{esc(ylabel)}</text>')
        return "".join(s)

    def line(self, xs, ys, cls, step=False) -> str:
        pts, out = [], []
        for x, y in zip(xs, ys):
            ok = np.isfinite(x) and np.isfinite(y) and (not self.ylog or y > 0) and (not self.xlog or x > 0)
            if ok:
                if step and pts:
                    pts.append(f"{self.X(x):.1f},{float(pts[-1].split(',')[1]):.1f}")
                pts.append(f"{self.X(x):.1f},{self.Y(min(max(y, self.ylim[0]), self.ylim[1])):.1f}")
            elif pts:
                out.append(pts)
                pts = []
        out.append(pts)
        return "".join(f'<polyline class="{cls}" points="{" ".join(p)}"/>' for p in out if len(p) > 1)

    def band(self, a, b, cls, label="", row=0) -> str:
        a, b = max(a, self.xlim[0]), min(b, self.xlim[1])
        s = f'<rect class="{cls}" x="{self.X(a):.1f}" y="{self.y0}" width="{max(self.X(b) - self.X(a), 0):.1f}" height="{self.h}"/>'
        if label:
            s += f'<text class="bl" x="{self.X(a) + 3:.1f}" y="{self.y0 + 13 + 14 * row}">{esc(label)}</text>'
        return s

    def hline(self, v, cls, label="") -> str:
        y = self.Y(v)
        s = f'<line class="{cls}" x1="{self.x0}" x2="{self.x0 + self.w}" y1="{y:.1f}" y2="{y:.1f}"/>'
        if label:
            s += f'<text class="bl" x="{self.x0 + self.w - 4}" y="{y - 4:.1f}" text-anchor="end">{esc(label)}</text>'
        return s

    def vline(self, v, cls, label="") -> str:
        x = self.X(v)
        s = f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{self.y0}" y2="{self.y0 + self.h}"/>'
        if label:
            s += f'<text class="bl" x="{x + 3:.1f}" y="{self.y0 + self.h - 6}">{esc(label)}</text>'
        return s


def svg(w, h, body, label) -> str:
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">{body}</svg>'


def legend(x, y, items) -> str:
    s = []
    for i, (cls, name) in enumerate(items):
        yy = y + 18 * i
        s.append(f'<line class="{cls}" x1="{x}" x2="{x + 22}" y1="{yy}" y2="{yy}"/><text x="{x + 28}" y="{yy + 4}">{esc(name)}</text>')
    return "".join(s)


def box(x, y, w, h, title, lines, cls="box", tcls="bt") -> str:
    s = [f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>',
         f'<text class="{tcls}" x="{x + w / 2}" y="{y + 20}" text-anchor="middle">{esc(title)}</text>']
    for i, ln in enumerate(lines):
        s.append(f'<text x="{x + w / 2}" y="{y + 38 + 15 * i}" text-anchor="middle">{esc(ln)}</text>')
    return "".join(s)


def arrow(x1, y1, x2, y2, mid="arr") -> str:
    return f'<line class="arrow" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#{mid})"/>'


def marker(mid="arr") -> str:
    return (f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path class="ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>')


# ----------------------------------------------------------------------------------------- facts
def _per_seed(A, system, fn) -> pd.Series:
    v = pd.DataFrame([{"seed": s, "v": fn(a)} for (c, s, _), a in A.items() if c == system])
    return v.groupby("seed").v.mean()


def collect() -> dict:
    """Every number on the page, from the stores and the prepared data."""
    F: dict = {}
    F["params"] = sum(p.numel() for p in FallCNN(_IN_CHANNELS["acc"]).parameters())
    F["prep"] = json.loads((PROCESSED / "prepare_log.json").read_text())

    st = Results(stillness.RUNS)
    A = {k: st.arrays(*k) for k in st.keys() if k[0] != "noise-floor"}
    F["A"] = A
    F["hours"] = next(iter(A.values()))["hours"].sum()
    F["people"] = len(next(iter(A.values()))["hours"])
    rate = {}
    for s in ("stage1", "two-stage"):
        for w in stillness.WAITS:
            v = _per_seed(A, s, lambda a: stillness._alarms(a, w).per_day(0.5))
            rate[(s, w)] = (v.mean(), v.std())
    F["rate"] = rate
    # per-recording detection of unseen-dataset falls (stage 1, before any check), macro over datasets
    rows = []
    for (c, s, _), a in A.items():
        if c != "stage1":
            continue
        m = a["fall_counted"]
        ds = np.array([g.split()[0] for g in a["fall_group"]])
        for d in np.unique(ds[m]):
            rows.append({"seed": s, "ds": d, "det": a["fall_detected"][m & (ds == d)].mean()})
    R = pd.DataFrame(rows)
    F["rec_sens"] = R.groupby("ds").det.mean()
    F["rec_sens_macro"] = R.groupby("seed").det.mean().mean()
    two = []
    for (c, s, d), a in A.items():
        if c == "two-stage":
            m = a["fall_counted"]
            two.append({"seed": s, "ds": d, "det": a["fall_detected"][m].mean()})
    F["rec_sens_two"] = pd.DataFrame(two).groupby("seed").det.mean().mean()

    # lab metrics and trade-off curves: leave-one-dataset-out models with / without Capture-24 negatives
    neg = Results(REPORTS / "negatives" / "runs")
    tab = neg.table()
    F["lab"] = tab[tab.condition.isin(["real", "c24-random"])].groupby(["condition", "fold"])[
        ["auroc", "ap", "sensitivity", "false_alarms_per_h", "fl_events_per_day_at_val_thr"]].mean()
    F["fl_real"] = tab[tab.condition == "real"].groupby("seed").fl_events_per_day_at_val_thr.mean().mean()
    F["fl_c24_lodo"] = tab[tab.condition == "c24-random"].groupby("seed").fl_events_per_day_at_val_thr.mean().mean()
    curves = {}
    for cond in ("real", "c24-random"):
        per = []
        for c, s, d in neg.keys():
            if c != cond:
                continue
            z = neg.arrays(c, s, d)
            sens = np.array([(z["falls"] >= g).mean() for g in z["grid"]])
            per.append([sens[z["ev"] <= b].max() if (z["ev"] <= b).any() else 0.0 for b in BUDGETS])
        curves[cond] = np.mean(per, axis=0)
    F["curves"] = curves

    # stillness: lab falls that pass, by group (stage 1, strict, default level)
    rows = []
    for (c, s, _), a in A.items():
        if c != "stage1":
            continue
        use = a["fall_detected"] & a["fall_counted"]
        for g in np.unique(a["fall_group"][use]):
            m = use & (a["fall_group"] == g)
            r = {"group": g}
            for w in (2, 5):
                known = m & ~np.isnan(a["fall_motion"][:, :w]).any(axis=1)
                r[f"n{w}"], r[f"k{w}"] = known.sum(), stillness.stays_still(a["fall_motion"][known], w).sum()
            rows.append(r)
    F["fall_pass"] = pd.DataFrame(rows).groupby("group").sum()
    noise = st.table()
    F["noise"] = noise[noise.condition == "noise-floor"][["source", "motion_g"]]

    # where alarms come from: alarms per day contributed by each activity
    pids = SOURCE.split(0).held_out
    label_hours = []
    for p in pids:
        u, c = np.unique(SOURCE.labels(p), return_counts=True)
        label_hours.append({k: n / FS / 3600 for k, n in zip(u, c)})
    F["label_share"] = pd.DataFrame(label_hours).sum() / F["hours"]
    contrib = {}
    for name, sysname, w in (("stage 1", "stage1", 0), ("two-stage", "two-stage", 0), ("+ 10 s stillness", "two-stage", 10)):
        keys = [k for k in A if k[0] == sysname]
        ev = pd.concat([stillness._alarms(A[k], w, labels=label_hours).by_label(0.5) for k in keys]).groupby("label").events.sum()
        contrib[name] = ev / (F["hours"] * len(keys)) * 24
    F["contrib"] = pd.DataFrame(contrib).fillna(0.0)
    F["examples"] = examples(A, pids)
    return F


def examples(A, pids) -> dict:
    """Signals for the example panels: two lab falls (with and without getting up) and two Capture-24 alarms."""
    from .loaders import LOADERS

    a = A[("stage1", 0, "all")]
    still5 = stillness.stays_still(np.nan_to_num(a["fall_motion"], nan=np.inf), 5)
    ok = a["fall_detected"] & a["fall_counted"] & ~np.isnan(a["fall_motion"][:, :5]).any(axis=1)
    pick_fall = np.where(ok & (a["fall_activity"] == "901") & still5)[0][0]
    pick_rec = np.where(ok & (a["fall_activity"] == "905") & ~still5)[0][0]
    recs = []
    for loader in LOADERS.values():  # the global recording order of prepare / stage2 / stillness
        try:
            recs += loader(RAW)[0]
        except FileNotFoundError:
            pass
    out = {}
    for key, i in (("fall", pick_fall), ("recovery", pick_rec)):
        r = recs[int(a["fall_rec"][i])]
        assert (r.dataset, str(r.activity)) == ("erciyes", a["fall_activity"][i])
        out[key] = {"acc": r.acc, "impact": r.impact_s, "activity": str(r.activity), "subject": r.subject}
    m = np.nan_to_num(a["det_motion"], nan=0.0)
    # a detection after which the wrist keeps moving (an everyday activity), and one followed by a minute of stillness
    moving = np.where(~stillness.stays_still(m, 2) & (a["det_lab"] == "household-chores"))[0][0]
    resting = np.where(stillness.stays_still(m, LONG_WAIT) & (a["det_t"] > 60))[0][0]
    for key, i in (("c24_moving", moving), ("c24_still", resting)):
        p = pids[int(a["det_w"][i])]
        out[key] = {"acc": SOURCE.signal(p), "t": float(a["det_t"][i]), "label": str(a["det_lab"][i]),
                    "scores": np.load(stillness.C24 / "scores" / f"{p}.npy")[0]}
    thr = pd.read_csv(stillness.C24 / "thresholds.csv")
    out["threshold"] = float(thr[(thr.seed == 0) & (thr.rule == "val-slice")].threshold.iloc[0])
    return out


# ------------------------------------------------------------------------------------- diagrams
def pipeline_svg(F) -> str:
    r = F["rate"]
    w, h, bw, gap = 980, 230, 152, 11
    xs = [8 + i * (bw + gap) for i in range(6)]
    items = [
        ("Accelerometer", ["x, y, z in g", f"resampled to {FS} Hz", "rotated to wrist frame"], "box"),
        ("Windows", [f"{WIN_S:g} s = {int(WIN_S * FS)} samples", f"a new one every {STRIDE_S:g} s"], "box"),
        ("Stage 1: CNN", [f"{F['params']:,} parameters", "fall score 0–1", "fires above threshold"], "box hl"),
        ("Stage 2 (optional)", ["impact |a| ≥ 2 g", "context −1 s … +4 s", "gradient boosting"], "box dim"),
        ("Stillness check", ["wrist stays still", "for 10–60 s", "after the decision"], "box hl2"),
        ("Prompt, then alert", ["“Did you fall?”", "countdown, then", "emergency call"], "box"),
    ]
    chips = [None, None, f"{r[('stage1', 0)][0]:.0f} alarms/day", f"{r[('two-stage', 0)][0]:.0f} alarms/day",
             f"{r[('stage1', 10)][0]:.2f} / day (10 s)", f"{r[('stage1', LONG_WAIT)][0]:.2f} / day ({LONG_WAIT} s)"]
    s = [marker("pa")]
    for i, (x, (t, lines, cls)) in enumerate(zip(xs, items)):
        s.append(box(x, 20, bw, 92, t, lines, cls))
        if i < 5:
            s.append(arrow(x + bw + 2, 66, x + bw + gap - 2, 66, "pa"))
    for i, c in enumerate(chips):
        if c:
            x = xs[i] + bw / 2
            cls = "chip good" if i >= 4 else "chip"
            s.append(f'<rect class="{cls}" x="{x - 66}" y="128" width="132" height="24" rx="12"/>'
                     f'<text class="chipt" x="{x}" y="144" text-anchor="middle">{esc(c)}</text>')
    s.append(f'<text x="10" y="180">Chips: free-living false alarms after each step, {F["people"]} held-out Capture-24 people, '
             f'{F["hours"]:,.0f} h, nobody the models trained on.</text>')
    s.append(f'<text x="10" y="198">Stage 2 halves alarms on its own, but once the stillness check is on, stage 1 alone does as well '
             f'({r[("stage1", 10)][0]:.2f} vs {r[("two-stage", 10)][0]:.2f} per day after 10 s).</text>')
    s.append('<text x="10" y="216">The stillness check delays every alert by its wait.</text>')
    return svg(w, h, "".join(s), "Detection pipeline")


def cnn_svg(F) -> str:
    T = int(WIN_S * FS)
    shapes = [(4, T), (32, T // 2), (64, T // 4), (128, T // 8)]
    w, h = 980, 270
    s = [marker("ca")]
    x, mid = 20, 130
    labels = ["input", "block 1", "block 2", "block 3"]
    centers = []
    for i, (c, t) in enumerate(shapes):
        bh = 20 + 100 * t / T
        bw = 10 + 8 * math.log2(c)
        cls = "tensor in" if i == 0 else "tensor"
        s.append(f'<rect class="{cls}" x="{x}" y="{mid - bh / 2:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3"/>')
        s.append(f'<text x="{x + bw / 2:.1f}" y="{mid + bh / 2 + 18:.1f}" text-anchor="middle">{c} × {t}</text>')
        s.append(f'<text class="sub" x="{x + bw / 2:.1f}" y="{mid + bh / 2 + 34:.1f}" text-anchor="middle">{labels[i]}</text>')
        centers.append((x, bw))
        x += bw + (135 if i < 3 else 60)
    ops = ["Conv1d k=5 → 32", "Conv1d k=5 → 64", "Conv1d k=5 → 128"]
    for i in range(3):
        x0, bw = centers[i]
        x1 = centers[i + 1][0]
        s.append(arrow(x0 + bw + 4, mid, x1 - 4, mid, "ca"))
        s.append(f'<text class="op" x="{(x0 + bw + x1) / 2:.1f}" y="{mid - 8}" text-anchor="middle">{ops[i]}</text>')
        s.append(f'<text class="sub" x="{(x0 + bw + x1) / 2:.1f}" y="{mid + 18}" text-anchor="middle">BN · ReLU · max-pool 2</text>')
    x0, bw = centers[-1]
    xg = x0 + bw + 60
    s.append(arrow(x0 + bw + 4, mid, xg - 4, mid, "ca"))
    s.append(f'<text class="op" x="{x0 + bw + 30:.1f}" y="{mid - 24}" text-anchor="middle">global</text>')
    s.append(f'<text class="op" x="{x0 + bw + 30:.1f}" y="{mid - 8}" text-anchor="middle">avg pool</text>')
    s.append(f'<rect class="tensor" x="{xg}" y="{mid - 50}" width="14" height="100" rx="3"/>'
             f'<text x="{xg + 7}" y="{mid + 68}" text-anchor="middle">128</text>')
    xd = xg + 44
    s.append(arrow(xg + 18, mid, xd - 4, mid, "ca"))
    s.append(box(xd, mid - 36, 124, 72, "Dropout 0.3", ["Linear 128 → 1", "sigmoid"], "box"))
    xs_ = xd + 150
    s.append(arrow(xd + 128, mid, xs_ - 4, mid, "ca"))
    s.append(box(xs_, mid - 30, 96, 60, "fall score", ["0 … 1"], "box hl"))
    s.append(f'<text x="20" y="22">Input channels: x, y, z and |a|, each divided by 4 g (the clip range shared by all datasets); '
             f'{T} samples = {WIN_S:g} s at {FS} Hz.</text>')
    s.append(f'<text x="20" y="250">{F["params"]:,} parameters. Box height ∝ time steps, width ∝ log2(channels).</text>')
    return svg(w, h, "".join(s), "CNN architecture")


def protocol_svg(F) -> str:
    w, h = 980, 250
    s = [marker("pr")]
    names = list(DATASETS.values())
    s.append('<text class="bt" x="20" y="22">Lab fall datasets: leave one dataset out</text>')
    for row, held in enumerate(names):
        y = 36 + row * 30
        for j, n in enumerate(names):
            cls = "cell test" if n == held else "cell train"
            s.append(f'<rect class="{cls}" x="{20 + j * 98}" y="{y}" width="92" height="24" rx="4"/>'
                     f'<text x="{66 + j * 98}" y="{y + 16}" text-anchor="middle">{esc(n)}</text>')
        s.append(f'<text class="sub" x="{20 + 4 * 98 + 4}" y="{y + 16}">fold {row + 1}</text>')
    s.append('<rect class="cell train" x="20" y="166" width="16" height="12"/><text x="42" y="177">train (a subject-wise slice sets the threshold)</text>')
    s.append('<rect class="cell test" x="20" y="186" width="16" height="12"/><text x="42" y="197">unseen dataset: test</text>')
    x0 = 520
    s.append(f'<text class="bt" x="{x0}" y="22">Capture-24 free-living wear (no falls)</text>')
    s.append(f'<rect class="cell train" x="{x0}" y="36" width="270" height="40" rx="4"/>'
             f'<text x="{x0 + 135}" y="54" text-anchor="middle">P001–P100</text>'
             f'<text class="sub" x="{x0 + 135}" y="69" text-anchor="middle">1,500 random 4 s windows each as negatives</text>')
    s.append(f'<rect class="cell test" x="{x0 + 280}" y="36" width="170" height="40" rx="4"/>'
             f'<text x="{x0 + 365}" y="54" text-anchor="middle">P101–P151</text>'
             f'<text class="sub" x="{x0 + 365}" y="69" text-anchor="middle">held out: {F["people"]} people, {F["hours"]:,.0f} h</text>')
    s.append(f'<text x="{x0}" y="104">Every window of the held-out people is scored; detections</text>'
             f'<text x="{x0}" y="121">closer than 30 s merge into one alarm event; rates are</text>'
             f'<text x="{x0}" y="138">alarm events per 24 h of wear (sleep included).</text>')
    s.append(f'<text x="{x0}" y="170">Sensitivity is measured on simulated falls of the unseen</text>'
             f'<text x="{x0}" y="187">lab dataset; false alarms on people nobody trained on.</text>')
    return svg(w, h, "".join(s), "Evaluation protocol")


# --------------------------------------------------------------------------------------- charts
def ladder_svg(F) -> str:
    r = F["rate"]
    steps = [("Stage 1 without Capture-24 negatives", F["fl_real"], "bad", "leave-one-dataset-out models"),
             ("+ Capture-24 negatives in training", r[("stage1", 0)][0], "warn", "final models"),
             ("+ stage 2 confirmation", r[("two-stage", 0)][0], "warn", "per fold"),
             ("+ stillness check, 10 s", r[("two-stage", 10)][0], "good", "alert 10 s later"),
             (f"+ stillness check, {LONG_WAIT} s", r[("two-stage", LONG_WAIT)][0], "good", f"alert {LONG_WAIT} s later")]
    w, h = 980, 300
    ax = Axes(300, 30, 560, 200, (0.1, 200), (0, len(steps)), xlog=True)
    s = [ax.frame([0.1, 0.3, 1, 3, 10, 30, 100], [], "alarm events per day (log scale)", "", "{:g}")]
    lo, hi = 3 / 7, 6 / 7
    s.append(f'<rect class="target" x="{ax.X(lo):.1f}" y="{ax.y0}" width="{ax.X(hi) - ax.X(lo):.1f}" height="{ax.h}"/>'
             f'<text class="bl" x="{ax.X(lo) + 3:.1f}" y="{ax.y0 + ax.h - 6}">target: 3–6 a week (assumed)</text>')
    for i, (name, v, cls, note) in enumerate(steps):
        y = ax.y0 + 6 + i * 39
        s.append(f'<rect class="hb {cls}" x="{ax.x0}" y="{y}" width="{ax.X(v) - ax.x0:.1f}" height="26" rx="3"/>')
        s.append(f'<text x="{ax.x0 - 8}" y="{y + 13}" text-anchor="end">{esc(name)}</text>')
        s.append(f'<text class="sub" x="{ax.x0 - 8}" y="{y + 26}" text-anchor="end">{esc(note)}</text>')
        s.append(f'<text class="val" x="{ax.X(v) + 6:.1f}" y="{y + 18}">{v:.2f}</text>' if v < 10 else
                 f'<text class="val" x="{ax.X(v) + 6:.1f}" y="{y + 18}">{v:.1f}</text>')
    return svg(w, h, "".join(s), "False alarms per day, step by step")


def tradeoff_svg(F) -> str:
    w, h = 980, 330
    ax = Axes(90, 20, 600, 250, (1, 400), (0.2, 1.0), xlog=True)
    s = [ax.frame([1, 3, 10, 30, 100, 300], [0.2, 0.4, 0.6, 0.8, 1.0], "free-living alarm events per day allowed (log scale)",
                  "fall windows detected", "{:g}", "{:.1f}")]
    s.append(ax.line(BUDGETS, F["curves"]["real"], "l2", step=True))
    s.append(ax.line(BUDGETS, F["curves"]["c24-random"], "l1", step=True))
    s.append(legend(720, 40, [("l1", "with Capture-24 negatives"), ("l2", "lab data only")]))
    for b in (10, 50):
        i = int(np.argmin(np.abs(BUDGETS - b)))
        v1, v2 = F["curves"]["c24-random"][i], F["curves"]["real"][i]
        s.append(f'<circle class="dot1" cx="{ax.X(BUDGETS[i]):.1f}" cy="{ax.Y(v1):.1f}" r="4"/>'
                 f'<text class="val" x="{ax.X(BUDGETS[i]) + 6:.1f}" y="{ax.Y(v1) - 8:.1f}">{v1:.2f}</text>')
        s.append(f'<circle class="dot2" cx="{ax.X(BUDGETS[i]):.1f}" cy="{ax.Y(v2):.1f}" r="4"/>'
                 f'<text class="val" x="{ax.X(BUDGETS[i]) + 6:.1f}" y="{ax.Y(v2) + 16:.1f}">{v2:.2f}</text>')
    s.append('<text x="720" y="100">Stage 1 only, no stillness check.</text>'
             '<text x="720" y="118">Threshold picked on the held-out</text>'
             '<text x="720" y="136">data for each budget (best case).</text>'
             '<text x="720" y="160">Mean over 4 unseen datasets,</text>'
             '<text x="720" y="178">3 seeds; per 4 s fall window.</text>')
    return svg(w, h, "".join(s), "Sensitivity against false-alarm budget")


def stillness_svg(F) -> str:
    r = F["rate"]
    waits = [w for w in stillness.WAITS if w > 0]
    w, h = 980, 320
    ax = Axes(90, 20, 600, 250, (2, 120), (0.08, 50), xlog=True, ylog=True)
    s = [ax.frame([2, 5, 10, 20, 30, 60, 120], [0.1, 0.3, 1, 3, 10, 30], "wait after the decision, s (log scale)",
                  "alarm events per day (log)", "{:g}", "{:g}")]
    lo, hi = 3 / 7, 6 / 7
    s.append(f'<rect class="target" x="{ax.x0}" y="{ax.Y(hi):.1f}" width="{ax.w}" height="{ax.Y(lo) - ax.Y(hi):.1f}"/>'
             f'<text class="bl" x="{ax.x0 + ax.w - 4}" y="{ax.Y(lo) + 13:.1f}" text-anchor="end">target: 3–6 a week (assumed)</text>')
    for sys, cls in (("stage1", "l1"), ("two-stage", "l2")):
        s.append(ax.hline(r[(sys, 0)][0], f"{cls} dash"))
        s.append(ax.line(waits, [r[(sys, x)][0] for x in waits], cls))
        for x in waits:
            s.append(f'<circle class="dot{cls[-1]}" cx="{ax.X(x):.1f}" cy="{ax.Y(r[(sys, x)][0]):.1f}" r="3"/>')
    s.append(legend(720, 40, [("l1", "stage 1"), ("l2", "two-stage"), ("l1 dash", "no check (stage 1)"), ("l2 dash", "no check (two-stage)")]))
    s.append(f'<text x="720" y="130">10 s: {r[("stage1", 10)][0]:.2f} / {r[("two-stage", 10)][0]:.2f} per day</text>'
             f'<text x="720" y="148">{LONG_WAIT} s: {r[("stage1", LONG_WAIT)][0]:.2f} / {r[("two-stage", LONG_WAIT)][0]:.2f} per day</text>'
             f'<text x="720" y="172">still = every second below</text><text x="720" y="190">{stillness.STILL_G:g} g of movement</text>')
    return svg(w, h, "".join(s), "Alarms against stillness wait")


def contrib_svg(F) -> str:
    C, share = F["contrib"], F["label_share"]
    order = C["stage 1"].sort_values(ascending=False).index
    w, rowh = 980, 30
    h = 30 + rowh * len(order) + 48
    ax = Axes(220, 30, 560, rowh * len(order), (0, max(C["stage 1"].max() * 1.15, 1)), (0, len(order)))
    ticks = [t for t in (0, 2, 4, 6, 8, 10, 12, 14, 16) if t <= ax.xlim[1]]
    s = [ax.frame(ticks, [], "alarm events per day of wear coming from this activity", "", "{:g}")]
    cols = [("stage 1", "a1"), ("two-stage", "a2"), ("+ 10 s stillness", "a3")]
    for i, lbl in enumerate(order):
        y = ax.y0 + i * rowh + 3
        s.append(f'<text x="{ax.x0 - 8}" y="{y + 15}" text-anchor="end">{esc(lbl)} ({share.get(lbl, 0):.0%} of wear)</text>')
        for j, (c, cls) in enumerate(cols):
            v = C.loc[lbl, c]
            s.append(f'<rect class="hb {cls}" x="{ax.x0}" y="{y + 8 * j}" width="{max(ax.X(v) - ax.x0, 0.5):.1f}" height="7"/>')
        s.append(f'<text class="val" x="{ax.X(C.loc[lbl, "stage 1"]) + 6:.1f}" y="{y + 12}">{C.loc[lbl, "stage 1"]:.1f}</text>')
    for j, (c, cls) in enumerate(cols):
        s.append(f'<rect class="hb {cls}" x="800" y="{40 + 20 * j}" width="22" height="10"/><text x="828" y="{49 + 20 * j}">{esc(c)}</text>')
    return svg(w, h, "".join(s), "Where alarms come from")


def lab_signal_svg(ex, title, passed) -> str:
    acc, imp = ex["acc"], ex["impact"]
    t = np.arange(len(acc)) / FS
    mag = np.linalg.norm(acc, axis=1)
    a0, a1 = max(0.0, imp - 3.0), t[-1]
    m = (t >= a0) & (t <= a1)
    w, h = 480, 250
    ymax = math.ceil(max(4.0, float(mag[m].max())))
    ax = Axes(52, 40, 410, 160, (a0 - imp, a1 - imp), (0, ymax))
    s = [f'<text class="bt" x="10" y="20">{esc(title)}</text>',
         ax.frame([x for x in range(-3, 12, 1) if a0 - imp <= x <= a1 - imp], range(ymax + 1), "seconds from impact", "|a| (g)", "{:g}", "{:g}"),
         ax.band(-stage2.PRE_S, stage2.POST_S, "bctx", "stage-2 context", 0),
         ax.band(stage2.POST_S, stage2.POST_S + 5, "bpass" if passed else "bfail",
                 "5 s still ✓" if passed else "5 s still ✗", 1),
         ax.hline(2.0, "dash muted", "2 g trigger"), ax.vline(0, "dash muted"),
         ax.line(t[m] - imp, mag[m], "l1")]
    return svg(w, h, "".join(s), title)


def c24_signal_svg(ex, title, thr, wait) -> str:
    acc, t0 = ex["acc"], ex["t"]
    a0, a1 = t0 - 20, t0 + max(wait, 10) + 15
    i0, i1 = int(a0 * FS), int(a1 * FS)
    t = np.arange(i0, i1) / FS
    mag = np.linalg.norm(acc[i0:i1], axis=1)
    sc = ex["scores"]
    ends = np.arange(len(sc)) * STRIDE_S + WIN_S
    k = (ends >= a0) & (ends <= a1)
    m = stillness.post_motion(acc, np.array([t0]), wait)
    passed = bool(stillness.stays_still(m, wait)[0])
    w, h = 480, 300
    ymax = math.ceil(max(3.0, float(mag.max())))
    top = Axes(52, 40, 410, 130, (a0 - t0, a1 - t0), (0, ymax))
    bot = Axes(52, 205, 410, 55, (a0 - t0, a1 - t0), (0, 1))
    xt = [x for x in range(-20, 80, 10) if a0 - t0 <= x <= a1 - t0]
    s = [f'<text class="bt" x="10" y="20">{esc(title)}</text>',
         top.frame(xt, range(ymax + 1), "", "|a| (g)", "{:g}", "{:g}"),
         top.band(-WIN_S, 0, "bctx", "window that fired", 0),
         top.band(0, wait, "bpass" if passed else "bfail", f"{wait} s still {'✓' if passed else '✗'}", 1),
         top.line(t - t0, mag, "l1"),
         bot.frame(xt, [0, 1], "seconds from the decision", "score", "{:g}", "{:g}"),
         bot.hline(thr, "dash muted"),
         bot.line(ends[k] - t0, sc[k], "l2", step=True)]
    return svg(w, h, "".join(s), title)


# ------------------------------------------------------------------------------------------- page
CSS = """
:root { --bg:#fff; --fg:#1b1f24; --muted:#586069; --border:#d8dee4; --surface:#f6f8fa;
        --accent:#0969da; --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --c4:#8250df; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --border:#30363d; --surface:#161b22;
          --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149; --c4:#bc8cff; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1040px; margin:0 auto; padding:2rem 1.25rem 4rem; }
h1 { font-size:1.9rem; line-height:1.2; margin:0 0 .3rem; }
h2 { font-size:1.35rem; margin:2.6rem 0 .6rem; padding-bottom:.3rem; border-bottom:1px solid var(--border); }
p, li { max-width:80ch; }
.meta, .note { color:var(--muted); font-size:.88rem; }
.hero { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem; margin:1.4rem 0; }
.card { border:1px solid var(--border); border-radius:8px; padding:.9rem 1rem; background:var(--surface); }
.card .big { font-size:1.9rem; font-weight:700; line-height:1.1; }
.card .lbl { color:var(--muted); font-size:.86rem; margin-top:.3rem; }
.card.good .big { color:var(--ok); } .card.warn .big { color:var(--warn); }
.fig { border:1px solid var(--border); border-radius:8px; padding:.6rem; margin:.6rem 0 .4rem; background:var(--bg); }
.pair { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:.8rem; }
.pair .fig { margin:0; }
svg { width:100%; height:auto; display:block; font-size:12.5px; }
svg text { fill:var(--fg); } svg text.sub, svg text.bl { fill:var(--muted); font-size:11.5px; }
svg text.bt { font-weight:600; font-size:13.5px; } svg text.op { font-weight:600; } svg text.val { font-weight:600; font-size:12px; }
svg .plotbg { fill:var(--bg); } svg .ax { fill:none; stroke:var(--border); } svg .gl { stroke:var(--border); stroke-dasharray:2 4; }
svg .box { fill:var(--surface); stroke:var(--border); } svg .box.hl { stroke:var(--accent); stroke-width:2; }
svg .box.hl2 { stroke:var(--ok); stroke-width:2; } svg .box.dim { stroke-dasharray:5 4; }
svg .arrow { stroke:var(--muted); stroke-width:1.5; } svg .ah { fill:var(--muted); }
svg .chip { fill:var(--surface); stroke:var(--warn); } svg .chip.good { stroke:var(--ok); } svg .chipt { font-weight:600; }
svg .tensor { fill:var(--accent); fill-opacity:.18; stroke:var(--accent); } svg .tensor.in { fill:var(--muted); stroke:var(--muted); }
svg .cell.train { fill:var(--accent); fill-opacity:.14; stroke:var(--accent); } svg .cell.test { fill:var(--warn); fill-opacity:.2; stroke:var(--warn); }
svg .l1, svg .l2 { fill:none; stroke-width:2; } svg .l1 { stroke:var(--accent); } svg .l2 { stroke:var(--warn); }
svg .dash { stroke-dasharray:6 4; stroke-width:1.3; } svg .muted { stroke:var(--muted); }
svg .dot1 { fill:var(--accent); } svg .dot2 { fill:var(--warn); }
svg .hb.bad { fill:var(--bad); } svg .hb.warn { fill:var(--warn); } svg .hb.good { fill:var(--ok); }
svg .hb.a1 { fill:var(--warn); } svg .hb.a2 { fill:var(--accent); } svg .hb.a3 { fill:var(--ok); }
svg .target { fill:var(--ok); fill-opacity:.12; }
svg .bctx { fill:var(--accent); fill-opacity:.10; } svg .bpass { fill:var(--ok); fill-opacity:.18; } svg .bfail { fill:var(--bad); fill-opacity:.16; }
.tablewrap { overflow-x:auto; border:1px solid var(--border); border-radius:8px; margin:.5rem 0 1rem; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:.45rem .7rem; border-bottom:1px solid var(--border); }
th { background:var(--surface); } tr:last-child td { border-bottom:0; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.callout { border:1px solid var(--warn); border-left-width:4px; border-radius:8px; padding:.8rem 1.1rem; background:var(--surface); }
code { background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:0 .3em; font-size:.88em; }
footer { margin-top:3rem; color:var(--muted); font-size:.85rem; border-top:1px solid var(--border); padding-top:1rem; }
"""


def page(F) -> str:
    r, prep, ex = F["rate"], F["prep"], F["examples"]
    lab = F["lab"]
    ds_rows = []
    for k, name in DATASETS.items():
        d = prep["datasets"].get(k)
        if d:
            ds_rows.append(f'<tr><td>{name}</td><td class="n">{d["subjects"]}</td><td class="n">{d["fall_recordings"]:,}</td>'
                           f'<td class="n">{d["adl_recordings"]:,}</td></tr>')
    lab_rows = []
    for k, name in DATASETS.items():
        if ("c24-random", k) not in lab.index:
            continue
        a, b = lab.loc[("real", k)], lab.loc[("c24-random", k)]
        lab_rows.append(
            f'<tr><td>{name}</td><td class="n">{b.auroc:.3f}</td><td class="n">{a.ap:.3f} → <b>{b.ap:.3f}</b></td>'
            f'<td class="n">{b.sensitivity:.2f}</td><td class="n">{F["rec_sens"].get(k, np.nan):.2f}</td>'
            f'<td class="n">{a.fl_events_per_day_at_val_thr:.0f} → <b>{b.fl_events_per_day_at_val_thr:.0f}</b></td></tr>')
    fp = F["fall_pass"]
    group_names = {"erciyes other": "Erciyes, stays down", "erciyes recovery": "Erciyes, gets up (905/906/914/916)",
                   "umafall": "UMAFall", "upfall": "UP-Fall", "weda": "WEDA-FALL"}
    noise = dict(zip(F["noise"].source, F["noise"].motion_g))
    fp_rows = []
    for g, name in group_names.items():
        if g not in fp.index:
            continue
        x = fp.loc[g]
        cells = "".join(f'<td class="n">{x[f"k{w}"] / x[f"n{w}"]:.2f} <span class="note">(n={x[f"n{w}"] // 3:.0f})</span></td>'
                        if x[f"n{w}"] >= 30 else '<td class="n note">too few</td>' for w in (2, 5))
        fp_rows.append(f'<tr><td>{name}</td><td class="n">{noise.get(g.split()[0], np.nan):.3f}</td>{cells}</tr>')
    hw = {k: v for k, v in prep["windows"].items() if isinstance(v, (int, float))}
    n_rec = sum(d["recordings"] for d in prep["datasets"].values())
    n_falls = sum(d["fall_recordings"] for d in prep["datasets"].values())

    fall_title = f'Lab fall that stays down (Erciyes {ex["fall"]["activity"]})'
    rec_title = f'Lab fall, then gets up (Erciyes {ex["recovery"]["activity"]})'
    mov_title = f'Free-living alarm while {ex["c24_moving"]["label"]}: rejected'
    still_title = f'Free-living alarm that survives {LONG_WAIT} s ({ex["c24_still"]["label"]})'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wrist fall detector: showcase</title><style>{CSS}</style></head>
<body><main>
<h1>Wrist fall detector: what it is and how it performs</h1>
<div class="meta">Generated {date.today():%d %B %Y} from the stored results (<code>uv run falldet showcase</code>). Accelerometer only, 25 Hz, any wrist-worn watch.
All falls are <b>simulated by volunteers</b>; the false alarms are measured on real free-living wear.</div>

<div class="hero">
  <div class="card"><div class="big">{F["params"] / 1000:.1f}k</div><div class="lbl">parameters in the window CNN; {WIN_S:g} s of accelerometer per decision</div></div>
  <div class="card warn"><div class="big">{F["rec_sens_macro"]:.2f}</div><div class="lbl">share of simulated fall recordings detected on a dataset the model never saw (stage 1, mean of 4 datasets)</div></div>
  <div class="card"><div class="big">{r[("stage1", 0)][0]:.0f}<small style="font-size:1rem"> / day</small></div><div class="lbl">false alarms from the CNN alone in free-living wear</div></div>
  <div class="card good"><div class="big">{r[("stage1", 10)][0]:.2f}<small style="font-size:1rem"> / day</small></div><div class="lbl">after a 10 s stillness check ({r[("stage1", 10)][0] * 7:.1f} a week); {r[("stage1", LONG_WAIT)][0]:.2f} after {LONG_WAIT} s. Cost on real falls: unknown</div></div>
</div>

<h2>1. How a fall becomes an alert</h2>
<div class="fig">{pipeline_svg(F)}</div>
<p class="note">Rates: stage 1 = final models with Capture-24 negatives at the validation-slice threshold; two-stage = per-fold models of the stage-2 experiment.
Mean of 3 seeds. Stage 2 is drawn dashed because with the stillness check it no longer reduces alarms but still drops falls followed by movement.</p>

<h2>2. The model</h2>
<div class="fig">{cnn_svg(F)}</div>
<p>Trained with AdamW and class-weighted binary cross-entropy, early stopping on validation average precision. Each batch is augmented in signal units:
random rotation (watch fit), left/right mirror, gain, time stretch, shift and noise, so the model does not learn a dataset's mounting or sensor.
Negatives are the lab activities plus 1,500 random 4 s windows from each of 100 Capture-24 participants' everyday life.</p>

<h2>3. Data and evaluation</h2>
<div class="tablewrap"><table><thead><tr><th>Lab dataset (simulated falls)</th><th class="n">subjects</th><th class="n">fall recordings</th><th class="n">activity recordings</th></tr></thead>
<tbody>{"".join(ds_rows)}<tr><td><b>total</b></td><td></td><td class="n"><b>{n_falls:,}</b></td><td class="n"><b>{n_rec - n_falls:,}</b></td></tr></tbody></table></div>
<p class="note">{hw.get("total", 0):,} windows after preparation, {hw.get("fall", 0):,} of them containing a fall impact. Every source is resampled to 25 Hz, clipped to ±4 g and rotated into one wrist frame.</p>
<div class="fig">{protocol_svg(F)}</div>

<h2>4. Detection on unseen datasets</h2>
<div class="tablewrap"><table><thead><tr><th>held-out dataset</th><th class="n">AUROC</th><th class="n">AUPRC: lab only → with negatives</th>
<th class="n">fall windows detected</th><th class="n">fall recordings detected</th><th class="n">free-living alarms/day: lab only → with negatives</th></tr></thead>
<tbody>{"".join(lab_rows)}</tbody></table></div>
<p class="note">Leave-one-dataset-out models, 3 seeds, threshold from a validation slice of the training subjects (no held-out data). Window = one 4 s window
around an impact; a recording counts as detected if any of its fall windows fires (nested models of the final fit).</p>
<div class="fig">{tradeoff_svg(F)}</div>

<h2>5. What the signals look like</h2>
<div class="pair">
  <div class="fig">{lab_signal_svg(ex["fall"], fall_title, True)}</div>
  <div class="fig">{lab_signal_svg(ex["recovery"], rec_title, False)}</div>
  <div class="fig">{c24_signal_svg(ex["c24_moving"], mov_title, ex["threshold"], 10)}</div>
  <div class="fig">{c24_signal_svg(ex["c24_still"], still_title, ex["threshold"], LONG_WAIT)}</div>
</div>
<p class="note">Top: acceleration magnitude. Bottom (free-living): stage-1 score of each 4 s window with its threshold (dashed). Free-living examples are held-out
Capture-24 people scored by the seed-0 final model; lab examples are falls that the nested model, trained without Erciyes, detected.</p>

<h2>6. False alarms, step by step</h2>
<div class="fig">{ladder_svg(F)}</div>
<div class="fig">{stillness_svg(F)}</div>
<div class="fig">{contrib_svg(F)}</div>
<p class="note">Without a check, alarms come from ordinary daytime movement (walking, chores, unlabelled time); with it, nearly all of those go and
what remains is a hard movement followed by rest (sitting, lying, sleep), which the wrist cannot tell apart from lying on the floor.</p>

<h2>7. What the stillness check costs on falls</h2>
<div class="tablewrap"><table><thead><tr><th>detected lab falls</th><th class="n">sensor noise at rest (g)</th>
<th class="n">still for 2 s</th><th class="n">still for 5 s</th></tr></thead><tbody>{"".join(fp_rows)}</tbody></table></div>
<p class="note">Stage 1, every second below {stillness.STILL_G:g} g, starting 4 s after the impact; n = recordings long enough to check, per seed.
Lab recordings stop a few seconds after the fall, so 10–60 s waits cannot be checked at all. Sensors with more noise at rest (UMAFall, WEDA-FALL)
rarely pass: the level has to be set on the target watch.</p>

<h2>8. Read this before trusting the numbers</h2>
<div class="callout"><ul>
<li><b>Simulated falls only.</b> Every sensitivity comes from volunteers falling on mats; real falls are slower, messier and often followed by attempts to get up. Treat detection rates as upper bounds.</li>
<li><b>The stillness check's cost is unmeasured.</b> It removes {1 - r[("stage1", 10)][0] / r[("stage1", 0)][0]:.0%} of free-living alarms after 10 s, but whether people who really fell keep the wrist still for 10–60 s needs new recordings: a minute or more after each fall, with and without getting up, on the target watch.</li>
<li><b>Free-living wear is a research accelerometer</b> (Axivity AX3, Oxford adults, about a day each), not a smartwatch, and not older adults.</li>
<li><b>Seed spread is large:</b> the CNN alone ranges {r[("stage1", 0)][0] - r[("stage1", 0)][1]:.0f}–{r[("stage1", 0)][0] + r[("stage1", 0)][1]:.0f} alarms/day (± 1 std over 3 seeds). The false-alarm target of a few per week is an assumption.</li>
</ul></div>

<footer>Sources: <code>reports/negatives</code>, <code>reports/stillness</code>, <code>reports/stage2</code>, <code>data/processed/prepare_log.json</code>.
Full write-up: <code>docs/project-summary.html</code>. Rebuild: <code>uv run falldet showcase</code>.</footer>
</main></body></html>
"""


def run() -> None:
    F = collect()
    OUT.write_text(page(F))
    print(f"showcase: {OUT}")
