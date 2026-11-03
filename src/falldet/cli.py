"""Command line: `uv run falldet prepare` then `uv run falldet baseline`."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .loaders import LOADERS
from .paths import PROCESSED, RAW
from .windows import build_windows


def prepare() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    recs, log = [], {"datasets": {}}
    for name, loader in LOADERS.items():
        try:
            r, info = loader(RAW)
        except FileNotFoundError:
            print(f"[skip] {name}: not found under {RAW}")
            continue
        src = Counter(x.impact_source for x in r if x.is_fall)
        log["datasets"][name] = {
            "recordings": len(r),
            "subjects": len({x.subject for x in r}),
            "fall_recordings": sum(x.is_fall for x in r),
            "adl_recordings": sum(not x.is_fall for x in r),
            "impact_source": dict(src),
            **info,
        }
        print(name, log["datasets"][name])
        recs += r
    X, meta, counts = build_windows(recs)
    log["windows"] = {
        "total": len(meta),
        "fall": int(meta.label.sum()),
        "adl": int((meta.label == 0).sum()),
        "by_dataset": {
            d: {"fall": int(g.label.sum()), "adl": int((g.label == 0).sum())}
            for d, g in meta.groupby("dataset")
        },
        **counts,
    }
    np.save(PROCESSED / "X.npy", X)
    meta.to_csv(PROCESSED / "meta.csv", index=False)
    (PROCESSED / "prepare_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log["windows"], indent=2))


def synth() -> None:
    from .synth import build_synthetic

    X, meta, info = build_synthetic(RAW)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED / "synthetic_X.npy", X)
    meta.to_csv(PROCESSED / "synthetic_meta.csv", index=False)
    (PROCESSED / "synthetic_log.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(prog="falldet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="raw datasets -> canonical windows in data/processed/")
    b = sub.add_parser("baseline", help="train and evaluate the baseline, write reports/")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--augment", action="store_true", help="augment training windows; writes reports/baseline_augmented/")
    c = sub.add_parser("compare", help="compare two baseline runs (paired by seed and fold)")
    c.add_argument("base", type=Path)
    c.add_argument("other", type=Path)
    f = sub.add_parser("finetune", help="per-device adaptation experiment, writes reports/finetune/")
    f.add_argument("--seeds", type=int, default=3)
    f.add_argument("--augment", action="store_true", help="augment training; writes reports/finetune_augmented/")
    t = sub.add_parser("threshold", help="threshold-transfer experiment, writes reports/threshold/")
    t.add_argument("--seeds", type=int, default=3)
    t.add_argument("--augment", action="store_true", help="augment training; writes reports/threshold_augmented/")
    a = sub.add_parser("ablate", help="per-transform augmentation ablation, writes reports/ablation_<channels>/")
    a.add_argument("--seeds", type=int, default=5)
    a.add_argument("--channels", choices=["acc", "imu"], default="acc")
    sub.add_parser("synth", help="LAFAN1 mocap -> synthetic wrist-IMU windows in data/processed/")
    s = sub.add_parser("synthetic", help="does synthetic data help? writes reports/synthetic_<channels>/")
    s.add_argument("--seeds", type=int, default=8)
    s.add_argument("--channels", choices=["acc", "imu"], default="acc")
    al = sub.add_parser("alarms", help="event-level false alarms and a daily estimate, writes reports/alarms/")
    al.add_argument("--seeds", type=int, default=5)
    fl = sub.add_parser("freeliving", help="measured free-living false alarms on Capture-24, writes reports/freeliving/")
    fl.add_argument("--seeds", type=int, default=5)
    ng = sub.add_parser("negatives", help="Capture-24 as training negatives, writes reports/negatives/")
    ng.add_argument("--seed", type=int, default=None, help="run one seed (run several in parallel), then --report")
    ng.add_argument("--report", action="store_true", help="aggregate the saved seeds into the report")
    ng.add_argument("--final", type=int, metavar="SEEDS", default=None,
                    help="final models with Capture-24 negatives: free-living report and comparison")
    ng.add_argument("--scale", action="store_true",
                    help="more negatives per person (with --seed N), or with --report the dose-response report")
    s2 = sub.add_parser("stage2", help="second-stage confirmation, writes reports/stage2/")
    s2.add_argument("--seed", type=int, default=None, help="run one seed (run several in parallel), then --report")
    s2.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "synth":
        synth()
    elif args.cmd == "stage2":
        from . import stage2

        stage2.report() if args.report else stage2.run(args.seed)
    elif args.cmd == "negatives":
        from . import negatives

        if args.final:
            negatives.final(args.final)
        elif args.scale:
            negatives.scale_report() if args.report else negatives.scale(args.seed)
        else:
            negatives.report() if args.report else negatives.run(args.seed)
    elif args.cmd == "freeliving":
        from .freeliving import run as freeliving

        freeliving(args.seeds)
    elif args.cmd == "alarms":
        from .budget_estimate import run as alarms

        alarms(args.seeds)
    elif args.cmd == "synthetic":
        from .synthetic_exp import run as synthetic

        synthetic(args.seeds, args.channels)
    elif args.cmd == "baseline":
        from .baseline import run

        run(args.seeds, args.augment)
    elif args.cmd == "compare":
        from .compare import run as compare

        compare(args.base, args.other)
    elif args.cmd == "ablate":
        from .ablation import run as ablate

        ablate(args.seeds, args.channels)
    elif args.cmd == "threshold":
        from .threshold import run as threshold

        threshold(args.seeds, args.augment)
    else:
        from .finetune import run

        run(args.seeds, args.augment)
