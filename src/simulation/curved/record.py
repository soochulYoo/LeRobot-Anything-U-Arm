"""Side-by-side video of a stiff and a compliant rotational impedance.

Same surface, same path, same translational gains, same target force.  The only
difference between the two halves of the frame is K_R, so anything you can see
is rotational stiffness.  The camera sits almost level with the slab, looking
across the traverse, because the thing to watch is the PAD'S EDGE: whether it
stays flat on the curve or rocks onto a corner.

Usage:  python3 record.py [--seed 0] [--hi 60] [--lo 3] [--fps 25]
"""
from __future__ import annotations

import argparse
import sys

import pathlib

import numpy as np
from PIL import Image

# APPEND, never insert: src/simulation contains a `mani_skill/` directory that
# would otherwise shadow the installed ManiSkill and send the agent loader off
# looking for assets it does not have.
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from record_demo import annotate, frame_of, save

import wipe as W
from surface import CurvedSurface

# Close and nearly level with the slab, with a narrow field of view: the subject
# is the PAD'S EDGE against the curve, and at a wide angle from 0.5 m away it was
# 15 pixels tall and the tilt was invisible.
CAM = {"eye": [0.60, -0.42, 0.125], "at": [0.575, 0.0, 0.060],
       "size": (780, 520), "fov": 0.42}


def record(surf, Kr: float, label: str, args) -> tuple[list, list]:
    frames, rows = [], []

    def cb(env, t, f, align, tilt):
        frames.append(annotate(
            frame_of(env), f"{label}   K_R = {Kr:g} Nm/rad",
            [f"t {t:4.1f} s     contact force {abs(f):5.1f} N",
             f"pad misalignment {align:5.1f} deg   (surface tilt {tilt:4.1f} deg)"],
            colour=(255, 210, 120) if Kr >= 20 else (150, 230, 255)))
        rows.append((t, abs(f), align))

    r = W.run_one(surf, Kr, args, render_mode="rgb_array", cam=CAM,
                  on_frame=cb, every=args.every)
    print(f"   {label:9s} K_R={Kr:5g}  align {r['align_mean']:5.2f} deg"
          f"  force {r['f_mean']:5.2f} N  rmse {r['force_rmse']:5.2f}"
          f"  peak {r['peak']:5.1f} N  contact {100*r['contact']:3.0f}%")
    return frames, rows


def main() -> None:
    ap = W.add_args(argparse.ArgumentParser())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hi", type=float, default=60.0)
    ap.add_argument("--lo", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="rot_stiffness")
    args = ap.parse_args()
    args.every = max(1, int(round(500.0 / args.fps)))   # sim runs at 500 Hz

    surf = CurvedSurface(seed=args.seed)
    st = surf.stats(args.pad_half)
    print(f"surface seed {args.seed}: tilt {st['tilt_mean']:.1f} deg mean"
          f" / {st['tilt_p95']:.1f} p95, swing across pad {st['swing_mean']:.1f} deg")
    hi, _ = record(surf, args.hi, "STIFF", args)
    lo, _ = record(surf, args.lo, "COMPLIANT", args)

    n = min(len(hi), len(lo))
    pair = [np.concatenate([hi[i], lo[i]], axis=1) for i in range(n)]
    save(pair, args.out, args.fps)


if __name__ == "__main__":
    main()
