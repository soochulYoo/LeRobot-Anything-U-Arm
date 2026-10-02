"""Step 1: does scrubbing the entrance widen the window, and where does it jam?

The aim-and-push rollout inserts when the aim is good and fails when it is not.
A demonstration worth recording does something else: it puts the peg on the
face, presses, and scrubs until the head drops into the aperture.  The question
here is whether that motion WORKS -- how much aiming error it absorbs -- because
there is no point collecting data for a search that does not search.

The geometry sets the stakes: the aperture is a square of half-width
radius + 3 mm with no chamfer, so the clearance is 3 mm and the edge is sharp.
Anything the scrub recovers beyond 3 mm it recovers by finding the hole, not by
being let in.

Three outcomes are distinguished, because they call for different fixes:
  inserted  head past the success line (15 mm short of the hole centre)
  dropped   head got past the face but did not go home -- jammed
  missed    the scrub never found the aperture at all

    python3 peg_search_study.py --errors 0,2,4,6,8,10 --episodes 4
    python3 peg_search_study.py --errors 6 --episodes 8 --push 0   # one cell
"""
from __future__ import annotations

import argparse
import copy

import numpy as np

import peg_insertion_cascade as P


def base_args(a) -> argparse.Namespace:
    return argparse.Namespace(
        seed=0, check=False, frames=False, episodes=1,
        magnitude=1200.0, ratio=a.ratio, inner_stiffness=3000.0,
        rot_stiffness=400.0, rot_ratio=a.rot_ratio, cc_frac=0.0,
        wrist_inertia=0.005, grip_back=0.040, f_push=12.0, f_limit=30.0,
        v_limit=0.06, standoff_margin=0.035, retarget=True, overdrive=0.010,
        align_s=4.0, insert_s=a.insert_s, duration=a.duration, lateral_mm=0.0,
        search_s=0.0, search_r0=0.001, search_growth=a.growth,
        search_turns=a.turns, search_press=a.press, search_depth=0.003,
        search_catch=0.005, outer="admittance")


def episode(seed: int, err_mm: float, args) -> dict:
    """One attempt with `err_mm` of aim error in a seed-dependent direction."""
    s = P.setup(seed, grip_back=args.grip_back)
    Rh, _ = P.hole_frame(s["u"])
    Ko = P.compose_axial(args.magnitude, args.ratio, Rh)
    rng = np.random.default_rng(1000 + seed)
    d = rng.normal(size=2)
    d /= max(np.linalg.norm(d), 1e-9)
    lat = 1e-3 * err_mm * (d[0] * Rh[:, 1] + d[1] * Rh[:, 2])
    r = P.rollout(s, Ko, lat, args)
    fd = r["face_depth"]
    s["env"].close()
    return dict(ok=bool(r["success"]), caught=bool(r.get("search_caught")),
                deepest=float(fd.max()), final=float(fd[-1]),
                found=r.get("search_found_s"), fmax=float(r["f"].max()),
                fmean=float(np.mean(r["f"][r["f"] > 0.5])) if (r["f"] > 0.5).any() else 0.0)


def cell(mode: str, err: float, a, args) -> dict:
    ins = jam = miss = 0
    found, fmax = [], []
    for seed in range(a.episodes):
        ar = copy.deepcopy(args)
        ar.seed = seed
        ar.search_s = a.search_s if mode == "search" else 0.0
        e = episode(seed, err, ar)
        fmax.append(e["fmax"])
        if e["ok"]:
            ins += 1
        elif e["deepest"] > 0.005:
            jam += 1                      # got in past the face, then stuck
        else:
            miss += 1
        if e["caught"]:
            found.append(e["found"] - ar.align_s)
    return dict(ins=ins, jam=jam, miss=miss, found=found, fmax=fmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default="0,2,4,6,8,10", help="mm of lateral aim error")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--search-s", type=float, default=4.0)
    ap.add_argument("--insert-s", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--growth", type=float, default=0.012)
    ap.add_argument("--turns", type=float, default=5.0)
    ap.add_argument("--press", type=float, default=8.0)
    ap.add_argument("--ratio", type=float, default=10.0, help="axial:lateral stiffness")
    ap.add_argument("--rot-ratio", type=float, default=1.0)
    ap.add_argument("--push", type=int, default=1, help="0 skips the aim-and-push arm")
    a = ap.parse_args()
    args = base_args(a)

    n = a.episodes
    print(f"{n} seeds per cell | spiral {1.0:.0f}..{(0.001 + a.growth) * 1e3:.0f} mm over "
          f"{a.turns:.0f} turns in {a.search_s:.0f} s, {a.press:.0f} N onto the face | "
          f"K axial:lateral {a.ratio:.0f}:1\n")
    print(f"{'aim err':>8} {'| aim-and-push':>22} {'| with search':>22}   {'caught':>8}")
    print(f"{'mm':>8} {'in':>6}{'jam':>6}{'miss':>6}{'Fpk':>5} "
          f"{'in':>6}{'jam':>6}{'miss':>6}{'Fpk':>5}   {'after':>8}")
    for err in [float(x) for x in a.errors.split(",")]:
        row = f"{err:7.0f} "
        for mode in (("push", "search") if a.push else ("search",)):
            c = cell(mode, err, a, args)
            row += (f"{c['ins']:5d}/{n}" if False else f"{c['ins']:6d}{c['jam']:6d}"
                    f"{c['miss']:6d}{np.mean(c['fmax']):5.0f} ")
            if mode == "search":
                row += ("  " + (f"{np.mean(c['found']):5.1f} s  "
                                f"{len(c['found'])}/{n}" if c["found"] else "      none"))
        if not a.push:
            row = f"{err:7.0f} " + " " * 23 + row[8:]
        print(row, flush=True)
    print("\nin = inserted, jam = got past the face but stuck, miss = never found the hole")
    print("Fpk = mean peak |f| in N (the admittance clips the measurement at 30 N)")


if __name__ == "__main__":
    main()
