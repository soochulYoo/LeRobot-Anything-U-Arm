"""The upper bound with no rule in it.

Every "oracle" so far has been a heuristic, so every negative result has been
open to the same objection: maybe the rule is bad, not the ceiling low. This
removes the rule. For each episode, sweep K_R and keep that episode's BEST
constant. Then compare

    mean over episodes of (that episode's best constant)      <- upper bound
    the single global constant that is best on average        <- what we ship

The first is unreachable by any policy that must infer the stiffness, because it
is handed the answer per episode. It is also a LOWER bound on what within-episode
scheduling could do, since it never changes K during the episode. So:

  gap ~ 0   the hidden variables (board curvature, where the glyph sits) do not
            ask for different stiffnesses. Surface estimation, RMA, a stiffness
            copilot -- none of them can win, and the oracle rule is exonerated.
  gap > 0   there IS room between conditions, and the hand-written oracle was
            failing to collect it.

    WIPE_PAD_R=0.040 python3 kr_per_episode.py --radius 0.15 --seed 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kr", default="60,30,10,3,1,0.3")
    ap.add_argument("--text", default="S")
    ap.add_argument("--letter", type=float, default=0.15)
    ap.add_argument("--jitter-mm", type=float, default=20.0)
    a = ap.parse_args()

    import smoke
    import wipe_scene as SC
    from oracle_kr import force_metrics
    os.environ["WIPE_SURFACE"] = "dome"
    os.environ["WIPE_DOME_R"] = str(a.radius)

    rng = np.random.default_rng(90_000 + a.seed)
    th = rng.uniform(0, 2 * np.pi)
    rad = 1e-3 * a.jitter_mm * np.sqrt(rng.uniform())
    off = (rad * np.cos(th), rad * np.sin(th))

    for kr in [float(x) for x in a.kr.split(",")]:
        r = smoke.run(text=a.text, curved=True, kr=kr, letter_height=a.letter,
                      seed=a.seed, verbose=False, offset_uv=off)
        m = force_metrics(r)
        print(json.dumps(dict(pad=SC.ERASER_R, radius=a.radius, seed=a.seed,
                              kr=kr, f_rmse=m["f_rmse"], erased=float(r["erased"]),
                              mis=float(r["mis_mean"]),
                              off_mm=[round(1e3 * x, 1) for x in off])), flush=True)


if __name__ == "__main__":
    main()
