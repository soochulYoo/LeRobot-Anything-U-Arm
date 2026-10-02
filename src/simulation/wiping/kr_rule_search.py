"""Is the CEILING low, or is the oracle's RULE bad?

Every environment so far says a constant K_R ties or beats the oracle. But the
oracle is one hand-written heuristic -- soften until the pad's own contact
moment carries it through the local swing,

    K_R = f_n * r_pad / (swing - tol)

-- and it is tuned for CONFORMANCE while it is scored on FORCE STABILITY. Its
f_rmse was worse than the best constant in all eight dome cells, which is what a
rule optimising the wrong thing looks like. "No schedule can win" and "this
schedule loses" are different claims and only the second has been shown.

So: stop designing rules and search them. A rule here is a monotone map from the
local surface swing to a stiffness, as five free knots on swing bins:

    swing < 2deg   2-5   5-10   10-20   >20deg
       k0          k1     k2      k3      k4

The oracle is one member of this family. So is EVERY constant (k0 = ... = k4),
which is the point: the constant is nested inside the search space, so if the
best rule found is no better than the best constant, that is a statement about
the ceiling and not about my taste in heuristics.

Held-out seeds matter here. With the constant nested, a rule fitted and scored
on the same episode can never lose, so the fit uses one seed and the comparison
uses others.

    WIPE_PAD_R=0.040 python3 kr_rule_search.py --radius 0.15 --n 10 --chunk 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

EDGES = np.array([2.0, 5.0, 10.0, 20.0])          # deg, bin boundaries


def rule(knots, r_pad, log=None):
    """K_R from the true local swing, as a step function of it."""
    k = np.asarray(knots, dtype=float)

    def f(s):
        last = getattr(s, "last", {}) or {}
        f_n = float(last.get("f_n", 0.0))
        uvh = last.get("contact_uvh")
        if uvh is None or f_n < 0.8:
            return float(k[0])                    # in the air: the flat-case value
        uv = np.asarray(uvh[:2], dtype=float)
        n0 = np.asarray(s.frame.normal_at(uv), dtype=float)
        swing = 0.0
        for d in ((r_pad, 0.0), (-r_pad, 0.0), (0.0, r_pad), (0.0, -r_pad)):
            n1 = np.asarray(s.frame.normal_at(uv + np.asarray(d)), dtype=float)
            c = float(n0 @ n1) / (np.linalg.norm(n0) * np.linalg.norm(n1))
            swing = max(swing, float(np.arccos(np.clip(c, -1.0, 1.0))))
        deg = np.degrees(swing)
        v = float(k[int(np.searchsorted(EDGES, deg))])
        if log is not None:
            log.append((v, deg))
        return v
    return f


def evaluate(knots, radius, seeds, args) -> dict:
    import smoke
    from oracle_kr import force_metrics
    import wipe_scene as SC
    os.environ["WIPE_SURFACE"] = "dome"
    os.environ["WIPE_DOME_R"] = str(radius)
    out = []
    for sd in seeds:
        rng = np.random.default_rng(90_000 + sd)
        th = rng.uniform(0, 2 * np.pi)
        rad = 1e-3 * args.jitter_mm * np.sqrt(rng.uniform())
        r = smoke.run(text=args.text, curved=True, kr=float(knots[0]),
                      letter_height=args.letter, seed=sd, verbose=False,
                      offset_uv=(rad * np.cos(th), rad * np.sin(th)),
                      kr_fn=rule(knots, SC.ERASER_R))
        m = force_metrics(r)
        out.append((m["f_rmse"], r["erased"]))
    return dict(f_rmse=float(np.mean([o[0] for o in out])),
                erased=float(np.mean([o[1] for o in out])))


def sample(rng, n, lo, hi):
    """Random rules, log-uniform.  A quarter are drawn CONSTANT on purpose: the
    comparison needs the nested case sampled as densely as the rest."""
    k = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(n, len(EDGES) + 1)))
    flat = rng.random(n) < 0.25
    k[flat] = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(flat.sum(), 1)))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=10, help="rules in this chunk")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--fit-seed", type=int, default=0)
    ap.add_argument("--k-lo", type=float, default=0.3)
    ap.add_argument("--k-hi", type=float, default=60.0)
    ap.add_argument("--text", default="S")
    ap.add_argument("--letter", type=float, default=0.15)
    ap.add_argument("--jitter-mm", type=float, default=20.0)
    ap.add_argument("--knots", default=None, help="evaluate ONE rule, comma list")
    ap.add_argument("--seeds", default=None, help="evaluate on these seeds")
    args = ap.parse_args()

    import wipe_scene as SC
    pad = SC.ERASER_R
    if args.knots:                                 # stage 2: held-out evaluation
        k = [float(x) for x in args.knots.split(",")]
        seeds = [int(x) for x in (args.seeds or "1,2,3,4").split(",")]
        r = evaluate(k, args.radius, seeds, args)
        print(json.dumps(dict(pad=pad, radius=args.radius, knots=k,
                              seeds=seeds, held_out=True, **r)), flush=True)
        return

    rng = np.random.default_rng(7000 + args.chunk)
    for k in sample(rng, args.n, args.k_lo, args.k_hi):
        r = evaluate(k, args.radius, [args.fit_seed], args)
        print(json.dumps(dict(pad=pad, radius=args.radius,
                              knots=[round(x, 3) for x in k],
                              const=bool(np.ptp(k) < 1e-9), **r)), flush=True)


if __name__ == "__main__":
    main()
