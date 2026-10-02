"""The per-episode best constant: an upper bound on any variable-stiffness policy,
with no oracle RULE to criticise.

Why this exists.  Across writing, wiping (flat / patchy / attitude / dome) and peg
insertion, a well-chosen CONSTANT stiffness has matched or beaten the oracle
schedule.  But "variable stiffness cannot win" and "that particular oracle rule is
bad" are different claims, and a hand-written oracle (`tol_deg = 0`, soft enough to
follow the whole swing) cannot separate them -- its own comments say it trades
force stability away, and its f_rmse being worse everywhere is that symptom.

So remove the rule.  Run the SAME episodes at every stiffness on a ladder and ask:

    GLOBAL BEST     argmax over k of (mean over episodes)      <- the baseline
    PER-EPISODE     mean over episodes of (max over k)         <- the bound

The second is what a policy that knew the episode perfectly would get while still
holding ONE stiffness for the whole episode.  It is therefore an upper bound on
any within-episode schedule too, and it involves no rule at all.  Three outcomes:

  * the two are equal        -> the hidden variation does not demand different
                               stiffness.  The question is closed and every
                               oracle rule is exonerated at once.
  * there is a gap, and the per-episode argmax CORRELATES with a hidden variable
                            -> variable stiffness has room, and the rules tried so
                               far were the problem.
  * there is a gap, and the argmax correlates with nothing
                            -> the gap is selection noise over the ladder, not a
                               learnable signal.  No policy can collect it.

The third case is the one a bare "oracle beats constant" number cannot tell from
the second, which is why the correlations below are reported next to the gap.

Only k_n (into the paper) is swept; k_t is held, so one axis is isolated.  Tilt is
the hidden variable that makes k_n matter at all: over a 37 mm glyph the surface
height ranges w tan(theta), so at k_n the force swings by k_n w tan(theta) about
the commanded press, and the sizing rule caps k_n at (f - F_lo)/delta.  Run it at
a WIDE tilt range (--tilt-deg 15 draws per-episode tilts of 0-21 deg) so that if
any setting makes the per-episode optimum move, this one does.

    python3 policy/best_constant.py --episodes 16 --tilt-deg 15 --workers 12
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse        # noqa: E402
import json            # noqa: E402
import pathlib         # noqa: E402
import sys             # noqa: E402
import time            # noqa: E402
import warnings        # noqa: E402

import numpy as np     # noqa: E402

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# the metrics the analysis is run on; +1 means bigger is better
METRICS = {"success": +1, "coverage": +1, "in_band": +1, "precision": +1}

_ENV = _CFG = None


def _init(cfg):
    global _ENV, _CFG
    from evaluate import WritingPolicyEnv
    # no cameras: the scripted writer is privileged in the strokes and finds the
    # paper by feel, so vision changes nothing and costs most of the wall clock
    _ENV = WritingPolicyEnv(control_hz=10.0, cameras=False)
    _CFG = cfg


def _job(job):
    import dataclasses

    import evaluate as EV
    import sim as SM
    k_n, seed = job
    spec = SM.TaskSpec.sample(seed, dz=_CFG["dz"], tilt_deg=_CFG["tilt_deg"],
                              mu=tuple(_CFG["mu"]))
    spec = dataclasses.replace(spec, text=_CFG["text"])
    pol = EV.ScriptedPolicy(_ENV, k_t=_CFG["k_t"], k_n=k_n, k_travel=_CFG["k_travel"],
                            f_target=_CFG["f_target"], spring=_CFG["spring"])
    res = EV.run_episode(_ENV, pol, spec)
    sim = _ENV.sim
    # the hidden variables, so the per-episode argmax can be correlated with them
    tilt = np.asarray(sim.frame.tilt, dtype=float).reshape(-1)
    out = {k: float(res[k]) for k in METRICS}
    out.update(k_n=float(k_n), seed=int(seed), success=float(res["success"]),
               contact_force=float(res["contact_force"]),
               peak_force=float(res["peak_force"]),
               tilt_deg=float(np.degrees(np.linalg.norm(tilt))),
               friction=float(spec.friction), dz_mm=1000.0 * float(spec.canvas_dz),
               letter_h=float(spec.letter_height))
    return out


def analyse(rows, ladder, seeds, metric, sign) -> dict:
    """Global-best vs per-episode-best for one metric."""
    M = np.full((len(ladder), len(seeds)), np.nan)
    for r in rows:
        M[ladder.index(r["k_n"]), seeds.index(r["seed"])] = r[metric]
    per_k = np.nanmean(M, axis=1)                       # mean over episodes, per k
    gi = int(np.nanargmax(sign * per_k))
    global_best, global_k = per_k[gi], ladder[gi]
    best_per_ep = np.nanmax(sign * M, axis=0) * sign    # max over k, per episode
    arg_per_ep = [ladder[int(i)] for i in np.nanargmax(sign * M, axis=0)]
    return dict(metric=metric, per_k=per_k.tolist(), global_k=global_k,
                global_best=float(global_best),
                per_episode=float(np.nanmean(best_per_ep)),
                gap=float(np.nanmean(best_per_ep) - global_best),
                argmax=arg_per_ep,
                n_differ=int(sum(a != global_k for a in arg_per_ep)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=float, nargs="+",
                    default=[200, 400, 700, 1200, 2000, 3500], help="k_n values, N/m")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=30_000)
    ap.add_argument("--text", default="S")
    ap.add_argument("--tilt-deg", type=float, default=15.0,
                    help="the hidden variable that makes k_n matter (demos: 5)")
    ap.add_argument("--dz", type=float, default=0.004)
    ap.add_argument("--mu", type=float, nargs=2, default=[0.2, 0.5])
    ap.add_argument("--k-t", type=float, default=2000.0, help="held, so one axis is isolated")
    ap.add_argument("--k-travel", type=float, default=800.0)
    ap.add_argument("--f-target", type=float, default=3.0)
    ap.add_argument("--spring", action="store_true", help="the spring-consistent layout")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ladder = [float(k) for k in args.ladder]
    seeds = [args.seed_offset + i for i in range(args.episodes)]
    cfg = dict(dz=args.dz, tilt_deg=args.tilt_deg, mu=args.mu, text=args.text,
               k_t=args.k_t, k_travel=args.k_travel, f_target=args.f_target,
               spring=bool(args.spring))
    jobs = [(k, s) for k in ladder for s in seeds]

    import multiprocessing as mp
    t0 = time.time()
    print(f"{len(ladder)} stiffnesses x {len(seeds)} episodes = {len(jobs)} runs, "
          f"text {args.text!r}, tilt +-{args.tilt_deg} deg, k_t {args.k_t:.0f} N/m, "
          f"layout {'spring' if args.spring else 'legacy'}")
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        rows = []
        for r in pool.imap_unordered(_job, jobs):
            rows.append(r)
            if len(rows) % 12 == 0:
                print(f"  {len(rows)}/{len(jobs)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # ---- the table, so the shape of the sweep is visible before the verdict ----
    print(f"\n  {'k_n':>6}" + "".join(f"{m:>12}" for m in METRICS) + f"{'contact N':>11}{'peak N':>8}")
    for k in ladder:
        rs = [r for r in rows if r["k_n"] == k]
        print(f"  {k:6.0f}" + "".join(f"{np.mean([r[m] for r in rs]):12.3f}" for m in METRICS)
              + f"{np.mean([r['contact_force'] for r in rs]):11.2f}"
              + f"{np.mean([r['peak_force'] for r in rs]):8.2f}")

    hidden = {h: np.array([r[h] for r in rows if r["k_n"] == ladder[0]])
              for h in ("tilt_deg", "friction", "dz_mm", "letter_h")}
    print(f"\n  GLOBAL BEST CONSTANT vs PER-EPISODE BEST CONSTANT")
    print(f"  {'metric':<11}{'best k_n':>9}{'global':>9}{'per-ep':>9}{'gap':>8}"
          f"{'eps differing':>15}   argmax correlation with the hidden variables")
    report = {}
    for m, sign in METRICS.items():
        a = analyse(rows, ladder, seeds, m, sign)
        report[m] = a
        am = np.array([float(x) for x in a["argmax"]])
        cors = []
        for h, v in hidden.items():
            if np.std(am) < 1e-9 or np.std(v) < 1e-9:
                cors.append(f"{h} n/a")
            else:
                cors.append(f"{h} {np.corrcoef(np.log(am), v)[0, 1]:+.2f}")
        print(f"  {m:<11}{a['global_k']:9.0f}{a['global_best']:9.3f}{a['per_episode']:9.3f}"
              f"{a['gap']:+8.3f}{a['n_differ']:>10}/{len(seeds):<4}   " + "  ".join(cors))

    print("\n  Read it this way:")
    print("    gap ~ 0                      the hidden variation does not demand a "
          "different stiffness; every oracle rule is exonerated at once")
    print("    gap > 0 and |corr| high      variable stiffness has room and the rules "
          "tried so far were the problem")
    print("    gap > 0 and |corr| ~ 0       the gap is ladder selection noise, not a "
          "learnable signal -- no policy can collect it")

    out = pathlib.Path(args.out or HERE / "best_constant.json")
    out.write_text(json.dumps(dict(config=vars(args), rows=rows, report=report), indent=1))
    print(f"\n  wrote {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
