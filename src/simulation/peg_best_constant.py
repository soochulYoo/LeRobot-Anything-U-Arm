"""Per-episode best constant stiffness on peg insertion: the rule-free upper bound.

PEG_DEMO_PLAN.md §10 compared one phase SCHEDULE (3000 -> 400 -> 3000) against
four constants and found the best constant tying or beating it at every
clearance.  Two things limit that result:

  * THE LADDER IS CENSORED AT THE BOTTOM.  400 N/m was the softest level tested
    and it won at 3 mm and 1 mm, with the ranking monotone -- softer is better.
    So the optimum may well lie BELOW the ladder, and "constant 400 ties the
    schedule" may be understating the constant.
  * IT COMPARES ONE HAND-WRITTEN SCHEDULE.  A schedule that loses says nothing
    about schedules in general, only about that one.

This removes both.  The same episodes are run at every stiffness on a ladder
extended DOWNWARD, and two numbers are compared:

    GLOBAL BEST     argmax_k  mean_episodes(success)      <- the baseline
    PER-EPISODE     mean_episodes  max_k(success)         <- the bound

The second is what a controller that knew the episode perfectly would get while
still holding ONE stiffness throughout.  It therefore upper-bounds any schedule
AND any policy, with no rule to design or criticise.  Three readings:

    gap ~ 0                      no schedule can win.  Stop; do not build one.
    gap > 0, argmax tracks the aim error   there IS room, and now a schedule has
                                 a reason to exist and something to condition on.
    gap > 0, argmax tracks nothing         ladder selection noise, not a signal a
                                 policy could collect.

The third is the one a bare "oracle beats constant" number cannot tell from the
second.  On the writing task the same method gave gap +0.000 on every task
metric with 0/24 episodes preferring a different stiffness, which closed the
question there.

One caveat to carry, from PEG_DEMO_PLAN.md's own note on the 1 mm block: at
1 mm the zero-aim-error cell is only 7/8, and that miss is the sag calibration's
residual, which is the size of the clearance itself.  Part of the 1 mm headroom
is this executor's precision rather than the task.  The 2 mm block is clean.

ONE STIFFNESS PER PROCESS, because SAPIEN will not survive more.  `episode()`
builds and closes an env every call, and 160 of those in one process
SEGFAULTS (core dumped, no Python traceback -- peg_bc_0.log).  The proven
footprint is peg_step4_clearance.sbatch's: 32 episodes per task, 10 tasks.  So
each task runs one (stiffness, clearance) cell and writes its own rows, and a
second pass aggregates them.

    sbatch peg_best_constant.sbatch                 # 10 tasks x 32 episodes
    python3 peg_best_constant.py --aggregate "peg_bc_k*_cl*.json"
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import peg_insertion_case1 as C1


# THE SEARCH HAS TO BE RESCALED WITH THE CLEARANCE, or a tight gap fails for a
# scripted reason rather than a physical one.  The head drops in only if it
# passes within `clearance` of the hole centre, so a spiral pitch wider than
# that capture window steps straight over the hole, and a fixed search time
# covers fewer turns as the pitch tightens.  These are peg_step4_clearance.sbatch's
# own numbers, so this sweep is comparable with the table in PEG_DEMO_PLAN.md
# section 10 rather than a differently-tuned script:
#     2 mm -> turns 6,  search_s 5,  duration 20
#     1 mm -> turns 12, search_s 10, duration 25
# turns is growth / clearance (0.012 m / 0.002 m = 6); for any other clearance
# that formula is used and search_s / duration are scaled off the 2 mm row.
SEARCH = {2.0: (6.0, 5.0, 20.0), 1.0: (12.0, 10.0, 25.0)}


def search_for(a, clearance_mm):
    if clearance_mm in SEARCH:
        return SEARCH[clearance_mm]
    turns = max(1.0, a.search_growth / (1e-3 * clearance_mm))
    return turns, 5.0 * (2.0 / clearance_mm), 20.0 + 5.0 * (2.0 / clearance_mm - 1.0)


def run_cell(base, k_axial, clearance_mm, err_mm, seed):
    """One episode at one stiffness.  `--no-schedule` so the level is held."""
    a = copy.deepcopy(base)
    a.no_schedule = True
    a.k_axial_search = float(k_axial)
    a.clearance_mm = float(clearance_mm)
    a.search_turns, a.search_s, a.duration = search_for(a, float(clearance_mm))
    C1.resolve(a)
    r = C1.episode(seed, err_mm, a)
    return dict(success=bool(r["success"]), caught=bool(r["caught"]),
                deepest_mm=1000.0 * float(r["deepest"]),
                clearance_mm=1000.0 * float(r["clearance"]))


def analyse(rows, ladder, cells):
    """Global-best vs per-episode-best over `cells` = [(clearance, err, seed)]."""
    M = np.full((len(ladder), len(cells)), np.nan)
    for r in rows:
        M[ladder.index(r["k"]), cells.index((r["cl"], r["err"], r["seed"]))] = r["success"]
    per_k = np.nanmean(M, axis=1)
    gi = int(np.nanargmax(per_k))
    best_per_cell = np.nanmax(M, axis=0)
    arg = [ladder[int(i)] for i in np.nanargmax(M, axis=0)]
    return dict(per_k=per_k.tolist(), global_k=ladder[gi], global_best=float(per_k[gi]),
                per_episode=float(np.nanmean(best_per_cell)),
                gap=float(np.nanmean(best_per_cell) - per_k[gi]),
                argmax=arg, n_differ=int(sum(a != ladder[gi] for a in arg)))


def report(rows, ladder, cls, cells) -> None:
    print(f"\n  {'k_axial':>8}" + "".join(f"{c:>6.0f}mm" for c in cls) + "   overall")
    for k in ladder:
        line = f"  {k:8.0f}"
        for c in cls:
            rs = [r for r in rows if r["k"] == k and r["cl"] == c]
            line += f"{100 * np.mean([r['success'] for r in rs]):7.1f}" if rs else "     --"
        rs = [r for r in rows if r["k"] == k]
        print(line + f"{100 * np.mean([r['success'] for r in rs]):9.1f}%")

    print(f"\n  {'scope':<12}{'best k':>8}{'global':>9}{'per-ep':>9}{'gap':>8}"
          f"{'cells differing':>17}   argmax vs aim error")
    for scope, sel in ([("all", cells)] if len(cls) > 1 else []) + \
                      [(f"{c:.0f} mm", [x for x in cells if x[0] == c]) for c in cls]:
        rs = [r for r in rows if (r["cl"], r["err"], r["seed"]) in sel]
        a = analyse(rs, ladder, sel)
        am = np.array([float(x) for x in a["argmax"]])
        ev = np.array([x[1] for x in sel], dtype=float)
        cor = (f"{np.corrcoef(np.log(am), ev)[0, 1]:+.2f}"
               if np.std(am) > 1e-9 and np.std(ev) > 1e-9 else "n/a")
        print(f"  {scope:<12}{a['global_k']:8.0f}{100 * a['global_best']:8.1f}%"
              f"{100 * a['per_episode']:8.1f}%{100 * a['gap']:+8.1f}"
              f"{a['n_differ']:>12}/{len(sel):<4}   r = {cor}")
    print("\n  gap ~ 0            no schedule can win -- do not build one")
    print("  gap > 0, |r| high  there is room, and a schedule has something to condition on")
    print("  gap > 0, |r| ~ 0   ladder selection noise, not a signal a policy can collect")


def main():
    ap = argparse.ArgumentParser()
    C1.add_args(ap)
    ap.add_argument("--ladder", type=float, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--clearances", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="peg_best_constant.json")
    ap.add_argument("--aggregate", default=None,
                    help="glob of per-cell JSONs to pool and analyse instead of running")
    args = ap.parse_args()

    if args.aggregate:
        import glob as _g
        files = sorted(_g.glob(args.aggregate))
        if not files:
            raise SystemExit(f"no files match {args.aggregate}")
        rows = [r for f in files for r in json.loads(pathlib.Path(f).read_text())["rows"]]
        ladder = sorted({r["k"] for r in rows})
        cls = sorted({r["cl"] for r in rows})
        cells = sorted({(r["cl"], r["err"], r["seed"]) for r in rows})
        missing = len(ladder) * len(cells) - len(rows)
        print(f"{len(files)} files, {len(rows)} episodes, "
              f"{len(ladder)} stiffnesses x {len(cells)} cells"
              + (f"  -- {missing} MISSING, analysis is on what exists" if missing else ""))
        report(rows, ladder, cls, cells)
        return

    ladder = [float(k) for k in args.ladder]
    errs = [float(e) for e in args.errors.split(",")]
    seeds = list(range(args.seeds))
    cells = [(c, e, s) for c in args.clearances for e in errs for s in seeds]
    print(f"{len(ladder)} stiffnesses x {len(cells)} episodes = {len(ladder) * len(cells)} runs")
    print(f"  ladder {ladder}\n  clearances {args.clearances} mm, aim errors {errs} mm, "
          f"{len(seeds)} seeds")

    t0 = time.time()
    rows = []
    for k in ladder:
        for (cl, err, sd) in cells:
            r = run_cell(args, k, cl, err, sd)
            r.update(k=k, cl=cl, err=err, seed=sd)
            rows.append(r)
            if len(rows) % 20 == 0:
                print(f"  {len(rows)}/{len(ladder) * len(cells)}  "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(
        dict(config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list))},
             rows=rows), indent=1))
    print(f"  wrote {args.out} ({len(rows)} episodes, {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
