"""The act variants over their seeds: closed loop first, open loop to explain it.

CLOSED LOOP IS THE RANKING.  It is the only number that asks what the policy is
for -- how much of the glyph came off an unseen board, inside the force band,
without tearing it.  `rollout.py` writes one results.json per seed and this
pools them.

Open loop is here to explain a ranking, not to set it.  Two variants can sit
0.05 mm apart on held-out chunks and land differently in the loop, most often
by choosing a rotational stiffness that does not let the pad lie on a surface
whose normal turns.  Which is why both tables carry K_R: `acc_kr` is whether
the policy predicted the demonstrated level, `K_R wiping` is what it actually
held while the pad was moving in contact, against what the demonstrations held
there.  A policy that nails position and misses K_R has learned the motions of
wiping without the compliance that makes it work.

    python3 compare_runs.py                 # runs/*/seed*/{train_log,eval/results}.json
    python3 compare_runs.py --root runs_krlow
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

COLS = [("pos1_mm", "pos1 mm", 1.0, 2), ("posH_mm", "posH mm", 1.0, 2),
        ("logk_err", "logK err", 1.0, 3), ("logkr_err", "logKR err", 1.0, 3),
        ("acc_xy", "acc xy %", 100.0, 1), ("acc_z", "acc z %", 100.0, 1),
        ("acc_kr", "acc K_R %", 100.0, 1), ("ft_N", "ft N", 1.0, 2),
        ("val_loss", "val loss", 1.0, 3)]


def mean_std(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


EVAL = [("success", "success %", 100.0, 0), ("erased", "erased %", 100.0, 1),
        ("in_band", "in band %", 100.0, 1), ("peak_force", "peak N", 1.0, 1),
        ("wipe_kr", "K_R wiping", 1.0, 2), ("approach_kr", "K_R appr", 1.0, 1),
        ("wipe_left", "mis/ask", 1.0, 2)]


def eval_rows(d: pathlib.Path):
    """One dict per seed: the episode means of its closed-loop run."""
    out, ref = [], {}
    for s in sorted(d.glob("seed*/eval/results.json")):
        try:
            r = json.loads(s.read_text())
        except json.JSONDecodeError:
            continue
        eps = r.get("episodes") or []
        if not eps:
            continue
        ref = r.get("demo_reference") or ref
        out.append({k: mean_std([e.get(k) for e in eps])[0] for k, _, _, _ in EVAL})
    return out, ref


def table(rows, cols, title, sort, reverse):
    if not rows:
        return
    rows = [r for r in rows if not math.isnan(r[2].get(sort, (float("nan"),))[0])] or rows
    rows.sort(key=lambda r: -r[2][sort][0] if reverse else r[2][sort][0])
    head = f"  {'variant':18s} {'n':>2s}  " + "  ".join(f"{t:>13s}" for _, t, _, _ in cols)
    print("\n" + title)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, n, st in rows:
        line = f"  {name:18s} {n:2d}  "
        for k, _, scale, prec in cols:
            m, s = st.get(k, (float("nan"), float("nan")))
            line += ("  " + (f"{'-':>13}" if math.isnan(m)
                             else f"{scale * m:>7.{prec}f}+-{scale * s:<5.{prec}f}"))
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--sort", default="acc_kr", help="column to rank the OPEN-loop table on")
    a = ap.parse_args()
    root = pathlib.Path(a.root)

    closed, ref = [], {}
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        rows, r = eval_rows(d)
        ref = r or ref
        if rows:
            closed.append((d.name, len(rows),
                           {k: mean_std([x[k] for x in rows]) for k, _, _, _ in EVAL}))
    table(closed, EVAL, "CLOSED LOOP -- unseen boards, ranked on success", "success", True)
    if ref and closed:
        first = next(iter(ref.values()))
        d = {k: float(np.mean([v[k] for v in ref.values()]))
             for k in first if isinstance(first[k], (int, float))}
        cells = []
        for k, _, sc, pr in EVAL:
            cells.append(f"{'-':>13}" if k not in d else f"{sc * d[k]:>7.{pr}f}{'':<6}")
        print(f"  {'the demonstrations':18s}  " + "  " + "  ".join(cells)
              + "     <- measured the same way")

    rows = []
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        last, steps, mins = [], [], []
        for s in sorted(d.glob("seed*/train_log.json")):
            try:
                log = json.loads(s.read_text())
            except json.JSONDecodeError:         # a run still writing its log
                continue
            if log:
                last.append(log[-1])
                steps.append(log[-1].get("step", 0))
                mins.append(log[-1].get("minutes", float("nan")))
        if last:
            rows.append((d.name, len(last), int(min(steps)), mean_std(mins)[0],
                         {k: mean_std([r.get(k) for r in last]) for k, _, _, _ in COLS}))
    if not rows:
        raise SystemExit(f"no train_log.json under {root}/*/seed*/")

    key = lambda r: (-r[4][a.sort][0] if a.sort.startswith("acc") else r[4][a.sort][0])
    rows.sort(key=lambda r: (math.inf if math.isnan(key(r)) else key(r)))

    head = f"  {'variant':18s} {'n':>2s} {'steps':>6s} {'min':>5s}  " + "  ".join(
        f"{t:>13s}" for _, t, _, _ in COLS)
    print("\nOPEN LOOP -- held-out demonstrations")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, n, steps, mins, st in rows:
        line = f"  {name:18s} {n:2d} {steps:6d} {mins:5.0f}  "
        for k, _, scale, prec in COLS:
            m, s = st[k]
            line += ("  " + (f"{'-':>13}" if math.isnan(m)
                             else f"{scale * m:>7.{prec}f}+-{scale * s:<5.{prec}f}"))
        print(line)
    print(f"\n  mean +- sd over seeds, at the last eval of each run; ranked on {a.sort}")


if __name__ == "__main__":
    main()
