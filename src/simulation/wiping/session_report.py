"""What a human session with interleaved arms actually showed.

    python3 session_report.py demos/human/attempts.jsonl
    python3 session_report.py demos/human/attempts.jsonl --metric erased --ref manual

Reads the rows `protocol.py --arms` wrote and groups them by arm.  It reports three
things, in this order, because that is the order in which they can invalidate each
other:

1. DID THE INTERLEAVING WORK.  The mean session position of each arm, and each arm's
   first half against its second half.  A person's hand improves over a session, so if
   one arm sat late in the session its numbers include that improvement.  If the
   positions are lopsided or an arm drifts hard, nothing below is attributable.

2. THE ARMS.  Per-arm outcomes, and the per-axis compliance split: `xy`/`z` are the
   PERSON's work in every arm and are comparable; `kr` is whoever owned K_R, so it is a
   description of the arm and not a score.  The mechanism a helper claims is that xy and
   z go up once K_R is off the person's hands.  An outcome that moves while xy and z do
   not has some other cause.

3. WHETHER THE DIFFERENCE IS REAL, which with a dozen episodes per arm is usually
   "cannot tell".  A bootstrap interval on the difference says so out loud, and the
   last column says how many episodes per arm the observed spread would need.  A pilot's
   job is to size the real session, not to declare a winner.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

METRICS = ("erased", "in_band", "success", "peak_force", "t")
AXES = ("xy", "z", "kr")


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    kept = [r for r in rows if r.get("arm")]
    if not kept:
        raise SystemExit(f"no rows carry an `arm`: {len(rows)} rows, none from --arms")
    for i, r in enumerate(kept):
        r["_i"] = i                      # session order, which is what order effects need
    return kept


def col(rows, key) -> np.ndarray:
    if key in AXES:
        return np.array([float(r["compliance"].get(f"axis_{key}", np.nan)) for r in rows])
    return np.array([float(r[key]) for r in rows], dtype=float)


def ci(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return (np.nan, np.nan)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def diff_ci(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    da = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    return tuple(np.percentile(da - db, [2.5, 97.5]))


def need(a: np.ndarray, b: np.ndarray) -> float:
    """Episodes per arm to resolve the difference these samples show, at 80% power.

    16 s^2 / d^2 with the pooled variance: the textbook two-sample sizing.  It is an
    estimate OF AN ESTIMATE -- d and s both come from the same few episodes -- so read it
    as an order of magnitude, and read a huge number as "this pilot saw nothing".
    """
    d = abs(a.mean() - b.mean())
    s2 = 0.5 * (a.var(ddof=1) + b.var(ddof=1)) if len(a) > 1 and len(b) > 1 else np.nan
    return np.inf if d < 1e-9 or not np.isfinite(s2) else 16.0 * s2 / d ** 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--metric", default="erased", choices=METRICS)
    ap.add_argument("--ref", default="manual", help="the arm the others are compared to")
    ap.add_argument("--min-compliance", type=float, default=0.0,
                    help="filter AFTER the fact, and only on the axes the person owned")
    args = ap.parse_args()

    rows = load(args.logs)
    if args.min_compliance > 0:
        keep = [r for r in rows
                if min(r["compliance"].get(f"axis_{a}", 1.0) for a in ("xy", "z"))
                >= args.min_compliance]
        print(f"  filtered on the person's own axes >= {args.min_compliance}: "
              f"{len(keep)}/{len(rows)} episodes\n")
        rows = keep
    arms = sorted({r["arm"] for r in rows})
    n_tot = len(rows)
    print(f"  {n_tot} episodes, {len(arms)} arms, "
          f"{len({r['text'] for r in rows})} case(s), "
          f"motion {'/'.join(sorted({r.get('motion', '?') for r in rows}))}\n")

    # ---- 1. did the interleaving work -------------------------------------------
    print("  INTERLEAVING   mean session position (an arm late in the session carries the")
    print("                 operator's own improvement; these should be close to 0.50)")
    m = args.metric
    for a in arms:
        r = [x for x in rows if x["arm"] == a]
        pos = np.mean([x["_i"] / max(1, n_tot - 1) for x in r])
        v = col(r, m)
        h = len(r) // 2
        d = (v[h:].mean() - v[:h].mean()) if h else np.nan
        print(f"    {a:<7} n {len(r):<3} position {pos:.2f}   "
              f"{m} 1st half {v[:h].mean() if h else np.nan:6.3f} -> "
              f"2nd {v[h:].mean():6.3f}  (drift {d:+.3f})")

    # ---- 2. the arms -------------------------------------------------------------
    print(f"\n  ARMS           {'n':>3}  " + "  ".join(f"{k:>10}" for k in METRICS)
          + "   | person's axes   K_R")
    for a in arms:
        r = [x for x in rows if x["arm"] == a]
        cells = "  ".join(f"{col(r, k).mean():10.3f}" for k in METRICS)
        ax = "  ".join(f"{k} {np.nanmean(col(r, k)):.2f}" for k in AXES)
        print(f"    {a:<7}      {len(r):>3}  {cells}   | {ax}")

    # ---- 3. is it real -----------------------------------------------------------
    if args.ref not in arms:
        print(f"\n  (no arm {args.ref!r}: skipping the comparison)")
        return
    ref = col([x for x in rows if x["arm"] == args.ref], m)
    lo, hi = ci(ref, seed=1)
    print(f"\n  {m.upper()} vs {args.ref}   ({args.ref} = {ref.mean():.3f} "
          f"[{lo:.3f}, {hi:.3f}] 95% CI, n {len(ref)})")
    for a in arms:
        if a == args.ref:
            continue
        v = col([x for x in rows if x["arm"] == a], m)
        lo, hi = diff_ci(v, ref, seed=2)
        verdict = ("cannot tell: the interval spans 0" if not (lo > 0 or hi < 0)
                   else "better" if lo > 0 else "worse")
        n_need = need(v, ref)
        print(f"    {a:<7} {v.mean():6.3f}  diff {v.mean() - ref.mean():+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  -> {verdict}"
              + (f";  ~{n_need:.0f}/arm to resolve it" if np.isfinite(n_need)
                 else ";  no difference to resolve"))
    print("\n  A 95% interval spanning 0 after a dozen episodes is the EXPECTED result.")
    print("  The pilot's output is the episode count in the last column, not a winner.")


if __name__ == "__main__":
    main()
