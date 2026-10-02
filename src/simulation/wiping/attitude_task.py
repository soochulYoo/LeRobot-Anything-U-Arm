"""Grinding-shaped wiping: the two bands want OPPOSITE things from the wrist.

Curvature alone did not justify adapting K_R -- a low constant was as good as
the oracle on every board we built, because a soft wrist costs nothing on a flat
stretch.  The senior researcher's point is the way out: find where a soft wrist
is not free.

In grinding it is not.  Two requirements live on the same workpiece:

  CONFORM band (curved).  The tool must lie on the surface.  An edge-loaded pad
  puts its whole force through a sliver of contact, which is how a grinder burns
  or gouges.  Error = misalignment against the TRUE LOCAL NORMAL  (`mis`).

  HOLD band (flat, a feature).  The tool must keep the attitude it was given --
  a chamfer has an angle, an edge has to stay sharp, and a wrist that tips
  rounds it off.  Error = departure from the COMMANDED attitude  (`yld`).

A constant K_R cannot serve both, and not because of a metric invented to make
it fail: the two errors are already logged, and the K_R sweep already moves them
in opposite directions (K_R 10 -> 0.3 takes `mis` 5.1 -> 4.0 while `yld` goes
1.3 -> 6.5).  The open question is whether a schedule that switches between the
bands beats every constant, or whether the cost of switching eats the gain --
which is what the curvature experiments taught us to check.

    python3 attitude_task.py --kr 60,10,3,1,0.3 --letter 0.15
"""
from __future__ import annotations

import argparse

import numpy as np

import smoke
import wipe_scene as SC
from patchy_surface import PatchySurface


def bands(u, surf):
    """Which requirement is in force at each sample: 1 conform, 0 hold."""
    w = surf.band_at(np.asarray(u))
    return w > 0.5


def defects(r, surf, tol_mis=3.0, tol_yld=3.0):
    """Per-band violations, in the units each band is specified in."""
    t = r["trace"]
    down = (t["f"] > 0.8) & np.isfinite(t["u"])
    if not down.any():
        return dict(conform=float("nan"), hold=float("nan"), score=float("nan"),
                    frac_conform=float("nan"))
    cf = bands(t["u"][down], surf)
    mis, yld = t["mis"][down], t["yld"][down]
    # a violation is the amount by which the band's own tolerance is exceeded
    v_conform = np.maximum(mis[cf] - tol_mis, 0.0) if cf.any() else np.zeros(1)
    v_hold = np.maximum(yld[~cf] - tol_yld, 0.0) if (~cf).any() else np.zeros(1)
    return dict(conform=float(v_conform.mean()), hold=float(v_hold.mean()),
                score=float(v_conform.mean() + v_hold.mean()),
                frac_conform=float(cf.mean()))


def switching_oracle(surf, r_pad, k_lo, k_hi, log=None, lead=0.0):
    """What each band asks for, with the true surface: soft enough to lie on the
    curve where conformance is required, stiff where the attitude is.

    `lead` is the distance AHEAD along the direction of travel that the decision
    is made at.  With lead = 0 the schedule reacts to the band the pad is already
    in, and K_R -- which is integrated state, not a parameter -- arrives late: the
    first measurement of this task showed the schedule losing to a constant while
    cleanly removing the attitude error it was aimed at, because the conformance
    error it caused on entry to the curved band was larger.  Looking ahead is the
    cheap test of whether that lateness is the whole of the cost.
    """
    state = {"uv": None, "dir": np.zeros(2)}

    def f(s):
        last = getattr(s, "last", {}) or {}
        f_n = float(last.get("f_n", 0.0))
        uvh = last.get("contact_uvh")
        if uvh is None or f_n < 0.8:
            return k_hi
        uv_now = np.asarray(uvh[:2], float)
        if state["uv"] is not None:
            d = uv_now - state["uv"]
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                state["dir"] = d / n
        state["uv"] = uv_now
        look = uv_now + lead * state["dir"]       # decide for where it is GOING
        u = float(look[0])
        if surf.band_at(u) <= 0.5:
            k = k_hi                                   # HOLD band: keep attitude
        else:
            uv = look
            n0 = np.asarray(s.frame.normal_at(uv), float)
            sw = 0.0
            for d in ((r_pad, 0.0), (-r_pad, 0.0), (0.0, r_pad), (0.0, -r_pad)):
                n1 = np.asarray(s.frame.normal_at(uv + np.asarray(d)), float)
                sw = max(sw, float(np.arccos(np.clip(
                    n0 @ n1 / (np.linalg.norm(n0) * np.linalg.norm(n1)), -1, 1))))
            k = k_hi if sw < 1e-3 else float(np.clip(f_n * r_pad / sw, k_lo, k_hi))
        if log is not None:
            log.append((k, float(s.ctl.kr)))      # commanded vs achieved
        return k
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kr", default="60,10,3,1,0.3")
    ap.add_argument("--text", default="S")
    ap.add_argument("--letter", type=float, default=0.15)
    ap.add_argument("--tol-mis", type=float, default=3.0)
    ap.add_argument("--tol-yld", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lead", default="0,0.02,0.05,0.09",
                   help="m of lookahead along the direction of travel")
    a = ap.parse_args()

    surf = PatchySurface(**SC.CurvedWipingEnv.SURF_PATCHY)
    krs = [float(x) for x in a.kr.split(",")]
    common = dict(text=a.text, curved=True, seed=a.seed, letter_height=a.letter,
                  verbose=False)
    print(f"tolerances: conform {a.tol_mis:g} deg of misalignment, "
          f"hold {a.tol_yld:g} deg of attitude drift\n")
    print("  condition     erased   mis   yld   conform   hold   DEFECT   in-band")
    rows = {}
    for kr in krs:
        r = smoke.run(kr=kr, **common)
        d = defects(r, surf, a.tol_mis, a.tol_yld)
        rows[kr] = (r, d)
        print(f"  const {kr:6g}  {100*r['erased']:5.1f}%  {r['mis_mean']:5.1f}"
              f" {r['yld_mean']:5.1f}   {d['conform']:6.2f} {d['hold']:6.2f}"
              f"   {d['score']:6.2f}    {100*r['in_band']:4.0f}%")

    best_o = None
    for lead in [float(x) for x in a.lead.split(",")]:
        log = []
        o = smoke.run(kr=max(krs), **common,
                      kr_fn=switching_oracle(surf, SC.ERASER_R, min(krs), max(krs),
                                             log, lead=lead))
        d = defects(o, surf, a.tol_mis, a.tol_yld)
        print(f"  ORACLE +{1000*lead:3.0f}mm {100*o['erased']:5.1f}%  {o['mis_mean']:5.1f}"
              f" {o['yld_mean']:5.1f}   {d['conform']:6.2f} {d['hold']:6.2f}"
              f"   {d['score']:6.2f}    {100*o['in_band']:4.0f}%")
        if log:
            cmd = np.array([c for c, _ in log])
            got = np.array([g for _, g in log])
            lag = np.abs(np.log10(np.maximum(cmd, 1e-3)) - np.log10(np.maximum(got, 1e-3)))
            print(f"      commanded {cmd.min():.2g}..{cmd.max():.2g}; the wrist actually"
                  f" held {got.min():.2g}..{got.max():.2g}, off by {lag.mean():.2f} decades"
                  f" on average")
        if best_o is None or d["score"] < best_o[1]["score"]:
            best_o = (o, d, lead)
    o, d, lead_best = best_o

    best = min(rows, key=lambda k: rows[k][1]["score"])
    bs = rows[best][1]["score"]
    print(f"\n  conform = mean degrees over the {100*d['frac_conform']:.0f}% of contact"
          f" time in curved bands that exceed {a.tol_mis:g} deg of misalignment")
    print(f"  hold    = the same for attitude drift on the flat bands")
    print(f"\n best constant K_R = {best:g}, defect {bs:.2f};"
          f"  best oracle {d['score']:.2f} at {1000*lead_best:.0f} mm of lookahead")
    if bs > 1e-9:
        print(f" the schedule removes {100*(bs - d['score'])/bs:.0f}% of the defect"
              " a constant cannot avoid")
    if log:
        k = np.asarray(log)
        print(f" oracle commanded K_R {k.min():.2g}..{k.max():.2g}"
              f" (median {np.median(k):.2g})")


if __name__ == "__main__":
    main()
