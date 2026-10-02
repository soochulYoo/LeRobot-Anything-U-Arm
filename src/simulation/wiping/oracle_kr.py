"""Is there a ceiling for ADAPTIVE rotational stiffness at all?

Step 1 of the research plan, and the one that decides whether the rest is worth
building.  Everything downstream -- regressing the surface from vision and
force, distilling a latent, any of it -- can only recover part of what an
ORACLE gets.  So measure the oracle first.

The oracle is given the true surface and sets K_R to the stiffness at which the
pad's own contact moment just rotates it through the local curvature:

    theta  = how far the true normal swings across the pad footprint
    K_R    = f_n * r / theta           clipped, and held HIGH off contact

against the best CONSTANT K_R over a sweep.  If the best constant is as good,
there is nothing to adapt to on this surface and no distillation can help --
which is a result, not a failure, and is far cheaper to learn now than after
building the estimator.

    python3 oracle_kr.py --kr 60,30,10,3,1,0.3
"""
from __future__ import annotations

import argparse

import numpy as np

import smoke
import wipe_scene as SC


def force_metrics(r, target=None):
    """What `curved/` measured and t90 cannot see: how steadily the pad holds
    force, and whether it keeps contact at all.  Erased% and t90 saturate --
    every condition gets the glyph off -- so the discrimination, if there is
    any, has to be in the force trace."""
    t, f = r["trace"]["t"], r["trace"]["f"]
    down = f > 0.8
    if not down.any():
        return dict(f_rmse=float("nan"), contact=0.0, f_min=float("nan"), f_cv=float("nan"))
    first = int(np.argmax(down))
    span = slice(first, len(f))                    # landing to the end of the raster
    ref = target if target is not None else float(f[down].mean())
    return dict(f_rmse=float(np.sqrt(np.mean((f[span] - ref) ** 2))),
                contact=float((f[span] > 0.8).mean()),
                f_min=float(f[span].min()),
                f_cv=float(f[down].std() / max(f[down].mean(), 1e-9)))


def oracle(r_pad, k_lo, k_hi, log=None, tol_deg=0.0, smooth=0.0, rate=None, dt=0.002):
    """K_R from the true surface.

    `tol_deg = 0` is the first rule: soften until the pad's own contact moment
    rotates it through the WHOLE local swing.  It conforms best and costs force
    stability, because a wrist that yields everywhere also rocks everywhere.

    `tol_deg > 0` is the rule that rule should have been: conform only the part
    of the swing beyond what can be tolerated, i.e. the STIFFEST wrist that
    still lies flat to within tol.  Stiffer than the first rule everywhere, so
    it should keep more of the force stability while conforming where it counts.

    `smooth` and `rate` exist because the schedule itself is a disturbance: a
    K_R that jumps injects a moment transient, and the energy tank pays for
    every bit of it.
    """
    tol = np.deg2rad(tol_deg)
    state = {"k": k_hi}

    def f(s):
        last = getattr(s, "last", {}) or {}
        f_n = float(last.get("f_n", 0.0))
        uvh = last.get("contact_uvh")
        if uvh is None or f_n < 0.8:
            return k_hi                       # in the air: hold what was commanded
        uv = np.asarray(uvh[:2], dtype=float)
        n0 = np.asarray(s.frame.normal_at(uv), dtype=float)
        swing = 0.0
        for d in ((r_pad, 0.0), (-r_pad, 0.0), (0.0, r_pad), (0.0, -r_pad)):
            n1 = np.asarray(s.frame.normal_at(uv + np.asarray(d)), dtype=float)
            c = float(n0 @ n1) / (np.linalg.norm(n0) * np.linalg.norm(n1))
            swing = max(swing, float(np.arccos(np.clip(c, -1.0, 1.0))))
        need = swing - tol                    # the part that has to be yielded
        if need <= 1e-3:
            k = k_hi                          # flat enough already: stay stiff
        else:
            k = float(np.clip(f_n * r_pad / need, k_lo, k_hi))
        if smooth > 0.0:
            k = (1.0 - smooth) * k + smooth * state["k"]
        if rate is not None:
            k = float(np.clip(k, state["k"] - rate * dt, state["k"] + rate * dt))
        state["k"] = k
        if log is not None:
            log.append((k, float(np.degrees(swing))))
        return k
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kr", default="60,30,10,3,1,0.3")
    p.add_argument("--text", default="S")
    p.add_argument("--board", default="curved")
    p.add_argument("--penetration", type=float, default=0.004)
    p.add_argument("--force", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--letter", type=float, default=0.035, help="glyph height, m")
    p.add_argument("--wrist-inertia", type=float, default=None)
    p.add_argument("--tol", default="0,1,2,3",
                   help="deg of misalignment the oracle is allowed to leave; "
                        "0 reproduces the conform-everything rule")
    p.add_argument("--smooth", type=float, default=0.0, help="EMA on the K_R command")
    p.add_argument("--rate", type=float, default=None, help="Nm/rad/s cap on the command")
    a = p.parse_args()

    krs = [float(x) for x in a.kr.split(",")]
    common = dict(text=a.text, penetration=a.penetration, curved=(a.board == "curved"),
                  seed=a.seed, wrist_inertia=a.wrist_inertia, f_target=a.force,
                  letter_height=a.letter, verbose=False)
    print(f"{a.board} board, glyph {a.text!r}, scripted raster -- only K_R differs\n")
    print("  condition     K_R    erased    t90    mis   yield   f_mean   peak  in-band"
          "   f_rmse  contact   f_min")
    rows = {}
    for kr in krs:
        r = smoke.run(kr=kr, **common)
        rows[f"{kr:g}"] = r
        f = r["trace"]["f"]
        m = force_metrics(r, a.force)
        print(f"  constant   {kr:6g}  {100*r['erased']:5.1f}%  {r['t90']:5.1f}"
              f"  {r['mis_mean']:5.1f}   {r['yld_mean']:5.1f}  {np.mean(f[f > 0.8]):6.2f}"
              f"  {r['peak_force']:5.1f}   {100*r['in_band']:4.0f}%"
              f"   {m['f_rmse']:6.2f}   {100*m['contact']:4.0f}%  {m['f_min']:6.2f}")

    # Does writing g.Kr per step reach the controller at all?  Inject a CONSTANT
    # through the same hook: if it lands on the constant row, the hook is fine
    # and any oracle/constant gap is behaviour, not plumbing.
    probe_kr = min(krs)
    pr = smoke.run(kr=max(krs), kr_fn=lambda s: probe_kr, **common)
    pm = force_metrics(pr, a.force)
    pf = pr["trace"]["f"]
    print(f"  hook-check {probe_kr:6g}  {100*pr['erased']:5.1f}%  {pr['t90']:5.1f}"
          f"  {pr['mis_mean']:5.1f}   {pr['yld_mean']:5.1f}  {np.mean(pf[pf > 0.8]):6.2f}"
          f"  {pr['peak_force']:5.1f}   {100*pr['in_band']:4.0f}%"
          f"   {pm['f_rmse']:6.2f}   {100*pm['contact']:4.0f}%  {pm['f_min']:6.2f}")

    oracles = {}
    for tol in [float(x) for x in a.tol.split(",")]:
        klog = []
        o = smoke.run(kr=max(krs), **common,
                      kr_fn=oracle(SC.ERASER_R, min(krs), max(krs), klog,
                                   tol_deg=tol, smooth=a.smooth, rate=a.rate))
        f = o["trace"]["f"]
        m = force_metrics(o, a.force)
        oracles[tol] = (o, m, klog)
        print(f"  ORACLE t{tol:<4g} adapt  {100*o['erased']:5.1f}%  {o['t90']:5.1f}"
              f"  {o['mis_mean']:5.1f}   {o['yld_mean']:5.1f}  {np.mean(f[f > 0.8]):6.2f}"
              f"  {o['peak_force']:5.1f}   {100*o['in_band']:4.0f}%"
              f"   {m['f_rmse']:6.2f}   {100*m['contact']:4.0f}%  {m['f_min']:6.2f}")
    tol_best = min(oracles, key=lambda t: oracles[t][1]["f_rmse"] - oracles[t][0]["erased"])
    o, m, klog = oracles[tol_best]
    print(f"\n  (the ORACLE row used below is tol = {tol_best:g} deg)")
    if klog:
        k_cmd = np.array([k for k, _ in klog])
        sw = np.array([s_ for _, s_ in klog])
        at_cap = float((k_cmd >= max(krs) - 1e-9).mean())
        print(f"\n what the oracle actually asked for: K_R {k_cmd.min():.2g}"
              f" .. {k_cmd.max():.2g} (median {np.median(k_cmd):.2g}),"
              f" {100*at_cap:.0f}% of steps at the {max(krs):g} cap")
        print(f" normal swing across the pad: {sw.mean():.2f} deg mean,"
              f" {np.percentile(sw, 95):.2f} p95"
              + ("   <- too flat for the pad to have anything to yield to"
                 if np.percentile(sw, 95) < 2.0 else ""))

    best = min(rows, key=lambda k: force_metrics(rows[k], a.force)["f_rmse"])
    b = rows[best]
    bm = force_metrics(b, a.force)
    print(f"\n best constant K_R = {best} (by force rmse {bm['f_rmse']:.2f} N)")
    print(f" oracle vs best constant:  f_rmse {bm['f_rmse']:.2f} -> {m['f_rmse']:.2f} N"
          f"   contact {100*bm['contact']:.0f} -> {100*m['contact']:.0f}%"
          f"   mis {b['mis_mean']:.1f} -> {o['mis_mean']:.1f} deg"
          f"   t90 {b['t90']:.1f} -> {o['t90']:.1f} s")
    same = abs(pr["mis_mean"] - rows[f"{probe_kr:g}"]["mis_mean"]) < 0.3
    print(f" hook-check: a constant {probe_kr:g} through the per-step hook gives"
          f" mis {pr['mis_mean']:.1f} vs {rows[f'{probe_kr:g}']['mis_mean']:.1f} for the same"
          f" value set at construction -> the hook "
          + ("WORKS, so the oracle gap is behaviour" if same else "is NOT reaching the controller"))
    spread = [force_metrics(rows[k], a.force)["f_rmse"] for k in rows]
    print(f" force rmse across the constant sweep: {min(spread):.2f} .. {max(spread):.2f} N"
          + ("   <- flat, so this metric does not discriminate either"
             if max(spread) - min(spread) < 0.15 else ""))
    print(" a ceiling worth chasing only if the oracle column is clearly better;"
          " if not, stop here -- no estimator can beat the oracle it approximates.")


if __name__ == "__main__":
    main()
