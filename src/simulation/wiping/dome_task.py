"""Does a constant rotational stiffness still win on POTTERY-grade curvature?

Four boards of the same shape and different crest radius, from a gentle swell to
something a vase would have.  Two things change as R shrinks and they pull in
opposite directions:

    swing across the pad = 2r/R        grows -> a stiff wrist is punished harder
    sagitta  r^2 / 2R                  grows -> the pad has to deform MORE to lie
                                                flat, and the felt only gives so
                                                much.  Past that the pad cannot
                                                conform at ANY stiffness.

For a 40 mm pad with 4 mm of give that limit is R = 200 mm.  So the interesting
prediction is not "softer is better forever": below 200 mm every stiffness
should degrade together, because the limit stops being the wrist and starts
being the tool.

A uniform dome is also the worst case for ADAPTING: one curvature everywhere
asks for one stiffness everywhere, so the oracle has nothing to switch between.
It is here as a control, not as a contender.

    python3 dome_task.py --radii 0.5,0.25,0.15,0.10
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import smoke
import wipe_scene as SC
from oracle_kr import force_metrics, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="0.50,0.25,0.15,0.10")
    ap.add_argument("--kr", default="60,10,3,1,0.3")
    ap.add_argument("--text", default="S")
    ap.add_argument("--letter", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter-mm", type=float, default=0.0,
                    help="randomise the glyph's place on the dome by up to this "
                         "much, so --seed actually varies the episode")
    a = ap.parse_args()

    r_pad, give = SC.ERASER_R, 0.004
    # The pad's half-width is fixed at IMPORT time from WIPE_PAD_R, so it cannot
    # be an argument here -- and that is exactly how two runs of this script
    # ended up describing different experiments under the same label: one shell
    # had WIPE_PAD_R=0.040 exported and the other did not, giving a 40 mm pad
    # (sagitta 5.3 mm, cannot conform at R=150) against a 15 mm one (0.8 mm,
    # conforms easily).  The size was printed both times and read neither time,
    # so it now goes on every verdict line too.
    rng = np.random.default_rng(90_000 + a.seed)
    th, rad = rng.uniform(0, 2 * np.pi), 1e-3 * a.jitter_mm * np.sqrt(rng.uniform())
    off = (rad * np.cos(th), rad * np.sin(th))
    krs = [float(x) for x in a.kr.split(",")]
    os.environ["WIPE_SURFACE"] = "dome"
    print(f"PAD HALF-WIDTH {1000*r_pad:.0f} mm (WIPE_PAD_R), felt give {1000*give:.0f} mm "
          f"-> a flat pad stops being able to conform below R = "
          f"{1000*r_pad**2/(2*give):.0f} mm\n")
    print("     R   swing  sagitta  conform?    K_R   erased   mis  yield  in-band  f_rmse")
    for R in [float(x) for x in a.radii.split(",")]:
        os.environ["WIPE_DOME_R"] = str(R)
        sw, sag = np.degrees(2 * r_pad / R), r_pad ** 2 / (2 * R)
        ok = "yes" if sag < give else "NO"
        rows = {}
        for i, kr in enumerate(krs):
            r = smoke.run(text=a.text, curved=True, kr=kr, letter_height=a.letter,
                          seed=a.seed, verbose=False, offset_uv=off)
            m = force_metrics(r)
            rows[kr] = (r, m)
            head = (f"{1000*R:6.0f}{sw:7.0f}°{1000*sag:8.1f}{ok:>10}"
                    if i == 0 else " " * 31)
            print(f"{head} {kr:6g}  {100*r['erased']:5.1f}%  {r['mis_mean']:5.1f}"
                  f" {r['yld_mean']:6.1f}   {100*r['in_band']:4.0f}%   {m['f_rmse']:6.2f}")
        o = smoke.run(text=a.text, curved=True, kr=max(krs), letter_height=a.letter,
                      seed=a.seed, verbose=False, offset_uv=off,
                      kr_fn=oracle(r_pad, min(krs), max(krs)))
        mo = force_metrics(o)
        print(f"{' ' * 31} ORACLE  {100*o['erased']:5.1f}%  {o['mis_mean']:5.1f}"
              f" {o['yld_mean']:6.1f}   {100*o['in_band']:4.0f}%   {mo['f_rmse']:6.2f}")
        best = min(rows, key=lambda k: rows[k][1]["f_rmse"])
        bm, be = rows[best][1]["f_rmse"], rows[best][0]["erased"]
        print(f"VERDICT pad {1000*r_pad:.0f}mm R {1000*R:.0f}mm seed {a.seed} "
              f"glyph +({1000*off[0]:+.0f},{1000*off[1]:+.0f})mm conform {ok}: "
              f"best constant {best:g} erased {100*be:.1f}% f_rmse {bm:.2f}"
              f" | oracle {100*o['erased']:.1f}% {mo['f_rmse']:.2f}"
              f" -> {'ORACLE' if mo['f_rmse'] < bm and o['erased'] >= be else 'constant'}"
              f" wins\n")


if __name__ == "__main__":
    main()
