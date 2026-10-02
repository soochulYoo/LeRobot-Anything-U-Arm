"""What rotational stiffness buys ON THE TASK, flat board against curved.

`curved_probe.py` measures K_R against pad alignment with no glyph and no
erasing, which is how it could be trusted to isolate the rotational loop.  This
asks the question the task cares about: does lying flush get the glyph off
faster, and does it hold the force band while doing it.

The scripted raster of `smoke.py` drives every condition, so the path, the
speed, the penetration and the translational stiffness are identical and only
K_R and the board change.

K_R = 60 vs 0.3, not 60 vs 3: a pad of half-width r pressing with f generates
at most f*r of contact moment, which for this 30 mm pad at ~3 N is 0.05 Nm, so
anything above ~1 Nm/rad is rigid and 60 vs 3 compares rigid with rigid.  That
mistake is what made the first comparison here read as a null result; see
CURVED_BOARD.md.

`--erase-work` raises the work each mark needs, which is what keeps the test
from saturating: with the default 0.040 N*m the raster covers the glyph twice
over and both conditions finish at 100%, so the difference can only show up in
the timing.

    python3 kr_task.py                        # both boards, K_R 60 vs 0.3
    python3 kr_task.py --erase-work 0.12      # make the glyph hard to shift
"""
from __future__ import annotations

import argparse

import numpy as np

import smoke


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kr", default="60,0.3", help="K_R values, Nm/rad")
    p.add_argument("--text", default="S")
    p.add_argument("--boards", default="flat,curved")
    p.add_argument("--erase-work", type=float, default=None)
    p.add_argument("--penetration", type=float, default=0.004)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wrist-inertia", type=float, default=None,
                   help="correct the inertia Dr is built from (true value ~0.2)")
    p.add_argument("--force", type=float, default=None,
                   help="N: regulate the normal force, so K_R cannot leak into it")
    a = p.parse_args()

    print(f"glyph {a.text!r}, scripted raster,"
          + (f" normal force held at {a.force:.1f} N," if a.force else
             f" penetration {1000*a.penetration:.0f} mm,")
          + f" erase_work {a.erase_work if a.erase_work else 0.040} N m")
    print("\n board    K_R   erased   t90    mis  yield   f_mean  peak  in-band  success")
    out = {}
    for board in a.boards.split(","):
        for kr in [float(x) for x in a.kr.split(",")]:
            r = smoke.run(a.text, penetration=a.penetration, curved=(board == "curved"),
                          kr=kr, seed=a.seed, erase_work=a.erase_work,
                          wrist_inertia=a.wrist_inertia, f_target=a.force,
                          verbose=False)
            f = r["trace"]["f"]
            out[(board, kr)] = r
            print(f" {board:6s} {kr:5g}  {100*r['erased']:5.1f}%  {r['t90']:5.1f}"
                  f"  {r['mis_mean']:5.1f}  {r['yld_mean']:5.1f}"
                  f"  {np.mean(f[f > 0.8]):6.2f}  {r['peak_force']:5.1f}"
                  f"   {100*r['in_band']:4.0f}%   {r['success']}")

    print("\n t90  pad-down seconds until 90% of the glyph was gone")
    print(" mis  pad face against the true local normal, mean while pressing")
    print(" yld  how far the pad moved off the vertical it was commanded")
    for board in a.boards.split(","):
        krs = [float(x) for x in a.kr.split(",")]
        if len(krs) == 2 and all((board, k) in out for k in krs):
            hi, lo = (out[(board, k)] for k in krs)
            print(f" {board}: K_R {krs[0]:g} -> {krs[1]:g} moves misalignment"
                  f" {hi['mis_mean']:.1f} -> {lo['mis_mean']:.1f} deg,"
                  f" t90 {hi['t90']:.1f} -> {lo['t90']:.1f} s,"
                  f" in-band {100*hi['in_band']:.0f} -> {100*lo['in_band']:.0f}%")


if __name__ == "__main__":
    main()
