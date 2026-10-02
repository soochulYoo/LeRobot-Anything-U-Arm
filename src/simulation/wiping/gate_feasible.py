"""Does a patchy board exist that variable K_R could possibly win on?

Every negative result so far (40 dome cells, a 5-knot rule search with the
constant nested, the grinding-shaped attitude task) was collected on a board
that violated one of two NECESSARY conditions.  Neither is about control, so
neither needs a simulator -- which is the point of this file.  It screens
candidate geometries in numpy before any of them costs CPU hours.

    (A) THE DEMAND EXISTS AND ROTATION IS WHAT ANSWERS IT.  The pad's face is a
        PLANE, so fit one over the footprint and the surface splits in two:

          TILT of the fitted plane  -- the part a rotation can take out.  This
            is the demand on K_R, and it has to be large in the curved bands and
            small in the flat ones or there is nothing to schedule.
          RESIDUAL to the fitted plane -- the part NO rotation can take out, at
            any stiffness.  `pad_give` has to absorb it; above that the pad
            cannot seat even at the perfect angle and the task is limited by
            geometry rather than by stiffness.

        So the window is residual < pad_give AND tilt varying along the path --
        which INVERTS the boundary this directory has been testing.
        `dome_boundary.sbatch` read "sagitta > give" as the regime adaptation
        should pay in; it is the regime where rotation is powerless, and a
        constant winning all five of those cells is what that predicts.

    (B) THE WRIST CAN FOLLOW.  The optimum is K_R* ~ f_n r / swing, and a
        rotational mode at that stiffness settles in ~4/(zeta w), w =
        sqrt(K_R*/Lambda_r).  If the pad crosses a band faster than that, no
        schedule can be tracked and the best a controller can do IS a constant.
        This is the condition the existing patchy board fails: 0.17 m bands at
        0.12 m/s is 1.4 s against a 3.1 s settling time.

    (C) The raster has to CROSS bands, at least three transitions, or the
        episode contains one regime change and a constant is right either side
        of it.  3 transitions need 1.5 periods inside the wiped extent, and the
        extent is bounded by the board.

(A) and (C) pull against each other -- (A) wants bumps about the pad's size,
(C) wants several bands inside 0.32 m of board -- which is why this is a screen
and not a calculation.

    python3 gate_feasible.py                 # the grid
    python3 gate_feasible.py --verbose       # every cell, including failures
"""
from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from patchy_surface import CorrugatedSurface, PatchySurface, swing   # noqa: E402

PAD_GIVE = 0.004           # m, WipeCriteria.pad_give
LAMBDA_R = 0.38            # kg m^2, the worst axis of the measured op-space rotational inertia
ZETA = 0.8                 # the controller's design damping ratio, after the matrix-Dr fix
N_TAU = 2.3                # time constants to 90% of a step -- what "acquired the tilt" means


def footprint_plane(surf, r_pad: float, xy, n: int = 9):
    """Split the surface under the pad into what rotation can and cannot fix.

    Returns (tilt_deg, residual_m): the fitted plane's tilt off horizontal --
    the rotation the wrist would have to acquire -- and the max |residual| to
    that plane, which is the gap the felt has to close at the best possible
    angle.  A plane IS the pad's face, so this is the whole decomposition.
    """
    g = np.linspace(-r_pad, r_pad, n)
    dx, dy = np.meshgrid(g, g, indexing="ij")
    keep = (dx ** 2 + dy ** 2) <= r_pad ** 2 + 1e-12        # the pad is a disc to the eraser model
    dx, dy = dx[keep], dy[keep]
    A = np.column_stack([dx, dy, np.ones_like(dx)])
    pinv = np.linalg.pinv(A)
    tilt, res = np.empty(len(xy)), np.empty(len(xy))
    for i, (x0, y0) in enumerate(xy):
        h = surf.height(x0 + dx, y0 + dy)
        c = pinv @ h
        tilt[i] = np.degrees(np.arctan(np.hypot(c[0], c[1])))
        res[i] = np.abs(h - A @ c).max()
    return tilt, res


def screen(r_pad, sigma, amp, period, speed, f_n, half=0.16, seed=3, n_bumps=16):
    surf = PatchySurface(half=half, seed=seed, n_bumps=n_bumps, sigma=sigma,
                         amp=amp, period=period, sharp=0.25)
    # Along the wipe, inside the reachable extent: the raster cannot get closer
    # to the rim than the pad's own half-width.
    u = np.linspace(-(half - r_pad), half - r_pad, 241)
    xy = np.column_stack([u, np.zeros_like(u)])
    sw = swing(surf, xy, r_pad)
    tilt, res = footprint_plane(surf, r_pad, xy)
    b = surf.band_at(u)
    flat, curv = b < 0.2, b > 0.8
    if not (flat.any() and curv.any()):
        return None
    sw_c, sw_f = float(sw[curv].mean()), float(sw[flat].mean())
    ti_c, ti_f = float(tilt[curv].mean()), float(tilt[flat].mean())

    # (B) K_R* is set by the DEMAND, not by the swing: a steady yield of theta
    # needs K_R <= M/theta and the pad can make at most M = f_n r of moment.
    kr_star = f_n * r_pad / np.radians(max(ti_c, 1e-6))
    settle = 4.0 / (ZETA * np.sqrt(kr_star / LAMBDA_R))
    t_band = (period / 2.0) / speed

    # (C) three transitions need 1.5 periods inside the wiped extent
    extent = 2.0 * (half - r_pad)
    return dict(
        r_pad=r_pad, sigma=sigma, amp=amp, period=period, speed=speed, f_n=f_n,
        swing_curved=sw_c, swing_flat=sw_f,
        tilt_curved=ti_c, tilt_flat=ti_f, contrast=ti_c / max(ti_f, 1e-9),
        res_curved_mm=1e3 * float(res[curv].mean()), res_flat_mm=1e3 * float(res[flat].mean()),
        kr_star=float(kr_star), settle=float(settle), t_band=float(t_band),
        margin_B=float(t_band / settle), periods=float(extent / period),
        A=bool(res[curv].mean() <= PAD_GIVE), B=bool(t_band > settle),
        C=bool(extent >= 1.5 * period),
        contrast_ok=bool(ti_c >= 8.0 and ti_f <= 2.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="corr", choices=("corr", "patchy"))
    ap.add_argument("--plateau", default="0.030,0.040,0.060,0.080")
    ap.add_argument("--flank", default="0.040,0.060,0.080,0.100")
    ap.add_argument("--slope", default="8,12,16,20")
    ap.add_argument("--fn", default="4,10,20")
    ap.add_argument("--half", type=float, default=0.16)
    ap.add_argument("--give", type=float, default=PAD_GIVE)
    ap.add_argument("--lam", type=float, default=LAMBDA_R)
    ap.add_argument("--pads", default="0.015,0.025,0.040")
    ap.add_argument("--sigma", default="0.020,0.030,0.045")
    ap.add_argument("--amp", default="0.008,0.014,0.020")
    ap.add_argument("--period", default="0.08,0.12,0.16,0.34")
    ap.add_argument("--speed", default="0.015,0.03,0.05,0.12")
    ap.add_argument("--f", type=float, default=4.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.family == "corr":
        main_corr(a)
        return
    grid = [[float(x) for x in s.split(",")]
            for s in (a.pads, a.sigma, a.amp, a.period, a.speed)]

    rows = [r for r in (screen(*c, f_n=a.f) for c in itertools.product(*grid)) if r]
    ok = [r for r in rows if r["A"] and r["B"] and r["C"] and r["contrast_ok"]]
    print(f"{len(rows)} cells screened, f_n = {a.f} N, pad_give = {1e3*PAD_GIVE:.0f} mm, "
          f"Lambda_r = {LAMBDA_R} kg m^2\n")
    hdr = (f"{'pad':>5} {'sig':>5} {'amp':>5} {'per':>5} {'v':>6} | "
           f"{'ti_c':>5} {'ti_f':>5} | {'res_c':>6} | {'K_R*':>6} {'settle':>6} "
           f"{'t_band':>6} {'B':>5} | {'cyc':>4} | A B C ct")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows if a.verbose else ok,
                    key=lambda r: -(r["margin_B"] * min(r["contrast"], 20))):
        print(f"{1e3*r['r_pad']:5.0f} {1e3*r['sigma']:5.0f} {1e3*r['amp']:5.0f} "
              f"{1e3*r['period']:5.0f} {r['speed']:6.3f} | "
              f"{r['tilt_curved']:5.1f} {r['tilt_flat']:5.1f} | "
              f"{r['res_curved_mm']:6.2f} | {r['kr_star']:6.2f} {r['settle']:6.2f} "
              f"{r['t_band']:6.2f} {r['margin_B']:5.2f} | {r['periods']:4.1f} | "
              f"{'Y' if r['A'] else '.'} {'Y' if r['B'] else '.'} "
              f"{'Y' if r['C'] else '.'} {'Y' if r['contrast_ok'] else '.'}")
    print(f"\n{len(ok)} of {len(rows)} cells pass all four.")
    if not ok:
        print("NONE -- the screen says no patchy board in this grid can be won on, "
              "and the gate should not be run until one is found.")



def screen_corr(r_pad, plateau, flank, slope_deg, speed, f_n,
                half=0.16, corner=0.004, give=PAD_GIVE, lam=LAMBDA_R):
    """The same screen on a trapezoidal corrugation, plus the condition that
    makes softness COST something.

    (A1) the flank's tilt is more than the felt can swallow:  2 r tan(th) > give
    (A2) the flank is flat enough that a rotation answers it: residual <= give
    (A3) the plateau asks for nothing:                        2 r tan(th) <= give
    (B)  the wrist can acquire the flank's tilt while on it:  t90 < flank/v
    (C)  the raster visits several facets:                    >= 4 boundaries
    (D)  AND IT CANNOT RECOVER ON THE PLATEAU:  plateau/v < 2 t90.  Without this
         the soft wrist settles flush again before the marks there are reached,
         softness is free, and a low constant wins -- which is every negative
         result in this directory.  (B) and (D) together are flank > plateau/2.
    """
    surf = CorrugatedSurface(half=half, plateau=plateau, flank=flank,
                             slope_deg=slope_deg, corner=corner)
    u = np.linspace(-(half - r_pad), half - r_pad, 121)
    xy = np.column_stack([u, np.zeros_like(u)])
    tilt, res = footprint_plane(surf, r_pad, xy)
    b = surf.band_at(u)
    fl, pl = b > 0.8, b < 0.2
    if not (fl.any() and pl.any()):
        return None
    ti_f, ti_p = float(tilt[fl].mean()), float(tilt[pl].mean())
    ask_f = 2.0 * r_pad * np.tan(np.radians(ti_f))        # m of height across the pad
    ask_p = 2.0 * r_pad * np.tan(np.radians(ti_p))
    seat = float((res[fl] <= give).mean())

    kr_star = f_n * r_pad / np.radians(max(ti_f, 1e-6))
    t90 = N_TAU / (ZETA * np.sqrt(kr_star / lam))
    t_flank, t_plat = flank / speed, plateau / speed
    crossings = int(np.count_nonzero(np.diff((b > 0.5).astype(int)) != 0))
    return dict(
        r_pad=r_pad, plateau=plateau, flank=flank, slope_deg=slope_deg,
        speed=speed, f_n=f_n, tilt_flank=ti_f, tilt_plat=ti_p,
        ask_flank_mm=1e3 * ask_f, ask_plat_mm=1e3 * ask_p, seat=seat,
        res_flank_mm=1e3 * float(res[fl].mean()), kr_star=float(kr_star),
        t90=float(t90), t_flank=float(t_flank), t_plat=float(t_plat),
        crossings=crossings,
        A1=bool(ask_f > give), A2=bool(seat >= 0.7), A3=bool(ask_p <= give),
        B=bool(t_flank > t90), C=bool(crossings >= 4), D=bool(t_plat < 2.0 * t90),
    )


KEYS = ("A1", "A2", "A3", "B", "C", "D")


def main_corr(a):
    grid = [[float(x) for x in s.split(",")]
            for s in (a.pads, a.plateau, a.flank, a.slope, a.speed, a.fn)]
    rows = [r for r in (screen_corr(*c, half=a.half, give=a.give, lam=a.lam)
                        for c in itertools.product(*grid)) if r]
    ok = [r for r in rows if all(r[k] for k in KEYS)]
    print(f"{len(rows)} corrugated cells, pad_give {1e3*a.give:.0f} mm, "
          f"Lambda_r {a.lam} kg m^2, board {1e3*2*a.half:.0f} mm\n")
    hdr = (f"{'pad':>4}{'plat':>5}{'flnk':>5}{'slp':>4}{'v':>6}{'f':>4} | "
           f"{'ti_f':>5}{'ask':>5}{'seat':>5} | {'K*':>5}{'t90':>5}{'tfl':>5}{'tpl':>5} | "
           f"{'x':>2} | " + " ".join(KEYS))
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows if a.verbose else ok,
                    key=lambda r: -(min(r['t_flank'] / r['t90'], 3) * r['seat'])):
        print(f"{1e3*r['r_pad']:4.0f}{1e3*r['plateau']:5.0f}{1e3*r['flank']:5.0f}"
              f"{r['slope_deg']:4.0f}{r['speed']:6.3f}{r['f_n']:4.0f} | "
              f"{r['tilt_flank']:5.1f}{r['ask_flank_mm']:5.1f}{r['seat']:5.2f} | "
              f"{r['kr_star']:5.2f}{r['t90']:5.2f}{r['t_flank']:5.2f}{r['t_plat']:5.2f} | "
              f"{r['crossings']:2d} | "
              + " ".join((' Y' if r[k] else ' .') for k in KEYS))
    print(f"\n{len(ok)} of {len(rows)} cells pass all six.")
    return ok
if __name__ == "__main__":
    main()
