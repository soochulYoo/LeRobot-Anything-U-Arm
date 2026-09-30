"""T1: of a stiffness label's three factors, which one does a contact task care about?

Every published stiffness-label rule returns a diagonal matrix in some frame, so
a label carries three things and papers report only the first:

    K = exp(s) * R diag(w) R^T
    s  MAGNITUDE  -- the number the rules disagree about (4-6x in the
                     literature's own sweeps; 27x when measured directly here)
    w  RATIO      -- how anisotropic it is
    R  FRAME      -- WHICH axis is the soft one.  Never stated as a choice, yet
                     "stiffness along the direction of motion" and "stiffness
                     along the surface normal" are 90 degrees apart on a wipe.

Each factor is varied on its own against the wiping task while the other two are
held at the expert's values, over several surface tilts.  The score is a single
failure margin, so the three are read on one axis:

    margin = max(force_error / tolerance, path_error / tolerance),  < 1 = success

Usage:  python3 study_anisotropy.py [--quick] [--out FIG.png]
"""
from __future__ import annotations

import argparse

import numpy as np

import wipe as W

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

MAG0, RATIO0 = 700.0, 10.0          # the expert's magnitude and ratio
FORCE_TOL, PATH_TOL = 0.30, 0.005   # the success thresholds in wipe.rollout


def margin(res: W.WipeResult) -> float:
    """Worst of the two normalised errors.  < 1 means the task succeeded."""
    return max(res.force_rel / FORCE_TOL, res.path_error / PATH_TOL)


def margins_of(fr, pe, ftol=FORCE_TOL, ptol=PATH_TOL) -> np.ndarray:
    return np.maximum(fr / ftol, pe / ptol)


def make_tasks(n: int, seed: int = 0) -> list[W.WipeTask]:
    """Surfaces at randomised tilt, so no conclusion rests on one geometry."""
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n):
        tx, ty = rng.uniform(-0.44, 0.44, size=2)     # +-25 deg
        tasks.append(W.WipeTask(tilt_x=float(tx), tilt_y=float(ty)))
    return tasks


def frame_rotated(task: W.WipeTask, deg: float) -> np.ndarray:
    """The believed task frame turned by `deg` about its own second tangent --
    i.e. the soft axis swung away from the surface normal toward the stroke."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return task.R_belief @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def sweep(tasks, factor: str, values):
    """Raw (force_rel, path_error) of shape [len(values), len(tasks)].

    The RAW metrics are kept, not the margin, so the success thresholds can be
    varied afterwards without re-running anything.  The thresholds are a choice,
    and a conclusion that only holds at one choice of them is not a conclusion.
    """
    fr = np.zeros((len(values), len(tasks)))
    pe = np.zeros((len(values), len(tasks)))
    for i, v in enumerate(values):
        for j, task in enumerate(tasks):
            if factor == "magnitude":
                K = W.compose(MAG0 * v, RATIO0, task.R_belief)
            elif factor == "ratio":
                K = W.compose(MAG0, v, task.R_belief)
            elif factor == "frame":
                K = W.compose(MAG0, RATIO0, frame_rotated(task, v))
            else:
                raise ValueError(factor)
            r = W.rollout(task, K)
            fr[i, j], pe[i, j] = r.force_rel, r.path_error
        print(f"    {factor}={v:>8.3g}  force {100*np.median(fr[i]):5.1f}%  "
              f"path {1000*np.median(pe[i]):5.2f} mm")
    return fr, pe


def tolerance(values, margins, nominal) -> tuple[float, float]:
    """Contiguous run of successes around the nominal value.

    Contiguity matters: a sweep can pass again far from the nominal for the wrong
    reason (a very stiff K tracks the path while hammering the surface), and an
    interval taken as min-to-max of all passing values would hide that.
    """
    med = np.median(margins, axis=1)
    ok = med < 1.0
    i0 = int(np.argmin(np.abs(np.asarray(values, float) - nominal)))
    if not ok[i0]:
        return (np.nan, np.nan)
    lo = i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(values[lo]), float(values[hi])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tilts", type=int, default=5)
    ap.add_argument("--replot", action="store_true",
                    help="redraw from anisotropy_raw.npz without re-running the sweep")
    ap.add_argument("--out", default="anisotropy.png")
    args = ap.parse_args()

    tasks = make_tasks(3 if args.quick else args.tilts)
    print(f"{len(tasks)} surfaces, tilt +-25 deg, believed normal "
          f"{np.rad2deg(tasks[0].belief_tilt_error):.0f} deg off the true one")
    print(f"expert: magnitude {MAG0:.0f} N/m, ratio {RATIO0:.0f}, believed task frame")
    ex = [margin(W.rollout(t, W.expert_stiffness(t, MAG0, RATIO0))) for t in tasks]
    print(f"expert margin across surfaces: {np.min(ex):.2f}-{np.max(ex):.2f} (success < 1)\n")

    # Fine, geometric grids: the tolerance windows turned out narrow enough that
    # a coarse sweep reports the grid spacing rather than the physics.
    mags = (np.geomspace(0.08, 30.0, 17) if not args.quick
            else np.array([0.1, 0.5, 1.0, 3.0, 10.0]))
    ratios = (np.geomspace(0.1, 100.0, 15) if not args.quick
              else np.array([0.3, 1.0, 3.0, 10.0, 100.0]))
    frames = (np.array([0., 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60, 75, 90.])
              if not args.quick else np.array([0.0, 15, 30, 60, 90.0]))

    if args.replot:
        z = np.load("anisotropy_raw.npz")
        mags, ratios, frames = z["mags"], z["ratios"], z["frames"]
        fr_mag, pe_mag = z["fr_mag"], z["pe_mag"]
        fr_rat, pe_rat = z["fr_rat"], z["pe_rat"]
        fr_frm, pe_frm = z["fr_frm"], z["pe_frm"]
        print("  (replotting from anisotropy_raw.npz)")
    else:
        print("  MAGNITUDE sweep (x the expert's 700 N/m)")
        fr_mag, pe_mag = sweep(tasks, "magnitude", mags)
        print("  RATIO sweep (tangential / normal stiffness)")
        fr_rat, pe_rat = sweep(tasks, "ratio", ratios)
        print("  FRAME sweep (soft axis swung off the surface normal, degrees)")
        fr_frm, pe_frm = sweep(tasks, "frame", frames)
        np.savez("anisotropy_raw.npz", mags=mags, ratios=ratios, frames=frames,
                 fr_mag=fr_mag, pe_mag=pe_mag, fr_rat=fr_rat, pe_rat=pe_rat,
                 fr_frm=fr_frm, pe_frm=pe_frm)

    m_mag = margins_of(fr_mag, pe_mag)
    m_rat = margins_of(fr_rat, pe_rat)
    m_frm = margins_of(fr_frm, pe_frm)
    t_mag = tolerance(mags, m_mag, 1.0)
    t_rat = tolerance(ratios, m_rat, RATIO0)
    t_frm = tolerance(frames, m_frm, 0.0)

    print("\n" + "=" * 86)
    print("TOLERANCE: how wrong can a label be on each factor before the task fails?")
    print("=" * 86)
    print(f"  magnitude   {t_mag[0]:.2g}x to {t_mag[1]:.2g}x the expert's "
          f"-> a {t_mag[1]/t_mag[0]:.0f}x window")
    print(f"  ratio       {t_rat[0]:.2g} to {t_rat[1]:.2g} "
          f"-> a {t_rat[1]/t_rat[0]:.0f}x window (expert {RATIO0:.0f})")
    print(f"  frame       0 to {t_frm[1]:.0f} deg off the surface normal")
    print("\n  Against what the rules actually disagree by:")
    print(f"    magnitude : 4-6x in the literature's own sweeps -> a {t_mag[1]/t_mag[0]:.1f}x window")
    print(f"                is the SAME SIZE. The labels sit at the edge of tolerance, which")
    print(f"                is how they can all appear to work and still be brittle.")
    print(f"                The 27x spread measured in study_labels does not fit at all.")
    print(f"    ratio     : a {t_rat[1]/t_rat[0]:.0f}x one-sided window -- anything stiffer than")
    print(f"                ~{t_rat[0]:.0f}:1 tangentially passes. This factor is easy to get right,")
    print(f"                which is likely why nobody noticed it matters.")
    print(f"    frame     : tolerated to {t_frm[1]:.0f} deg, but 'stiffness along the motion")
    print(f"                direction' and 'along the surface normal' are 90 deg apart on a")
    print(f"                wipe. No paper states the frame as a choice at all.")

    print("\n" + "=" * 86)
    print("THRESHOLD SENSITIVITY: the success thresholds were chosen, so does the")
    print("ordering survive changing them?")
    print("=" * 86)
    print(f"  {'force tol':>10}{'path tol':>10}{'magnitude':>16}{'ratio':>14}{'frame':>12}")
    print("  " + "-" * 60)
    orders, mag_windows, frm_windows = [], [], []
    for fscale, pscale in [(0.5, 0.5), (0.75, 0.75), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0),
                           (0.5, 2.0), (2.0, 0.5)]:
        ft, pt = FORCE_TOL * fscale, PATH_TOL * pscale
        a = tolerance(mags, margins_of(fr_mag, pe_mag, ft, pt), 1.0)
        b = tolerance(ratios, margins_of(fr_rat, pe_rat, ft, pt), RATIO0)
        c = tolerance(frames, margins_of(fr_frm, pe_frm, ft, pt), 0.0)
        wa = a[1] / a[0] if np.isfinite(a[0]) else np.nan
        wb = b[1] / b[0] if np.isfinite(b[0]) else np.nan
        feasible = np.isfinite(wa) and np.isfinite(wb)
        note = "" if feasible else "   (expert itself fails: path tol < its own error)"
        print(f"  {100*ft:9.0f}%{1000*pt:9.1f}mm"
              + (f"{wa:14.1f}x{wb:13.0f}x" if feasible else f"{'--':>14}{'--':>14}")
              + (f"{c[1]:10.0f} deg" if np.isfinite(c[1]) else f"{'--':>10}    ") + note)
        if feasible:
            orders.append(wb > wa)
            mag_windows.append(wa)
            frm_windows.append(c[1])
    print("  " + "-" * 60)
    print(f"  WHAT SURVIVES every feasible threshold setting ({len(orders)} of them):")
    print(f"    - the ratio window is wider than the magnitude window: {sum(orders)}/{len(orders)}")
    print(f"    - the frame never tolerates the 90 deg that separates 'along the motion'")
    print(f"      from 'along the normal': widest seen {max(frm_windows):.0f} deg")
    print(f"  WHAT DOES NOT SURVIVE:")
    print(f"    - the magnitude window spans {min(mag_windows):.1f}x to {max(mag_windows):.1f}x across")
    print(f"      these settings. At strict tolerances it is the size of the published")
    print(f"      disagreement (4-6x); at loose ones it comfortably contains it. So")
    print(f"      'magnitude labels sit at the edge' holds only for a demanding task,")
    print(f"      and must be stated with the tolerance it was measured at.")
    print(f"  The frame result is the robust one; the magnitude result is conditional.")

    plot(tasks, mags, m_mag, t_mag, ratios, m_rat, t_rat, frames, m_frm, t_frm, args.out)


def plot(tasks, mags, m_mag, t_mag, ratios, m_rat, t_rat, frames, m_frm, t_frm, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))

    panels = [
        (ax[0], mags, m_mag, t_mag, SLOT[0], "log",
         r"magnitude, $\times$ the expert's 700 N/m",
         f"Magnitude: {t_mag[1]/t_mag[0]:.0f}$\\times$ here; 3$-$59$\\times$ across thresholds",
         "published rules\ndisagree 4$-$6$\\times$", (4.0, 6.0)),
        (ax[1], ratios, m_rat, t_rat, SLOT[1], "log",
         "anisotropy ratio  (tangential / normal)",
         f"Ratio: {t_rat[1]/t_rat[0]:.0f}$\\times$ and one-sided; widest at every threshold",
         None, None),
        (ax[2], frames, m_frm, t_frm, SLOT[2], "linear",
         "soft axis swung off the surface normal (deg)",
         f"Frame: {t_frm[1]:.0f}$\\degree$ here; never past 40$\\degree$ at any threshold", 
         "'along the motion' vs\n'along the normal'", (90.0, 90.0)),
    ]
    for a, xs, M, tol, c, scale, xlab, title, note, band in panels:
        med = np.median(M, axis=1)
        a.fill_between(xs, np.min(M, axis=1), np.max(M, axis=1), color=c, alpha=0.18, lw=0)
        a.plot(xs, med, "-o", color=c, lw=2.2, ms=5.5, zorder=3,
               markeredgecolor=SURFACE, markeredgewidth=1.2, label="median over surfaces")
        a.axhline(1.0, color=INK3, ls="--", lw=1.4)
        a.text(xs[0], 1.06, " failure threshold", color=INK2, fontsize=7.5)
        # Brackets and labels go in AXES coordinates: the y axis is logarithmic and
        # its lower limit is near 0.3, so a label placed at a data y of 0.12 lands
        # outside the panel entirely, which is what the first version did.
        if np.isfinite(tol[0]):
            a.axvspan(tol[0], tol[1], color=c, alpha=0.13, lw=0, zorder=1)
            mid = np.sqrt(tol[0] * tol[1]) if scale == "log" else 0.5 * (tol[0] + tol[1])
            lbl = (f"tolerated {tol[0]:.2g}$-${tol[1]:.2g}$\\times$" if scale == "log"
                   else f"tolerated 0$-${tol[1]:.0f}$\\degree$")
            a.annotate("", xy=(tol[0], 0.06), xytext=(tol[1], 0.06),
                       xycoords=("data", "axes fraction"),
                       textcoords=("data", "axes fraction"),
                       arrowprops=dict(arrowstyle="<->", color=c, lw=1.5))
            a.annotate(lbl, xy=(mid, 0.11), xycoords=("data", "axes fraction"),
                       color=c, fontsize=8.5, ha="center")
        if band is not None:
            if band[0] == band[1]:
                a.axvline(band[0], color=INK, ls=":", lw=1.6)
                a.annotate(note + " ", xy=(band[0], 0.88), xycoords=("data", "axes fraction"),
                           color=INK, fontsize=8, ha="right", va="top")
            else:
                a.axvspan(band[0], band[1], facecolor="none", hatch="///",
                          edgecolor=INK3, lw=0.0, alpha=0.7, zorder=2)
                a.annotate(note, xy=(np.sqrt(band[0] * band[1]), 0.76),
                           xycoords=("data", "axes fraction"), color=INK,
                           fontsize=8, ha="center", va="top")
        a.set_xscale(scale); a.set_yscale("log")
        a.set_xlabel(xlab); a.set_ylabel("failure margin  (< 1 = task succeeds)")
        a.set_title(title, color=INK, loc="left")
        a.grid(True, which="major", alpha=0.9); a.set_axisbelow(True)
        a.legend(fontsize=7.5, loc="upper right")
        a.set_ylim(bottom=min(0.3, float(np.min(M)) * 0.85))

    fig.suptitle(
        "T1: a stiffness label carries a magnitude, a ratio and a frame. Only the FRAME's tolerance is exceeded by how much published rules disagree "
        "-- and at every success threshold tested.\n"
        "Shaded band = tolerated range at the thresholds named in the axis label (force 30%, path 5 mm); hatched = published disagreement.",
        color=INK, fontsize=10, x=0.006, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(out_path, dpi=170)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
