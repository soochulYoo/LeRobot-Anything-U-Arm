"""How much rotational stiffness, and does curvature explain why it matters?

Two things are swept together, because one without the other proves nothing:

  CURVED   the random Gaussian-bump slab, whose normal turns as the tool travels
  FLAT     the same slab with zero amplitude -- the normal never turns, so K_R
           has nothing to align to and the sweep should come out FLAT

The flat arm is the control.  An earlier version of this study showed a large
K_R effect on the flat slab too, which meant the effect was not alignment at
all: the rotational damping had been set from a guessed inertia of 5e-3 kg m^2
against a true operational-space value of 0.136, so every run was ringing at a
damping ratio of 0.15.  With Dr built from the measured inertia the flat arm is
null and the curved arm is not, which is what makes the curved result readable.

Usage:  python3 sweep.py [--seeds 3] [--out rot_stiffness.png] [--replot]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import pickle

import numpy as np

import wipe as W
from surface import CurvedSurface

KR = [1.0, 3.0, 10.0, 30.0, 60.0, 100.0]
TRACE_KR = (60.0, 3.0)          # the pair in the video, and in the trace panels
TRACE_SEED = 1

# House palette, validated: CVD dE 24.7 (protan) / 33.6 (normal) on this surface.
STIFF, SOFT = "#eb6834", "#2a78d6"
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"


def job(spec: tuple) -> dict:
    seed, Kr, curved = spec
    import argparse as _a
    args = W.add_args(_a.ArgumentParser()).parse_args([])
    surf = CurvedSurface(seed=seed) if curved else CurvedSurface(amp=0.0, seed=seed)
    r = W.run_one(surf, Kr, args)
    out = {k: r[k] for k in ("Kr", "align_mean", "align_p95", "force_rmse",
                             "f_mean", "peak", "contact", "tilt_mean")}
    out.update(seed=seed, curved=curved)
    if curved and seed == TRACE_SEED and Kr in TRACE_KR:
        # decimated to 50 Hz: the panels are 8 s wide and 4000 points would
        # just be ink
        s = slice(None, None, 10)
        out["trace"] = {"t": r["t"][s], "f": r["f_filt"][s],
                        "align": r["align"][s], "tilt": r["tilt"][s]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="rot_stiffness.png")
    ap.add_argument("--cache", default="rot_stiffness.pkl")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    if args.replot:
        rows = pickle.loads(cache.read_bytes())
        print(f"replotting {len(rows)} runs from {cache}")
    else:
        specs = [(s, k, c) for c in (True, False)
                 for k in KR for s in range(args.seeds)]
        print(f"{len(specs)} runs on {args.workers} workers")
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(job, specs)
        cache.write_bytes(pickle.dumps(rows))

    d = [r for r in rows if r.get("diverged")]
    if d:
        print("  %d of %d runs diverged and are excluded: %s"
              % (len(d), len(rows),
                 ", ".join(f"{'curved' if r['curved'] else 'flat'} K_R={r['Kr']:g}"
                           f" seed {r['seed']}" for r in d)))
    table(rows)
    draw(rows, args)


def ok(rows, curved, k):
    """Runs at this setting that did not diverge."""
    return [r for r in rows if r["curved"] is curved and r["Kr"] == k
            and not r.get("diverged", False)]


def agg(rows, curved, key):
    m, sd = [], []
    for k in KR:
        v = [r[key] for r in ok(rows, curved, k)]
        m.append(np.mean(v) if v else np.nan)
        sd.append(np.std(v) if v else np.nan)
    return np.array(m), np.array(sd)


def table(rows) -> None:
    print("\nCURVED slab -- mean over seeds")
    print("   K_R      align      force rmse      peak      contact")
    for k in KR:
        v = ok(rows, True, k)
        print("  %5.0f   %5.2f deg   %6.2f N    %6.1f N     %3.0f%%"
              % (k, np.mean([x["align_mean"] for x in v]),
                 np.mean([x["force_rmse"] for x in v]),
                 np.mean([x["peak"] for x in v]),
                 100 * np.mean([x["contact"] for x in v])))
    print("\nFLAT control -- the normal never turns; this arm should be level")
    print("   K_R      align      force rmse      peak      contact")
    for k in KR:
        v = ok(rows, False, k)
        print("  %5.0f   %5.2f deg   %6.2f N    %6.1f N     %3.0f%%"
              % (k, np.mean([x["align_mean"] for x in v]),
                 np.mean([x["force_rmse"] for x in v]),
                 np.mean([x["peak"] for x in v]),
                 100 * np.mean([x["contact"] for x in v])))


def draw(rows, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(14.6, 8.2), facecolor=SURFACE)
    for a in ax.ravel():
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK3)
        a.tick_params(colors=INK2, labelsize=8.5)
        a.grid(True, color=INK3, alpha=0.22, lw=0.6)
        a.set_axisbelow(True)

    panels = [("align_mean", "pad misalignment  [deg]", "A  is the pad flush?"),
              ("force_rmse", "force error, RMS  [N]", "B  is the force held?"),
              ("peak", "peak contact force  [N]", "C  worst spike"),
              ("contact", "in contact  [%]", "D  does it stay on the surface?")]
    for i, (key, ylab, title) in enumerate(panels[:3]):
        a = ax[0, i]
        _sweep_panel(a, rows, key, ylab, title, scale=1.0, legend=(i == 0))
    _sweep_panel(ax[1, 0], rows, "contact", panels[3][1], panels[3][2],
                 scale=100.0, legend=False)

    # E, F -- the two videoed settings over time, on one surface
    tr = {r["Kr"]: r["trace"] for r in rows if "trace" in r}
    for a, key, ylab, title in (
            (ax[1, 1], "align", "pad misalignment  [deg]",
             f"E  alignment over one traverse  (seed {TRACE_SEED})"),
            (ax[1, 2], "f", "contact force  [N]",
             f"F  force over the same traverse  (seed {TRACE_SEED})")):
        for Kr, col, lab in ((TRACE_KR[0], STIFF, f"stiff  $K_R$={TRACE_KR[0]:g}"),
                             (TRACE_KR[1], SOFT, f"compliant  $K_R$={TRACE_KR[1]:g}")):
            if Kr in tr:
                a.plot(tr[Kr]["t"], tr[Kr][key], color=col, lw=2.0, label=lab)
        if key == "align" and TRACE_KR[0] in tr:
            a.plot(tr[TRACE_KR[0]]["t"], tr[TRACE_KR[0]]["tilt"], color=INK,
                   lw=1.2, ls=(0, (5, 4)), label="surface tilt under the pad")
        if key == "f":
            a.axhline(8.0, color=INK, lw=1.2, ls=(0, (5, 4)), label="target 8 N")
        a.set_xlabel("time  [s]", color=INK2, fontsize=9)
        a.set_ylabel(ylab, color=INK2, fontsize=9)
        a.set_title(title, color=INK, fontsize=10.5, loc="left")
        a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    fig.suptitle("Rotational stiffness on a curved surface: a flat pad can only"
                 " lie flat if it is allowed to turn",
                 color=INK, fontsize=13, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"\n  wrote {args.out}")


def _sweep_panel(a, rows, key, ylab, title, scale, legend) -> None:
    import numpy as np
    for curved, col, lab, z in ((True, SOFT, "curved slab", 4),
                                (False, INK3, "flat control", 3)):
        m, sd = agg(rows, curved, key)
        m, sd = m * scale, sd * scale
        a.fill_between(KR, m - sd, m + sd, color=col, alpha=0.15, lw=0, zorder=z)
        a.plot(KR, m, color=col, lw=2.0, marker="o", ms=5.5, zorder=z + 1,
               label=lab, ls="-" if curved else (0, (5, 4)))
    # The two settings shown in the video, marked on the curve they came from.
    m, _ = agg(rows, True, key)
    for Kr, col in ((TRACE_KR[0], STIFF), (TRACE_KR[1], SOFT)):
        if Kr in KR:
            a.plot([Kr], [m[KR.index(Kr)] * scale], marker="o", ms=11, mfc="none",
                   mec=col, mew=2.2, zorder=8)
    a.set_xscale("log")
    a.set_xticks(KR); a.set_xticklabels([f"{k:g}" for k in KR])
    a.set_xlabel("rotational stiffness $K_R$  [Nm/rad]", color=INK2, fontsize=9)
    a.set_ylabel(ylab, color=INK2, fontsize=9)
    a.set_title(title, color=INK, fontsize=10.5, loc="left")
    if legend:
        a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")


if __name__ == "__main__":
    main()
