"""cascade vs Case 1 vs Case 2, on the SAME spiral wipe -> case_spiral.png

study_cascade_spiral.py ablates the cascade against ITSELF (stiff inner, no
outer).  This one swaps in the manuscript's two single-interface executors
instead, under the identical delayed operator, surface and seed.

THE HEADLINE, which the static comparison in analytic.py cannot see.  On a
quasistatic push Case 1 and Case 2 are indistinguishable -- both are one series
compliance, both deliver 5.308 N.  Put them on a moving, undulating, stiff
surface and they separate completely, and not because of the admittance:

    ke = 5000 N/m, Ts = 20 ms, ma = 3  ->  Eq. 15 needs da >= 67.20 Ns/m
    the cascade's own gains give           da = Ba + Br = 65 Ns/m

Case 2 misses the stability boundary by 2.2 Ns/m -- 3% -- on gains that the
cascade runs happily, because the cascade terminates in a Cartesian impedance
rather than a lag servo.  The same numbers are safe in one structure and
divergent in the other.  That is the comparison worth publishing.

Usage:  python3 study_case_spiral.py [--seeds 8] [--out case_spiral.png]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np

import analytic as A
import case_spiral as CS
import spiral_task as S

# 3 categorical slots (the repo's house palette, validated) + 1 STATUS colour.
# The failing Case 2 is not a fourth peer controller -- it is the same
# controller in a configuration that violates Eq. 15 -- so it wears the
# reserved critical red and is always labelled, never distinguished by hue alone.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
CRITICAL = "#d03b3b"
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

# (label, controller, kwargs, task-override, colour)
CONFIGS = [
    ("cascade",              "cascade", {},              {},             SLOT[0]),
    ("Case 1",               "Case 1",  {},              {},             SLOT[1]),
    ("Case 2  da=150",       "Case 2",  {},              {"Br": 110.0},  SLOT[2]),
    ("Case 2  da=65",        "Case 2",  {},              {},             CRITICAL),
]
LABELS = [c[0] for c in CONFIGS]
COLOURS = {c[0]: c[4] for c in CONFIGS}

_I = S.SpiralIntent()
# Starts just after contact, not at 0.0: the first logged sample sits one dt
# PAST t=settle, so a grid point at exactly 0.0 falls outside every episode's
# interpolation range and produces an all-NaN column (a "mean of empty slice").
T_GRID = np.linspace(0.05, 0.97 * _I.duration, 240)     # s since contact


def episode(job) -> dict:
    import dataclasses
    seed, label = job
    cfg = next(c for c in CONFIGS if c[0] == label)
    _, controller, kw, over, _ = cfg
    task = dataclasses.replace(S.SpiralTask(), **over)
    intent = S.SpiralIntent()
    op, surface = S.Operator(seed=seed), S.RandomSurface(seed=seed)
    log = CS.demonstrate(task, intent, op, surface, controller, **kw)
    g = CS.gap(log)

    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
    t = np.asarray(log["t"])
    sel = t >= task.settle
    rel = np.asarray(log["x"])[sel] - task.wall * n
    xy = np.column_stack([rel @ t1, rel @ t2])
    f = np.asarray(log["f_normal"])[sel]
    ts = t[sel] - task.settle

    out = {"seed": seed, "label": label, **g}
    out["f_vs_t"] = np.interp(T_GRID, ts, f, left=np.nan, right=np.nan)
    if seed == 3:
        out["xy"] = xy
    return out


def band(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    return np.nanmean(a, axis=0), np.nanstd(a, axis=0)


def figure(by, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 4, figsize=(16.4, 8.4), facecolor=SURFACE)
    for a in ax.ravel():
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(INK3)
        a.tick_params(colors=INK2, labelsize=8.5)
        a.grid(True, color=INK3, alpha=0.22, lw=0.6)
        a.set_axisbelow(True)

    intent = S.SpiralIntent()
    th = np.linspace(0.5 * np.pi, 2 * np.pi * intent.turns, 800)
    rr = intent.pitch * th / (2 * np.pi)
    ideal_x, ideal_y = rr * np.cos(th) * 1000, rr * np.sin(th) * 1000
    lim = 1.12 * max(np.abs(ideal_x).max(), np.abs(ideal_y).max())

    # ---- row 1: the path each executor actually drew, shared limits ----
    for i, lab in enumerate(LABELS):
        a = ax[0, i]
        xy = next((r["xy"] for r in by[lab] if "xy" in r), None)
        a.plot(ideal_x, ideal_y, color=INK, lw=1.0, ls=(0, (5, 4)), zorder=6,
               label="intended spiral")
        if xy is not None:
            a.plot(xy[:, 0] * 1000, xy[:, 1] * 1000, color=COLOURS[lab], lw=1.5, alpha=0.95)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        pr = np.mean([r["path_result"] for r in by[lab]]) * 1000
        a.set_title(f"{lab}\npath RMS {pr:.2f} mm", color=INK, fontsize=10.5, pad=8)
        a.set_xlabel("surface axis 1 (mm)", color=INK2, fontsize=9)
        if i == 0:
            a.set_ylabel("surface axis 2 (mm)", color=INK2, fontsize=9)
            a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    # ---- (e) normal force vs time: the three that hold contact ----
    a = ax[1, 0]
    a.axhline(intent.f_normal, color=INK, lw=1.0, ls=(0, (5, 4)), zorder=6)
    a.annotate(f"intended {intent.f_normal:.0f} N", (T_GRID[0], intent.f_normal),
               textcoords="offset points", xytext=(2, -13), ha="left",
               color=INK, fontsize=8.5, zorder=7)
    for lab in LABELS[:3]:
        mu, sd = band(by[lab], "f_vs_t")
        a.fill_between(T_GRID, mu - sd, mu + sd, color=COLOURS[lab], alpha=0.20, lw=0)
        a.plot(T_GRID, mu, color=COLOURS[lab], lw=1.6, label=lab)
    # Quote the SAME statistic the summary table prints, not the time-grid mean
    # -- the grid stops at 0.97*duration and reported 61 N against the table's 64.
    f_bad = np.mean([r["f_mean"] for r in by[LABELS[3]]])
    a.set_ylim(0, 26)
    a.annotate(f"{LABELS[3]}: mean {f_bad:.0f} N, off scale",
               (0.03, 0.06), xycoords="axes fraction", color=CRITICAL, fontsize=8.5,
               fontweight="bold", va="bottom")
    a.set_title("normal force held against the surface", color=INK, fontsize=10.5, pad=8)
    a.set_xlabel("time since contact (s)", color=INK2, fontsize=9)
    a.set_ylabel("normal force (N)", color=INK2, fontsize=9)
    a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    # ---- (f) force error, log scale: the failure is 40x, not 40% ----
    a = ax[1, 1]
    xs = np.arange(len(LABELS))
    for i, lab in enumerate(LABELS):
        v = np.array([r["force_result"] for r in by[lab]])
        a.bar(i, v.mean(), width=0.62, color=COLOURS[lab], edgecolor=SURFACE, lw=2.0, zorder=3)
        a.errorbar(i, v.mean(), yerr=v.std(), color=INK2, lw=1.2, capsize=3, zorder=4)
        a.annotate(f"{v.mean():.2f}", (i, v.mean() + v.std()), textcoords="offset points",
                   xytext=(0, 7), ha="center", color=INK, fontsize=8.5, zorder=5)
    a.set_yscale("log")
    a.set_ylim(0.4, 260)
    a.set_xticks(xs); a.set_xticklabels([l.replace("  ", "\n") for l in LABELS], fontsize=8)
    a.set_title("force-tracking error  (RMS vs intended)", color=INK, fontsize=10.5, pad=8)
    a.set_ylabel("N, log scale", color=INK2, fontsize=9)

    # ---- (g) path error, linear ----
    a = ax[1, 2]
    for i, lab in enumerate(LABELS):
        v = np.array([r["path_result"] for r in by[lab]]) * 1000
        a.bar(i, v.mean(), width=0.62, color=COLOURS[lab], edgecolor=SURFACE, lw=2.0, zorder=3)
        a.errorbar(i, v.mean(), yerr=v.std(), color=INK2, lw=1.2, capsize=3, zorder=4)
        a.annotate(f"{v.mean():.2f}", (i, v.mean() + v.std()), textcoords="offset points",
                   xytext=(0, 7), ha="center", color=INK, fontsize=8.5, zorder=5)
    a.set_ylim(0, 56)
    a.set_xticks(xs); a.set_xticklabels([l.replace("  ", "\n") for l in LABELS], fontsize=8)
    a.set_title("path error  (RMS off the intended spiral)", color=INK, fontsize=10.5, pad=8)
    a.set_ylabel("mm", color=INK2, fontsize=9)

    # ---- (h) the mechanism: Eq. 15's boundary, and where each run sits ----
    a = ax[1, 3]
    task = S.SpiralTask()
    kes = np.logspace(2.3, 4.3, 300)
    dmin = np.array([A.routh_da_min(task.Ma, task.Ka, k, 0.020) for k in kes])
    a.fill_between(kes, 0, dmin, color=CRITICAL, alpha=0.13, lw=0)
    a.plot(kes, dmin, color=CRITICAL, lw=1.8, label="Eq. 15 boundary, $T_s$=20 ms")
    dmin5 = np.array([A.routh_da_min(task.Ma, task.Ka, k, 0.005) for k in kes])
    a.plot(kes, dmin5, color=INK2, lw=1.3, ls=(0, (4, 3)), label="$T_s$=5 ms")
    a.annotate("UNSTABLE", (0.93, 0.05), xycoords="axes fraction", color=CRITICAL,
               fontsize=9.5, fontweight="bold", ha="right")
    a.scatter([task.ke], [task.Ba + task.Br], s=64, color=CRITICAL, zorder=6,
              edgecolor=SURFACE, lw=1.6)
    a.annotate(f"Case 2  da=65\n(the cascade's own gains,\n{A.routh_da_min(task.Ma, task.Ka, task.ke, 0.020) - (task.Ba + task.Br):.1f} Ns/m short)",
               (task.ke, task.Ba + task.Br), textcoords="offset points",
               xytext=(-14, -30), ha="right", color=CRITICAL, fontsize=8.5, zorder=7)
    a.scatter([task.ke], [150.0], s=64, color=SLOT[2], zorder=6, edgecolor=SURFACE, lw=1.6)
    a.annotate("Case 2  da=150", (task.ke, 150.0), textcoords="offset points",
               xytext=(-10, 8), ha="right", color=SLOT[2], fontsize=8.5, zorder=7)
    a.set_xscale("log")
    a.set_ylim(0, 260)
    a.set_title("why Case 2 fails here, and only here", color=INK, fontsize=10.5, pad=8)
    a.set_xlabel("environment stiffness $k_e$ (N/m)", color=INK2, fontsize=9)
    a.set_ylabel("total damping $d_a=b_a+b_r$ (Ns/m)", color=INK2, fontsize=9)
    a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    fig.suptitle("The same spiral wipe under three executors: one operator, one surface, one dt",
                 color=INK, fontsize=13.5, y=0.985)
    fig.text(0.5, 0.945, "Cascade and Case 1 are near-identical here; Case 2 is stable only "
             "once its damping clears the Eq. 15 boundary the cascade never has to satisfy.",
             ha="center", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="case_spiral.png")
    args = ap.parse_args()

    jobs = [(s, lab) for lab in LABELS for s in range(args.seeds)]
    # Capped: each worker's path metric holds a (chunk, N_curve) block and a
    # full episode log, so oversubscribing cores here buys nothing and risks
    # the OOM that an uncapped pool actually hit.
    with mp.Pool(min(8, mp.cpu_count(), len(jobs))) as pool:
        rows = pool.map(episode, jobs)
    by = {lab: [r for r in rows if r["label"] == lab] for lab in LABELS}

    print(f"\n{'executor':<18} {'path RMS':>10} {'force RMS':>11} {'f mean':>9} "
          f"{'f std':>8} {'contact':>9}")
    for lab in LABELS:
        r = by[lab]
        print(f"{lab:<18} {1000*np.mean([x['path_result'] for x in r]):>8.2f}mm "
              f"{np.mean([x['force_result'] for x in r]):>10.2f}N "
              f"{np.mean([x['f_mean'] for x in r]):>8.2f}N "
              f"{np.mean([x['f_std'] for x in r]):>7.2f}N "
              f"{100*np.mean([x['contact'] for x in r]):>8.1f}%")
    figure(by, args.out)


if __name__ == "__main__":
    main()
