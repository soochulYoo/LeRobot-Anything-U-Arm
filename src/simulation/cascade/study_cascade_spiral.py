"""Cascade vs outer-only vs inner-only, on the SAME spiral demonstrations.

The ablation in study_ablation.py swapped controllers under a scripted FR3 wipe.
This one swaps them under the human operator: the same delayed visual loop, the
same delayed haptic loop, the same random surface, the same seed -- only the
follower changes.  That matters because the operator is a feedback controller
too, and a follower that fights the surface is a follower the operator has to
correct for.  Whether the two layers earn their keep is a question about the
whole loop, not about the follower alone.

    cascade      outer admittance (Ka) + compliant inner impedance (Ki 4000)
    outer only   the same admittance, but a STIFF inner loop (Ki 30000): the
                 force is regulated by moving the reference, not by yielding
    inner only   no admittance at all -- the decoded operator command drives the
                 inner impedance directly, so NOTHING in the loop reads force

What the three should trade off, if the cascade story is right: the inner loop
buys quiet contact, the outer loop buys force regulation, and dropping either
one shows up on a different axis of the figure rather than on all of them.

Usage:  python3 study_cascade_spiral.py [--episodes 50] [--out spiral_ablation.png]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import pickle

import numpy as np

import spiral_task as S

CONTROLLERS = ["cascade", "outer only", "inner only"]
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

# One common grid per axis, so 50 episodes of different length can be averaged.
F_BINS = np.linspace(0.0, 30.0, 61)         # normal force histogram, N
_I = S.SpiralIntent()
R_GRID = np.linspace(0.6 * _I.pitch, _I.pitch * _I.turns, 70)   # ideal radius, m
T_GRID = np.linspace(0.0, 0.97 * _I.duration, 200)              # since contact, s


def episode(job: tuple) -> dict:
    """One (seed, controller) demonstration, reduced to what the figure needs."""
    seed, controller = job
    task = S.SpiralTask()
    intent = S.SpiralIntent()
    # Matched across controllers: same operator, same surface, same seed.
    op = S.Operator(seed=seed)
    surface = S.RandomSurface(seed=seed)
    log = S.demonstrate(task, intent, op, surface, controller)
    g = S.gap(log)

    t = np.asarray(log["t"])
    sel = t >= task.settle
    n = task.R[:, 2]
    t1, t2 = task.R[:, 0], task.R[:, 1]
    f = np.asarray(log["f_normal"])[sel, 0]
    ts = t[sel] - task.settle

    # In-plane tool position, expressed in the surface's own two axes.
    rel = np.asarray(log["x"])[sel] - task.wall * n
    xy = np.column_stack([rel @ t1, rel @ t2])
    r_ideal, dev = _polar_error(xy, intent.pitch)

    # A diverged run must never be averaged in silently.  "inner only" is
    # unstable when the operator's arm is stiff (>~2000 N/m along the normal):
    # with no admittance in the path, the visual loop's 200 ms delay closes on a
    # near-rigid hand-to-tool map.  The nominal arm here is 225 N/m and stable,
    # but the guard is what makes that a checked fact rather than an assumption.
    peak = float(np.max(np.abs(f)))
    assert peak < 1e3 and np.all(np.isfinite(f)), (
        f"{controller} seed {seed} diverged: peak |f| = {peak:.3g} N")

    out = {
        "seed": seed, "controller": controller,
        "path": g["path_result"], "force_rms": g["force_result"],
        "f_mean": g["f_mean"], "contact": g["contact"],
        "peak": peak,
        # The follower's own contribution to roughness: force energy above 5 Hz,
        # which is faster than either human loop can answer.
        "chatter": _hf_rms(f, float(np.mean(np.diff(t)))),
        # Who yields, in mm along the contact normal.  x_c -> x_r is the
        # admittance; x_r -> x is the inner impedance.
        "defl_outer": float(np.mean(
            (np.asarray(log["x_c"])[sel] - np.asarray(log["x_r"])[sel]) @ n)) * 1000,
        "defl_inner": float(np.mean(
            (np.asarray(log["x_r"])[sel] - np.asarray(log["x"])[sel]) @ n)) * 1000,
        # At the full rate, not from the 200-point time grid: the whole question
        # about inner-only is its tail, and a decimated trace loses the tail.
        "f_hist": np.histogram(f, bins=F_BINS, density=True)[0],
        "dev_vs_r": _resample(r_ideal, dev, R_GRID),
        "f_vs_t": _resample(ts, f, T_GRID),
    }
    if seed == 0:                                # one episode drawn in full
        out["xy"] = xy[::10]
    return out


def _polar_error(xy: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial deviation from r = pitch*theta/2pi, against the ideal radius.

    The same measure plot_demos.py uses: what "constant spacing" means is the
    right radius at the right angle, which a distance-to-curve hides.  The first
    half-turn is dropped because the angle is ill-conditioned near the centre.
    """
    r = np.linalg.norm(xy, axis=1)
    th = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    th = th - th[0]
    # One scalar per episode: the phase that best explains the measured radii.
    # d/dphi of sum (r - pitch*(th+phi)/2pi)^2 = 0.  Without this the band across
    # episodes is dominated by where each one happened to start, not by spacing.
    keep0 = pitch * th / (2 * np.pi) > 0.5 * pitch
    if keep0.sum() >= 2:
        phi = (2 * np.pi / pitch) * np.mean(r[keep0]) - np.mean(th[keep0])
        th = th + phi
    r_ideal = pitch * th / (2 * np.pi)
    keep = r_ideal > 0.5 * pitch
    return r_ideal[keep], (r - r_ideal)[keep]


def _resample(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Onto a common grid, NaN outside the episode's own span.

    np.interp would clamp to the end values instead, which would draw a flat
    tail that no episode actually contains.
    """
    if x.size < 2:
        return np.full(grid.shape, np.nan)
    order = np.argsort(x)
    out = np.interp(grid, x[order], y[order])
    return np.where((grid >= x[order][0]) & (grid <= x[order][-1]), out, np.nan)


def _hf_rms(f: np.ndarray, dt: float) -> float:
    """RMS of the force above 5 Hz: what the operator cannot answer."""
    if f.size < 8 or not np.isfinite(dt) or dt <= 0:
        return float("nan")
    spec = np.fft.rfft(f - np.mean(f))
    freq = np.fft.rfftfreq(f.size, dt)
    spec[freq < 5.0] = 0.0
    return float(np.sqrt(np.mean(np.fft.irfft(spec, n=f.size) ** 2)))


def band(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and +-1 sd across episodes, ignoring grid points nobody reached."""
    M = np.array([r[key] for r in rows], dtype=float)
    keep = np.sum(np.isfinite(M), axis=0) >= max(3, 0.5 * len(rows))
    mu = np.full(keep.shape, np.nan)
    sd = np.full(keep.shape, np.nan)
    # Only the kept columns go to nanmean; an all-NaN column would warn and the
    # warning would be the only sign that the grid runs past the episodes.
    mu[keep] = np.nanmean(M[:, keep], axis=0)
    sd[keep] = np.nanstd(M[:, keep], axis=0)
    return mu, sd, keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="spiral_ablation.png")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="spiral_ablation.pkl")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cache instead of re-simulating")
    args = ap.parse_args()

    cache = pathlib.Path(args.cache)
    if args.replot:
        # 150 demonstrations is a few minutes; redrawing should not cost that.
        rows = pickle.loads(cache.read_bytes())
        assert len({r["seed"] for r in rows}) == args.episodes, (
            f"cache holds {len({r['seed'] for r in rows})} episodes, asked for "
            f"{args.episodes} -- rerun without --replot")
        print(f"replotting {len(rows)} runs from {cache}")
    else:
        jobs = [(s, c) for c in CONTROLLERS for s in range(args.episodes)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(episode, jobs)
        cache.write_bytes(pickle.dumps(rows))
    by = {c: [r for r in rows if r["controller"] == c] for c in CONTROLLERS}

    target = S.SpiralIntent().f_normal
    print(f"\nSPIRAL WIPE -- {args.episodes} episodes each, matched operator and surface")
    print("=" * 92)
    print("  controller     spacing err      force err       mean force"
          "       peak      >5 Hz    contact")
    print("  " + "-" * 88)
    for c in CONTROLLERS:
        R = by[c]
        def ms(k, scale=1.0):
            v = np.array([r[k] for r in R]) * scale
            return f"{v.mean():6.2f}+-{v.std():4.2f}"
        v_peak = np.array([r["peak"] for r in R])
        print(f"  {c:<13}{ms('path', 1000)} mm  {ms('force_rms')} N  {ms('f_mean')} N"
              f"  {v_peak.mean():5.1f}+-{v_peak.std():4.1f} N  {ms('chatter')} N"
              f"   {100*np.mean([r['contact'] for r in R]):3.0f}%")
    print("  " + "-" * 88)

    draw(by, target, args)


def draw(by: dict, target: float, args) -> None:
    """One row per controller, four views per row.

    The three shared one set of axes before and the traces landed on top of one
    another -- cascade and outer-only are within 5% of each other, so the figure
    hid its own main result.  A row each, on SHARED LIMITS, with the other two
    greyed in behind, shows both the overlap and the one place the three
    genuinely part company.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 4, figsize=(16.0, 10.2), facecolor=SURFACE)
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
    fc = 0.5 * (F_BINS[:-1] + F_BINS[1:])

    # Shared limits, so a row is read against the others and not against itself.
    stats = {c: (band(by[c], "dev_vs_r"), band(by[c], "f_vs_t")) for c in CONTROLLERS}
    dlim = 1.05 * max(np.nanmax(np.abs(d[0]) + d[1]) for d, _ in stats.values()) * 1000
    flim = (0.0, 1.05 * max(np.nanmax(f[0] + f[1]) for _, f in stats.values()))
    hists = {c: np.mean([r["f_hist"] for r in by[c]], axis=0) for c in CONTROLLERS}
    hmax = 1.08 * max(h.max() for h in hists.values())

    for i, (c, col) in enumerate(zip(CONTROLLERS, SLOT)):
        rows = by[c]

        # 1 -- the path, against the spiral they meant to draw
        a = ax[i, 0]
        xy = next((r["xy"] for r in rows if "xy" in r), None)
        if xy is not None:
            a.plot(xy[:, 0] * 1000, xy[:, 1] * 1000, color=col, lw=1.6, alpha=0.95)
        a.plot(rr * np.cos(th) * 1000, rr * np.sin(th) * 1000, color=INK,
               lw=1.0, ls=(0, (5, 4)), zorder=6)
        a.set_aspect("equal")
        a.set_ylabel(c, color=INK, fontsize=12, labelpad=12)
        # The deflection budget, in words: two numbers, and they are the reason
        # the first two rows look alike.
        do = float(np.mean([r["defl_outer"] for r in rows]))
        di = float(np.mean([r["defl_inner"] for r in rows]))
        a.annotate("admittance yields %5.2f mm\ninner loop yields %5.2f mm" % (do, di),
                   (0.02, 0.02), xycoords="axes fraction", color=INK2,
                   fontsize=8, family="monospace", va="bottom", zorder=8,
                   bbox=dict(facecolor=SURFACE, edgecolor="none",
                             alpha=0.88, pad=2.0))
        if i == 0:
            a.set_title("path  (dashed = intended spiral)", color=INK,
                        fontsize=10.5, loc="left")
        if i == 2:
            a.set_xlabel("x  [mm]", color=INK2, fontsize=9)

        # 2 -- spacing error vs radius, with the other two greyed in
        a = ax[i, 1]
        for other in CONTROLLERS:
            if other != c:
                a.plot(R_GRID * 1000, stats[other][0][0] * 1000,
                       color=INK3, lw=1.2, alpha=0.55)
        mu, sd, _ = stats[c][0]
        a.fill_between(R_GRID * 1000, (mu - sd) * 1000, (mu + sd) * 1000,
                       color=col, alpha=0.18, lw=0)
        a.plot(R_GRID * 1000, mu * 1000, color=col, lw=2.2)
        a.axhline(0.0, color=INK, lw=1.0, ls=(0, (5, 4)))
        a.set_ylim(-dlim, dlim)
        rms = 1000 * np.mean([np.sqrt(np.nanmean(np.asarray(r["dev_vs_r"]) ** 2))
                              for r in rows])
        a.annotate("radial RMS %.2f mm" % rms,
                   (0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                   color=INK, fontsize=9)
        if i == 0:
            a.set_title("spacing error, mean $\\pm$1 sd of %d" % args.episodes,
                        color=INK, fontsize=10.5, loc="left")
        if i == 2:
            a.set_xlabel("ideal radius  [mm]", color=INK2, fontsize=9)
        a.set_ylabel("radial error  [mm]", color=INK2, fontsize=9)

        # 3 -- force against the force they meant to hold
        a = ax[i, 2]
        for other in CONTROLLERS:
            if other != c:
                a.plot(T_GRID, stats[other][1][0], color=INK3, lw=1.2, alpha=0.55)
        mu, sd, _ = stats[c][1]
        a.fill_between(T_GRID, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        a.plot(T_GRID, mu, color=col, lw=2.2)
        a.axhline(target, color=INK, lw=1.0, ls=(0, (5, 4)))
        a.set_ylim(*flim)
        a.annotate("RMS %.2f N" % np.mean([r["force_rms"] for r in rows]),
                   (0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                   color=INK, fontsize=9)
        if i == 0:
            a.annotate("intended %.0f N" % target, (T_GRID[-1], target), color=INK,
                       fontsize=8, ha="right", va="top",
                       textcoords="offset points", xytext=(0, -4))
            a.set_title("normal force held", color=INK, fontsize=10.5, loc="left")
        if i == 2:
            a.set_xlabel("time since contact  [s]", color=INK2, fontsize=9)
        a.set_ylabel("force  [N]", color=INK2, fontsize=9)

        # 4 -- the whole force distribution.  The mean in column 3 hides the
        # tail, and the tail is what a wipe actually fails on.
        a = ax[i, 3]
        for other in CONTROLLERS:
            if other != c:
                a.plot(fc, hists[other], color=INK3, lw=1.2, alpha=0.55)
        a.fill_between(fc, 0, hists[c], color=col, alpha=0.30, lw=0)
        a.plot(fc, hists[c], color=col, lw=2.2)
        a.axvline(target, color=INK, lw=1.0, ls=(0, (5, 4)))
        a.set_ylim(0, hmax)
        a.annotate("peak %.0f N\nin contact %.0f%%"
                   % (np.mean([r["peak"] for r in rows]),
                      100 * np.mean([r["contact"] for r in rows])),
                   (0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
                   color=INK, fontsize=9)
        if i == 0:
            a.set_title("force distribution, all samples", color=INK,
                        fontsize=10.5, loc="left")
        if i == 2:
            a.set_xlabel("normal force  [N]", color=INK2, fontsize=9)
        a.set_ylabel("density", color=INK2, fontsize=9)

    fig.suptitle("Outward-spiral wipe under a human operator: cascade vs each"
                 " layer alone", color=INK, fontsize=13.5, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print("\n  wrote %s" % args.out)


if __name__ == "__main__":
    main()
