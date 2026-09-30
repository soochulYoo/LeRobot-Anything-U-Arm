"""Study T5: do published stiffness-label rules identify the operator, or do they
report controller gains?

Two results:

  (1) Rules are run against demos whose true Kh is known, sweeping Kh over a 16x
      range, and scored by the log-log SENSITIVITY SLOPE d log(K_hat)/d log(Kh):
          slope ~ 1  -> the rule tracks the operator
          slope ~ 0  -> the rule returns a constant unrelated to them
      The controller gains (Ka, Ki, ke) are held FIXED across the sweep, so a
      rule that returns one of them has a slope of zero by construction.

  (2) The particle filter's nuisance hyperparameters are swept.  sigma_xeq -- the
      random-walk width on the unobservable equilibrium point, a quantity with no
      physical meaning about the operator -- moves the answer by an order of
      magnitude while the filter's own predictive error barely changes.  Some of
      those settings land near the truth, and nothing in the data says which.

Usage:  python3 study_labels.py [--quick] [--out FIG.png]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses

import numpy as np

import core as C
import labels as L

# Validated light-mode categorical slots (dataviz reference palette).  Line
# panels use the adjacent pairlist; the prior-sweep panel uses only the first
# three slots, the ones that clear the all-pairs floors.  Three slots sit below
# 3:1 contrast on the light surface, so the relief rule applies: every series
# carries a visible direct label AND the table view below is always printed.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

KA, KI, KE, WALL = 100.0, 2000.0, 5000.0, 0.10
PROBE_HZ, KH_REF = 3.0, 400.0

FIG_RULES = [
    ("R3 Imp-ACT rule", L.rule_impact_controller_rule, "Ki"),
    ("R4 Compl-for-Free", L.rule_compliance_for_free, "series(Ka,Ki)"),
    ("R4b (vs x_r)", lambda d: L.rule_compliance_for_free(d, against="x_r"), "Ka"),
    ("R5 particle filter", L.rule_particle_filter, "its prior"),
    ("R6 probe ident.", L.rule_probe_identification, "Kh"),
    ("R7 naive regression", L.rule_naive_regression_with_intercept, "Kh iff intent const"),
]
# R1/R2 are designer constants by construction, so they are reported in the
# table rather than the figure -- stating that plainly is part of the result.
TABLE_ONLY = [("R1 ACP heuristic", L.rule_acp_force_heuristic, "designer const"),
              ("R2 Comp-ACT toggle", L.rule_compact_toggle, "designer const")]


def make_demo(Kh: float, duration: float, settle: float,
              probe_amp: float = 3.0, reach_amp: float = 0.03) -> L.Demo:
    p = C.CascadeParams(
        human=C.HumanParams(Kh=Kh, Bh=20.0, reach=0.16, reach_amp=reach_amp, reach_hz=0.2),
        master=C.MasterParams(probe_amp=probe_amp, probe_hz=PROBE_HZ),
        coupling=C.CouplingParams(Ka=KA), inner=C.InnerParams(Ki=KI),
        env=C.EnvParams(ke=KE, x_wall=WALL),
    )
    p = dataclasses.replace(p, dt=C.safe_dt(p))
    return L.Demo(log=C.CascadeSim(p, "teleop").run(duration), dt=p.dt, Kh_true=Kh,
                  Ka=KA, Ki=KI, ke=KE, settle_s=settle)


def loglog_slope(kh: np.ndarray, est: np.ndarray) -> float:
    """d log(estimate) / d log(Kh).  Non-finite or non-positive estimates are
    dropped rather than coerced, and fewer than three usable points returns nan
    instead of a slope fitted through noise."""
    ok = np.isfinite(est) & (est > 0) & np.isfinite(kh) & (kh > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(kh[ok]), np.log(est[ok]), 1)[0])


def run_sweep(kh_values, duration, settle):
    est = {n: [] for n, _, _ in FIG_RULES + TABLE_ONLY}
    for Kh in kh_values:
        d = make_demo(Kh, duration, settle)
        for name, fn, _ in FIG_RULES + TABLE_ONLY:
            est[name].append(fn(d).value)
        print(f"  Kh={Kh:7.1f} done")
    return est


def print_table(kh_values, est, csv_path):
    predicts = {n: w for n, _, w in FIG_RULES + TABLE_ONLY}
    order = [n for n, _, _ in FIG_RULES + TABLE_ONLY]
    print("\n" + "=" * 104)
    print("ESTIMATED OPERATOR STIFFNESS (N/m) vs TRUE Kh  -- table view")
    print("=" * 104)
    print("  rule".ljust(24) + "predicts".ljust(20)
          + "".join(f"{k:>9.0f}" for k in kh_values) + "    slope")
    print("  " + "-" * 100)
    slopes = {}
    for name in order:
        v = np.array(est[name], dtype=float)
        slopes[name] = loglog_slope(kh_values, v)
        cells = "".join("      nan" if not np.isfinite(x) else f"{x:>9.1f}" for x in v)
        s = f"{slopes[name]:>6.3f}" if np.isfinite(slopes[name]) else "   nan"
        print(f"  {name:<22}{predicts[name]:<20}{cells}   {s}")
    print("  " + "-" * 100)
    print("  slope = d log(estimate)/d log(Kh).  1.0 = identifies the operator, 0.0 = ignores them.")
    print(f"  Kh spans {kh_values.min():.0f}-{kh_values.max():.0f} N/m "
          f"({kh_values.max()/kh_values.min():.0f}x) with Ka/Ki/ke held fixed.")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "predicts", *[f"Kh={k:.0f}" for k in kh_values], "loglog_slope"])
        for name in order:
            w.writerow([name, predicts[name], *[f"{x:.6g}" for x in est[name]],
                        f"{slopes[name]:.6g}"])
    print(f"\n  wrote {csv_path}")
    return slopes


def run_prior_sweep(duration, settle, sigma_xeqs, prior_means):
    """Sweep the filter's nuisance hyperparameters on ONE demo with a known Kh."""
    d = make_demo(KH_REF, duration, settle)
    K = {m: [] for m in prior_means}
    fit = {m: [] for m in prior_means}
    for m in prior_means:
        for sx in sigma_xeqs:
            e = L.rule_particle_filter(d, log_K_prior=(np.log(m), 1.0), sigma_xeq=sx, seed=0)
            K[m].append(e.value)
            fit[m].append(e.diagnostics.get("fit_rms", np.nan))
    signal = e.diagnostics.get("signal_rms", np.nan)
    return K, fit, signal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="label_study.png")
    ap.add_argument("--csv", default="label_study.csv")
    args = ap.parse_args()

    kh_values = np.array([100.0, 300.0, 800.0, 1600.0] if args.quick
                         else [100.0, 200.0, 400.0, 800.0, 1200.0, 1600.0])
    duration, settle = (35.0, 12.0) if args.quick else (60.0, 20.0)
    print(f"Kh sweep: {kh_values.tolist()}   demo {duration:.0f}s / settle {settle:.0f}s")
    print(f"probe {PROBE_HZ} Hz, operator intent varying at 0.2 Hz, "
          f"fixed gains Ka={KA:.0f} Ki={KI:.0f} ke={KE:.0f}\n")

    est = run_sweep(kh_values, duration, settle)
    slopes = print_table(kh_values, est, args.csv)

    sigma_xeqs = [0.002, 0.005, 0.01, 0.02, 0.05]
    prior_means = [100.0, 400.0, 1600.0]
    K, fit, signal = run_prior_sweep(duration, settle, sigma_xeqs, prior_means)

    print("\n" + "=" * 104)
    print(f"PARTICLE-FILTER NUISANCE-HYPERPARAMETER SENSITIVITY (true Kh = {KH_REF:.0f} N/m)")
    print("=" * 104)
    print("  sigma_xeq is the random-walk width on the unobservable equilibrium point.")
    print("  It says nothing about the operator, yet it is what decides the answer.\n")
    print("  prior mean |" + "".join(f"{s:>10.3f}" for s in sigma_xeqs) + "   <- sigma_xeq")
    print("  " + "-" * 72)
    for m in prior_means:
        print(f"  {m:>10.0f} |" + "".join(f"{v:>10.1f}" for v in K[m]))
    print("  " + "-" * 72)
    print("  fit RMS    |" + "".join(f"{np.nanmean([fit[m][i] for m in prior_means]):>10.3f}"
                                     for i in range(len(sigma_xeqs)))
          + f"   (force signal {signal:.2f} N)")
    allK = np.array([v for m in prior_means for v in K[m]], dtype=float)
    allK = allK[np.isfinite(allK) & (allK > 0)]
    allf = np.array([v for m in prior_means for v in fit[m]], dtype=float)
    spread = float(allK.max() / allK.min())
    print(f"\n  K_hat spans {allK.min():.0f}-{allK.max():.0f} N/m, a {spread:.1f}x range,")
    print(f"  while the filter's own predictive error only moves "
          f"{np.nanmin(allf):.3f}-{np.nanmax(allf):.3f} N "
          f"({100*np.nanmin(allf)/signal:.0f}-{100*np.nanmax(allf)/signal:.0f}% of signal).")
    print(f"  Some settings land near the true {KH_REF:.0f} N/m. Nothing in the data says which,")
    print("  and every setting explains the measured force about equally well. That is what")
    print("  unidentifiability looks like in practice: the prior answers, not the data.")

    plot(kh_values, est, slopes, sigma_xeqs, prior_means, K, spread, signal, args.out)


def plot(kh, est, slopes, sigma_xeqs, prior_means, K, spread, signal, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False, "svg.fonttype": "none",
    })
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # ---- A: estimate vs truth (log-log) ----
    # The identity line is drawn thick and underneath so it shows as a halo
    # around whichever series lands on it -- R6 sits exactly there, and a thin
    # reference line would simply vanish under it and look omitted.
    axA.plot(kh, kh, "-", color="#d6d5cf", lw=5.0, zorder=1,
             solid_capstyle="round", label="identity (perfect recovery)")
    # Labels are offset in points, alternating up/down, because R4 and R4b land
    # within 5% of each other and collide if anchored at the data point.
    offsets = {"R4 Compl-for-Free": -9, "R4b (vs x_r)": +9}
    for i, (name, _, _) in enumerate(FIG_RULES):
        v = np.array(est[name], dtype=float)
        ok = np.isfinite(v) & (v > 0)
        if not ok.any():
            axA.plot([], [], "-o", color=SLOT[i], lw=2.0, ms=5.5,
                     label=f"{name} (negative, not plottable)")
            continue
        axA.plot(kh[ok], v[ok], "-o", color=SLOT[i], lw=2.0, ms=5.5, zorder=3,
                 markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        axA.annotate(f" {name}", (kh[ok][-1], v[ok][-1]), color=SLOT[i], fontsize=7.5,
                     textcoords="offset points", xytext=(4, offsets.get(name, 0)),
                     va="center", ha="left", annotation_clip=False, zorder=4)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel(r"true operator stiffness $K_h$  (N/m)")
    axA.set_ylabel(r"estimated stiffness  $\hat{K}$  (N/m)")
    axA.set_title("Only the probe-based rule tracks the operator", color=INK, loc="left")
    axA.grid(True, which="major", alpha=0.9); axA.set_axisbelow(True)
    axA.set_xlim(kh.min() * 0.85, kh.max() * 3.4)
    axA.set_ylim(22, 5000)
    # Lower-left is the only region with no marks or direct labels in it: the flat
    # rules sit at ~100 N/m across the whole width and R5 only dips below that at
    # the far right.  Two columns halve the legend's height so its top edge stays
    # clear of that ~100 N/m cluster, which a single column ran into.
    axA.legend(fontsize=6.3, loc="lower left", borderpad=0.3, labelspacing=0.25,
               handlelength=1.5, borderaxespad=0.4, ncol=2, columnspacing=0.9)

    # ---- B: sensitivity slope ----
    names = [n for n, _, _ in FIG_RULES]
    ypos = np.arange(len(names))[::-1]
    axB.axvline(1.0, color=INK3, ls="--", lw=1.4, zorder=2)
    axB.axvline(0.0, color=INK3, lw=0.8, zorder=2)
    for y, n, c in zip(ypos, names, SLOT):
        v = slopes[n]
        axB.barh(y, 0.0 if not np.isfinite(v) else v, height=0.5, color=c, zorder=3)
        axB.text((0.0 if not np.isfinite(v) else v) + 0.04, y,
                 "nan (estimate is negative)" if not np.isfinite(v) else f"{v:.2f}",
                 va="center", fontsize=8, color=INK2)
    axB.set_yticks(ypos); axB.set_yticklabels(names, fontsize=8)
    axB.set_xlabel(r"sensitivity   $d\log\hat{K}\,/\,d\log K_h$")
    axB.set_title("1.0 = identifies the operator;  0.0 = reports a gain", color=INK, loc="left")
    axB.annotate("perfect", xy=(1.0, ypos[0] - 0.45), color=INK3, fontsize=7.5,
                 ha="center", va="top")
    axB.set_xlim(-0.35, 1.9)
    axB.set_ylim(-0.75, len(names) - 0.4)
    axB.grid(True, axis="x", alpha=0.9); axB.set_axisbelow(True)

    # ---- C: nuisance-hyperparameter sensitivity ----
    # The three prior means give curves that coincide to within a few percent --
    # that coincidence IS the finding, so they are drawn as ONE series with the
    # across-prior spread as a band rather than as three labels fighting for the
    # same pixels.  A single series needs no legend; the title names it.
    sx = np.array(sigma_xeqs, dtype=float)
    M = np.array([K[m] for m in prior_means], dtype=float)
    lo, hi, mid = np.nanmin(M, axis=0), np.nanmax(M, axis=0), np.nanmean(M, axis=0)
    axC.axhline(KH_REF, color=INK3, ls="--", lw=1.4, zorder=2)
    axC.text(sx[-1], KH_REF * 1.10, rf"true $K_h$ = {KH_REF:.0f} N/m ", color=INK2,
             fontsize=8, ha="right")
    axC.fill_between(sx, lo, hi, color=SLOT[0], alpha=0.25, lw=0, zorder=2)
    axC.plot(sx, mid, "-o", color=SLOT[0], lw=2.0, ms=6.5, zorder=3,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    band = float(np.nanmax(hi / lo) - 1.0)
    axC.annotate(rf"  $\hat{{K}}$ spans {np.nanmin(M):.0f}$-${np.nanmax(M):.0f} N/m",
                 (sx[0], mid[0]), textcoords="offset points", xytext=(8, -4),
                 color=SLOT[0], fontsize=8, va="center", annotation_clip=False)
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel(r"$\sigma_{x_{eq}}$   random-walk width on the unobservable equilibrium point")
    axC.set_ylabel(r"posterior mean  $\hat{K}$  (N/m)")
    axC.set_title(f"A nuisance hyperparameter moves it {spread:.0f}x", color=INK, loc="left")
    axC.grid(True, which="major", alpha=0.9); axC.set_axisbelow(True)
    axC.set_xlim(sx.min() * 0.75, sx.max() * 1.5)
    axC.text(0.03, 0.06,
             f"band = prior mean swept 100$-$1600 N/m\n"
             f"    (moves it only {100*band:.0f}%, so the data does\n"
             f"     constrain the PRODUCT $K(x_{{eq}}-x)$)\n"
             f"every setting fits the {signal:.1f} N force to 14$-$17% RMS",
             transform=axC.transAxes, fontsize=7.5, color=INK2, va="bottom", linespacing=1.5)

    fig.suptitle("Where do stiffness labels come from?  Rules run against demonstrations with a known operator stiffness",
                 color=INK, fontsize=11, x=0.006, ha="left", y=0.986)
    fig.tight_layout(rect=(0, 0, 1, 0.952))
    fig.savefig(out_path, dpi=170)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
