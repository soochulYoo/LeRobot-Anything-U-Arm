"""The delta x tilt curves: success against how far the surface moves in contact.

One line per arm of policy/slurm/robust_experiment.sh, pooled over seeds.  Reads
every policy/runs/*/seed*/dz_sweep/dz_sweep.json it can find, so a partial sweep
still plots.

What to read off it:
  * COVERAGE IS THE PANEL THAT MOVES, and the mechanism matters: an over-stiff
    k_n on a tilted paper swings the force below the ink threshold and the pen
    comes off the paper, so the glyph is never written.  in_band averages over
    pen-down steps and therefore cannot see it.
  * THE TWO LAYOUTS ARE TRAPPED DIFFERENTLY, which is the prediction to check.
    Legacy fixes the DEPTH, so force = k_n x depth: soft means weak means no ink,
    stiff means the force swings off the paper on a tilt.  It is squeezed from
    both sides and has an interior optimum it must find per episode.  Spring
    commands f_d directly, so soft keeps 3 N at every tilt and the low-end cost
    disappears -- the spring arm should be able to go soft and hold coverage
    across the whole sweep where the legacy arm cannot.
  * WHERE EACH CURVE FALLS OFF.  At the demonstrations' own 5 deg every
    structure succeeds and commands the demonstrated K to 0.1%
    (policy/runs/COMPARISON.md), so the left edge carries no information.  The
    knee does.
  * CONTACT FORCE vs tilt.  This is the spring layout's own claim: f_d is
    commanded directly, so it should stay flat while the legacy arm's force drifts
    -- in legacy the force IS K times the gap, and tilt moves the gap by
    w tan(theta) across a stroke.
  * THE CROSSOVER between delta = 2 mm and delta = 8 mm.  The 8 mm arm should be
    slightly worse at nominal and better on the right.  No crossover means
    --w-robust changed nothing, and that is a result worth reporting as one.

    python3 policy/plot_stress.py --runs policy/runs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np

# arm -> (label, colour), in the order they should be read
ARMS = [
    (re.compile(r"^(?!spring_).*"), "legacy (x_d + K)", "#8a8880"),
    (re.compile(r"^spring_(?!.*_d\d)"), "spring, K unsupervised", "#2a78d6"),
    (re.compile(r"^spring_.*_d2$"), "spring + robust, delta 2 mm", "#1baf7a"),
    (re.compile(r"^spring_.*_d8$"), "spring + robust, delta 8 mm", "#eb6834"),
]


def arm_of(name: str) -> int | None:
    for i, (pat, _, _) in enumerate(ARMS):
        if pat.match(name):
            return i
    return None


def collect(root: pathlib.Path) -> dict:
    """{arm index: {dz_mm: {metric: [per-seed values]}}}"""
    out: dict = {}
    for f in sorted(root.glob("*/seed*/stress/tilt_sweep.json")):
        name = f.parts[-4]
        a = arm_of(name)
        if a is None:
            continue
        for row in json.loads(f.read_text())["curve"]:
            d = out.setdefault(a, {}).setdefault(round(row["x"]), {})
            for k in ("success", "coverage", "in_band", "contact_f", "peak_force", "contact_kn"):
                d.setdefault(k, []).append(row[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="policy/runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.runs)
    data = collect(root)
    if not data:
        raise SystemExit(f"no */seed*/stress/tilt_sweep.json under {root} -- "
                         "run policy/slurm/robust_experiment.sh first")

    # COVERAGE, NOT in_band, is the panel that moves.  Measured at tilt 5 -> 20
    # deg on act_ft_input/seed0: success 100 -> 50%, coverage 100 -> 64.7%, and
    # in_band held at 99.3%.  in_band is averaged over PEN-DOWN steps only, so an
    # over-stiff k_n that lifts the pen off the paper does not register there at
    # all -- it registers as glyph that never got written.  At 20 deg the surface
    # ranges 13.5 mm across the glyph, so k_n = 1000 N/m swings the force by
    # +-6.7 N about 3 N and the low half of that is below the 0.8 N ink threshold.
    panels = [("success", "success", 100.0, "%"),
              ("coverage", "coverage", 100.0, "%"),
              ("contact_f", "contact force", 1.0, "N"),
              ("peak_force", "peak force", 1.0, "N")]
    print(f"  {'tilt':>7}  " + "".join(f"{lab:>30}" for _, lab, _, _ in panels[:2]))
    for a in sorted(data):
        lab = ARMS[a][1]
        print(f"\n{lab}")
        for dz in sorted(data[a]):
            d = data[a][dz]
            n = len(d["success"])
            print(f"  {dz:7d}  " + "".join(
                f"{sc * np.mean(d[k]):>22.1f} +-{sc * np.std(d[k]):<5.1f}"
                for k, _, sc, _ in panels[:2]) + f"   ({n} seed{'s' if n > 1 else ''})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.6))
    for j, (key, lab, sc, unit) in enumerate(panels):
        for a in sorted(data):
            dzs = sorted(data[a])
            m = np.array([sc * np.mean(data[a][dz][key]) for dz in dzs])
            e = np.array([sc * np.std(data[a][dz][key]) for dz in dzs])
            _, name, col = ARMS[a]
            ax[j].plot(dzs, m, "-o", color=col, ms=4, label=name)
            ax[j].fill_between(dzs, m - e, m + e, color=col, alpha=0.15, lw=0)
        ax[j].set_xlabel("paper tilt (deg)")
        ax[j].set_ylabel(f"{lab} ({unit})")
        # the demonstrations' own range: everything left of this is in distribution
        ax[j].axvspan(0, 5, color="#8a8880", alpha=0.10, lw=0)
        if key == "contact_f":
            ax[j].axhspan(1, 6, color="#1baf7a", alpha=0.08, lw=0)
        if key == "peak_force":
            ax[j].axhline(12.0, color="#d12", lw=0.8, ls="--")   # Criteria.tear_force
        ax[j].grid(alpha=0.2, lw=0.5)
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].set_title("shaded: the demos' own tilt range", fontsize=8, loc="left")
    fig.tight_layout()
    out = pathlib.Path(args.out or root / "STRESS_SWEEP.png")
    fig.savefig(out, dpi=130)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
