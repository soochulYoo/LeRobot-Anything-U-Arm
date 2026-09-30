"""Draw what the 300 demonstrations actually contain.

Three things, because three things differ and the difference is the point:

  the INTENT      an Archimedean spiral of fixed pitch -- constant spacing
  the OPERATOR    what they commanded, which is not the intent: they are
                  correcting a surface they cannot see, 200 ms late
  the ROBOT       where the tool went, which lags the command through the
                  coupling, the admittance and the inner impedance

and the force, which is the other half of the intent and the half a pose-only
record throws away.

Spacing is measured in POLAR coordinates against the ideal r = pitch * theta /
2pi, not as a distance to the curve.  The distance-to-curve is small (2.2 mm)
because the tool is always near SOME part of the spiral; what a wipe needs is
the right radius at the right angle, and that is what "constant spacing" means.

Usage:  python3 plot_demos.py [--demos demos/spiral_v1] [--out demo_overview.png]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"


def inplane(p: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    q = p - origin
    return np.column_stack([q @ R[:, 0], q @ R[:, 1]])


def polar_error(xy: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial deviation from the ideal spiral, against unwrapped angle.

    Returns (radius_of_the_ideal_at_that_angle, deviation).  Points inside the
    first half-turn are dropped: the angle is ill-conditioned near the centre
    and a millimetre of wobble there reads as a huge radial error.
    """
    r = np.linalg.norm(xy, axis=1)
    th = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    th = th - th[0]
    r_ideal = pitch * th / (2 * np.pi)
    keep = r_ideal > 0.5 * pitch
    return r_ideal[keep], (r - r_ideal)[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="demos/spiral_v1")
    ap.add_argument("--out", default="demo_overview.png")
    ap.add_argument("--show-ep", type=int, default=0)
    args = ap.parse_args()

    root = pathlib.Path(args.demos)
    meta = json.loads((root / "index.json").read_text())
    eps = [e for e in meta["episodes"] if e["ok"]]
    print(f"{len(eps)} demonstrations from {root}")

    grids = np.linspace(0.0, 1.0, 40)
    dev_rob, dev_hum, force = [], [], []
    for e in eps:
        d = np.load(root / f"ep_{e['seed']:05d}.npz")
        R = d["task_frame"]
        org = d["obs_x"][0]
        rob = inplane(d["obs_x"], R, org)
        hum = inplane(d["diag_x_c"], R, org)
        for store, xy in ((dev_rob, rob), (dev_hum, hum)):
            r_i, dv = polar_error(xy, e["pitch"])
            if len(r_i) > 10:
                u = np.clip(r_i / r_i.max(), 0, 1)
                store.append(np.interp(grids, u, dv))
        f = d["obs_f_normal"] / float(d["intent_f"])
        force.append(np.interp(np.linspace(0, 1, 200),
                               np.linspace(0, 1, len(f)), f))
    dev_rob = np.vstack(dev_rob); dev_hum = np.vstack(dev_hum)
    force = np.vstack(force)

    print(f"  spacing error, robot    {1000*np.abs(dev_rob).mean():5.2f} mm mean, "
          f"{1000*dev_rob.std():5.2f} mm s.d.")
    print(f"  spacing error, operator {1000*np.abs(dev_hum).mean():5.2f} mm mean, "
          f"{1000*dev_hum.std():5.2f} mm s.d.")
    print(f"  force / intended        {force.mean():5.3f} +- {force.std():.3f}")
    plot(root, eps, args.show_ep, grids, dev_rob, dev_hum, force, args.out)


def plot(root, eps, show_ep, grids, dev_rob, dev_hum, force, out_path):
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
    fig, ax = plt.subplots(1, 3, figsize=(15.8, 5.2))

    # --- A: one episode, all three ---
    e = next(x for x in eps if x["seed"] == show_ep)
    d = np.load(root / f"ep_{show_ep:05d}.npz")
    R, org = d["task_frame"], d["obs_x"][0]
    itn = np.column_stack([d["intent_xy"] @ R[:, 0], d["intent_xy"] @ R[:, 1]])
    a = ax[0]
    a.plot(1000 * itn[:, 0], 1000 * itn[:, 1], color=INK, lw=2.2, ls="--", zorder=4,
           label=f"intent: {1000*e['pitch']:.0f} mm pitch")
    a.plot(*(1000 * inplane(d["diag_x_c"], R, org)).T, color=SLOT[1], lw=1.5, zorder=3,
           label="operator's command")
    a.plot(*(1000 * inplane(d["obs_x"], R, org)).T, color=SLOT[0], lw=1.5, zorder=2,
           label="robot")
    a.set_aspect("equal")
    a.set_xlabel("surface tangent $t_1$ (mm)"); a.set_ylabel("surface tangent $t_2$ (mm)")
    a.set_title(f"One demonstration (episode {show_ep})", color=INK, loc="left")
    a.grid(True, alpha=0.9); a.set_axisbelow(True)
    a.legend(fontsize=7.5, loc="upper right")

    # --- B: spacing error over all episodes ---
    b = ax[1]
    for i, (dv, nm) in enumerate(((dev_hum, "operator's command"), (dev_rob, "robot"))):
        c = SLOT[1] if i == 0 else SLOT[0]
        mu, sd = 1000 * dv.mean(axis=0), 1000 * dv.std(axis=0)
        b.fill_between(grids, mu - sd, mu + sd, color=c, alpha=0.22, lw=0)
        b.plot(grids, mu, color=c, lw=2.0, label=f"{nm}   |err| {np.abs(1000*dv).mean():.1f} mm")
    b.axhline(0.0, color=INK, ls="--", lw=1.4)
    b.annotate(" the intended spacing", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
               color=INK, fontsize=8, va="bottom")
    b.set_xlabel("fraction of the way out along the spiral")
    b.set_ylabel("radial deviation from the ideal spiral (mm)")
    b.set_title(f"Spacing error over {dev_rob.shape[0]//1} runs, band = 1 s.d.",
                color=INK, loc="left")
    b.grid(True, alpha=0.9); b.set_axisbelow(True)
    b.legend(fontsize=7.5, loc="upper left")

    # --- C: force ---
    c = ax[2]
    u = np.linspace(0, 1, force.shape[1])
    mu, sd = force.mean(axis=0), force.std(axis=0)
    c.fill_between(u, mu - sd, mu + sd, color=SLOT[2], alpha=0.25, lw=0)
    c.plot(u, mu, color=SLOT[2], lw=2.0, label=f"mean {mu.mean():.3f} of intended")
    c.plot(u, np.percentile(force, 5, axis=0), color=SLOT[2], lw=0.9, ls=":",
           label="5th / 95th percentile")
    c.plot(u, np.percentile(force, 95, axis=0), color=SLOT[2], lw=0.9, ls=":")
    c.axhline(1.0, color=INK, ls="--", lw=1.4)
    c.annotate(" the intended force", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
               color=INK, fontsize=8, va="bottom")
    c.set_xlabel("fraction of the way through the spiral")
    c.set_ylabel("contact force / intended force")
    c.set_title("Force, normalised by each run's own target", color=INK, loc="left")
    c.set_ylim(0.3, 1.8)
    c.grid(True, alpha=0.9); c.set_axisbelow(True)
    c.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(f"{len(eps)} spiral-wipe demonstrations: the operator commands a correction they cannot make in time, and the robot follows that -- neither is the intent.",
                 color=INK, fontsize=10.5, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out_path, dpi=170)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
