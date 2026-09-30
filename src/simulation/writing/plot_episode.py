"""One recorded episode on one page: what was written, how hard, how stiff,
and what the cameras saw.

Usage:  python3 plot_episode.py demos/writing_v1/ep_00001.h5 [--out ep1.png]
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": INK3, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK2,
        "grid.color": "#e6e5e0", "legend.frameon": False,
    })

    with h5py.File(args.path) as f:
        full = {k: f["full"][k][:] for k in f["full"]}
        obs = {k: f["obs"][k] for k in f["obs"]}
        goal_uv, mask = f["goal/strokes_uv"][:], f["goal/mask"][:]
        ink = f["privileged/ink_uv"][:]
        m = json.loads(f.attrs["metrics"])
        crit = json.loads(f.attrs["criteria"])
        phases = json.loads(f.attrs["phases"])
        text = f.attrs["text"]
        n_img = len(obs["t"])
        picks = np.linspace(0, n_img - 1, 4).round().astype(int)
        cams = [k for k in obs if k.startswith("rgb_")]
        thumbs = {k: [obs[k][i] for i in picks] for k in cams}
        t_img = obs["t"][:]

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.3, 1, 1, 1.1])
    ax = fig.add_subplot(gs[0, :2])
    for s, mk in zip(goal_uv, mask):
        if mk.any():
            ax.plot(1000 * s[mk, 0], 1000 * s[mk, 1], color=INK3, lw=7, alpha=0.35,
                    solid_capstyle="round")
    if len(ink):
        ax.plot(1000 * ink[:, 0], 1000 * ink[:, 1], ".", color=SLOT[0], ms=2.5)
    ax.set_aspect("equal")
    ax.set_title(f"'{text}'  coverage {100 * m['coverage']:.0f}%  precision "
                 f"{100 * m['precision']:.0f}%  (tolerance {1000 * crit['tol']:.0f} mm, grey)",
                 loc="left")
    ax.set_xlabel("u (mm)"); ax.set_ylabel("v (mm)")

    t = full["t"]
    ph = full["phase"]
    def shade(a):
        down = np.isin(ph, [phases.index(p) for p in ("press", "stroke", "finish")])
        a.fill_between(t, 0, 1, where=down, transform=a.get_xaxis_transform(),
                       color="#e6e5e0", lw=0, label="pen meant to be down")

    a1 = fig.add_subplot(gs[1, :])
    shade(a1)
    a1.plot(t, np.asarray(full["f_contact"]) @ np.array([0, 0, 1.0]), color=SLOT[3], lw=0.8,
            alpha=0.7, label="sensor (25 Hz)")
    a1.plot(t, full["f_n"], color=SLOT[0], lw=1.6, label="pressure (5 Hz): inks and tears")
    a1.plot(t, np.asarray(full["f_fb"])[:, 2], color=SLOT[1], lw=1.0, ls="--",
            label="felt at the master")
    lo, hi = crit["force_band"]
    a1.axhspan(lo, hi, color=SLOT[2], alpha=0.08, lw=0)
    a1.set_ylim(-1, max(8.0, hi + 1))
    a1.set_ylabel("normal force (N)")
    a1.set_title(f"force: {100 * m['in_band']:.0f}% of pen-down time in the "
                 f"{lo:.0f}-{hi:.0f} N band (green); pressure peak {m['peak_force']:.1f} N", loc="left")
    a1.legend(fontsize=7.5, ncol=4, loc="upper right")

    a2 = fig.add_subplot(gs[2, :], sharex=a1)
    shade(a2)
    kd, kr = full["k_diag"], full["k_req"]
    for i, (lab, c) in enumerate((("k_u", SLOT[0]), ("k_v", SLOT[2]), ("k_n", SLOT[1]))):
        a2.plot(t, kd[:, i], color=c, lw=1.6, label=f"{lab} applied")
        a2.plot(t, kr[:, i], color=c, lw=0.8, ls=":")
    a2.set_ylabel("K_p (N/m)")
    a2.set_title("Case 1 stiffness along the paper (u, v) and off it (n); dotted = requested. "
                 f"Tank gate min alpha {np.min(full['alpha']):.2f}", loc="left")
    a2.legend(fontsize=7.5, ncol=4, loc="upper right")
    a2.set_xlabel("time (s)")

    for j, i in enumerate(picks):
        a = fig.add_subplot(gs[3, j])
        row = [thumbs[k][j] for k in cams]
        a.imshow(np.concatenate(row, axis=1))
        a.set_axis_off()
        a.set_title(f"t = {t_img[i]:.1f} s  ({' | '.join(c[4:].replace('_camera', '') for c in cams)})",
                    fontsize=8)
    fig.suptitle(f"{args.path}", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    out = args.out or str(args.path).replace(".h5", ".png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
