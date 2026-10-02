"""Mean contact-force profile over an episode, per structure and task.

Every evaluation episode of every seed, averaged onto one time axis, against the
force the demonstrations produced.  Episodes of a task all run the same length
(1.3x the longest demonstration of that text), so the axis is directly shared.

The band the score cares about is drawn behind: 1-6 N must hold for 85% of
pen-down time, and below 0.8 N the pen leaves no ink.

    python3 policy/plot_force.py --out policy/runs/FORCE.png
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

STRUCTS = ["ft_input", "uni_dir", "unified", "cross_cond"]
FAMILY = {"CoFA flow": STRUCTS,
          "Comp-ACT": [f"act_{n}" for n in STRUCTS] + ["act_film", "act_ft_tokens"]}
TASKS = ["S", "7", "<star>"]
# Okabe-Ito: a published colour-blind-safe qualitative set, assigned in fixed order
HUES = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
DEMO = "#111827"
BAND, INK = (1.0, 6.0), 0.8


def profiles(root: pathlib.Path, name: str, case: int):
    """(n_episodes, T) of normal force, resampled onto the longest common axis."""
    ts, fs = [], []
    for p in sorted((root / name).glob("seed*/eval/traces.npz")):
        z = np.load(p, allow_pickle=False)
        for key in (k for k in z.files if k.endswith("_f") and k.startswith(f"c{case}_")):
            t = z[key[:-2] + "_t"]
            if len(t) > 1:
                ts.append(t), fs.append(z[key])
    if not fs:
        return None, None
    T = max(len(f) for f in fs)
    grid = np.linspace(0, max(t[-1] for t in ts), T)
    return grid, np.stack([np.interp(grid, t, f) for t, f in zip(ts, fs)])


def demo_profiles(data: pathlib.Path, case: int):
    ts, fs = [], []
    for p in sorted(data.glob("*/ep_*.h5")):
        with h5py.File(p) as h:
            if int(h.attrs["case"]) != case:
                continue
            ts.append(h["full/t"][:]), fs.append(h["full/f_n"][:])
    if not fs:
        return None, None
    T = max(len(f) for f in fs)
    grid = np.linspace(0, max(t[-1] for t in ts), T)
    return grid, np.stack([np.interp(grid, t, f) for t, f in zip(ts, fs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="policy/runs")
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--out", default="policy/runs/FORCE.png")
    a = ap.parse_args()
    root = pathlib.Path(a.runs)

    fams = {f: [n for n in ns if list((root / n).glob("seed*/eval/traces.npz"))]
            for f, ns in FAMILY.items()}
    fams = {f: ns for f, ns in fams.items() if ns}
    if not fams:
        raise SystemExit(f"no traces under {root} -- rerun the evaluation "
                         "(rollout.py only started saving force traces recently)")

    fig, axes = plt.subplots(len(fams), len(TASKS), figsize=(4.4 * len(TASKS), 3.3 * len(fams)),
                             squeeze=False, sharey=True)
    for i, (fam, names) in enumerate(fams.items()):
        for j, task in enumerate(TASKS):
            ax = axes[i][j]
            ax.axhspan(*BAND, color="#9CA3AF", alpha=0.13, lw=0)
            ax.axhline(INK, color="#9CA3AF", lw=0.8, ls=":")
            gd, fd = demo_profiles(pathlib.Path(a.data), j)
            if fd is not None:
                ax.plot(gd, fd.mean(0), color=DEMO, lw=1.6, ls="--", label="demonstrations")
            for k, n in enumerate(names):
                g, f = profiles(root, n, j)
                if f is None:
                    continue
                m, sd = f.mean(0), f.std(0)
                ax.fill_between(g, m - sd, m + sd, color=HUES[k % len(HUES)], alpha=0.13, lw=0)
                ax.plot(g, m, color=HUES[k % len(HUES)], lw=1.6,
                        label=f"{n.replace('act_', '')}  ({len(f)} eps)")
            ax.set_title(f"{fam} — {task}", fontsize=10, color="#374151")
            ax.set_xlabel("time (s)", fontsize=9)
            if j == 0:
                ax.set_ylabel("contact force (N)", fontsize=9)
            ax.grid(alpha=0.18, lw=0.5)
            for sp in ax.spines.values():
                sp.set_color("#D1D5DB")
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Mean contact force over an episode — all evaluation episodes and seeds\n"
                 "shaded: ±1 SD across episodes;  grey band: the 1–6 N the score requires;  "
                 "dotted: 0.8 N ink threshold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
