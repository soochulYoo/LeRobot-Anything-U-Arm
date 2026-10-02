"""The scripted operator's skills side by side: the same randomization written
by each (one row per skill), and when its stiffness levels arrived.

Usage:  python3 plot_tiers.py demos/scripted [--attempt 0] [--out demos/scripted/overview.png]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="the folder holding one folder per skill")
    ap.add_argument("--skills", nargs="+", default=["good", "normal", "bad"])
    ap.add_argument("--attempt", type=int, default=0, help="which randomization of each case to show")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": INK3, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK2,
        "grid.color": "#e6e5e0", "legend.frameon": False,
    })

    root = pathlib.Path(args.root)
    logs = {s: [json.loads(l) for l in (root / s / "attempts.jsonl").read_text().splitlines()]
            for s in args.skills}
    texts = list(dict.fromkeys(e["text"] for e in logs[args.skills[0]]))
    n_c = len(texts) + 1
    fig, axes = plt.subplots(len(args.skills), n_c, figsize=(3.1 * n_c + 1.2, 3.0 * len(args.skills) + 0.6),
                             squeeze=False, gridspec_kw={"width_ratios": [1] * len(texts) + [1.7]})
    bands = []
    for i, skill in enumerate(args.skills):
        rows = logs[skill]
        for j, text in enumerate(texts):
            mine = [e for e in rows if e["text"] == text]
            e = next(e for e in mine if e["attempt"] == args.attempt)
            ax = axes[i, j]
            with h5py.File(e["saved"]) as f:
                goal_uv, mask = f["goal/strokes_uv"][:], f["goal/mask"][:]
                ink = f["privileged/ink_uv"][:]
                tol = json.loads(f.attrs["criteria"])["tol"]
                if j == 0:
                    op = json.loads(f.attrs["operator"])
                    t, kd, ph = f["full/t"][:], f["full/k_diag"][:], f["full/phase"][:]
                    phases = json.loads(f.attrs["phases"])
            for s, mk in zip(goal_uv, mask):
                if mk.any():
                    bands += ax.plot(1000 * s[mk, 0], 1000 * s[mk, 1], color=INK3, alpha=0.3,
                                     solid_capstyle="round")
            if len(ink):
                ax.plot(1000 * ink[:, 0], 1000 * ink[:, 1], ".", color=SLOT[0], ms=2.5)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ok = sum(x["success"] for x in mine)
            ax.set_title(f"{text!r}, {ok}/{len(mine)} succeed.  This one: "
                         f"{'success' if e['success'] else 'fails'}\n"
                         f"coverage {100 * e['coverage']:.0f}%, precision {100 * e['precision']:.0f}%",
                         loc="left", fontsize=8)
            if j == 0:
                all_ok = sum(x["success"] for x in rows)
                ch = np.mean([x["chamfer_mm"] for x in rows])
                comp = np.mean([x["compliance"]["overall"] for x in rows])
                ax.set_ylabel(f"{skill.upper()}\n{all_ok}/{len(rows)} succeed\nchamfer {ch:.2f} mm\n"
                              f"compliance {100 * comp:.0f}%", rotation=0, ha="right", va="center",
                              color=INK, fontsize=9.5, labelpad=12)

        # when the levels arrived, for the first case's episode
        ax = axes[i, -1]
        down = np.isin(ph, [phases.index(p) for p in ("stroke",)])
        ax.fill_between(t, 0, 1, where=down, transform=ax.get_xaxis_transform(), color="#e6e5e0", lw=0,
                        label="pen writing")
        ax.plot(t, kd[:, 0], color=SLOT[0], lw=2, label="K along the paper (xy)")
        ax.plot(t, kd[:, 2], color=SLOT[1], lw=2, label="K into the paper (z)")
        ax.set_ylim(0, 3400)
        ax.set_yticks([500, 1000, 3000])
        ax.grid(axis="y", lw=0.6)
        ax.set_ylabel("N/m")
        ax.set_title(f"stiffness in that {texts[0]!r}:  keys {op['motion_delay']:.2f} s late, "
                     f"levels {op['stiffness_delay']:.1f} s late", loc="left", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper right", ncol=1)
        if i == len(args.skills) - 1:
            ax.set_xlabel("time (s)")
    fig.suptitle(f"{root}: the same randomization (attempt {args.attempt}) at each skill. "
                 f"Grey = target with its {1000 * tol:.0f} mm tolerance, blue = ink.",
                 x=0.01, ha="left", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.canvas.draw()
    for line in bands:                           # the tolerance band at its true width: +-tol
        ax = line.axes
        pt_per_mm = ax.get_window_extent().width / fig.dpi * 72 / np.ptp(ax.get_xlim())
        line.set_linewidth(2000 * tol * pt_per_mm)
    out = args.out or str(root / "overview.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
