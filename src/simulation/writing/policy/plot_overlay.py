"""Every evaluation episode of every variant, drawn on one page.

One panel per (structure, task): all episodes overlaid, each mapped into its own
target's bounding box, because the paper is randomized per episode -- letter
size, slant and placement all change, so raw canvas coordinates would smear
together for reasons that have nothing to do with the policy.

Colour carries success, and never alone: the panel prints the success rate and
failed episodes are drawn heavier, so the page survives greyscale printing.

    python3 policy/plot_overlay.py --out policy/runs/OVERLAY.png
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

STRUCTS = ["ft_input", "uni_dir", "unified", "cross_cond"]
ORDER = STRUCTS + [f"act_{n}" for n in STRUCTS]
CASE = {n: c for n, c in zip(STRUCTS, "abcd")}
CASE.update({f"act_{n}": c for n, c in zip(STRUCTS, "abcd")})
TASKS = ["S", "7", "<star>"]
OK, BAD, TGT = "#1D6996", "#D95F02", "#111827"      # blue / orange: safe under CVD


def norm(pts, tgt):
    """Both into the target's own frame, so episodes can be laid over each other."""
    lo, hi = tgt.min(0), tgt.max(0)
    c, s = 0.5 * (lo + hi), max(float((hi - lo).max()), 1e-6)
    return (pts - c) / s, (tgt - c) / s


def panels(root: pathlib.Path, name: str):
    """Every episode of every seed, grouped by task index."""
    out = {c: [] for c in range(len(TASKS))}
    for d in sorted((root / name).glob("seed*/eval/traces.npz")):
        z = np.load(d, allow_pickle=False)
        for key in (k for k in z.files if k.endswith("_ink")):
            tag = key[:-4]
            c = int(tag.split("_")[0][1:])
            tgt = z[f"{tag}_tgt"]
            if not len(tgt):
                continue
            ink, t = norm(z[key].reshape(-1, 2), tgt)
            out.setdefault(c, []).append((ink, t, bool(z[f"{tag}_ok"][0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="policy/runs")
    ap.add_argument("--out", default="policy/runs/OVERLAY.png")
    a = ap.parse_args()
    root = pathlib.Path(a.runs)

    names = [n for n in ORDER if list((root / n).glob("seed*/eval/traces.npz"))]
    if not names:
        raise SystemExit(f"no traces.npz under {root} -- run the evaluation first")
    data = {n: panels(root, n) for n in names}

    fig, axes = plt.subplots(len(names), len(TASKS),
                             figsize=(3.1 * len(TASKS), 3.1 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, task in enumerate(TASKS):
            ax = axes[i][j]
            eps = data[n].get(j, [])
            good = sum(1 for _, _, ok in eps if ok)
            for ink, tgt, ok in eps:
                if len(ink):
                    ax.plot(ink[:, 0], ink[:, 1], ".", ms=0.7, alpha=0.30 if ok else 0.55,
                            color=OK if ok else BAD, mew=0)
            if eps:
                tgt = eps[0][1]
                ax.plot(tgt[:, 0], tgt[:, 1], "-", lw=1.4, color=TGT, alpha=0.85)
            ax.set_aspect("equal")
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#D1D5DB")
            ax.set_title(f"{task}   {100 * good / max(1, len(eps)):.0f}%  ({len(eps)} eps)",
                         fontsize=9, color="#374151")
            if j == 0:
                ax.set_ylabel(f"({CASE[n]}) {n}", fontsize=10)

    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=OK, label="success"),
                        plt.Line2D([], [], marker="o", ls="", color=BAD, label="failure"),
                        plt.Line2D([], [], color=TGT, lw=1.4, label="target glyph")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Every evaluation episode, in each episode's own target frame", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}  ({len(names)} structures x {len(TASKS)} tasks)")


if __name__ == "__main__":
    main()
