"""Is the stiffness label worth learning, or is it a lookup table?

The question this answers is the one that decided the writing comparison without
anyone asking it.  In policy/runs/COMPARISON.md all eight structures command the
demonstrated stiffness to 0.0-0.7% and sit at `acc xy` 97.7-97.9%, which reads
like eight policies that learned impedance.  It is not: protocol.EXPECTED makes K
a FOUR-ENTRY FUNCTION OF THE WRITER'S PHASE, so a phase classifier plus a lookup
reproduces the label exactly and there is nothing left for an architecture to do.

So measure that directly.  For a demonstration directory, report

  PHASE-ONLY ACCURACY   predict the level from the writer's phase alone, taking
        the majority level per phase.  This is the lookup table, and it is the
        number any architecture comparison has to beat to mean anything.  Near
        100% means the dataset cannot distinguish policies on stiffness.
  LEVEL SPREAD IN CONTACT   how often each level is actually used.  A single
        level in contact is a constant, not a decision.
  FRICTION COUPLING   mutual information between the level and the paper's
        friction, which is privileged and invisible to a camera.  This is what
        makes the label carry information a policy has to INFER.
  DRAG-ONLY ACCURACY   predict the level from the in-plane contact force alone,
        majority per bin.  If this beats phase-only, the label has moved from the
        phase to the force -- which is the point of protocol.py --adaptive-xy.

    python3 policy/label_check.py demos/protocol_v1 demos/protocol_adaptive
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LEVELS = ("LOW", "MID", "HIGH")
INK = 0.8        # N, sim.Criteria.ink_force -- the contact test


def majority_accuracy(group: np.ndarray, label: np.ndarray) -> float:
    """Accuracy of the best possible predictor that sees only `group`.

    Per group, predict its most common label.  No model can do better from that
    input alone, so this is an upper bound rather than one classifier's score.
    """
    if len(label) == 0:
        return float("nan")
    hit = 0
    for g in np.unique(group):
        m = group == g
        hit += np.bincount(label[m]).max()
    return hit / len(label)


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """I(x; y) in bits, x continuous (binned by quantile), y discrete."""
    if len(x) < 2 or len(np.unique(y)) < 2:
        return 0.0
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    xb = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    joint = np.zeros((xb.max() + 1, y.max() + 1))
    np.add.at(joint, (xb, y), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))


def audit(root: pathlib.Path) -> dict:
    files = sorted(root.glob("*/ep_*.h5"))
    if not files:
        raise SystemExit(f"no */ep_*.h5 under {root}")
    phase, lvl_xy, lvl_z, drag, mu_ep, lvl_ep = [], [], [], [], [], []
    for f in files:
        with h5py.File(f) as h:
            ph = h["full/phase"][:].astype(int)
            kl = h["full/k_level"][:].astype(int)
            fn = h["full/f_n"][:].astype(float)
            fc = h["full/f_contact"][:].astype(float)
            n = h["privileged/canvas_R"][:][:, 2].astype(float)
            mu = float(h["privileged/friction"][()])
        # the operator's own contact test, and the drag it feels there
        down = fn >= INK
        tan = np.linalg.norm(fc - (fc @ n)[:, None] * n[None], axis=1)
        phase.append(ph)
        lvl_xy.append(kl[:, 0])
        lvl_z.append(kl[:, 1])
        drag.append(tan)
        if down.any():
            # one row per episode: the level it mostly held in contact, against mu
            mu_ep.append(mu)
            lvl_ep.append(int(np.bincount(kl[down, 0]).argmax()))
    phase = np.concatenate(phase)
    lvl_xy, lvl_z = np.concatenate(lvl_xy), np.concatenate(lvl_z)
    drag = np.concatenate(drag)
    ok = lvl_xy >= 0
    phase, lvl_xy, lvl_z, drag = phase[ok], lvl_xy[ok], lvl_z[ok], drag[ok]
    dbin = np.digitize(drag, np.quantile(drag[drag > 0], np.linspace(0, 1, 9)[1:-1])
                       if (drag > 0).any() else [0.0])
    return dict(
        episodes=len(files), frames=len(lvl_xy),
        xy_hist=np.bincount(lvl_xy, minlength=3) / len(lvl_xy),
        z_hist=np.bincount(lvl_z, minlength=3) / len(lvl_z),
        phase_xy=majority_accuracy(phase, lvl_xy),
        phase_z=majority_accuracy(phase, lvl_z),
        drag_xy=majority_accuracy(dbin, lvl_xy),
        phase_drag_xy=majority_accuracy(phase * 16 + dbin, lvl_xy),
        mi_mu_xy=mutual_information(np.asarray(mu_ep), np.asarray(lvl_ep), bins=6),
        mu_range=(min(mu_ep), max(mu_ep)) if mu_ep else (float("nan"),) * 2,
        ep_levels=np.bincount(np.asarray(lvl_ep), minlength=3) / max(1, len(lvl_ep)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    args = ap.parse_args()
    for r in args.roots:
        a = audit(pathlib.Path(r))
        print(f"\n{r}   {a['episodes']} episodes, {a['frames']} labelled frames")
        print(f"  friction {a['mu_range'][0]:.2f}-{a['mu_range'][1]:.2f}")
        print(f"  xy level used        " + "  ".join(
            f"{n} {100 * p:5.1f}%" for n, p in zip(LEVELS, a["xy_hist"])))
        print(f"  xy level per episode " + "  ".join(
            f"{n} {100 * p:5.1f}%" for n, p in zip(LEVELS, a["ep_levels"]))
            + "   (the level it mostly held in contact)")
        print(f"  BEST PREDICTOR OF THE xy LABEL, by what it is allowed to see:")
        print(f"    phase only          {100 * a['phase_xy']:5.1f}%   <- the lookup table;"
              f" a comparison must beat this to mean anything")
        print(f"    drag only           {100 * a['drag_xy']:5.1f}%   <- the force")
        print(f"    phase + drag        {100 * a['phase_drag_xy']:5.1f}%")
        print(f"  I(friction ; xy level) {a['mi_mu_xy']:.3f} bits"
              f"   <- information a camera cannot supply (max {np.log2(3):.2f})")
        print(f"  z level: phase only {100 * a['phase_z']:5.1f}%, used " + "  ".join(
            f"{n} {100 * p:4.1f}%" for n, p in zip(LEVELS, a["z_hist"])))


if __name__ == "__main__":
    main()
