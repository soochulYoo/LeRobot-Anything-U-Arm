"""Demos -> (observation, local surface geometry) pairs for step 2.

The oracle sets K_R from the true surface.  Step 2 asks how much of that an
estimator recovers from what the robot can actually see, so the target is the
quantity the oracle rule consumes -- not the pad's misalignment, which depends
on the pad's own pose and is therefore not a property of the environment:

    swing(u, v) = max angle between the true normal at the contact point and
                  at the four points one pad half-width away

Everything needed is in the demo.  `privileged/surf_half_amp_sigma_seed`
reconstructs the CurvedSurface exactly, so the target is closed form at any
contact point rather than read off a mesh.

Inputs are what the robot has at inference: the wrist camera, the contact
force, the joint torques, and proprioception.  Note there is no six-axis F/T in
these recordings -- the contact MOMENT, which is what most directly reveals how
the pad is sitting, has to be inferred from `tau` through the Jacobian.

    python3 surface_dataset.py demos/wipe_curved --out data/surface.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from curved.surface import CurvedSurface          # noqa: E402

import wipe_scene as SC                           # noqa: E402


def swing_at(surf, xy, r):
    """Normal swing across a pad of half-width r centred at xy, in degrees."""
    n0 = surf.normal(xy[..., 0], xy[..., 1])
    out = np.zeros(len(xy))
    for d in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
        q = xy + np.asarray(d)
        n1 = surf.normal(q[..., 0], q[..., 1])
        c = np.clip(np.einsum("ij,ij->i", n0, n1), -1.0, 1.0)
        out = np.maximum(out, np.degrees(np.arccos(c)))
    return out


def episode(path, r_pad):
    with h5py.File(path) as f:
        half, amp, sigma, seed = f["privileged/surf_half_amp_sigma_seed"][:]
        surf = CurvedSurface(half=float(half), amp=float(amp),
                             sigma=float(sigma), seed=int(seed))
        o = f["obs"]
        tcp = o["tcp_pos"][:].astype(np.float64)
        org = f["privileged/canvas_origin"][:]
        R = f["privileged/canvas_R"][:]
        uv = ((tcp - org) @ R)[:, :2]              # contact point in board axes
        d = dict(
            wrist=o["rgb_wrist_camera"][:], top=o["rgb_top_camera"][:],
            f=o["f_contact"][:], tau=o["tau"][:], qpos=o["qpos"][:], qvel=o["qvel"][:],
            k=o["k_diag"][:], kr=o["kr"][:], tcp=tcp.astype(np.float32),
            quat=o["tcp_quat"][:], uv=uv.astype(np.float32),
            swing=swing_at(surf, uv, r_pad).astype(np.float32),
        )
        # only frames where the pad is actually on the board: off contact there
        # is no surface under it to estimate, and no K_R decision to make
        fn = np.linalg.norm(d["f"], axis=1)
        d["down"] = (fn > 0.8).astype(np.bool_)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="data/surface.npz")
    ap.add_argument("--pad", type=float, default=None, help="pad half-width, m")
    a = ap.parse_args()
    r_pad = a.pad if a.pad is not None else SC.ERASER_R

    files = sorted(pathlib.Path(a.root).glob("*/ep_*.h5"))
    if not files:
        raise SystemExit(f"no episodes under {a.root}")
    parts = []
    for i, p in enumerate(files):
        e = episode(p, r_pad)
        e["ep"] = np.full(len(e["swing"]), i, np.int32)
        parts.append(e)
    d = {k: np.concatenate([e[k] for e in parts]) for k in parts[0]}

    sw, down = d["swing"], d["down"]
    print(f"{len(files)} episodes, {len(sw)} frames, {down.sum()} in contact")
    print(f"pad half-width {1000*r_pad:.0f} mm")
    print(f"swing over frames in contact: mean {sw[down].mean():.2f} deg, "
          f"p5 {np.percentile(sw[down], 5):.2f}, p95 {np.percentile(sw[down], 95):.2f}, "
          f"max {sw[down].max():.2f}")
    print(f"  a regressor that always predicts the mean scores "
          f"{sw[down].std():.2f} deg rmse -- that is the number to beat")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
