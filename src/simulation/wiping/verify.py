"""Replay a collected wiping demo from its stored labels, and ask whether each
label carries anything.

`../writing/collect.py:replay` does this for (x_d, K): drive the recorded
APPLIED values back through the controller on the same board and check the
episode comes out the same, then freeze K at its episode mean and check it does
NOT -- because a stiffness label that changes nothing is not a label.

Wiping adds K_R, so there is a third control:

    recorded    all three labels as collected
    K const     K_p frozen at its episode mean
    K_R const   K_R frozen at its episode mean

On the curved board the last one is the interesting one.  The protocol holds
K_R HIGH to travel and LOW to wipe; freezing it at the mean of those gives a
wrist that is too soft to carry the pad between rows and too stiff to let the
board set its angle while wiping, so if the schedule matters at all, it shows
up here as less of the glyph removed.

    python3 verify.py demos/wipe_curved/S_curved
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import controller as C  # noqa: E402
import sim as SM  # noqa: E402
from wipe_sim import CurvedWipingSim, WipingSim  # noqa: E402


def replay(sim, path, mode: str = "recorded") -> dict:
    """Drive the recorded labels back through the controller, at the physics
    rate, interpolating the 100 Hz record."""
    import h5py
    with h5py.File(path) as f:
        spec = SM.TaskSpec(**json.loads(f.attrs["spec"]))
        t = f["full/t"][:].astype(float)
        xd = f["full/x_d"][:].astype(float)
        K = f["full/K"][:].astype(float)
        kr = f["full/kr"][:].astype(float)
        f_demo = f["full/f_n"][:].astype(float)
    if mode == "k_const":
        K = np.repeat(K.mean(axis=0, keepdims=True), len(K), axis=0)
    if mode == "kr_const":
        kr = np.full_like(kr, kr.mean())
    sim.reset(spec)
    sim.ctl.x_d = xd[0].copy()
    sim.ctl.K = K[0].copy()
    sim.ctl.kr = float(kr[0])
    out, j = [], 0
    while sim.t < t[-1]:
        tn = sim.t + sim.dt
        while j + 1 < len(t) - 1 and t[j + 1] <= tn:
            j += 1
        w = np.clip((tn - t[j]) / max(t[j + 1] - t[j], 1e-9), 0.0, 1.0)
        x_next = xd[j] + w * (xd[j + 1] - xd[j])
        K_next = K[j] + w * (K[j + 1] - K[j])
        kr_next = kr[j] + w * (kr[j + 1] - kr[j])
        rec = sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt,
                                       (K_next - sim.ctl.K) / sim.dt,
                                       Ur=(kr_next - sim.ctl.kr) / sim.dt))
        mis = np.nan
        if rec["f_n"] > 0.8:
            uv = rec["contact_uvh"][:2]
            n = (sim.frame.normal_at(uv) if hasattr(sim.frame, "normal_at")
                 else sim.frame.normal)
            mis = np.degrees(np.arccos(np.clip(-rec["R"][:, 2] @ n, -1.0, 1.0)))
        out.append((rec["t"], rec["f_n"], mis))
    out = np.array(out)
    f_on_demo = np.interp(t, out[:, 0], out[:, 1])
    res = sim.score()
    res["force_rmse"] = float(np.sqrt(np.mean((f_on_demo - f_demo) ** 2)))
    res["mis"] = float(np.nanmean(out[:, 2])) if np.any(~np.isnan(out[:, 2])) else float("nan")
    # What the board asked for over the glyph, so `mis` has a denominator: a
    # misalignment of 9 deg is the whole demand on one board and half of it on
    # another, and this episode's tilt is randomised on top of the curvature.
    nrm = (np.stack([sim.frame.normal_at(uv) for uv in sim.marks_uv])
           if hasattr(sim.frame, "normal_at")
           else np.repeat(sim.frame.normal[None], len(sim.marks_uv), 0))
    res["ask"] = float(np.degrees(np.arccos(np.clip(nrm @ [0.0, 0.0, 1.0], -1, 1))).mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="a case directory of ep_*.h5")
    ap.add_argument("-n", type=int, default=3)
    a = ap.parse_args()
    import h5py
    files = sorted(pathlib.Path(a.dir).glob("ep_*.h5"))[:a.n]
    if not files:
        raise SystemExit(f"no ep_*.h5 in {a.dir}")
    with h5py.File(files[0]) as f:
        board = f.attrs.get("board", "flat")
    sim = (CurvedWipingSim if board == "curved" else WipingSim)(cameras=False)
    print(f"  board {board}, replaying {len(files)} episodes from their stored labels\n")
    print("  episode        mode        erased  in-band   mis   ask   force RMSE")
    for p in files:
        for mode, name in (("recorded", "recorded"), ("k_const", "K const"),
                           ("kr_const", "K_R const")):
            r = replay(sim, p, mode)
            print(f"  {p.name:14s} {name:10s}  {100*r['erased']:5.1f}%  {100*r['in_band']:5.1f}%"
                  f"  {r['mis']:5.1f} {r['ask']:5.1f}  {r['force_rmse']:6.2f} N"
                  + ("" if r["success"] else f"   ({r['fail_reason']})"))
        print()
    sim.close()


if __name__ == "__main__":
    main()
