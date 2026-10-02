"""How much of the oracle's ceiling does the ESTIMATOR recover?

The oracle reads the true surface; the regressor reads only what the robot can
see, at 1.07 deg rmse.  The question this answers is whether that error is small
enough to matter, because the rule divides by the estimate:

    K_R = f_n * r / swing

so near the low end of the swing distribution (p5 is 8.5 deg) a one-degree error
is already a ten-percent error in K_R.  If recovery is poor it may be the RULE's
sensitivity rather than the estimate's quality, which is why `--floor` and
`--log-blend` are here: a floor on the swing, and interpolating K_R in log space,
both make the rule less sharp without making the estimate any better.

The estimator runs at policy rate, not physics rate: a camera at 500 Hz is not
a thing, and K_R holds between updates like any other commanded quantity.

    python3 recover.py --ckpt data/surface_reg.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import smoke
import wipe_scene as SC
from oracle_kr import force_metrics, oracle
from train_surface import Net

VEC_KEYS = ("f_contact", "tau", "qpos", "qvel", "k_diag", "tcp_pos", "tcp_quat")


def estimator(ckpt, r_pad, k_lo, k_hi, every, floor, log=None, dev="cpu"):
    """The same rule the oracle uses, with the surface ESTIMATED from the
    observation instead of read off the true height field."""
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    mu, sd = ck["mu"], ck["sd"]
    net = Net(len(mu)).to(dev).eval()
    with torch.no_grad():                      # build the lazy layers, then load
        net(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64), torch.zeros(1, len(mu)))
    net.load_state_dict(ck["model"])
    state = {"k": k_hi, "i": -10 ** 9}

    def f(s):
        last = getattr(s, "last", {}) or {}
        if float(last.get("f_n", 0.0)) < 0.8:
            state["k"] = k_hi                  # off contact: nothing to estimate
            return k_hi
        if s.step_i - state["i"] < every:
            return state["k"]                  # held between policy frames
        state["i"] = s.step_i
        o = s.observe(images=True)
        img = lambda k: torch.as_tensor(
            np.asarray(o[k], np.float32)[::2, ::2][None] / 255.0).permute(0, 3, 1, 2)
        v = np.concatenate([np.asarray(o[k], np.float32).ravel() for k in VEC_KEYS]
                           + [np.float32([s.ctl.kr])])
        with torch.no_grad():
            swing = float(net(img("rgb_wrist_camera"), img("rgb_top_camera"),
                              torch.as_tensor(((v - mu) / sd)[None].astype(np.float32))))
        swing = max(swing, floor)
        k = float(np.clip(float(last["f_n"]) * r_pad / np.deg2rad(swing), k_lo, k_hi))
        if log is not None:
            log.append((k, swing))
        state["k"] = k
        return k
    return f


def row(tag, r, m):
    f = r["trace"]["f"]
    print(f"  {tag:<16} {100*r['erased']:5.1f}%  {r['t90']:5.1f}  {r['mis_mean']:5.1f}"
          f"   {r['yld_mean']:5.1f}  {np.mean(f[f > 0.8]):6.2f}  {r['peak_force']:5.1f}"
          f"   {100*r['in_band']:4.0f}%   {m['f_rmse']:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/surface_reg.pt")
    ap.add_argument("--kr", default="60,3,0.3")
    ap.add_argument("--text", default="S")
    ap.add_argument("--every", type=int, default=50, help="physics steps between estimates")
    ap.add_argument("--floor", type=float, default=0.0, help="deg, a floor on the estimate")
    ap.add_argument("--wrist-inertia", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    krs = [float(x) for x in a.kr.split(",")]
    common = dict(text=a.text, curved=True, seed=a.seed, verbose=False,
                  wrist_inertia=a.wrist_inertia)
    print("  condition        erased    t90    mis   yield   f_mean   peak  in-band   f_rmse")
    rows = {}
    for kr in krs:
        r = smoke.run(kr=kr, **common)
        rows[kr] = r
        row(f"constant {kr:g}", r, force_metrics(r))

    olog = []
    o = smoke.run(kr=max(krs), kr_fn=oracle(SC.ERASER_R, min(krs), max(krs), olog), **common)
    row("ORACLE (true)", o, force_metrics(o))

    elog = []
    e = smoke.run(kr=max(krs), cameras=True, **common,
                  kr_fn=estimator(a.ckpt, SC.ERASER_R, min(krs), max(krs),
                                  a.every, a.floor, elog))
    row("ESTIMATOR", e, force_metrics(e))

    best = max(rows, key=lambda k: rows[k]["erased"])
    b, o_e, e_e = rows[best]["erased"], o["erased"], e["erased"]
    gap = o_e - b
    print(f"\n best constant K_R = {best:g}: erased {100*b:.1f}%")
    print(f" oracle: {100*o_e:.1f}%   estimator: {100*e_e:.1f}%")
    if abs(gap) > 1e-6:
        print(f" ceiling {100*gap:+.1f} pp, estimator recovers {100*(e_e-b)/gap:.0f}% of it")
    if elog:
        k = np.array([x for x, _ in elog]); sw = np.array([y for _, y in elog])
        ok = np.array([y for _, y in olog]) if olog else None
        print(f" estimated swing {sw.mean():.1f} deg mean ({sw.min():.1f}..{sw.max():.1f});"
              f" true swing {np.mean([y for _, y in olog]):.1f} deg"
              if olog else "")
        print(f" K_R commanded {k.min():.2g}..{k.max():.2g} (median {np.median(k):.2g})")


if __name__ == "__main__":
    main()
