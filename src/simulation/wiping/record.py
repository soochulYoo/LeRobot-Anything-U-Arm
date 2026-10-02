"""The scripted eraser, as a video.

Same three rows as the writing recordings -- the scene, the contact force, the
commanded stiffness -- with the share of the glyph still on the board drawn over
the force axis, since that is what the task is scored on.

    python3 record.py --text S --out wipe_S.mp4
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import controller as C          # noqa: E402
import sim as WSM               # noqa: E402
import wipe_scene as SC         # noqa: E402
from wipe_sim import WipingSim  # noqa: E402

BAND = (1.0, 6.0)
HUES = {"u": "#0072B2", "v": "#56B4E9", "n": "#E69F00"}
FORCE, LEFT, ROT = "#009E73", "#CC79A7", "#7D3C98"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="S")
    ap.add_argument("--penetration", type=float, default=0.004)
    ap.add_argument("--k-lat", type=float, default=1000.0)
    ap.add_argument("--k-n", type=float, default=600.0)
    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="wipe.mp4")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    s = WipingSim(cameras=False, render_mode="rgb_array")
    spec = WSM.TaskSpec(text=a.text, letter_height=0.035, time_limit=60.0, seed=0)
    s.reset(spec)
    every = max(1, int(round(1.0 / (a.fps * s.dt))))        # capture at --fps
    print(f"{len(s.marks_uv)} marks on the board, capturing every {every} physics steps")

    lo, hi = s.marks_uv.min(0), s.marks_uv.max(0)
    pad = SC.ERASER_R
    way = []
    for i, v in enumerate(np.arange(lo[1] - pad / 2, hi[1] + pad, pad)):
        us = [lo[0] - pad, hi[0] + pad][:: 1 if i % 2 == 0 else -1]
        way += [np.array([us[0], v]), np.array([us[1], v])]

    K = C.k_world(np.array([a.k_lat, a.k_lat, a.k_n]), s.W)
    x_d = s.ctl.x_d.copy()
    plan = ([(s.frame.to_world(np.r_[way[0], -a.penetration]), 1.0)]
            + [(s.frame.to_world(np.r_[w, -a.penetration]), None) for w in way])

    frames, t, f, k, left, kr = [], [], [], [], [], []

    def capture():
        r = s.env.render()
        r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
        frames.append(r[0] if r.ndim == 4 else r)
        t.append(s.t), f.append(s.last["f_n"]), k.append(np.array(s.k_diag()))
        kr.append(float(s.ctl.g.Kr))        # rotational stiffness: fixed, not commanded
        left.append(float((~s.gone).mean()))

    for target, dwell in plan:
        n_steps = (int(dwell / s.dt) if dwell is not None
                   else max(1, int(np.linalg.norm(target - x_d) / a.speed / s.dt)))
        start = x_d.copy()
        for i in range(1, n_steps + 1):
            nxt = start + (target - start) * (i / n_steps)
            s.step(C.Case1Proposal((nxt - s.ctl.x_d) / s.dt, (K - s.ctl.K) / s.dt))
            x_d = nxt
            if s.step_i % every == 0:
                capture()
            if s.t >= spec.time_limit:
                break
        if s.t >= spec.time_limit:
            break
    capture()

    res = s.score()
    t, f, k, left, kr = map(np.asarray, (t, f, k, left, kr))
    print(f"erased {100 * res['erased']:.1f}%  peak {res['peak_force']:.1f} N  "
          f"{len(frames)} frames")

    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100 + 4.0), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[h / 100, 2.0, 2.0], hspace=0.35)
    ax_img, ax_f, ax_k = (fig.add_subplot(gs[i]) for i in range(3))
    im = ax_img.imshow(frames[0])
    ax_img.axis("off")

    ax_f.axhspan(*BAND, color="#9CA3AF", alpha=0.15, lw=0)
    (ln_f,) = ax_f.plot([], [], color=FORCE, lw=1.8, label="contact force")
    ax_f.set_xlim(0, t[-1]), ax_f.set_ylim(-0.3, max(7.0, f.max() * 1.15))
    ax_f.set_ylabel("force (N)"), ax_f.grid(alpha=0.18, lw=0.5)
    ax_l = ax_f.twinx()
    (ln_l,) = ax_l.plot([], [], color=LEFT, lw=1.8, ls="--", label="glyph left")
    ax_l.set_ylim(-0.03, 1.03), ax_l.set_ylabel("glyph left", color=LEFT)
    ax_f.legend(handles=[ln_f, ln_l], fontsize=8, frameon=False, loc="upper right")

    lns = {}
    for axis in "uvn":
        (lns[axis],) = ax_k.plot([], [], color=HUES[axis], lw=1.6, label=f"K_{axis}")
    ax_k.set_xlim(0, t[-1]), ax_k.set_ylim(0, max(k.max() * 1.15, 1.0))
    ax_k.set_ylabel("translational K (N/m)"), ax_k.set_xlabel("time (s)")
    ax_k.grid(alpha=0.18, lw=0.5)
    # Rotational stiffness lives on its own axis: different units (Nm/rad), and
    # it is NOT part of the action -- the controller holds it at a constant,
    # which is exactly the thing a flat eraser pad on a tilted board would want
    # to vary.  Drawn so that constancy is visible rather than assumed.
    ax_r = ax_k.twinx()
    (ln_r,) = ax_r.plot([], [], color=ROT, lw=1.6, ls="--", label="K_r (fixed)")
    ax_r.set_ylim(0, max(kr.max() * 1.6, 1.0))
    ax_r.set_ylabel("rotational K (Nm/rad)", color=ROT)
    ax_r.tick_params(axis="y", colors=ROT)
    ax_k.legend(handles=[lns["u"], lns["v"], lns["n"], ln_r],
                fontsize=8, frameon=False, ncol=4)
    cur = [ax.axvline(0, color="#6B7280", lw=0.8) for ax in (ax_f, ax_k)]

    with imageio.get_writer(a.out, fps=a.fps, macro_block_size=1) as vid:
        for i in range(len(frames)):
            im.set_data(frames[i])
            ax_img.set_title(f"wiping '{a.text}'   erased {100 * (1 - left[i]):.0f}%"
                             f"   force {f[i]:.1f} N", fontsize=11)
            ln_f.set_data(t[:i + 1], f[:i + 1])
            ln_l.set_data(t[:i + 1], left[:i + 1])
            for j, axis in enumerate("uvn"):
                lns[axis].set_data(t[:i + 1], k[:i + 1, j])
            ln_r.set_data(t[:i + 1], kr[:i + 1])
            for c in cur:
                c.set_xdata([t[i], t[i]])
            fig.canvas.draw()
            vid.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
