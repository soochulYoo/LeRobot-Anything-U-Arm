"""One episode as a video: the writing, the force, and the stiffness it chose.

Three rows on one canvas, sharing a time axis:
    1. the scene as the camera sees it
    2. contact force, with the 1-6 N band the score requires
    3. the commanded Cartesian stiffness, in-plane and normal

Frames are captured once per policy step, the same cadence the trace is logged
at, so the cursor in rows 2 and 3 is exactly the moment shown in row 1.

    python3 policy/record_episode.py --ckpt policy/runs/act_ft_input/seed0/model.pkl \\
        --text S --out episode.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

BAND, INK = (1.0, 6.0), 0.8
HUES = {"u": "#0072B2", "v": "#56B4E9", "n": "#E69F00"}
FORCE = "#009E73"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default="S")
    ap.add_argument("--case", type=int, default=None, help="0 S, 1 '7', 2 <star>")
    ap.add_argument("--attempt", type=int, default=500, help="500+ is unseen by training")
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--n-exec", type=int, default=3)
    ap.add_argument("--fps", type=int, default=10, help="10 is real time")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    import protocol as P
    from evaluate import WritingPolicyEnv
    from rollout import FlowPolicy, run_episode

    texts = ["S", "7", "<star>"]
    case = a.case if a.case is not None else texts.index(a.text)
    rows = [json.loads(l) for l in (pathlib.Path(a.data) / "attempts.jsonl").read_text().splitlines()]
    duration = 1.3 * max(r["t"] for r in rows if r["text"] == texts[case])

    env = WritingPolicyEnv(control_hz=10.0, cameras=True, render_mode="rgb_array")
    env.record_video = True
    pol = FlowPolicy(a.ckpt, n_exec=a.n_exec)
    spec, _, seed = P.episode_spec(texts[case], case, a.attempt, 60_000)
    res = run_episode(env, pol, dataclasses.replace(spec, time_limit=duration))

    frames = [np.asarray(f) for f in env.frames]
    t = np.asarray(res["trace"]["t"], float)
    f = np.asarray(res["trace"]["f"], float)
    k = np.asarray(res["trace"]["k"], float)          # (T, 3) u, v, n
    n = min(len(frames), len(t))
    frames, t, f, k = frames[:n], t[:n], f[:n], k[:n]
    print(f"{texts[case]} seed {seed}: {'SUCCESS' if res['success'] else 'fail: ' + res['fail_reason']}"
          f"  {n} frames  peak {f.max():.1f} N")

    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, (h / 100) + 4.0), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[h / 100, 2.0, 2.0], hspace=0.35)
    ax_img, ax_f, ax_k = (fig.add_subplot(gs[i]) for i in range(3))

    im = ax_img.imshow(frames[0])
    ax_img.axis("off")
    ax_img.set_title(f"{texts[case]}   seed {seed}   "
                     f"{'SUCCESS' if res['success'] else res['fail_reason']}", fontsize=10)

    ax_f.axhspan(*BAND, color="#9CA3AF", alpha=0.15, lw=0)
    ax_f.axhline(INK, color="#9CA3AF", lw=0.8, ls=":")
    (ln_f,) = ax_f.plot([], [], color=FORCE, lw=1.8)
    cur_f = ax_f.axvline(0, color="#6B7280", lw=0.8)
    ax_f.set_xlim(0, t[-1]), ax_f.set_ylim(-0.3, max(6.5, f.max() * 1.15))
    ax_f.set_ylabel("force (N)"), ax_f.grid(alpha=0.18, lw=0.5)

    lns_k = {}
    for i, axis in enumerate("uvn"):
        (lns_k[axis],) = ax_k.plot([], [], color=HUES[axis], lw=1.6, label=f"K_{axis}")
    cur_k = ax_k.axvline(0, color="#6B7280", lw=0.8)
    ax_k.set_xlim(0, t[-1]), ax_k.set_ylim(0, k.max() * 1.15)
    ax_k.set_ylabel("stiffness (N/m)"), ax_k.set_xlabel("time (s)")
    ax_k.grid(alpha=0.18, lw=0.5), ax_k.legend(fontsize=8, frameon=False, ncol=3)
    for ax in (ax_f, ax_k):
        for sp in ax.spines.values():
            sp.set_color("#D1D5DB")

    out = a.out or str(pathlib.Path(a.ckpt).parent / f"episode_{texts[case].strip('<>')}.mp4")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=a.fps, macro_block_size=1) as vid:
        for i in range(n):
            im.set_data(frames[i])
            ln_f.set_data(t[:i + 1], f[:i + 1])
            for j, axis in enumerate("uvn"):
                lns_k[axis].set_data(t[:i + 1], k[:i + 1, j])
            cur_f.set_xdata([t[i], t[i]]), cur_k.set_xdata([t[i], t[i]])
            fig.canvas.draw()
            vid.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    print(f"wrote {out}  ({n} frames at {a.fps} fps)")


if __name__ == "__main__":
    main()
