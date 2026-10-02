"""Wiping with a stiff vs a compliant rotational impedance, side by side.

Same board, same path, same translational gains, same NORMAL FORCE.  The only
difference between the two halves is K_R, so anything that differs is
rotational stiffness.

THREE THINGS HAD TO BE RIGHT BEFORE THIS COMPARISON MEANT ANYTHING, and the
first version of this file had none of them (CURVED_BOARD.md has the
measurements):

  THE BOARD MUST ASK FOR SOMETHING.  A rigid pad on a board whose normal is
  where the pad was told to point has nothing to comply with.  `--curved` puts
  the glyph on a surface that asks the pad for ~9 deg of tilt and swings ~5 deg
  across the pad's own width; `--tilt` asks for a constant wedge instead, which
  also works but is a weaker question.

  K_R MUST BE SWEPT WHERE THE CONTACT MOMENT LIVES.  A pad of half-width r
  pressing with f generates at most f*r -- 0.05 Nm for this pad at 3 N -- so
  anything above ~1 Nm/rad is rigid.  The default pair is 60 vs 0.3, not 60 vs
  3, which compared rigid with rigid and read as a null result.

  THE FORCE MUST BE HELD, NOT THE DEPTH.  At a fixed penetration the force
  depends on how the pad is lying: 2.5 N stiff against 3.7 N compliant, and
  since erasure is work, the comparison then reports a force effect under a
  rotational name.  `--force` regulates it (`smoke.py` has the loop); it is on
  by default here for that reason.

The driver is `smoke.py:run` -- the same one `kr_task.py` measures with, so the
video cannot drift away from the numbers.

    python3 record_rot.py --curved --out wipe_rot_curved.mp4
    python3 record_rot.py --tilt 8 --hi 60 --lo 0.3 --out wipe_rot.mp4
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import smoke  # noqa: E402  the scripted driver, shared with kr_task.py

BAND = (1.0, 6.0)
STIFF, SOFT = "#E69F00", "#0072B2"     # the two conditions, everywhere


def run(Kr, a) -> dict:
    """One condition, with a frame grabbed every time the driver samples."""
    frames = []

    def grab(s):
        r = s.env.render()
        r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
        frames.append(r[0] if r.ndim == 4 else r)

    every = max(1, int(round(1.0 / (a.fps * (1.0 / 500)))))
    r = smoke.run(a.text, penetration=a.penetration, k_lat=a.k_lat, k_n=a.k_n,
                  speed=a.speed, curved=a.curved, tilt=a.tilt, kr=float(Kr),
                  wrist_inertia=a.wrist_inertia, f_target=a.force,
                  erase_work=a.erase_work, every=every, on_sample=grab,
                  render_mode="rgb_array", verbose=False)
    tr = r["trace"]
    print(f"   K_R={Kr:5g}  erased {100*r['erased']:5.1f}%  misalign {r['mis_mean']:4.1f} deg"
          f"  (board asks {r['ask_mean']:4.1f})  yield {r['yld_mean']:4.1f} deg"
          f"  force {np.mean(tr['f'][tr['f'] > 0.8]):5.2f} N  peak {r['peak_force']:5.1f} N"
          f"  in-band {100*r['in_band']:4.0f}%")
    return dict(frames=frames, score=r, ask=r["ask_mean"],
                **{k: tr[k] for k in ("t", "f", "erased", "mis")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="S")
    ap.add_argument("--curved", action="store_true", help="curved board")
    ap.add_argument("--tilt", type=float, default=0.0, help="board tilt, degrees")
    ap.add_argument("--hi", type=float, default=60.0, help="stiff K_R, Nm/rad")
    ap.add_argument("--lo", type=float, default=0.3, help="compliant K_R, Nm/rad")
    ap.add_argument("--force", type=float, default=3.0, help="N, regulated; 0 for depth")
    ap.add_argument("--wrist-inertia", type=float, default=None,
                    help="restore the OLD guessed-scalar Dr (0.01 was the stock guess)")
    ap.add_argument("--erase-work", type=float, default=0.12,
                    help="N m per mark.  The default 0.040 lets the raster cover the "
                         "glyph twice over, so both conditions finish at 100%")
    ap.add_argument("--penetration", type=float, default=0.004)
    ap.add_argument("--k-lat", type=float, default=1000.0)
    ap.add_argument("--k-n", type=float, default=600.0)
    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="wipe_rot.mp4")
    a = ap.parse_args()
    a.force = a.force or None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    print(f"board {'curved' if a.curved else f'tilted {a.tilt:.0f} deg'},"
          f" commanded orientation vertical,"
          f" {'force held at %.1f N' % a.force if a.force else 'fixed depth'}")
    hi, lo = run(a.hi, a), run(a.lo, a)
    n = min(len(hi["frames"]), len(lo["frames"]))
    h, w = hi["frames"][0].shape[:2]
    ask = hi["ask"]

    fig = plt.figure(figsize=(2 * w / 100, h / 100 + 4.2), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[h / 100, 2.0, 2.0], hspace=0.38, wspace=0.05)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_f, ax_m = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    im_a, im_b = ax_a.imshow(hi["frames"][0]), ax_b.imshow(lo["frames"][0])
    for ax in (ax_a, ax_b):
        ax.axis("off")

    ax_f.axhspan(*BAND, color="#9CA3AF", alpha=0.15, lw=0)
    (lf_a,) = ax_f.plot([], [], color=STIFF, lw=1.8, label=f"STIFF  K_R={a.hi:g}")
    (lf_b,) = ax_f.plot([], [], color=SOFT, lw=1.8, label=f"COMPLIANT  K_R={a.lo:g}")
    ax_f.set_xlim(0, max(hi["t"][-1], lo["t"][-1]))
    ax_f.set_ylim(-0.3, max(7.0, hi["f"].max(), lo["f"].max()) * 1.15)
    ax_f.set_ylabel("contact force (N)"), ax_f.grid(alpha=0.18, lw=0.5)
    ax_f.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")

    (lm_a,) = ax_m.plot([], [], color=STIFF, lw=1.8)
    (lm_b,) = ax_m.plot([], [], color=SOFT, lw=1.8)
    ax_m.axhline(ask, color="#6B7280", lw=0.8, ls=":")
    ax_m.text(0.01, ask, f" the board asks for {ask:.1f}°", fontsize=7.5, color="#6B7280",
              va="bottom", transform=ax_m.get_yaxis_transform())
    ax_m.set_xlim(ax_f.get_xlim()), ax_m.set_ylim(0, max(ask * 1.8, 2.0))
    ax_m.set_ylabel("pad misalignment (deg)"), ax_m.set_xlabel("time (s)")
    ax_m.grid(alpha=0.18, lw=0.5)
    ax_e = ax_m.twinx()
    (le_a,) = ax_e.plot([], [], color=STIFF, lw=1.5, ls="--")
    (le_b,) = ax_e.plot([], [], color=SOFT, lw=1.5, ls="--")
    ax_e.set_ylim(-0.03, 1.03), ax_e.set_ylabel("erased (dashed)")
    cur = [ax.axvline(0, color="#6B7280", lw=0.8) for ax in (ax_f, ax_m)]

    with imageio.get_writer(a.out, fps=a.fps, macro_block_size=1) as vid:
        for i in range(n):
            im_a.set_data(hi["frames"][i]), im_b.set_data(lo["frames"][i])
            ax_a.set_title(f"STIFF  K_R = {a.hi:g} Nm/rad    erased {100*hi['erased'][i]:.0f}%"
                           f"   {hi['f'][i]:.1f} N", fontsize=10, color=STIFF)
            ax_b.set_title(f"COMPLIANT  K_R = {a.lo:g} Nm/rad    erased {100*lo['erased'][i]:.0f}%"
                           f"   {lo['f'][i]:.1f} N", fontsize=10, color=SOFT)
            for ln, d, key in ((lf_a, hi, "f"), (lf_b, lo, "f"),
                               (lm_a, hi, "mis"), (lm_b, lo, "mis"),
                               (le_a, hi, "erased"), (le_b, lo, "erased")):
                ln.set_data(d["t"][:i + 1], d[key][:i + 1])
            for c in cur:
                c.set_xdata([hi["t"][i], hi["t"][i]])
            fig.canvas.draw()
            vid.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    print(f"wrote {a.out}  ({n} frames)")


if __name__ == "__main__":
    main()
