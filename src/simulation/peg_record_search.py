"""Side-by-side video: aim-and-push against scrubbing, at the same aim error.

Same seed, same peg, same hole, same stiffness schedule, same aim error. The
only difference between the two halves of the frame is whether the executor
spirals on the face before it pushes, so anything visible is the search.

The thing to watch is the moment of failure on the left. Aim-and-push presses
a peg that is only partly over the aperture, catches one corner on the sharp
edge -- the hole has 3 mm of clearance and no chamfer -- and wedges: the depth
trace stops while the force trace climbs. On the right the same peg is dragged
across the opening until it drops in.

Rendering needs a GPU, so this runs under peg_record.sbatch, not on a CPU node.

Usage:  python3 peg_record_search.py --err 4 --seed 2 --out peg_search_4mm
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from record_demo import _font, frame_of, save

import peg_insertion_case1 as Q
import peg_insertion_cascade as P

# Looking across the box face at a shallow angle: the subject is the peg's head
# against the aperture, so the frame has to show the face (where the spiral is)
# and the depth (how far in it got) at once.
CAM = {"eye": [0.42, 0.10, 0.30], "at": [0.00, 0.26, 0.11], "size": (680, 520)}


def trace(img: np.ndarray, series: list[tuple[str, np.ndarray, float, tuple]],
          t: np.ndarray, upto: int, h: int = 120) -> np.ndarray:
    """A strip of normalised traces under the frame, drawn up to frame `upto`."""
    w = img.shape[1]
    strip = Image.new("RGB", (w, h), (16, 16, 18))
    d = ImageDraw.Draw(strip)
    n = max(2, len(t))
    for name, y, top, col in series:
        pts = []
        for i in range(min(upto + 1, len(y))):
            x = int(w * i / (n - 1))
            v = float(np.clip(y[i] / top, 0.0, 1.0))
            pts.append((x, int(h - 8 - v * (h - 26))))
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    for i, (name, _, top, col) in enumerate(series):
        d.text((10 + 190 * i, 4), f"{name} (0..{top:g})", font=_font(14), fill=col)
    return np.vstack([img, np.asarray(strip)])


def binned(res: dict, td: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-rate depth and force reduced to one value per frame.

    Force takes the MAX within each frame's window and depth the last value:
    sampling force at 20 Hz drops the millisecond-scale contact spikes that are
    the whole point of the trace.
    """
    t, f, d = res["t"], res["f"], res["depth"]
    edges = np.append(td, td[-1] + (td[-1] - td[-2] if len(td) > 1 else 1.0))
    idx = np.clip(np.searchsorted(edges, t, "right") - 1, 0, len(td) - 1)
    fmax = np.zeros(len(td))
    dlast = np.zeros(len(td))
    np.maximum.at(fmax, idx, f)
    dlast[idx] = d
    return dlast, fmax


def label(img: np.ndarray, title: str, lines: list[str], colour) -> np.ndarray:
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 34 + 22 * len(lines)], fill=(0, 0, 0, 160))
    d.text((12, 6), title, font=_font(22), fill=colour)
    for i, ln in enumerate(lines):
        d.text((12, 34 + 22 * i), ln, font=_font(17), fill=(235, 235, 235))
    return np.asarray(im)


def run_arm(search_s: float, err_mm: float, seed: int, args) -> tuple[list, dict]:
    """One arm, with frames captured at `args.fps`."""
    a = argparse.Namespace(**vars(args))
    a.search_s = search_s
    Q.resolve(a)
    s = Q.setup(seed, a, render_mode="rgb_array", cam=CAM)
    Rh, _ = P.hole_frame(s["u"])
    rng = np.random.default_rng(1000 + seed)
    d = rng.normal(size=2)
    d /= max(np.linalg.norm(d), 1e-9)
    lat = 1e-3 * err_mm * (d[0] * Rh[:, 1] + d[1] * Rh[:, 2])

    raw = []
    every = max(1, int(round(1.0 / (args.fps * s["dt"]))))
    r = Q.rollout(s, a, lat, on_frame=lambda t, depth, f: raw.append(
        (frame_of(s["env"]), t, depth, f)), every=every)
    s["env"].close()
    return raw, r


def main() -> None:
    ap = argparse.ArgumentParser()
    Q.add_args(ap)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reveal-s", type=float, default=2.0,
                    help="s before the end at which the verdict appears")
    args = ap.parse_args()
    args.episodes = 1

    out = args.out or f"peg_search_{args.err:.0f}mm"
    print(f"aim error {args.err:.0f} mm, seed {args.seed}")
    left, rl = run_arm(0.0, args.err, args.seed, args)
    print(f"  aim-and-push: success={rl['success']} deepest "
          f"{rl['deepest']*1e3:.1f} mm Fpk {rl['f'].max():.1f} N")
    right, rr = run_arm(args.search_s, args.err, args.seed, args)
    print(f"  with search:  success={rr['success']} deepest "
          f"{rr['deepest']*1e3:.1f} mm Fpk {rr['f'].max():.1f} N caught "
          f"{rr['caught']}")

    def verdict(r) -> str:
        """What actually happened, not just pass/fail.

        "jam" in the sweep's tally means the head got past the face and did not
        go home.  On these seeds that splits into two different events, and the
        video should not call both of them the same thing: the peg can stay
        wedged in the mouth, or the wedge can load the grasp hard enough that
        the fingers lose it and the peg ends up on the table.
        """
        if r["success"]:
            return "INSERTED"
        if r["deepest"] <= 0.005:
            return "NEVER FOUND IT"
        return ("WEDGED, THEN LOST THE PEG" if r["depth"][-1] < 0.0
                else "WEDGED IN THE MOUTH")

    # Both arms run the same duration, so the frame lists line up; zip to the
    # shorter one anyway rather than assuming it.
    n = min(len(left), len(right))
    td = np.array([f[1] for f in left[:n]])
    frames = []
    for i in range(n):
        panes = []
        for (raw, res, name, col) in ((left, rl, "aim and push", (250, 170, 150)),
                                      (right, rr, "scrub, then push", (150, 230, 150))):
            img, t, depth, f = raw[i]
            dep, frc = binned(res, td)
            # The verdict is a REVEAL, not a caption: labelling the left pane
            # "JAMMED" from the first frame both spoils it and is false at the
            # time it is shown.
            tail = (f"-> {verdict(res)}" if t >= td[-1] - args.reveal_s else "")
            # Peak over EVERY physics step up to now, not over the 20 fps
            # samples: contact spikes last a few milliseconds, so the sampled
            # maximum read 33 N on an episode whose real peak was 52 N.
            peak = float(res["f"][res["t"] <= t + 1e-9].max())
            img = label(img, name, [
                f"aim error {args.err:.0f} mm   (clearance 3 mm, no chamfer)",
                f"head {1000*depth:+7.1f} mm past the face",
                f"contact force {f:5.1f} N   (peak so far {peak:4.1f} N)",
                f"t = {t:5.2f} s      {tail}"], col)
            panes.append(trace(img, [("depth mm", 1000 * dep, 120.0, (120, 200, 255)),
                                     ("force N", frc, 60.0, (255, 140, 120))],
                               td, i))
        frames.append(np.hstack(panes))
    save(frames, out, args.fps)


if __name__ == "__main__":
    main()
