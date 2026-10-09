"""The synthetic operator for wiping: the writer, handed a raster.

`../writing/teleop.py` already models the part that makes an operator an
operator -- the visual loop that keeps the tool on the line it can only see
`visual_delay` late, the haptic loop that presses until the felt force matches
what was intended, the arm on the master, the motor noise, and the phase
machine travel -> settle -> descend -> press -> stroke -> lift.  None of that
is about writing.  What makes it write is the list of strokes it is given.

So a wiper IS that writer with the raster as its strokes.  Measured: the
unmodified `SyntheticWriter`, handed four raster rows, takes the glyph off both
boards -- 100% erased, 94-95% of pad-down time inside the force band, 3.6-4.1 N
peak (its own soft landing, against 9.1 N for the scripted driver in smoke.py).

TWO THINGS THAT FOLLOW, AND MATTER:

  IT FINDS THE SURFACE BY FEEL.  The haptic loop presses against what it
  measures, so nothing here is given the board's height profile -- unlike
  `smoke.py`, which is handed `frame.to_world` and therefore the shape.  On the
  curved board that is the difference between a demonstration a policy could
  imitate from force and vision and one it could not.

  IT CHOOSES K_R.  A flat pad on a surface whose normal turns has a rotational
  stiffness to pick, which a ball-point pen never did (CURVED_BOARD.md: K_R 60
  -> 0.3 moves the glyph off from 50% to 84% on the curve, and buys nothing on
  the flat).  `kr_target` is the schedule; `protocol.py` overrides it with the
  operator's own levels, exactly as it overrides the translational one.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import teleop as T  # noqa: E402  ../writing/teleop.py
import wipe_scene as SC  # noqa: E402


def raster_rows(marks_uv: np.ndarray, pad: float = SC.ERASER_R,
                margin: float = 1.0) -> list[np.ndarray]:
    """Boustrophedon rows over the marks' bounding box, one pad wide.

    Rows are spaced by the pad's HALF-width, so consecutive passes overlap by
    half: a pad that is a little misaligned still covers the gap its neighbour
    missed, which is what a person does and what the curved board needs.
    """
    lo, hi = marks_uv.min(0), marks_uv.max(0)
    out = []
    for i, v in enumerate(np.arange(lo[1] - pad / 2, hi[1] + pad, pad)):
        us = [lo[0] - margin * pad, hi[0] + margin * pad][:: 1 if i % 2 == 0 else -1]
        out.append(np.array([[us[0], v], [us[1], v]]))
    return out


@dataclass
class WiperStyle(T.WriterStyle):
    """`WriterStyle` with the numbers a wiper differs on, and K_R.

    Erasing is work -- pressure times sliding -- so a wiper presses harder and
    moves faster than a writer, and has no corner to stiffen for.
    """
    f_intent: float = 3.5        # N.  The band is 1-6; erasure is f * slide
    v_write: float = 0.045       # m/s along a row, half again a writer's
    corner_boost: float = 0.0    # a straight row has no corners
    # ---- the rotational stiffness they command, Nm/rad ----
    # The levels the protocol uses, as a default schedule for anyone driving
    # this wiper directly.  0.3 is below the knee at f*r ~ 0.05 Nm, 30 is
    # rigid; see CURVED_BOARD.md for the sweep that puts the knee near 1.
    kr_travel: float = 30.0      # off the board: a loose wrist flops
    kr_land: float = 30.0        # land FLAT, not on a corner
    kr_wipe: float = 0.3         # let the board set the pad's angle
    kr_tau: float = 0.15         # s, the same co-contraction constant as K_p

    @staticmethod
    def sample(seed: int) -> "WiperStyle":
        base = T.WriterStyle.sample(seed)
        r = np.random.default_rng(40_000 + seed)
        return WiperStyle(
            **{k: v for k, v in base.__dict__.items()
               if k not in ("f_intent", "v_write", "corner_boost")},
            f_intent=float(r.uniform(2.5, 4.5)),
            v_write=float(r.uniform(0.035, 0.055)),
            corner_boost=0.0)


class SyntheticWiper(T.SyntheticWriter):
    """The writer, driving a raster instead of a glyph."""

    def __init__(self, sim, style: WiperStyle, margin: float = 1.0):
        super().__init__(sim, style)
        self.rows = raster_rows(sim.marks_uv, SC.ERASER_R, margin)
        self.strokes = [T._curvature(r) for r in self.rows]
        self.kr_cmd = float(style.kr_travel)

    def kr_target(self) -> float:
        """The rotational stiffness for the phase, low-passed like a muscle.

        Call it once per step, after `act`: it reads the phase `act` left.
        """
        st = self.st
        want = {"travel": st.kr_travel, "settle": st.kr_land, "descend": st.kr_land,
                "press": st.kr_wipe, "stroke": st.kr_wipe, "finish": st.kr_wipe,
                "lift": st.kr_travel, "done": st.kr_travel,
                "finished": st.kr_travel}[self.phase]
        self.kr_cmd += (want - self.kr_cmd) * min(1.0, self.dt / st.kr_tau)
        return self.kr_cmd


class GlyphWiper(SyntheticWiper):
    """A wiper that ERASES ALONG THE GLYPH, the way a hand does.

    `SyntheticWiper` rasters: boustrophedon rows over the marks' bounding box.  That
    is the easy way to cover ink and it is not how anyone wipes a letter off a board
    -- a hand traces the S.  Two things follow, and both are the reason this class
    exists:

      THE PATH IS LONGER and it is mostly NOT axis-aligned, so the pad is dragged in
      a direction that keeps turning.

      THE WRIST HAS SOMETHING TO DO.  A raster row is straight, so the pad's heading
      is constant down it and K_R is asked for almost nothing; on a curve the contact
      patch turns under the pad the whole way.  A flat board rastered is a task with
      no rotational demand in it at all, and "K_R does not matter" measured on that
      is a statement about the path, not about the wrist.

    It is the parent's own behaviour, unsuppressed: `SyntheticWriter` already follows
    `sim.target.strokes`, and SyntheticWiper replaces them with rows.  This puts them
    back.
    """

    def __init__(self, sim, style: "WiperStyle", margin: float = 1.0):
        super().__init__(sim, style, margin)
        self.rows = [np.asarray(s, float) for s in sim.target.strokes]
        self.strokes = [T._curvature(s) for s in sim.target.strokes]
