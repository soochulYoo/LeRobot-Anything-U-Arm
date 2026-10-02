"""One wiping episode: the writing simulator with the eraser's verbs.

Module names are deliberately `wipe_*`: the writing package uses flat imports
(`import sim`), so a sibling file called `sim.py` would shadow it depending on
sys.path order -- which it did, once.

The board starts with the glyph already on it; the task is to take it off.
Everything about physics, contact force, the Case 1 controller, the energy tank
and the wrench filtering is inherited from `../writing/sim.py` -- only what the
tool does to the board changes, through the two hooks that file exposes.

Erasing is modelled as WORK, not as a switch:

    wear_i  +=  pressure * |slide|      for every mark i under the pad
    mark i is gone when wear_i >= erase_work

so a mark comes off by being pressed AND rubbed.  Pressing harder erases in
fewer passes, hovering erases nothing, and dragging a dry pad over the glyph
without force does nothing either -- which is what makes this a force task
rather than a reaching task.

Scoring mirrors writing's two-sided check:

    erased     >= 90%   of the marks are gone          (writing: coverage)
    in-band    >= 85%   of pad-down time inside 1-6 N  (same)
    no tear, no collision                              (same)
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import glyphs as G            # noqa: E402  writing/glyphs.py
import sim as SM              # noqa: E402  writing/sim.py, the simulator we extend
import wipe_scene as SC       # noqa: E402  our eraser scene


def board_height(frame, uv):
    """Surface height above the board's mean plane at (u, v).

    Zero for a frame that has no shape, which is every flat `CanvasFrame`, so
    one code path serves both boards.  Asked for by name rather than branched
    on a flag: the only thing the erasure model needs to know about a curved
    board is this function.
    """
    h = getattr(frame, "height", None)
    return 0.0 if h is None else h(uv)


@dataclasses.dataclass
class WipeCriteria(SM.Criteria):
    erase_work: float = 0.040      # N*m of pressure x sliding to lift one mark
    min_erased: float = 0.90
    mark_spacing: float = 0.0015   # m between the marks laid on the board
    pad_give: float = 0.004        # m the felt compresses.  A RIGID tilted disc
                                   # touches along a line of zero area and erases
                                   # nothing; a real eraser squashes, and that
                                   # compression is what sets the patch width.


class WipingSim(SM.WritingSim):
    """`WritingSim` with the eraser's contact point and the eraser's effect."""

    ENV_ID = "TeleopWiping-v1"
    TOOL_LINK = "eraser"

    @staticmethod
    def tool_urdf() -> str:
        return SC.PandaEraser.urdf_path

    def __init__(self, *args, criteria: WipeCriteria | None = None, **kwargs):
        super().__init__(*args, criteria=criteria or WipeCriteria(), **kwargs)

    # ---- the tool -------------------------------------------------------
    def contact_point(self, rec, n):
        """The pad's face centre.  The pad's AXIS is stashed here because the
        next hook needs to know how flush the pad is lying."""
        self._pad_axis = rec["R"][:, 2]
        return rec["p"]

    def on_contact(self, f_n, uvh, down) -> None:
        if not down:
            self._last_uv = None
            return
        uv = np.asarray(uvh[:2], dtype=float)
        slide = 0.0 if self._last_uv is None else float(np.linalg.norm(uv - self._last_uv))
        self._last_uv = uv.copy()
        if slide <= 0.0 or not len(self.marks_uv):
            return
        # Where the pad actually touches.  A rigid disc tilted by theta against
        # the board penetrates by d(x) = d0 - x . g across its face, so only the
        # part with d > 0 is in contact: a chord of the disc, not the whole disc.
        # Without this the model is blind to pad geometry and rotational
        # stiffness has no way to matter -- which is exactly what the first
        # K_R comparison showed.
        a = self.frame.R.T @ self._pad_axis          # pad axis in canvas axes
        if a[2] > 0:
            a = -a                                   # point it into the board
        if abs(a[2]) < 1e-9:
            return
        d = self.marks_uv - uv
        # THE BOARD'S OWN SHAPE.  The pad's face is a plane, the board is not:
        # a mark sitting dh higher than the contact point is pressed dh harder,
        # and one in a dip may not be touched at all.  This is the whole of
        # what the curved board changes in the erasure model, and it is exactly
        # zero on a flat one, so the flat behaviour is unchanged to the bit.
        # First order in the pad's tilt, exact in the surface: the pad's
        # penetration at mark i is (surface at i) - (face plane at i).
        dh = board_height(self.frame, self.marks_uv) - board_height(self.frame, uv)
        pen = dh - uvh[2] - d @ (-a[:2] / a[2])      # how deep the face is at each mark
        squash = pen + self.crit.pad_give            # how hard the felt is pressed
        under = (np.linalg.norm(d, axis=1) <= SC.ERASER_R) & (squash > 0) & ~self.gone
        if not under.any():
            return
        # The normal force spreads over that patch in proportion to how far the
        # felt is compressed (a linear spring bed).  Normalised so a flush pad
        # reproduces the old uniform model exactly, and a tilted one puts the
        # same force through a narrow crescent: it rubs harder over less glyph.
        share = squash[under] / squash[under].sum() * under.sum()
        self.wear[under] += f_n * slide * share
        newly = under & (self.wear >= self.crit.erase_work)
        for k in np.nonzero(newly)[0]:
            self.u.wipe(int(k))
        self.gone |= newly

    # ---- episode --------------------------------------------------------
    def reset(self, spec: SM.TaskSpec, hover: float = 0.040) -> dict:
        # no grey guide: on this task the marks themselves are the target
        obs = super().reset(dataclasses.replace(spec, show_template=False), hover=hover)
        marks = np.concatenate([G.resample(s, self.crit.mark_spacing)
                                for s in self.target.strokes])
        marks = marks[:self.u.MAX_INK]
        self.u.show_ink(marks)
        self.marks_uv = marks
        self.wear = np.zeros(len(marks))
        self.gone = np.zeros(len(marks), dtype=bool)
        self._last_uv = None
        return obs

    # ---- what a policy sees, and what it is asked for ------------------
    def observe(self, images: bool = True) -> dict:
        """Writing's observation plus K_R, which is an action on this task."""
        obs = super().observe(images)
        obs["kr"] = np.float32(self.ctl.kr)
        return obs

    def goal(self) -> dict:
        """The marks to remove, in the BELIEVED frame -- the mirror of writing's
        goal, which is the strokes to draw.  Where they truly sit is
        privileged, and on a curved board that includes the surface under them.
        """
        g = super().goal()
        g["marks_uv"] = np.asarray(self.marks_uv, dtype=np.float32)
        g["marks_world"] = self.belief.to_world(self.marks_uv).astype(np.float32)
        return g

    def score(self) -> dict:
        cr = self.crit
        erased = float(self.gone.mean()) if len(self.gone) else 0.0
        in_band = self.in_band_steps / max(1, self.pen_down_steps)
        res = dict(erased=erased, n_marks=int(len(self.gone)),
                   n_left=int((~self.gone).sum()), in_band=float(in_band),
                   peak_force=float(self.peak), peak_force_fast=float(self.peak_fast),
                   pad_down_s=self.pen_down_steps * self.dt,
                   torn=bool(self.torn), collided=bool(self.collided),
                   t=self.t, tank_E=float(self.ctl.E))
        checks = {
            "erased": erased >= cr.min_erased,
            "force": in_band >= cr.min_in_band,
            "no_tear": not self.torn,
            "no_collision": not self.collided,
        }
        res["checks"] = checks
        res["success"] = bool(all(checks.values()))
        res["fail_reason"] = ",".join(k for k, v in checks.items() if not v)
        return res


class CurvedWipingSim(WipingSim):
    """The wiping episode on a curved board.

    Everything that makes the board curved is in `wipe_scene.CurvedWipingEnv`
    and `wipe_scene.CurvedFrame`; all that is left here is to adopt the frame
    the env built when it posed the slab.  `writing/sim.py` constructs a flat
    `CanvasFrame` from the episode's height error and tilt and hands it to
    `place_canvas`, which is where the env puts it together with the surface it
    owns -- so the curved frame only exists once that has happened, i.e. after
    `super().reset`.  The marks are already laid on the surface by then,
    because `show_ink` goes through the env's `_dot_pose`, which has had the
    curved frame all along.

    What this does NOT change is the controller or the command: the pad is
    still told to hold one vertical orientation, and the board still never
    tells it otherwise.  That is what K_R is being asked to absorb.
    """

    ENV_ID = "TeleopWipingCurved-v1"

    @property
    def surf(self):
        return self.u.surf

    def reset(self, spec: SM.TaskSpec, hover: float = 0.040) -> dict:
        obs = super().reset(spec, hover)
        self.frame = self.u._frame
        return obs

    def privileged(self) -> dict:
        """The board's shape is privileged exactly as its pose is.

        As an ARRAY, not the dict it is declared as: every entry of this
        mapping is written out as an h5 dataset, and a dict is not one.  The
        order is the one named below, and the full spec is in the file's attrs.
        """
        s = self.u.SURF
        return dict(super().privileged(),
                    surf_half_amp_sigma_seed=np.array(
                        [s["half"], s["amp"], s["sigma"], s["seed"]], dtype=float),
                    board_h_ptp=float(np.ptp(self.frame.height(self.marks_uv))))
