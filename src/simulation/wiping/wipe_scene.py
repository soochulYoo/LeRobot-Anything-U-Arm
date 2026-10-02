"""Whiteboard wiping: the writing scene with an eraser pad instead of a pen.

The reverse of `../writing`: the board starts with the glyph already on it and
the robot has to take it off.  Everything that is not about the tool is reused
by subclassing -- the Panda, the canvas, the cameras, the ink dots, the frame
convention -- so the two tasks cannot drift apart in the parts they share.

What differs is the tool, and it is not a cosmetic change:

  * a ball-point pen touches at ONE point, so its orientation barely matters;
    an eraser is a FLAT PAD, and a pad meets a tilted board on an edge unless
    its orientation is right.  Wiping is therefore the task where rotational
    compliance earns its keep, which the writing task could not exercise.
    The pad is a BOX, not a cylinder: a face-on-face contact gives the solver
    several contact points and therefore a moment for K_R to resist, which a
    cylinder resolved to a single point did not (see ../curved/wipe.py).
  * the pad sweeps an AREA, so erasing is about covering the glyph with the
    footprint while pressing, not about tracing a line.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

import numpy as np
import sapien
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.utils.registration import register_env
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, qmult

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "writing"))
# APPEND for the sibling study: `src/simulation` holds a vendored `mani_skill/`
# that must not reach the front of sys.path (../curved/record.py was bitten).
sys.path.append(str(HERE.parent / "curved"))

import scene as W  # noqa: E402  the writing scene, reused wholesale
from surface import CurvedSurface  # noqa: E402  ../curved/surface.py

# Pad half-width.  An env var because the URDF is generated when this module is
# imported, so the size has to be fixed before that: `WIPE_PAD_R=0.040 python ...`.
# It also sets how far the normal swings across the pad, which is the whole of
# what rotational stiffness has to work against.
ERASER_R = float(os.environ.get("WIPE_PAD_R", 0.015))   # m
ERASER_H = 0.020          # m, pad thickness
PAD_ROUND = 0.002         # m, radius of the pad's rim (see `_pad_mesh`)
TIP_Z = W.PEN_TIP_Z       # keep the TCP convention: panda_hand_tcp at the contact face


def _pad_mesh(out: pathlib.Path) -> str:
    """The pad's collision shape: a box with a ROUNDED RIM, written to `out`.

    A sharp-edged box cannot be dragged over a triangle mesh.  PhysX generates
    contacts against the mesh's internal edges, with normals pointing sideways,
    so the pad meets what is effectively a vertical wall; measured on a FLAT
    mesh slab, where the only correct answer is a smooth slide, a box pad
    stalled dead with 17 N mean and a 74 N peak against a 2.4 N command
    (`curved_probe.py --amp 0`).  Flat box-on-box contact is analytic and never
    showed this, which is why the flat-canvas wiping task did not reveal it and
    the curved board does: a curved board HAS to be a mesh.

    Rounding the rim puts the contact on a curved surface whose normal follows
    the slab, which is also what the corner of a felt eraser does.  The flat
    face is untouched -- 26 mm of it -- so the pad still meets the board
    face-on and still transmits the moment K_R has to resist.

    Outer dimensions are exactly the box's (2*ERASER_R, 2*ERASER_R, ERASER_H),
    so ERASER_R stays the pad's half-width and the erasure model in wipe_sim.py
    needs no adjustment.
    """
    import trimesh
    path = out / (f"eraser_pad_{1000*ERASER_R:g}x{1000*ERASER_H:g}"
                  f"r{1000*PAD_ROUND:g}.obj")
    if not path.exists():
        r = PAD_ROUND
        hs = (ERASER_R - r, ERASER_R - r, ERASER_H / 2 - r)
        # Minkowski sum of the shrunken box and a sphere, as the convex hull of
        # a sphere at each of its eight corners.  216 hull vertices at
        # subdivisions=2, inside PhysX's 255-vertex convex limit.
        balls = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    b = trimesh.creation.icosphere(subdivisions=2, radius=r)
                    b.apply_translation([sx * hs[0], sy * hs[1], sz * hs[2]])
                    balls.append(b)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.obj")
        trimesh.util.concatenate(balls).convex_hull.export(tmp)
        os.replace(tmp, path)
    return str(path)


def _eraser_urdf(pad_mesh: str) -> str:
    """panda_stick.urdf with the stick replaced by a flat eraser pad.

    Generated for the same reason the pen is: the mesh paths have to be absolute.
    """
    src_dir = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda"
    txt = (src_dir / "panda_stick.urdf").read_text()
    txt = txt.replace('filename="franka_description/', f'filename="{src_dir}/franka_description/')
    txt = re.sub(r'\s*<visual>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</visual>', "",
                 txt, count=1, flags=re.S)
    txt = re.sub(r'\s*<collision>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</collision>', "",
                 txt, count=1, flags=re.S)
    stem_lo, stem_hi = 0.05, TIP_Z - ERASER_H
    pad_z = TIP_Z - ERASER_H / 2
    pad = f"""
  <link name="eraser">
    <visual>
      <origin xyz="0 0 {(stem_lo + stem_hi) / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder radius="0.005" length="{stem_hi - stem_lo:.4f}"/></geometry>
      <material name="holder"><color rgba="0.20 0.20 0.22 1"/></material>
    </visual>
    <visual>
      <origin xyz="0 0 {pad_z:.4f}" rpy="0 0 0"/>
      <geometry><mesh filename="{pad_mesh}"/></geometry>
      <material name="felt"><color rgba="0.85 0.45 0.25 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {(stem_lo + stem_hi - 0.004) / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder radius="0.005" length="{stem_hi - 0.004 - stem_lo:.4f}"/></geometry>
    </collision>
    <collision>
      <origin xyz="0 0 {pad_z:.4f}" rpy="0 0 0"/>
      <geometry><mesh filename="{pad_mesh}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="0.03"/>
      <inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="2e-5"/>
    </inertial>
  </link>
  <joint name="eraser_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="eraser"/>
  </joint>
"""
    return txt.replace("</robot>", pad + "</robot>")


def _ensure_assets() -> str:
    """Write the eraser URDF **and its SRDF**, and return the URDF path.

    THE SRDF IS NOT OPTIONAL.  ManiSkill's loader looks for `<stem>.srdf` beside
    the URDF and, finding none, disables no self-collisions at all.  panda_hand
    and panda_link7 are rigidly connected through the two fixed joints at
    panda_link8 but are not parent and child, so PhysX does not filter them --
    and their collision meshes overlap by 23 mm by construction.  The result is
    a contact the solver can never resolve, pinning the wrist: the pad then
    holds whatever orientation it was commanded no matter what K_R says, and a
    traverse stalls outright once the lateral spring loads up.  The first K_R
    comparison here (9.9 deg misaligned at K_R = 60, 8.8 at 0.3, i.e. no effect)
    was measured in that state.  ../writing/scene.py writes its pen SRDF for
    this reason; the eraser was copied without it.
    """
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    urdf = out / "panda_eraser.urdf"
    txt = _eraser_urdf(_pad_mesh(out))
    if not urdf.exists() or urdf.read_text() != txt:
        W._write_atomic(urdf, txt)
    src = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda" / "panda_stick.srdf"
    srdf = out / "panda_eraser.srdf"
    s = src.read_text().replace(
        "</robot>", '    <disable_collisions link1="panda_hand" link2="eraser" reason="Adjacent"/>\n'
                    '    <disable_collisions link1="panda_link7" link2="eraser" reason="Adjacent"/>\n</robot>')
    if not srdf.exists() or srdf.read_text() != s:
        W._write_atomic(srdf, s)
    return str(urdf)


@register_agent()
class PandaEraser(PandaStick):
    """PandaStick with a flat eraser pad; same 7 joints and TCP convention."""
    uid = "panda_eraser"
    urdf_path = _ensure_assets()


@register_env("TeleopWiping-v1", max_episode_steps=1_000_000)
class TeleopWipingEnv(W.TeleopWritingEnv):
    """The writing scene with the eraser agent.  The ink dots are the marks to
    remove rather than the marks being made, so the only new verb is `wipe`."""

    SUPPORTED_ROBOTS = ["panda_eraser"]
    agent: PandaEraser

    def __init__(self, *args, robot_uids="panda_eraser", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def show_ink(self, uv_points: np.ndarray) -> int:
        """Lay the marks that have to come off."""
        n = min(len(uv_points), self.MAX_INK)
        for i in range(n):
            self.ink_dots[i].set_pose(self._dot_pose(uv_points[i], self.INK_THICK / 2))
        return n

    def wipe(self, k: int) -> None:
        """Take mark k off the board."""
        self.ink_dots[k].set_pose(sapien.Pose(p=[0, 0, -1.0]))


# --------------------------------------------------------------------------- #
# the curved board
# --------------------------------------------------------------------------- #
def align_z(R: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`R` with its third column swung onto `n`, staying right-handed.

    Used to stand a mark up on the LOCAL normal instead of the board's mean
    one.  The first column is kept as far as the new normal allows, so a row of
    marks does not spin about its own axis as it crosses a bump.
    """
    z = np.asarray(n, dtype=float)
    z = z / np.linalg.norm(z)
    x = R[:, 0] - (R[:, 0] @ z) * z
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class CurvedFrame(W.CanvasFrame):
    """`CanvasFrame` for a board that is not flat.

    Same contract, so `writing/sim.py` and the scripted drivers do not know the
    difference: `(u, v)` are still the in-plane coordinates of the board's mean
    plane, and the third coordinate is still "off the board".  What changes is
    that it is measured from the LOCAL SURFACE rather than from the plane:

        to_canvas(p)[2]      height of p above the surface under it
        to_world([u, v, k])  the point k above the surface at (u, v)

    so `to_world(np.r_[uv, -penetration])` keeps meaning "press in by this
    much", and every raster driver written for the flat board presses correctly
    on the curve without being told.  That is deliberate: a scripted
    demonstrator is allowed to know the surface (it would see it), while the
    CONTROLLER still only gets a position command and still holds whatever
    orientation it was given.

    The field is offset so that `height(0, 0) = 0`: the board's centre lands
    exactly where the flat board's did, which keeps the start pose, the hover
    height and the IK of every episode unchanged.

    `normal` is the MEAN-PLANE normal, as on a flat board.  Code that wants the
    true local one asks for `normal_at(uv)`.  The one place in `writing/sim.py`
    that uses `normal` is the projection of the contact wrench onto "into the
    board", where the mean plane costs at most a few per cent of the magnitude
    (the local normal is within 10-16 deg of it) and the erasure threshold is a
    free parameter anyway.  The place where the local normal is NOT optional is
    the contact geometry in `wipe_sim.on_contact`, which asks for it by name.
    """

    def __init__(self, origin, tilt_x: float = 0.0, tilt_y: float = 0.0,
                 surf: CurvedSurface | None = None):
        super().__init__(origin, tilt_x, tilt_y)
        self.surf = surf if surf is not None else CurvedSurface()
        self.h0 = float(self.surf.height(0.0, 0.0))

    def height(self, uv) -> np.ndarray:
        """Surface height above the mean plane at board coordinates (u, v)."""
        uv = np.asarray(uv, dtype=float)
        return self.surf.height(uv[..., 0], uv[..., 1]) - self.h0

    def normal_at(self, uv) -> np.ndarray:
        """The TRUE outward normal at (u, v), in world coordinates."""
        uv = np.asarray(uv, dtype=float)
        return self.surf.normal(uv[..., 0], uv[..., 1]) @ self.R.T

    def to_world(self, uvh) -> np.ndarray:
        uvh = np.asarray(uvh, dtype=float)
        pad = np.zeros(uvh.shape[:-1] + (3,))
        pad[..., :uvh.shape[-1]] = uvh
        pad[..., 2] += self.height(pad[..., :2])
        return self.origin + pad @ self.R.T

    def to_canvas(self, p) -> np.ndarray:
        uvw = (np.asarray(p, dtype=float) - self.origin) @ self.R
        out = np.array(uvw, dtype=float)
        out[..., 2] -= self.height(uvw[..., :2])
        return out


@register_env("TeleopWipingCurved-v1", max_episode_steps=1_000_000)
class CurvedWipingEnv(TeleopWipingEnv):
    """The wiping scene with the paper replaced by a slab of curved board.

    A SUBCLASS rather than a flag, so the flat scene is not merely the default
    but is bit-for-bit the code it always was: nothing below runs unless this
    env id is the one asked for.

    THE SHAPE IS PART OF THE SCENE, THE POSE IS PART OF THE EPISODE.  The
    height field is baked into a collision mesh at reconfigure, so `SURF` is a
    class attribute read once when the scene is built, while height error and
    tilt stay per-episode as they are on the flat board (the slab is KINEMATIC,
    like the paper, so `place_canvas` can still re-pose it every reset).  A run
    that wants several shapes wants several envs; set `SURF` before building.

    `self.canvas` IS REBOUND TO THE SLAB.  Everything that reads the board --
    the pad/board contact force in `writing/sim.py:step`, the collision check
    on the other links, the friction material -- goes through `self.canvas`, so
    rebinding the name is what makes all of it work on the curve untouched.
    The flat box is parked below the floor and kept as `self.flat_canvas`.
    """

    # 35 mm bumps, not ../curved/surface.py's default 50: a bump of amplitude a
    # and width s has radius of curvature ~ s^2/a, and what matters is how much
    # the normal turns ACROSS THE TOOL.  Our pad is 30 mm where ../curved
    # welded a 50 mm one, so the same demand needs a sharper surface -- 8.9 deg
    # of tilt and 3.9 deg of swing across the pad along the traverse that
    # `curved_probe.py` measures K_R on.  `half` matches CANVAS_HALF's long
    # side, so the board is square where the paper was 320x240 mm.
    SURF = dict(half=0.16, amp=0.010, sigma=0.035, seed=3)
    # WIPE_SURFACE=patchy swaps the uniformly bumpy slab for one whose curvature
    # changes band to band along the wipe, so a single traverse crosses a regime
    # where K_R is irrelevant (flat, ~3 deg of swing across the pad) and one
    # where it decides the task (~12 deg).  A constant K_R cannot serve both;
    # that is the whole point of the environment.  See patchy_surface.py.
    SURF_PATCHY = dict(half=0.16, seed=3)
    MESH_N = 161                      # 2 mm facets, as curved_probe.py uses

    def __init__(self, *args, **kwargs):
        mode = os.environ.get("WIPE_SURFACE", "curved")
        if mode == "dome":
            from patchy_surface import DomeSurface
            self.surf = DomeSurface(half=0.16, sigma=0.05, n_bumps=1,
                                    radius=float(os.environ.get("WIPE_DOME_R", 0.25)))
        elif mode == "patchy":
            from patchy_surface import PatchySurface
            self.surf = PatchySurface(**self.SURF_PATCHY)
        elif mode == "corr":
            # Flat facets of alternating tilt.  The only family inside
            # residual <= pad_give < 2 r tan(tilt); gate_feasible.py --family corr
            # screens the geometry, so the numbers belong on the command line.
            from patchy_surface import CorrugatedSurface
            self.surf = CorrugatedSurface(
                half=0.16, seed=3,
                plateau=float(os.environ.get("WIPE_CORR_PLATEAU", 0.030)),
                flank=float(os.environ.get("WIPE_CORR_FLANK", 0.100)),
                slope_deg=float(os.environ.get("WIPE_CORR_SLOPE", 12.0)))
        else:
            self.surf = CurvedSurface(**self.SURF)
        super().__init__(*args, **kwargs)

    def _slab_path(self) -> str:
        out = HERE / "assets"
        out.mkdir(exist_ok=True)
        path = out / f"slab_{self.surf.key(self.MESH_N)}.obj"
        if not path.exists():
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.obj")
            self.surf.mesh(self.MESH_N).export(tmp)
            os.replace(tmp, path)
        return str(path)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.slab_material = sapien.physx.PhysxMaterial(0.3, 0.3, 0.0)
        b = self.scene.create_actor_builder()
        # Nonconvex: the convex hull of this mesh is a dome and would erase
        # every dent, which is half of what makes the board curved.  Kinematic
        # rather than static so the pose stays per-episode; PhysX allows a
        # triangle mesh on anything that is not dynamic.
        b.add_nonconvex_collision_from_file(self._slab_path(), material=self.slab_material)
        b.add_visual_from_file(self._slab_path(), material=sapien.render.RenderMaterial(
            base_color=[0.95, 0.95, 0.93, 1]))
        b.initial_pose = sapien.Pose(p=[0.0, 0.0, -5.0])
        slab = b.build_kinematic(name="canvas_curved")
        self.flat_canvas, self.canvas = self.canvas, slab

    def place_canvas(self, frame: W.CanvasFrame, friction: float) -> None:
        """Pose the slab, and build the CURVED frame the rest of the run uses.

        `writing/sim.py` hands in the flat frame it made from the episode's
        height error and tilt; the surface belongs to the env, so the env is
        where the two are put together.  `wipe_sim.CurvedWipingSim` reads the
        result back out of `self._frame`.
        """
        self.flat_canvas.set_pose(sapien.Pose(p=[0.0, 0.0, -5.0]))
        self._frame = CurvedFrame(frame.origin, *frame.tilt, surf=self.surf)
        # The mesh carries the raw height field; the frame's is offset to put
        # zero at the board's centre, so the actor has to be dropped by h(0,0)
        # for the two to agree.
        self.canvas.set_pose(sapien.Pose(
            p=self._frame.origin - self._frame.h0 * self._frame.normal,
            q=mat2quat(self._frame.R)))
        self.slab_material.set_static_friction(float(friction))
        self.slab_material.set_dynamic_friction(float(friction))

    def _dot_pose(self, uv, lift: float) -> sapien.Pose:
        """A mark lying ON the surface, standing on the LOCAL normal."""
        f = self._frame
        return sapien.Pose(p=f.to_world(np.r_[uv, lift]),
                           q=qmult(mat2quat(align_z(f.R, f.normal_at(uv))),
                                   euler2quat(0, np.pi / 2, 0)))
