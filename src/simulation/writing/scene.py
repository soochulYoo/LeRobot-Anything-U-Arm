"""The writing scene: a Panda holding a ball-point pen over a sheet of paper.

WHY NOT ManiSkill's DrawSVG / TableTopFreeDraw
------------------------------------------------
Those tasks are kinematic by their own admission ("We do not actually check if
the robot contacts the table"): ink appears whenever the TCP is within 5 mm of
the canvas, so force cannot affect success and there is nothing for a stiffness
to do.  Here the paper is a real collision body and ink is laid down only while
the MEASURED pen/paper contact force is above `ink_force` -- press too lightly
and nothing is written, press too hard and the paper tears.  That makes force,
and therefore the impedance that produces it, part of the task.

THE PEN is a sphere, not the flat-ended stick of panda_stick.  The wipe study
(../cascade/README.md) found that an edge on a plane catches as soon as
tangential motion starts -- contact collapsed to 60% with fingers on a box and
held 100% with a sphere on a plate.  A ball-point is the same geometry.  The pen
is its own link, so pen/paper contact can be told apart from the hand hitting
the paper.

THE PAPER'S POSE IS RANDOM and not told to the operator: height within a few
millimetres and a few degrees of tilt.  That is the disturbance a compliant
normal axis absorbs and a stiff one turns into force error, which is what makes
a variable stiffness worth demonstrating.
"""
from __future__ import annotations

import os
import pathlib
import re

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, qmult

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

HERE = pathlib.Path(__file__).resolve().parent

# ---- pen geometry, in the panda_hand frame (z runs down the pen) ----
PEN_BALL_R = 0.004
PEN_TIP_Z = 0.150                 # the ball's lowest point; the TCP sits here
PEN_BODY_R = 0.006


def _pen_urdf() -> str:
    """panda_stick.urdf with the flat stick replaced by a ball-point pen link.

    Generated rather than shipped: the mesh paths must be absolute to load from
    outside ManiSkill's asset tree, which makes the file machine-specific.
    """
    src_dir = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda"
    txt = (src_dir / "panda_stick.urdf").read_text()
    txt = txt.replace('filename="franka_description/',
                      f'filename="{src_dir}/franka_description/')
    # drop the stick's visual and collision from panda_hand
    txt = re.sub(r'\s*<visual>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</visual>', "",
                 txt, count=1, flags=re.S)
    txt = re.sub(r'\s*<collision>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</collision>', "",
                 txt, count=1, flags=re.S)
    body_lo, body_hi = 0.05, PEN_TIP_Z - 2 * PEN_BALL_R + 0.001
    ball_z = PEN_TIP_Z - PEN_BALL_R
    pen = f"""
  <link name="pen">
    <visual>
      <origin xyz="0 0 {(body_lo + body_hi) / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder radius="{PEN_BODY_R}" length="{body_hi - body_lo:.4f}"/></geometry>
      <material name="pen_body"><color rgba="0.15 0.25 0.55 1"/></material>
    </visual>
    <visual>
      <origin xyz="0 0 {ball_z:.4f}" rpy="0 0 0"/>
      <geometry><sphere radius="{PEN_BALL_R}"/></geometry>
      <material name="pen_tip"><color rgba="0.8 0.8 0.82 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {(body_lo + body_hi - 0.004) / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder radius="{PEN_BODY_R}" length="{body_hi - 0.004 - body_lo:.4f}"/></geometry>
    </collision>
    <collision>
      <origin xyz="0 0 {ball_z:.4f}" rpy="0 0 0"/>
      <geometry><sphere radius="{PEN_BALL_R}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="0.02"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-6"/>
    </inertial>
  </link>
  <joint name="pen_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="pen"/>
  </joint>
"""
    txt = txt.replace("</robot>", pen + "</robot>")
    return txt


def _ensure_assets() -> str:
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    urdf = out / "panda_pen.urdf"
    txt = _pen_urdf()
    if not urdf.exists() or urdf.read_text() != txt:
        _write_atomic(urdf, txt)
    srdf_src = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda" / "panda_stick.srdf"
    srdf = out / "panda_pen.srdf"
    s = srdf_src.read_text().replace(
        "</robot>", '    <disable_collisions link1="panda_hand" link2="pen" reason="Adjacent"/>\n'
                    '    <disable_collisions link1="panda_link7" link2="pen" reason="Adjacent"/>\n</robot>')
    if not srdf.exists() or srdf.read_text() != s:
        _write_atomic(srdf, s)
    return str(urdf)


def _write_atomic(path: pathlib.Path, text: str) -> None:
    """Many worker processes import this module at once; none may read a
    half-written robot description."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@register_agent()
class PandaPen(PandaStick):
    """PandaStick with a ball-point pen: same 7 joints, same TCP convention
    (panda_hand_tcp at the lowest point of the tip)."""
    uid = "panda_pen"
    urdf_path = _ensure_assets()


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def rot_xy(tilt_x: float, tilt_y: float) -> np.ndarray:
    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


# Writing direction.  The operator stands where the robot is, facing +x: their
# left is +y, so text runs toward -y and "up the letter" is +x (away from them).
# Columns are (u, v, n): canvas u, canvas v, outward normal.  Right-handed.
CANVAS_AXES = np.array([[0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])


class CanvasFrame:
    """The paper as a rigid frame: world = origin + R @ [u, v, n]."""

    def __init__(self, origin, tilt_x: float = 0.0, tilt_y: float = 0.0):
        self.origin = np.asarray(origin, dtype=float)
        self.R = rot_xy(tilt_x, tilt_y) @ CANVAS_AXES
        self.tilt = (float(tilt_x), float(tilt_y))

    @property
    def normal(self) -> np.ndarray:
        return self.R[:, 2]

    def to_world(self, uv) -> np.ndarray:
        uv = np.asarray(uv, dtype=float)
        pad = np.zeros(uv.shape[:-1] + (3,))
        pad[..., :uv.shape[-1]] = uv
        return self.origin + pad @ self.R.T

    def to_canvas(self, p) -> np.ndarray:
        """World -> (u, v, height above the paper)."""
        return (np.asarray(p, dtype=float) - self.origin) @ self.R


# --------------------------------------------------------------------------- #
# the environment
# --------------------------------------------------------------------------- #
@register_env("TeleopWriting-v1", max_episode_steps=1_000_000)
class TeleopWritingEnv(BaseEnv):
    """Scene container.  Stepping, control, ink and scoring live in sim.py.

    This env is never stepped through BaseEnv.step: the Case 1 controller runs
    a torque law at every physics step and the operator runs at the same rate,
    so the scene is stepped directly, as every other SAPIEN script in this
    repository does.  What BaseEnv provides is the scene, the agent, the
    cameras and the reset/reconfigure machinery.
    """
    SUPPORTED_ROBOTS = ["panda_pen"]
    agent: PandaPen

    # nominal paper: top-surface centre, and half extents (u, v, thickness)
    CANVAS_CENTER = np.array([0.50, 0.0, 0.10])
    CANVAS_HALF = np.array([0.16, 0.12, 0.01])     # along world y, world x, z
    INK_R = 0.0012
    INK_THICK = 0.0004
    TEMPLATE_R = 0.0007
    MAX_INK = 2500
    MAX_TEMPLATE = 900

    def __init__(self, *args, image_size: int = 128, wrist_camera: bool = True,
                 robot_uids="panda_pen", **kwargs):
        self.image_size = int(image_size)
        self.use_wrist_camera = bool(wrist_camera)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---- config ----
    @property
    def _default_sim_config(self):
        # 500 Hz: study_layers.py put an impedance loop's floor at 200 Hz, and
        # the peg study found the ManiSkill default of 100 Hz unusable.
        return SimConfig(sim_freq=500, control_freq=500,
                         scene_config=SceneConfig(contact_offset=0.002,
                                                  solver_position_iterations=15,
                                                  solver_velocity_iterations=1))

    @property
    def _default_sensor_configs(self):
        c = self.CANVAS_CENTER
        s = self.image_size
        # From across the desk, 40 deg off vertical: steep enough that the hand
        # (15 cm above the tip) shadows paper BEYOND the text rather than the
        # text itself, and rolled 180 deg (up = -z) so the writing reads upright
        # in the image, as it does to the operator.  The field is ~0.2 m wide.
        cams = [CameraConfig(uid="top_camera",
                             pose=sapien_utils.look_at(c + [0.30, 0.0, 0.36], c, up=(0, 0, -1)),
                             width=s, height=s, fov=0.42, near=0.01, far=10)]
        if self.use_wrist_camera:
            # Looking down the pen from beside the hand, so the tip and the ink
            # just behind it are in view.  Mounted on panda_hand: z runs down
            # the pen, the tip is at z = PEN_TIP_Z.  Sensors are set up after
            # the agent is loaded, so the link exists by now.
            hand = sapien_utils.get_obj_by_name(self.agent.robot.links, "panda_hand")
            cams.append(CameraConfig(uid="wrist_camera",
                                     pose=sapien_utils.look_at([0.075, 0.0, 0.02],
                                                               [0.0, 0.0, PEN_TIP_Z + 0.01]),
                                     width=s, height=s, fov=1.1, near=0.01, far=10,
                                     mount=hand))
        return cams

    @property
    def _default_human_render_camera_configs(self):
        c = self.CANVAS_CENTER
        return CameraConfig(uid="render_camera",
                            pose=sapien_utils.look_at(c + [0.30, -0.26, 0.24], c + [-0.03, 0.0, 0.04]),
                            width=960, height=720, fov=0.8, near=0.01, far=10)

    # The keyboard operator's view (interactive.py): from behind and to the
    # right of where they stand, so the writing reads left to right, J/L move
    # the pen left/right on screen, I/K away/toward, and the hand does not hide
    # the tip.  The render camera sits across the desk, where the text is
    # upside down and every key is mirrored.
    VIEWER_EYE = np.array([-0.30, -0.15, 0.32])       # relative to CANVAS_CENTER
    VIEWER_AT = np.array([0.02, 0.0, 0.0])

    @property
    def _default_viewer_camera_configs(self):
        c = self.CANVAS_CENTER
        return CameraConfig(uid="viewer",
                            pose=sapien_utils.look_at(c + self.VIEWER_EYE, c + self.VIEWER_AT),
                            width=1280, height=900, fov=0.9, near=0.01, far=100,
                            shader_pack="default")

    def _setup_viewer(self):
        super()._setup_viewer()               # aims it at the render camera
        # Re-aim through the viewer's own fly-camera state: a bare
        # set_camera_pose is overridden by it (the view backs off ~10 cm and
        # widens).  Its yaw runs opposite to atan2 in the world frame.
        c = self.CANVAS_CENTER
        eye = c + self.VIEWER_EYE
        d = (c + self.VIEWER_AT) - eye
        d = d / np.linalg.norm(d)
        self._viewer.set_camera_xyz(*eye)
        self._viewer.set_camera_rpy(0.0, -np.arcsin(-d[2]), -np.arctan2(d[1], d[0]))
        self._viewer.window.set_camera_parameters(0.01, 100.0, 0.9)

    # ---- scene ----
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        c, h = self.CANVAS_CENTER, self.CANVAS_HALF
        ground = self.scene.create_actor_builder()
        ground.add_box_visual(half_size=[1.5, 1.5, 0.01],
                              material=sapien.render.RenderMaterial(base_color=[0.78, 0.78, 0.76, 1]))
        ground.add_box_collision(half_size=[1.5, 1.5, 0.01])
        ground.initial_pose = sapien.Pose(p=[0, 0, -0.01])
        ground.build_static(name="ground")

        desk_top = c[2] - 2 * h[2] - 0.004
        desk = self.scene.create_actor_builder()
        desk.add_box_visual(half_size=[0.24, 0.30, desk_top / 2],
                            material=sapien.render.RenderMaterial(base_color=[0.45, 0.33, 0.24, 1]))
        desk.initial_pose = sapien.Pose(p=[c[0], c[1], desk_top / 2])
        desk.build_static(name="desk")

        # The paper: kinematic, so its pose can be re-randomised every reset
        # without rebuilding the scene.  The material is kept so friction can be
        # re-randomised the same way.
        self.paper_material = sapien.physx.PhysxMaterial(0.3, 0.3, 0.0)
        b = self.scene.create_actor_builder()
        b.add_box_visual(half_size=[h[1], h[0], h[2]],
                         material=sapien.render.RenderMaterial(base_color=[0.97, 0.97, 0.95, 1]))
        b.add_box_collision(half_size=[h[1], h[0], h[2]], material=self.paper_material)
        b.initial_pose = sapien.Pose(p=c - [0, 0, h[2]])
        self.canvas = b.build_kinematic(name="canvas")

        def dots(n, r, color, prefix):
            out = []
            for i in range(n):
                db = self.scene.create_actor_builder()
                db.add_cylinder_visual(radius=r, half_length=self.INK_THICK / 2,
                                       material=sapien.render.RenderMaterial(base_color=color))
                db.initial_pose = sapien.Pose(p=[0, 0, -1.0])
                out.append(db.build_kinematic(name=f"{prefix}_{i}"))
            return out

        self.ink_dots = dots(self.MAX_INK, self.INK_R, [0.05, 0.10, 0.45, 1], "ink")
        self.template_dots = dots(self.MAX_TEMPLATE, self.TEMPLATE_R, [0.72, 0.72, 0.72, 1], "tmpl")

    # ---- per-episode placement, called by sim.py after reset ----
    def place_canvas(self, frame: CanvasFrame, friction: float) -> None:
        h = self.CANVAS_HALF
        q = mat2quat(frame.R @ CANVAS_AXES.T)          # box axes: world-aligned at zero tilt
        centre = frame.origin - h[2] * frame.normal
        self.canvas.set_pose(sapien.Pose(p=centre, q=q))
        self.paper_material.set_static_friction(float(friction))
        self.paper_material.set_dynamic_friction(float(friction))
        self._dot_q = qmult(mat2quat(frame.R), euler2quat(0, np.pi / 2, 0))
        self._frame = frame

    def _dot_pose(self, uv, lift: float) -> sapien.Pose:
        p = self._frame.to_world(np.r_[uv, lift])
        return sapien.Pose(p=p, q=self._dot_q)

    def hide_all_dots(self) -> None:
        hidden = sapien.Pose(p=[0, 0, -1.0])
        for d in self.ink_dots + self.template_dots:
            d.set_pose(hidden)

    def show_template(self, uv_points: np.ndarray) -> int:
        n = min(len(uv_points), self.MAX_TEMPLATE)
        for i in range(n):
            self.template_dots[i].set_pose(self._dot_pose(uv_points[i], self.INK_THICK / 2))
        return n

    def put_ink(self, k: int, uv) -> bool:
        if k >= self.MAX_INK:
            return False
        # a hair above the template so ink always renders over it
        self.ink_dots[k].set_pose(self._dot_pose(uv, self.INK_THICK / 2 + 0.0002))
        return True

    # ---- BaseEnv hooks ----
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

    def evaluate(self):
        # scoring lives in sim.WritingSim.score(); BaseEnv only needs the keys
        z = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        return {"success": z, "fail": z}

    def _get_obs_extra(self, info: dict):
        return dict()

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
