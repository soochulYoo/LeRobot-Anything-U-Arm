"""Scenes for two tasks whose stiff direction TURNS as the work goes: a box to flip up
against a wall, and a door to pull open.  Both are done with a STICK.

`../wiping` is one surface the pad stays on; `../peg` is one axis the peg goes down.  In
both the direction the tool has to be stiff along is fixed for the whole episode.  Here
it is the direction the tool is MOVING, and that turns through most of a right angle:

    FLIP    robot --- box --- wall.  A box lies on the desk with its far end at a wall.
            The stick's ball tip presses on the near end and carries it up and over, the
            box pivoting on the wall, until it stands on its far end against it.  The
            tip PRESSES across its path and MOVES along it: soft push, hard move.
    DOOR    robot --- door.  A door standing on the desk, hinged on a vertical edge,
            that opens AWAY from the arm.  The stick's ball tip comes up to the face,
            near the free edge, and PUSHES it open against a latch and a closer.  The
            place it pushes on travels an arc about the hinge, and nothing holds the
            ball there but friction: stiff along the push, soft across it, and the push
            has turned 50 degrees by the end.

            (Two earlier doors were PULLED: one by a tip that the simulator fastened to
            a strip on top of the door, one by a stick hung behind a vertical bar
            handle.  Both are in this file's history; the task was changed to a push.)

The arm is the Panda the pen and the pad are on, with a ball-tipped stick pointing
DOWN -- the pose the pen writes in, which is why both scenes stand on the desk.  The
scene is stepped directly under the Case 1 controller, as the writing scene is: what
BaseEnv gives is the scene, the agent and the cameras.

Everything an episode varies is set after the build, on objects built once: the box's
place, mass and friction; the door's place, its yaw, its latch and closer.  Nothing is
reconfigured, because reconfiguring closes the operator's window.
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import sapien
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "writing"))

from mani_skill import PACKAGE_ASSET_DIR  # noqa: E402
from mani_skill.agents.registration import register_agent  # noqa: E402
from mani_skill.agents.robots.panda.panda_stick import PandaStick  # noqa: E402
from mani_skill.envs.sapien_env import BaseEnv  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from mani_skill.utils.registration import register_env  # noqa: E402
from mani_skill.utils.structs.types import SceneConfig, SimConfig  # noqa: E402

TABLE_TOP = 0.10        # m, the desk the work stands on: the height the pen writes at
TIP_Z = 0.150           # m down the hand's axis: the ball's far end, where the TCP is
# The door's stick is LONGER.  The ball pushes the face half way up it, and the hand is
# then straight above the ball: with 150 mm of stick the hand's own body would be level
# with the door's top edge, and on it.
TIP_Z_LONG = 0.220
BALL_R = 0.012          # m: a rubber ball on the end of the stick
STEM_R = 0.006


def _stick_urdf(tip_z: float = TIP_Z) -> str:
    """panda_stick.urdf with the flat stick replaced by a ball-tipped one, `tip_z` long.

    Generated rather than shipped, as the pen's and the pad's are: the mesh paths must
    be absolute to load from outside ManiSkill's asset tree.
    """
    src_dir = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda"
    txt = (src_dir / "panda_stick.urdf").read_text()
    txt = txt.replace('filename="franka_description/', f'filename="{src_dir}/franka_description/')
    txt = re.sub(r'\s*<visual>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</visual>', "",
                 txt, count=1, flags=re.S)
    txt = re.sub(r'\s*<collision>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</collision>', "",
                 txt, count=1, flags=re.S)
    lo, hi = 0.05, tip_z - 2 * BALL_R + 0.002
    ball_z = tip_z - BALL_R
    # the TCP rides at the end of the ball, wherever that is
    txt = txt.replace('<origin rpy="0 0 0" xyz="0 0 0.15"/>\n    <parent link="panda_hand"/>\n'
                      '    <child link="panda_hand_tcp"/>',
                      f'<origin rpy="0 0 0" xyz="0 0 {tip_z:.4f}"/>\n    <parent link="panda_hand"/>\n'
                      f'    <child link="panda_hand_tcp"/>')
    assert f'xyz="0 0 {tip_z:.4f}"' in txt, "the TCP joint was not where it was expected"
    stick = f"""
  <link name="stick">
    <visual>
      <origin xyz="0 0 {(lo + hi) / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder radius="{STEM_R}" length="{hi - lo:.4f}"/></geometry>
      <material name="stem"><color rgba="0.85 0.80 0.70 1"/></material>
    </visual>
    <visual>
      <origin xyz="0 0 {ball_z:.4f}" rpy="0 0 0"/>
      <geometry><sphere radius="{BALL_R}"/></geometry>
      <material name="ball"><color rgba="0.15 0.35 0.85 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {(lo + hi - 0.004) / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder radius="{STEM_R}" length="{hi - 0.004 - lo:.4f}"/></geometry>
    </collision>
    <collision>
      <origin xyz="0 0 {ball_z:.4f}" rpy="0 0 0"/>
      <geometry><sphere radius="{BALL_R}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="0.03"/>
      <inertia ixx="2e-5" ixy="0" ixz="0" iyy="2e-5" iyz="0" izz="2e-6"/>
    </inertial>
  </link>
  <joint name="stick_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="panda_hand"/>
    <child link="stick"/>
  </joint>
"""
    return txt.replace("</robot>", stick + "</robot>")


def _write_atomic(path: pathlib.Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _ensure_assets(name: str = "panda_ballstick", tip_z: float = TIP_Z) -> str:
    """The stick's URDF and its SRDF.  The SRDF is not optional: without it the hand
    and link 7 collide with what is bolted to them and the wrist is pinned -- see
    `wiping/wipe_scene._ensure_assets`, where that cost a comparison."""
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    urdf = out / f"{name}.urdf"
    txt = _stick_urdf(tip_z)
    if not urdf.exists() or urdf.read_text() != txt:
        _write_atomic(urdf, txt)
    src = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda" / "panda_stick.srdf"
    srdf = out / f"{name}.srdf"
    s = src.read_text().replace(
        "</robot>", '    <disable_collisions link1="panda_hand" link2="stick" reason="Adjacent"/>\n'
                    '    <disable_collisions link1="panda_link7" link2="stick" reason="Adjacent"/>\n</robot>')
    if not srdf.exists() or srdf.read_text() != s:
        _write_atomic(srdf, s)
    return str(urdf)


@register_agent()
class PandaBallStick(PandaStick):
    """PandaStick with a ball-tipped stick; same 7 joints and TCP convention."""
    uid = "panda_ballstick"
    urdf_path = _ensure_assets()


@register_agent()
class PandaLongStick(PandaStick):
    """The same stick, 70 mm longer: the door's.  See TIP_Z_LONG."""
    uid = "panda_longstick"
    urdf_path = _ensure_assets("panda_longstick", TIP_Z_LONG)


class ContactEnv(BaseEnv):
    """What the two scenes share: the arm, the desk, the cameras, the window."""

    SUPPORTED_ROBOTS = ["panda_ballstick", "panda_longstick"]
    ROBOT = "panda_ballstick"
    agent: PandaBallStick
    LOOK_AT = np.array([0.50, 0.0, 0.17])     # where the work is, for every camera
    DESK_HALF = np.array([0.30, 0.36, TABLE_TOP / 2])
    DESK_AT = np.array([0.58, 0.0, TABLE_TOP / 2])

    def __init__(self, *args, image_size: int = 128, robot_uids=None, **kwargs):
        self.image_size = int(image_size)
        super().__init__(*args, robot_uids=robot_uids or self.ROBOT, **kwargs)

    @property
    def _default_sim_config(self):
        # 500 Hz and the writing scene's solver settings: an impedance loop cannot live
        # at ManiSkill's 100 Hz, and these are the iterations that scene was tuned on.
        return SimConfig(sim_freq=500, control_freq=500,
                         scene_config=SceneConfig(contact_offset=0.002,
                                                  solver_position_iterations=15,
                                                  solver_velocity_iterations=1))

    @property
    def _default_sensor_configs(self):
        c, s = self.LOOK_AT, self.image_size
        # Named as wiping's two streams are, because the recorder, the console's camera
        # tap and a deployed helper read `rgb_top_camera` and `rgb_wrist_camera`.  Both
        # are FIXED in the world.  The second looks along the desk from the side, which
        # is the view in which a box leaning on a wall, or a door part open, can be
        # told from one that is not.
        return [CameraConfig(uid="top_camera",
                             pose=sapien_utils.look_at(c + [-0.02, 0.0, 0.60], c,
                                                       up=(1, 0, 0)),
                             width=s, height=s, fov=0.95, near=0.01, far=10),
                CameraConfig(uid="wrist_camera",
                             pose=sapien_utils.look_at(c + self.SIDE_EYE, c),
                             width=s, height=s, fov=0.80, near=0.01, far=10)]

    @property
    def _default_human_render_camera_configs(self):
        c = self.LOOK_AT
        return CameraConfig(uid="render_camera",
                            pose=sapien_utils.look_at(c + self.RENDER_EYE, c),
                            width=960, height=720, fov=0.8, near=0.01, far=10)

    # The operator stands behind the arm's right shoulder: forward on the screen is
    # forward for the tool, and the hand does not hide what the tip is about to touch.
    VIEWER_EYE = np.array([-0.50, -0.42, 0.36])       # relative to LOOK_AT
    VIEWER_AT = np.array([0.0, 0.0, 0.0])
    SIDE_EYE = np.array([-0.04, -0.60, 0.12])         # the second camera
    RENDER_EYE = np.array([-0.28, -0.62, 0.30])       # the picture a recording is made from

    @property
    def _default_viewer_camera_configs(self):
        c = self.LOOK_AT
        return CameraConfig(uid="viewer",
                            pose=sapien_utils.look_at(c + self.VIEWER_EYE, c + self.VIEWER_AT),
                            width=1280, height=900, fov=0.9, near=0.01, far=100,
                            shader_pack="default")

    def _setup_viewer(self):
        super()._setup_viewer()
        # Through the viewer's own fly-camera state, as `writing/scene.py` learned: a
        # bare set_camera_pose is overridden by it.
        c = self.LOOK_AT
        eye = c + self.VIEWER_EYE
        d = (c + self.VIEWER_AT) - eye
        d = d / np.linalg.norm(d)
        self._viewer.set_camera_xyz(*eye)
        self._viewer.set_camera_rpy(0.0, -np.arcsin(-d[2]), -np.arctan2(d[1], d[0]))
        self._viewer.window.set_camera_parameters(0.01, 100.0, 0.9)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, 0.0]))

    def _load_scene(self, options: dict):
        ground = self.scene.create_actor_builder()
        ground.add_box_visual(half_size=[1.5, 1.5, 0.01],
                              material=sapien.render.RenderMaterial(
                                  base_color=[0.78, 0.78, 0.76, 1]))
        ground.add_box_collision(half_size=[1.5, 1.5, 0.01])
        ground.initial_pose = sapien.Pose(p=[0, 0, -0.01])
        ground.build_static(name="ground")
        # One material for everything that stands still -- the desk, and the wall --
        # kept so an episode can set how easily the work slides on it.
        self.still_material = sapien.physx.PhysxMaterial(0.4, 0.4, 0.0)
        desk = self.scene.create_actor_builder()
        desk.add_box_visual(half_size=self.DESK_HALF,
                            material=sapien.render.RenderMaterial(
                                base_color=[0.76, 0.62, 0.42, 1]))
        desk.add_box_collision(half_size=self.DESK_HALF, material=self.still_material)
        desk.initial_pose = sapien.Pose(p=self.DESK_AT)
        self.desk = desk.build_static(name="desk")
        self._load_task()

    def _load_task(self) -> None:
        raise NotImplementedError

    # ---- BaseEnv hooks: scoring lives in contact_tasks.py, as it does for writing ----
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

    def evaluate(self):
        z = torch.zeros(self.num_envs, device=self.device, dtype=bool)
        return {"success": z, "fail": z}

    def _get_obs_extra(self, info: dict):
        return dict()

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


# --------------------------------------------------------------------------- #
@register_env("TeleopFlip-v1", max_episode_steps=1_000_000)
class FlipEnv(ContactEnv):
    """robot --- box --- wall.  The box lies flat, long way towards the wall."""

    # half extents: along the push (towards the wall), across, and up.  Lying, it is
    # 160 long and 50 high; flipped up, it stands 160 high on a 50 x 80 end.
    BOX_HALF = np.array([0.080, 0.040, 0.025])
    BOX_MASS = 0.4                               # kg, the nominal; an episode rescales it
    WALL_X = 0.64                                # m: the face the box is flipped against
    WALL_HALF = np.array([0.030, 0.14, 0.075])

    def _load_task(self) -> None:
        w = self.scene.create_actor_builder()
        w.add_box_visual(half_size=self.WALL_HALF, material=sapien.render.RenderMaterial(
            base_color=[0.92, 0.92, 0.90, 1]))
        w.add_box_collision(half_size=self.WALL_HALF, material=self.still_material)
        w.initial_pose = sapien.Pose(p=[self.WALL_X + self.WALL_HALF[0], 0.0,
                                        TABLE_TOP + self.WALL_HALF[2]])
        self.wall = w.build_static(name="wall")

        h = self.BOX_HALF
        self.box_material = sapien.physx.PhysxMaterial(0.5, 0.5, 0.0)
        b = self.scene.create_actor_builder()
        b.add_box_visual(half_size=h, material=sapien.render.RenderMaterial(
            base_color=[0.10, 0.30, 0.80, 1]))
        # the near end, where the tip goes, is the dark one: a helper finds it by eye
        b.add_box_visual(pose=sapien.Pose([-h[0] - 0.0005, 0, 0]),
                         half_size=[0.0005, h[1], h[2]],
                         material=sapien.render.RenderMaterial(
                             base_color=[0.12, 0.12, 0.14, 1]))
        b.add_box_collision(half_size=h, material=self.box_material,
                            density=self.BOX_MASS / float(8 * np.prod(h)))
        b.initial_pose = sapien.Pose(p=[self.WALL_X - h[0], 0.0, TABLE_TOP + h[2]])
        self.box = b.build(name="box")


# --------------------------------------------------------------------------- #
@register_env("TeleopDrawer-v1", max_episode_steps=1_000_000)
class DrawerEnv(ContactEnv):
    """robot --- drawers.  A three-tier chest on the desk; the TOP drawer is pulled out.

    THE GRIPPER, NOT THE STICK.  Flipping and the door are done with a ball on a stick,
    which can push and cannot pull.  This task uses the peg's robot -- the plain Panda,
    with its two fingers -- and opens with the handle already GRASPED, as the peg opens
    already gripped.

    SO THE HANDLE IS A RECTANGULAR PILLAR standing straight out of the front, the way a
    real drawer pull does.  The fingers come down from above and close on its two side
    faces, and the drawer comes out on friction -- which is ample: 0.7 against 120 N a
    finger is about 170 N of hold against a pull that peaks near 50.

    (A fixed constraint would hold the handle too, and would read ZERO force -- a drive
    is not a contact.  Measured on the first cut: the wrench went silent, the detent
    never broke and the drawer never moved.)

    THREE TIERS, and the top one is the task, because the other two are what makes
    finding it a thing you have to do by eye: they are the same bar at the same stand-off
    and differ only in height.

    THE STOP IS THE POINT.  The drawer runs on a prismatic joint with a hard limit at
    `TRAVEL`, and a `detent` of static friction to break before it moves at all.  So the
    along axis has to be stiff to get it started and must not still be stiff when it
    reaches the end -- there is no constant that does both, which is what this task is
    here to show.
    """
    SUPPORTED_ROBOTS = ["panda"]
    ROBOT = "panda"                          # the peg's robot: fingers, not a stick
    LOOK_AT = np.array([0.50, 0.0, 0.17])
    # A TABLETOP chest.  The first cut was 150 x 180 x 240 half extents -- a 300 x 360 x
    # 480 mm chest that filled half the desk and put the top drawer 400 mm up, out of
    # the cameras and awkward for the arm.  Three 60 mm tiers on a 200 x 240 footprint
    # sit where the other two tasks' work sits.
    BODY = np.array([0.100, 0.120, 0.090])   # m half extents: deep, wide, tall
    FRONT_AT = 0.56                          # m: x of the closed drawer fronts
    N_TIERS = 3
    TRAVEL = 0.110                           # m: the hard stop, fully out
    DRAWER_MASS = 0.6                        # kg
    FRONT_T = 0.014                          # m: the drawer front's thickness
    # THE PILLAR.  Half extents: how far it stands out of the front, half its width
    # (what the fingers close on, against the Panda's 80 mm of opening), and half its
    # height.  Nothing stands behind it and nothing needs to: the grip is friction.
    PILLAR = np.array([0.026, 0.009, 0.013])

    def bar_dx(self) -> float:
        """x of the pillar's MID-LENGTH, relative to the closed front's outer face --
        where the fingers close.

        The task grasps with this and `_load_task` builds with it, so the two cannot
        drift apart.  They did once, by one FRONT_T, and the tool started the episode
        inside the handle: 10.8 kN on the first step.
        """
        return -self.PILLAR[0]

    def _tier_z(self, i: int) -> float:
        """Centre height of tier i, 0 = top."""
        h = 2 * self.BODY[2] / self.N_TIERS
        return TABLE_TOP + 2 * self.BODY[2] - (i + 0.5) * h

    def _load_task(self) -> None:
        d, w, _ = self.BODY
        self.front_material = sapien.physx.PhysxMaterial(0.7, 0.7, 0.0)
        # The pillar is GRIPPY -- a drawer pull is wood or rubber in a hand, not steel
        # on steel, and the grasp here is friction along the pull.  The peg's ball uses
        # 1.5 for the same reason.
        self.grip_material = sapien.physx.PhysxMaterial(1.2, 1.2, 0.0)
        builder = self.scene.create_articulation_builder()

        # THE CARCASS, seen and felt: the drawers come out of it and the arm can hit it
        case = builder.create_link_builder(parent=None)
        case.set_name("case")
        case.add_box_visual(half_size=self.BODY.tolist(),
                            material=sapien.render.RenderMaterial(
                                base_color=[0.38, 0.33, 0.29, 1]))
        case.add_box_collision(half_size=self.BODY.tolist(), material=self.still_material)

        hh = self.BODY[2] / self.N_TIERS          # half the height of one tier
        self.drawers, self.fronts = [], []
        for i in range(self.N_TIERS):
            dz = self._tier_z(i) - (TABLE_TOP + self.BODY[2])    # relative to the carcass
            lk = builder.create_link_builder(case)
            lk.set_name(f"drawer{i}")
            # the FRONT, the only part the ball ever meets
            fx = d + self.FRONT_T / 2
            colour = [0.80, 0.56, 0.32, 1] if i else [0.86, 0.64, 0.38, 1]
            lk.add_box_visual(pose=sapien.Pose([-fx, 0, dz]),
                              half_size=[self.FRONT_T / 2, w * 0.96, hh * 0.92],
                              material=sapien.render.RenderMaterial(base_color=colour))
            lk.add_box_collision(pose=sapien.Pose([-fx, 0, dz]),
                                 half_size=[self.FRONT_T / 2, w * 0.96, hh * 0.92],
                                 material=self.front_material,
                                 density=self.DRAWER_MASS / (2 * self.FRONT_T * w * hh))
            # THE BOX behind the front, so a drawer that is open looks like a drawer
            # and not a floating panel.  VISUAL ONLY: the carcass is one solid collider
            # and a drawer body with collision would be inside it, so the two would
            # fight on the first step.  Nothing has to touch the inside of a drawer --
            # the fingers only ever meet the pillar.
            inner_t = 0.004
            back_x, front_x = d - 0.006, -fx + self.FRONT_T / 2
            cx, cdx = 0.5 * (back_x + front_x), 0.5 * (back_x - front_x)
            iw, ih = w * 0.93, hh * 0.78
            box = sapien.render.RenderMaterial(base_color=[0.62, 0.45, 0.28, 1],
                                               roughness=0.85)
            for pos, half in (
                ([cx, 0.0, dz - ih + inner_t], [cdx, iw, inner_t]),          # bottom
                ([cx, iw - inner_t, dz], [cdx, inner_t, ih]),                # left
                ([cx, -(iw - inner_t), dz], [cdx, inner_t, ih]),             # right
                ([back_x - inner_t, 0.0, dz], [inner_t, iw, ih]),            # back
            ):
                lk.add_box_visual(pose=sapien.Pose(pos), half_size=half, material=box)

            # THE PILLAR: one box out of the middle of the front.
            wood = sapien.render.RenderMaterial(base_color=[0.78, 0.62, 0.40, 1],
                                                roughness=0.7)
            bx = self.bar_dx() - (d + self.FRONT_T)
            lk.add_box_visual(pose=sapien.Pose([bx, 0.0, dz]),
                              half_size=self.PILLAR.tolist(), material=wood)
            lk.add_box_collision(pose=sapien.Pose([bx, 0.0, dz]),
                                 half_size=self.PILLAR.tolist(),
                                 material=self.grip_material, density=500.0)
            # a prismatic joint slides along its own x, which here is world -x: out,
            # towards the arm.  The limit IS the stop this task is about.
            lk.set_joint_name(f"slide{i}")
            q = [0.0, 0.0, 0.0, 1.0]        # x -> -x
            lk.set_joint_properties(type="prismatic", limits=[[0.0, self.TRAVEL]],
                                    pose_in_parent=sapien.Pose([0, 0, 0], q),
                                    pose_in_child=sapien.Pose([0, 0, 0], q),
                                    friction=0.0, damping=0.0)
        builder.initial_pose = sapien.Pose(
            p=[self.FRONT_AT + d + self.FRONT_T, 0.0, TABLE_TOP + self.BODY[2]])
        self.chest = builder.build("chest", fix_root_link=True)
        for i in range(self.N_TIERS):
            self.fronts.append(sapien_utils.get_obj_by_name(self.chest.get_links(),
                                                            f"drawer{i}"))
        self.slides = list(self.chest.get_active_joints())
        self.slide = self.slides[0]            # the TOP drawer: the one that is the task


# --------------------------------------------------------------------------- #
@register_env("TeleopDoor-v1", max_episode_steps=1_000_000)
class DoorEnv(ContactEnv):
    """robot --- door.  A door on the desk that is PUSHED open, away from the arm.

    The link frame of the panel sits ON the hinge line, at the panel's mid height, with
    the closed panel running along -y and the arm on its -x side.  The joint turns it
    about +z, so a positive angle swings the free edge AWAY from the arm.

    Nothing is fastened to the stick and there is no handle: the ball presses on the
    face and friction is all that keeps it on the spot it started on.  A PUSH PLATE is
    painted where the hand is meant to push -- seen, not felt -- because a helper has to
    find the place by eye, and how far along it the ball lands is the episode's.
    """
    ROBOT = "panda_longstick"
    LOOK_AT = np.array([0.56, 0.0, 0.20])
    PANEL_W, PANEL_H, PANEL_T = 0.32, 0.24, 0.016      # m: width, height, thickness
    PANEL_MASS = 1.0
    PLATE_R = (0.13, 0.27)                             # m from the hinge: where to push
    PLATE_HALF_H = 0.045
    HINGE_AT = np.array([0.50, 0.16, TABLE_TOP + 0.003 + 0.24 / 2])   # an episode moves it
    ANGLE_MAX = 1.75                                   # rad: the stop

    def _load_task(self) -> None:
        w, h, t = self.PANEL_W, self.PANEL_H, self.PANEL_T
        # the face the ball pushes on: it has to carry the ball round by friction
        self.face_material = sapien.physx.PhysxMaterial(0.6, 0.6, 0.0)
        builder = self.scene.create_articulation_builder()

        post = builder.create_link_builder(parent=None)
        post.set_name("post")
        # seen, not felt: a post the arm could snag on would be a second task
        post.add_box_visual(half_size=[0.010, 0.010, h / 2 + 0.012],
                            material=sapien.render.RenderMaterial(
                                base_color=[0.30, 0.30, 0.32, 1]))

        panel = builder.create_link_builder(post)
        panel.set_name("panel")
        panel.add_box_visual(pose=sapien.Pose([0, -w / 2, 0]), half_size=[t / 2, w / 2, h / 2],
                             material=sapien.render.RenderMaterial(
                                 base_color=[0.72, 0.50, 0.30, 1]))
        panel.add_box_collision(pose=sapien.Pose([0, -w / 2, 0]),
                                half_size=[t / 2, w / 2, h / 2],
                                material=self.face_material,
                                density=self.PANEL_MASS / (w * h * t))
        r0, r1 = self.PLATE_R
        panel.add_box_visual(pose=sapien.Pose([-(t / 2 + 0.0005), -(r0 + r1) / 2, 0.0]),
                             half_size=[0.0005, (r1 - r0) / 2, self.PLATE_HALF_H],
                             material=sapien.render.RenderMaterial(
                                 base_color=[0.75, 0.76, 0.78, 1], metallic=0.5,
                                 roughness=0.4))
        # SAPIEN turns a revolute joint about its own x: this carries x onto world +z.
        q = [0.70710678, 0.0, -0.70710678, 0.0]
        panel.set_joint_name("hinge")
        panel.set_joint_properties(type="revolute", limits=[[0.0, self.ANGLE_MAX]],
                                   pose_in_parent=sapien.Pose([0, 0, 0], q),
                                   pose_in_child=sapien.Pose([0, 0, 0], q),
                                   friction=0.0, damping=0.0)
        builder.initial_pose = sapien.Pose(p=self.HINGE_AT)
        self.door = builder.build("door", fix_root_link=True)
        self.panel = sapien_utils.get_obj_by_name(self.door.get_links(), "panel")
        self.hinge = self.door.get_active_joints()[0]
