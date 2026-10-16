"""Insertions where the stiffness has to CHANGE during the task.

    snap      a buckle pushed DOWN into its housing                   K_n: stiff -> soft
    plug      a two-blade plug pushed FORWARD into a wall socket      K_t: stiff -> soft
    wallsnap  a snap-fit's male half pushed into its catch in a wall  K_t and K_n, both

Both are built so that NO CONSTANT STIFFNESS DOES THE TASK, with the position-error cap
(`ContactSim.E_MAX`) as what makes that so: with the reference held within E_MAX of the
tool, an operator cannot reach further to make up for a soft arm, and the most force the
arm can put out along an axis is K x E_MAX.

SNAP.  The buckle rides down a rail into its housing.  The prongs resist with a force
that rises with the stroke to `f_click`; past `x_click` they snap through, the buckle
runs free for a few millimetres onto its stop and is latched.  At the 20 mm cap, along
makes at most 6 / 20 / 60 N at low / mid / high, and a click needs 30-40 N: only HIGH
clicks.  Once it has, HIGH goes on pressing the latched parts together with up to 60 N
-- into a part that is done.  So: stiff to the click, soft after it.

PLUG.  The plug drags a cord the controller knows nothing about (`cable`, N down).  It is
carried level and pushed FORWARD into a socket on a wall ahead of the arm, so the
vertical is ACROSS the push, and K_t is what holds it up.
At the 6 mm cap K_t can hold at most 1.8 / 4.8 / 12 N at low / mid / high against 5-7 N
of cord: below HIGH the plug falls.  But the socket is not quite where it was believed
to be: going in, the funnel has to slide the blades 2.5-3.5 mm along the wall, and a stiff
K_t pushes back on exactly that -- K_t x (offset - clearance) of side load on the
blades, for as long as they are in the hole.  So: stiff to carry, soft once in the mouth.
(That last needs a FIRM GRIP on the operator's side -- see `PlugSim.VR_MAP`.)

Registered into `contact_tasks.SIMS` on import, as `wok_probe` registers the wok, so the
stick collector (`protocol.py --task snap|plug`) runs them.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contact_tasks as CT  # noqa: E402  sets the path to writing/ and registers its envs
import contact_scene as SC  # noqa: E402
import sapien  # noqa: E402
import torch  # noqa: E402
from contact_tasks import UP, ContactCriteria, ContactSim, rot_z  # noqa: E402
from mani_skill import PACKAGE_ASSET_DIR  # noqa: E402
from mani_skill.agents.registration import register_agent  # noqa: E402
from mani_skill.agents.robots.panda.panda_stick import PandaStick  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from mani_skill.utils.registration import register_env  # noqa: E402
from transforms3d.quaternions import mat2quat  # noqa: E402

Array = np.ndarray
FWD = np.array([1.0, 0.0, 0.0])        # the plug's push: at the wall ahead
ALONG_WALL = np.array([0.0, 1.0, 0.0])  # across the push and across gravity
_QY90 = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]    # x -> -z
# Neither task has a landing or a turn to judge, and both push harder than a box press
# by design: the stick tasks' 20 N across / 45 N / 4 Nm limits would fail every click.
_NO_STICK_LIMITS = staticmethod(lambda: ContactCriteria(across_force=1e9, overload=1e9,
                                                        moment=1e9))


# =========================================================================== SNAP
@register_env("TeleopSnap-v1", max_episode_steps=1_000_000)
class SnapEnv(SC.ContactEnv):
    """robot --- buckle --- housing.  The housing stands on the desk, mouth up; the
    buckle's male half sits in its mouth on a vertical rail, flat top up, and the ball
    comes down on that top."""

    LOOK_AT = np.array([0.52, 0.0, 0.17])
    HOUSING = np.array([0.032, 0.026, 0.030])    # m half extents: the female half
    PLATE = np.array([0.022, 0.020, 0.004])      # m: the male half's top, pushed on
    STROKE_MAX = 0.026                           # m: longer than any episode's stroke
    MALE_MASS = 0.03                             # kg
    # THE LATCH, DRAWN ONLY: its physics is the rail's drive (see SnapSim).  Two prongs
    # hang from the plate with a hook at each tip, and two catches stand in the housing.
    # The hooks overlap the catches by their whole width, so the prongs have to BEND in
    # to ride over them and spring straight under them at the click -- which is what a
    # snap-fit looks like, exaggerated to read at video size.
    PRONG = np.array([0.004, 0.003, 0.012])      # half extents
    PRONG_Y = 0.013                              # m: each prong's centre from the middle
    HOOK = np.array([0.004, 0.0015, 0.002])      # half: out 3 mm from the prong's face
    LEDGE = np.array([0.005, 0.0025, 0.0015])    # half: a catch
    LEDGE_Y = PRONG_Y + PRONG[1] + LEDGE[1]      # its inner face at the prong's outer one
    BEND = 2 * HOOK[1] + 0.0002                  # m the tip moves in to clear a catch
    LEAD = 0.0015                                # m of stroke the hook takes to ride on

    def root_z(self, stroke: float) -> float:
        """Height of the prongs' roots (the plate's underside) above the housing's
        centre, at `stroke`."""
        return float(self.HOUSING[2] + self.STROKE_MAX - stroke)

    def hook_top(self, stroke: float) -> float:
        """Height of a hook's top above the housing's centre, at `stroke`."""
        return self.root_z(stroke) - 2 * self.PRONG[2] + 2 * self.HOOK[2]

    def place_catches(self, centre, R, x_click: float) -> None:
        """Put the catches in the housing (centred at `centre`, turned by `R`) where the
        hooks pass under them at `x_click`."""
        self._frame = (np.asarray(centre, float), np.asarray(R, float), float(x_click))
        z = self.hook_top(x_click) + self.LEDGE[2]
        self.catches.set_pose(sapien.Pose(p=(self._frame[0] + R @ [0, 0, z]).tolist(),
                                          q=mat2quat(R)))

    def draw_latch(self, stroke: float, clicked: bool) -> None:
        """Pose the two prongs at `stroke`: straight, or bent in while a hook is riding
        over its catch.  From the click on they are straight, the hooks under."""
        centre, R, x_click = self._frame
        bend = 0.0
        if not clicked:
            on = (self.hook_top(x_click) + 2 * self.LEDGE[2]) - (self.hook_top(stroke)
                                                                 - 2 * self.HOOK[2])
            bend = self.BEND * float(np.clip(on / self.LEAD, 0.0, 1.0))
        th = float(np.arcsin(bend / (2 * self.PRONG[2])))
        for sy, actor in zip((-1, 1), self.prongs):
            a = -sy * th                                     # tip in, towards the middle
            Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
            p = centre + R @ [0.0, sy * self.PRONG_Y, self.root_z(stroke)]
            actor.set_pose(sapien.Pose(p=p.tolist(), q=mat2quat(R @ Rx)))

    def _load_task(self) -> None:
        b = self.scene.create_articulation_builder()
        root = b.create_link_builder(parent=None)
        root.set_name("housing")
        dark = sapien.render.RenderMaterial(base_color=[0.10, 0.10, 0.11, 1], roughness=0.6)
        root.add_box_visual(half_size=self.HOUSING.tolist(), material=dark)
        root.add_box_collision(half_size=self.HOUSING.tolist(), material=self.still_material)
        # the mouth, so a person can see where the prongs go
        root.add_box_visual(pose=sapien.Pose([0, 0, self.HOUSING[2] + 0.0004]),
                            half_size=[self.PLATE[0] * 0.8, self.PLATE[1] * 0.5, 0.0005],
                            material=sapien.render.RenderMaterial(
                                base_color=[0.02, 0.02, 0.02, 1]))
        # THE CATCHES the prongs' hooks snap under, inside the housing: seen, not felt
        # (the click is the rail's drive letting go -- see SnapSim).  Only visible when
        # the housing is drawn see-through, as the presentation video draws it.  An actor
        # of their own, because a shape cannot be moved once attached and they have to
        # be where the hooks pass them AT the episode's click: `place_catches`.
        amber = sapien.render.RenderMaterial(base_color=[0.95, 0.55, 0.10, 1])
        cb = self.scene.create_actor_builder()
        for sy in (-1, 1):
            cb.add_box_visual(pose=sapien.Pose([0, sy * self.LEDGE_Y, 0.0]),
                              half_size=self.LEDGE.tolist(), material=amber)
        cb.initial_pose = sapien.Pose(p=[0.52, 0.0, SC.TABLE_TOP + self.HOUSING[2]])
        self.catches = cb.build_kinematic(name="catches")
        male = b.create_link_builder(root)
        male.set_name("male")
        grey = sapien.render.RenderMaterial(base_color=[0.55, 0.55, 0.58, 1], roughness=0.5)
        # THE TOP, where the ball pushes.  At zero stroke its underside stands
        # STROKE_MAX above the housing, so even the longest stroke stops short of it:
        # the stop is the rail's limit, never the two parts meeting.
        z0 = self.HOUSING[2] + self.STROKE_MAX + self.PLATE[2]
        male.add_box_visual(pose=sapien.Pose([0, 0, z0]), half_size=self.PLATE.tolist(),
                            material=grey)
        male.add_box_collision(pose=sapien.Pose([0, 0, z0]), half_size=self.PLATE.tolist(),
                               material=sapien.physx.PhysxMaterial(0.8, 0.8, 0.0),
                               density=self.MALE_MASS / float(8 * np.prod(self.PLATE)))
        # the centre bar, down into the housing: seen, not felt
        male.add_box_visual(pose=sapien.Pose([0, 0, z0 - self.PLATE[2] - 0.010]),
                            half_size=[0.005, 0.004, 0.010], material=grey)
        # THE PRONGS, each an actor of its own so it can bend (`draw_latch`): framed at
        # its root, hanging down, its hook out towards its catch
        self.prongs = []
        for sy in (-1, 1):
            pb = self.scene.create_actor_builder()
            pb.add_box_visual(pose=sapien.Pose([0, 0, -self.PRONG[2]]),
                              half_size=self.PRONG.tolist(), material=grey)
            pb.add_box_visual(pose=sapien.Pose([0, sy * (self.PRONG[1] + self.HOOK[1]),
                                                -2 * self.PRONG[2] + self.HOOK[2]]),
                              half_size=self.HOOK.tolist(), material=grey)
            pb.initial_pose = sapien.Pose(p=[0.52, sy * self.PRONG_Y, 0.2])
            self.prongs.append(pb.build_kinematic(name=f"prong{'LR'[(sy + 1) // 2]}"))
        male.set_joint_name("insert")
        # a prismatic joint slides along its own x, here turned to world -z: down
        male.set_joint_properties(type="prismatic", limits=[[0.0, self.STROKE_MAX]],
                                  pose_in_parent=sapien.Pose([0, 0, 0], _QY90),
                                  pose_in_child=sapien.Pose([0, 0, 0], _QY90),
                                  friction=0.0, damping=0.0)
        b.initial_pose = sapien.Pose(p=[0.52, 0.0, SC.TABLE_TOP + self.HOUSING[2]])
        self.buckle = b.build("buckle", fix_root_link=True)
        links = self.buckle.get_links()
        self.housing = sapien_utils.get_obj_by_name(links, "housing")
        self.male = sapien_utils.get_obj_by_name(links, "male")
        self.rail = self.buckle.get_active_joints()[0]

    def top_z(self) -> float:
        """World z of the male half's top face at zero stroke."""
        return float(SC.TABLE_TOP + 2 * self.HOUSING[2] + self.STROKE_MAX
                     + 2 * self.PLATE[2])


@dataclasses.dataclass
class SnapSpec:
    seed: int = 0
    xy: tuple = (0.52, 0.0)          # m: where the housing stands
    yaw: float = 0.0
    f_click: float = 35.0            # N the prongs resist with at the click
    x_click: float = 0.016           # m of stroke to the click
    free: float = 0.006              # m from the click to the stop
    drag: float = 20.0               # N s/m along the rail
    belief_err: float = 0.0          # m up: told minus true, so WHEN the top is met
    standoff: float = 0.040          # m above the told top, where the ball starts
    time_limit: float = 20.0

    @staticmethod
    def sample(seed: int) -> "SnapSpec":
        r = np.random.default_rng(seed)
        return SnapSpec(seed=seed, xy=(float(r.uniform(0.48, 0.56)),
                                       float(r.uniform(-0.06, 0.06))),
                        yaw=float(np.radians(r.uniform(-15.0, 15.0))),
                        f_click=float(r.uniform(30.0, 40.0)),
                        x_click=float(r.uniform(0.013, 0.018)),
                        free=float(r.uniform(0.004, 0.007)),
                        drag=float(r.uniform(10.0, 30.0)),
                        belief_err=float(r.uniform(-0.010, 0.010)),
                        standoff=float(r.uniform(0.030, 0.060)))


class SnapSim(ContactSim):
    """Push the buckle down until it clicks; then stop pressing on it.

    THE CLICK IS A SPRING THAT LETS GO.  The rail's drive resists with
    `f_click / x_click` per metre of stroke -- the prongs bending -- and the moment the
    stroke passes `x_click` the drive goes slack: the prongs have snapped through.  The
    buckle runs `free` onto its stop and the rail's lower limit moves up behind it:
    latched, it cannot come back out.  The hand feels the push rise and then fall away
    under it, which is the cue a person times the change to.

    WHAT FAILS.  Not clicking (`clicked`), and pressing on it afterwards: the sustained
    force on the latched buckle, from `HOLD_GRACE` after the click to the end, must stay
    under `HOLD_MAX` (`gentle`).  The grace is a person's reaction and the stiffness
    ramp, so a change that comes when it should passes and one a second late does not.
    """
    ENV_ID = "TeleopSnap-v1"
    TASK = "snap"
    TOOL_LINK = "stick"
    LANDS = False                # the push is the task; the touch is not judged
    WORK_ON = "touch"            # pushing begins the moment the ball is on the top
    FIXED_FRAME = True           # along IS the push, for the whole episode
    SEATS = True                 # the protocol's `stop` row comes in at the click
    START_ON_TABLE = True        # a scripted user starts on the table's first row
    START_KR = 0.3
    E_MAX = 0.020                # m: along makes at most 6 / 20 / 60 N at the cap
    # m/s the reference may pull away from the tool (controller 1c): without it a
    # trigger squeezed on the buckle punched it through at MID (6/6 clicks).  0.03, not
    # the plug's 0.1: a buckle YIELDS under the push, so the feed-forward of a reference
    # creeping on toward the limit is never cut as "blocked", and D x 0.1 m/s is ~9 N at
    # MID -- 20 + 9 clicked it 6/6.  D x 0.03 is under 3 N.
    E_RATE = 0.03
    # N: THE FORCE CEILING (controller 1c): the reach is min(20 mm, 45 N / K) per axis, so
    # 6 / 20 / 45 N at LOW / MID / HIGH.  A little over the 30-40 N click, so HIGH still
    # clicks; and HIGH left on after the click presses 45 N, not 60 -- still a failure
    # against HOLD_MAX, now without the overload.
    F_SAFE = 45.0
    PRESS_FORCE = 45.0           # N, a full trigger in VR: the click needs 30-40
    # s after the click before the press is judged: a reaction (0.2-0.45 s) and the
    # 0.15 s ramp down from HIGH (three time constants), so on time passes
    HOLD_GRACE = 1.0
    HOLD_MAX = 12.0              # N, sustained, on the latched buckle
    # s judged after the grace; the task then ends the episode itself (`finished`).  A
    # person who ended it sooner -- DONE straight after the click -- was never judged on
    # the press at all, so ending early is a failure of its own (`held`).
    HOLD_S = 1.0
    # FOR A PERSON (2026-10-16): K_n's MID is no use here -- 20 N cannot click and is
    # over HOLD_MAX after -- so one stick push crosses it; softening takes 0.05 s, not
    # 0.15; and the click is buzzed and flashed (`cue`).  Together ~0.9 s to react
    # before the press is judged, against ~0.5 with two pushes and the slow ramp.
    TWO_LEVEL = (1,)
    RAMP_DOWN = 0.05
    APPROACH = -UP
    # outward normal of the work as the tip comes at it: up.  The trigger pushes down.
    HAND_AXES = np.eye(3)
    CRITERIA = _NO_STICK_LIMITS
    ROW_EXTRA = ("clicked", "click_s", "hold_peak", "touch_s")   # into attempts.jsonl

    def _work(self):
        return self.u.male

    def stroke(self) -> float:
        return float(self.u.buckle.get_qpos()[0, 0])

    def reset(self, spec: SnapSpec) -> dict:
        self.spec = spec
        self.env.reset(seed=spec.seed)
        u = self.u
        centre = [spec.xy[0], spec.xy[1], SC.TABLE_TOP + u.HOUSING[2]]
        u.buckle.set_pose(sapien.Pose(p=centre, q=mat2quat(rot_z(spec.yaw))))
        u.buckle.set_qpos(torch.zeros((1, 1)))
        u.buckle.set_qvel(torch.zeros((1, 1)))
        # THE PRONGS as a drive, as the drawer's detent is: the joint's `friction` in
        # this engine is a coefficient on the load through it, not a force
        u.rail.set_friction(0.0)
        u.rail.set_drive_properties(float(spec.f_click / spec.x_click), float(spec.drag),
                                    1e6, "force")
        u.rail.set_drive_target(0.0)
        self._limit(0.0)
        u.place_catches(centre, rot_z(spec.yaw), spec.x_click)
        u.draw_latch(0.0, False)
        self.tip_on = np.array([spec.xy[0], spec.xy[1], u.top_z()])   # the ball's bottom
        self.told = self.tip_on + spec.belief_err * UP
        self._begin(self.told + spec.standoff * UP, -UP)
        self.clicked_t = None
        self._cued = False
        self.hold_peak = 0.0
        self.max_x = 0.0
        return self.observe()

    def _limit(self, lo: float) -> None:
        s = self.spec
        self.u.rail.set_limits(np.array([[float(lo), float(s.x_click + s.free)]],
                                        dtype=np.float32))

    def clicked(self) -> bool:
        return getattr(self, "clicked_t", None) is not None

    def seated(self) -> bool:
        """The protocol's `stop`: from the click on."""
        return self.clicked()

    def _over(self) -> bool:
        return self.clicked()

    def held(self) -> bool:
        return self.clicked() and self.t - self.clicked_t >= self.HOLD_GRACE + self.HOLD_S - 1e-6

    def cue(self) -> str | None:
        if self.clicked() and not self._cued:
            self._cued = True
            return "click"
        return None

    def finished(self) -> bool:
        return self.held()

    def _force(self, rec) -> Array:
        return self.u.scene.get_pairwise_contact_forces(self.tool, self.u.male)[0].cpu().numpy()

    def _task_step(self, rec, touching: bool) -> None:
        x = self.stroke()
        self.max_x = max(self.max_x, x)
        if self.clicked_t is None and x >= self.spec.x_click:
            self.clicked_t = self.t
            self.u.rail.set_drive_properties(0.0, float(self.spec.drag), 1e6, "force")
            self._limit(self.spec.x_click)               # latched
        self.engaged = self.clicked()
        self.u.draw_latch(x, self.clicked())
        if self.clicked() and self.t - self.clicked_t >= self.HOLD_GRACE:
            self.hold_peak = max(self.hold_peak, self.f_slow)

    def _task_fields(self) -> dict:
        if not hasattr(self, "spec"):
            return dict(progress=0.0, stroke=0.0, clicked=False)
        x = self.stroke()
        full = self.spec.x_click + self.spec.free
        return dict(progress=float(np.clip(x / full, 0.0, 1.0)), stroke=x,
                    clicked=self.clicked())

    def privileged(self) -> dict:
        s = self.spec
        return {"xy": np.asarray(s.xy), "yaw": s.yaw, "f_click": s.f_click,
                "x_click": s.x_click, "free": s.free, "drag": s.drag,
                "belief_err": s.belief_err}

    def score(self) -> dict:
        res, checks = self._common_score()
        full = self.spec.x_click + self.spec.free
        res.update(clicked=self.clicked(), click_s=self.clicked_t,
                   touch_s=self.t_touch,             # touch -> click: how long the push took
                   hold_peak=float(self.hold_peak), stroke_m=float(self.stroke()),
                   progress=float(np.clip(self.max_x / full, 0.0, 1.0)))
        checks = {"clicked": self.clicked(), "held": self.held(),
                  "gentle": self.hold_peak <= self.HOLD_MAX,
                  "no_collision": checks["no_collision"]}
        return self._finish(res, checks)


# =========================================================================== PLUG
# The plug, in the HAND's frame (z along the hand's axis, out of the flange).  Held with
# the hand's x UP (the drawer's grasp), so: blades side by side along y and thin along
# it, tall along x, long along z -- a US plug, its slots upright and side by side.
BLADE = np.array([0.0015, 0.0040, 0.0110])   # half extents
BLADE_GAP = 0.009                             # m: each blade's centre from the middle
BODY = np.array([0.018, 0.016, 0.020])        # half extents of the plug's body
BODY_AT = 0.050                               # m out of the flange: clear of the hand
TIP_PLUG = BODY_AT + 2 * BODY[2] + 2 * BLADE[2]   # the TCP: the blades' tips
CORD_LEN = 0.14                               # m of cord drawn hanging below the plug
DRAW_CORD = False                             # no cord since 2026-10-15


def _plug_urdf(blade=BLADE) -> str:
    """panda_stick.urdf with the stick replaced by a plug, the TCP at the blades' tips,
    between them.  `blade`: the blades' half extents (the wall snap's are longer)."""
    BLADE = np.asarray(blade, float)
    TIP_PLUG = BODY_AT + 2 * BODY[2] + 2 * BLADE[2]
    src_dir = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda"
    txt = (src_dir / "panda_stick.urdf").read_text()
    txt = txt.replace('filename="franka_description/', f'filename="{src_dir}/franka_description/')
    txt = re.sub(r'\s*<visual>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</visual>', "",
                 txt, count=1, flags=re.S)
    txt = re.sub(r'\s*<collision>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</collision>', "",
                 txt, count=1, flags=re.S)
    txt = txt.replace('<origin rpy="0 0 0" xyz="0 0 0.15"/>\n    <parent link="panda_hand"/>\n'
                      '    <child link="panda_hand_tcp"/>',
                      f'<origin rpy="0 0 0" xyz="0 0 {TIP_PLUG:.4f}"/>\n    <parent link="panda_hand"/>\n'
                      f'    <child link="panda_hand_tcp"/>')
    assert f'xyz="0 0 {TIP_PLUG:.4f}"' in txt, "the TCP joint was not where it was expected"
    bz = BODY_AT + BODY[2]
    tz = BODY_AT + 2 * BODY[2] + BLADE[2]
    blades = "".join(f"""
    <visual><origin xyz="0 {s * BLADE_GAP:.4f} {tz:.4f}" rpy="0 0 0"/>
      <geometry><box size="{2 * BLADE[1]:.4f} {2 * BLADE[0]:.4f} {2 * BLADE[2]:.4f}"/></geometry>
      <material name="brass"><color rgba="0.80 0.65 0.30 1"/></material></visual>
    <collision><origin xyz="0 {s * BLADE_GAP:.4f} {tz:.4f}" rpy="0 0 0"/>
      <geometry><box size="{2 * BLADE[1]:.4f} {2 * BLADE[0]:.4f} {2 * BLADE[2]:.4f}"/></geometry></collision>"""
                     for s in (-1, 1))
    # THE CORD, drawn only: its pull is `PlugSim.step`'s force, not this.  Down out of
    # the body's underside -- the hand's -x is down as the plug is held -- so a person
    # can see what K_t is holding up.
    cord = f"""
    <visual><origin xyz="{-(BODY[0] + CORD_LEN / 2):.4f} 0 {BODY_AT + 0.008:.4f}" rpy="0 1.5708 0"/>
      <geometry><cylinder radius="0.0035" length="{CORD_LEN:.4f}"/></geometry>
      <material name="cord"><color rgba="0.12 0.12 0.13 1"/></material></visual>"""
    plug = f"""
  <link name="plug">
    <visual><origin xyz="0 0 {bz:.4f}" rpy="0 0 0"/>
      <geometry><box size="{2 * BODY[0]:.4f} {2 * BODY[1]:.4f} {2 * BODY[2]:.4f}"/></geometry>
      <material name="plugbody"><color rgba="0.08 0.08 0.09 1"/></material></visual>
    <collision><origin xyz="0 0 {bz:.4f}" rpy="0 0 0"/>
      <geometry><box size="{2 * BODY[0]:.4f} {2 * BODY[1]:.4f} {2 * BODY[2]:.4f}"/></geometry></collision>{blades}{cord if DRAW_CORD else ""}
    <inertial><origin xyz="0 0 {bz:.4f}" rpy="0 0 0"/><mass value="0.08"/>
      <inertia ixx="3e-5" ixy="0" ixz="0" iyy="3e-5" iyz="0" izz="1e-5"/></inertial>
  </link>
  <joint name="plug_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/><parent link="panda_hand"/><child link="plug"/>
  </joint>
"""
    return txt.replace("</robot>", plug + "</robot>")


def _plug_assets(name: str = "panda_plug", blade=BLADE) -> str:
    """The plug's URDF and SRDF, as `contact_scene._ensure_assets` writes the stick's."""
    out = SC.HERE / "assets"
    out.mkdir(exist_ok=True)
    urdf = out / f"{name}.urdf"
    txt = _plug_urdf(blade)
    if not urdf.exists() or urdf.read_text() != txt:
        SC._write_atomic(urdf, txt)
    src = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda" / "panda_stick.srdf"
    srdf = out / f"{name}.srdf"
    s = src.read_text().replace(
        "</robot>", '    <disable_collisions link1="panda_hand" link2="plug" reason="Adjacent"/>\n'
                    '    <disable_collisions link1="panda_link7" link2="plug" reason="Adjacent"/>\n</robot>')
    if not srdf.exists() or srdf.read_text() != s:
        SC._write_atomic(srdf, s)
    return str(urdf)


@register_agent()
class PandaPlug(PandaStick):
    """PandaStick holding a plug instead of a stick; same 7 joints and TCP convention."""
    uid = "panda_plug"
    urdf_path = _plug_assets()


@register_env("TeleopPlug-v1", max_episode_steps=1_000_000)
class PlugEnv(SC.ContactEnv):
    """robot --- plug --- wall.  The wall stands IN FRONT of the arm, across the desk,
    with a two-hole socket in it facing the arm; the plug is held level, blades pointing
    forward at it -- so pushing it in is moving the hand forward, as in a room.

    (It stood to the arm's LEFT first, and a VR operator pushed the plug in by moving
    their hand sideways: the controller and the plug did not move alike (2026-10-14).
    In front it has to be HIGH -- the socket at 0.50-0.54 m: holding a tool level and
    pointing forward, this Panda's fifth joint runs out of range between 0.22 and 0.42 m,
    and is clear of its stops by 0.2 rad or more above 0.50 across the whole carry.)

    THE SOCKET, in its own frame: x along the wall, y into it, z up; its face at y = 0.
    The actor is turned so its y is the world's +x (`Q_FRONT`).
    Two layers.  The FRONT `CHAMFER` deep is a funnel round each hole -- four slabs at
    45 degrees, so a blade a few millimetres off is slid in rather than stopped; the
    BACK, to `DEPTH`, is the holes themselves, `HOLE` half wide, with the wall behind
    them as the stop."""

    SUPPORTED_ROBOTS = ["panda_plug"]
    ROBOT = "panda_plug"
    LOOK_AT = np.array([0.48, 0.0, 0.48])
    SIDE_EYE = np.array([0.0, -0.55, 0.05])          # from the side, along the wall
    WALL_X = 0.54                         # m: the socket's face, nominal
    # the socket's frame turned -90 deg about z: its y (into the wall) is the world's +x
    Q_FRONT = [float(np.cos(-np.pi / 4)), 0.0, 0.0, float(np.sin(-np.pi / 4))]
    # down to the desk from a socket at ~0.52 m: centred 0.15 m below the socket
    WALL_HALF = np.array([0.25, 0.010, 0.25])
    WALL_DROP = 0.15
    HOLE = np.array([BLADE[0] + 0.0003, BLADE[1] + 0.0010])   # half: x (thick), z (wide)
    # THE HOLE IS A STEP WIDER THAN THE FUNNEL'S NARROW END.  Flush, the cord pressed the
    # blade on the bottom slab to its end and its leading edge caught the hole's floor,
    # which was exactly as high: stuck 5 mm in at every stiffness.  A step down cannot
    # catch.
    STEP = 0.0002                          # m
    CHAMFER = 0.005                        # m, deep and wide
    DEPTH = 0.020                          # m: the face to the stop
    PLATE_HALF = 0.035                     # m: half the faceplate, wide and high
    SLAB = 0.0005                          # m: half a chamfer slab's thickness
    MU = 0.2                               # brass on plastic
    CAVITY = 0.0                           # m behind the holes before the wall: none

    def _load_task(self) -> None:
        mat = sapien.physx.PhysxMaterial(self.MU, self.MU, 0.0)
        white = sapien.render.RenderMaterial(base_color=[0.93, 0.93, 0.91, 1])
        plate = sapien.render.RenderMaterial(base_color=[0.97, 0.97, 0.95, 1])
        dark = sapien.render.RenderMaterial(base_color=[0.70, 0.70, 0.68, 1])
        b = self.scene.create_actor_builder()

        def box(pos, half, q=(1.0, 0.0, 0.0, 0.0), look=plate):
            b.add_box_visual(pose=sapien.Pose(list(pos), list(q)), half_size=list(half),
                             material=look)
            b.add_box_collision(pose=sapien.Pose(list(pos), list(q)), half_size=list(half),
                                material=mat)

        P, hx, hz, g = self.PLATE_HALF, self.HOLE[0], self.HOLE[1], BLADE_GAP
        c, D, T = self.CHAMFER, self.DEPTH, self.SLAB
        # THE WALL, behind: its face is the stop the blades go home against -- or, with
        # a CAVITY, the back of the room the wall snap's catch slides in
        box((0.0, D + self.CAVITY + self.WALL_HALF[1], -self.WALL_DROP), self.WALL_HALF,
            look=white)
        self._cavity(box)
        # THE BACK LAYER, c to D: solid round the two holes, a STEP wider than the funnel
        bx, bz = hx + self.STEP, hz + self.STEP
        cy, hy = (c + D) / 2, (D - c) / 2
        for sx in (-1, 1):
            box((sx * (P + g + bx) / 2, cy, 0.0), ((P - g - bx) / 2, hy, P))     # outside
        box((0.0, cy, 0.0), (g - bx, hy, P))                                    # between
        for sz in (-1, 1):
            box((0.0, cy, sz * (P + bz) / 2), (g + bx, hy, (P - bz) / 2))       # above, below
        # THE FRONT LAYER, 0 to c: solid outside the two funnels' mouths
        cy, hy = c / 2, c / 2
        m = g + hx + c                       # the funnels' mouths reach out to here in x
        for sx in (-1, 1):
            box((sx * (P + m) / 2, cy, 0.0), ((P - m) / 2, hy, P))
        for sz in (-1, 1):
            box((0.0, cy, sz * (P + hz + c) / 2), (m, hy, (P - hz - c) / 2))
        # THE FUNNELS: each mouth's four edges, a slab from the face at (edge + c) to the
        # hole's edge c deep, its inner face on that line
        L = c * np.sqrt(2.0) / 2
        for x0 in (-g, g):
            for sx in (-1, 1):               # the two edges across x
                a = sx * np.pi / 4           # about z: the slab's y runs (-sx, 1)/sqrt2
                # half its thickness back into the material, away from the hole and
                # into the wall, so the face the blade meets lies ON the line
                ctr = (np.array([x0 + sx * (hx + c / 2), c / 2, 0.0])
                       + T * np.array([sx, 1.0, 0.0]) / np.sqrt(2.0))
                box(ctr, (T, L, hz + c), (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)), dark)
            for sz in (-1, 1):               # the two edges across z
                a = -sz * np.pi / 4          # about x: the slab's y runs (1, -sz)/sqrt2
                ctr = (np.array([x0, c / 2, sz * (hz + c / 2)])
                       + T * np.array([0.0, 1.0, sz]) / np.sqrt(2.0))
                box(ctr, (hx + c, L, T), (np.cos(a / 2), np.sin(a / 2), 0.0, 0.0), dark)
        # the holes, dark at the back, so a person can see where they are
        for x0 in (-g, g):
            b.add_box_visual(pose=sapien.Pose([x0, D - 0.0005, 0.0]), half_size=[hx, 0.0004, hz],
                             material=sapien.render.RenderMaterial(
                                 base_color=[0.05, 0.05, 0.05, 1]))
        b.initial_pose = sapien.Pose(p=[self.WALL_X, 0.0, 0.52], q=self.Q_FRONT)
        # kinematic, not static: an episode moves it
        self.socket = b.build_kinematic(name="socket")
        self._load_catch()

    def _cavity(self, box) -> None:
        """The walls of the room behind the holes (the wall snap's); none here."""

    def _load_catch(self) -> None:
        """What sits behind the holes (the wall snap's catch); nothing here."""


@dataclasses.dataclass
class PlugSpec:
    seed: int = 0
    socket_x: float = 0.54           # m: the socket's face, ahead of the arm
    socket_y: float = 0.0            # m along the wall
    socket_z: float = 0.52           # m: the holes' centre height
    cable: float = 0.0               # N, down, on the plug: a cord it drags (none now)
    err_y: float = 0.0               # m along the wall: told minus true
    err_z: float = 0.0               # m up: told minus true
    carry: tuple = (-0.10, 0.05)     # m (x, y) from the told mouth to where it starts
    time_limit: float = 20.0

    @staticmethod
    def sample(seed: int) -> "PlugSpec":
        r = np.random.default_rng(seed)
        sign = 1.0 if r.uniform() < 0.5 else -1.0
        side = 1.0 if r.uniform() < 0.5 else -1.0
        return PlugSpec(seed=seed, socket_x=float(r.uniform(0.52, 0.56)),
                        socket_y=float(r.uniform(-0.05, 0.05)),
                        # HIGH, where a level forward reach is clear of the joint stops
                        # (PlugEnv); and the hand is then far above the desk, so a dropped
                        # plug cannot be caught at socket height by the hand resting on it
                        # -- 21 of 100 "successes" at K_t LOW when the sockets were low.
                        socket_z=float(r.uniform(0.50, 0.54)),
                        # NO CORD (the user's choice, 2026-10-15: "just use plug").  The
                        # field stays, at 0, so logs before and after read alike.
                        cable=0.0,
                        # ALONG THE WALL, the offset the funnel has to slide out: 2.8-3.8
                        # mm either way -- inside the 5.3 mm it captures, with room for the
                        # hand's own error -- and past the hole's 0.5 mm of play by enough
                        # that HIGH K_t pushes back with ~4-5.5 N and MID with under 2.5
                        err_y=float(sign * r.uniform(0.0028, 0.0038)),
                        err_z=float(r.uniform(-0.0005, 0.0005)),
                        # carried in from back and to one side
                        carry=(float(r.uniform(-0.12, -0.08)),
                               float(side * r.uniform(0.03, 0.07))))


class PlugSim(ContactSim):
    """Carry the plug to the socket and push it home.

    THE CORD is `cable` newtons down on the plug every physics step, applied to its body
    and known to nothing else: the controller compensates the arm and the plug's own
    weight and not this, so it is held up by K_t alone.  It comes on over `CABLE_RAMP`
    seconds -- the cord lifting off the desk -- so the first stiffness a person sets is
    not judged before they can set it.

    WHAT FAILS.  Not getting in (`inserted`: the blades within 2 mm of the stop), and a
    side load on the blades while they are in: the sustained force the socket pushes
    them with ALONG THE WALL -- across the insertion and across gravity, so the cord's
    weight is not in it -- from `HOLD_GRACE` after they enter to the end, must stay
    under `SIDE_MAX` (`no_bend`).
    """
    ENV_ID = "TeleopPlug-v1"
    TASK = "plug"
    TOOL_LINK = "plug"
    LANDS = False
    WORK_ON = "touch"            # in the mouth the moment the blades meet the socket
    FIXED_FRAME = True           # along IS the push, for the whole episode
    SEATS = True                 # the protocol's `stop` row comes in once it is home
    START_ON_TABLE = True
    START_KR = 30.0
    E_MAX = 0.006                # m: K_t holds at most 1.8 / 4.8 / 12 N at the cap
    # PER AXIS, in the stiffness frame (across, across, along): 6 mm in the wall plane,
    # which is the cord's test, and 15 mm into the wall, for the push.  On the sphere of
    # E_MAX the cord's sag used up the whole limit and a VR operator could move the
    # plug neither in nor out, nor lift it back: "the robot does not follow".
    E_BOX = (0.006, 0.006, 0.015)
    E_RATE = 0.10                # m/s, as the snap's
    # N: the force ceiling, as the snap's.  Across it does not bind -- HIGH K_t makes 12 N
    # at the 6 mm side -- and along it caps the push into the socket at 15 N (5 mm at
    # K_n HIGH), which the insertion needs a third of.
    F_SAFE = 15.0
    CABLE_RAMP = 1.0             # s
    HOLD_GRACE = 1.0             # s after entering before the side load is judged: as
                                 # the snap's, a reaction and the ramp down
    # s it is held home, judged, before the task ends the episode itself.  With the
    # trigger the plug went home 0.2 s after it touched and the episode ended before the
    # side load was ever judged: K_t HIGH passed 8/8.
    HOLD_S = 1.0
    SIDE_MAX = 3.0               # N, sustained, along the wall on the blades
    SIDE_LP_HZ = 5.0
    MU_BLADE = 0.2
    # level, blades forward at the wall; the hand's x up (the drawer's grasp), its y
    # along the wall
    R_START = np.column_stack([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    APPROACH = FWD
    # outward normal of the work as the tip comes at it: -x, out of the wall -- the VR
    # mapping's own frame, so the trigger pushes the way "away from you" moves
    HAND_AXES = CT.VR_AXES
    CRITERIA = _NO_STICK_LIMITS
    # A FIRM GRIP, as the wok's (see `protocol.run_episode`, VR_MAP).  The default VR
    # hand holds its handle through 300 N/m with the arm's force reflected onto it, which
    # is softer than K_t at MID or HIGH: in series with it, the arm's stiffness hardly
    # reaches the socket, and measured going in, HIGH pushed back with 1 N while the
    # funnel slid it 2 mm -- the hand gave way, so no K_t could be wrong.  A person
    # holding a plug to a socket holds it where they aim it.  The scripted hand is
    # given the same grip (`PlugScript`).
    # ...firm in the WALL PLANE only.  Along the push the grip is the default 300 N/m,
    # so the trigger's push moves the plug: through 10 kN/m its 20 N moved the handle
    # 2 mm and "the trigger does not work at all" (2026-10-14).
    # max_force 100 N, not 30: the grip's clip drags the controller's mapped target along
    # with the handle once its spring passes max_force, and at 10 kN/m 30 N is 3 mm of
    # slack -- brisk moves hit it and the mapping slipped, mostly DOWN under the cord
    # ("the end effector falls, it does not follow the controller").  What reaches the
    # work is bounded on the robot's side (E_BOX, F_SAFE), not by this.
    VR_MAP = dict(k_hand=10000.0, b_hand=140.0, max_force=100.0,
                  k_hand_press=300.0, b_hand_press=35.0)
    ROW_EXTRA = ("inserted", "insert_s", "side_peak")
    # THE TRIGGER PUSHES THE PLUG IN, as the snap's pushes the buckle down: 30 N at a full
    # pull, along -HAND_AXES[:, 2] (into the wall), on top of what the grip holds --
    # through the soft press-axis grip (VR_MAP), so it moves the plug up to ~10 cm ahead
    # of a hand held still.  15 N was too little: a hand held 2 cm short of the face
    # left ~3 N of push at the funnel, where a plug at LOW K_t rests on the 45-degree
    # lower edge and is pushed back by about its cord's weight -- 0/8 went in.
    PRESS_FORCE = 30.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOT the stick's rubber ball: brass blades.  At 1.5 a 45-degree chamfer jams.
        brass = sapien.physx.PhysxMaterial(self.MU_BLADE, self.MU_BLADE, 0.0)
        for s in self.tool._objs[0].get_collision_shapes():
            s.set_physical_material(brass)
        self.body = self.tool._objs[0]

    def _work(self):
        return self.u.socket

    def mouth(self) -> Array:
        """The socket's face, centred between the holes."""
        s = self.spec
        return np.array([s.socket_x, s.socket_y, s.socket_z])

    def depth(self) -> float:
        """How far the blades' tips are IN THE SOCKET, m: past its face, and lined up
        with the holes -- within the funnel's mouth along the wall and up and down.
        -1 when they are not lined up, wherever they are.

        Past the face alone is not in: a plug dropped below the faceplate went 18 mm
        past its plane, under it, and "inserted" 100 of 100 episodes with K_t pinned
        LOW once the sockets were raised clear of the desk."""
        _, p, _, _ = self.ctl.tip_state()
        d = p - self.mouth()
        u = self.u
        if (abs(float(d @ ALONG_WALL)) > u.HOLE[0] + u.CHAMFER
                or abs(float(d[2])) > u.HOLE[1] - BLADE[1] + u.CHAMFER):
            return -1.0
        return float(d @ FWD)

    def reset(self, spec: PlugSpec) -> dict:
        self.spec = spec
        self.env.reset(seed=spec.seed)
        self.u.socket.set_pose(sapien.Pose(p=self.mouth().tolist(), q=self.u.Q_FRONT))
        self.tip_on = self.mouth()
        self.told = self.tip_on + spec.err_y * ALONG_WALL + spec.err_z * UP
        self._begin(self.told + np.array([spec.carry[0], spec.carry[1], 0.0]), FWD)
        self.inserted_t = None
        self._cued = False
        self.side_slow = 0.0
        self.side_peak = 0.0
        self.max_depth = -1.0
        return self.observe()

    def step(self, prop) -> dict:
        # the cord, on the body, for the physics step about to be taken
        f = float(self.spec.cable) * min(1.0, self.t / self.CABLE_RAMP)
        self.body.add_force_torque(np.array([0.0, 0.0, -f], np.float32),
                                   np.zeros(3, np.float32))
        return super().step(prop)

    def _force(self, rec) -> Array:
        # kept: `_task_step` filters its along-the-wall part for the side load
        self._f_now = self.u.scene.get_pairwise_contact_forces(
            self.tool, self.u.socket)[0].cpu().numpy()
        return self._f_now

    def inserted(self) -> bool:
        return getattr(self, "inserted_t", None) is not None

    def seated(self) -> bool:
        """The protocol's `stop`: home."""
        return self.inserted()

    def _over(self) -> bool:
        return self.inserted()

    def held(self) -> bool:
        """Home, and held there until the side load has been judged for HOLD_S."""
        if not self.inserted() or self.t_touch is None:
            return False
        return self.t >= max(self.inserted_t, self.t_touch + self.HOLD_GRACE) + self.HOLD_S - 1e-6

    def finished(self) -> bool:
        return self.held()

    def cue(self) -> str | None:
        if self.t_touch is not None and not self._cued and self.depth() > 0.0:
            self._cued = True
            return "in"
        return None

    def _task_step(self, rec, touching: bool) -> None:
        d = self.depth()
        self.max_depth = max(self.max_depth, d)
        if self.inserted_t is None and d >= self.u.DEPTH - 0.002:
            self.inserted_t = self.t
        self.engaged = d > 0.0
        b = self.dt / (self.dt + 1.0 / (2 * np.pi * self.SIDE_LP_HZ))
        self.side_slow += b * (float(self._f_now @ ALONG_WALL) - self.side_slow)
        if self.t_touch is not None and self.t - self.t_touch >= self.HOLD_GRACE:
            self.side_peak = max(self.side_peak, abs(self.side_slow))

    def _task_fields(self) -> dict:
        if not hasattr(self, "spec"):
            return dict(progress=0.0, depth=-1.0)
        d = self.depth()
        return dict(progress=float(np.clip(d / (self.u.DEPTH - 0.002), 0.0, 1.0)), depth=d)

    def privileged(self) -> dict:
        s = self.spec
        return {"socket_x": s.socket_x, "socket_y": s.socket_y, "socket_z": s.socket_z,
                "cable": s.cable, "err_y": s.err_y, "err_z": s.err_z}

    def score(self) -> dict:
        res, checks = self._common_score()
        full = self.u.DEPTH - 0.002
        res.update(inserted=self.inserted(), insert_s=self.inserted_t,
                   side_peak=float(self.side_peak), depth_m=float(self.depth()),
                   progress=float(np.clip(self.max_depth / full, 0.0, 1.0)))
        checks = {"inserted": self.inserted(), "held": self.held(),
                  "no_bend": self.side_peak <= self.SIDE_MAX,
                  "no_collision": checks["no_collision"]}
        return self._finish(res, checks)


# =========================================================================== WALL SNAP
# The plug, its blades longer: the MALE HALF of a snap-fit, held by the arm.  Long
# enough that the tips go through the holes and through the catch's whole stroke behind
# them with the body still clear of the faceplate (`WallSnapEnv`).
PRONG_BLADE = np.array([BLADE[0], BLADE[1], 0.0195])    # half: 39 mm long


@register_agent()
class PandaProng(PandaStick):
    """PandaStick holding the wall snap's male half: the plug with longer blades."""
    uid = "panda_prong"
    urdf_path = _plug_assets("panda_prong", PRONG_BLADE)


@register_env("TeleopWallSnap-v1", max_episode_steps=1_000_000)
class WallSnapEnv(PlugEnv):
    """robot --- male half --- wall.  The plug's wall and socket -- the same funnels,
    the same holes, shorter -- with a ROOM behind the holes, and in it the CATCH: a plate
    across both holes that the prongs' tips meet and push back.  It stands for the
    female half's spring arms: it resists until the click, then lets go, and the part is
    latched (`WallSnapSim`).

    In the socket's frame (x along the wall, y into it, z up): the funnels 0 to CHAMFER,
    the holes to DEPTH, the catch's face GAP behind them, and the wall CAVITY behind
    that.  The tips meet the catch at DEPTH + GAP and are seated at
    DEPTH + GAP + x_click + free -- 34.5 mm at most, against blades 39 mm long.

    THE CATCH IS DRAWN, NOT FELT: its push is a force along the insertion
    (`WallSnapSim.step`).  As a contact it held the tips sideways by FRICTION, which
    this engine does not report -- the socket's walls read 4-6 N on the blades at K_t
    LOW while the arm pushed sideways with 1.1 N (2026-10-16)."""

    SUPPORTED_ROBOTS = ["panda_prong"]
    ROBOT = "panda_prong"
    DEPTH = 0.015                  # m: the face to the end of the holes (5 funnel, 10 hole)
    GAP = 0.0005                   # m from the holes' end to the catch's face, at rest
    CATCH = np.array([0.016, 0.0015, 0.010])     # half: across both blades, thin, tall
    STROKE_MAX = 0.020             # m: longer than any episode's x_click + free
    CAVITY = GAP + 2 * CATCH[1] + STROKE_MAX + 0.001
    ROOM = np.array([0.020, 0.013])              # half: the room's inside, x and z

    def _cavity(self, box) -> None:
        P, D, C = self.PLATE_HALF, self.DEPTH, self.CAVITY
        rx, rz = self.ROOM
        cy, hy = D + C / 2, C / 2
        for sx in (-1, 1):
            box((sx * (P + rx) / 2, cy, 0.0), ((P - rx) / 2, hy, P))
        for sz in (-1, 1):
            box((0.0, cy, sz * (P + rz) / 2), (rx, hy, (P - rz) / 2))

    def _load_catch(self) -> None:
        b = self.scene.create_actor_builder()
        b.add_box_visual(half_size=self.CATCH.tolist(),
                         material=sapien.render.RenderMaterial(base_color=[0.95, 0.55, 0.10, 1]))
        b.initial_pose = sapien.Pose(p=[self.WALL_X + self.DEPTH, 0.0, 0.52], q=self.Q_FRONT)
        self.catch = b.build_kinematic(name="catch")

    def place_catch(self, mouth, stroke: float) -> None:
        """Draw the catch `stroke` in from rest, behind the socket's face at `mouth`."""
        y = self.DEPTH + self.GAP + float(stroke) + self.CATCH[1]
        self.catch.set_pose(sapien.Pose(p=(np.asarray(mouth, float) + y * FWD).tolist(),
                                        q=self.Q_FRONT))


@dataclasses.dataclass
class WallSnapSpec(PlugSpec):
    f_click: float = 35.0            # N the catch resists with at the click
    x_click: float = 0.012           # m of the catch's stroke to the click
    free: float = 0.004              # m from the click to the stop
    drag: float = 20.0               # N s/m along the catch's rail

    @staticmethod
    def sample(seed: int) -> "WallSnapSpec":
        # the plug's scene, draw for draw, and the latch from a stream of its own
        base = dataclasses.asdict(PlugSpec.sample(seed))
        r = np.random.default_rng([seed, 41])
        return WallSnapSpec(**base, f_click=float(r.uniform(30.0, 40.0)),
                            x_click=float(r.uniform(0.010, 0.014)),
                            free=float(r.uniform(0.003, 0.005)),
                            drag=float(r.uniform(10.0, 30.0)))


class WallSnapSim(PlugSim):
    """Carry the male half to the wall, into the holes, and push it on until it clicks;
    then stop pressing on it.  The plug and the snap in one task, K_t and K_n each
    changing once:

        K_t   the plug's: in the holes, a stiff K_t pushes the blades against their sides
              for as long as the funnel's correction is held -- soft from the entry
        K_n   the snap's: the catch resists 30-40 N to the click, which only HIGH reaches
              at the reach limit; pressing on afterwards is pressing a latched part --
              soft from the click

    THE CATCH is the snap's rail as a force along the insertion, on the part: from the
    tips meeting it, `f_click / x_click` per metre of stroke and `drag` per m/s; slack
    the moment the stroke passes `x_click`; and from then on a LATCH -- the part runs
    `free` onto its seat and can go neither further in nor back out past the click,
    both through a stiff spring (K_SEAT).  Along the insertion only, so it cannot hold
    the part sideways: the side load is what the socket's walls do, which is what the
    stiffness sets.  The sensor is the socket's contact and the catch's force together.

    WHAT FAILS.  No click (`clicked`); a side load on the blades over SIDE_MAX from
    HOLD_GRACE after they first touch to the end, but for the click's own jolt
    (`no_bend`, the plug's); a load on the
    latched part over HOLD_MAX, pushed or pulled, from HOLD_GRACE after the click to the
    end (`gentle`, the snap's); ending before that has been judged for HOLD_S (`held`).

    A LIGHT PART (the user's, 2026-10-16): no cord and no load the controller does not
    know about, so nothing in free space is judged.  Whether K_t's free-space level
    then matters is measured, not assumed (`stiffness_helper.envs.axis_check`).
    """
    ENV_ID = "TeleopWallSnap-v1"
    TASK = "wallsnap"
    # the plug's 6 mm in the wall plane, and the snap's 20 mm along: K_n makes at most
    # 6 / 20 / 45 N at LOW / MID / HIGH with F_SAFE, so only HIGH clicks, as the snap's
    E_BOX = (0.006, 0.006, 0.020)
    E_RATE = 0.03                # as the snap's: the catch YIELDS under the push
    F_SAFE = 45.0                # as the snap's; across it does not bind (2000 x 6 mm)
    PRESS_FORCE = 45.0           # N, a full trigger in VR: the click needs 30-40
    HOLD_MAX = 12.0              # N, sustained, on the latched part: the snap's
    K_SEAT, B_SEAT = 5.0e4, 300.0    # N/m, N s/m: the seat and the latch, ~1 mm at 45 N
    # s after the click when the side load is NOT judged: the catch letting go of a
    # 30-45 N push throws the part onto its seat, and the arm's inertia, coupled across
    # the push, rattles the blades in their holes -- 5-8 N for ~50 ms at K_t LOW, whose
    # sustained side load is 1 N.  The jolt is the catch's, not the stiffness's.
    CLICK_JOLT = 0.3
    # FOR A PERSON, as the snap's: K_t's MID is no use once in and K_n's none at the
    # catch, so one stick push crosses each; softening is fast; both events are buzzed
    TWO_LEVEL = (0, 1)
    RAMP_DOWN = 0.05
    ROW_EXTRA = ("inserted", "insert_s", "side_peak", "clicked", "click_s", "hold_peak",
                 "touch_s")

    def reset(self, spec: WallSnapSpec) -> dict:
        self.clicked_t = None
        self.cx = 0.0                                # the catch's stroke
        self.f_catch = 0.0                           # N, on the part, back out along it
        super().reset(spec)
        self._cued_click = False
        self.press_slow = 0.0
        self.hold_peak = 0.0
        self.max_x = 0.0
        self.u.place_catch(self.mouth(), 0.0)
        return self.observe()

    def stroke(self) -> float:
        return float(self.cx)

    def _tip_stroke(self) -> tuple[float, float]:
        """How far past the catch's rest the tips are (m; -1 when not in the holes),
        and how fast they are going in (m/s)."""
        d = self.depth()
        if d < 0.0:
            return -1.0, 0.0
        _, _, v, _ = self.ctl.tip_state()
        return d - (self.u.DEPTH + self.u.GAP), float(v @ FWD)

    def step(self, prop) -> dict:
        s = self.spec
        x, xd = self._tip_stroke()
        f = 0.0
        if not self.clicked():
            if x > 0.0:
                f = max(0.0, s.f_click / s.x_click * x + s.drag * xd)
            self.cx = float(np.clip(x, 0.0, s.x_click))
        else:
            seat = s.x_click + s.free
            if x > seat:
                f = self.K_SEAT * (x - seat) + self.B_SEAT * xd
            elif -1.0 < x < s.x_click:                   # the hooks: it cannot come out
                f = self.K_SEAT * (x - s.x_click) + self.B_SEAT * xd
            self.cx = float(np.clip(x, s.x_click, seat))
        self.f_catch = float(f)
        if f != 0.0:
            self.body.add_force_torque((-f * FWD).astype(np.float32),
                                       np.zeros(3, np.float32))
        return super().step(prop)

    def _limit(self, lo: float) -> None:
        pass

    def clicked(self) -> bool:
        return getattr(self, "clicked_t", None) is not None

    def seated(self) -> bool:
        """The protocol's `stop`: from the click on."""
        return self.clicked()

    def _over(self) -> bool:
        return self.clicked()

    def held(self) -> bool:
        if not self.clicked() or self.t_touch is None:
            return False
        return self.t >= max(self.clicked_t, self.t_touch) + self.HOLD_GRACE + self.HOLD_S - 1e-6

    def cue(self) -> str | None:
        c = super().cue()                            # "in": the blades are in the mouth
        if c is None and self.clicked() and not self._cued_click:
            self._cued_click = True
            c = "click"
        return c

    def _force(self, rec) -> Array:
        self._f_now = (self.u.scene.get_pairwise_contact_forces(
            self.tool, self.u.socket)[0].cpu().numpy() - self.f_catch * FWD)
        return self._f_now

    def _task_step(self, rec, touching: bool) -> None:
        judged = self.side_peak
        super()._task_step(rec, touching)            # depth, in, the side load
        if self.clicked() and self.t - self.clicked_t < self.CLICK_JOLT:
            self.side_peak = judged
        s = self.spec
        self.max_x = max(self.max_x, self.cx)
        if self.clicked_t is None and self._tip_stroke()[0] >= s.x_click:
            self.clicked_t = self.t
        # the load on the part along the insertion, either way: pushed on its seat or
        # pulled against its hooks
        b = self.dt / (self.dt + 1.0 / (2 * np.pi * self.crit.pressure_lp_hz))
        self.press_slow += b * (abs(float(self._f_now @ FWD)) - self.press_slow)
        if self.clicked() and self.t - self.clicked_t >= self.HOLD_GRACE:
            self.hold_peak = max(self.hold_peak, self.press_slow)
        if self.step_i % 10 == 0:
            self.u.place_catch(self.mouth(), self.cx)

    def _full(self) -> float:
        s = self.spec
        return self.u.DEPTH + self.u.GAP + s.x_click + s.free

    def _task_fields(self) -> dict:
        if not hasattr(self, "spec") or not hasattr(self, "max_x"):
            return dict(progress=0.0, depth=-1.0, stroke=0.0, clicked=False)
        d = self.depth()
        return dict(progress=float(np.clip(d / self._full(), 0.0, 1.0)), depth=d,
                    stroke=self.stroke(), clicked=self.clicked())

    def privileged(self) -> dict:
        s = self.spec
        return {**super().privileged(), "f_click": s.f_click, "x_click": s.x_click,
                "free": s.free, "drag": s.drag}

    def score(self) -> dict:
        res, checks = self._common_score()
        res.update(inserted=self.inserted(), insert_s=self.inserted_t,
                   side_peak=float(self.side_peak), depth_m=float(self.depth()),
                   clicked=self.clicked(), click_s=self.clicked_t, touch_s=self.t_touch,
                   hold_peak=float(self.hold_peak), stroke_m=float(self.stroke()),
                   progress=float(np.clip(self.max_depth / self._full(), 0.0, 1.0)))
        checks = {"clicked": self.clicked(), "held": self.held(),
                  "no_bend": self.side_peak <= self.SIDE_MAX,
                  "gentle": self.hold_peak <= self.HOLD_MAX,
                  "no_collision": checks["no_collision"]}
        return self._finish(res, checks)


CT.SIMS["snap"] = (SnapSim, SnapSpec)
CT.SIMS["plug"] = (PlugSim, PlugSpec)
CT.SIMS["wallsnap"] = (WallSnapSim, WallSnapSpec)
