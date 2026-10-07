"""Peg insertion as a HAND-DRIVEN task: the simulator `../writing` and `../wiping` are,
with a peg in the gripper and a hole to find.

`../peg_insertion_case1.py` is the SCRIPTED study of this scene -- a spiral search under
the Case 1 controller, and the measurements in ../PEG_DEMO_PLAN.md.  This file is the
other half: the same scene under the same controller with the motion handed to a person,
so `teleop.TeleopSession`, `interactive.VRHand` and `collect.Recorder` run on it
unchanged.  It offers exactly what those three read off `sim.WritingSim` -- `ctl`, `W`,
`dt`, `step`, `last`, `observe`, `score`, `goal`, `privileged` -- and nothing else is
shared, because nothing else is the same task.

WHAT THE HAND IS GIVEN, and what it is not.

  ALIGNED AND CARRIED TO A STAND-OFF, by `reset`.  The proposal has no rotational
  reference velocity, so turning the peg onto the hole axis is not something a hand on
  a 3-DoF handle can do; the scripted study slerps R_d in free space for the same
  reason and says so.  It is done here before the episode starts and is not recorded.

  A WEIGHTLESS PEG.  ManiSkill disables link gravity on a fixed-base arm, so the arm is
  weightless and the grasped peg is not: 0.4-0.5 kg sags the hand 13.8 mm on a 300 N/m
  lateral axis, against 3 mm of clearance.  The scripted study calibrates that into a
  frozen bias, which is only valid at the stiffness it was measured at -- and a person
  changes the stiffness.  A real arm compensates its payload, so the peg's gravity is
  switched off here instead and the sag never exists.

  NOT THE HOLE.  The peg starts in front of the face and off the hole by `PegSpec.offset`,
  so there is something to find by sight and by feel.

THE FRAME.  `W = (e1, e2, n)` with `n` the entrance face's OUTWARD normal, i.e. minus the
hole axis.  That is the convention the pen and the pad use -- the third column is what a
"press" pushes against -- so the trigger presses INTO the hole and the stiffness triple
reads (lateral, lateral, axial).

THE VIEW.  The operator's window stands 65 degrees round from the hole's axis, because
from behind the peg the hand that holds it hides the hole; `vr_axes` makes the
controller follow that window, so the hand and the screen agree.  See `_view`.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # the scripted peg study lives there
sys.path.insert(0, str(HERE.parent / "writing"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import controller as C  # noqa: E402  writing/controller.py, as teleop.py imports it
import mani_skill.envs  # noqa: E402,F401
from mani_skill.utils.structs.pose import Pose  # noqa: E402
from transforms3d.quaternions import mat2quat  # noqa: E402

Array = np.ndarray

# Fixed in the world, because the box moves a few centimetres per episode and a camera
# that followed it would hand the policy the hole's position for free.
TOP_EYE, TOP_AT = [0.02, 0.16, 0.62], [0.0, 0.20, 0.10]
SIDE_EYE, SIDE_AT = [0.46, 0.10, 0.22], [0.0, 0.22, 0.12]
# The window's FIRST frame, before an episode has put a hole anywhere: see `_view`.
VIEW_EYE, VIEW_AT = [0.42, -0.10, 0.34], [0.0, 0.26, 0.10]


def _slerp(R0: Array, R1: Array, a: float) -> Array:
    """R0 turned the fraction `a` of the way to R1."""
    S = R0.T @ R1
    c = np.clip((np.trace(S) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return R0.copy()
    w = th / (2.0 * np.sin(th)) * C.vee(S - S.T) * a
    th = float(np.linalg.norm(w))
    k = w / th if th > 1e-12 else np.zeros(3)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return R0 @ (np.eye(3) + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx))


def _ease(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return 0.5 - 0.5 * np.cos(np.pi * a)


def _align(R0: Array, a_from: Array, a_to: Array) -> Array:
    """R0 turned by the smallest rotation carrying `a_from` onto `a_to`."""
    a = a_from / np.linalg.norm(a_from)
    b = a_to / np.linalg.norm(a_to)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return R0.copy()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + vx + vx @ vx / (1.0 + c)) @ R0


@dataclasses.dataclass
class PegSpec:
    """Everything that varies between episodes.  The box pose comes from `seed`."""
    seed: int = 0
    offset: tuple = (0.0, 0.0)        # m across the face (e1, e2): where the peg starts
    standoff: float = 0.035           # m in front of the face
    time_limit: float = 90.0

    @staticmethod
    def sample(seed: int, max_offset: float = 0.012) -> "PegSpec":
        """Off the hole by up to 12 mm, in a random direction: four clearances, which is
        where the scripted search stopped being a formality (PEG_DEMO_PLAN.md, step 1)."""
        rng = np.random.default_rng(seed)
        th, r = rng.uniform(0, 2 * np.pi), max_offset * np.sqrt(rng.uniform())
        return PegSpec(seed=seed, offset=(float(r * np.cos(th)), float(r * np.sin(th))))


@dataclasses.dataclass
class PegCriteria:
    touch_force: float = 0.8          # N on the peg from the box: it is touching
    catch: float = 0.005              # m past the entrance face: the head is in the hole
    # A good press.  The scripted study found 4 N best and both 2 N and 8 N worse -- a
    # light press skates over the aperture and a heavy one lets the sharp edge bite.
    force_band: tuple = (1.0, 8.0)
    # Above ~50 N the wedge pulls the peg out of the fingers (PEG_DEMO_PLAN.md, step 2).
    jam_force: float = 50.0
    slip: float = 0.015               # m the peg may move in the hand before it is lost
    force_lp_hz: float = 25.0         # the force sensor: observations and logs
    pressure_lp_hz: float = 5.0       # sustained force: the band and the jam


class PegSim:
    ENV_ID = "PegInsertionSide-v1"
    TOOL_LINK = "peg"

    def __init__(self, image_size: int = 128, cameras: bool = True,
                 gains: C.Case1Gains | None = None, criteria: PegCriteria | None = None,
                 render_mode: str | None = None, build_seed: int = 0,
                 clearance_mm: float = 0.0, grip_back: float = 0.040):
        from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils
        from mani_skill.utils.structs.types import SimConfig
        self.cameras = cameras
        self.grip_back = float(grip_back)
        self.crit = criteria or PegCriteria()
        if clearance_mm > 0.0:
            PegInsertionSideEnv._clearance = clearance_mm * 1e-3
        size = int(image_size)

        # Named after wiping's two streams on purpose: the recorder, the console's
        # camera tap and a deployed helper all read `rgb_top_camera` and
        # `rgb_wrist_camera`.  The second is a fixed side view here, not a wrist camera
        # -- the task's own wrist camera body strikes the box face at 168 N while the
        # peg is still 35 mm short (see ../peg_insertion_cascade.py), so the arm is the
        # plain panda -- and it is the view where the gap to the face can be seen.
        @property
        def _sensors(env):
            return [CameraConfig("top_camera", sapien_utils.look_at(TOP_EYE, TOP_AT),
                                 size, size, np.pi / 2, 0.01, 100),
                    CameraConfig("wrist_camera", sapien_utils.look_at(SIDE_EYE, SIDE_AT),
                                 size, size, np.pi / 2, 0.01, 100)]

        @property
        def _window(env):
            return CameraConfig("render_camera", sapien_utils.look_at(VIEW_EYE, VIEW_AT),
                                960, 720, 1.0, 0.01, 100)

        old = (PegInsertionSideEnv._default_sensor_configs,
               PegInsertionSideEnv._default_human_render_camera_configs)
        PegInsertionSideEnv._default_sensor_configs = _sensors
        PegInsertionSideEnv._default_human_render_camera_configs = _window
        try:
            # 500 Hz, not the task's 100: an inner Cartesian impedance cannot live at
            # 10 ms (../peg_insertion_cascade.py has the arithmetic).  ONE env for the
            # whole session and never reconfigured, because reconfiguring closes the
            # viewer.  So the peg is one size throughout -- and, built this way, the
            # SAME size in every session: 171 x 40 mm in a 46 mm hole whatever the
            # seed, measured, not intended.  What changes between episodes is where
            # the box is and which way the hole points.  Not reconfiguring has to
            # be asked for: a single CPU env reconfigures on every reset by default,
            # which rebuilds the robot under the controller that was holding it -- the
            # arm then takes 34 Nm and does not move, with nothing to say why.
            self.env = gym.make(self.ENV_ID, num_envs=1, sim_backend="cpu",
                                robot_uids="panda", render_mode=render_mode,
                                reconfiguration_freq=0,
                                sim_config=SimConfig(sim_freq=500, control_freq=500))
            self.env.reset(seed=build_seed)
        finally:
            (PegInsertionSideEnv._default_sensor_configs,
             PegInsertionSideEnv._default_human_render_camera_configs) = old
        self.u = self.env.unwrapped
        self.dt = 1.0 / self.u.sim_freq
        self.robot = self.u.agent.robot
        self.nq = len(self.robot.active_joints)
        g = gains or C.Case1Gains()
        # The fingers are not part of the impedance: a zero limit on their two entries
        # leaves the grasp's own drive as the only thing holding the peg, which is also
        # why disable_joint_drives() is never called here -- it would open the hand.
        g.tau_limit = np.concatenate([C.PANDA_TAU_LIMIT, np.zeros(self.nq - 7)])
        self.ctl = C.Case1Controller(self.robot, g, self.u.agent.urdf_path,
                                     tcp_name=self.u.agent.tcp.name)
        self.half_len = float(self.u.peg_half_sizes[0, 0])
        self.half_w = float(self.u.peg_half_sizes[0, 1])
        self.hole_half_w = float(self.u.box_hole_radii[0])
        import sapien
        body = self.u.peg._objs[0].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent)
        body.disable_gravity = True                 # a compensated payload; see the top
        self.W = np.eye(3)
        self.K0 = np.array([1000.0, 1000.0, 1000.0])
        self.last: dict = {}

    # ---- the scene's own geometry ----------------------------------------
    def _hole(self) -> tuple[Array, Array]:
        hp = self.u.box_hole_pose
        return (hp.to_transformation_matrix()[0, :3, :3].cpu().numpy(),
                hp.p[0].cpu().numpy())

    def _peg_axis(self) -> Array:
        R = self.u.peg.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
        a = R[:, 0]
        return a if a @ self.axis > 0 else -a

    def head(self) -> Array:
        return self.u.peg.pose.p[0].cpu().numpy() + self.half_len * self._peg_axis()

    def depth(self) -> tuple[float, Array, bool]:
        """(m past the entrance face, the head in the hole frame, seated)."""
        ok, at = self.u.has_peg_inserted()
        at = at[0].cpu().numpy()
        return float(at[0]) + self.half_len, at, bool(ok[0])

    # ---- where the operator stands ----------------------------------------
    VIEW_OFF = np.radians(65.0)     # how far round from the hole's axis the window looks
    VIEW_DIST, VIEW_UP = 0.42, 0.20  # m from the entrance, and above it

    def _view(self) -> tuple[Array, Array]:
        """(eye, horizontal view direction) for THIS episode's hole.

        NOT FROM BEHIND THE PEG.  That is the obvious place -- it makes "push forward"
        and "into the hole" the same gesture -- and from there the hand that holds the
        peg covers the hole completely, and at 40 or 50 degrees round it still covers the
        peg's head.  So the window stands 65 degrees round, on the side away from the
        robot's base: the peg is seen nearly side-on with the gap in front of it, and
        the entrance is still open enough to see the hole in it.
        """
        up = np.array([0.0, 0.0, 1.0])
        a = self.axis - (self.axis @ up) * up
        a /= np.linalg.norm(a)
        s = np.cross(a, up)
        base = self.robot.pose.p[0].cpu().numpy()
        if s @ (self.face - base) < 0:
            s = -s                                   # away from the arm, not through it
        f = np.cos(self.VIEW_OFF) * a - np.sin(self.VIEW_OFF) * s
        return self.face - self.VIEW_DIST * f + self.VIEW_UP * up, f

    @property
    def vr_axes(self) -> Array:
        """WebXR (right, up, back) -> world, with FORWARD into the window.

        The controller follows the VIEW, not the hole: what moves away from you on the
        screen is what you pushed away from you.  Going in is therefore a diagonal
        push, the way it looks -- and the trigger still presses along the hole itself,
        because that comes from `W`, not from here.
        """
        up = np.array([0.0, 0.0, 1.0])
        _, f = self._view()
        return np.column_stack([np.cross(f, up), up, -f])

    @property
    def hole_bearing(self) -> float:
        """Degrees the hole runs to the RIGHT of straight ahead in the window (negative
        is left): the one number an operator needs to know which way "in" is."""
        up = np.array([0.0, 0.0, 1.0])
        _, f = self._view()
        return float(np.degrees(np.arctan2(self.axis @ np.cross(f, up), self.axis @ f)))

    def aim_viewer(self, viewer) -> None:
        """Put the operator's window where `_view` says, through the viewer's own
        fly-camera state, as `writing/scene.py` does it."""
        eye, _ = self._view()
        d = self.face - eye
        d /= np.linalg.norm(d)
        viewer.set_camera_xyz(*eye)
        viewer.set_camera_rpy(0.0, -np.arcsin(-d[2]), -np.arctan2(d[1], d[0]))

    # ---- the grasp, as ../peg_insertion_cascade.py's setup does it ---------
    def _grasp(self) -> None:
        u, robot = self.u, self.robot
        Rt = u.agent.tcp.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
        pt = u.agent.tcp.pose.p[0].cpu().numpy()
        Rp = np.column_stack([Rt[:, 0], Rt[:, 2], np.cross(Rt[:, 0], Rt[:, 2])])
        if np.linalg.det(Rp) < 0:
            Rp[:, 2] *= -1.0
        # Gripped BEHIND its centre: held at the centre the hand strikes the box face
        # while the head is still short of seated.
        centre = pt + self.grip_back * Rp[:, 0]
        u.peg.set_pose(Pose.create_from_pq(
            torch.tensor(centre, dtype=torch.float32)[None, :],
            torch.tensor(mat2quat(Rp), dtype=torch.float32)[None, :]))
        q = robot.get_qpos()[0].cpu().numpy().copy()
        q[-2:] = self.half_w + 0.004
        robot.set_qpos(q[None, :])
        robot.set_qvel(np.zeros((1, self.nq)))
        u.peg.set_linear_velocity(torch.zeros(1, 3))
        u.peg.set_angular_velocity(torch.zeros(1, 3))
        # Stepping the scene directly bypasses the env's controller, so every joint has
        # to be told to hold where it is or the arm drives to its zero configuration.
        for i, j in enumerate(robot.active_joints):
            j.set_drive_target(0.0 if "finger" in j.name else float(q[i]))
        for _ in range(400):
            u.scene.step()
        names = set(u.agent.arm_joint_names)
        for j in robot.active_joints:
            if j.name in names:
                j.set_drive_properties(0.0, 0.0, force_limit=1000.0)   # torque control

    # ------------------------------------------------------------------ #
    def reset(self, spec: PegSpec, settle: float = 0.4, carry: float = 2.5) -> dict:
        self.spec = spec
        cr = self.crit
        self.env.reset(seed=spec.seed)
        self._grasp()
        Rh, hole_p = self._hole()
        self.axis, e1, e2 = Rh[:, 0], Rh[:, 1], Rh[:, 2]
        self.hole_p = hole_p
        self.W = np.column_stack([e1, e2, -self.axis])
        self.face = hole_p - self.half_len * self.axis     # the entrance, at its centre

        R0, p0, _, _ = self.ctl.tip_state()
        self.peg_in_tcp = R0.T @ (self.u.peg.pose.p[0].cpu().numpy() - p0)
        R_aim = _align(R0, R0[:, 0], self.axis)
        q = self.robot.get_qpos()[0].cpu().numpy()
        # Carried stiff: nothing is touching, and a soft arm would arrive somewhere else.
        self.ctl.reset_state(p0, R0, C.k_world(np.array([1500.0, 1500.0, 3000.0]), self.W),
                             q_rest=q, kr=80.0)
        head0 = self.head()
        self.start = (self.face - spec.standoff * self.axis
                      + spec.offset[0] * e1 + spec.offset[1] * e2)
        n = int(carry / self.dt)
        for i in range(n + int(settle / self.dt)):
            a = i / n
            self.ctl.R_d = _slerp(R0, R_aim, min(1.0, a / 0.7))
            head_t = head0 + _ease(a) * (self.start - head0)
            x_t = head_t - self.half_len * self._peg_axis() - self.ctl.R_d @ self.peg_in_tcp
            Vd = (x_t - self.ctl.x_d) / self.dt
            sp = float(np.linalg.norm(Vd))
            if sp > 0.10:
                Vd *= 0.10 / sp
            rec = self.ctl.compute(C.Case1Proposal(Vd), self.dt)
            self.u.scene.step()
            self.ctl.advance()
        # THE EPISODE STARTS HERE, and so does the tank: the carry is not the person's.
        R1, p1, _, _ = self.ctl.tip_state()
        self.ctl.reset_state(p1, self.ctl.R_d, C.k_world(self.K0, self.W),
                             q_rest=self.robot.get_qpos()[0].cpu().numpy(), kr=20.0)
        self.grip0 = float(np.linalg.norm(self.u.peg.pose.p[0].cpu().numpy() - p1))

        self.t = 0.0
        self.step_i = 0
        self.f_filt = np.zeros(3)
        self.f_slow = 0.0
        self.peak = 0.0
        self.peak_fast = 0.0
        self.touch_steps = 0
        self.in_band_steps = 0
        self.f_touch_sum = 0.0
        self.deepest = -spec.standoff
        self.caught_t = None
        self.jammed = False
        self.dropped = False
        self.ink_uv: list = []                       # the recorder's; a peg lays none
        d, at, ok = self.depth()
        self.last = self.ctl.compute(C.Case1Proposal(np.zeros(3)), self.dt)   # prime state
        self.last.update(t=0.0, f_raw=np.zeros(3), f_filt=np.zeros(3), f_n=0.0,
                         f_sensor_n=0.0, pen_down=False, depth=d, seated=ok, in_hole=False,
                         contact_uvh=np.array([at[1], at[2], -d]), n_ink=0,
                         x_d_next=self.ctl.x_d.copy(), K_next=self.ctl.K.copy())
        self.robot.set_qf(torch.zeros((1, self.nq)))
        return self.observe()

    # ------------------------------------------------------------------ #
    def step(self, prop: C.Case1Proposal) -> dict:
        """One physics step, with `sim.WritingSim.step`'s time convention."""
        cr = self.crit
        t0 = self.t
        rec = self.ctl.compute(prop, self.dt)
        self.u.scene.step()
        self.ctl.advance()

        # On the peg, from the box, in the world.  Its MAGNITUDE is the pressure here:
        # on the face the reaction is along the hole, in the hole it is across it, and a
        # jam is whichever of the two is climbing.
        f_raw = self.u.scene.get_pairwise_contact_forces(
            self.u.peg, self.u.box)[0].cpu().numpy()
        a = self.dt / (self.dt + 1.0 / (2 * np.pi * cr.force_lp_hz))
        self.f_filt = self.f_filt + a * (f_raw - self.f_filt)
        b = self.dt / (self.dt + 1.0 / (2 * np.pi * cr.pressure_lp_hz))
        self.f_slow += b * (float(np.linalg.norm(f_raw)) - self.f_slow)
        f_n = self.f_slow
        n = self.W[:, 2]
        f_sensor_n = float(self.f_filt @ n)

        d, at, ok = self.depth()
        touching = f_n >= cr.touch_force
        in_hole = d >= cr.catch
        if touching:
            self.touch_steps += 1
            self.f_touch_sum += f_n
            lo, hi = cr.force_band
            self.in_band_steps += int(lo <= f_n <= hi)
        if in_hole and self.caught_t is None:
            self.caught_t = t0
        self.deepest = max(self.deepest, d)
        self.peak = max(self.peak, f_n)
        self.peak_fast = max(self.peak_fast, float(np.linalg.norm(self.f_filt)))
        self.jammed |= f_n > cr.jam_force
        if self.step_i % 25 == 0:
            grip = float(np.linalg.norm(self.u.peg.pose.p[0].cpu().numpy() - rec["p"]))
            self.dropped |= abs(grip - self.grip0) > cr.slip

        self.t += self.dt
        self.step_i += 1
        # `pen_down` and `n_ink` are the recorder's names for "in contact" and "progress";
        # progress here is millimetres of the head past the face.
        rec.update(t=t0, f_raw=f_raw, f_filt=self.f_filt.copy(), f_n=f_n,
                   f_sensor_n=f_sensor_n, pen_down=bool(touching), depth=d, seated=ok,
                   in_hole=bool(in_hole), contact_uvh=np.array([at[1], at[2], -d]),
                   n_ink=int(round(1000.0 * max(0.0, d))),
                   x_d_next=self.ctl.x_d.copy(), K_next=self.ctl.K.copy())
        self.last = rec
        return rec

    # ------------------------------------------------------------------ #
    def k_diag(self, K: Array | None = None) -> Array:
        """Stiffness along (e1, e2, n): lateral, lateral, axial."""
        K = self.ctl.K if K is None else K
        return np.diag(self.W.T @ K @ self.W).copy()

    def images(self) -> dict:
        if not self.cameras:
            return {}
        data = self.u._get_obs_sensor_data()
        return {k: v["rgb"][0].cpu().numpy() for k, v in data.items() if "rgb" in v}

    def observe(self, images: bool = True) -> dict:
        """What a policy may see.  The hole's pose is NOT here."""
        q = self.robot.get_qpos()[0].cpu().numpy()
        qd = self.robot.get_qvel()[0].cpu().numpy()
        _, p, v, w = self.ctl.tip_state()
        obs = {
            "t": self.t,
            "qpos": q.astype(np.float32), "qvel": qd.astype(np.float32),
            "tau": self.last["tau"].astype(np.float32),
            "tcp_pos": p.astype(np.float32),
            "tcp_quat": self.ctl.tcp.pose.q[0].cpu().numpy().astype(np.float32),
            "tcp_vel": v.astype(np.float32), "tcp_angvel": w.astype(np.float32),
            "f_contact": self.f_filt.astype(np.float32),
            "x_d": self.ctl.x_d.astype(np.float32),
            "k_diag": self.k_diag().astype(np.float32),
            "tank_E": np.float32(self.ctl.E),
        }
        if images:
            for k, v in self.images().items():
                obs[f"rgb_{k}"] = v
        return obs

    def goal(self) -> dict:
        """What the operator was told: the frame the stiffness is named in, and where
        the peg started.  Not the hole -- finding it is the task."""
        return {"text": "peg", "belief_origin": self.start.copy(), "belief_R": self.W.copy(),
                "peg_half_len": np.float64(self.half_len),
                "peg_half_w": np.float64(self.half_w)}

    def privileged(self) -> dict:
        return {"hole_p": self.hole_p, "hole_R": np.column_stack(
                    [self.axis, self.W[:, 0], self.W[:, 1]]),
                "hole_half_w": self.hole_half_w, "offset": np.asarray(self.spec.offset),
                "standoff": self.spec.standoff}

    # ------------------------------------------------------------------ #
    def score(self) -> dict:
        cr = self.crit
        d, _, ok = self.depth()
        # A peg that went in without ever touching the face was never out of band.
        in_band = (self.in_band_steps / self.touch_steps) if self.touch_steps else 1.0
        # Seated is the head within 15 mm of the box's centre, and the box is as deep as
        # the peg is long, so that is this far past the entrance face.
        seat = self.half_len - 0.015
        res = dict(inserted=bool(ok), depth_mm=1000.0 * d, deepest_mm=1000.0 * self.deepest,
                   progress=float(np.clip(self.deepest / seat, 0.0, 1.0)),
                   caught_s=self.caught_t, in_band=float(in_band),
                   peak_force=float(self.peak), peak_force_fast=float(self.peak_fast),
                   contact_force=float(self.f_touch_sum / max(1, self.touch_steps)),
                   contact_s=self.touch_steps * self.dt, jammed=bool(self.jammed),
                   dropped=bool(self.dropped), t=self.t, tank_E=float(self.ctl.E))
        checks = {"inserted": bool(ok), "no_jam": not self.jammed,
                  "held": not self.dropped}
        res["checks"] = checks
        res["success"] = bool(all(checks.values()))
        res["fail_reason"] = ",".join(k for k, v in checks.items() if not v)
        return res

    def close(self) -> None:
        self.env.close()
