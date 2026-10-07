"""Flipping a box up against a wall, and pulling a door open, with a STICK, as
HAND-DRIVEN tasks under the Case 1 controller.

`../peg/peg_sim.py` is the model: a simulator that offers exactly what
`teleop.TeleopSession`, `interactive.VRHand` and `collect.Recorder` read off
`sim.WritingSim` -- `ctl`, `W`, `dt`, `step`, `last`, `observe`, `score`, `goal`,
`privileged` -- so those three run on it unchanged.

WHAT IS NEW HERE IS THE FRAME.  In writing, wiping and the peg, `W` is fixed for the
episode: the paper does not turn and neither does the hole.  Here the tip travels an
arc -- round the wall with the end of a box, round a hinge with a door's handle -- and
what the work asks for is named relative to that travel, not to the room:

    HARD MOVE   stiff ALONG the way the tip is going, so it goes where it is sent
    SOFT PUSH   soft ACROSS it, which is where the tip presses on the box, and where
                everything the hand has wrong about the arc ends up

So the stiffness frame follows the MOTION:

    W(t) = (u, v, n),   n = the direction the reference is being moved in

`n` is the way the reference has come over the last `TRAIL` of its own path, and it is
held while the reference is not getting anywhere.  It is the operator's own command, so
nothing privileged is in it and a real arm has it for free; a stick with a ball on the
end does not turn with the work, so the tool's frame would not do.

  (Over PATH, not over time.  The reference's velocity, smoothed over a tenth of a
  second, was tried first and was unstable against a door that had not unlatched yet:
  the stiff axis followed the velocity, the force fed back to the hand was along the
  stiff axis and pushed the reference round, and the frame chased its own tail -- 30 N
  swings on a door that had not moved.  A reference that is only jiggling in place has
  come from nowhere, and a direction taken over its path says so.)

The triple then reads

    K_t   ACROSS the motion -- the press, and the two ways the tip could be pushed off
    K_n   ALONG it
    K_R   the wrist

WHAT IS HIDDEN, per episode, and so has to be felt or seen:

    FLIP   how far from the wall the box lies and how it is turned, what it weighs,
           how easily it slides -- and where its near end really is, so WHEN the tip
           meets it
    DOOR   where the door is and which way it faces (so where the hinge is, and the
           arc), how far from the hinge the ball lands, how hard the latch holds, how
           strong the closer is -- and how far away the door's face really is, so WHEN
           the ball meets it

WHAT COUNTS AS DONE, and what counts as breaking something, is `ContactCriteria`: the
numbers are stated there once, before anything was measured against them.
"""
from __future__ import annotations

import collections
import dataclasses
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "writing"))
sys.path.insert(0, str(HERE))

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
import torch  # noqa: E402

import contact_scene as SC  # noqa: E402
import controller as C  # noqa: E402  writing/controller.py, as teleop.py imports it
from mani_skill.utils import sapien_utils  # noqa: E402
from transforms3d.quaternions import mat2quat  # noqa: E402

Array = np.ndarray

R_DOWN = np.diag([1.0, -1.0, -1.0])     # the pose the pen writes in: the stick points down
FWD = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])
# WebXR (right, up, back) -> world for an operator standing behind the arm: away from
# you is forward for the tip.
VR_AXES = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def rot_z(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def frame_along(n: Array) -> Array:
    """A right-handed (u, v, n) whose third axis is `n`.  Which u and v is of no
    consequence to the stiffness -- K_t is the same on both -- so they are picked to be
    steady: u is level whenever n is not near vertical."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    ref = UP if abs(n @ UP) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(n, u), n])


@dataclasses.dataclass
class ContactCriteria:
    touch_force: float = 0.8          # N on the tool: it is touching
    force_band: tuple = (1.0, 15.0)   # N, a working force; the console draws it
    # THE LANDING, where there is one (the door: the tip arrives, and then pulls).  The
    # hardest the tool may be pressing over the first `land_window` seconds of touching,
    # before it has taken hold.
    land_force: float = 8.0           # N
    land_window: float = 0.35         # s
    # WHILE WORKING.  Force ACROSS the motion: on the box it is the press, and past this
    # the box is being crushed into the wall; on the door it is the arm fighting the arc.
    across_force: float = 20.0        # N
    moment: float = 4.0               # Nm at the wrist: the arm fighting a turn
    overload: float = 45.0            # N sustained, in any direction
    collision_force: float = 1.0      # N, any link but the tool on the work
    force_lp_hz: float = 25.0         # the force sensor: observations and logs
    pressure_lp_hz: float = 5.0       # sustained force: the band and the overload


class ContactSim:
    """What the two tasks share.  A subclass says what the work is and how it scores."""
    ENV_ID = ""
    TASK = ""
    TOOL_LINK = "stick"
    START_KR = 3.0
    R_START = R_DOWN       # how the stick is held, for the whole episode
    LANDS = True           # is there a landing to judge?  see ContactCriteria
    # WHEN THE WORK BEGINS, once the tip has met it -- which is when the protocol's WORK
    # row may come in.  "turn": when the tip is travelling across the way it came (the
    # box: it leans, then carries).  "dwell": when the landing is over (the door: it
    # arrives, and then pushes the way it was already going, so there is no turn to see).
    WORK_ON = "turn"
    # THE HAND'S OWN FRAME, which is not the stiffness frame: its third column is the
    # work's outward normal AS THE TIP COMES AT IT, fixed for the episode.  A trigger
    # (or the press key) pushes against that column, so "press" goes into the work --
    # read off `W`, which follows the motion, it would push back along whatever way
    # the tip happened to be going.  The first two are what a keyboard's I/J/K/L move
    # along.  Each task says its own.
    HAND_AXES = VR_AXES
    TRAIL = 0.005          # m of the reference's own path its direction is taken over
    TRAIL_STEP = 0.001     # m between the points kept along it
    TRAIL_GONE = 0.5       # of TRAIL: the least it must have got, end to end, to count

    def __init__(self, image_size: int = 128, cameras: bool = True,
                 gains: C.Case1Gains | None = None,
                 criteria: ContactCriteria | None = None, render_mode: str | None = None):
        self.cameras = cameras
        self.crit = criteria or ContactCriteria()
        # ONE env for the session and never reconfigured: reconfiguring rebuilds the
        # robot under the controller holding it, and closes the operator's window.
        self.env = gym.make(self.ENV_ID, num_envs=1, sim_backend="cpu",
                            obs_mode="rgb" if cameras else "state",
                            render_mode=render_mode, image_size=image_size,
                            reconfiguration_freq=0)
        self.env.reset(seed=0)
        self.u = self.env.unwrapped
        self.dt = 1.0 / self.u.sim_freq
        self.robot = self.u.agent.robot
        self.nq = len(self.robot.active_joints)
        self.ctl = C.Case1Controller(self.robot, gains or C.Case1Gains(),
                                     self.u.agent.urdf_path)
        self.qlim = self.robot.get_qlimits()[0].cpu().numpy()
        self._q_like = None        # the arm's configuration last time: the next start's seed
        links = self.robot.get_links()
        self.tool = sapien_utils.get_obj_by_name(links, self.TOOL_LINK)
        self.others = [sapien_utils.get_obj_by_name(links, n)
                       for n in ("panda_hand", "panda_link7", "panda_link6", "panda_link5")]
        # A rubber ball: it has to be able to carry the end of a box up by friction.
        grip = sapien.physx.PhysxMaterial(1.5, 1.5, 0.0)
        for s in self.tool._objs[0].get_collision_shapes():
            s.set_physical_material(grip)
        self.W = frame_along(FWD)
        self.K0 = np.array([1000.0, 1000.0, 1000.0])
        self.last: dict = {}

    # ---- the start, common to both ------------------------------------------
    def _begin(self, p0: Array, along: Array) -> None:
        """Put the tip at `p0`, at rest, about to move `along`, with the books open."""
        q = self._start_q(np.asarray(p0, float))
        self.robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))
        self.robot.set_qvel(torch.zeros((1, self.nq)))
        self.ctl.disable_joint_drives()
        R, p, _, _ = self.ctl.tip_state()
        self.n = np.asarray(along, float) / np.linalg.norm(along)
        self.W0 = frame_along(self.n)
        self.W = self.W0.copy()
        self._trail = collections.deque([p.copy()],
                                        maxlen=int(round(self.TRAIL / self.TRAIL_STEP)) + 1)
        self.ctl.reset_state(p, R, C.k_world(self.K0, self.W), q_rest=q, kr=self.START_KR)
        self.start = p.copy()
        self.t = 0.0
        self.step_i = 0
        self.f_filt = np.zeros(3)
        self.f_slow = 0.0
        self.peak = 0.0
        self.peak_fast = 0.0
        self.peak_across = 0.0
        self.peak_moment = 0.0
        self.land_peak = 0.0
        self.t_touch = None
        self.touch_steps = 0
        self.in_band_steps = 0
        self.f_touch_sum = 0.0
        self.collided = False
        self.engaged = False
        self.ink_uv: list = []                # the recorder's; nothing is laid here
        self.last = self.ctl.compute(C.Case1Proposal(np.zeros(3)), self.dt)   # prime state
        self.last.update(t=0.0, f_raw=np.zeros(3), f_filt=np.zeros(3), f_n=0.0,
                         f_sensor_n=0.0, pen_down=False, engaged=False, across=0.0,
                         moment=0.0, contact_uvh=np.zeros(3), n_ink=0, W=self.W.copy(),
                         x_d_next=self.ctl.x_d.copy(), K_next=self.ctl.K.copy(),
                         **self._task_fields())
        self.robot.set_qf(torch.zeros((1, self.nq)))

    def _room(self, q: Array) -> float:
        """How far the nearest joint is from its stop, rad."""
        return float(np.min(np.minimum(q - self.qlim[:, 0], self.qlim[:, 1] - q)))

    def _start_q(self, p0: Array, room: float = 0.25) -> Array:
        """Joint angles that put the tip at `p0` in this task's pose, well inside the
        arm's limits.

        The solver takes whatever is nearest its seed and does not know about joint
        stops, and a pose other than the pen's has no keyframe to seed it from.  So:
        the configuration the last episode started in, then the rest pose, and failing
        both a search from seeds spread over the joint range -- the same seeds every
        time, so the arm starts an episode the way it started the last one.
        """
        rest = self.u.agent.keyframes["rest"].qpos
        rest = np.asarray(rest.cpu() if hasattr(rest, "cpu") else rest, float).reshape(-1)
        best = None
        for seed in ([self._q_like] if self._q_like is not None else []) + [rest]:
            q, ok = self.ctl.ik(p0, self.R_START, seed)
            if ok and self._room(q) >= room:
                best = q
                break
        if best is None:
            rng = np.random.default_rng(0)
            for _ in range(80):
                seed = self.qlim[:, 0] + rng.uniform(0.15, 0.85, self.nq) * (
                    self.qlim[:, 1] - self.qlim[:, 0])
                q, ok = self.ctl.ik(p0, self.R_START, seed)
                if ok and (best is None or self._room(q) > self._room(best)):
                    best = q
            if best is None or self._room(best) <= 0.0:
                raise RuntimeError(f"the start pose {np.round(p0, 3)} is out of the "
                                   f"arm's reach in this task's pose")
        self._q_like = np.asarray(best, float).copy()
        return self._q_like

    # ---- hooks ----------------------------------------------------------------
    def _force(self, rec) -> Array:
        """The force on the tool from the work, world frame."""
        raise NotImplementedError

    def _work(self):
        """The body any other link must not touch."""
        raise NotImplementedError

    def _task_step(self, rec, touching: bool) -> None:
        pass

    def _over(self) -> bool:
        """The work is done: nothing after this is held against the stiffness."""
        return False

    def _task_fields(self) -> dict:
        """`progress` in [0, 1], and whatever else the task wants in every record."""
        return dict(progress=0.0)

    # ------------------------------------------------------------------ #
    def step(self, prop: C.Case1Proposal) -> dict:
        """One physics step, with `sim.WritingSim.step`'s time convention."""
        cr = self.crit
        t0 = self.t
        rec = self.ctl.compute(prop, self.dt)
        self.u.scene.step()
        self.ctl.advance()
        # WHICH WAY THE HAND IS GOING, from where the reference has come: the frame the
        # NEXT stiffness request is named in.
        if np.linalg.norm(self.ctl.x_d - self._trail[-1]) >= self.TRAIL_STEP:
            self._trail.append(self.ctl.x_d.copy())
            come = self._trail[-1] - self._trail[0]
            far = float(np.linalg.norm(come))
            if len(self._trail) == self._trail.maxlen and far >= self.TRAIL_GONE * self.TRAIL:
                self.n = come / far
                self.W = frame_along(self.n)
        n = self.n

        f_raw = self._force(rec)
        a = self.dt / (self.dt + 1.0 / (2 * np.pi * cr.force_lp_hz))
        self.f_filt = self.f_filt + a * (f_raw - self.f_filt)
        b = self.dt / (self.dt + 1.0 / (2 * np.pi * cr.pressure_lp_hz))
        self.f_slow += b * (float(np.linalg.norm(f_raw)) - self.f_slow)
        f_n = self.f_slow
        fast = float(np.linalg.norm(self.f_filt))
        across = float(np.linalg.norm(self.f_filt - (self.f_filt @ n) * n))
        moment = float(np.linalg.norm(self.ctl.elastic_moment(rec["R"])))

        touching = bool(f_n >= cr.touch_force or self.engaged)
        if touching:
            if self.t_touch is None:
                self.t_touch = t0
            self.touch_steps += 1
            self.f_touch_sum += f_n
            lo, hi = cr.force_band
            self.in_band_steps += int(lo <= f_n <= hi)
        # The SUSTAINED force, as the band and the overload are: what a landing does
        # wrong is go on pressing after it has arrived, and that is the spring between
        # the tool and where it was told to be -- which the stiffness sets.  The first
        # few milliseconds of two rigid things meeting are set by speed and mass, are in
        # `peak_force_fast`, and no stiffness chosen here changes them.  And only until
        # the tool has TAKEN HOLD: after that the hand is working, and what happens is
        # the work's, judged by the overload.
        if (self.LANDS and self.t_touch is not None and not self.engaged
                and t0 - self.t_touch <= cr.land_window):
            self.land_peak = max(self.land_peak, f_n)
        self.peak = max(self.peak, f_n)
        self.peak_fast = max(self.peak_fast, fast)
        # ON THE WORK, AND UNTIL IT IS DONE.  Once the box is up or the door is open
        # the tool comes away, and what the arm does to itself then is not something it
        # did to the work.
        if touching and not self._over():
            self.peak_across = max(self.peak_across, across)
            self.peak_moment = max(self.peak_moment, moment)
        if self.step_i % 10 == 0:
            for lk in self.others:
                fo = self.u.scene.get_pairwise_contact_forces(lk, self._work())[0].cpu().numpy()
                self.collided |= float(np.linalg.norm(fo)) > cr.collision_force
        self._task_step(rec, touching)

        self.t += self.dt
        self.step_i += 1
        fields = self._task_fields()
        rec.update(t=t0, f_raw=f_raw, f_filt=self.f_filt.copy(), f_n=f_n,
                   f_sensor_n=float(self.f_filt @ n), pen_down=touching,
                   engaged=bool(self.engaged), across=across, moment=moment,
                   contact_uvh=np.zeros(3), n_ink=int(round(1000.0 * fields["progress"])),
                   W=self.W.copy(), x_d_next=self.ctl.x_d.copy(),
                   K_next=self.ctl.K.copy(), **fields)
        self.last = rec
        return rec

    # ------------------------------------------------------------------ #
    def k_diag(self, K: Array | None = None) -> Array:
        """Stiffness (across, across, along the motion), as the frame stands now."""
        K = self.ctl.K if K is None else K
        return np.diag(self.W.T @ K @ self.W).copy()

    def images(self) -> dict:
        if not self.cameras:
            return {}
        data = self.u._get_obs_sensor_data()
        return {k: v["rgb"][0].cpu().numpy() for k, v in data.items() if "rgb" in v}

    def observe(self, images: bool = True) -> dict:
        """What a policy may see.  Nothing about where the work is, or what it weighs."""
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
        """What the operator was told: where the tip started and the frame the
        stiffness is named in at that moment.  Not where the work is."""
        return {"text": self.TASK, "belief_origin": self.start.copy(),
                "belief_R": self.W0.copy()}

    def _common_score(self) -> tuple[dict, dict]:
        cr = self.crit
        in_band = (self.in_band_steps / self.touch_steps) if self.touch_steps else 1.0
        res = dict(in_band=float(in_band), peak_force=float(self.peak),
                   peak_force_fast=float(self.peak_fast), land_peak=float(self.land_peak),
                   peak_across=float(self.peak_across), peak_moment=float(self.peak_moment),
                   contact_force=float(self.f_touch_sum / max(1, self.touch_steps)),
                   contact_s=self.touch_steps * self.dt, collided=bool(self.collided),
                   t=self.t, tank_E=float(self.ctl.E))
        checks = {"soft_landing": self.land_peak <= cr.land_force,
                  "no_overload": (self.peak <= cr.overload
                                  and self.peak_across <= cr.across_force
                                  and self.peak_moment <= cr.moment),
                  "no_collision": not self.collided}
        return res, checks

    @staticmethod
    def _finish(res: dict, checks: dict) -> dict:
        res["checks"] = checks
        res["success"] = bool(all(checks.values()))
        res["fail_reason"] = ",".join(k for k, v in checks.items() if not v)
        return res

    def close(self) -> None:
        self.env.close()


# --------------------------------------------------------------------------- #
#                         FLIP:  robot --- box --- wall
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FlipSpec:
    """Everything that varies between episodes.  All of it is hidden from the hand
    except where it was TOLD the near end is, which is `belief_err` out."""
    seed: int = 0
    gap: float = 0.0                  # m between the box's far end and the wall
    y: float = 0.0                    # m off the arm's centre line
    yaw: float = 0.0                  # rad
    mass: float = 0.4                 # kg
    mu: float = 0.35                  # the desk and the wall
    press_h: float = 0.70             # of the box's height: where on the near end to press
    belief_err: float = 0.0           # m along the approach: told minus true
    standoff: float = 0.050           # m short of the TOLD near end, where the tip starts
    time_limit: float = 30.0

    @staticmethod
    def sample(seed: int) -> "FlipSpec":
        rng = np.random.default_rng(seed)
        return FlipSpec(seed=seed,
                        gap=float(rng.uniform(0.0, 0.015)),
                        y=float(rng.uniform(-0.03, 0.03)),
                        yaw=float(np.deg2rad(rng.uniform(-5.0, 5.0))),
                        mass=float(rng.uniform(0.30, 0.70)),
                        mu=float(rng.uniform(0.20, 0.50)),
                        # WHEN the tip meets the box: up to 15 mm sooner or later than told
                        belief_err=float(rng.uniform(-0.015, 0.015)))


class FlipSim(ContactSim):
    ENV_ID = "TeleopFlip-v1"
    TASK = "flip"
    LANDS = False          # pressing on the box IS the work: there is no landing apart
    UPRIGHT = np.radians(80.0)        # standing on its far end, give or take a lean

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body = self.u.box._objs[0].find_component_by_type(
            sapien.physx.PhysxRigidDynamicComponent)
        self._m0 = float(self.body.mass)
        self._I0 = np.asarray(self.body.inertia, dtype=float).copy()

    def _work(self):
        return self.u.box

    def _box(self) -> tuple[Array, Array]:
        T = self.u.box.pose.to_transformation_matrix()[0].cpu().numpy()
        return T[:3, :3], T[:3, 3]

    def rise(self) -> float:
        """How far the near end has been carried up: 0 lying, 90 degrees standing."""
        R = self._box()[0]
        return float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))

    def face_inward(self) -> Array:
        """Into the near end: the way the tip presses."""
        return self._box()[0][:, 0]

    def face_up(self) -> Array:
        """Along the near end, the way it has to be carried: up, then over to the wall."""
        return self._box()[0][:, 2]

    def reset(self, spec: FlipSpec) -> dict:
        self.spec = spec
        self.env.reset(seed=spec.seed)
        h = self.u.BOX_HALF
        Rb = rot_z(spec.yaw)
        # turned, its far corners are not level: back it off so neither is in the wall
        cx = self.u.WALL_X - spec.gap - h[0] * np.cos(spec.yaw) - h[1] * abs(np.sin(spec.yaw))
        centre = np.array([cx, spec.y, SC.TABLE_TOP + h[2]])
        self.u.box.set_pose(sapien.Pose(p=centre, q=mat2quat(Rb)))
        self.u.box.set_linear_velocity(torch.zeros(1, 3))
        self.u.box.set_angular_velocity(torch.zeros(1, 3))
        self.body.set_mass(float(spec.mass))
        self.body.set_inertia(self._I0 * spec.mass / self._m0)
        for f in (self.u.still_material.set_static_friction,
                  self.u.still_material.set_dynamic_friction):
            f(float(spec.mu))
        # where the ball should press: on the near end, `press_h` of the way up it
        self.face = centre - h[0] * Rb[:, 0] + (2 * spec.press_h - 1.0) * h[2] * UP
        ball = self.face - SC.BALL_R * Rb[:, 0]
        self.tip_on = ball - SC.BALL_R * UP               # the TCP is the ball's lowest point
        self.told = self.tip_on + spec.belief_err * FWD
        self._begin(self.told - spec.standoff * FWD, FWD)
        self.max_rise = 0.0
        self.up_t = None
        return self.observe()

    def _force(self, rec) -> Array:
        return self.u.scene.get_pairwise_contact_forces(
            self.tool, self.u.box)[0].cpu().numpy()

    def _task_step(self, rec, touching: bool) -> None:
        r = self.rise()
        self.max_rise = max(self.max_rise, r)
        if r >= self.UPRIGHT and self.up_t is None:
            self.up_t = self.t

    def _over(self) -> bool:
        return self.up_t is not None

    def _task_fields(self) -> dict:
        r = self.rise() if hasattr(self, "spec") else 0.0
        return dict(progress=float(np.clip(r / self.UPRIGHT, 0.0, 1.0)), rise=r,
                    upright=bool(r >= self.UPRIGHT))

    def privileged(self) -> dict:
        s = self.spec
        return {"gap": s.gap, "y": s.y, "yaw": s.yaw, "mass": s.mass, "mu": s.mu,
                "belief_err": s.belief_err, "face": self.face}

    def score(self) -> dict:
        res, checks = self._common_score()
        r = self.rise()
        res.update(upright=bool(r >= self.UPRIGHT), rise_deg=float(np.degrees(r)),
                   max_rise_deg=float(np.degrees(self.max_rise)),
                   progress=float(np.clip(self.max_rise / self.UPRIGHT, 0.0, 1.0)),
                   up_s=self.up_t)
        # STANDING WHEN IT IS OVER, not merely stood up at some point: a box carried to
        # 80 degrees and dropped has been dropped.
        checks = {"upright": bool(r >= self.UPRIGHT), **checks}
        return self._finish(res, checks)


# --------------------------------------------------------------------------- #
#                           DOOR:  robot --- door
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DoorSpec:
    seed: int = 0
    hinge_xy: tuple = (0.50, 0.16)    # m
    yaw: float = 0.0                  # rad: which way the closed door faces
    push_r: float = 0.20              # m from the hinge: where the ball lands
    closer: float = 1.0               # Nm/rad pushing it shut
    drag: float = 0.25                # Nm s/rad in the hinge
    latch: float = 8.0                # N of push before it lets go
    belief_err: float = 0.0           # m along the approach: told minus true
    standoff: float = 0.050           # m short of the TOLD face, where the tip starts
    target: float = float(np.radians(50.0))
    time_limit: float = 30.0

    @staticmethod
    def sample(seed: int) -> "DoorSpec":
        rng = np.random.default_rng(seed)
        return DoorSpec(seed=seed,
                        hinge_xy=(float(rng.uniform(0.48, 0.52)), float(rng.uniform(0.14, 0.18))),
                        yaw=float(np.deg2rad(rng.uniform(-6.0, 6.0))),
                        push_r=float(rng.uniform(0.16, 0.24)),
                        closer=float(rng.uniform(0.6, 1.5)),
                        drag=float(rng.uniform(0.15, 0.35)),
                        latch=float(rng.uniform(5.0, 11.0)),
                        # WHEN the ball meets the door: 12 mm nearer or farther than told
                        belief_err=float(rng.uniform(-0.012, 0.012)))


class DoorSim(ContactSim):
    """A door pushed open with the ball of a stick.

    WHAT THE HAND HAS TO DO: bring the ball up to the door's face, on the push plate (it
    does not know how far the face is); push, until the latch lets go at a force nobody
    told it; and keep pushing as the door swings away, to 50 degrees, and hold it there
    against the closer.

    NOTHING HOLDS THE BALL TO THE DOOR.  The spot it is on swings round the hinge, so
    it moves sideways as well as away; the ball goes with it only as far as friction
    carries it and the arm lets it.  Pushed straight ahead by an arm that will not give
    sideways, the ball skates along the face to the free edge and off it, and the
    closer shuts the door behind the stick.
    """
    ENV_ID = "TeleopDoor-v1"
    TASK = "door"
    WORK_ON = "dwell"
    LATCHED = 0.004        # rad of play while the latch holds
    LATCH_S = 0.05         # s the push has to exceed the latch for

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hinge = self.u.hinge._objs[0]

    def _work(self):
        return self.u.panel

    def _panel(self) -> tuple[Array, Array]:
        T = self.u.panel.pose.to_transformation_matrix()[0].cpu().numpy()
        return T[:3, :3], T[:3, 3]

    def into(self) -> Array:
        """Into the door's face, away from the arm: the direction of the push."""
        return self._panel()[0][:, 0]

    def angle(self) -> float:
        return float(self.u.door.get_qpos()[0, 0])

    def reset(self, spec: DoorSpec) -> dict:
        self.spec = spec
        self.env.reset(seed=spec.seed)
        z = float(self.u.HINGE_AT[2])
        Rd = rot_z(spec.yaw)
        self.u.door.set_pose(sapien.Pose(p=[spec.hinge_xy[0], spec.hinge_xy[1], z],
                                         q=mat2quat(Rd)))
        self.u.door.set_qpos(torch.zeros((1, 1)))
        self.u.door.set_qvel(torch.zeros((1, 1)))
        # The closer: a spring towards shut, and a drag so a door that is let go does
        # not slam.  NOT the joint's own `friction`: in this engine that is a
        # coefficient on the load through the hinge, not a torque -- at 0.18 a door let
        # go at 50 degrees stayed there against a 1.3 Nm/rad closer.
        self.hinge.set_friction(0.0)
        self.hinge.set_drive_properties(float(spec.closer), float(spec.drag), 1e6, "force")
        self.hinge.set_drive_target(0.0)
        self._limit(self.LATCHED)
        hinge = np.array([spec.hinge_xy[0], spec.hinge_xy[1], z])
        # where the ball should land: on the face, `push_r` from the hinge, half way up
        self.face = hinge + Rd @ np.array([-self.u.PANEL_T / 2, -spec.push_r, 0.0])
        self.hinge_p = hinge
        ball = self.face - SC.BALL_R * Rd[:, 0]
        self.tip_on = ball - SC.BALL_R * UP               # the TCP is the ball's lowest point
        self.told = self.tip_on + spec.belief_err * FWD
        self._begin(self.told - spec.standoff * FWD, FWD)
        self.unlatched = False
        self._push = 0.0
        self.max_angle = 0.0
        self.open_t = None
        self.lost = False            # the ball came off a door that was on its way
        return self.observe()

    def _limit(self, hi: float) -> None:
        self.u.hinge.set_limits(np.array([[0.0, float(hi)]], dtype=np.float32))

    def _force(self, rec) -> Array:
        return self.u.scene.get_pairwise_contact_forces(
            self.tool, self.u.panel)[0].cpu().numpy()

    def _task_step(self, rec, touching: bool) -> None:
        # the door pushing back on the ball is the ball pushing the door
        push = -float(self.f_filt @ self.into())
        if not self.unlatched:
            self._push = self._push + self.dt if push >= self.spec.latch else 0.0
            if self._push >= self.LATCH_S:
                self.unlatched = True
                self._limit(self.u.ANGLE_MAX)
        self.engaged = self.unlatched             # from here the door goes with the stick
        a = self.angle()
        self.max_angle = max(self.max_angle, a)
        if a >= self.spec.target and self.open_t is None:
            self.open_t = self.t
        if self.unlatched and self.open_t is None and not touching and a > np.radians(3.0):
            self.lost = True

    def _over(self) -> bool:
        return self.open_t is not None

    def _task_fields(self) -> dict:
        a = self.angle() if hasattr(self, "spec") else 0.0
        tgt = self.spec.target if hasattr(self, "spec") else 1.0
        return dict(progress=float(np.clip(a / tgt, 0.0, 1.0)), angle=a,
                    unlatched=bool(getattr(self, "unlatched", False)))

    def privileged(self) -> dict:
        s = self.spec
        return {"hinge_xy": np.asarray(s.hinge_xy), "yaw": s.yaw, "push_r": s.push_r,
                "closer": s.closer, "drag": s.drag, "latch": s.latch,
                "belief_err": s.belief_err, "face": self.face}

    def score(self) -> dict:
        res, checks = self._common_score()
        a = self.angle()
        res.update(opened=bool(a >= self.spec.target), angle_deg=float(np.degrees(a)),
                   max_angle_deg=float(np.degrees(self.max_angle)),
                   progress=float(np.clip(self.max_angle / self.spec.target, 0.0, 1.0)),
                   unlatched=bool(self.unlatched), lost=bool(self.lost),
                   open_s=self.open_t)
        # OPEN WHEN IT IS OVER: the closer is pushing back, so a door that was opened
        # and let go is a door that is shut.
        checks = {"opened": bool(a >= self.spec.target - np.radians(3.0)), **checks}
        return self._finish(res, checks)


SIMS = {"flip": (FlipSim, FlipSpec), "door": (DoorSim, DoorSpec)}
