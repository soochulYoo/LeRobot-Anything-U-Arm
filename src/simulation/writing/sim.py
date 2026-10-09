"""WritingSim: one episode of writing, stepped at the physics rate.

Both ways of driving the pen go through here and nowhere else:

    teleop.py   human -> master -> (x_d, K) request -> WritingSim.step
    evaluate.py policy ---------> (x_d, K) request -> WritingSim.step

so a policy is evaluated in exactly the scene, controller, ink rule and
success check its demonstrations were collected in.

INK IS A FUNCTION OF FORCE.  Every physics step the pen/paper contact force is
read and filtered twice:

    sensor    25 Hz   what a force sensor reports; observed by a policy, logged
    pressure   5 Hz   the SUSTAINED force; ink, pen-down, the force band and
                      tearing are all judged on this one signal

Two filters because PhysX contact is rigid.  A landing at 15 mm/s delivers
0.14 N s in ONE 2 ms step (68 N raw, 16 N at 25 Hz), and a light touch
chatters, each re-contact a one-step impulse.  Judged on the sensor signal, a
2 mm/s touch at 0.3 N laid 8 dots of ink and a gentle landing "tore" the paper.
Neither is what the words mean.  On the pressure signal a landing reads ~4 N,
a light touch leaves no ink, and slamming the pen down at 5 cm/s still tears.
Ink lags the pen by the filter's 32 ms, well under a millimetre at writing
speed.

SUCCESS (Criteria) -- all of:
    coverage   >= 90%   of the target's points have ink within `tol`
    precision  >= 90%   of the ink lies within `tol` of the target
                        (a pen that is not lifted between strokes fails here)
    in_band    >= 85%   of pen-down time has pressure inside `force_band`
    no tear             pressure never exceeded `tear_force`
    no collision        nothing but the pen touched the paper
"""
from __future__ import annotations

import os

import dataclasses
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402

import controller as C  # noqa: E402
import glyphs as G  # noqa: E402
import scene as S  # noqa: E402

Array = np.ndarray

# pen straight down: tool z along -world z
R_PEN_DOWN = np.diag([1.0, -1.0, -1.0])


@dataclass
class TaskSpec:
    """Everything that varies between episodes.  The canvas pose and friction
    are HIDDEN from the operator and from a policy (see `privileged()`)."""
    text: str = "HI"
    letter_height: float = 0.035
    slant: float = 0.0
    offset_uv: tuple = (0.0, 0.0)       # text centre on the paper, m
    canvas_dz: float = 0.0              # paper height error, m
    tilt_x: float = 0.0                 # rad
    tilt_y: float = 0.0
    friction: float = 0.3
    show_template: bool = True
    time_limit: float = 90.0
    seed: int = 0

    @staticmethod
    def sample(seed: int, vocab: list[str] | None = None, max_len: int = 3,
               dz: float = 0.004, tilt_deg: float = 5.0,
               mu: tuple = (0.2, 0.5)) -> "TaskSpec":
        """`mu` is the paper's friction range, and it is the one hidden variable
        a camera cannot see: measured out of fold, vision explains R^2 = -0.20 and
        0.00 of the two TANGENTIAL force axes against 0.87 of the normal one
        (policy/SPRING.md).  The default (0.2, 0.5) is narrow enough that a single
        xy stiffness covers all of it, which is why protocol.py fixes xy at LOW
        and why every structure's `acc xy` sits at 98% with nothing to learn.
        Widening it is what gives the stiffness a job -- see protocol.py --mu.
        """
        rng = np.random.default_rng(seed)
        if vocab is None:
            chars = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]
            shapes = ["<circle>", "<square>", "<triangle>", "<star>", "<spiral>", "<wave>"]
            if rng.random() < 0.2:
                text = str(rng.choice(shapes))
            else:
                n = int(rng.integers(1, max_len + 1))
                text = "".join(rng.choice(chars, size=n))
        else:
            text = str(rng.choice(vocab))
        n_tok = max(1, len(G.tokenize(text)))
        h = float(rng.uniform(0.030, 0.040))
        width = n_tok * h * (4 + 1.6) / 6
        return TaskSpec(
            text=text, letter_height=h,
            slant=float(rng.uniform(-0.15, 0.15)),
            offset_uv=(float(rng.uniform(-0.02, 0.02) if width < 0.12 else 0.0),
                       float(rng.uniform(-0.02, 0.02))),
            canvas_dz=float(rng.uniform(-dz, dz)),
            tilt_x=float(np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))),
            tilt_y=float(np.deg2rad(rng.uniform(-tilt_deg, tilt_deg))),
            friction=float(rng.uniform(*mu)),
            seed=seed)


@dataclass
class Criteria:
    tol: float = 0.003                  # m, for both coverage and precision
    min_coverage: float = 0.90
    min_precision: float = 0.90
    ink_force: float = 0.8              # N, below this the pen skates without ink
    tear_force: float = 12.0            # N, above this the paper tears
    force_band: tuple = (1.0, 6.0)      # N, a good writing force
    min_in_band: float = 0.85
    ink_spacing: float = 0.0008         # m between deposited ink dots
    force_lp_hz: float = 25.0           # the force sensor: observations and logs
    pressure_lp_hz: float = 5.0         # sustained force: ink, band, tear (see top)
    collision_force: float = 1.0        # N, any non-pen link on the paper


class WritingSim:
    # The tool, so ../wiping can be this simulator with an eraser on the wrist.
    ENV_ID = "TeleopWriting-v1"
    TOOL_LINK = "pen"

    @staticmethod
    def tool_urdf() -> str:
        return S.PandaPen.urdf_path

    def __init__(self, image_size: int = 128, wrist_camera: bool = True,
                 cameras: bool = True, gains: C.Case1Gains | None = None,
                 criteria: Criteria | None = None, render_mode: str | None = None,
                 sim_freq: int | None = None):
        # sim_freq overrides scene.py's 500 Hz.  speedup.py needs it: the
        # time-scaling symmetry is a CONTINUOUS-time identity, and the discrete
        # loop is conjugate step for step only when a c-times faster execution
        # also steps c times faster, so one step covers the same phase.
        self.cameras = cameras
        extra = {} if sim_freq is None else dict(
            sim_config=dict(sim_freq=int(sim_freq), control_freq=int(sim_freq)))
        # RENDER DEVICE WITH AN INDEX, because ManiSkill asks for one without.  Its
        # default render_backend is "gpu", which it maps to "sapien_cuda" and then
        # builds `sapien.Device("cuda")` when no device id was given -- and on this
        # cluster that raises `failed to find device "cuda"` while `sapien.Device(
        # "cuda:0")` returns the RTX 3090 happily.  ManiSkill catches the RuntimeError
        # and falls back to CPU rendering, which is not a degraded mode but an
        # unusable one: two tasks rasterising 128x128 cameras on llvmpipe ran eight
        # hours at 350% CPU and produced no episodes at all.  Passing "cuda:0" goes
        # through `parse_backend_device_id`, which splits it, so the device id arrives.
        render_backend = os.environ.get("MS_RENDER_BACKEND")
        if render_backend is None:
            # ...but only where ManiSkill parses an index.  The lab PC's installed
            # 3.0.0b21 looks the name up as given and raised KeyError: 'cuda:0' before
            # any scene was built; its "gpu" finds the device without one.
            from mani_skill.envs.utils.system import backend as _msb
            render_backend = ("cuda:0" if hasattr(_msb, "parse_backend_device_id")
                              else "gpu")
        self.env = gym.make(self.ENV_ID, num_envs=1, sim_backend="cpu",
                            render_backend=render_backend,
                            obs_mode="rgb" if cameras else "state",
                            render_mode=render_mode,
                            image_size=image_size, wrist_camera=wrist_camera, **extra)
        self.env.reset(seed=0)
        self.u = self.env.unwrapped
        self.dt = 1.0 / self.u.sim_freq
        self.crit = criteria or Criteria()
        self.robot = self.u.agent.robot
        self.ctl = C.Case1Controller(self.robot, gains or C.Case1Gains(), self.tool_urdf())
        links = self.robot.get_links()
        self.pen = sapien_utils.get_obj_by_name(links, self.TOOL_LINK)
        self.others = [sapien_utils.get_obj_by_name(links, n)
                       for n in ("panda_hand", "panda_link7", "panda_link6", "panda_link5")]
        # What the operator and any policy BELIEVE: the nominal, untilted paper.
        self.belief = S.CanvasFrame(self.u.CANVAS_CENTER)
        self.W = self.belief.R                 # writing frame (u, v, n), stiffness axes

    # ------------------------------------------------------------------ #
    def reset(self, spec: TaskSpec, hover: float = 0.040) -> dict:
        self.spec = spec
        self.env.reset(seed=spec.seed)
        c = self.u.CANVAS_CENTER + np.array([0.0, 0.0, spec.canvas_dz])
        self.frame = S.CanvasFrame(c, spec.tilt_x, spec.tilt_y)
        self.u.place_canvas(self.frame, spec.friction)
        self.u.hide_all_dots()

        tgt = G.layout(spec.text, spec.letter_height, slant=spec.slant)
        off = np.asarray(spec.offset_uv, dtype=float)
        self.target = dataclasses.replace(tgt, strokes=[s + off for s in tgt.strokes])
        self.target_pts = self.target.points
        if spec.show_template:
            self.u.show_template(np.concatenate(
                [G.resample(s, 0.0012) for s in self.target.strokes]))

        # Start hovering over the middle of the text, pen down, at rest.
        p0 = self.belief.to_world(np.r_[off, hover])
        q_init = self.u.agent.keyframes["rest"].qpos
        q, ok = self.ctl.ik(p0, R_PEN_DOWN, q_init)
        if not ok:
            raise RuntimeError(f"no IK solution for the start pose {p0}")
        self.robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))
        self.robot.set_qvel(torch.zeros((1, len(q))))
        self.ctl.disable_joint_drives()
        self.K0 = np.array([1000.0, 1000.0, 1000.0])
        # Seed the reference at where the tip actually IS, not at the IK
        # target: the IK residual (~0.1 mm) would otherwise show up as the arm
        # settling in the first half second of every episode.
        _, p_act, _, _ = self.ctl.tip_state()
        self.ctl.reset_state(p_act, R_PEN_DOWN, C.k_world(self.K0, self.W), q_rest=q)

        self.t = 0.0
        self.step_i = 0
        self.f_filt = np.zeros(3)
        self.f_slow = 0.0
        self.peak_fast = 0.0
        self.ink_uv: list[Array] = []
        self._last_ink = None
        self.pen_down_steps = 0
        self.in_band_steps = 0
        self.f_down_sum = 0.0
        self.peak = 0.0
        self.torn = False
        self.collided = False
        self.last = self.ctl.compute(C.Case1Proposal(np.zeros(3)), self.dt)  # prime state
        # ... with the fields step() adds, so a caller can read them before the
        # first step (a status line drawn on the very first frame did)
        self.last.update(t=0.0, f_raw=np.zeros(3), f_filt=np.zeros(3), f_n=0.0, f_sensor_n=0.0,
                         pen_down=False, contact_uvh=np.zeros(3), n_ink=0,
                         x_d_next=self.ctl.x_d.copy(), K_next=self.ctl.K.copy())
        self.robot.set_qf(torch.zeros((1, len(q))))
        return self.observe()

    # ---- what the tool is and what it does to the board -----------------
    # Two hooks, so wiping (../wiping) can be the same simulator with a
    # different tool rather than a copy of it that drifts.
    def contact_point(self, rec, n):
        """Where the tool touches: the ball's centre pushed one radius down."""
        return rec["p"] - S.PEN_BALL_R * rec["R"][:, 2] - S.PEN_BALL_R * n

    def on_contact(self, f_n, uvh, down) -> None:
        """Writing lays ink along the contact path; wiping takes it off."""
        if not down:
            self._last_ink = None
            return
        uv = uvh[:2]
        if self._last_ink is None or np.linalg.norm(uv - self._last_ink) >= self.crit.ink_spacing:
            if self.u.put_ink(len(self.ink_uv), uv):
                self.ink_uv.append(uv.copy())
            self._last_ink = uv

    # ------------------------------------------------------------------ #
    def step(self, prop: C.Case1Proposal) -> dict:
        """One physics step: law at t, physics, then integrate, then ink.

        TIME CONVENTION of the returned record: every state (q, p, x_d, K, t) is
        at the START of the step, where the law was evaluated; forces and ink
        are what happened DURING the step; x_d_next / K_next are the applied
        values after it.  A request for the next step (x_d_req) is therefore
        compared against x_d_next, never x_d.
        """
        cr = self.crit
        t0 = self.t
        rec = self.ctl.compute(prop, self.dt)
        self.u.scene.step()
        self.ctl.advance()

        f_raw = self.u.scene.get_pairwise_contact_forces(self.pen, self.u.canvas)[0].cpu().numpy()
        a = self.dt / (self.dt + 1.0 / (2 * np.pi * cr.force_lp_hz))
        self.f_filt = self.f_filt + a * (f_raw - self.f_filt)
        n = self.frame.normal
        f_sensor_n = float(self.f_filt @ n)
        b = self.dt / (self.dt + 1.0 / (2 * np.pi * cr.pressure_lp_hz))
        self.f_slow += b * (float(f_raw @ n) - self.f_slow)
        f_n = self.f_slow                       # the pressure: what writes and tears

        uvh = self.frame.to_canvas(self.contact_point(rec, n))

        down = f_n >= cr.ink_force
        if down:
            self.pen_down_steps += 1
            self.f_down_sum += f_n
            lo, hi = cr.force_band
            self.in_band_steps += int(lo <= f_n <= hi)
        self.on_contact(f_n, uvh, down)
        self.peak = max(self.peak, f_n)
        self.peak_fast = max(self.peak_fast, f_sensor_n)
        self.torn |= f_n > cr.tear_force
        if self.step_i % 10 == 0:
            for lk in self.others:
                fo = self.u.scene.get_pairwise_contact_forces(lk, self.u.canvas)[0].cpu().numpy()
                self.collided |= float(np.linalg.norm(fo)) > cr.collision_force

        self.t += self.dt
        self.step_i += 1
        rec.update(t=t0, f_raw=f_raw, f_filt=self.f_filt.copy(), f_n=f_n, f_sensor_n=f_sensor_n,
                   pen_down=down, contact_uvh=uvh, n_ink=len(self.ink_uv),
                   x_d_next=self.ctl.x_d.copy(), K_next=self.ctl.K.copy())
        self.last = rec
        return rec

    # ------------------------------------------------------------------ #
    def k_diag(self, K: Array | None = None) -> Array:
        """Stiffness along the writing axes (u, v, n)."""
        K = self.ctl.K if K is None else K
        return np.diag(self.W.T @ K @ self.W).copy()

    def images(self) -> dict:
        if not self.cameras:
            return {}
        data = self.u._get_obs_sensor_data()
        return {k: v["rgb"][0].cpu().numpy() for k, v in data.items() if "rgb" in v}

    def observe(self, images: bool = True) -> dict:
        """What a policy may see, all at the current time.  Canvas pose and
        friction are NOT here.  `tau` is the torque applied over the last step."""
        q = self.robot.get_qpos()[0].cpu().numpy()
        qd = self.robot.get_qvel()[0].cpu().numpy()
        _, p, v, w = self.ctl.tip_state()
        q_tcp = self.ctl.tcp.pose.q[0].cpu().numpy()
        obs = {
            "t": self.t,
            "qpos": q.astype(np.float32), "qvel": qd.astype(np.float32),
            "tau": self.last["tau"].astype(np.float32),
            "tcp_pos": p.astype(np.float32), "tcp_quat": q_tcp.astype(np.float32),
            "tcp_vel": v.astype(np.float32), "tcp_angvel": w.astype(np.float32),
            "f_contact": self.f_filt.astype(np.float32),
            "x_d": self.ctl.x_d.astype(np.float32),
            "k_diag": self.k_diag().astype(np.float32),
            "tank_E": np.float32(self.ctl.E),
            "n_ink": len(self.ink_uv),
        }
        if images:
            for k, v in self.images().items():
                obs[f"rgb_{k}"] = v
        return obs

    def goal(self) -> dict:
        """The target in the BELIEVED canvas frame and in world coordinates."""
        pad, mask = self.target.padded()
        return {"text": self.spec.text, "strokes_uv": pad, "mask": mask,
                "strokes_world": self.belief.to_world(pad).astype(np.float32),
                "belief_origin": self.belief.origin, "belief_R": self.belief.R}

    def privileged(self) -> dict:
        return {"canvas_origin": self.frame.origin, "canvas_R": self.frame.R,
                "canvas_dz": self.spec.canvas_dz, "tilt": self.frame.tilt,
                "friction": self.spec.friction}

    # ------------------------------------------------------------------ #
    def score(self) -> dict:
        from scipy.spatial import cKDTree
        cr = self.crit
        T = self.target_pts
        ink = np.array(self.ink_uv) if self.ink_uv else np.zeros((0, 2))
        if len(ink):
            d_t, _ = cKDTree(ink).query(T)
            d_i, _ = cKDTree(T).query(ink)
            coverage = float(np.mean(d_t <= cr.tol))
            precision = float(np.mean(d_i <= cr.tol))
            chamfer = float(0.5 * (d_t.mean() + d_i.mean()))
        else:
            coverage = precision = 0.0
            chamfer = float("inf")
        in_band = self.in_band_steps / max(1, self.pen_down_steps)
        res = dict(coverage=coverage, precision=precision, chamfer_mm=1000 * chamfer,
                   in_band=float(in_band), peak_force=float(self.peak),
                   contact_force=float(self.f_down_sum / max(1, self.pen_down_steps)),
                   peak_force_fast=float(self.peak_fast),
                   pen_down_s=self.pen_down_steps * self.dt,
                   torn=bool(self.torn), collided=bool(self.collided),
                   n_ink=len(ink), t=self.t, tank_E=float(self.ctl.E))
        checks = {
            "coverage": coverage >= cr.min_coverage,
            "precision": precision >= cr.min_precision,
            "force": in_band >= cr.min_in_band,
            "no_tear": not self.torn,
            "no_collision": not self.collided,
        }
        res["checks"] = checks
        res["success"] = bool(all(checks.values()))
        res["fail_reason"] = ",".join(k for k, v in checks.items() if not v)
        return res

    def close(self) -> None:
        self.env.close()
