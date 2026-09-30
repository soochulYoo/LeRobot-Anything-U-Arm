"""An operator who INTENDS a constant-pitch spiral, on a surface that will not
let them draw one.

Everything in this package so far has been blocked by the same thing: the
operator's intent is unobservable, so every label rule substitutes a proxy and
none of them can be checked.  Here the intent is known by construction -- an
Archimedean spiral of fixed pitch, pressed at a constant normal force -- and the
surface is a random height field the operator cannot see.  What separates intent
from execution is then measurable rather than assumed.

THE OPERATOR.  What makes a human an operator is not degrees of freedom, it is
DELAY.  A zero-delay model tracks any surface perfectly and leaves no gap at all.
This one has:

    visual loop   sees where the tool actually is, 200 ms late, and corrects the
                  IN-PLANE error only -- the path is what they can see
    haptic loop   feels the contact force, 60 ms late, and adjusts how hard they
                  press -- force is what they can feel, and they feel it sooner
                  than they see anything
    motor noise   signal-dependent, so a hard push scatters more than a light one

The two loops have different delays and act on different axes, which is the same
split the cascade makes and is why an operator's own behaviour is worth
imitating rather than replacing.

THE TESTABLE CLAIM.  The operator's COMMAND is smooth because they cannot react
to the surface in time; the RESULT is rough because the surface acts instantly.
So a label taken from the command side should carry the intent, and a label
taken from the executed pose should carry the disturbance.  `gap()` measures both.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

import core as C

Array = np.ndarray


# --------------------------------------------------------------------------- #
# the surface the operator cannot see
# --------------------------------------------------------------------------- #
@dataclass
class RandomSurface:
    """Smooth random height field over the tangent plane.

    A sum of random sinusoids rather than discrete bumps: a wiped surface is
    undulating, not studded, and sharp features produce impact spikes that swamp
    everything the experiment is trying to measure.
    """
    # Calibrated so the SURFACE dominates, not the operator.  On a flat plate
    # this operator draws the spiral to 1.34 mm on its own; at 2.5 mm RMS with a
    # 15 mm length scale the error is ~3x that, so the gap being measured is the
    # disturbance rather than the demonstrator's own tremor.  What matters is
    # the SLOPE, amp/length_scale -- that is what tilts the contact normal and
    # pushes the tool off the path.  Height alone barely disturbs a wipe.
    amp: float = 0.0025          # m, RMS height
    length_scale: float = 0.015  # m, characteristic wavelength
    n_modes: int = 6
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.k = rng.normal(0.0, 1.0 / self.length_scale, size=(self.n_modes, 2))
        self.phase = rng.uniform(0.0, 2 * np.pi, size=self.n_modes)
        self.w = rng.normal(0.0, 1.0, size=self.n_modes)
        self.w /= np.sqrt(np.sum(self.w ** 2))

    def height(self, s1: float, s2: float) -> float:
        return float(self.amp * np.sum(
            self.w * np.sin(self.k[:, 0] * s1 + self.k[:, 1] * s2 + self.phase)))

    def grad(self, s1: float, s2: float) -> tuple[float, float]:
        """Analytic slope, (dh/ds1, dh/ds2) -- what tilts the contact normal."""
        c = self.amp * self.w * np.cos(self.k[:, 0] * s1 + self.k[:, 1] * s2 + self.phase)
        return float(np.sum(c * self.k[:, 0])), float(np.sum(c * self.k[:, 1]))


# --------------------------------------------------------------------------- #
# the intent
# --------------------------------------------------------------------------- #
@dataclass
class SpiralIntent:
    """Archimedean spiral r = pitch * theta / (2 pi): constant spacing between
    turns, which is what a wipe wants -- no gaps and no double coverage."""
    pitch: float = 0.022         # m between successive turns
    turns: float = 3.5
    duration: float = 18.0
    f_normal: float = 8.0        # N, intended press

    def radius(self, u: float) -> float:
        return self.pitch * self.turns * u

    def offset(self, t: float, t0: float, t1: np.ndarray, t2: np.ndarray) -> Array:
        u = float(np.clip((t - t0) / self.duration, 0.0, 1.0))
        th = 2 * np.pi * self.turns * u
        r = self.radius(u)
        return r * (np.cos(th) * t1 + np.sin(th) * t2)

    def curve(self, t0: float, t1: np.ndarray, t2: np.ndarray, n: int = 3000) -> Array:
        us = np.linspace(0.0, 1.0, n)
        th = 2 * np.pi * self.turns * us
        r = self.pitch * self.turns * us
        return (r[:, None] * (np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2))


@dataclass
class Operator:
    """Arm impedance plus the two delayed correction loops."""
    magnitude: float = 900.0     # arm stiffness, task frame
    ratio: float = 8.0           # stiff along the stroke, soft into the surface
    Bh: float = 25.0
    visual_gain: float = 0.6     # how much of the seen path error they take out
    visual_delay: float = 0.20
    haptic_gain: float = 0.004   # m per N of felt force error, per second
    haptic_delay: float = 0.06
    motor_noise: float = 0.02
    press0: float = 0.060        # initial guess at how deep to press
    seed: int = 0


# --------------------------------------------------------------------------- #
# the demonstration
# --------------------------------------------------------------------------- #
@dataclass
class SpiralTask:
    tilt_x: float = 0.30
    tilt_y: float = -0.17
    wall: float = 0.10
    ke: float = 5000.0
    de: float = 10.0
    mu: float = 0.3
    settle: float = 4.0
    Ka: float = 300.0
    Ba: float = 40.0
    Ma: float = 3.0
    Br: float = 25.0
    Ki: float = 4000.0
    Di: float = 180.0
    Ki_stiff: float = 30000.0     # the "outer only" inner loop: rigid tracking
    Ms: float = 2.0

    @property
    def R(self) -> Array:
        cx, sx = np.cos(self.tilt_x), np.sin(self.tilt_x)
        cy, sy = np.cos(self.tilt_y), np.sin(self.tilt_y)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return Ry @ Rx


def make_operator_target(task: "SpiralTask", intent: "SpiralIntent", op: "Operator"):
    """The operator's two delayed correction loops, as a `target_fn(t, obs)`.

    Extracted so that the cascade and the two single-interface executors
    (case_spiral.py) are driven by the IDENTICAL operator.  That is the whole
    basis of the comparison: if the operator differed between runs, a difference
    in path or force error would say nothing about the follower.

    `obs` carries the DELAYED observations, exactly as core.CascadeSim builds
    them: {"x": tool pose seen `visual_delay` ago, "f": contact force felt
    `haptic_delay` ago, "t": the time those observations were taken}.

    Returns (target_fn, dt_holder).  The caller must set dt_holder["dt"] before
    the first call -- the haptic loop integrates a rate, so it needs the step.
    """
    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
    state = {"press": op.press0}
    dt_holder: dict = {}

    def target(t: float, obs: dict) -> Array:
        ideal = intent.offset(t, task.settle, t1, t2)
        corr = np.zeros(3)
        if t > task.settle + op.visual_delay:
            seen_ideal = intent.offset(obs["t"], task.settle, t1, t2)
            seen_xy = obs["x"] - float(obs["x"] @ n) * n
            corr = op.visual_gain * (seen_ideal - seen_xy)
        # haptic loop: press harder or softer until it feels like f_normal
        f_felt = float(obs["f"] @ n)
        state["press"] += op.haptic_gain * (intent.f_normal - f_felt) * dt_holder["dt"]
        state["press"] = float(np.clip(state["press"], 0.0, 0.25))
        ramp = min(1.0, t / max(task.settle * 0.6, 1e-9))
        return (task.wall + ramp * state["press"]) * n + ideal + corr

    return target, dt_holder


def demonstrate(task: SpiralTask, intent: SpiralIntent, op: Operator,
                surface: RandomSurface, controller: str = "cascade") -> dict:
    """One teleoperated spiral wipe.  Returns the log plus the intent it was
    trying to realise, so the two can be compared directly."""
    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]      # n is the PENETRATION direction
    Kh = _compose(op.magnitude, op.ratio, R)
    # The three controllers differ only in the follower.  "outer only" keeps the
    # admittance and makes the inner loop rigid; "inner only" keeps the inner
    # compliance and removes the admittance, so the operator's decoded command
    # drives the impedance directly and nothing in the loop sees the force.
    if controller == "outer only":
        _Ki, _Di = task.Ki_stiff, 2.0 * 0.8 * np.sqrt(task.Ki_stiff * task.Ms)
    else:
        _Ki, _Di = task.Ki, task.Di
    target, dt_holder = make_operator_target(task, intent, op)

    base = C.CascadeParams(
        n=3,
        human=C.HumanParams(Kh=Kh, Bh=op.Bh, target_fn=target,
                            visual_delay=op.visual_delay, haptic_delay=op.haptic_delay,
                            motor_noise=op.motor_noise, noise_seed=op.seed),
        master=C.MasterParams(Mm=2.0, Bm=10.0),
        coupling=C.CouplingParams(Ka=task.Ka, Ba=task.Ba),
        outer=C.OuterParams(Ma=task.Ma, Br=task.Br,
                            rigid=(controller == "inner only")),
        inner=C.InnerParams(Ki=_Ki, Di=_Di, Ms=task.Ms),
        env=C.EnvParams(ke=task.ke, de=task.de, x_wall=task.wall, normal=n, mu=task.mu),
    )
    prm = dataclasses.replace(base, dt=C.safe_dt(base))
    dt_holder["dt"] = prm.dt
    def slope(_t, x):
        g1, g2 = surface.grad(float(x @ t1), float(x @ t2))
        return g1 * t1 + g2 * t2

    prm = dataclasses.replace(
        prm, env=dataclasses.replace(prm.env, slope_fn=slope))
    sim = C.CascadeSim(prm, "teleop",
                       wall_offset_fn=lambda t, x: surface.height(float(x @ t1), float(x @ t2)))
    for attr in ("x_r", "x", "x_c", "x_m"):
        setattr(sim, attr, task.wall * n.copy())
    sim._x_r0 = sim.x_r.copy()
    log = sim.run(task.settle + intent.duration + 1.0)
    log["_dt"] = prm.dt
    log["_task"], log["_intent"], log["_op"], log["_surface"] = task, intent, op, surface
    log["_Kh_true"] = Kh                     # the operator's real arm stiffness: the answer key
    log["_controller"] = controller
    return log


def _compose(magnitude: float, ratio: float, frame: Array) -> Array:
    w = np.array([ratio, ratio, 1.0], dtype=float)
    w = w / np.prod(w) ** (1.0 / 3.0)
    return magnitude * (frame @ np.diag(w) @ frame.T)


# --------------------------------------------------------------------------- #
# how far the result is from the intent
# --------------------------------------------------------------------------- #
def gap(log: dict) -> dict:
    """Intent-vs-result, measured on the RESULT and on the COMMAND separately.

    The claim under test is that the operator's command stays close to what they
    meant -- they cannot react to the surface inside their own delay -- while the
    executed pose carries the disturbance.  If so, a pose label taken from the
    command is a better record of intent than one taken from the tool.
    """
    task, intent = log["_task"], log["_intent"]
    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
    t = log["t"]
    sel = t >= task.settle
    curve = intent.curve(task.settle, t1, t2)          # ideal in-plane points

    def inplane(v: Array) -> Array:
        return v - np.outer(v @ n, n)

    def path_rms(pos: Array, chunk: int = 512) -> float:
        """Chunked: the one-shot form allocates an (N_samples, N_curve, 3)
        tensor, which at 19k samples against a 3k-point curve is 1.4 GB for a
        single call -- survivable once, an out-of-memory kill as soon as several
        episodes run in parallel.  Bounds the peak; changes no result."""
        p = inplane(pos - task.wall * n)
        best = np.empty(len(p))
        for i in range(0, len(p), chunk):
            blk = p[i:i + chunk]
            d = np.linalg.norm(blk[:, None, :] - curve[None, :, :], axis=2)
            best[i:i + chunk] = d.min(axis=1)
        return float(np.sqrt(np.mean(best ** 2)))

    res = {
        "path_result": path_rms(np.asarray(log["x"])[sel]),
        "path_command": path_rms(np.asarray(log["x_c"])[sel]),
        "force_result": float(np.sqrt(np.mean(
            (np.asarray(log["f_normal"])[sel, 0] - intent.f_normal) ** 2))),
        "force_command": float(np.sqrt(np.mean(
            (np.asarray(log["f_ch"])[sel] @ n - intent.f_normal) ** 2))),
        "contact": float(np.mean(np.asarray(log["f_normal"])[sel, 0] > 0.1 * intent.f_normal)),
    }
    res["f_mean"] = float(np.mean(np.asarray(log["f_normal"])[sel, 0]))
    return res
