"""Lean cascade simulator: inner Cartesian impedance + outer admittance.

Why this exists separately from haptic_teleop_fr3_bilateral.py: that file runs
the cascade through SAPIEN contact, IK and a redundant arm, so when a number
looks wrong there is no way to tell whether the controller, the IK or the
contact model is at fault.  Here the same controller chain runs against a
scalar spring environment whose steady state is known in closed form
(analytic.py), so the controller can be proven correct first and ported second.

See analytic.py for the sign convention.  Two drive modes:

  TELEOP  -- human spring -> master -> (delayed channel) -> coupling -> outer
             admittance -> inner impedance -> slave -> environment.
             This GENERATES demonstrations, with the human's true Kh known.

  POLICY  -- a policy writes (x_ref, f_d, Ko) into the outer admittance, which
             then drives the same inner impedance -> slave -> environment.
             This REPLAYS / deploys, and is the action space under study.

Both modes share every downstream stage, which is the property that makes
"demo and deploy are the same controller" checkable rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

Array = np.ndarray


def _mat(x, n: int, name: str) -> Array:
    """Coerce a gain to an (n, n) matrix.

    A scalar k becomes k*I (isotropic), a length-n sequence becomes a diagonal,
    and an (n, n) array is taken as given.  Full matrices are the point: an
    anisotropic stiffness is only meaningful as a matrix, and a DIAGONAL matrix
    silently assumes the anisotropy axes coincide with the world axes -- which
    is exactly the assumption the tilted-surface study is built to break.
    """
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return float(a) * np.eye(n)
    if a.ndim == 1:
        if a.size == 1:
            return float(a[0]) * np.eye(n)
        if a.size != n:
            raise ValueError(f"{name}: length {a.size} gain for an {n}-D system")
        return np.diag(a)
    if a.shape != (n, n):
        raise ValueError(f"{name}: expected scalar, ({n},) or ({n},{n}); got {a.shape}")
    return a.copy()


def _kmax(K) -> float:
    """Largest eigenvalue of a stiffness/damping gain (symmetric part)."""
    a = np.asarray(K, dtype=float)
    if a.ndim == 2:
        return float(np.max(np.linalg.eigvalsh(0.5 * (a + a.T))))
    return float(np.max(np.atleast_1d(a)))


def _mmin(M) -> float:
    """Smallest eigenvalue of an inertia gain (symmetric part)."""
    a = np.asarray(M, dtype=float)
    if a.ndim == 2:
        return float(np.min(np.linalg.eigvalsh(0.5 * (a + a.T))))
    return float(np.min(np.atleast_1d(a)))


def _vec(x, n: int, name: str) -> Array:
    """Broadcast a scalar or sequence to a float array of shape (n,)."""
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.size == 1:
        a = np.repeat(a, n)
    if a.shape != (n,):
        raise ValueError(f"{name}: expected scalar or shape ({n},), got shape {a.shape}")
    return a.copy()


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass
class HumanParams:
    """Operator arm as a spring-damper reaching for an intended target.

    Kh is the GROUND TRUTH the label study tries to recover.  `reach` is the
    intended displacement from the master's start pose, positive = into the wall.
    A spring-damper (not a constant force) is required: a constant f_h drives
    f_m -> f_h at steady state by construction regardless of the rest of the
    chain, so it can never reveal anything about the channel or environment.
    """
    Kh: float = 400.0
    Bh: float = 20.0
    reach: float = 0.16
    # A wipe demonstration needs the operator to move in 3-D, not to press along
    # one axis, so `target_fn` overrides `reach` when given.  Kh may be a matrix:
    # an operator wiping a surface is genuinely stiffer along the stroke than
    # into the surface, and THAT anisotropy, in the task frame, is the ground
    # truth the label rules are asked to recover.
    #
    # target_fn is called as target_fn(t, obs) where obs carries the operator's
    # DELAYED view of the world:  {"x": follower position, "f": felt force,
    # "t": the time those observations were taken}.  The delay is what makes an
    # operator an operator.  A zero-delay model tracks any surface perfectly and
    # there is no gap between what it intended and what it did -- which is the
    # very gap this scenario exists to measure.
    target_fn: "Callable[..., Array] | None" = None
    visual_delay: float = 0.20     # s, seeing where the tool actually is
    haptic_delay: float = 0.06     # s, feeling the contact force
    motor_noise: float = 0.0       # signal-dependent: sigma = this * |f_h|
    noise_seed: int = 0
    # The operator's intent is not constant in reality -- they vary how hard
    # they press.  This matters for identifiability, not just realism: with a
    # CONSTANT target, a regression carrying a single intercept recovers Kh
    # exactly with no probe at all, because the whole unobservable intent is one
    # unknown number.  Once the target varies, the intent can absorb any
    # stiffness and the probe is what separates them -- in frequency, since
    # intent lives near reach_hz and the probe sits at master.probe_hz.
    reach_amp: float = 0.0   # m, amplitude of the slow intent variation
    reach_hz: float = 0.2    # Hz, how fast the operator changes their mind


@dataclass
class MasterParams:
    """Master (leader) virtual dynamics, plus the optional identification probe.

    The probe is a force injected HERE, at the device, not into f_h: physically
    a probe is the handle shaking the operator's hand and the operator's arm
    impedance resisting.  Injecting it into f_h instead would just be the
    operator shaking themselves, which excites nothing about Kh.  With the
    probe on, regressing the handle force against handle motion becomes
    well-conditioned and Kh is identifiable; with it off, the only motion in the
    record is the slow quasi-static drift, which is collinear with the whole
    series chain -- that is the identifiability claim the label study measures.
    """
    Mm: float = 2.0
    Bm: float = 10.0
    # Scalar, or per-axis.  Identifying a full 3x3 arm stiffness needs THREE
    # distinct frequencies: each gives 3 complex equations, so three give 18 real
    # equations for the 9 + 9 unknowns of (Kh, Bh) -- exactly determined.  One
    # frequency on every axis at once leaves the matrix unidentifiable no matter
    # how long the recording runs.
    probe_amp: "float | Array" = 0.0   # N, device-side perturbation force
    probe_hz: "float | Array" = 3.0    # Hz


@dataclass
class ChannelParams:
    """Transport delay only ("direct" mode of the FR3 script).  The wave-variable
    scattering transform is NOT reproduced here; the label study runs at zero
    delay to isolate the labelling question, then sweeps delay as a robustness
    check, and the passivity argument stays in the FR3 script where it lives."""
    delay_f: float = 0.0  # master -> slave, s
    delay_b: float = 0.0  # slave -> master, s


@dataclass
class CouplingParams:
    """Virtual coupling between the decoded command x_c and the admittance
    reference x_r.  f_ch = Ka (x_c - x_r) + Ba (v_c - v_r)."""
    Ka: float = 100.0
    Ba: float = 25.0


@dataclass
class OuterParams:
    """Outer admittance:  Ma a_r = f_drive - f_e - Br v_r."""
    Ma: float = 3.0
    Br: float = 25.0
    # rigid=True turns the outer layer OFF: the reference is handed straight
    # through (x_r := x_ref, v_r := 0) with no admittance dynamics at all, so the
    # inner impedance faces the disturbance alone.  This is the ablation that
    # separates the two layers' roles.
    rigid: bool = False
    # Safety net recommended over tuning an energy tank: bound the reference
    # speed directly, so approach speed is bounded regardless of policy error.
    v_limit: float = np.inf


@dataclass
class InnerParams:
    """Inner Cartesian impedance + the slave's own inertia.

    f_cmd = Ki (x_r - x) + Di (v_r - v);   Ms a = f_cmd - f_e.
    This is the stage that haptic_teleop_fr3_bilateral.py:565 skips ("inner
    impedance skipped per request").  Ki enters the steady state exactly like
    the environment stiffness, so Ki >> Ko is a hard requirement, not a
    preference -- see analytic.contact_force_policy.
    """
    Ki: float = 2000.0
    Di: float = 80.0
    Ms: float = 2.0


@dataclass
class EnvParams:
    """Unilateral spring-damper wall at x_wall, resisting penetration."""
    ke: float = 5000.0
    de: float = 10.0
    x_wall: float = 0.10
    # Surface orientation.  `normal` is the unit direction of PENETRATION (the
    # tool presses along +normal into the material) and the plane sits at
    # normal . x = x_wall.  None means the first axis, which reproduces the
    # 1-DOF wall exactly.  Tilting this is what makes an anisotropic stiffness
    # able to point the wrong way.
    normal: Array | None = None
    # Coulomb friction, tangential force <= mu * |normal force|, opposing
    # tangential motion.  Without it a wiping stroke costs nothing and the
    # tangential stiffness has nothing to act against, so mu > 0 is a
    # requirement for the anisotropy study, not a refinement.
    mu: float = 0.0
    # Local surface slope, as d(height)/d(tangent) for the two tangent axes at
    # the tool's position.  An undulating surface does not only move up and
    # down: its normal TILTS, and the tilt is what pushes a tool sideways off a
    # path.  Leaving it out makes a bumpy surface a purely vertical disturbance,
    # which a wipe barely notices -- the tangential motion then tracks as well
    # over bumps as over glass, and the scenario measures nothing.
    # Returns the surface GRADIENT at the tool, as a world 3-vector lying in the
    # tangent plane.  A world vector rather than two components in some basis:
    # the environment's own tangent basis is arbitrary, and having the caller
    # resolve the gradient in whatever frame it thinks in removes a silent
    # mismatch between the two.
    slope_fn: "Callable[[float, Array], Array] | None" = None


@dataclass
class CascadeParams:
    human: HumanParams = field(default_factory=HumanParams)
    master: MasterParams = field(default_factory=MasterParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    coupling: CouplingParams = field(default_factory=CouplingParams)
    outer: OuterParams = field(default_factory=OuterParams)
    inner: InnerParams = field(default_factory=InnerParams)
    env: EnvParams = field(default_factory=EnvParams)
    n: int = 1
    dt: float = 1e-3


class DelayLine:
    """Fixed transport delay for an (n,)-vector at a known dt.

    Semantics match haptic_teleop_fr3_bilateral.DelayLine: push what is sent
    now, receive what was sent `delay` ago.  Zero delay is pass-through.
    """

    def __init__(self, delay_s: float, dt: float, n: int):
        if delay_s < 0:
            raise ValueError("delay must be >= 0")
        self.n_steps = int(round(delay_s / dt))
        self.buf = [np.zeros(n) for _ in range(self.n_steps)]

    def push_and_get(self, value: Array) -> Array:
        if self.n_steps == 0:
            return value.copy()
        self.buf.append(value.copy())
        return self.buf.pop(0)


# --------------------------------------------------------------------------- #
# explicit-integrator stability
# --------------------------------------------------------------------------- #
def stability_limit(p: "CascadeParams") -> tuple[float, str]:
    """Largest dt for which semi-implicit Euler stays stable on this chain.

    Two families of constraint, checked for every mass/stiffness and
    mass/damping pair in the cascade:
        stiffness:  dt < 2 sqrt(m/k)     (undamped oscillator limit)
        damping:    dt < 2 m / b         (explicit damping goes unstable first)
    The tightest one is returned with the name of the pair that set it, so an
    unstable configuration reports WHICH element is too stiff rather than just
    producing NaNs several seconds into a run.

    A rigid inner impedance (Ki -> inf) is therefore not reachable with an
    explicit integrator at any useful dt; use a large finite Ki and accept the
    residual Ko/Ki bias, or switch that stage to an implicit solve.
    """
    worst_dt, worst_name = np.inf, "none"
    stiff_pairs = [
        ("inner Ki / slave Ms", p.inner.Ki, p.inner.Ms),
        ("env ke / slave Ms", p.env.ke, p.inner.Ms),
        ("coupling Ka / outer Ma", p.coupling.Ka, p.outer.Ma),
        ("human Kh / master Mm", p.human.Kh, p.master.Mm),
    ]
    for name, k_g, m_g in stiff_pairs:
        k, m = _kmax(k_g), _mmin(m_g)
        if k <= 0 or not np.isfinite(k) or m <= 0:
            continue
        dt_k = 2.0 * np.sqrt(m / k)
        if dt_k < worst_dt:
            worst_dt, worst_name = dt_k, name
    damp_pairs = [
        ("inner Di / slave Ms", p.inner.Di, p.inner.Ms),
        ("env de / slave Ms", p.env.de, p.inner.Ms),
        ("outer Br / outer Ma", p.outer.Br, p.outer.Ma),
        ("coupling Ba / outer Ma", p.coupling.Ba, p.outer.Ma),
        ("master Bm / master Mm", p.master.Bm, p.master.Mm),
        ("human Bh / master Mm", p.human.Bh, p.master.Mm),
    ]
    for name, b_g, m_g in damp_pairs:
        b, m = _kmax(b_g), _mmin(m_g)
        if b <= 0 or not np.isfinite(b) or m <= 0:
            continue
        dt_b = 2.0 * m / b
        if dt_b < worst_dt:
            worst_dt, worst_name = dt_b, name
    return float(worst_dt), worst_name


def safe_dt(p: "CascadeParams", margin: float = 0.05, round_to_pow10: bool = True) -> float:
    """A dt with `margin` of the stability limit.  Experiments should call this
    instead of hard-coding dt, so changing a gain can never silently push a
    sweep past the stability boundary."""
    dt_max, _ = stability_limit(p)
    dt = margin * dt_max
    if round_to_pow10:  # round DOWN to 1/2/5 x 10^k so logs stay legible
        exp = np.floor(np.log10(dt))
        mant = dt / 10.0**exp
        mant = 5.0 if mant >= 5.0 else (2.0 if mant >= 2.0 else 1.0)
        dt = mant * 10.0**exp
    return float(dt)


# --------------------------------------------------------------------------- #
# action for POLICY mode
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """The unified force-impedance action under study.

    GAUGE: the (x_ref, f_d, Ko) triple is over-parameterised -- the
    instantaneous drive f_d + Ko(x_ref - x_r) is determined by two numbers, not
    three.  The labelling rule must therefore FIX A GAUGE or the decomposition
    is not identifiable, and replaying labels taken from two different gauges
    double-counts the force.  This package fixes  x_ref := x_r  (see
    labels.py), which makes f_d carry all of the contact intent and the spring
    term a pure tracking/regularisation term.  `spring_share()` below is the
    diagnostic that checks the gauge actually held at deploy time.
    """
    x_ref: Array
    f_d: Array
    Ko: Array

    @staticmethod
    def _apply(K, d: Array) -> Array:
        return (K @ d) if np.ndim(K) == 2 else (K * d)

    def drive(self, x_r: Array) -> Array:
        return self.f_d + self._apply(self.Ko, self.x_ref - x_r)

    def spring_share(self, x_r: Array) -> Array:
        """|Ko (x_ref - x_r)| / |drive|: the gauge diagnostic.  Under the fixed
        gauge this should stay near 0 in contact.  If it drifts toward 1 the
        learned decomposition has collapsed back to the equilibrium-point
        parameterisation and any force-action advantage is accidental."""
        d = self.drive(x_r)
        spring = self._apply(self.Ko, self.x_ref - x_r)
        nd = float(np.linalg.norm(d))
        return np.full(d.shape, float(np.linalg.norm(spring)) / nd if nd > 1e-12 else 0.0)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #
def unilateral_wall(env: EnvParams, n_hat: Array, t: float, x: Array, v: Array,
                    wall: float) -> "tuple[Array, float, float]":
    """Unilateral spring-damper wall reaction, >= 0, resisting penetration.

    Extracted as a free function so the two single-interface simulators
    (case1.py, case2.py) face the IDENTICAL environment as the full cascade.
    Sharing one contact model is what makes a cross-case force comparison
    mean anything: a difference in the reported force is then attributable to
    the controller structure alone, which is the entire point of comparing
    them.  Returns (f, f_normal, penetration).
    """
    delta = float(n_hat @ x) - wall          # penetration depth along the normal
    if delta <= 0.0:
        return np.zeros(x.shape[0]), 0.0, delta

    # Tilt the contact normal by the local slope.  n_local ~ n - g1 t1 - g2 t2
    # for a height field of gradient (g1, g2) over the tangent plane.
    if env.slope_fn is not None:
        g = np.asarray(env.slope_fn(t, x), dtype=float).reshape(x.shape[0])
        n_hat = n_hat - g
        n_hat = n_hat / float(np.linalg.norm(n_hat))

    v_n = float(n_hat @ v)
    # Clamped at 0 so the damper can never pull the tool INTO the surface,
    # which an unclamped ke*d + de*v model does on separation.
    f_n = max(0.0, env.ke * delta + env.de * v_n)
    f = f_n * n_hat

    if env.mu > 0.0 and f_n > 0.0:
        v_t = v - v_n * n_hat
        speed = float(np.linalg.norm(v_t))
        if speed > 1e-9:
            # f_e is SUBTRACTED from the drive, so a resisting friction force
            # enters with a PLUS sign along the tangential velocity.
            f = f + (env.mu * f_n) * (v_t / speed)
    return f, f_n, delta


class CascadeSim:
    """Semi-implicit (symplectic) Euler integration of the whole chain.

    Velocity is updated before position at every stage, matching
    haptic_teleop_fr3_bilateral.py.  The two places that could form an
    algebraic loop use the previous step's value and are marked; this is the
    same choice the FR3 script makes, so the two implementations stay
    comparable step for step.
    """

    def __init__(self, params: CascadeParams, mode: Literal["teleop", "policy"] = "teleop",
                 wall_offset_fn: Callable[[float, Array], float] | None = None):
        self.p = params
        self.mode = mode
        # A moving or uneven surface: x_wall = x_wall + wall_offset_fn(t, x).
        # It takes the tool POSITION as well as the time because surface
        # waviness is a function of where the tool is, not of when it got there
        # -- a wipe over a bumpy surface and a wipe over a drifting flat one are
        # different disturbances and the study needs both.
        self.wall_offset_fn = wall_offset_fn
        n, dt = params.n, params.dt
        self.n, self.dt = n, dt

        self._check_stability()

        # --- state, all shape (n,) ---
        self.x_m = np.zeros(n)   # master virtual position (0 = its start pose)
        self.v_m = np.zeros(n)
        self.x_c = np.zeros(n)   # decoded slave-side command
        self.x_r = np.zeros(n)   # outer admittance reference
        self.v_r = np.zeros(n)
        self.x = np.zeros(n)     # slave actual position
        self.v = np.zeros(n)

        # Every gain is coerced to an (n, n) matrix ONCE, here, so the step loop
        # is plain matrix algebra and a scalar, a diagonal and a full anisotropic
        # gain all take the identical code path.  A gain that behaves
        # differently depending on how it was spelled would make the anisotropy
        # study impossible to trust.
        g = params
        self.G = {
            "Kh": _mat(g.human.Kh, n, "Kh"), "Bh": _mat(g.human.Bh, n, "Bh"),
            "Mm": _mat(g.master.Mm, n, "Mm"), "Bm": _mat(g.master.Bm, n, "Bm"),
            "Ka": _mat(g.coupling.Ka, n, "Ka"), "Ba": _mat(g.coupling.Ba, n, "Ba"),
            "Ma": _mat(g.outer.Ma, n, "Ma"), "Br": _mat(g.outer.Br, n, "Br"),
            "Ki": _mat(g.inner.Ki, n, "Ki"), "Di": _mat(g.inner.Di, n, "Di"),
            "Ms": _mat(g.inner.Ms, n, "Ms"),
        }
        nrm = g.env.normal
        if nrm is None:
            nrm = np.zeros(n); nrm[0] = 1.0
        nrm = np.asarray(nrm, dtype=float).reshape(n)
        norm = float(np.linalg.norm(nrm))
        if norm < 1e-12:
            raise ValueError("env.normal must be a non-zero vector")
        self._n_hat = nrm / norm

        self.f_ch = np.zeros(n)  # previous step's coupling wrench (for the backward line)
        self._x_r0 = np.zeros(n)  # outer reference at t=0, for the retreat measurement
        self.wall_now = params.env.x_wall
        self.f_normal = 0.0
        self.penetration = 0.0
        self._vis_line = DelayLine(params.human.visual_delay, dt, n)
        self._hap_line = DelayLine(params.human.haptic_delay, dt, n)
        self._noise_rng = np.random.default_rng(params.human.noise_seed)
        self._probe_amp = _vec(params.master.probe_amp, n, "probe_amp")
        self._probe_hz = _vec(params.master.probe_hz, n, "probe_hz")
        self._probe_on = bool(np.any(self._probe_amp != 0.0))
        self.t = 0.0

        self._fwd = DelayLine(params.channel.delay_f, dt, n)
        self._bwd = DelayLine(params.channel.delay_b, dt, n)

    # ---------------- validation ----------------
    def _check_stability(self) -> None:
        dt_max, source = stability_limit(self.p)
        if self.dt >= dt_max:
            raise ValueError(
                f"dt={self.dt:.2e}s is at or past the explicit stability limit "
                f"{dt_max:.2e}s set by {source}. Try dt={safe_dt(self.p):.2e}s "
                f"(see cascade.core.safe_dt)."
            )
        self.dt_limit, self.dt_limit_source = dt_max, source

    # ---------------- environment ----------------
    def env_force(self) -> Array:
        """Unilateral wall reaction, >= 0, resisting penetration.  The contact
        model itself lives in the free function `unilateral_wall` so that
        case1.py and case2.py face the identical environment."""
        p = self.p.env
        wall = p.x_wall + (self.wall_offset_fn(self.t, self.x) if self.wall_offset_fn else 0.0)
        self.wall_now = wall
        f, self.f_normal, self.penetration = unilateral_wall(
            p, self._n_hat, self.t, self.x, self.v, wall)
        return f

    # ---------------- one step ----------------
    def step(self, action: Action | None = None) -> dict:
        """Advance one dt.

        STRUCTURE: every force is evaluated from the state at time t, the record
        is taken, and only then is every state integrated to t+dt.  This matters
        beyond tidiness -- an earlier version integrated each stage as it went
        and logged the post-update state next to the pre-update force, so the
        record carried a one-step lag between force and pose.  Any label rule
        that regresses force against displacement then sees a phase error of
        w*dt, which at a 3 Hz probe silently biased an identified stiffness by
        ~4%.  A log that is meant to have labels extracted from it must be
        internally consistent, so the record below satisfies, exactly:
            f_e   == env(x, v)
            f_h   == Kh (target - x_m) - Bh v_m
            f_ch  == Ka (x_c - x_r) + Ba (v_c - v_r)
            f_cmd == Ki (x_r - x)    + Di (v_r - v)
        """
        p, dt, G = self.p, self.dt, self.G
        if self.mode == "policy" and action is None:
            raise ValueError("policy mode requires an action")
        if self.mode == "teleop" and action is not None:
            raise ValueError("teleop mode is driven by the human model; pass action=None")

        # ---------------- all forces, from the state at time t ----------------
        f_e = self.env_force()
        f_probe = np.zeros(self.n)

        if self.mode == "teleop":
            h = p.human
            if h.target_fn is not None:
                obs = {"x": self._vis_line.push_and_get(self.x),
                       "f": self._hap_line.push_and_get(f_e),
                       "t": max(0.0, self.t - h.visual_delay)}
                target = np.asarray(h.target_fn(self.t, obs), dtype=float).reshape(self.n)
            else:
                target = h.reach
                if h.reach_amp != 0.0:
                    target = target + h.reach_amp * np.sin(2.0 * np.pi * h.reach_hz * self.t)
            f_h = G["Kh"] @ (target - self.x_m) - G["Bh"] @ self.v_m
            if h.motor_noise > 0.0:
                # Signal-dependent noise: human motor output scatters in
                # proportion to how hard it is pushing, which is why a light
                # touch is steadier than a hard one.  Constant-variance noise
                # would be the wrong model and would wash out at high force.
                f_h = f_h + self._noise_rng.normal(
                    0.0, h.motor_noise * float(np.linalg.norm(f_h)) + 1e-12, self.n)

            # channel: velocity forward, coupling wrench backward.  The backward
            # line carries the PREVIOUS step's f_ch, which is what breaks the
            # algebraic loop; the FR3 script makes the same choice.
            v_c = self._fwd.push_and_get(self.v_m)
            f_m = self._bwd.push_and_get(self.f_ch)
            if self._probe_on:
                f_probe = self._probe_amp * np.sin(2.0 * np.pi * self._probe_hz * self.t)

            f_ch = G["Ka"] @ (self.x_c - self.x_r) + G["Ba"] @ (v_c - self.v_r)
            f_drive = f_ch
        else:
            f_h = np.zeros(self.n)
            f_m = np.zeros(self.n)
            v_c = np.zeros(self.n)
            f_ch = np.zeros(self.n)
            f_drive = action.drive(self.x_r)

        if p.outer.rigid:
            # Hand the command through untouched, so the inner impedance is all
            # that stands between the command and the world and nothing responds
            # to the measured force.  In TELEOP that command is the decoded
            # x_c -- the operator's own hand, with the admittance removed from
            # between them and the surface.
            if self.mode == "policy":
                self.x_r = action.x_ref.copy()
                self.v_r = np.zeros(self.n)
            else:
                self.x_r = self.x_c.copy()
                self.v_r = v_c.copy()

        f_cmd = G["Ki"] @ (self.x_r - self.x) + G["Di"] @ (self.v_r - self.v)

        # ---------------- record (state and forces both at time t) ----------------
        rec = {
            "t": self.t,
            "f_h": f_h.copy(), "f_m": f_m.copy(), "f_ch": f_ch.copy(),
            "f_probe": f_probe.copy(), "f_drive": f_drive.copy(),
            "f_cmd": f_cmd.copy(), "f_e": f_e.copy(), "v_c": v_c.copy(),
            "x_m": self.x_m.copy(), "v_m": self.v_m.copy(),
            "x_c": self.x_c.copy(), "x_r": self.x_r.copy(), "v_r": self.v_r.copy(),
            "x": self.x.copy(), "v": self.v.copy(),
            "x_wall": np.full(self.n, self.wall_now),
            "defl_inner": (self.x_r - self.x).copy(),   # what the inner spring carries
            "defl_outer": (self.x_r - self._x_r0).copy(),  # how far the outer layer retreated
            "f_normal": np.full(self.n, self.f_normal),     # scalar force along the surface normal
            "penetration": np.full(self.n, self.penetration),
        }

        # ---------------- integrate every state to t+dt ----------------
        if self.mode == "teleop":
            a_m = np.linalg.solve(G["Mm"], f_h - f_m + f_probe - G["Bm"] @ self.v_m)
            self.v_m = self.v_m + a_m * dt
            self.x_m = self.x_m + self.v_m * dt
            self.x_c = self.x_c + v_c * dt
            self.f_ch = f_ch
        else:
            self.f_ch = f_ch

        if p.outer.rigid:
            self.t += dt
            a_s = np.linalg.solve(G["Ms"], f_cmd - f_e)
            self.v = self.v + a_s * dt
            self.x = self.x + self.v * dt
            return rec

        a_r = np.linalg.solve(G["Ma"], f_drive - f_e - G["Br"] @ self.v_r)
        self.v_r = self.v_r + a_r * dt
        if np.isfinite(p.outer.v_limit):
            self.v_r = np.clip(self.v_r, -p.outer.v_limit, p.outer.v_limit)
        self.x_r = self.x_r + self.v_r * dt

        a_s = np.linalg.solve(G["Ms"], f_cmd - f_e)
        self.v = self.v + a_s * dt
        self.x = self.x + self.v * dt

        self.t += dt
        return rec

    # ---------------- convenience ----------------
    def run(self, duration: float, action_fn: Callable[[float, dict], Action] | None = None) -> dict:
        """Integrate for `duration` seconds, returning stacked logs.

        action_fn(t, last) -> Action is called every step in policy mode. It
        receives the previous step's record so a closed-loop policy can see the
        controller's internal state, which is itself one of the input-design
        questions under study.
        """
        n_steps = int(round(duration / self.dt))
        if n_steps <= 0:
            raise ValueError(f"duration {duration} is shorter than one dt ({self.dt})")
        recs: list[dict] = []
        last: dict | None = None
        for _ in range(n_steps):
            act = None
            if self.mode == "policy":
                if action_fn is None:
                    raise ValueError("policy mode needs action_fn")
                act = action_fn(self.t, last)
            last = self.step(act)
            recs.append(last)
        out = {k: np.array([r[k] for r in recs]) for k in recs[0]}
        if not np.all(np.isfinite(out["f_e"])):
            raise FloatingPointError("simulation diverged (non-finite f_e); dt too large?")
        return out


def quintic_pos(t: float, t0: float, t1: float, total: float) -> float:
    """Quintic (smooth-step) reference POSITION: 0 before t0, `total` after t1.

    Quintic rather than linear because Theorem 1 assumes differentiable applied
    references; a linear ramp puts a step into V_d, which is a delta in the
    reference power and shows up as an unbounded term in the energy balance.
    """
    if t <= t0:
        return 0.0
    if t >= t1:
        return total
    s = (t - t0) / (t1 - t0)
    return total * s ** 3 * (10.0 - 15.0 * s + 6.0 * s ** 2)


def quintic_vel(t: float, t0: float, t1: float, total: float) -> float:
    """Time derivative of quintic_pos -- the reference VELOCITY V_d."""
    if t <= t0 or t >= t1:
        return 0.0
    s = (t - t0) / (t1 - t0)
    return total * (30.0 * s ** 2 - 60.0 * s ** 3 + 30.0 * s ** 4) / (t1 - t0)


def steady_state_of(log: dict, key: str, window_s: float, dt: float) -> Array:
    """Mean of `key` over the final `window_s` seconds -- the empirical steady
    state to compare against analytic.py."""
    k = max(1, int(round(window_s / dt)))
    return np.asarray(log[key])[-k:].mean(axis=0)
