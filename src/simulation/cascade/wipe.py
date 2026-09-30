"""Tilted-surface wiping task: the contact scenario the anisotropy study (T1) needs.

Nothing in ManiSkill fits.  Its drawing tasks are explicitly kinematic -- see
draw.py's own comment, "we do not actually check if the robot contacts the
table" -- so stiffness cannot affect their success at all.  PegInsertionSide is
the only tabletop task with real contact, but it randomises the box about z
only, needs a grasp phase first, and scores success geometrically.  A wipe has
none of those problems: no grasp, a task frame that varies in two tilt angles,
and outcomes that depend directly on how compliance is oriented.

THE TASK.  A plane of unknown tilt.  The tool must hold a target NORMAL force
while sliding TANGENTIALLY along a stroke, against Coulomb friction.  The two
directions want opposite compliance, which is what makes anisotropy decisive:

    along the normal   -> soft, so an unknown surface height does not turn into
                          a force error
    along the tangent  -> stiff, so friction does not drag the tool off the
                          stroke path

Get the magnitude wrong and the task degrades.  Get the FRAME wrong -- apply the
soft axis along the tangent and the stiff axis along the normal -- and it fails
outright, in two different ways at once.  Separating those outcomes is the
experiment.

THE THREE FACTORS.  Every published stiffness-label rule returns a diagonal
matrix in some frame, so a label carries three things, not one:

    K = exp(s) * R diag(w) R^T ,   sum(log w) = 0

    s  magnitude   (trace)          -- the number the rules disagree about 4-6x
    w  ratio       (eigenvalue spread)
    R  frame       (eigenvector orientation)  -- the CHOICE nobody states

study_anisotropy.py varies the three independently against this task.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

import core as C

Array = np.ndarray


def rot_xy(tilt_x: float, tilt_y: float) -> Array:
    """Surface orientation from two tilt angles (about world x, then world y)."""
    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


@dataclass
class WipeTask:
    """One episode.  The task frame is [t1, t2, n]: two tangents and the normal.

    `normal` is the direction of PENETRATION, matching core.EnvParams, so the
    tool presses along +n into the surface.
    """
    tilt_x: float = 0.0          # rad
    tilt_y: float = 0.0          # rad
    f_target: float = 8.0        # N along the normal
    stroke_len: float = 0.10     # m, peak-to-peak
    stroke_hz: float = 0.25      # one full back-and-forth per 4 s
    mu: float = 0.3
    ke: float = 5000.0
    de: float = 10.0
    wall: float = 0.10           # TRUE plane position, at n . x = wall
    # Surface uncertainty.  Without it there is no reason to be soft along the
    # normal at all -- on a perfectly known flat plane a stiffer normal axis
    # simply tracks better, and the experiment would reward the wrong thing.
    # A real wiped surface is neither flat nor exactly where you believed.
    wave_amp: float = 0.0008     # m, undulation amplitude
    wave_len: float = 0.025      # m, undulation wavelength along the stroke
    height_error: float = 0.002  # m, how wrong the commanded surface height is
    # The dominant real uncertainty in wiping is not height but ORIENTATION.  A
    # believed normal a few degrees off means the surface height under the tool
    # drifts by millimetres across the stroke, at the stroke frequency -- inside
    # the outer layer's band, which is the only regime where the outer normal
    # stiffness controls the force error at all.  The fast waviness above sits
    # ABOVE that band and is answered by the inner layer and by inertia, so on
    # its own it makes the normal stiffness look irrelevant.
    belief_tilt_error: float = 0.0873  # rad (5 deg)

    # These two were CALIBRATED, and the calibration is part of the method: the
    # operating point is chosen so that both metrics are sensitive to their own
    # axis and neither requirement is trivially met or impossible.  At 0.8 mm /
    # 5 deg the force error spans 5.5x across the normal-stiffness range and the
    # path error spans 4.2x across the tangential range, with the expert inside
    # the success region on both with margin (0.34 and 0.61 of tolerance).
    # A task where only one metric responds would collapse the three-factor
    # question into a single number and answer it by construction.
    settle_s: float = 3.0        # press on before the stroke starts
    duration: float = 15.0
    Ki: float = 4000.0           # inner impedance (isotropic; the inner layer is
    Di: float = 180.0            # not what this study varies)
    Ms: float = 2.0
    Ma: float = 3.0
    Br: float = 25.0

    @property
    def R_task(self) -> Array:
        """TRUE task frame; columns are [t1, t2, n] in world coordinates."""
        return rot_xy(self.tilt_x, self.tilt_y)

    @property
    def R_belief(self) -> Array:
        """What the command believes the task frame is: the true frame rotated by
        belief_tilt_error about t2.  Everything the controller emits -- x_ref,
        f_d, and the label's own frame -- is expressed against THIS."""
        e = self.belief_tilt_error
        c, s = np.cos(e), np.sin(e)
        return self.R_task @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @property
    def normal(self) -> Array:
        return self.R_task[:, 2]

    @property
    def tangent(self) -> Array:
        return self.R_task[:, 0]

    @property
    def normal_belief(self) -> Array:
        return self.R_belief[:, 2]

    @property
    def tangent_belief(self) -> Array:
        return self.R_belief[:, 0]

    def stroke_offset(self, t: float) -> float:
        """Tangential position of the commanded stroke, m."""
        if t < self.settle_s:
            return 0.0
        return 0.5 * self.stroke_len * np.sin(2 * np.pi * self.stroke_hz * (t - self.settle_s))

    def believed_wall(self) -> float:
        """Where the command THINKS the surface is.  The controller only ever
        sees this; the true plane is `wall`."""
        return self.wall + self.height_error

    def surface_offset(self, x: Array) -> float:
        """Undulation of the true surface at the tool's tangential position."""
        if self.wave_amp == 0.0:
            return 0.0
        s = float(np.asarray(x) @ self.tangent)
        return self.wave_amp * np.sin(2 * np.pi * s / self.wave_len)

    def x_ref(self, t: float) -> Array:
        """Commanded reference: on the BELIEVED surface, sliding along its t1."""
        return (self.believed_wall() * self.normal_belief
                + self.stroke_offset(t) * self.tangent_belief)

    def f_d(self, t: float) -> Array:
        """Commanded force: normal only.  The tangential drag is left to the
        stiffness, which is the whole reason the tangential gain matters."""
        ramp = min(1.0, t / max(self.settle_s * 0.5, 1e-9))
        return ramp * self.f_target * self.normal_belief

    def params(self, dt: float | None = None) -> C.CascadeParams:
        base = C.CascadeParams(
            n=3,
            outer=C.OuterParams(Ma=self.Ma, Br=self.Br),
            inner=C.InnerParams(Ki=self.Ki, Di=self.Di, Ms=self.Ms),
            env=C.EnvParams(ke=self.ke, de=self.de, x_wall=self.wall,
                            normal=self.normal, mu=self.mu),
        )
        return dataclasses.replace(base, dt=C.safe_dt(base) if dt is None else dt)

    def wall_fn(self):
        """Hand to CascadeSim so the surface undulates under the tool."""
        return lambda t, x: self.surface_offset(x)


# --------------------------------------------------------------------------- #
# stiffness factorisation
# --------------------------------------------------------------------------- #
def compose(magnitude: float, ratio: float, frame: Array) -> Array:
    """Build K = exp(s) * R diag(w) R^T with the three factors separated.

    `magnitude` is the geometric mean stiffness (N/m).  `ratio` is tangential /
    normal stiffness; ratio > 1 means stiff along the tangents and soft along
    the normal, which is the compliance a wipe wants.  `frame` has columns
    [t1, t2, n].  det(diag(w)) == 1, so the magnitude and the ratio cannot leak
    into each other -- the factorisation is what the experiment rests on.
    """
    w = np.array([ratio, ratio, 1.0], dtype=float)
    w = w / np.prod(w) ** (1.0 / 3.0)     # unit determinant
    return magnitude * (frame @ np.diag(w) @ frame.T)


def decompose(K: Array) -> tuple[float, float, Array]:
    """Inverse of `compose`: (magnitude, ratio, frame) from an SPD matrix.

    magnitude = det(K)^(1/3), the geometric mean eigenvalue; ratio is the
    largest/smallest eigenvalue; the frame's columns are the eigenvectors,
    ordered so the SMALLEST eigenvalue is last -- i.e. the soft axis is the
    third column, matching compose()'s convention that the normal is soft.
    """
    Ksym = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(Ksym)
    order = np.argsort(w)[::-1]          # descending: soft axis ends up last
    w, V = w[order], V[:, order]
    if np.linalg.det(V) < 0:
        V[:, -1] *= -1.0
    magnitude = float(np.prod(w) ** (1.0 / 3.0))
    ratio = float(w[0] / max(w[-1], 1e-12))
    return magnitude, ratio, V


# --------------------------------------------------------------------------- #
# rollout and metrics
# --------------------------------------------------------------------------- #
@dataclass
class WipeResult:
    contact_loss: float      # fraction of stroke time out of contact
    force_rmse: float        # N, normal-force error against the target
    force_rel: float         # the same, relative to the target
    path_error: float        # m, RMS tangential deviation from the stroke
    peak_force: float        # N
    success: bool
    log: dict = field(repr=False, default_factory=dict)


def rollout(task: WipeTask, Ko: Array, contact_frac: float = 0.05,
            force_tol: float = 0.30, path_tol: float = 0.005) -> WipeResult:
    """Run one wipe with outer coupling stiffness `Ko` (a 3x3 world-frame matrix).

    Success needs all three: stay in contact, hold the normal force, and keep to
    the stroke.  One criterion alone would let a degenerate controller pass --
    a very soft K holds force beautifully while sliding off the path, and a very
    stiff K tracks the path while hammering the surface.
    """
    p = task.params()
    sim = C.CascadeSim(p, "policy", wall_offset_fn=task.wall_fn())
    n_hat, t_hat = task.normal, task.tangent
    sim.x_r = task.wall * n_hat.copy()
    sim.x = task.wall * n_hat.copy()
    sim._x_r0 = sim.x_r.copy()

    Ko = np.asarray(Ko, dtype=float)

    def action(t, last):
        return C.Action(task.x_ref(t), task.f_d(t), Ko)

    log = sim.run(task.duration, action)
    t = log["t"]
    stroke = t >= task.settle_s
    if not stroke.any():
        raise ValueError("duration shorter than settle_s: no stroke to score")

    f_n = np.asarray(log["f_normal"])[:, 0][stroke]
    x = np.asarray(log["x"])[stroke]
    ts = t[stroke]

    in_contact = f_n > 0.1 * task.f_target
    contact_loss = float(1.0 - in_contact.mean())

    err = f_n - task.f_target
    force_rmse = float(np.sqrt(np.mean(err ** 2)))

    t_bel = task.tangent_belief
    want = np.array([task.stroke_offset(float(ti)) for ti in ts])
    got = x @ t_bel - task.believed_wall() * float(task.normal_belief @ t_bel)
    path_error = float(np.sqrt(np.mean((got - want) ** 2)))

    ok = (contact_loss <= contact_frac
          and force_rmse <= force_tol * task.f_target
          and path_error <= path_tol)
    return WipeResult(contact_loss, force_rmse, force_rmse / task.f_target,
                      path_error, float(f_n.max()), bool(ok), log)


def expert_stiffness(task: WipeTask, magnitude: float = 700.0, ratio: float = 10.0) -> Array:
    """The best stiffness actually available: soft along the normal, stiff along
    the tangents, in the BELIEVED task frame.

    The believed frame, not the true one -- an operator cannot label against a
    surface orientation nobody knows.  belief_tilt_error is therefore a floor on
    how well any label can do, which is what makes the frame sweep meaningful:
    it measures tolerance to error ON TOP of the error everyone already has.
    """
    return compose(magnitude, ratio, task.R_belief)


# --------------------------------------------------------------------------- #
# demonstration: a synthetic operator wiping the surface
# --------------------------------------------------------------------------- #
@dataclass
class OperatorParams:
    """The operator whose stiffness the label rules are asked to recover.

    `magnitude` and `ratio` describe an ANISOTROPIC arm stiffness expressed in
    the TRUE task frame -- soft into the surface, stiff along the stroke.  That
    frame is the ground truth the rules are scored against.  The operator
    presses `press` past the surface, which is what sets the contact force
    through the whole series chain.
    """
    magnitude: float = 900.0
    ratio: float = 8.0
    frame_error_deg: float = 0.0
    Bh: float = 25.0
    press: float = 0.085
    probe_amp: float = 0.0               # N, per-axis device probe (0 = none)
    probe_hz: tuple = (2.7, 3.9, 5.3)    # distinct per axis; see MasterParams
    Ka: float = 300.0
    Ba: float = 40.0
    Mm: float = 2.0
    Bm: float = 10.0


def operator_stiffness(task: WipeTask, op: "OperatorParams") -> Array:
    th = np.deg2rad(op.frame_error_deg)
    c, s = np.cos(th), np.sin(th)
    R = task.R_task @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return compose(op.magnitude, op.ratio, R)


def demonstrate(task: WipeTask, op: "OperatorParams | None" = None) -> dict:
    """Run the full teleop chain and return the demonstration log.

    The operator follows a target TRAJECTORY -- press in, then stroke along the
    surface -- rather than reaching for one point, so the record carries both a
    sustained normal force and sustained tangential motion.  Both are needed: a
    rule keyed on force and a rule keyed on motion pick different frames from
    the same demonstration, and measuring that difference is the point.
    """
    op = op or OperatorParams()
    n_hat, t_hat = task.normal, task.tangent
    Kh = operator_stiffness(task, op)

    def target(t: float) -> Array:
        press = op.press * min(1.0, t / max(task.settle_s * 0.5, 1e-9))
        return (task.wall + press) * n_hat + task.stroke_offset(t) * t_hat

    base = C.CascadeParams(
        n=3,
        human=C.HumanParams(Kh=Kh, Bh=op.Bh, target_fn=target),
        master=C.MasterParams(Mm=op.Mm, Bm=op.Bm, probe_amp=op.probe_amp,
                              probe_hz=np.array(op.probe_hz, dtype=float)),
        coupling=C.CouplingParams(Ka=op.Ka, Ba=op.Ba),
        outer=C.OuterParams(Ma=task.Ma, Br=task.Br),
        inner=C.InnerParams(Ki=task.Ki, Di=task.Di, Ms=task.Ms),
        env=C.EnvParams(ke=task.ke, de=task.de, x_wall=task.wall,
                        normal=n_hat, mu=task.mu),
    )
    prm = dataclasses.replace(base, dt=C.safe_dt(base))
    sim = C.CascadeSim(prm, "teleop", wall_offset_fn=task.wall_fn())
    for attr in ("x_r", "x", "x_c", "x_m"):
        setattr(sim, attr, task.wall * n_hat.copy())
    sim._x_r0 = sim.x_r.copy()
    log = sim.run(task.duration)
    log["_dt"] = prm.dt
    log["_Kh_true"] = Kh
    log["_task"] = task
    log["_op"] = op
    return log
