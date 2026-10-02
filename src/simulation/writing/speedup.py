"""Theorem 1 of the retiming theory, tested in this simulator.

THE CLAIM (Accelerated Demonstration Reproduction on SE(3), Thm 1).  Run the
demonstration clock c times faster and, at the same time, set

    K_p, K_R -> c^2 K     D, D_q -> c D     U_N -> c^2 U_N     g_d,c(t) = g_d,0(ct)

and the closed loop reproduces the baseline orbit EXACTLY:

    q_c(t) = q_0(ct),      v_c(t) = c v_0(ct).

Pose error at matched demonstration phase is preserved -- finite rotations
included -- while physical twist error scales by c.  Proposition 2 says the
exponents 2 and 1 are then also necessary for this controller class, which has
no acceleration feedforward; Corollary 3 says torque scales as c^2, power as
c^3 and stored energy as c^2.

TWO OF THE FOUR LINES ARE ALREADY FREE IN controller.py.  Both damping designs
are built on a square root of the stiffness,

    D  = zeta (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2)      -> K  -> c^2 K gives D  -> c D
    Dr = 2 zeta sqrt(kr) Lambda_r^1/2                    -> kr -> c^2 kr gives Dr -> c Dr

and Lambda is unchanged at a matched configuration, so scaling the STIFFNESSES
is the whole prescription.  The test below verifies D_c = c D_0 from the log
rather than assuming it.

WHAT A RETIMER STILL HAS TO CARRY.  The action is (x_d, k_diag), so the policy
can scale K_p by itself.  Everything else lives in Case1Gains and does not:

    Kr / ctl.kr         the rotational stiffness of the geometric potential,
                        which is a controller STATE and so is written there
    kr_lo, kr_hi        its bounds, which bracket the schedule like k_lo/k_hi
    null_kp, null_kd    the posture potential U_N and its damping D_q
    k_lo, k_hi          the applied-stiffness bounds -- Eq. (11)'s K-bar
    E0, Ec              the tank: stored energy scales as c^2, power as c^3,
                        so an unscaled tank gates a scaled plan
    tau_limit           saturation is the theorem's delta_tau

Each is a flag in `Exponents`, so --ablate prices them one at a time.

WHY THE TEST CHANGES sim_freq.  Theorem 1 is a continuous-time statement.  The
implemented loop (law at t, physics, then integrate the reference) is conjugate
STEP FOR STEP only if the faster execution also steps c times faster, so that
one physics step covers the same phase increment.  That is the default here:
sim_freq -> c sim_freq.  `--fixed-rate` keeps 500 Hz and measures what is left,
which is the sampling part of the implementation discrepancy beta_imp of VI-C.

FREE MOTION means fe = 0.  The reference stays above the paper and every run
asserts the pen/paper contact force never leaves zero.

ONE HYPOTHESIS THIS SIMULATOR DOES NOT GRANT.  `--audit` finds that gravity is
exactly compensated (ManiSkill disables link gravity for a fixed-base arm, so
g_q = 0 and Corollary 3's torque law is the clean b T_0), but that SAPIEN gives
every link a default linear and angular damping of 0.05.  That is a torque
b qdot: it scales as c, NOT as c^2, so it is unmatched dissipation of exactly
the kind Assumption 1 excludes and the manuscript sends to an explicit
discrepancy model.  The identity runs therefore zero it; --keep-link-damping
puts it back and measures what that single term costs.

A scheduled R_d (--rot-amp) does work that Case 1's tank does not meter, since
Case1Proposal carries only a linear Vd.  The conjugacy is unaffected -- the
rotational spring and R_d' scale as c^2 and c like everything else -- but the
reported tank numbers are Case-1-exact only at --rot-amp 0.

Usage (from src/simulation/writing):
    python3 speedup.py --audit                   # Assumption 1, checked in this sim
    python3 speedup.py --c 2                     # the Table I / Fig. 1 analogue
    python3 speedup.py --c 2 --ablate Kr bounds tank null
    python3 speedup.py --c 2 --fixed-rate        # the sampling defect instead
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import controller as C  # noqa: E402
import sim as SM  # noqa: E402

Array = np.ndarray
BASE_HZ = 500


# --------------------------------------------------------------------------- #
# the scaling law
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Exponents:
    """Which coefficients the retimer actually scales.  All true is Theorem 1."""
    Kp: bool = True        # the action's k_diag    -> c^2   (the policy's own output)
    Kr: bool = True        # Case1Gains.Kr          -> c^2
    null: bool = True      # null_kp -> c^2, null_kd -> c    (U_N, D_q)
    bounds: bool = True    # k_lo, k_hi             -> c^2   (Eq. 11's K-bar)
    tank: bool = True      # E0, Ec                 -> c^2   (energy scales as c^2)
    tau: bool = True       # tau_limit              -> c^2   (delta_tau)

    @staticmethod
    def none() -> "Exponents":
        return Exponents(False, False, False, False, False, False)

    def drop(self, name: str) -> "Exponents":
        return dataclasses.replace(self, **{name: False})

    def missing(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if not getattr(self, f.name)]


FIELDS = [f.name for f in dataclasses.fields(Exponents)]


def scaled_gains(base: C.Case1Gains, c: float, e: Exponents, p: float = 2.0,
                 c_lo: float | None = None) -> C.Case1Gains:
    """Theorem 1's coefficient prescription, restricted to the flags in `e`.

    D and Dr are absent on purpose: the factorization and the 2 zeta sqrt(I Kr)
    designs make them follow K and Kr by themselves.

    `p` is the stiffness exponent, 2 in the theorem.  It exists so the converse
    can be swept: because both damping designs take a square root of the
    stiffness, choosing K -> c^p K forces D -> c^(p/2) D, so the implemented
    controller can only ever sit on the curve (p, p/2) and Theorem 1's (2, 1) is
    one point of it.  Proposition 2 says that point is the only one that
    preserves the operator; `--sweep` checks it.
    """
    b = c ** p
    b_lo = (c if c_lo is None else c_lo) ** p
    g = dataclasses.replace(base)
    if e.Kr:
        g.Kr = base.Kr * b          # reset_state seeds ctl.kr from this
    if e.null:
        g.null_kp, g.null_kd = base.null_kp * b, base.null_kd * c ** (0.5 * p)
    if e.bounds:
        # The bounds must BRACKET the whole schedule, so they scale with the
        # clock's extremes and not with one number.  Scaling the floor by the
        # PEAK rate instead silently clips the schedule back up wherever the
        # clock is slow -- the applied stiffness is then max(r^2 K_0, k_lo),
        # which is not the similarity law, and the orbit it produces is not the
        # baseline's.  A constant clock hides this, because then there is only
        # one rate and the floor never catches up with the schedule.
        g.k_lo, g.k_hi = base.k_lo * b_lo, base.k_hi * b
        g.kr_lo, g.kr_hi = base.kr_lo * b_lo, base.kr_hi * b
    if e.tank:
        g.E0, g.Ec = base.E0 * b, base.Ec * b
    if e.tau:
        g.tau_limit = base.tau_limit * b
    return g


# --------------------------------------------------------------------------- #
# the execution clock
# --------------------------------------------------------------------------- #
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


@dataclasses.dataclass
class Clock:
    """ds/dt = r(s) > 0: how fast demonstration phase is consumed.

    Theorem 1 is the constant-r case.  For a general r the phase-domain defect
    of Theorem 4 keeps a term -(r'/r) M p even when the gains are scheduled by
    the similarity law, and Eq. (15)'s u_clock = (rdot/r) M v = r'(s) M v is the
    active torque that cancels it.  `dr` is dr/ds, supplied analytically so the
    correction is not a finite difference of the thing it is correcting.
    """
    name: str
    r: object
    dr: object
    const: float | None = None

    @staticmethod
    def uniform(c: float) -> "Clock":
        c = float(c)
        return Clock(f"r = {c:g}", lambda s: c, lambda s: 0.0, c)

    @staticmethod
    def sine(a: float, S: float, n: int = 1) -> "Clock":
        """r(s) = 1 + a sin^2(n pi s / S): the manuscript's clock at n = 1.

        It starts and ends at the demonstrated rate, so the retiming is interior
        and no initial condition has to be reinterpreted, and its duration is
        S / sqrt(1 + a) for every n.

        `n` exists because the clock term is driven by r', not by r.  Raising n
        multiplies sup |d log r / ds| by n while leaving the rate range and the
        total duration alone -- which is the manuscript's own point, that the
        condition bounds |d log r/ds| = |a|/r^2 and not the speed factor. It is
        also what makes the term measurable here: the floor of this comparison
        is set by the baseline, and n raises the signal without raising it.
        """
        w = n * np.pi / S
        return Clock(f"r = 1 + {a:g} sin^2({n if n != 1 else ''}pi s / {S:g})",
                     lambda s: 1.0 + a * np.sin(w * s) ** 2,
                     lambda s: a * w * np.sin(2.0 * w * s))

    def peak(self, S: float) -> float:
        return float(max(self.r(x) for x in np.linspace(0.0, S, 2001)))

    def floor(self, S: float) -> float:
        return float(min(self.r(x) for x in np.linspace(0.0, S, 2001)))

    def duration(self, S: float, n: int = 200001) -> float:
        g = np.linspace(0.0, S, n)
        return float(TRAPZ(1.0 / np.array([self.r(x) for x in g]), g))

    def h_bound(self, S: float) -> float:
        """sup |d log r / ds| = sup |a| / r^2, the quantity Corollary 5 bounds --
        not the speed factor.  A fast but smoothly clocked execution can have a
        smaller h than a slow but abruptly clocked one."""
        g = np.linspace(0.0, S, 20001)
        return float(np.max([abs(self.dr(x)) / self.r(x) for x in g]))


# --------------------------------------------------------------------------- #
# a free-motion demonstration to reproduce
# --------------------------------------------------------------------------- #
def h5(u):
    """The quintic the manuscript uses: value, slope and curvature zero at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


class FreeReference:
    """A smooth SE(3) free-motion demonstration, parameterized by phase s.

    Built like the manuscript's constant-clock study: a travel along the paper's
    u axis with lateral and vertical excursions, noncommuting roll/pitch/yaw at
    three different frequencies, and a stiffness schedule that goes up and comes
    back so the stiffness RATE is exercised in both signs.  Everything starts
    with zero value AND zero slope, so a run from rest matches the reference
    initial condition exactly -- which Theorem 1 requires of both clocks.

    The height offset is never negative, so a reference anchored at the hover
    pose can only move away from the paper: fe = 0 by construction.
    """

    def __init__(self, S: float = 6.0, rot_amp_deg: float = 8.0,
                 k_a=(400.0, 400.0, 800.0), k_b=(1200.0, 1200.0, 2400.0),
                 du: float = 0.07, dv: float = 0.03, dh: float = 0.025):
        self.S = float(S)
        self.A = np.deg2rad(rot_amp_deg)
        self.k_a, self.k_b = np.asarray(k_a, float), np.asarray(k_b, float)
        self.du, self.dv, self.dh = du, dv, dh
        self.origin = None          # bound to the achieved tip pose at the first reset
        self.W = None               # the writing frame (u, v, n)
        self.R0 = None

    def bind(self, p0: Array, R0: Array, W: Array) -> None:
        if self.origin is None:
            self.origin, self.R0, self.W = np.asarray(p0, float).copy(), R0.copy(), W.copy()
        else:                       # every run must start from the same state
            assert np.allclose(self.origin, p0, atol=1e-12), "reset is not reproducible"

    # ---- the three prescribed functions of phase ----
    def x(self, s: float) -> Array:
        w = np.clip(s / self.S, 0.0, 1.0)
        d = np.array([self.du * h5(w),
                      self.dv * 0.5 * (1 - np.cos(2 * np.pi * w)),
                      self.dh * 0.5 * (1 - np.cos(4 * np.pi * w))])
        return self.origin + self.W @ d

    def R(self, s: float) -> Array:
        w = np.clip(s / self.S, 0.0, 1.0)
        a, b, g = (self.A * 0.5 * (1 - np.cos(n * np.pi * w)) for n in (2, 4, 6))
        return self.R0 @ rot_z(g) @ rot_y(b) @ rot_x(a)

    def k(self, s: float) -> Array:
        w = np.clip(s / self.S, 0.0, 1.0)
        return self.k_a + (self.k_b - self.k_a) * 0.5 * (1 - np.cos(2 * np.pi * w))


# --------------------------------------------------------------------------- #
# one execution
# --------------------------------------------------------------------------- #
SPEC = SM.TaskSpec(text="I", letter_height=0.035, time_limit=1e9, seed=0,
                   canvas_dz=0.0, tilt_x=0.0, tilt_y=0.0, show_template=False)


def set_link_damping(sim: SM.WritingSim, value: float | None) -> float:
    """SAPIEN's per-link viscous drag.  `None` leaves it alone and returns what
    it is; a number sets every link of the arm to it.  It survives reset, so it
    is applied once per run, after the reset."""
    links = [o for lk in sim.robot.get_links() for o in getattr(lk, "_objs", [lk])]
    if value is not None:
        for o in links:
            o.linear_damping = float(value)
            o.angular_damping = float(value)
    return float(links[0].linear_damping)


def run_free(sim: SM.WritingSim, ref: FreeReference, clock, e: Exponents,
             base: C.Case1Gains, link_damping: float | None = 0.0,
             p: float = 2.0, correct: bool = False) -> dict:
    """Execute `ref` under the clock `clock` and the gain scaling `e`.

    The proposals are deadbeat, exactly as evaluate.WritingPolicyEnv builds
    them: Vd = (x_ref(s + ds) - x_d)/dt and Up = (K_ref(s + ds) - K)/dt, so with
    an open gate the applied reference lands on the phase sample.  Under a
    constant retiming Vd picks up c and Up picks up c^3, which is Corollary 3's
    power law.

    The phase advances by ds = r(s) dt, explicit Euler and deliberately so: the
    physics takes q <- q + v dt, and that equals the phase update q <- q + p ds
    with p = v/r only when ds = r(s) dt exactly.  A "better" midpoint rule for
    the clock would desynchronise it from the integrator it has to agree with.

    GAIN SCHEDULE.  The similarity law is applied pointwise in phase: the
    stiffnesses become r(s)^p times the demonstrated ones, and the damping
    designs carry r(s)^(p/2) by themselves.  The coefficients that are bounds
    rather than schedules -- k_lo/k_hi, the tank, tau_limit -- are sized once at
    the clock's peak, so they never bind and never silently gate the plan.

    `correct` adds Eq. (15)'s u_clock = r'(s) M(q) qdot through Case 1's gated
    active slot.  It is evaluated at the measured state at the start of the
    step, which is where the control law is evaluated, and M is the joint-space
    inertia the theorem names -- not the operational-space Lambda.
    """
    if not isinstance(clock, Clock):
        clock = Clock.uniform(float(clock))
    r_peak = clock.peak(ref.S)
    r_floor = clock.floor(ref.S)
    g = scaled_gains(base, r_peak, e, p, c_lo=r_floor)
    sim.ctl.g = g
    sim.reset(SPEC)
    set_link_damping(sim, link_damping)
    _, p0, _, _ = sim.ctl.tip_state()
    q0 = sim.robot.get_qpos()[0].cpu().numpy()
    ref.bind(p0, SM.R_PEN_DOWN, sim.W)

    kpow = (lambda rv: rv ** p) if e.Kp else (lambda rv: 1.0)
    r0 = clock.r(0.0)
    sim.ctl.reset_state(ref.x(0.0), ref.R(0.0),
                        C.k_world(kpow(r0) * ref.k(0.0), sim.W), q_rest=q0)

    dt = sim.dt
    keys = ("q", "qd", "p", "R", "x_d", "K", "D", "tau", "tau_req", "f_raw", "u_tau")
    log = {k: [] for k in keys + ("s", "r", "alpha", "E", "p_prop")}
    # Every step is a FULL step: ds = r(s) dt is the identity the scheme rests on
    # (the physics takes q <- q + v dt, which is q <- q + p ds only for that ds),
    # so a final partial step to land exactly on S would break it for one step
    # and leave an endpoint spike that dominates a max-over-phase. The run stops
    # at the last whole step instead, and the baseline uses the same rule, which
    # keeps the retimed phase grid inside the baseline's.
    s = 0.0
    while s + float(clock.r(s)) * dt <= ref.S + 1e-12:
        rv = float(clock.r(s))
        s_next = s + rv * dt
        # the law is evaluated at phase s, so the scheduled coefficients are too
        if e.Kr:
            # controller.py promoted the rotational stiffness to a STATE with
            # its own rate slot, so compute() reads ctl.kr and no longer reads
            # g.Kr.  Scheduling has to be written there; g.Kr only seeds it.
            g.Kr = base.Kr * rv ** p
            sim.ctl.kr = base.Kr * rv ** p
        if e.null:
            g.null_kp = base.null_kp * rv ** p
            g.null_kd = base.null_kd * rv ** (0.5 * p)
        u_tau = None
        if correct:
            q = sim.robot.get_qpos()[0].cpu().numpy().astype(float)
            qd = sim.robot.get_qvel()[0].cpu().numpy().astype(float)
            u_tau = float(clock.dr(s)) * (sim.ctl.pm.compute_generalized_mass_matrix(q) @ qd)
        # R_d is the commanded equilibrium AT THE CURRENT PHASE, like x_d: the law
        # is evaluated at s, and sim.step's x_d is the value the previous advance()
        # left at s.  Reading the rotation one sample ahead instead advances it by
        # dt while the clock advances the rest by r dt, which is a g_d the two runs
        # do not share.
        sim.ctl.R_d = ref.R(s)
        x_next = ref.x(s_next)
        K_next = C.k_world(kpow(float(clock.r(s_next))) * ref.k(s_next), sim.W)
        rec = sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / dt,
                                       (K_next - sim.ctl.K) / dt, u_tau=u_tau))
        log["s"].append(s)
        log["r"].append(rv)
        for key in keys:
            log[key].append(np.asarray(rec[key], float).copy())
        for key in ("alpha", "E", "p_prop"):
            log[key].append(float(rec[key]))
        s = s_next
    out = {k: np.asarray(v) for k, v in log.items()}
    out["dt"], out["sim_freq"] = dt, 1.0 / dt
    out["c"] = clock.const if clock.const is not None else float("nan")
    out["clock"], out["corrected"] = clock.name, correct
    out["exec_s"] = float(len(out["s"]) * dt)
    out["exp_p"] = p          # NOT "p": that key is the logged tip position
    out["exps"] = e
    # the reference, sampled on the same phase grid, for the error curves
    out["x_ref"] = np.array([ref.x(x) for x in out["s"]])
    out["R_ref"] = np.array([ref.R(x) for x in out["s"]])
    return out


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def rot_angle(Ra: Array, Rb: Array) -> Array:
    """Geodesic angle between stacks of rotations, rad.

    Through the chordal norm, not arccos(tr/2 - 1/2).  Near identity arccos has
    an infinite slope, so float32 poses report a 1e-3 rad error where the two
    rotations are bit-identical -- which would hide exactly the agreement this
    file is written to measure.  ||E - I||_F = 2 sqrt(2) sin(theta/2) is exact
    on all of [0, pi] and well conditioned at zero.
    """
    E = np.einsum("...ji,...jk->...ik", Ra, Rb)
    d = np.linalg.norm(E - np.eye(3), axis=(-2, -1))
    return 2.0 * np.arcsin(np.clip(d / (2.0 * np.sqrt(2.0)), -1.0, 1.0))


def tracking_error(log: dict) -> tuple[Array, Array]:
    """Error from the commanded equilibrium: metres and radians, per phase sample."""
    return (np.linalg.norm(log["p"] - log["x_ref"], axis=1),
            rot_angle(log["R"], log["R_ref"]))


def align(a: dict, b: dict) -> tuple[dict, dict]:
    """Put two runs on a common phase grid.  With sim_freq scaled by c the grids
    are already identical; at a fixed rate the coarser one is sub-sampled from
    the finer when c is an integer, and interpolated otherwise."""
    sa, sb = a["s"], b["s"]
    if len(sa) == len(sb) and np.allclose(sa, sb, atol=1e-12):
        return a, b
    keys = [k for k in a if isinstance(a[k], np.ndarray) and a[k].shape[:1] == sa.shape]
    idx = np.searchsorted(sa, sb)
    idx = np.clip(idx, 0, len(sa) - 1)
    if np.allclose(sa[idx], sb, atol=1e-9):           # exact sub-sample
        return {k: (a[k][idx] if k in keys else a[k]) for k in a}, b
    # A nonuniform clock gives a nonuniform phase grid, so the baseline has to be
    # resampled.  Cubic, not linear: linear interpolation of a smooth orbit at
    # this step size leaves ~1e-7 m, which is the size of the residual the clock
    # correction is supposed to be judged on.
    from scipy.interpolate import CubicSpline
    out = {}
    for k in a:
        if k not in keys:
            out[k] = a[k]
            continue
        flat = a[k].reshape(len(sa), -1)
        out[k] = CubicSpline(sa, flat, axis=0)(sb).reshape((len(sb),) + a[k].shape[1:])
    out["s"] = sb
    return out, b


def conjugacy(base_log: dict, fast_log: dict) -> dict:
    """How far the retimed run is from the baseline orbit at matched phase.

    Written in the clock's own terms, so it covers both studies: the physical
    twist is normalized by r(s) and the coefficients by r(s)^2, which reduces to
    Theorem 1's c and c^2 when r is constant.  The twist comparison IS the
    theorem's statement -- phase-normalized twist error is preserved while
    physical twist error scales -- so it has to be the normalized one.
    """
    a, b = align(base_log, fast_log)
    r = b["r"]
    dq = np.abs(b["q"] - a["q"]).max()
    dv = np.abs(b["qd"] / r[:, None] - a["qd"]).max()
    dp = np.linalg.norm(b["p"] - a["p"], axis=1).max()
    dR = rot_angle(b["R"], a["R"]).max()
    # the coefficient laws the theorem also asserts, read back from the log
    rel = lambda x, y: float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-12))
    dD = rel(b["D"] / r[:, None, None], a["D"])
    dK = rel(b["K"] / (r ** 2)[:, None, None], a["K"])
    dtau = rel(b["tau"] / (r ** 2)[:, None], a["tau"])
    return dict(dq=float(dq), dv=float(dv), dp_m=float(dp), dR_rad=float(dR),
                rel_D=float(dD), rel_K=float(dK), rel_tau=float(dtau),
                u_tau_peak=float(np.abs(b["u_tau"]).max()),
                orbit_mm=float(1000 * np.linalg.norm(b["p"] - a["p"], axis=1).max()),
                orbit_deg=float(np.rad2deg(rot_angle(b["R"], a["R"]).max())))


def resources(base_log: dict, fast_log: dict) -> dict:
    """Corollary 3's ratios, measured: speed c, torque c^2, power c^3, energy c^2."""
    c = fast_log["c"]
    r = lambda k, f: (f(fast_log[k]) / f(base_log[k]) if abs(f(base_log[k])) > 1e-12 else np.nan)
    peak = lambda x: float(np.abs(x).max())
    drained = lambda L: float(L["E"][0] - L["E"][-1])
    return dict(
        c=float(c), speed=r("qd", peak), torque=r("tau_req", peak),
        power=r("p_prop", peak),
        energy=(drained(fast_log) / drained(base_log) if abs(drained(base_log)) > 1e-12 else np.nan),
        expect=dict(speed=float(c), torque=float(c ** 2), power=float(c ** 3),
                    energy=float(c ** 2)),
        alpha_min=float(fast_log["alpha"].min()),
        tank_left=float(fast_log["E"][-1]),
        sat=float(np.abs(fast_log["tau"] - fast_log["tau_req"]).max()),
        contact=float(np.abs(fast_log["f_raw"]).max()))


def admissible_b(base_log: dict, base: C.Case1Gains, ref: FreeReference,
                 v_lim: Array | None = None) -> dict:
    """Corollary 3's exact uniform-scaling set (Eq. 11) for this baseline.

    b = c^2.  Gravity is absent from this robot (ManiSkill disables link gravity
    for a fixed-base arm), so the torque test is the clean b |T_0| <= tau-bar
    rather than the general g_q + b T_0 form.
    """
    v_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]) if v_lim is None else v_lim
    qd, tau = np.abs(base_log["qd"]).max(0), np.abs(base_log["tau_req"]).max(0)
    b_vel = float((v_lim / np.maximum(qd, 1e-12)).min() ** 2)
    b_tau = float((np.asarray(base.tau_limit) / np.maximum(tau, 1e-12)).min())
    k_peak = float(np.max([ref.k(s).max() for s in np.linspace(0, ref.S, 201)]))
    b_k = base.k_hi / k_peak
    b = min(b_vel, b_tau, b_k)
    return dict(b_velocity=b_vel, b_torque=b_tau, b_stiffness=float(b_k),
                b_max=float(b), c_max=float(np.sqrt(b)),
                binding=min((("velocity", b_vel), ("torque", b_tau),
                             ("stiffness", b_k)), key=lambda t: t[1])[0])


# --------------------------------------------------------------------------- #
# Assumption 1, checked rather than assumed
# --------------------------------------------------------------------------- #
def kinetic(sim: SM.WritingSim, q: Array, qd: Array) -> float:
    return 0.5 * float(qd @ sim.ctl.pm.compute_generalized_mass_matrix(q) @ qd)


def coast(sim: SM.WritingSim, v0: Array, n: int, damping: float | None) -> dict:
    """Let the arm coast with zero commanded torque, logging q and kinetic energy.

    With gravity off, no contact and no model damping, free motion conserves
    kinetic energy exactly; whatever it loses is the integrator's, not a force
    in the model.
    """
    import torch
    sim.reset(SPEC)
    set_link_damping(sim, damping)
    sim.robot.set_qvel(torch.tensor(np.asarray(v0, np.float32)[None]))
    zero = torch.zeros((1, sim.ctl.nq), dtype=torch.float32)
    qs, E = [], []
    for _ in range(n):
        q = sim.robot.get_qpos()[0].cpu().numpy().astype(float)
        qd = sim.robot.get_qvel()[0].cpu().numpy().astype(float)
        qs.append(q)
        E.append(kinetic(sim, q, qd))
        sim.robot.set_qf(zero)
        sim.u.scene.step()
    return dict(q=np.asarray(qs), E=np.asarray(E))


def audit(sim_at, c: float = 2.0, S: float = 1.0) -> dict:
    """The hypotheses of Theorem 1, checked rather than assumed.

    Three separable questions, in the order they can break the theorem:

      gravity      free fall at zero torque.  Assumption 1 wants g_q removed;
                   ManiSkill disables link gravity for a fixed-base arm, so it
                   is removed exactly and Corollary 3's torque law is b T_0.
      dissipation  does coasting conserve kinetic energy?  SAPIEN gives every
                   link a default drag of 0.05, which is a torque b qdot: it
                   scales as c, not c^2, and is exactly the unmatched term
                   Assumption 1 excludes.  Zeroing it leaves only the
                   integrator's own loss, which is not a force in the model.
      conjugacy    is the INTEGRATOR itself conjugate?  Coast from v0 at dt and
                   from c v0 at dt/c and compare q step by step, with no
                   controller in the loop.  If this fails, nothing downstream
                   can succeed; if it passes at the float32 noise floor, that
                   floor -- not the theorem -- is what the full test can reach.
    """
    import torch
    base = sim_at(BASE_HZ)
    fast = sim_at(int(round(BASE_HZ * c)))

    base.reset(SPEC)
    q0 = base.robot.get_qpos()[0].cpu().numpy().astype(float)
    zero = torch.zeros((1, base.ctl.nq), dtype=torch.float32)
    drag = set_link_damping(base, None)
    base.robot.set_qvel(zero)
    for _ in range(400):
        base.robot.set_qf(zero)
        base.u.scene.step()
    fall = float(np.abs(base.robot.get_qpos()[0].cpu().numpy() - q0).max())

    v0 = 0.4 * np.random.default_rng(0).standard_normal(base.ctl.nq)
    n = int(round(S * BASE_HZ))
    with_drag = coast(base, v0, n, drag)
    no_drag = coast(base, v0, n, 0.0)
    rel = lambda E: float(abs(E[-1] - E[0]) / E[0])

    slow = no_drag                                   # dt, v0
    quick = coast(fast, c * v0, n, 0.0)              # dt/c, c v0  -> q_c(t) = q_0(ct)
    dq = float(np.abs(quick["q"] - slow["q"]).max())

    return dict(
        gravity_fall_rad=fall, gravity_off=bool(fall < 1e-4),
        sapien_link_damping=drag,
        ke_drift_with_drag=rel(with_drag["E"]), ke_drift_no_drag=rel(no_drag["E"]),
        coast_horizon_s=S, integrator_c=float(c), integrator_dq_rad=dq,
        integrator_conjugate=bool(dq < 1e-4))


# --------------------------------------------------------------------------- #
# Corollary 5: how fast may the clock vary WITHOUT the active correction
# --------------------------------------------------------------------------- #
def corollary5(S: float = 6.0, A: float = 1.4, n: int = 3, m: float = 1.0,
               k: float = 400.0, zeta: float = 1.4, amp: float = 0.12) -> dict:
    """The comparison bound, instantiated on a scalar GIC exactly as the
    manuscript does -- and for the same reason: Corollary 5 needs a contraction
    rate for the baseline in a declared metric, and contraction is an extra
    hypothesis, not something GIC passivity hands you on a 7-DOF arm.

    This is the result that makes retiming usable in contact, where supplying
    r' M v is exactly what one does not want to do: it says when scheduled gains
    alone approximate similarity, and by how much they miss.

    Scalar GIC with a clock, in phase coordinates (x' = p, p = xdot/r):

        x' = p,     m p' = -k (x - x_d(s)) - d p   -   (r'/r) m p

    The last term is the whole defect: the exact phase field is F_0(s,y) - h Pi_v y
    with h = r'/r and Pi_v = diag(0, 1), which is Corollary 5's hypothesis
    verbatim.  Subtracting the baseline orbit and applying the incremental
    contraction inequality gives

        ||y(s) - y_0(s)||_P  <=  e^{-lam* s} E_0 + (hbar V_P / lam*) (1 - e^{-lam* s})

    with lam* = lam - hbar L_Pi, L_Pi = ||P^1/2 Pi_v P^-1/2||, V_P = sup ||Pi_v y_0||_P.
    The metric is the eigenvector metric of the baseline A, which for an
    overdamped second-order system attains lam = -max Re eig(A) exactly.
    """
    from scipy.integrate import solve_ivp
    d = zeta * 2.0 * np.sqrt(m * k)
    clock = Clock.sine(A, S, n)
    xd = lambda s: amp * h5(s / S)
    A_m = np.array([[0.0, 1.0], [-k / m, -d / m]])
    w, V = np.linalg.eig(A_m)
    assert np.all(np.isreal(w)), "pick an overdamped case so the metric is real"
    w, V = w.real, V.real
    lam = float(-w.max())                       # contraction rate in this metric
    Vi = np.linalg.inv(V)
    P = Vi.T @ Vi                               # ||y||_P = ||V^-1 y||
    Ph = sym_sqrt_np(P)
    Pih = np.linalg.inv(Ph)
    Pi_v = np.diag([0.0, 1.0])
    L_Pi = float(np.linalg.norm(Ph @ Pi_v @ Pih, 2))
    h_bar = clock.h_bound(S)
    lam_star = lam - h_bar * L_Pi

    f0 = lambda s, y: [y[1], (-k * (y[0] - xd(s)) - d * y[1]) / m]
    fh = lambda s, y: [y[1], (-k * (y[0] - xd(s)) - d * y[1]) / m
                       - (clock.dr(s) / clock.r(s)) * y[1]]
    grid = np.linspace(0.0, S, 20001)
    y0 = solve_ivp(f0, (0, S), [0.0, 0.0], t_eval=grid, rtol=1e-11, atol=1e-13).y
    yh = solve_ivp(fh, (0, S), [0.0, 0.0], t_eval=grid, rtol=1e-11, atol=1e-13).y
    nrm = lambda z: np.sqrt(np.einsum("in,ij,jn->n", z, P, z))
    err = nrm(yh - y0)
    V_P = float(nrm(Pi_v @ y0).max())
    bound = (h_bar * V_P / lam_star) * (1.0 - np.exp(-lam_star * grid)) if lam_star > 0 else None
    return dict(lam=lam, L_Pi=L_Pi, h_bar=h_bar, lam_star=float(lam_star), V_P=V_P,
                feasible=bool(lam_star > 0), m=m, k=k, zeta=zeta, d=float(d),
                peak_err=float(err.max()),
                peak_bound=float(bound.max()) if bound is not None else float("inf"),
                s=grid, err=err, bound=bound, clock=clock.name)


def sym_sqrt_np(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def peak_table(rows: list[tuple[str, dict]]) -> str:
    out = [f"  {'execution':<34}{'peak position (mm)':>20}{'peak orientation (deg)':>24}"]
    for name, log in rows:
        ep, er = tracking_error(log)
        out.append(f"  {name:<34}{1000 * ep.max():>20.4f}{np.rad2deg(er).max():>24.4f}")
    return "\n".join(out)


def plot_clock(base_log, runs, clock, S, out_png) -> None:
    """The manuscript's Fig. 2: the clock, and what it costs with and without
    the correction, both against the baseline orbit at matched phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    g = np.linspace(0.0, S, 1001)
    ax[0].plot(g, [clock.r(x) for x in g], color="#2a78d6")
    ax[0].set_ylabel("clock rate r(s)")
    style = [dict(color="#8a8880", lw=1.4, label="uniform control: the floor"),
             dict(color="#2a78d6", lw=1.8, label="scaled gains only"),
             dict(color="#eb6834", lw=1.8, ls="--", label="with clock correction")]
    for (name, log), st in zip(runs, style):
        st = dict(st, label=name)
        a, b = align(base_log, log)
        dp = 1000 * np.linalg.norm(b["p"] - a["p"], axis=1)
        dr = np.rad2deg(rot_angle(b["R"], a["R"]))
        ax[1].plot(b["s"], dp, **st)
        ax[2].plot(b["s"], dr, **st)
    ax[1].set_ylabel("orbit difference (mm)")
    ax[2].set_ylabel("orbit rotation difference (deg)")
    for a_ in ax:
        a_.set_xlabel("demonstration time s (s)")
        a_.grid(alpha=0.25)
    for a_ in ax[1:]:
        a_.legend(fontsize=8)
    fig.suptitle(f"Nonuniform clock {clock.name}: the correction cancels the "
                 f"analytically derived clock term", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}")


def plot_sweep(ps, defects, c, out_png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.semilogy(ps, np.maximum(defects, 1e-18), "o-", color="#2a78d6")
    ax.axvline(2.0, color="#eb6834", lw=1.0, ls=":")
    ax.annotate("Theorem 1", (2.0, max(defects)), color="#eb6834", fontsize=8,
                ha="center", va="top")
    ax.set_xlabel("stiffness exponent p   (K -> c^p K, hence D -> c^(p/2) D)")
    ax.set_ylabel("max |p_c(s) - p_0(s)|  (m)")
    ax.set_title(f"Proposition 2, numerically: only p = 2 preserves the orbit (c = {c:g})",
                 fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}")


def plot(rows, out_png, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5ad6", "#8a8880"]
    for i, (name, log) in enumerate(rows):
        ep, er = tracking_error(log)
        w = log["s"] / log["s"][-1]
        st = dict(color=colors[i % len(colors)], lw=2.6 if i == 0 else 1.6,
                  ls="-" if i != 2 else "--", label=name)
        ax[0].plot(w, 1000 * ep, **st)
        ax[1].plot(w, np.rad2deg(er), **st)
    ax[0].set_ylabel("position error (mm)")
    ax[1].set_ylabel("rotation error (deg)")
    for a in ax:
        a.set_xlabel("demonstration phase")
        a.legend(fontsize=8)
        a.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"  wrote {out_png}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=2.0, help="execution-rate factor")
    ap.add_argument("--S", type=float, default=6.0, help="demonstration duration, s")
    ap.add_argument("--rot-amp", type=float, default=8.0, help="R_d schedule amplitude, deg")
    ap.add_argument("--fixed-rate", action="store_true",
                    help="keep 500 Hz for the fast runs: measures the sampling defect")
    ap.add_argument("--ablate", nargs="*", default=[], choices=FIELDS,
                    help="also run with each of these coefficients left unscaled")
    ap.add_argument("--keep-link-damping", action="store_true",
                    help="leave SAPIEN's 0.05 link drag in: unmatched dissipation, scales as c")
    ap.add_argument("--clock", type=float, default=None, metavar="A",
                    help="nonuniform-clock study with r(s) = 1 + A sin^2(pi s/S)")
    ap.add_argument("--clock-n", type=int, default=1,
                    help="half-periods of the clock: raises r' and the clock term "
                         "without touching the rate range, the duration or the floor")
    ap.add_argument("--hz", nargs="*", type=int, default=[1000, 2000],
                    help="physics rates for --clock.  The floor of this comparison "
                         "grows with the rate, so lower is sharper -- but only down "
                         "to the rate that still resolves the r^2-scaled loop")
    ap.add_argument("--sweep", nargs="*", type=float, default=None,
                    metavar="P", help="sweep the stiffness exponent p instead "
                                      "(default 1 1.5 1.75 1.9 2 2.1 2.25 2.5 3)")
    ap.add_argument("--corollary5", action="store_true",
                    help="the comparison bound for scheduled gains WITHOUT the "
                         "active clock correction, on a scalar GIC")
    ap.add_argument("--audit", action="store_true", help="check Assumption 1 and exit")
    ap.add_argument("--out", default="logs/speedup")
    args = ap.parse_args()

    base = C.Case1Gains()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.corollary5:
        r = corollary5(S=args.S, A=args.clock if args.clock else 1.4, n=args.clock_n)
        print("Corollary 5: scheduled gains without the active correction\n")
        print(f"  scalar GIC  m {r['m']:g} kg, k {r['k']:g} N/m, zeta {r['zeta']:g} "
              f"-> d {r['d']:.2f} N s/m")
        print(f"  clock       {r['clock']}")
        print(f"  metric      eigenvector metric of the baseline A, contraction rate "
              f"lam = {r['lam']:.4f} 1/s")
        print(f"              L_Pi = {r['L_Pi']:.4f},  V_P = {r['V_P']:.5f}")
        print(f"  clock       hbar = sup |d log r / ds| = {r['h_bar']:.4f} 1/s")
        print(f"  -> lam* = lam - hbar L_Pi = {r['lam_star']:.4f} "
              f"{'> 0, the bound holds' if r['feasible'] else '<= 0, NO BOUND: the clock varies too fast for this metric'}")
        if r["feasible"]:
            print(f"\n  peak metric error {r['peak_err']:.6e}")
            print(f"  comparison bound  {r['peak_bound']:.6e}   "
                  f"(conservative by {r['peak_bound'] / max(r['peak_err'], 1e-30):.1f}x)")
            print("\n  The bound is driven by hbar = sup |d log r/ds| = |a|/r^2, NOT by the")
            print("  speed factor: a fast but smoothly clocked execution can satisfy it where")
            print("  a slow but abruptly clocked one cannot.  That is the knob a retimer has")
            print("  in contact, where supplying r' M v actively is the thing to avoid.")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6.2, 4))
            ax.plot(r["s"], r["err"], color="#2a78d6", label="actual metric error")
            ax.plot(r["s"], r["bound"], color="#eb6834", ls="--", label="Corollary 5 bound")
            ax.set_xlabel("demonstration time s (s)")
            ax.set_ylabel("metric error")
            ax.set_yscale("log")
            top = max(r["bound"].max(), r["err"].max())
            ax.set_ylim(top * 1e-4, top * 2)     # both start at zero; the decades
            ax.grid(alpha=0.25, which="both")    # below the signal are not the story
            ax.legend(fontsize=8)
            ax.set_title("Scheduled gains, no clock correction: bound vs truth", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / "corollary5.png", dpi=120)
            print(f"  wrote {out / 'corollary5.png'}")
        (out / "corollary5.json").write_text(json.dumps(
            {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}, indent=1))
        return

    if args.audit:
        sims: dict[int, SM.WritingSim] = {}

        def sim_at(hz):
            if hz not in sims:
                sims[hz] = SM.WritingSim(cameras=False, sim_freq=hz)
            return sims[hz]

        a = audit(sim_at, c=args.c)
        print("Assumption 1 in this simulator\n")
        print(f"  gravity       zero-torque fall over 0.8 s: {a['gravity_fall_rad']:.3e} rad")
        print(f"                -> {'REMOVED exactly (ManiSkill disables link gravity): g_q = 0' if a['gravity_off'] else 'ACTIVE: g_q does not scale and breaks the c^2 law'}")
        print(f"  dissipation   coasting kinetic-energy drift over {a['coast_horizon_s']} s")
        print(f"                SAPIEN link damping {a['sapien_link_damping']:.3f}: "
              f"{100 * a['ke_drift_with_drag']:.3f} %   <- a b*qdot torque, scales as c NOT c^2")
        print(f"                zeroed:                 "
              f"{100 * a['ke_drift_no_drag']:.3f} %   <- the integrator's own loss, not a model force")
        print(f"  conjugacy     PhysX alone, no controller: coast at (dt, v0) vs (dt/{a['integrator_c']:g}, "
              f"{a['integrator_c']:g} v0)")
        print(f"                max |q_c(t) - q_0(ct)| = {a['integrator_dq_rad']:.3e} rad"
              f"  -> {'CONJUGATE at the float32 state floor' if a['integrator_conjugate'] else 'NOT CONJUGATE: the identity cannot be reached'}")
        (out / "audit.json").write_text(json.dumps(a, indent=1))
        for sv in sims.values():
            sv.close()
        return

    c = args.c
    fast_hz = BASE_HZ if args.fixed_rate else int(round(BASE_HZ * c))
    assert abs(fast_hz - BASE_HZ * c) < 1e-9 or args.fixed_rate, \
        f"c * {BASE_HZ} must be an integer sim rate (got {BASE_HZ * c})"
    ref = FreeReference(S=args.S, rot_amp_deg=args.rot_amp)
    drag = None if args.keep_link_damping else 0.0

    sims: dict[int, SM.WritingSim] = {}

    def sim_at(hz):
        if hz not in sims:
            sims[hz] = SM.WritingSim(cameras=False, sim_freq=hz)
        return sims[hz]

    print(f"link damping: {'SAPIEN default 0.05 (unmatched, scales as c)' if drag is None else 'zeroed (Assumption 1)'}")
    if args.clock is None:
        print(f"baseline {BASE_HZ} Hz, fast {fast_hz} Hz, c = {c}, S = {args.S} s "
              f"-> execution time {args.S / c:.4f} s")
    base_log = (None if args.clock is not None
                else run_free(sim_at(BASE_HZ), ref, 1.0, Exponents(), base, drag))

    if args.clock is not None:
        clk = Clock.sine(args.clock, args.S, args.clock_n)
        T = clk.duration(args.S)
        c_eq = args.S / T
        print(f"\nTheorem 4, the nonuniform clock:  {clk.name}")
        print(f"  peak rate {clk.peak(args.S):.4f},  sup |d log r/ds| = {clk.h_bound(args.S):.4f} 1/s")
        print(f"  execution {T:.6f} s against a {args.S:g} s demonstration -> {c_eq:.4f}x overall")
        print("\n  Gains follow the similarity law pointwise in phase, K = r(s)^2 K_0.")
        print("  That alone leaves Theorem 4's -(r'/r) M p term; Eq. (15)'s")
        print("  u_clock = r'(s) M(q) qdot is the active torque that cancels it.")
        print("\n  THE FLOOR.  Unlike the constant-clock test, a nonuniform clock cannot")
        print("  give the two runs the same phase grid, so they take different numbers of")
        print("  float32 physics steps and their roundoff no longer cancels.  The control")
        print(f"  is a UNIFORM clock at the same mean rate {c_eq:.4f}x on the same physics")
        print("  rate: Theorem 1 says its true defect is exactly zero, so whatever it")
        print("  reports is this comparison's floor and nothing below it means anything.\n")
        print(f"  {'physics':>9}{'steps':>8}{'floor':>11}{'no correction':>15}{'corrected':>11}"
              f"{'signal/floor':>14}{'peak u_clock':>14}{'K = r^2 K_0':>13}")
        print(f"  {'(Hz)':>9}{'base/re':>8}{'(mm)':>11}{'(mm)':>15}{'(mm)':>11}{'':>14}"
              f"{'(N m)':>14}{'(rel)':>13}")
        table, last = [], None
        for hz in args.hz:
            sm = sim_at(hz)
            bl = run_free(sm, ref, Clock.uniform(1.0), Exponents(), base, drag)
            ctl_ = run_free(sm, ref, Clock.uniform(c_eq), Exponents(), base, drag)
            off = run_free(sm, ref, clk, Exponents(), base, drag, correct=False)
            on = run_free(sm, ref, clk, Exponents(), base, drag, correct=True)
            k_fl, k_off, k_on = (conjugacy(bl, x) for x in (ctl_, off, on))
            table.append(dict(hz=hz, floor=k_fl, off=k_off, on=k_on))
            print(f"  {hz:>9d}{len(bl['s'])}/{len(off['s']):<4d}{k_fl['orbit_mm']:>11.5f}"
                  f"{k_off['orbit_mm']:>15.5f}{k_on['orbit_mm']:>11.5f}"
                  f"{k_off['orbit_mm'] / max(k_fl['orbit_mm'], 1e-30):>14.1f}"
                  f"{k_on['u_tau_peak']:>14.4f}{k_on['rel_K']:>13.2e}")
            last = (bl, ctl_, off, on, k_fl, k_off, k_on)
        bl, ctl_, off, on, k_fl, k_off, k_on = last
        hz = args.hz[-1]
        print(f"\n  The floor GROWS with the physics rate (more steps, more float32), so the")
        print(f"  sharpest test is the LOWEST rate that still resolves the loop -- and the")
        print(f"  scaled loop, not the baseline one.  The rotational stiffness peaks at")
        print(f"  r_peak^2 Kr here, and at 500 Hz that loop is no longer resolved: the")
        print(f"  nonuniform runs there report millimetres and the correction makes them")
        print(f"  worse, which is a sampled-controller artefact and not a defect of the")
        print(f"  model.  Raising --clock-n raises r' and the clock term with it, while")
        print(f"  leaving the rate range, the duration and the floor alone.")
        print(f"\n  at {hz} Hz, in full:")
        print(f"  {'run':<30}{'max |dq| (rad)':>16}{'|dv/r| (rad/s)':>16}{'tip (m)':>12}"
              f"{'rot (deg)':>12}{'alpha_min':>11}")
        for nm, kk, lg in ((f"uniform {c_eq:.3f}x control (floor)", k_fl, ctl_),
                           ("nonuniform, scaled gains", k_off, off),
                           ("nonuniform + clock correction", k_on, on)):
            print(f"  {nm:<30}{kk['dq']:>16.3e}{kk['dv']:>16.3e}{kk['dp_m']:>12.3e}"
                  f"{kk['orbit_deg']:>12.3e}{lg['alpha'].min():>11.6f}")
        dEb, dEon = (float(bl["E"][0] - bl["E"][-1]), float(on["E"][0] - on["E"][-1]))
        dEoff = float(off["E"][0] - off["E"][-1])
        print(f"\n  The correction is ACTIVE and Case 1 makes it pay.  The tank drains")
        print(f"  {dEon:.4f} J with it against {dEoff:.4f} J without and {dEb:.4f} J on the")
        print(f"  baseline; its peak torque is {k_on['u_tau_peak']:.3f} N m, which a real actuator")
        print(f"  limit would have to hold and which no passivity argument supplies.")
        plot_clock(bl, [("uniform control: the floor", ctl_),
                        ("scaled gains only", off),
                        ("with clock correction", on)],
                   clk, args.S, out / f"theorem4_a{args.clock:g}n{args.clock_n}.png")
        (out / f"theorem4_a{args.clock:g}n{args.clock_n}.json").write_text(json.dumps(dict(
            clock=clk.name, A=args.clock, n=args.clock_n, S=args.S, exec_s=T, c_eq=c_eq,
            h_bound=clk.h_bound(args.S), peak_r=clk.peak(args.S),
            rates=[dict(hz=t["hz"], floor_mm=t["floor"]["orbit_mm"],
                        uncorrected_mm=t["off"]["orbit_mm"],
                        corrected_mm=t["on"]["orbit_mm"],
                        u_clock_peak_Nm=t["on"]["u_tau_peak"]) for t in table]), indent=1))
        for sv in sims.values():
            sv.close()
        return

    rows = [("1x, baseline gains", base_log)]
    fixed = run_free(sim_at(fast_hz), ref, c, Exponents.none(), base, drag)
    rows.append((f"{c:g}x, fixed gains", fixed))

    full = run_free(sim_at(fast_hz), ref, c, Exponents(), base, drag)
    rows.append((f"{c:g}x, c^2 K, c D (Thm 1)", full))

    ablations = {}
    for name in args.ablate:
        ablations[name] = run_free(sim_at(fast_hz), ref, c, Exponents().drop(name), base, drag)
        rows.append((f"{c:g}x, Thm 1 without {name}", ablations[name]))

    if args.sweep is not None:
        ps = args.sweep or [1.0, 1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 3.0]
        print("\nProposition 2: the converse, swept over the stiffness exponent")
        print(f"  {'p':>6}{'D exponent':>13}{'max |dq| (rad)':>18}{'tip (m)':>13}{'rot (rad)':>13}")
        def_p = []
        for pv in ps:
            lg = run_free(sim_at(fast_hz), ref, c, Exponents(), base, drag, p=pv)
            k = conjugacy(base_log, lg)
            def_p.append(k["dp_m"])
            print(f"  {pv:>6.3g}{0.5 * pv:>13.3g}{k['dq']:>18.3e}{k['dp_m']:>13.3e}{k['dR_rad']:>13.3e}")
        plot_sweep(ps, np.asarray(def_p), c, out / f"proposition2_c{c:g}.png")
        (out / f"proposition2_c{c:g}.json").write_text(json.dumps(
            dict(c=c, p=list(ps), tip_defect_m=[float(x) for x in def_p]), indent=1))
        for sv in sims.values():
            sv.close()
        return

    print("\n" + peak_table(rows))

    print(f"\nconjugacy against the baseline orbit at matched phase  (q_c(t) = q_0(ct))")
    print(f"  {'execution':<34}{'max |dq| (rad)':>16}{'max |dv/c| (rad/s)':>20}"
          f"{'tip (m)':>12}{'rot (rad)':>12}")
    summary = {}
    for name, log in rows[1:]:
        k = conjugacy(base_log, log)
        summary[name] = k
        print(f"  {name:<34}{k['dq']:>16.3e}{k['dv']:>20.3e}{k['dp_m']:>12.3e}{k['dR_rad']:>12.3e}")

    print("\ncoefficient laws read back from the log  (relative error)")
    print(f"  {'execution':<34}{'K_c = c^2 K_0':>16}{'D_c = c D_0':>16}{'tau_c = c^2 tau_0':>20}")
    for name, log in rows[1:]:
        k = summary[name]
        print(f"  {name:<34}{k['rel_K']:>16.3e}{k['rel_D']:>16.3e}{k['rel_tau']:>20.3e}")

    # The tank and the torque limit cannot show up in a trajectory defect until
    # they actually bind, so report the margin instead of only the error: an
    # unscaled tank is wrong the moment c^2 of the baseline drain exceeds it.
    drain0 = float(base_log["E"][0] - base_log["E"][-1])
    print("\nresource margins  (a coefficient left unscaled costs nothing until it binds)")
    print(f"  {'execution':<34}{'alpha_min':>11}{'tank left (J)':>15}{'max |sat| (N m)':>17}")
    for name, log in rows:
        sat = float(np.abs(log["tau"] - log["tau_req"]).max())
        print(f"  {name:<34}{log['alpha'].min():>11.6f}{log['E'][-1]:>15.3f}{sat:>17.3e}")
    if drain0 > 1e-12:
        c_gate = float(np.sqrt((base.E0 - base.Ec) / drain0))
        print(f"  baseline drains {drain0:.4f} J, and the drain scales as c^2, so an UNSCALED "
              f"tank\n  ({base.E0:g} J, knee {base.Ec:g} J) first gates at c = {c_gate:.2f} "
              f"on this reference.")

    r = resources(base_log, full)
    print("\nCorollary 3, measured on the Theorem 1 run")
    for key in ("speed", "torque", "power", "energy"):
        print(f"  {key:<10} measured {r[key]:9.5f}   expected c^"
              f"{ {'speed': 1, 'torque': 2, 'power': 3, 'energy': 2}[key] } = {r['expect'][key]:9.5f}")
    print(f"  tank       alpha_min {r['alpha_min']:.6f}, {r['tank_left']:.3f} J left"
          f"    saturation {r['sat']:.3e} N m    contact {r['contact']:.3e} N")

    adm = admissible_b(base_log, base, ref)
    print("\nEq. (11), the exact admissible uniform-scaling set for this baseline")
    print(f"  velocity b <= {adm['b_velocity']:.3f}   torque b <= {adm['b_torque']:.3f}"
          f"   stiffness b <= {adm['b_stiffness']:.3f}")
    print(f"  -> b_max {adm['b_max']:.3f}, c_max {adm['c_max']:.3f}  (binding: {adm['binding']})")

    tag = (f"c{c:g}" + ("_fixedrate" if args.fixed_rate else "")
           + ("_drag" if args.keep_link_damping else ""))
    plot(rows[:3], out / f"theorem1_{tag}.png",
         f"Free-motion time-scaling symmetry at {c:g}x "
         f"({'fixed 500 Hz' if args.fixed_rate else f'{fast_hz} Hz'})")
    (out / f"theorem1_{tag}.json").write_text(json.dumps(dict(
        c=c, S=args.S, rot_amp_deg=args.rot_amp, fixed_rate=args.fixed_rate,
        link_damping=("sapien_default" if drag is None else 0.0),
        fast_hz=fast_hz, conjugacy=summary, resources=r, admissible=adm,
        peaks={n: dict(pos_mm=float(1000 * tracking_error(l)[0].max()),
                       rot_deg=float(np.rad2deg(tracking_error(l)[1]).max()))
               for n, l in rows}), indent=1))
    for s in sims.values():
        s.close()


if __name__ == "__main__":
    main()
