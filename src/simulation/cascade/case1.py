"""Case 1: direct geometric impedance on a torque-accessible arm, with the
energy tank that certifies it (manuscript Sec. III-B, Theorem 1).

WHAT MAKES THIS "CASE 1" AND NOT THE CASCADE
---------------------------------------------
There is NO outer admittance state.  core.py integrates (x_r, v_r) from a
virtual mass and then has an inner impedance chase it; here the GIC stiffness
talks to the environment directly and the only reference is the applied
equilibrium x_d, which moves at the commanded V_d.  That single structural
difference is the whole reason this file exists: the cascade's headline result
(Ki must exceed Ko, because the inner impedance is a second series compliance)
simply has no counterpart here.  analytic.contact_force_case1 shows the series
chain collapsing from 1 + Ko/ke + Ko/Ki to 1 + Kp/ke.

The second difference is the one the cascade is actually missing.  Case 1
carries a COMMON POWER GATE (Eq. 11): a single scalar alpha multiplies the
active torque, the reference velocity and the gain rates together, and a tank
integrates -alpha * p_prop.  The passive baseline tau_0 is never scaled.  That
gate is an ENFORCED passivity mechanism -- unlike the cascade's H_ch/E_ch
comparison, which is a numerical bug-check that throttles nothing.

TRANSLATIONAL REDUCTION
------------------------
Same reduction the rest of this package uses: n independent translational axes,
J = I, v = V_s, so the geometric error is p_de = x - x_d and the potential
gradient (Eq. 5) is f_G = Kp p_de.  Rotation is out of scope here exactly as it
is in core.py -- the SE(3) form lives in
../haptic_teleop_fr3_bilateral.geometric_impedance_torque.

Consequences of the reduction, stated rather than hidden:
  - There is no null space, so the posture potential U_N and its gradient drop
    out.  That is not cosmetic: the manuscript's own Panda run reports 5.113 N
    against a closed form predicting 5.31 N, and it attributes the gap to the
    unprojected posture gradient contributing task wrench.  This file cannot
    reproduce that gap, and a test here passing does not certify the 7-joint
    case.
  - delta_tau (applied-torque discrepancy) is NOT dropped.  It is the term the
    manuscript insists cannot be omitted, so it is modelled explicitly by a
    zero-order hold plus magnitude/rate clipping, and it appears in the energy
    identity as v . delta_tau.

WHAT IS VERIFIED (tests_cases.py)
----------------------------------
  1. Steady-state force matches analytic.contact_force_case1.
  2. The Theorem 1 identity Hdot_T = -f_e.V_s - D_T + p_T + v.delta_tau holds,
     with residual converging at FIRST ORDER in dt -- the same refinement check
     the manuscript reports (0.297 -> 0.149 mJ when the step is halved).
  3. With the tank on and delta_tau = 0, H_T + E is non-increasing.
  4. E >= 0 is never violated.
  5. With the tank OFF, an over-aggressive proposal injects energy that the
     gated run does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core import Array, EnvParams, _kmax, _mat, quintic_pos, quintic_vel, unilateral_wall


@dataclass
class Case1Params:
    """Plant, GIC baseline, tank and torque interface.

    Defaults follow the manuscript's Sec. V-A Panda study where they transfer
    to a point mass: Kp = 300 N/m, translational damping 35 Ns/m, a 1000 N/m
    wall 12 mm away, and an equilibrium commanded 35 mm past the start.
    """
    # ---- plant:  Ms a = tau_appl - f_e ----
    Ms: float = 2.0
    # ---- passive geometric baseline, Eq. 9:  tau_0 = -f_G - D V_s ----
    Kp: float = 300.0
    D: float = 35.0
    # Applied-coefficient bounds (Eq. 11's 0 < k I <= K <= kbar I).  The gate
    # scales the gain RATE, so without a bound an adaptive proposal can still
    # walk the stiffness anywhere it likes over time; the bound is part of the
    # theorem's hypotheses, not a safety extra.
    kp_lo: float = 1.0
    kp_hi: float = 5000.0
    # ---- energy tank (Eq. 11) ----
    tank: bool = True
    # E0 must exceed Ec or the gate throttles from the first step.  Left at 1.0
    # this is not a bug but it IS a trap: the run then silently tracks a
    # different reference than the one requested, which is exactly the
    # requested-vs-applied confusion the recording spec warns about.
    E0: float = 10.0      # initial tank energy, J
    Ec: float = 1.0       # gate knee: alpha = min(1, E/Ec)
    # ---- torque interface ----
    # hold_steps > 1 is a zero-order hold: the law is evaluated once and the
    # stale torque is applied for the next few substeps.  This is what MAKES
    # delta_tau nonzero, and the manuscript is explicit that the continuous
    # theorem cannot be identified with a held implementation without it.
    hold_steps: int = 1
    tau_limit: float = np.inf        # Nm (here N, translational)
    tau_rate_limit: float = np.inf   # N/s
    env: EnvParams = field(default_factory=lambda: EnvParams(ke=1000.0, de=8.0, x_wall=0.012))
    n: int = 1
    dt: float = 5e-4


@dataclass
class Case1Proposal:
    """What the haptic operator or the policy REQUESTS, before the gate.

    Kept separate from the applied values on purpose: the manuscript's recording
    spec distinguishes requested / admitted / applied, and conflating them is
    precisely the labelling error it warns about.  `Up` is the requested rate of
    change of Kp, i.e. Eq. 11's U_p, not a stiffness target.
    """
    u: Array            # active torque proposal u^prop
    Vd: Array           # reference velocity proposal V_d^prop
    Up: Array | None = None   # gain-rate proposal U_p (n, n); None = zero


def stability_limit(p: Case1Params) -> "tuple[float, str]":
    """Explicit semi-implicit-Euler step bound, same form as core.stability_limit."""
    worst, name = np.inf, "none"
    for nm, k in (("Kp / Ms", p.Kp), ("env ke / Ms", p.env.ke)):
        k = _kmax(k)                      # largest eigenvalue: an anisotropic Kp
                                          # is a matrix, and its max ELEMENT is
                                          # not its stiffest direction
        if k > 0 and np.isfinite(k):
            d = 2.0 * np.sqrt(p.Ms / k)
            if d < worst:
                worst, name = d, nm
    for nm, b in (("D / Ms", p.D), ("env de / Ms", p.env.de)):
        b = _kmax(b)
        if b > 0 and np.isfinite(b):
            d = 2.0 * p.Ms / b
            if d < worst:
                worst, name = d, nm
    return float(worst), name


class Case1Sim:
    """Semi-implicit Euler, forces evaluated at t, state integrated to t+dt.

    Step ordering matches core.CascadeSim exactly so the two are comparable step
    for step: every force comes from the state at time t, the record is taken,
    and only then is anything integrated.
    """

    def __init__(self, params: Case1Params,
                 wall_offset_fn: "callable | None" = None):
        # wall_offset_fn(t, x) -> scalar height, exactly core.CascadeSim's
        # signature, so the same random surface drives all three executors.
        self.p = params
        self.wall_offset_fn = wall_offset_fn
        n, dt = params.n, params.dt
        self.n, self.dt = n, dt
        dt_max, src = stability_limit(params)
        if dt >= dt_max:
            raise ValueError(f"dt={dt:.2e}s is at or past the stability limit "
                             f"{dt_max:.2e}s set by {src}")
        self.dt_limit, self.dt_limit_source = dt_max, src

        self.x = np.zeros(n)      # slave actual position
        self.v = np.zeros(n)      # slave actual velocity (= V_s, since J = I)
        self.x_d = np.zeros(n)    # APPLIED equilibrium (the integrated reference)
        self.Kp = _mat(params.Kp, n, "Kp")
        self.D = _mat(params.D, n, "D")
        self.Ms = _mat(params.Ms, n, "Ms")
        self.E = float(params.E0)
        self.t = 0.0

        # Contact normal, resolved the same way core.CascadeSim does it: None
        # means the first axis (the 1-DOF wall), a vector means a tilted plate.
        nrm = params.env.normal
        if nrm is None:
            nrm = np.zeros(n); nrm[0] = 1.0
        nrm = np.asarray(nrm, dtype=float).reshape(n)
        if float(np.linalg.norm(nrm)) < 1e-12:
            raise ValueError("env.normal must be a non-zero vector")
        self._n_hat = nrm / float(np.linalg.norm(nrm))
        self.wall_now = params.env.x_wall
        self._tau_held = np.zeros(n)
        self._tau_prev = np.zeros(n)
        self._k = 0
        self.resid_int = 0.0      # running integral of the Theorem 1 residual

    # ---------------- one step ----------------
    def step(self, prop: Case1Proposal | None = None) -> dict:
        n, dt, P = self.n, self.dt, self.p
        if prop is None:
            prop = Case1Proposal(np.zeros(n), np.zeros(n))
        u_p = np.asarray(prop.u, dtype=float).reshape(n)
        Vd_p = np.asarray(prop.Vd, dtype=float).reshape(n)
        Up = np.zeros((n, n)) if prop.Up is None else _mat(prop.Up, n, "Up")

        # ---- 1. geometric error and potential gradient (Eq. 5, translational) ----
        p_de = self.x - self.x_d
        f_G = self.Kp @ p_de

        # ---- 2. environment (SHARED with core.py -- same contact model) ----
        wall = P.env.x_wall + (self.wall_offset_fn(self.t, self.x) if self.wall_offset_fn else 0.0)
        self.wall_now = wall
        f_e, f_normal, pen = unilateral_wall(
            P.env, self._n_hat, self.t, self.x, self.v, wall)

        # ---- 3. proposal power, evaluated at the CURRENT APPLIED state ----
        # The manuscript requires this evaluation point so that p_T = alpha *
        # p_prop exactly; evaluating it at the proposed state instead would
        # break the cancellation the tank relies on.
        pK_prop = 0.5 * float(p_de @ (Up @ p_de))
        p_prop = float(self.v @ u_p) - float(f_G @ Vd_p) + pK_prop

        # ---- 4. common power gate (Eq. 11) ----
        alpha = min(1.0, self.E / P.Ec) if P.tank else 1.0
        u = alpha * u_p
        Vd = alpha * Vd_p
        Kp_dot = alpha * Up
        p_T = alpha * p_prop

        # ---- 5. passive geometric baseline (Eq. 9), NEVER scaled ----
        # The measured contact wrench is deliberately absent: no cancellation.
        tau_0 = -f_G - self.D @ self.v
        tau_req = tau_0 + u

        # ---- 6. torque interface: zero-order hold, then magnitude/rate limits ----
        if self._k % max(1, P.hold_steps) == 0:
            self._tau_held = tau_req.copy()
        tau_appl = self._tau_held.copy()
        if np.isfinite(P.tau_rate_limit):
            dmax = P.tau_rate_limit * dt
            tau_appl = np.clip(tau_appl, self._tau_prev - dmax, self._tau_prev + dmax)
        if np.isfinite(P.tau_limit):
            tau_appl = np.clip(tau_appl, -P.tau_limit, P.tau_limit)
        # delta_tau is measured against what the CONTINUOUS law wants right now,
        # which is what makes the held-torque defect show up here rather than
        # silently biasing the energy balance.
        d_tau = tau_appl - tau_req

        # ---- 7. energy accounting (Theorem 1) ----
        T_kin = 0.5 * float(self.v @ (self.Ms @ self.v))
        P_pot = 0.5 * float(p_de @ (self.Kp @ p_de))
        H_T = T_kin + P_pot
        D_T = float(self.v @ (self.D @ self.v))
        rhs = -float(f_e @ self.v) - D_T + p_T + float(self.v @ d_tau)

        rec = {
            "t": self.t, "x": self.x.copy(), "v": self.v.copy(), "x_d": self.x_d.copy(),
            "f_e": f_e.copy(), "f_G": f_G.copy(), "f_normal": f_normal, "pen": pen,
            "tau_0": tau_0.copy(), "tau_req": tau_req.copy(), "tau_appl": tau_appl.copy(),
            "d_tau": d_tau.copy(), "alpha": alpha, "E": self.E,
            "p_prop": p_prop, "p_T": p_T, "H_T": H_T, "D_T": D_T,
            "kp": float(self.Kp[0, 0]), "f_normal_v": f_normal, "wall": wall,
        }

        # ---- 8. integrate to t+dt (velocity before position) ----
        a = np.linalg.solve(self.Ms, tau_appl - f_e)
        self.v = self.v + a * dt
        self.x = self.x + self.v * dt
        self.x_d = self.x_d + Vd * dt
        if np.any(Kp_dot):
            self.Kp = np.clip(self.Kp + Kp_dot * dt, P.kp_lo, P.kp_hi)
        self.E = self.E - p_T * dt      # Edot = -alpha * p_prop = -p_T
        self._tau_prev = tau_appl
        self._k += 1
        self.t += dt

        # ---- 9. the Theorem 1 residual, as a forward difference ----
        p_de_n = self.x - self.x_d
        H_next = (0.5 * float(self.v @ (self.Ms @ self.v))
                  + 0.5 * float(p_de_n @ (self.Kp @ p_de_n)))
        resid = (H_next - H_T) / dt - rhs
        self.resid_int += resid * dt
        rec["H_T_next"] = H_next
        rec["resid"] = resid
        rec["HE"] = H_next + self.E          # the certified sum, post-update
        return rec

    def run(self, duration: float, prop_fn=None) -> dict:
        n_steps = int(round(duration / self.dt))
        if n_steps <= 0:
            raise ValueError(f"duration {duration} is shorter than one dt ({self.dt})")
        recs = []
        for _ in range(n_steps):
            recs.append(self.step(prop_fn(self.t) if prop_fn else None))
        return {k: np.array([r[k] for r in recs]) for k in recs[0]}


# The reference-trajectory helpers live in core.py so Case 1, Case 2 and the
# cascade are all driven by the IDENTICAL ramp; re-exported here for callers.
quintic_ramp = quintic_vel
__all__ = ["Case1Params", "Case1Proposal", "Case1Sim", "stability_limit",
           "quintic_pos", "quintic_vel", "quintic_ramp"]
