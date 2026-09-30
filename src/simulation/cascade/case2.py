"""Case 2: geometric admittance driving an opaque motion servo
(manuscript Sec. III-C, Prop. 1 and Prop. 2).

WHAT MAKES THIS "CASE 2" AND NOT THE CASCADE
---------------------------------------------
core.py's chain ends with an inner Cartesian impedance the controller owns:
f_cmd = Ki(x_r - x) + Di(v_r - v), applied as force.  Here there is NO inner
law at all.  The admittance emits a VELOCITY command, and what happens between
that command and the tool is a manufacturer servo the controller cannot see
into: a first-order lag Ts, a velocity/acceleration limiter and a delay queue.
Everything that made the cascade analysable -- knowing the inner stiffness,
being able to put it in a series-compliance formula -- is gone.

Three consequences, all of which this file makes measurable:

  1. STEADY STATE loses a term but gains an unknown.  The series chain is just
     1 + Ka/ke (analytic.contact_force_case2), simpler than the cascade's
     1 + Ko/ke + Ko/Ki.  But the servo invariant c = x_r - x_s - Ts v_s selects
     WHICH equilibrium, and after any clipping or delay c is no longer zero --
     so the same gains settle at a different force.  The cascade has no
     equivalent free parameter.

  2. PASSIVITY is not certified, only audited.  Prop. 1's identity
        Sdot_a = f_ch.V_c - f_e.V_s - D_a + r_M
     holds exactly, but r_M = f_e.e_s + ftilde.A_sr V_r is a DEFECT, computed
     from the true contact force and the true tracking error.  It is
     retrospective.  Nothing in the loop reacts to it.  This is the structural
     contrast with case1.py, where the tank actually throttles.

  3. STABILITY becomes a property of the plant, not the integrator.  Prop. 2's
     Routh condition (analytic.routh_case2) can fail with every virtual gain
     positive.  core.stability_limit is a step-size bound and says nothing
     about this; refining dt does not rescue a run that violates Eq. 15.

SIGN CONVENTION: as everywhere in this package, positive = into the wall and
f_e >= 0 is the reaction, so the admittance reads Ma v_rdot = f_ch - f_e - Br v_r.

STEP ORDERING
--------------
Deliberately NOT core.py's "evaluate everything at t, then integrate".  The
manuscript prescribes: virtual velocity advanced explicitly, virtual position
using the UPDATED velocity, then the servo integrated exactly under the held
command.  Table I is only reproducible under that ordering, so it is followed
here and the deviation is flagged rather than quietly normalised.

WHAT IS VERIFIED (tests_cases.py)
----------------------------------
  1. Steady-state force matches analytic.contact_force_case2, including the
     shifted equilibrium when c != 0 after clipping.
  2. Prop. 1's port-defect identity holds, residual first-order in dt.
  3. r_M -> 0 under exact tracking and sensing (Ts -> 0, no filter), and the
     virtual storage then certifies the real port -- the one case where the
     audit is a guarantee.
  4. Prop. 2's Routh criterion agrees with the reduced cubic's poles over
     randomised parameter sets.
  5. Table I is reproduced qualitatively: stable at ke=1000, oscillatory at
     ke=10000, recovered by raising da.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core import Array, EnvParams, _mat, quintic_pos, unilateral_wall


@dataclass
class Case2Params:
    """Virtual admittance, sensing, and the manufacturer servo model.

    Defaults are the manuscript's Sec. V-B generic servo: ma = 1 kg,
    da = ba + br = 30 Ns/m, ka = 300 N/m, 2 ms command period, 0.10 m/s and
    0.50 m/s^2 applied limits, against the same 12 mm gap as the Panda run.
    """
    # ---- virtual admittance (Eq. 12, A_cr = A_sr = I) ----
    Ma: float = 1.0
    Ka: float = 300.0
    # Ba and Br are split out because Eq. 12 has them at different ports, but
    # Prop. 2 only ever sees the SUM da = ba + br.  With a fixed command
    # (v_c = 0) they are indistinguishable, which is why the manuscript can
    # quote a single da; keep them separate so the Prop. 1 identity test can
    # exercise the Ba(v_c - v_r) term that a fixed command hides.
    Ba: float = 0.0
    Br: float = 30.0
    # ---- manufacturer servo:  Ts vdot_s + v_s = u,  xdot_s = v_s ----
    # Ts is a DECLARED MODEL PARAMETER, not identified UR firmware.
    Ts: float = 0.020
    v_limit: float = 0.10      # m/s,  applied velocity limit
    a_limit: float = 0.50      # m/s^2, applied acceleration limit
    delay_ticks: int = 0       # command transport delay, in whole dt steps
    # Anti-windup.  False reproduces the manuscript's choice ("Reference state
    # continues integrating during clipping; no unreported anti-windup is
    # added"), which is what EXPOSES reference windup rather than hiding it.
    anti_windup: bool = False
    # ---- sensing: causal first-order filter on the measured wrench ----
    # tau = 0 is a perfect sensor, which is the only setting under which
    # ftilde = 0 and Prop. 1's defect can vanish.
    f_filter_tau: float = 0.0
    f_bias: float = 0.0
    env: EnvParams = field(default_factory=lambda: EnvParams(ke=1000.0, de=0.0, x_wall=0.012))
    n: int = 1
    dt: float = 2e-3


class Case2Sim:
    """Geometric admittance + limits + delay + first-order lag servo."""

    def __init__(self, params: Case2Params,
                 wall_offset_fn: "callable | None" = None):
        self.p = params
        self.wall_offset_fn = wall_offset_fn
        n, dt = params.n, params.dt
        self.n, self.dt = n, dt

        self.x_r = np.zeros(n)     # virtual admittance reference
        self.v_r = np.zeros(n)
        self.x_s = np.zeros(n)     # ACTUAL tool state, behind the servo
        self.v_s = np.zeros(n)
        self.f_hat = np.zeros(n)   # filtered wrench measurement
        self.t = 0.0

        self.Ma = _mat(params.Ma, n, "Ma")
        self.Ka = _mat(params.Ka, n, "Ka")
        self.Ba = _mat(params.Ba, n, "Ba")
        self.Br = _mat(params.Br, n, "Br")
        nrm = params.env.normal
        if nrm is None:
            nrm = np.zeros(n); nrm[0] = 1.0
        nrm = np.asarray(nrm, dtype=float).reshape(n)
        if float(np.linalg.norm(nrm)) < 1e-12:
            raise ValueError("env.normal must be a non-zero vector")
        self._n_hat = nrm / float(np.linalg.norm(nrm))
        self.wall_now = params.env.x_wall
        self._u_prev = np.zeros(n)
        self._queue: list[Array] = [np.zeros(n) for _ in range(max(0, params.delay_ticks))]
        self.resid_int = 0.0
        self.rM_int = 0.0

    # ---------------- servo ----------------
    def _servo(self, u: Array) -> None:
        """Exact integration of Ts vdot_s + v_s = u under a HELD command.

        Exact rather than Euler because the lag is the object of study: an Euler
        servo would add its own O(dt) lag on top of Ts, and Prop. 2's condition
        is a statement about Ts alone.  Ts = 0 degenerates to a pure velocity
        source, which is the ideal-servo limit the defect test needs.
        """
        dt = self.dt
        if self.p.Ts <= 0.0:
            self.x_s = self.x_s + u * dt
            self.v_s = u.copy()
            return
        e = float(np.exp(-dt / self.p.Ts))
        dv = self.v_s - u
        self.x_s = self.x_s + u * dt + dv * self.p.Ts * (1.0 - e)
        self.v_s = u + dv * e

    # ---------------- one step ----------------
    def step(self, x_c: Array, v_c: Array | None = None) -> dict:
        n, dt, P = self.n, self.dt, self.p
        x_c = np.atleast_1d(np.asarray(x_c, dtype=float)).reshape(n)
        v_c = np.zeros(n) if v_c is None else np.atleast_1d(np.asarray(v_c, dtype=float)).reshape(n)

        # ---- 1. TRUE environment reaction, at the actual tool ----
        wall = P.env.x_wall + (self.wall_offset_fn(self.t, self.x_s) if self.wall_offset_fn else 0.0)
        self.wall_now = wall
        f_e, f_normal, pen = unilateral_wall(
            P.env, self._n_hat, self.t, self.x_s, self.v_s, wall)

        # ---- 2. what the controller actually SEES (causal filter + bias) ----
        if P.f_filter_tau > 0.0:
            a = dt / (P.f_filter_tau + dt)
            self.f_hat = self.f_hat + a * (f_e - self.f_hat)
        else:
            self.f_hat = f_e.copy()
        f_hat = self.f_hat + P.f_bias
        f_tilde = f_e - f_hat                     # the sensing half of the defect

        # ---- 3. geometric coupling at c (Eq. 12's f_ch) ----
        e_a = v_c - self.v_r                      # etilde_a, with A_cr = I
        f_Ga = self.Ka @ (x_c - self.x_r)
        f_ch = f_Ga + self.Ba @ e_a

        # ---- 4. Prop. 1 bookkeeping, all evaluated BEFORE the update ----
        S_a = 0.5 * float(self.v_r @ (self.Ma @ self.v_r)) \
            + 0.5 * float((x_c - self.x_r) @ (self.Ka @ (x_c - self.x_r)))
        D_a = float(e_a @ (self.Ba @ e_a)) + float(self.v_r @ (self.Br @ self.v_r))
        e_s = self.v_s - self.v_r                 # tracking half of the defect
        r_M = float(f_e @ e_s) + float(f_tilde @ self.v_r)
        rhs = float(f_ch @ v_c) - float(f_e @ self.v_s) - D_a + r_M

        # ---- 5. virtual admittance: velocity explicit, position from the NEW
        #         velocity (the manuscript's prescribed ordering) ----
        a_r = np.linalg.solve(self.Ma, f_ch - f_hat - self.Br @ self.v_r)
        self.v_r = self.v_r + a_r * dt
        self.x_r = self.x_r + self.v_r * dt

        # ---- 6. motion adapter -> limits -> delay queue ----
        u_req = self.v_r.copy()
        u = u_req.copy()
        if np.isfinite(P.a_limit):
            du = P.a_limit * dt
            u = np.clip(u, self._u_prev - du, self._u_prev + du)
        if np.isfinite(P.v_limit):
            u = np.clip(u, -P.v_limit, P.v_limit)
        clipped = bool(np.any(np.abs(u - u_req) > 1e-12))
        if clipped and P.anti_windup:
            # Back-propagating the clip into the virtual state is what a deployed
            # controller must declare; off by default so windup stays visible.
            self.v_r = u.copy()
        self._u_prev = u
        if P.delay_ticks > 0:
            self._queue.append(u)
            u = self._queue.pop(0)

        # ---- 7. the servo the controller cannot see into ----
        self._servo(u)

        # ---- 8. the invariant that selects WHICH equilibrium is reached ----
        c_inv = self.x_r - self.x_s - P.Ts * self.v_s

        # ---- 9. Prop. 1 residual, forward difference ----
        S_next = 0.5 * float(self.v_r @ (self.Ma @ self.v_r)) \
            + 0.5 * float((x_c - self.x_r) @ (self.Ka @ (x_c - self.x_r)))
        resid = (S_next - S_a) / dt - rhs
        self.resid_int += resid * dt
        self.rM_int += r_M * dt
        self.t += dt

        return {
            "t": self.t - dt, "x_c": x_c.copy(), "v_c": v_c.copy(),
            "x_r": self.x_r.copy(), "v_r": self.v_r.copy(),
            "x_s": self.x_s.copy(), "v_s": self.v_s.copy(),
            "f_e": f_e.copy(), "f_hat": f_hat.copy(), "f_tilde": f_tilde.copy(),
            "f_ch": f_ch.copy(), "f_normal": f_normal, "pen": pen,
            "e_s": e_s.copy(), "r_M": r_M, "S_a": S_a, "D_a": D_a,
            "resid": resid, "u_req": u_req.copy(), "u_appl": u.copy(),
            "clipped": float(clipped), "c_inv": c_inv.copy(), "wall": wall,
        }

    def run(self, duration: float, cmd_fn) -> dict:
        n_steps = int(round(duration / self.dt))
        if n_steps <= 0:
            raise ValueError(f"duration {duration} is shorter than one dt ({self.dt})")
        recs = []
        for _ in range(n_steps):
            xc, vc = cmd_fn(self.t)
            recs.append(self.step(xc, vc))
        return {k: np.array([r[k] for r in recs]) for k in recs[0]}


def reduced_poles(ma: float, da: float, ka: float, ke: float, Ts: float) -> Array:
    """Roots of the Prop. 2 reduced cubic

        ma Ts s^3 + (ma + da Ts) s^2 + (da + ka Ts) s + (ka + ke),

    obtained by eliminating x_r through the invariant c = x_r - x_s - Ts v_s.
    Provided so analytic.routh_case2 can be checked against actual eigenvalues
    rather than trusted -- the Routh reduction is exactly the kind of algebra
    that is easy to get subtly wrong and impossible to notice.
    """
    return np.roots([ma * Ts, ma + da * Ts, da + ka * Ts, ka + ke])


def wall_command(t: float, total: float = 0.035, t0: float = 0.5, t1: float = 2.0):
    """The manuscript's normal command: quintic to `total` over [t0, t1].
    Same ramp core.py and case1.py use, so the three cases are driven alike."""
    return np.array([quintic_pos(t, t0, t1, total)]), np.zeros(1)
