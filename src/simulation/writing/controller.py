"""Case 1 on the arm: geometric impedance with a TIME-VARYING K_p, gated by the
energy tank.

This is ../cascade/case1.py lifted from a point mass onto the Panda.  Same
structure, same ordering, same names:

    tau_0  = J_b^T ( -f_G - D e_v ) + null-space posture     (Eq. 9, never scaled)
    alpha  = min(1, E / Ec)                                  (Eq. 11, common gate)
    V_d    = alpha V_d^prop,   Kdot = alpha U_p              (applied = gated proposal)
    Edot   = -alpha p_prop,    p_prop = -f_G . V_d^prop + 1/2 p_de^T U_p p_de

What changes on the arm:

  * K_p is a 3x3 SPD matrix in WORLD coordinates, built from three stiffnesses
    along the WRITING frame (u, v along the paper, n off it).  The GIC form
    f_p = -R^T K (p - x_d) keeps that matrix attached to the task rather than to
    whatever orientation the wrist holds -- the frame question
    ../cascade/README.md found the task cannot absorb.
  * K_p VARIES.  The proposal carries U_p, a stiffness RATE, exactly as Case 1
    defines it; a stiffness TARGET is turned into a rate by the caller.  The
    rate is gated with the reference velocity by the same alpha, so a stiffness
    change that would inject energy (stiffening a stretched spring) drains the
    tank just as a reference motion does.  That is the whole reason variable
    impedance needs the tank: without it a stiffness schedule is an energy
    source.
  * Damping follows stiffness AND the arm's own inertia, by the factorization
    design D = zeta (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2), with Lambda the
    operational-space inertia at the tip, recomputed as the arm moves.  Holding
    D fixed while K moves by 10x would take the loop from overdamped to ringing
    within one stroke.  And Lambda is not a detail: the first version assumed a
    1.5 kg tip, the real one is ~7 kg vertically and ~1 kg sideways at the
    writing pose, and the normal axis sat at zeta ~0.3 -- the pen overshot the
    hover height by 5 mm and landed on the paper at travel speed.  D is dissipative whatever it is, so this costs
    the tank nothing.  Its feed-forward D V_d is NOT dissipative, so it rides
    in Case 1's active slot u, where the gate scales it and the tank pays.
  * THE ROTATIONAL DAMPING FOLLOWS THE SAME RULE, and for a while it did not.
    Dr was built from a GUESSED scalar inertia, which makes the real damping
    ratio zeta sqrt(I_guess / Lambda_rot) rather than zeta: against the
    measured [0.005, 0.18, 0.38] kg m^2 at the tip, the guess 0.01 put it at
    0.13-0.19 on two of three axes, a Q of about 4.  Nothing showed on a single
    smooth traverse, which is why it survived; on a wiping raster, which
    REVERSES every second, it rang the rotational mode hard enough to lose the
    task -- 0% of the glyph erased at K_R = 1 against 66% once the inertia was
    measured (../wiping/CURVED_BOARD.md).  Dr is now the matrix
    zeta (Lambda_r^1/2 K_R^1/2 + K_R^1/2 Lambda_r^1/2), with Lambda_r the
    rotational block of the same operational-space inertia, in the BODY frame
    because that is the frame the damping acts in.
  * K_R IS PART OF THE PROPOSAL, like K_p.  The rotational potential is
    tr(K_R (I - R_d^T R)), so a stiffness rate U_r injects
    U_r tr(I - R_d^T R) = U_r 2(1 - cos theta) of power -- nonnegative when
    stiffening, exactly as 1/2 p_de^T U_p p_de is -- and it is metered into
    p_prop and gated by the same alpha.  It is a 3x3 in the TOOL BODY frame,
    so "comply in tilt, resist twist" is expressible; a scalar still works and
    is what the writing task and the wiping protocol pass.
  * WHAT K_R IS NOT: the stiffness the tool feels.  The elastic moment is
    1/2 vee(K_R R_d^T R - R^T R_d K_R), whose small-angle gradient is
    1/2 (tr(K_R) I - K_R) -- so the felt stiffness about an axis is half the
    sum of the OTHER TWO K_R entries and does not depend on that axis's own
    entry at all.  K_R = diag(50, 5, 50) is soft about y as a parameter and the
    stiffest axis as a measurement.  felt_from_KR / KR_from_felt convert, and a
    human-impedance label or a perturbation experiment measures the FELT one, so
    that is the side a policy should predict.  Isotropic K_R hides this
    completely (K_eff = kr I), which is why the writing task never saw it.
  * The compliance centre is the pen TIP (panda_hand_tcp sits at the ball's
    lowest point), so a rotational error pivots the pen about the point that
    writes instead of dragging the tip sideways.
  * Torque saturation at the Panda's joint limits is the delta_tau of the
    theorem: modelled, clipped and logged, never dropped.
  * A proposal may carry an active JOINT torque u_tau.  Nothing in the writing
    task uses it; the retiming clock correction of speedup.py does, and it is
    gated, metered and clipped exactly like the task-space active term.

WHAT IS LOGGED, per the recording spec case1.py follows: the REQUESTED values
(x_d_req, K_req), the ADMITTED gate alpha, and the APPLIED values (x_d, K).
Labelling a demonstration with the requested ones is wrong whenever the gate
has closed, and the log is the only place that difference is visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sapien.wrapper.pinocchio_model import PinocchioModel

from mani_skill.utils import sapien_utils

Array = np.ndarray
PANDA_TAU_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def vee(S: Array) -> Array:
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def quat_wxyz_to_R(q: Array) -> Array:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def sym_sqrt(K: Array) -> Array:
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


def k_world(k_diag: Array, W: Array) -> Array:
    """Stiffnesses along the columns of frame W -> world matrix W diag(k) W^T."""
    return (W * np.asarray(k_diag, dtype=float)) @ W.T


# --------------------------------------------------------------------------- #
# K_R IS NOT THE STIFFNESS THE TOOL FEELS.
#
# The elastic moment below is  e_R = 1/2 vee(K_R R_d^T R - R^T R_d K_R).  Put
# R_d^T R = I + s_hat for a small body rotation s and use
# (K s_hat + s_hat K)^vee = (tr(K) I - K) s:
#
#     e_R  ->  1/2 (tr(K_R) I - K_R) s  =:  K_eff s
#
# so the stiffness an operator, a perturbation experiment or a human-impedance
# label measures is K_eff, and the i-th diagonal entry of K_eff is
# 1/2 (sum of the OTHER TWO entries of K_R) -- it does not depend on K_R[i, i]
# at all.  Learning K_R from a felt label is therefore learning the wrong
# quantity, and on an anisotropic profile it inverts the ordering:
# K_R = diag(50, 5, 50) is soft about y as a parameter and the STIFFEST axis as
# a measurement (K_eff = diag(27.5, 50, 27.5)).
#
# A scalar K_R hides all of it: K_R = kr I gives K_eff = kr I exactly, which is
# why nothing in the writing task ever noticed.
def as_KR(value) -> Array:
    """A scalar, a 3-vector or a 3x3 -> the 3x3 K_R (or K_R rate) it means."""
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        return float(a) * np.eye(3)
    if a.ndim == 1:
        return np.diag(a)
    return 0.5 * (a + a.T)


def felt_from_KR(Kr: Array) -> Array:
    """The rotational stiffness the tool actually feels, from the GIC gain."""
    Kr = as_KR(Kr)
    return 0.5 * (np.trace(Kr) * np.eye(3) - Kr)


def KR_from_felt(K_eff: Array) -> Array:
    """The GIC gain that yields a given felt stiffness: tr(K_eff) I - 2 K_eff.

    Inverse of felt_from_KR, exactly (tr(K_eff) = tr(K_R) for the pair).  The
    result is NOT positive definite for every positive-definite K_eff: with
    K_eff = diag(a, b, c) it needs b + c > a on every axis, so no felt axis may
    be stiffer than the other two together.  "Resist twist, comply in tilt" --
    felt (5, 5, 50) -- wants K_R = diag(50, 50, -40) and is unreachable.  The
    potential stays positive there, but GIC's and GUFIC's passivity proofs
    assume K_R > 0 and `advance` clips the eigenvalues at `kr_lo`, so the
    request is rejected rather than silently approximated.  `felt_reachable`
    is the test to run BEFORE asking for a profile.
    """
    K_eff = as_KR(K_eff)
    return np.trace(K_eff) * np.eye(3) - 2.0 * K_eff


def felt_reachable(K_eff: Array, kr_lo: float = 0.0) -> bool:
    """Is `K_eff` realizable by a K_R whose eigenvalues all clear `kr_lo`?"""
    return bool(np.linalg.eigvalsh(KR_from_felt(K_eff)).min() >= kr_lo)


@dataclass
class Case1Gains:
    # Applied-stiffness bounds (Eq. 11's  k I <= K <= kbar I).  The gate scales
    # the RATE, so without a bound a proposal could still walk K anywhere over
    # time; the bound is a hypothesis of the theorem, not a safety extra.
    k_lo: float = 100.0
    k_hi: float = 4000.0
    kr_lo: float = 0.1            # the same bound for the rotational stiffness
    kr_hi: float = 100.0
    zeta: float = 0.8
    lambda_every: int = 5         # physics steps between inertia updates
    # Nm/rad, the INITIAL rotational stiffness: a scalar (isotropic, the only
    # thing the writing task needs) or a 3-vector / 3x3 in the TOOL BODY frame.
    # It is the GIC GAIN K_R, not the felt stiffness -- see felt_from_KR above,
    # and pass KR_from_felt(K_eff) to command a felt profile.
    Kr: float | Array = 80.0
    # None: Dr is built from the MEASURED rotational inertia, which is what it
    # should always have been.  A number restores the old guessed scalar and is
    # kept only so the difference stays reproducible -- see the docstring.
    wrist_inertia: float | None = None
    null_kp: float = 5.0
    null_kd: float = 1.0
    # ---- energy tank (Eq. 11) ----
    tank: bool = True
    E0: float = 20.0              # J.  A writing episode drains ~1-3 J through friction
    Ec: float = 1.0               # J, gate knee
    tau_limit: Array = field(default_factory=lambda: PANDA_TAU_LIMIT.copy())


@dataclass
class Case1Proposal:
    """What the operator (or a policy) REQUESTS, before the gate.

    No task-space active wrench is part of the request: the only active term
    this task needs is the damping feed-forward, which the controller derives
    from V_d (see compute()).  A policy's whole action is therefore (x_d, K).

    `u_tau` is the one exception, and it is a JOINT-space torque because the
    only thing that asks for it is a joint-space quantity: the retiming clock
    correction u_clock = (rdot/r) M v of the time-scaling theory, which cancels
    the -(r'/r) M p defect a nonuniform execution clock leaves behind
    (speedup.py).  It is active, not passive -- while the clock accelerates it
    SUPPLIES power -- so it goes in Case 1's gated slot and its power qdot . u
    is metered into p_prop like any other active term, and it is clipped at the
    torque limit with everything else.  Leaving it out of the accounting would
    put an unmetered energy source inside a controller whose whole point is
    that there is none.
    """
    Vd: Array                     # reference velocity proposal, world, m/s
    Up: Array | None = None       # stiffness-rate proposal, world 3x3, N/m/s
    Ur: float | Array | None = None   # rotational stiffness rate, Nm/rad/s;
                                  # scalar (isotropic) or 3-vector / 3x3, body
    u_tau: Array | None = None    # active joint torque proposal, N m


class Case1Controller:
    def __init__(self, robot, gains: Case1Gains, urdf_path: str,
                 tcp_name: str = "panda_hand_tcp"):
        self.robot = robot
        self.g = gains
        with open(urdf_path) as f:
            self.pm = PinocchioModel(f.read(), [0.0, 0.0, -9.81])
        self.pm.set_joint_order([j.name for j in robot.active_joints])
        links = [l.name for l in robot.get_links()]
        self.pm.set_link_order(links)
        self.ee = links.index(tcp_name)
        self.tcp = sapien_utils.get_obj_by_name(robot.get_links(), tcp_name)
        self.nq = len(robot.active_joints)
        ql = robot.get_qlimits()[0].cpu().numpy()
        self.q_mid = 0.5 * (ql[:, 0] + ql[:, 1])
        self.q_rest = None
        self._Lam = None
        self._k = 0
        self.reset_state(np.zeros(3), np.eye(3), np.eye(3) * 1000.0)

    # ------------------------------------------------------------------ #
    def disable_joint_drives(self) -> None:
        """A joint PD left on puts a stiff position servo in series with the
        impedance, and the measured compliance is then neither one."""
        for j in self.robot.active_joints:
            j.set_drive_properties(0.0, 0.0, force_limit=1000.0)

    def reset_state(self, x_d: Array, R_d: Array, K: Array, q_rest: Array | None = None,
                    kr: float | None = None) -> None:
        self.x_d = np.asarray(x_d, dtype=float).copy()
        self.R_d = np.asarray(R_d, dtype=float).copy()
        self.K = np.asarray(K, dtype=float).copy()
        self.Kr = as_KR(self.g.Kr if kr is None else kr)
        self.E = float(self.g.E0)
        self._Lam = None
        self._k = 0
        self.q_rest = self.q_mid.copy() if q_rest is None else np.asarray(q_rest, float).copy()

    # The scalar view, for every caller that has one level of rotational
    # stiffness to set (the wiping protocol, the tests): exact when K_R is
    # isotropic, which is the only case any of them uses.
    @property
    def kr(self) -> float:
        return float(np.trace(self.Kr) / 3.0)

    @kr.setter
    def kr(self, value) -> None:
        self.Kr = as_KR(value)

    def tip_state(self) -> tuple[Array, Array, Array, Array]:
        """(R, p, v_world, w_world) of the pen tip."""
        q = self.robot.get_qpos()[0].cpu().numpy()
        qd = self.robot.get_qvel()[0].cpu().numpy()
        pose = self.tcp.pose
        R = quat_wxyz_to_R(pose.q[0].cpu().numpy())
        p = pose.p[0].cpu().numpy().astype(float)
        J_b = self.pm.compute_single_link_local_jacobian(q, self.ee)
        V_b = J_b @ qd
        return R, p, R @ V_b[:3], R @ V_b[3:]

    def inertia(self, q: Array, R: Array, J_b: Array) -> Array:
        """Translational block of the 6-D operational-space inertia at the tip.
        The 6-D block, not the 3-D one: the pen's orientation is held by Kr, so
        the tip does not get the 3-D value's free-rotation discount.

        The ROTATIONAL block falls out of the same inverse and is kept for Dr,
        rotated into the BODY frame: the rotational damping acts on V_b[3:],
        and Lambda_world = blkdiag(R, R) Lambda_body blkdiag(R, R)^T.
        """
        if self._k % max(1, self.g.lambda_every) == 0 or self._Lam is None:
            M = self.pm.compute_generalized_mass_matrix(q)
            Jw = np.vstack([R @ J_b[:3], R @ J_b[3:]])
            L6 = np.linalg.inv(Jw @ np.linalg.solve(M, Jw.T))
            self._Lam = 0.5 * (L6[:3, :3] + L6[:3, :3].T)
            self._Lam_sqrt = sym_sqrt(self._Lam)
            Lr = R.T @ L6[3:, 3:] @ R
            self._Lam_r = 0.5 * (Lr + Lr.T)
            self._Lam_r_sqrt = sym_sqrt(self._Lam_r)
        return self._Lam

    def elastic_moment(self, R: Array) -> Array:
        """GIC's rotational elastic moment e_R, body frame, at the current K_R
        and R_d.  The applied moment is -e_R (plus damping), so the stiffness a
        measurement sees is d(e_R)/ds = felt_from_KR(K_R) -- NOT K_R."""
        return 0.5 * vee(self.Kr @ self.R_d.T @ R - R.T @ self.R_d @ self.Kr)

    def damping(self, K: Array) -> Array:
        """Factorization design (Albu-Schaeffer et al., 2003)."""
        A, K1 = self._Lam_sqrt, sym_sqrt(K)
        return self.g.zeta * (A @ K1 + K1 @ A)

    # ------------------------------------------------------------------ #
    def compute(self, prop: Case1Proposal, dt: float) -> dict:
        """Evaluate the law at the CURRENT applied state and apply the torque.

        Nothing is integrated here: case1.Case1Sim takes every force from the
        state at t, records, and only then integrates.  `advance()` does the
        integration after the physics step, so the ordering is identical.
        """
        g = self.g
        q = self.robot.get_qpos()[0].cpu().numpy()
        qd = self.robot.get_qvel()[0].cpu().numpy()
        pose = self.tcp.pose
        R = quat_wxyz_to_R(pose.q[0].cpu().numpy())
        p = pose.p[0].cpu().numpy().astype(float)
        J_b = self.pm.compute_single_link_local_jacobian(q, self.ee)   # body Jacobian at the tip
        V_b = J_b @ qd
        v = R @ V_b[:3]

        Vd_p = np.asarray(prop.Vd, dtype=float).reshape(3)
        Up = np.zeros((3, 3)) if prop.Up is None else np.asarray(prop.Up, dtype=float)
        Up = 0.5 * (Up + Up.T)
        tau_u_p = (np.zeros(self.nq) if prop.u_tau is None
                   else np.asarray(prop.u_tau, dtype=float).reshape(self.nq))

        # ---- 1. geometric error and potential gradient (world) ----
        p_de = p - self.x_d
        f_G = self.K @ p_de

        # ---- 2. active proposal: damping feed-forward ----
        # Case 1's baseline damps the ABSOLUTE velocity (tau_0 = -f_G - D V_s),
        # which makes a moving reference lag by D V_d / K -- 1.3 mm at a
        # writing speed of 3 cm/s.  Damping the velocity RELATIVE to V_d
        # removes the lag, but the extra term D V_d does work, and hiding it
        # inside tau_0 would put an unaccounted energy source in the "passive"
        # baseline.  It goes in the u slot instead, where the gate sees it.
        Lam = self.inertia(q, R, J_b)
        D = self.damping(self.K)
        u_p = D @ Vd_p

        # ---- 3. proposal power at the CURRENT applied state ----
        # The rotational stiffness rate's share: d/dt of tr(K_R (I - R_d^T R))
        # through K_R alone.  tr(I - R_d^T R) = 2(1 - cos theta) >= 0, so
        # stiffening always injects and loosening always drains, which is the
        # rotational copy of 1/2 p_de^T U_p p_de.
        # Matrix form: d/dt of tr(K_R (I - R_d^T R)) through K_R is
        # tr(U_r (I - R_d^T R)), which reduces to U_r tr(I - R_d^T R) when U_r
        # is isotropic -- so the scalar callers and tests are unchanged.
        Ur_p = np.zeros((3, 3)) if prop.Ur is None else as_KR(prop.Ur)
        rot_gap_m = np.eye(3) - self.R_d.T @ R
        p_prop = (float(v @ u_p) + float(qd @ tau_u_p)
                  - float(f_G @ Vd_p) + 0.5 * float(p_de @ Up @ p_de)
                  + float(np.trace(Ur_p @ rot_gap_m)))

        # ---- 4. common power gate ----
        alpha = min(1.0, max(self.E, 0.0) / g.Ec) if g.tank else 1.0
        u = alpha * u_p
        tau_u = alpha * tau_u_p
        Vd = alpha * Vd_p
        Kdot = alpha * Up
        Krdot = alpha * Ur_p
        p_T = alpha * p_prop

        # ---- 5. passive geometric baseline (never scaled) + gated u ----
        F_lin = -f_G - D @ v + u
        # The GIC elastic moment for a MATRIX K_R.  With K_R = kr I this is
        # kr * vee(skew part) and Dr is 2 zeta sqrt(kr) Lambda_r^1/2, i.e.
        # exactly the scalar law it replaces.  What it felt like all along is
        # felt_from_KR(K_R) = 1/2 (tr(K_R) I - K_R), which is kr I only here.
        Kr_sqrt = sym_sqrt(self.Kr)
        e_R = self.elastic_moment(R)
        Dr = (g.zeta * (self._Lam_r_sqrt @ Kr_sqrt + Kr_sqrt @ self._Lam_r_sqrt)
              if g.wrist_inertia is None else
              2.0 * g.zeta * np.sqrt(g.wrist_inertia) * Kr_sqrt)
        F_b = np.concatenate([R.T @ F_lin, -e_R - Dr @ V_b[3:]])
        tau = J_b.T @ F_b
        N = np.eye(self.nq) - J_b.T @ np.linalg.pinv(J_b.T)
        tau_req = tau + N @ (g.null_kp * (self.q_rest - q) - g.null_kd * qd) + tau_u

        # ---- 6. torque interface: saturation IS delta_tau ----
        tau_appl = np.clip(tau_req, -g.tau_limit, g.tau_limit)
        d_tau = tau_appl - tau_req
        self.robot.set_qf(tau_appl[None, :])

        self._pending = (Vd, Kdot, Krdot, p_T, dt)
        return {
            "p": p, "R": R, "v": v, "w": R @ V_b[3:],
            "q": q, "qd": qd,
            "x_d": self.x_d.copy(), "K": self.K.copy(), "D": D, "Lambda": Lam.copy(),
            "kr": self.kr, "Kr": self.Kr.copy(), "Kr_eff": felt_from_KR(self.Kr),
            "Dr": Dr.copy(), "Lambda_r": self._Lam_r.copy(),
            "f_G": f_G, "alpha": alpha, "E": self.E, "p_prop": p_prop, "p_T": p_T,
            "u_tau": tau_u,
            "tau_req": tau_req, "tau": tau_appl, "d_tau": d_tau,
            "H_T": 0.5 * float(p_de @ self.K @ p_de),
        }

    def advance(self) -> None:
        """Integrate the applied reference, stiffness and tank to t + dt."""
        Vd, Kdot, Krdot, p_T, dt = self._pending
        self.x_d = self.x_d + Vd * dt
        if np.any(Krdot):
            # Eigenvalue projection, as for K.  This is where an UNREACHABLE
            # felt profile is refused: KR_from_felt can be indefinite, and the
            # clip at kr_lo > 0 is what rejects it instead of approximating it.
            w, V = np.linalg.eigh(0.5 * (self.Kr + Krdot * dt + (self.Kr + Krdot * dt).T))
            self.Kr = (V * np.clip(w, self.g.kr_lo, self.g.kr_hi)) @ V.T
        if np.any(Kdot):
            K = self.K + Kdot * dt
            # project onto the bounded SPD set, eigenvalue-wise
            w, V = np.linalg.eigh(0.5 * (K + K.T))
            self.K = (V * np.clip(w, self.g.k_lo, self.g.k_hi)) @ V.T
        self.E = self.E - p_T * dt
        self._k += 1

    # ------------------------------------------------------------------ #
    def ik(self, p: Array, R: Array, q_init: Array) -> tuple[Array, bool]:
        import sapien
        from scipy.spatial.transform import Rotation
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        q, ok, _ = self.pm.compute_inverse_kinematics(
            self.ee, sapien.Pose(p=p, q=[w, x, y, z]), initial_qpos=q_init,
            active_qmask=np.ones(self.nq, dtype=np.int32), max_iterations=200)
        return q, bool(ok)
