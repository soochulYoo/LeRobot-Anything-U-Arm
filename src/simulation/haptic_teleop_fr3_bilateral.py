"""Full bilateral cascade: human wrench -> master -> transformer -> wave channel
(with real transmission delay) -> virtual coupling -> outer admittance -> slave,
and back: slave/environment -> admittance -> coupling -> wave channel -> master
-> "felt" reflected wrench. This is the fuller loop from Sec. II-IV of the
manuscript; haptic_teleop_fr3_demo.py implements only the forward half (a
scripted/kinematic master, no wave channel, no reflected force).

TRANSLATION ONLY (for now). Orientation is frozen at each arm's starting
value throughout. This is a deliberate scope cut: an earlier version that
carried rotation through the wave channel + coupling produced the same kind
of frame-coupling instability (the slave slipping/drifting) already seen
elsewhere in this project. Since the requested test scenario -- a uniform
downward push, feeling the box through the channel -- is inherently a 1D/3D
translational experiment, dropping rotation removes a large source of risk
without losing anything the test needs. Add it back only once this is solid.

WHAT'S DIFFERENT FROM haptic_teleop_fr3_demo.py
------------------------------------------------
- The master is now a real (if virtual) DYNAMICAL SYSTEM: it responds to a
  human wrench f_h minus the reflected wrench f_m, via its own mass/damping
  (paper Eq. 7), instead of being scripted or keyboard-teleported. There is no
  real haptic device, so f_h is a synthetic HUMAN IMPEDANCE MODEL: a
  spring-damper reaching for a target displacement, f_h = K_h(p_target - p_m)
  - B_h V_m, NOT a constant force. A constant force is the wrong test signal
  here: with no spring in the master's own dynamics (Eq. 7 has none), any
  constant f_h forces V_m -> 0 and f_m -> f_h at steady state BY
  CONSTRUCTION, regardless of K_a, M_a, or how stiff the box is -- so a
  constant-force test can never show anything about the environment or the
  channel, only reproduce the number you dialed in. A spring-damper human has
  a real equilibrium set by the ratio of K_h to the coupling/environment
  stiffness, which is exactly the paper's own Sec. V-A quasistatic result
  (delta_x_s/delta_x_m = k_a/(k_a+k_e)) -- and it's also closer to how a real
  human actually pushes (they stop increasing force as they feel resistance,
  they don't keep shoving with the same constant force forever).
- There is an actual WAVE CHANNEL (paper Eq. 9-12) with a real transmission
  delay (Tf forward, Tb backward), selectable via --channel-mode wave. A
  --channel-mode direct alternative is also provided: the same signals with
  only a plain transport delay and NO scattering transform, to make the
  wave-variable formalism's actual contribution visible by comparison (the
  classical Anderson-Spong result: naive delayed force/velocity feedback can
  go non-passive -- inject energy -- as delay grows; the wave transform
  structurally cannot).
- The coupling wrench f_ch genuinely flows in BOTH directions: forward into
  the outer admittance, and backward through the channel into f_m.
- Channel passivity is logged and checked numerically each step: H_ch (energy
  presently stored in transit, computed from the wave buffers -- a sum of
  squares, so >= 0 by construction) is compared against the accumulated port
  power integral from Eq. 12 (Hdot_ch = F_c.V_c^tx - f_ch.V_c); for the wave
  channel these must match almost exactly (a numerical check that the
  implementation is bug-free, not just "probably passive"), and for the
  direct channel the same integral can go negative at large delay -- the
  actual demonstration of what the channel buys you.

HOW THE MASTER'S "PHYSICS" IS SIMULATED
-----------------------------------------
Per this repo's established finding (see haptic_teleop_fr3_demo.py's
docstring): torque-controlling a real SAPIEN articulation with hand-computed
gravity compensation was unreliable here. So neither arm is torque-controlled.
The master's dynamics (Eq. 7, simplified to a translation-only virtual point
mass) are integrated in Python; the actual xArm6 in the scene just
kinematically tracks that virtual position via IK + its own PD position
drive -- the same trick already used for the slave, applied to both arms.

WAVE-CHANNEL DISCRETE-TIME SOLVE
-----------------------------------
The paper's Eq. 10 defines four wave variables from (F_c, V_c^tx) at the
master and (f_ch, V_c) at the slave. At the master, V_c^tx is directly known
(the master's own measured velocity), so F_c is a direct, explicit solve from
the received (delayed) wave. At the slave, V_c is not independently known --
only f_ch is defined in terms of it (the coupling law, Eq. 13) -- so naively
this looks circular. It isn't: f_ch's elastic part depends on the PREVIOUS
step's decoded position g_c (already integrated), not on this step's
velocity, so substituting the coupling law into the wave equation gives one
clean linear equation in the one true unknown, V_c, each step.

OTHER SIMPLIFICATIONS (in addition to the ones in haptic_teleop_fr3_demo.py):
- The coupling/admittance transports (A_cr, A_sr) are treated as IDENTITY
  rather than the full SE(3) adjoint: master and slave don't share a
  workspace, and the adjoint's moment-arm term is what caused a runaway
  instability earlier in this project when the two frames sit far apart.
- Master and slave inertial/damping parameters are chosen for a legible demo,
  not fit to any real hardware.

Usage:
    python haptic_teleop_fr3_bilateral.py --duration 10.0 --human-reach -0.08
    python haptic_teleop_fr3_bilateral.py --realtime --human-reach -0.08
    python haptic_teleop_fr3_bilateral.py --channel-mode direct --Tf 0.2 --Tb 0.2
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import sapien

import gymnasium as gym
import mani_skill.envs  # noqa: F401
from mani_skill.utils import sapien_utils

from haptic_teleop_fr3_demo import (
    HapticTeleopDemoEnv,  # noqa: F401  (registers "HapticTeleopDemo-v1")
    R_to_quat_wxyz,
    build_pinocchio,
    joint_limit_avoidance_grad,
    quat_wxyz_to_R,
)


@dataclass
class BilateralGains:
    # master virtual dynamics (Eq. 7, translation only)
    Mm: np.ndarray = field(default_factory=lambda: np.diag([2.0] * 3))
    Bm: np.ndarray = field(default_factory=lambda: np.diag([10.0] * 3))
    # human-as-impedance model: f_h = Kh(p_target - pm) - Bh*Vm. This is k_e
    # in the paper's Sec. V-A quasistatic transmission ratio k_a/(k_a+k_e).
    Kh: float = 400.0
    Bh: float = 20.0
    # wave channel impedance-matching gain b (Eq. 10); larger = more "rigid"-feeling
    b: np.ndarray = field(default_factory=lambda: np.diag([10.0] * 3))
    Tf: float = 0.05  # forward (master -> slave) transmission delay, s
    Tb: float = 0.05  # backward (slave -> master) transmission delay, s
    channel_mode: str = "wave"  # "wave" (Eq. 9-12) or "direct" (plain delay, no scattering transform)
    # virtual coupling (Eq. 13), A_cr/A_sr treated as identity -- see docstring
    Ka: np.ndarray = field(default_factory=lambda: np.diag([100.0] * 3))
    Ba: np.ndarray = field(default_factory=lambda: np.diag([25.0] * 3))
    # outer admittance (Eq. 15), A_cr/A_sr treated as identity
    Ma: np.ndarray = field(default_factory=lambda: np.diag([3.0] * 3))
    Br: np.ndarray = field(default_factory=lambda: np.diag([25.0] * 3))
    f_e_limit: float = 15.0
    # ---- inner Cartesian impedance (--inner impedance) ----
    # The stage the original version skipped; see do_step step 6.  Ki enters the
    # cascade's steady state exactly like the environment stiffness -- the series
    # compliance is 1 + Ka/ke + Ka/Ki -- so Ki must sit well above the outer
    # coupling Ka or the commanded force never arrives.  Validated against the
    # closed form in cascade/analytic.py.
    Ki: np.ndarray = field(default_factory=lambda: np.diag([2000.0] * 3))
    Di: np.ndarray = field(default_factory=lambda: np.diag([120.0] * 3))
    # Rotational stiffness, Nm/rad.  A scalar means Kr*I; a 3x3 matrix is an
    # ANISOTROPIC rotational stiffness, which GIC's error term
    # e_R = 1/2 vee(K_R R_d^T R - R^T R_d K_R) already supports -- it is written
    # for a matrix K_R and only ever handed a scalar here.  For an insertion the
    # useful shape is soft about the two axes perpendicular to the hole, so the
    # peg can pivot into alignment, and stiff about the hole axis.
    Kr: "float | np.ndarray" = 60.0
    # Matched to the wrist's inertia (~0.003 kg m^2), as 2 zeta sqrt(I Kr).  The
    # previous 6.0 Nms/rad came from 2 zeta sqrt(Kr), i.e. from assuming
    # 1 kg m^2, and sits past the explicit stability limit 2I/dt = 3 Nms/rad at
    # a 2 ms step -- stable enough to run, but it is one source of the contact
    # chatter seen in the wipe experiments.
    Dr: float = 0.85        # rotational damping, Nms/rad
    # Where the COMPLIANCE CENTRE sits, as an offset in the end-effector frame.
    # Moving the controlled point is the same thing as adding the coupling
    # blocks of a 6x6 stiffness: expressing a block-diagonal K at a point offset
    # by r gives K_tr = -K_t r_hat and K_rr = K_r - r_hat K_t r_hat.  A block
    # diagonal stiffness at the wrist can only put the compliance centre at the
    # wrist; an RCC works by putting it at the tool tip, so a lateral force
    # there pivots the tool into alignment instead of jamming it.
    tool_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    null_kp: float = 5.0    # null-space posture pull toward joint mid-range
    null_kd: float = 3.0    # null-space damping
    # Reachable-workspace clamp for pc/pr before they're ever handed to IK.
    # compute_inverse_kinematics does not gracefully saturate for unreachable
    # targets -- it returns increasingly nonsensical joint solutions the
    # further the target drifts past the real workspace, which without this
    # clamp shows up as the slave suddenly slipping sideways in x/y for no
    # apparent reason once the admittance reference overshoots past the box.
    workspace_lo: np.ndarray = field(default_factory=lambda: np.array([0.40, -0.20, -0.02]))
    workspace_hi: np.ndarray = field(default_factory=lambda: np.array([0.75, 0.20, 0.35]))


def clamp_with_velocity(p, v, lo, hi):
    """Clamp position to [lo, hi], and zero the velocity component wherever
    the clamp is active, so the integrator doesn't keep "pressing" against an
    invisible wall with unbounded stored velocity."""
    p_clamped = np.clip(p, lo, hi)
    v_clamped = v.copy()
    v_clamped[(p <= lo) & (v < 0)] = 0.0
    v_clamped[(p >= hi) & (v > 0)] = 0.0
    return p_clamped, v_clamped


def get_pose(link):
    pose = link.pose
    p = pose.p.cpu().numpy().reshape(3)
    q = pose.q.cpu().numpy().reshape(4)
    return quat_wxyz_to_R(q), p


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector, for the orientation error term."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-9:
        return np.zeros(3)
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def vee(S: np.ndarray) -> np.ndarray:
    """so(3) skew matrix -> 3-vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def skew(a: np.ndarray) -> np.ndarray:
    """3-vector -> so(3) skew matrix, with skew(a) b = a x b."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def geometric_impedance_torque(agent, link, pmodel, ee_idx, idx, gains,
                               p_ref, v_ref, R_ref, q_mid):
    """Geometric Impedance Control on SE(3) (Seo et al.), the inner loop.

    Why this rather than the world-frame Cartesian law below: the stiffness is
    applied in the DESIRED (task) frame and only then rotated into the body
    frame,

        f_p = -R^T R_d K_p R_d^T (p - p_d)
        e_R = 1/2 vee(K_R R_d^T R - R^T R_d K_R),   m_R = -e_R
        F_b = [f_p; m_R] - K_d e_v,                 tau = J_b^T F_b

    so an ANISOTROPIC K_p stays attached to the task, not to whatever
    orientation the end effector happens to hold.  The naive law applies K in
    the world frame while taking the rotation error in the body frame, which is
    frame-inconsistent: tilt the task and the soft axis no longer points along
    the surface normal.  That is precisely the quantity the anisotropy study
    (T1) measures, so the controller must not corrupt it.

    Two further consequences.  e_R is the gradient of a genuine potential on
    SO(3), so the law is a passive energy-shaping controller rather than a
    heuristic error feedback.  And SAPIEN's compute_single_link_local_jacobian
    already returns the BODY Jacobian, so GIC needs no frame rotation of J at
    all -- the naive law did, and getting that rotation wrong is silent.

    e_v uses Ad_{g^-1 g_d} V_d, which for a translating reference with fixed
    desired orientation reduces exactly to [R^T v_ref; 0].
    """
    robot = agent.robot
    q = robot.get_qpos()[0].cpu().numpy()
    qd = robot.get_qvel()[0].cpu().numpy()
    R, p = get_pose(link)

    J_b = pmodel.compute_single_link_local_jacobian(q, ee_idx)[:, idx]  # body Jacobian

    # Shift the controlled point to the compliance centre.  The velocity of a
    # body point a (body coords) is v + w x a, so the Jacobian there is
    # [[I, -a_hat], [0, I]] J_b, and the same matrix transposed carries the
    # wrench back to the wrist.  Everything below then acts AT the tool tip.
    a = np.asarray(gains.tool_offset, dtype=float).reshape(3)
    if np.any(a):
        Ad = np.block([[np.eye(3), -skew(a)], [np.zeros((3, 3)), np.eye(3)]])
        J_b = Ad @ J_b
        p = p + R @ a
    V_b = J_b @ qd[idx]                                                 # [v_b; w_b]

    K_p = gains.Ki
    K_R = gains.Kr if np.ndim(gains.Kr) == 2 else gains.Kr * np.eye(3)
    f_p = -R.T @ R_ref @ K_p @ R_ref.T @ (p - p_ref)
    e_R = 0.5 * vee(K_R @ R_ref.T @ R - R.T @ R_ref @ K_R)

    e_v = V_b - np.concatenate([R.T @ v_ref, np.zeros(3)])
    D_r = gains.Dr if np.ndim(gains.Dr) == 2 else gains.Dr * np.eye(3)
    K_d = np.block([[gains.Di, np.zeros((3, 3))],
                    [np.zeros((3, 3)), D_r]])
    F_b = np.concatenate([f_p, -e_R]) - K_d @ e_v
    tau = J_b.T @ F_b

    N = np.eye(len(idx)) - J_b.T @ np.linalg.pinv(J_b.T)
    tau = tau + N @ (gains.null_kp * (q_mid - q[idx]) - gains.null_kd * qd[idx])
    return tau, R @ F_b[:3]   # report the applied force in WORLD frame, as the log expects


def cartesian_impedance_torque(agent, link, pmodel, ee_idx, idx, gains,
                               p_ref, v_ref, R_ref, q_mid):
    """Inner Cartesian impedance as joint torques: tau = J^T [f; m] + null-space.

        f = Ki (p_ref - p) + Di (v_ref - v)
        m = Kr log(R_ref R^T) - Dr omega

    NO GRAVITY COMPENSATION, deliberately.  Every link in this scene carries
    disable_gravity=True, so the arm does not fall on its own; a compensation
    torque here cancels nothing and instead ACCELERATES a weightless arm.
    Measured: zero torque drifts 0.00 mm in 0.5 s, while applying
    compute_passive_force() throws the end effector 998 mm.  The torque itself
    is correct -- compute_passive_force and pinocchio's inverse dynamics agree
    to the digit -- it simply has nothing to cancel here.  Restore the term only
    if gravity is ever enabled on these links.

    compute_single_link_local_jacobian returns the Jacobian in the LINK frame;
    both its linear and angular blocks are rotated into world by R before use
    (verified against a finite-difference end-effector velocity to 2.5e-4).
    """
    robot = agent.robot
    q = robot.get_qpos()[0].cpu().numpy()
    qd = robot.get_qvel()[0].cpu().numpy()
    R_now, p_now = get_pose(link)

    J_local = pmodel.compute_single_link_local_jacobian(q, ee_idx)
    J = np.vstack([R_now @ J_local[:3], R_now @ J_local[3:]])[:, idx]
    twist = J @ qd[idx]

    f_cmd = gains.Ki @ (p_ref - p_now) + gains.Di @ (v_ref - twist[:3])
    m_cmd = gains.Kr * so3_log(R_ref @ R_now.T) - gains.Dr * twist[3:]
    tau = J.T @ np.concatenate([f_cmd, m_cmd])

    # Posture term projected into the torque null space so it cannot disturb the
    # commanded wrench -- the redundant joint is kept near mid-range the same way
    # track_via_ik's null_gain does it for the IK path.
    N = np.eye(len(idx)) - J.T @ np.linalg.pinv(J.T)
    tau = tau + N @ (gains.null_kp * (q_mid - q[idx]) - gains.null_kd * qd[idx])
    return tau, f_cmd


class DelayLine:
    """Fixed-length transport delay for a 3-vector signal, at a known dt."""

    def __init__(self, delay_s: float, dt: float):
        self.n = max(0, int(round(delay_s / dt)))
        self.buf = [np.zeros(3) for _ in range(self.n)]

    def push_and_get(self, value: np.ndarray) -> np.ndarray:
        """Push `value` (sent now) and return what arrives now (sent `delay` ago)."""
        if self.n == 0:
            return value
        self.buf.append(value.copy())
        return self.buf.pop(0)

    def stored_energy(self, dt: float) -> float:
        """Discretization of paper Eq. 11's 0.5 * integral(||w||^2) over the
        in-flight window: each buffered sample represents dt of transit time."""
        return 0.5 * dt * sum(float(w @ w) for w in self.buf)


def run_realtime(args, unwrapped, viewer, dt, do_step, log_step, log_every):
    """Real-time playback: a fixed-timestep accumulator ties simulated time to
    the wall clock regardless of how fast rendering happens to be (same
    pattern as haptic_teleop_fr3_interactive.py), a live matplotlib window
    shows the reflected/contact forces, and O/L adjust the human push force
    live so you can feel how the equilibrium force changes without
    restarting. ESC or closing the viewer window quits.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["toolbar"] = "none"
    plt.ion()
    fig, (ax_fm, ax_hist) = plt.subplots(1, 2, figsize=(9, 4))
    bar = ax_fm.bar(["f_h", "f_m", "f_e"], [0, 0, 0], color=["tab:gray", "tab:red", "tab:green"])
    ax_fm.set_ylim(-20, 20)
    ax_fm.set_title("z-forces (N): human push / felt at master / contact")
    hist_len = 300
    fm_hist = np.zeros(hist_len)
    (line,) = ax_hist.plot(fm_hist, color="tab:red")
    ax_hist.set_ylim(0, 20)
    ax_hist.set_title("|f_m| recent history")
    fig.tight_layout()
    fig.show()

    print("Real-time bilateral cascade. Press O/L to increase/decrease how far the human reaches, ESC to quit.")

    step = 0
    t = 0.0
    last_time = time.time()
    accumulator = 0.0
    plot_t0 = time.time()
    fps_t0 = time.time()
    fps_count = 0
    MAX_SUBSTEPS = 50
    plus_down_prev = minus_down_prev = False

    try:
        while not viewer.window.should_close:
            now = time.time()
            accumulator += min(now - last_time, 0.1)
            last_time = now

            if viewer.window.key_down("esc"):
                break
            plus_down = viewer.window.key_down("o")
            minus_down = viewer.window.key_down("l")
            if plus_down and not plus_down_prev:
                args.human_reach -= 0.01  # reach further down
                print(f"[human reach] {args.human_reach * 100:.1f} cm")
            if minus_down and not minus_down_prev:
                args.human_reach += 0.01
                print(f"[human reach] {args.human_reach * 100:.1f} cm")
            plus_down_prev, minus_down_prev = plus_down, minus_down

            n_sub = 0
            f_e = f_m = f_ch = f_h = pgs = H_ch = E_ch_val = None
            while accumulator >= dt and n_sub < MAX_SUBSTEPS:
                accumulator -= dt
                n_sub += 1
                f_e, f_m, f_ch, f_h, pgs, H_ch, E_ch_val = do_step(t)
                if step % log_every == 0:
                    log_step(t, f_e, f_m, f_ch, f_h, pgs, H_ch, E_ch_val)
                step += 1
                t += dt
                fps_count += 1

            unwrapped.render_human()

            if f_m is not None and now - plot_t0 >= 0.1:
                for rect, val in zip(bar, [f_h[2], f_m[2], f_e[2]]):
                    rect.set_height(val)
                fm_hist = np.roll(fm_hist, -1)
                fm_hist[-1] = np.linalg.norm(f_m)
                line.set_ydata(fm_hist)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plot_t0 = now

            if now - fps_t0 >= 1.0:
                print(
                    f"[perf] {fps_count / (now - fps_t0):.1f} physics steps/s   "
                    f"H_ch={H_ch:.4g} E_ch={E_ch_val:.4g}"
                )
                fps_t0 = now
                fps_count = 0
    finally:
        plt.close(fig)


def run(args):
    env = gym.make(
        "HapticTeleopDemo-v1",
        num_envs=1,
        sim_backend="cpu",
        render_mode="human" if args.realtime else None,
    )
    env.reset(seed=0)
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    dt = 1.0 / unwrapped.sim_freq

    viewer = None
    if args.realtime:
        unwrapped.render_human()
        viewer = unwrapped.viewer

    master, slave = unwrapped.agent.agents
    gains = BilateralGains(
        Kh=args.Kh, Bh=args.Bh, Tf=args.Tf, Tb=args.Tb, channel_mode=args.channel_mode,
        Ki=np.diag([args.Ki] * 3), Di=np.diag([args.Di] * 3),
    )

    # --- both arms: their own PD position drive; both are tracked via IK from
    # a virtual state, never torque-controlled directly (see module docstring) ---
    def setup_position_drive(agent):
        names = set(agent.arm_joint_names)
        joints = [j for j in agent.robot.active_joints if j.name in names]
        n = len(joints)
        stiffness = np.broadcast_to(agent.arm_stiffness, n)
        damping = np.broadcast_to(agent.arm_damping, n)
        force_limit = np.broadcast_to(agent.arm_force_limit, n)
        for i, j in enumerate(joints):
            j.set_drive_properties(float(stiffness[i]), float(damping[i]), force_limit=float(force_limit[i]))
        idx = np.array([i for i, j in enumerate(agent.robot.active_joints) if j.name in names])
        pmodel, link_order = build_pinocchio(agent.urdf_path, agent.robot)
        ee_index = link_order.index(agent.ee_link_name)
        qmask = np.zeros(len(agent.robot.active_joints), dtype=np.int32)
        qmask[idx] = 1
        qlimits = agent.robot.get_qlimits()[0].cpu().numpy()[idx]
        q_mid = (qlimits[:, 0] + qlimits[:, 1]) / 2.0
        q_half_range = (qlimits[:, 1] - qlimits[:, 0]) / 2.0
        return joints, idx, pmodel, ee_index, qmask, q_mid, q_half_range

    master_joints, master_idx, master_pmodel, master_ee_idx, master_qmask, _, _ = setup_position_drive(master)
    slave_joints, slave_idx, slave_pmodel, slave_ee_idx, slave_qmask, slave_q_mid, slave_q_half = setup_position_drive(slave)

    # In impedance mode the slave is torque controlled, so its joint PD drive has
    # to be switched off: leaving it on would put a stiff position servo in
    # series with the impedance and the measured compliance would be neither one.
    n_active_slave = len(slave.robot.active_joints)
    INNER_LAW = {"geometric": geometric_impedance_torque,
                 "cartesian": cartesian_impedance_torque}
    if args.inner in INNER_LAW:
        for j in slave_joints:
            j.set_drive_properties(0.0, 0.0, force_limit=1000.0)
        print(f"[inner] {args.inner} impedance  Kp={np.diag(gains.Ki)} N/m  "
              f"Dp={np.diag(gains.Di)} Ns/m  Kr={gains.Kr} Nm/rad  (slave joint PD disabled)")
    else:
        print("[inner] IK + joint PD position drive (no inner impedance)")

    def track_via_ik(
        agent, joints, idx, pmodel, ee_idx, qmask, R_target, p_target, t,
        q_mid=None, q_half_range=None, null_gain=0.0, quiet_fails=True,
    ):
        qpos_full = agent.robot.get_qpos()[0].cpu().numpy()
        target_pose = sapien.Pose(p=p_target, q=R_to_quat_wxyz(R_target))
        q_ik, ik_ok, ik_err = pmodel.compute_inverse_kinematics(
            ee_idx, target_pose, initial_qpos=qpos_full, active_qmask=qmask, max_iterations=20,
        )
        q_arm = q_ik[idx]

        # Null-space safety net for redundant arms (n joints > 6 task dims):
        # compute_inverse_kinematics has no joint-limit-avoidance or posture
        # objective of its own -- it just returns *a* valid solution near the
        # warm-start. This nudges the redundant DOF toward the center of its
        # range WITHOUT disturbing the achieved task pose (it's projected
        # through the null space), and only when a joint is actually close to
        # a limit (the dead-zone in joint_limit_avoidance_grad) so it doesn't
        # drag a joint away from a perfectly comfortable solution the way a
        # plain "always pull to center" term did earlier in this project.
        if null_gain > 0 and len(idx) > 6:
            pmodel.compute_forward_kinematics(q_ik)
            J = pmodel.compute_single_link_local_jacobian(q_ik, ee_idx)[:, idx]
            N = np.eye(len(idx)) - np.linalg.pinv(J) @ J
            grad = joint_limit_avoidance_grad(q_arm, q_mid, q_half_range)
            q_arm = q_arm + null_gain * (N @ grad)

        agent.robot.set_joint_drive_targets(q_arm[None, :], joints=joints)
        if not ik_ok and not quiet_fails:
            print(f"[IK FAIL] t={t:.3f} target={p_target} err={ik_err}")
        return ik_ok

    master_link = sapien_utils.get_obj_by_name(master.robot.get_links(), master.ee_link_name)
    slave_link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)

    # orientations are frozen at their starting values throughout (see docstring)
    R_master_fixed, pm = get_pose(master_link)
    _, pc = get_pose(slave_link)   # decoded slave-side command position g_c
    R_slave_fixed, pr = get_pose(slave_link)  # admittance reference position

    Vm = np.zeros(3)  # master's own virtual velocity
    Vr = np.zeros(3)  # admittance reference velocity

    # human-as-impedance: a spring-damper reaching for a fixed target offset
    # from the master's start pose (see module docstring for why this replaces
    # a constant test force).
    pm0 = pm.copy()  # fixed anchor; p_h_target is recomputed from this + args.human_reach
    # each step, so --realtime's O/L keys (which change args.human_reach) take effect live.

    fwd_line = DelayLine(gains.Tf, dt)   # master -> slave (w+, or raw Vc_tx in "direct" mode)
    bwd_line = DelayLine(gains.Tb, dt)   # slave -> master (w-, or raw f_ch in "direct" mode)
    E_ch = 0.0  # accumulated channel port-power integral (Eq. 12), for the passivity check

    log = {k: [] for k in ["t", "pm", "pc", "pr", "ps", "fe", "fm", "fch", "fh", "Hch", "Ech"]}

    b = gains.b
    sqrt_b = np.sqrt(b)
    inv_sqrt_b = np.diag(1.0 / np.diag(sqrt_b))
    Ba = gains.Ba

    def do_step(t):
        """One physics step of the full cascade. Mutates the enclosing
        pm/pc/pr/Vm/Vr/E_ch state via nonlocal so both the batch loop and the
        real-time loop below can share this single implementation."""
        nonlocal pm, pc, pr, Vm, Vr, E_ch

        p_h_target = pm0 + np.array([0.0, 0.0, args.human_reach])
        f_h_world = gains.Kh * (p_h_target - pm) - gains.Bh * Vm

        # ---- 3. transformer: V_c^tx = A @ V_m, A = identity here ----
        Vc_tx = Vm.copy()

        if gains.channel_mode == "wave":
            # ---- 4a. wave channel, MASTER side: F_c is a direct solve (Eq. 10) ----
            w_minus_m = bwd_line.push_and_get(np.zeros(3))  # placeholder; real push below
            F_c = np.sqrt(2) * sqrt_b @ w_minus_m + b @ Vc_tx
            w_plus_m = w_minus_m + np.sqrt(2) * sqrt_b @ Vc_tx
            fwd_line_out = fwd_line.push_and_get(w_plus_m)  # this is w+_s(t) at the slave
        else:  # "direct": plain transport delay, no scattering transform at all
            F_c = bwd_line.push_and_get(np.zeros(3))  # placeholder; real push below
            fwd_line_out = fwd_line.push_and_get(Vc_tx)  # this is V_c(t) at the slave, straight

        f_m = F_c  # transformer back to master, A^T = identity
        Vm_dot = np.linalg.solve(gains.Mm, f_h_world - f_m - gains.Bm @ Vm)
        Vm = Vm + Vm_dot * dt
        pm = pm + Vm * dt
        track_via_ik(master, master_joints, master_idx, master_pmodel, master_ee_idx, master_qmask, R_master_fixed, pm, t)

        # ---- 4b. channel, SLAVE side: get V_c, then compute f_ch (Eq. 13) ----
        spring_prev = gains.Ka @ (pc - pr)  # uses the PREVIOUS step's decoded g_c -- not circular

        if gains.channel_mode == "wave":
            w_plus_s = fwd_line_out
            lhs = Ba @ inv_sqrt_b + sqrt_b
            rhs = np.sqrt(2) * w_plus_s - inv_sqrt_b @ spring_prev + inv_sqrt_b @ (Ba @ Vr)
            Vc = np.linalg.solve(lhs, rhs)
            f_ch = spring_prev + Ba @ (Vc - Vr)
            w_minus_s = (inv_sqrt_b @ f_ch - sqrt_b @ Vc) / np.sqrt(2)
            if bwd_line.n > 0:
                bwd_line.buf[-1] = w_minus_s  # replace the placeholder pushed above
        else:  # "direct"
            Vc = fwd_line_out
            f_ch = spring_prev + Ba @ (Vc - Vr)
            if bwd_line.n > 0:
                bwd_line.buf[-1] = f_ch  # replace the placeholder pushed above -- no transform

        # ---- passivity bookkeeping (Eq. 12): Hdot_ch = F_c.V_c^tx - f_ch.V_c.
        # For the wave channel this must match H_ch (below) almost exactly --
        # a numerical bug check, since H_ch is a sum of squares and so is
        # structurally >= 0 (the passivity guarantee itself). For "direct"
        # mode this same generic port formula is still computed, and can go
        # unboundedly negative at large delay -- that's the actual
        # demonstration of what the wave transform buys you.
        E_ch += (float(F_c @ Vc_tx) - float(f_ch @ Vc)) * dt
        H_ch = (fwd_line.stored_energy(dt) + bwd_line.stored_energy(dt)) if gains.channel_mode == "wave" else float("nan")

        # pc is deliberately NOT clamped to the workspace: it represents
        # "where the human's command wants the slave to be," and letting it
        # keep integrating past the box is what makes the coupling spring
        # term (Ka @ (pc - pr)) grow with how hard the push continues past
        # the wall -- clamping both pc and pr to the same box would zero out
        # that gap and flatten the felt force to a constant.
        pc = pc + Vc * dt  # integrate the decoded slave-side command position

        # ---- contact force (reused mechanism from haptic_teleop_fr3_demo.py) ----
        # f_e_raw is what actually gets logged/plotted; f_e_ctrl (clipped) is
        # only for the admittance math below, where the clip is a stability
        # safety net, not a sensor limitation -- clipping the logged value
        # too would flatten every contact scenario to the same plateau (as it
        # did before this fix), hiding the real force/depth relationship.
        contact = slave.robot.get_net_contact_forces(unwrapped.CONTACT_LINK_NAMES)[0].sum(axis=0)
        f_e_raw = contact.cpu().numpy()
        f_e = np.clip(f_e_raw, -gains.f_e_limit, gains.f_e_limit)

        # ---- 5. outer admittance (Eq. 15, A_cr = A_sr = identity) ----
        # SIGN: f_e here is get_net_contact_forces, i.e. the reaction ON the
        # robot, already a world vector pointing OPPOSITE to the penetration
        # (the human pushes -z and f_e_z comes back +150 N).  It therefore has
        # to be ADDED.  Subtracting it -- as this line did originally -- made
        # the wall's resistance drive the reference further INTO the wall:
        # positive feedback that ran away the instant contact was made and
        # stopped only at the workspace clamp, at 200+ N of contact force with
        # pr_z pinned to workspace_lo.  (That runaway is what the clamp comment
        # above describes as the reference "overshooting past the box".)
        # The scalar form in cascade/analytic.py subtracts f_e because there the
        # positive axis points INTO the wall; along that axis both agree.
        Vr_dot = np.linalg.solve(gains.Ma, f_ch + f_e - gains.Br @ Vr)
        Vr = Vr + Vr_dot * dt
        pr = pr + Vr * dt
        pr, Vr = clamp_with_velocity(pr, Vr, gains.workspace_lo, gains.workspace_hi)

        # ---- 6. inner loop: the slave follows the admittance reference ----
        # Two interchangeable inner loops, which is exactly the comparison the
        # cascade study needs.  "ik" is the original position-tracking baseline
        # (IK plus the Panda's own joint PD, no Cartesian compliance at all);
        # "impedance" is the inner Cartesian impedance the cascade calls for.
        # The outer admittance above is identical in both, so any difference in
        # contact force is attributable to the inner loop alone.
        _, pgs = get_pose(slave_link)
        if args.inner in INNER_LAW:
            tau, f_inner = INNER_LAW[args.inner](
                slave, slave_link, slave_pmodel, slave_ee_idx, slave_idx, gains,
                pr, Vr, R_slave_fixed, slave_q_mid,
            )
            qf = np.zeros(n_active_slave)
            qf[slave_idx] = tau
            slave.robot.set_qf(qf[None, :])
        else:
            track_via_ik(
                slave, slave_joints, slave_idx, slave_pmodel, slave_ee_idx, slave_qmask, R_slave_fixed, pr, t,
                q_mid=slave_q_mid, q_half_range=slave_q_half, null_gain=args.null_gain,
            )

        scene.step()
        return f_e_raw, f_m, f_ch, f_h_world, pgs, H_ch, E_ch

    def log_step(t, f_e, f_m, f_ch, f_h, pgs, H_ch, E_ch_val):
        log["t"].append(t)
        log["pm"].append(pm.copy())
        log["pc"].append(pc.copy())
        log["pr"].append(pr.copy())
        log["ps"].append(pgs.copy())
        log["fe"].append(f_e.copy())
        log["fm"].append(f_m.copy())
        log["fch"].append(f_ch.copy())
        log["fh"].append(f_h.copy())
        log["Hch"].append(H_ch)
        log["Ech"].append(E_ch_val)

    log_every = max(1, int(round(1.0 / (dt * args.log_hz))))

    if not args.realtime:
        n_steps = int(args.duration / dt)
        for step in range(n_steps):
            t = step * dt
            f_e, f_m, f_ch, f_h, pgs, H_ch, E_ch_val = do_step(t)
            if args.debug and step % 200 == 0:
                print(
                    f"t={t:5.2f} pm={np.round(pm,3)} pc={np.round(pc,3)} pr={np.round(pr,3)} "
                    f"pgs={np.round(pgs,3)} f_e={np.round(f_e,2)} f_m={np.round(f_m,2)} "
                    f"H_ch={H_ch:.4g} E_ch={E_ch_val:.4g}"
                )
            if step % log_every == 0:
                log_step(t, f_e, f_m, f_ch, f_h, pgs, H_ch, E_ch_val)
    else:
        run_realtime(args, unwrapped, viewer, dt, do_step, log_step, log_every)

    env.close()
    return log


def plot_log(log, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(log["t"])
    pm, pc, pr, ps = (np.array(log[k]) for k in ["pm", "pc", "pr", "ps"])
    fe, fm, fch, fh = (np.array(log[k]) for k in ["fe", "fm", "fch", "fh"])
    H_ch = np.array(log["Hch"])
    E_ch = np.array(log["Ech"])

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    axes[0].plot(t, pm[:, 2], label="master z (virtual)")
    axes[0].plot(t, pc[:, 2], "--", label="decoded g_c z (channel out)")
    axes[0].plot(t, pr[:, 2], ":", label="admittance ref z")
    axes[0].plot(t, ps[:, 2], "-.", label="slave actual z")
    axes[0].set_ylabel("z (m)")
    axes[0].set_title("Full cascade: master -> channel -> coupling/admittance -> slave (z only)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, np.linalg.norm(fe, axis=1), color="tab:green", label="|f_e| (slave-environment)")
    axes[1].set_ylabel("N")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, np.linalg.norm(fh, axis=1), color="tab:gray", label="|f_h| (human push)")
    axes[2].plot(t, np.linalg.norm(fm, axis=1), color="tab:red", label="|f_m| (felt at master)")
    axes[2].plot(t, np.linalg.norm(fch, axis=1), color="tab:purple", alpha=0.6, label="|f_ch| (coupling)")
    axes[2].set_ylabel("N")
    axes[2].set_title("Reflected wrench: what the human would feel through the channel")
    axes[2].legend(fontsize=8)

    axes[3].plot(t, H_ch, color="tab:blue", label="H_ch (stored, from wave buffers)")
    axes[3].plot(t, E_ch, "--", color="tab:orange", label="E_ch (accumulated port power, Eq. 12)")
    axes[3].axhline(0, color="k", linewidth=0.5)
    axes[3].set_ylabel("J")
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("Channel passivity check: H_ch should match E_ch almost exactly (wave mode)")
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def sweep(args):
    """Run the cascade to steady state for a range of --human-reach values and
    plot the resulting equilibrium |f_e| vs |f_h| -- the actual force/force
    transmission characteristic of the whole channel+coupling+environment,
    for both channel modes if requested. This is the same steady-state
    argument discussed earlier (Sec. V-A's k_a/(k_a+k_e)): each reach value
    puts the human spring at a different working point, and this plot is
    that curve traced out empirically instead of asserted analytically."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reaches = np.linspace(args.sweep_min, args.sweep_max, args.sweep_n)
    modes = ["wave", "direct"] if args.sweep_compare else [args.channel_mode]

    results = {m: {"fh": [], "fe": [], "fm": []} for m in modes}
    for mode in modes:
        for reach in reaches:
            run_args = argparse.Namespace(**vars(args))
            run_args.human_reach = float(reach)
            run_args.channel_mode = mode
            run_args.realtime = False
            run_args.debug = False
            print(f"[sweep] mode={mode} human_reach={reach:.3f} ...")
            log = run(run_args)

            # Steady-state window: wait --sweep-settle seconds past CONTACT
            # ONSET (not just "the last N% of the run"), so the impact
            # transient itself is never included, regardless of how early or
            # late contact happens to occur for a given reach depth.
            fe_norms = np.array([np.linalg.norm(v) for v in log["fe"]])
            t_arr = np.array(log["t"])
            contact_idx = np.argmax(fe_norms > 0.5)  # first index where contact starts
            has_contact = fe_norms[contact_idx] > 0.5
            if has_contact:
                settle_start = t_arr[contact_idx] + args.sweep_settle
                window = t_arr >= settle_start
                if not window.any():  # run ended before the settle period elapsed
                    window = t_arr >= t_arr[-1] - 1e-9  # fall back to the last sample
                    print(
                        f"  [warn] contact at t={t_arr[contact_idx]:.2f}s but run ended before "
                        f"the {args.sweep_settle:.1f}s settle window -- consider --duration higher"
                    )
            else:
                window = t_arr >= t_arr[-1] - 1e-9  # no contact: just use the tail, it's ~0 anyway

            fh_ss = np.mean([np.linalg.norm(v) for v, m in zip(log["fh"], window) if m])
            fe_ss = np.mean([np.linalg.norm(v) for v, m in zip(log["fe"], window) if m])
            fm_ss = np.mean([np.linalg.norm(v) for v, m in zip(log["fm"], window) if m])
            results[mode]["fh"].append(fh_ss)
            results[mode]["fe"].append(fe_ss)
            results[mode]["fm"].append(fm_ss)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"wave": "tab:blue", "direct": "tab:orange"}
    for mode in modes:
        fh = np.array(results[mode]["fh"])
        fe = np.array(results[mode]["fe"])
        order = np.argsort(fh)
        ax.plot(fh[order], fe[order], "o-", color=colors.get(mode, None), label=f"{mode} channel")
    lims = [0, max(np.max(results[m]["fh"]) for m in modes) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="f_e = f_h (perfect transmission)")
    ax.set_xlim(lims)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("steady-state |f_h| (N, human push)")
    ax.set_ylabel("steady-state |f_e| (N, contact force)")
    ax.set_title("Force transmission: contact force felt vs. how hard the human pushes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.sweep_out, dpi=150)
    print(f"Saved sweep plot to {args.sweep_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--human-reach", type=float, default=-0.16,
        help="the human's intended downward hand displacement, m (spring-damper target offset)",
    )
    parser.add_argument("--Kh", type=float, default=400.0, help="human arm stiffness (k_e in paper Sec. V-A)")
    parser.add_argument("--Bh", type=float, default=20.0, help="human arm damping")
    parser.add_argument("--channel-mode", choices=["wave", "direct"], default="wave")
    parser.add_argument("--Tf", type=float, default=0.05, help="forward (master->slave) delay, s")
    parser.add_argument("--Tb", type=float, default=0.05, help="backward (slave->master) delay, s")
    parser.add_argument("--log-hz", type=float, default=100.0)
    parser.add_argument("--out", type=str, default="haptic_teleop_bilateral_result.png")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--realtime", action="store_true",
        help="open a live 3D viewer + force display instead of running headless and saving a plot",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="instead of a single run, sweep --human-reach and plot the steady-state f_e vs f_h curve",
    )
    parser.add_argument("--sweep-min", type=float, default=-0.06, help="shallowest reach in the sweep, m")
    parser.add_argument("--sweep-max", type=float, default=-0.30, help="deepest reach in the sweep, m")
    parser.add_argument("--sweep-n", type=int, default=10, help="number of reach values in the sweep")
    parser.add_argument(
        "--sweep-settle", type=float, default=1.5,
        help="seconds to wait after contact ONSET before averaging f_h/f_e, to exclude the impact transient",
    )
    parser.add_argument(
        "--sweep-compare", action="store_true",
        help="sweep both wave and direct channel modes and overlay them",
    )
    parser.add_argument("--sweep-out", type=str, default="haptic_teleop_fe_vs_fh.png")
    parser.add_argument("--Ki", type=float, default=2000.0,
                        help="inner Cartesian stiffness, N/m (--inner impedance only)")
    parser.add_argument("--Di", type=float, default=120.0,
                        help="inner Cartesian damping, Ns/m (--inner impedance only)")
    parser.add_argument(
        "--inner", choices=["ik", "cartesian", "geometric"], default="geometric",
        help="inner loop: 'ik' = IK + joint PD position tracking (the original baseline, "
             "no Cartesian compliance); 'cartesian' = naive world-frame impedance; "
             "'geometric' = Geometric Impedance Control on SE(3) (default)",
    )
    parser.add_argument(
        "--null-gain", type=float, default=0.05,
        help="strength of the post-IK null-space nudge that keeps the slave's redundant "
             "joint away from its limits without affecting the achieved task pose; 0 disables it",
    )
    args = parser.parse_args()

    if args.sweep:
        sweep(args)
    else:
        log = run(args)
        if not args.realtime:
            plot_log(log, args.out)
