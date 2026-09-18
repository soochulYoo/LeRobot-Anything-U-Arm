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
  real haptic device, so f_h here is a synthetic constant test wrench (a
  uniform push), and f_m is logged/plotted rather than physically rendered.
- There is an actual WAVE CHANNEL (paper Eq. 9-12) with a real transmission
  delay (Tf forward, Tb backward), not a direct connection.
- The coupling wrench f_ch genuinely flows in BOTH directions: forward into
  the outer admittance, and backward through the channel into f_m.

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
    python haptic_teleop_fr3_bilateral.py --duration 10.0 --human-force -5.0
    python haptic_teleop_fr3_bilateral.py --realtime --human-force -3.0
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
    quat_wxyz_to_R,
)


@dataclass
class BilateralGains:
    # master virtual dynamics (Eq. 7, translation only)
    Mm: np.ndarray = field(default_factory=lambda: np.diag([2.0] * 3))
    Bm: np.ndarray = field(default_factory=lambda: np.diag([10.0] * 3))
    # wave channel impedance-matching gain b (Eq. 10); larger = more "rigid"-feeling
    b: np.ndarray = field(default_factory=lambda: np.diag([10.0] * 3))
    Tf: float = 0.05  # forward (master -> slave) transmission delay, s
    Tb: float = 0.05  # backward (slave -> master) transmission delay, s
    # virtual coupling (Eq. 13), A_cr/A_sr treated as identity -- see docstring
    Ka: np.ndarray = field(default_factory=lambda: np.diag([100.0] * 3))
    Ba: np.ndarray = field(default_factory=lambda: np.diag([25.0] * 3))
    # outer admittance (Eq. 15), A_cr/A_sr treated as identity
    Ma: np.ndarray = field(default_factory=lambda: np.diag([3.0] * 3))
    Br: np.ndarray = field(default_factory=lambda: np.diag([25.0] * 3))
    f_e_limit: float = 15.0
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

    print("Real-time bilateral cascade. Press O/L to increase/decrease the human push force, ESC to quit.")

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
                args.human_force -= 1.0  # more negative = push down harder
                print(f"[human force] {args.human_force:.1f} N")
            if minus_down and not minus_down_prev:
                args.human_force += 1.0
                print(f"[human force] {args.human_force:.1f} N")
            plus_down_prev, minus_down_prev = plus_down, minus_down

            n_sub = 0
            f_e = f_m = f_ch = pgs = None
            while accumulator >= dt and n_sub < MAX_SUBSTEPS:
                accumulator -= dt
                n_sub += 1
                f_e, f_m, f_ch, pgs = do_step()
                if step % log_every == 0:
                    log_step(t, f_e, f_m, f_ch, pgs)
                step += 1
                t += dt
                fps_count += 1

            unwrapped.render_human()

            if f_m is not None and now - plot_t0 >= 0.1:
                for rect, val in zip(bar, [args.human_force, f_m[2], f_e[2]]):
                    rect.set_height(val)
                fm_hist = np.roll(fm_hist, -1)
                fm_hist[-1] = np.linalg.norm(f_m)
                line.set_ydata(fm_hist)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plot_t0 = now

            if now - fps_t0 >= 1.0:
                print(f"[perf] {fps_count / (now - fps_t0):.1f} physics steps/s")
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
    gains = BilateralGains()

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
        return joints, idx, pmodel, ee_index, qmask

    master_joints, master_idx, master_pmodel, master_ee_idx, master_qmask = setup_position_drive(master)
    slave_joints, slave_idx, slave_pmodel, slave_ee_idx, slave_qmask = setup_position_drive(slave)

    def track_via_ik(agent, joints, idx, pmodel, ee_idx, qmask, R_target, p_target):
        qpos_full = agent.robot.get_qpos()[0].cpu().numpy()
        target_pose = sapien.Pose(p=p_target, q=R_to_quat_wxyz(R_target))
        q_ik, ik_ok, ik_err = pmodel.compute_inverse_kinematics(
            ee_idx, target_pose, initial_qpos=qpos_full, active_qmask=qmask, max_iterations=20,
        )
        agent.robot.set_joint_drive_targets(q_ik[idx][None, :], joints=joints)
        return ik_ok

    master_link = sapien_utils.get_obj_by_name(master.robot.get_links(), master.ee_link_name)
    slave_link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)

    # orientations are frozen at their starting values throughout (see docstring)
    R_master_fixed, pm = get_pose(master_link)
    _, pc = get_pose(slave_link)   # decoded slave-side command position g_c
    R_slave_fixed, pr = get_pose(slave_link)  # admittance reference position
    Rc = R_master_fixed  # only used for logging consistency; not otherwise needed

    Vm = np.zeros(3)  # master's own virtual velocity
    Vr = np.zeros(3)  # admittance reference velocity

    fwd_line = DelayLine(gains.Tf, dt)   # master -> slave (w+)
    bwd_line = DelayLine(gains.Tb, dt)   # slave -> master (w-)

    f_h_world = np.array([0.0, 0.0, args.human_force])  # constant test "human push"

    log = {k: [] for k in ["t", "pm", "pc", "pr", "ps", "fe", "fm", "fch"]}

    b = gains.b
    sqrt_b = np.sqrt(b)
    inv_sqrt_b = np.diag(1.0 / np.diag(sqrt_b))
    Ba = gains.Ba

    def do_step():
        """One physics step of the full cascade. Mutates the enclosing
        pm/pc/pr/Vm/Vr state via nonlocal so both the batch loop and the
        real-time loop below can share this single implementation."""
        nonlocal pm, pc, pr, Vm, Vr

        # ---- 3. transformer: V_c^tx = A @ V_m, A = identity here ----
        Vc_tx = Vm.copy()

        # ---- 4a. wave channel, MASTER side: F_c is a direct solve (Eq. 10) ----
        w_minus_m = bwd_line.push_and_get(np.zeros(3))  # placeholder push; real push below
        F_c = np.sqrt(2) * sqrt_b @ w_minus_m + b @ Vc_tx
        w_plus_m = w_minus_m + np.sqrt(2) * sqrt_b @ Vc_tx
        fwd_line_out = fwd_line.push_and_get(w_plus_m)  # this is w+_s(t) at the slave

        f_m = F_c  # transformer back to master, A^T = identity
        Vm_dot = np.linalg.solve(gains.Mm, f_h_world - f_m - gains.Bm @ Vm)
        Vm = Vm + Vm_dot * dt
        pm = pm + Vm * dt
        track_via_ik(master, master_joints, master_idx, master_pmodel, master_ee_idx, master_qmask, R_master_fixed, pm)

        # ---- 4b. wave channel, SLAVE side: solve for V_c, then f_ch ----
        w_plus_s = fwd_line_out
        spring_prev = gains.Ka @ (pc - pr)  # uses the PREVIOUS step's decoded g_c -- not circular

        lhs = Ba @ inv_sqrt_b + sqrt_b
        rhs = np.sqrt(2) * w_plus_s - inv_sqrt_b @ spring_prev + inv_sqrt_b @ (Ba @ Vr)
        Vc = np.linalg.solve(lhs, rhs)

        f_ch = spring_prev + Ba @ (Vc - Vr)
        w_minus_s = (inv_sqrt_b @ f_ch - sqrt_b @ Vc) / np.sqrt(2)
        if bwd_line.n > 0:
            bwd_line.buf[-1] = w_minus_s  # replace the placeholder pushed above with the real value

        # pc is deliberately NOT clamped to the workspace: it represents
        # "where the human's command wants the slave to be," and letting it
        # keep integrating past the box is what makes the coupling spring
        # term (Ka @ (pc - pr)) grow with how hard the push continues past
        # the wall -- clamping both pc and pr to the same box would zero out
        # that gap and flatten the felt force to a constant.
        pc = pc + Vc * dt  # integrate the decoded slave-side command position

        # ---- contact force (reused mechanism from haptic_teleop_fr3_demo.py) ----
        contact = slave.robot.get_net_contact_forces(unwrapped.CONTACT_LINK_NAMES)[0].sum(axis=0)
        f_e = np.clip(contact.cpu().numpy(), -gains.f_e_limit, gains.f_e_limit)

        # ---- 5. outer admittance (Eq. 15, A_cr = A_sr = identity) ----
        Vr_dot = np.linalg.solve(gains.Ma, f_ch - f_e - gains.Br @ Vr)
        Vr = Vr + Vr_dot * dt
        pr = pr + Vr * dt
        pr, Vr = clamp_with_velocity(pr, Vr, gains.workspace_lo, gains.workspace_hi)

        # ---- 6. slave tracks the admittance reference (inner impedance skipped
        # per request -- IK + Panda's own PD position drive instead) ----
        _, pgs = get_pose(slave_link)
        track_via_ik(slave, slave_joints, slave_idx, slave_pmodel, slave_ee_idx, slave_qmask, R_slave_fixed, pr)

        scene.step()
        return f_e, f_m, f_ch, pgs

    def log_step(t, f_e, f_m, f_ch, pgs):
        log["t"].append(t)
        log["pm"].append(pm.copy())
        log["pc"].append(pc.copy())
        log["pr"].append(pr.copy())
        log["ps"].append(pgs.copy())
        log["fe"].append(f_e.copy())
        log["fm"].append(f_m.copy())
        log["fch"].append(f_ch.copy())

    log_every = max(1, int(round(1.0 / (dt * args.log_hz))))

    if not args.realtime:
        n_steps = int(args.duration / dt)
        for step in range(n_steps):
            t = step * dt
            f_e, f_m, f_ch, pgs = do_step()
            if args.debug and step % 200 == 0:
                print(
                    f"t={t:5.2f} pm={np.round(pm,3)} pc={np.round(pc,3)} pr={np.round(pr,3)} "
                    f"pgs={np.round(pgs,3)} f_e={np.round(f_e,2)} f_m={np.round(f_m,2)}"
                )
            if step % log_every == 0:
                log_step(t, f_e, f_m, f_ch, pgs)
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
    fe, fm, fch = (np.array(log[k]) for k in ["fe", "fm", "fch"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

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

    axes[2].plot(t, np.linalg.norm(fm, axis=1), color="tab:red", label="|f_m| (felt at master)")
    axes[2].plot(t, np.linalg.norm(fch, axis=1), color="tab:purple", alpha=0.6, label="|f_ch| (coupling)")
    axes[2].set_ylabel("N")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Reflected wrench: what the human would feel through the channel")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--human-force", type=float, default=-5.0, help="constant downward test wrench, N")
    parser.add_argument("--log-hz", type=float, default=100.0)
    parser.add_argument("--out", type=str, default="haptic_teleop_bilateral_result.png")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--realtime", action="store_true",
        help="open a live 3D viewer + force display instead of running headless and saving a plot",
    )
    args = parser.parse_args()

    log = run(args)
    if not args.realtime:
        plot_log(log, args.out)
