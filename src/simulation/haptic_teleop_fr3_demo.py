"""Bilateral teleoperation demo: 6-DOF master (xArm6) -> Panda (Franka FR3 stand-in) slave.

This reproduces, in simplified form, the S1 "virtual coupling + outer admittance +
inner geometric impedance control (GIC)" cascade described in the geometric-impedance
teleoperation manuscript the user shared, using ManiSkill/SAPIEN for physics.

WHAT THIS SCRIPT DOES
----------------------
- Loads two independent articulations in one scene:
    * master: xarm6_nogripper (6 joints) -- driven KINEMATICALLY (qpos is scripted,
      not physically simulated) since there is no physical haptic device or human
      operator in this simulation. Its forward kinematics gives the "master command"
      pose g_c(t); only the RELATIVE motion of its end effector from t=0 is mapped
      onto the slave's workspace, so the two arms do not need to share a workspace.
    * slave: panda (7 joints). ManiSkill does not ship a Franka FR3 URDF/asset, only
      the Franka Panda (kinematically near-identical 7-DOF arm), so Panda stands in
      for FR3 here. The slave is driven by Panda's own PD joint-position drive
      (its default gains, the same ones countless ManiSkill tasks use), with a
      per-step inverse-kinematics solve providing the joint target -- see
      simplification 2 below for why this replaced a hand-written torque law.
- Virtual coupling (paper Eq. 13) connects the master's relative command to a 6D
  dynamic admittance reference (g_r, V_r) (paper Eq. 15). The slave's inner loop
  tracks that reference via IK + Panda's own position drive (see below), rather
  than the paper's geometric impedance torque law.
- The slave presses into a static box; the resulting contact force f_e is fed back
  into the admittance equation (this is what "closes the loop" for haptic feedback)
  and is logged as the reflected force signal, since there is no physical haptic
  device to actually render it on.

SIMPLIFICATIONS relative to the manuscript (so nobody mistakes this for a faithful
reproduction of its theorems):
  1. No wave-channel / scattering transform: master and admittance are coupled
     directly (equivalent to the paper's own zero-communication-delay ablation in
     Sec. XI, not the delayed bilateral case of Sec. IV).
  2. The inner controller is NOT a torque-level impedance law at all: an earlier
     Jacobian-transpose + hand-computed gravity-compensation version (using
     SAPIEN's compute_passive_force) was found empirically to let the arm
     drift and slam a joint into its limit even with zero commanded motion and
     the exact negated gravity torque applied -- compute_passive_force matched
     an independent Pinocchio computation exactly, so the discrepancy is
     between that reported torque and what SAPIEN's own physics integration
     actually needs, not a bug in the coupling/impedance math above it. Rather
     than keep chasing that, the inner loop now solves inverse kinematics for
     the admittance reference each step and hands the joint target to Panda's
     own PD position drive, which is known to hold the arm correctly. The
     trade-off: the slave is now stiff at the contact interface itself (not a
     soft torque-level impedance); compliance under contact comes only from
     the admittance reference (above) yielding to the measured contact force,
     not from the inner loop reacting to it directly.
  3. The energy-tank passivity controller (Sec. VI/VII) is implemented only as a
     DIAGNOSTIC energy bookkeeping plot (informational), not as an enforced,
     torque-scaling passivity guarantee. Do not read the "tank energy" plot as a
     certified passivity result -- it is illustrative only.

Usage:
    python haptic_teleop_fr3_demo.py --duration 8.0 --out haptic_teleop_result.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers all envs/agents)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig
from sapien.wrapper.pinocchio_model import PinocchioModel


# --------------------------------------------------------------------------- #
# SE(3) helpers (mirrors the manuscript's Sec. III conventions: twists/wrenches
# ordered translation-first, Ad(g) as in its Eq. 1, geometric error as in Eq. 4-5)
# --------------------------------------------------------------------------- #


def hat(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def log_SO3(Rmat: np.ndarray) -> np.ndarray:
    """Axis-angle (rotation) vector such that R = Exp(log_SO3(R))."""
    return R.from_matrix(Rmat).as_rotvec()


def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return R.from_quat([x, y, z, w]).as_matrix()


def R_to_quat_wxyz(Rmat: np.ndarray) -> np.ndarray:
    x, y, z, w = R.from_matrix(Rmat).as_quat()
    return np.array([w, x, y, z])


def joint_limit_avoidance_grad(q, q_mid, q_half_range, margin=0.5):
    """Null-space joint-centering gradient with a dead zone.

    Zero within `margin` of each joint's range center, ramping up to full
    strength only as a joint actually approaches its limit. A plain
    "always pull toward center" potential (no dead zone) drags a joint away
    from a perfectly fine starting pose whenever that pose isn't exactly
    centered in its range -- which is what made the arm visibly drift right
    after reset even with no commanded motion.
    """
    normalized = (q - q_mid) / q_half_range  # in [-1, 1]
    excess = np.sign(normalized) * np.clip(np.abs(normalized) - margin, 0, None) / (1 - margin)
    return -excess


def pose_inv(Rm: np.ndarray, p: np.ndarray):
    Ri = Rm.T
    return Ri, -Ri @ p


def pose_mul(R1, p1, R2, p2):
    return R1 @ R2, R1 @ p2 + p1


def adjoint(Rm: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Ad_g for g=(R,p), twists ordered (linear, angular) -- manuscript Eq. 1."""
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = Rm
    Ad[:3, 3:] = hat(p) @ Rm
    Ad[3:, 3:] = Rm
    return Ad


def geometric_error_wrench(Rs, ps, Rd, pd, Kp, KR):
    """Manuscript Eq. 4-5: g_de = g_d^{-1} g, P(g,gd), f_G(g,gd).

    Returns (P, fG) with fG = [R_de^T Kp p_de ; vee(KR R_de - R_de^T KR)],
    a body wrench expressed at the "g" (here: slave) frame.
    """
    Rde = Rd.T @ Rs
    pde = Rd.T @ (ps - pd)
    P = 0.5 * pde @ Kp @ pde + np.trace(KR @ (np.eye(3) - Rde))
    f_lin = Rde.T @ Kp @ pde
    f_rot = vee(KR @ Rde - Rde.T @ KR)
    return P, np.concatenate([f_lin, f_rot])


# --------------------------------------------------------------------------- #
# ManiSkill task: two independent articulations + one static contact box
# --------------------------------------------------------------------------- #


@register_env("HapticTeleopDemo-v1", max_episode_steps=100_000)
class HapticTeleopDemoEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("xarm6_nogripper", "panda")]
    MASTER_ROOT_POSE = sapien.Pose(p=[0.0, 0.9, 0.0])
    SLAVE_ROOT_POSE = sapien.Pose(p=[0.0, 0.0, 0.0])
    BOX_HALF_SIZE = [0.12, 0.15, 0.04]
    BOX_POSITION = [0.615, 0.0, 0.04]
    CONTACT_LINK_NAMES = ["panda_leftfinger", "panda_rightfinger"]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", ("xarm6_nogripper", "panda"))
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=500)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.1, 1.0, 0.9], [0.3, 0.2, 0.1])
        return dict()

    def _load_agent(self, options: dict):
        super()._load_agent(options, [self.MASTER_ROOT_POSE, self.SLAVE_ROOT_POSE])

    def _load_scene(self, options: dict):
        self.box = actors.build_box(
            self.scene,
            half_sizes=self.BOX_HALF_SIZE,
            color=[0.6, 0.35, 0.2, 1.0],
            name="contact_box",
            body_type="static",
            initial_pose=sapien.Pose(p=self.BOX_POSITION),
        )
        self.ground = actors.build_box(
            self.scene,
            half_sizes=[2.0, 2.0, 0.02],
            color=[0.8, 0.8, 0.8, 1.0],
            name="ground",
            body_type="static",
            initial_pose=sapien.Pose(p=[0, 0, -0.02]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        master, slave = self.agent.agents
        master.robot.set_qpos(master.keyframes["rest"].qpos)
        slave.robot.set_qpos(slave.keyframes["rest"].qpos)

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: dict):
        return dict()

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


# --------------------------------------------------------------------------- #
# Controller: virtual coupling -> outer admittance -> inner GIC
# --------------------------------------------------------------------------- #


@dataclass
class ControllerGains:
    """Gains for the virtual coupling + outer admittance. There is no inner
    impedance law anymore (see the module docstring): the slave tracks the
    admittance reference via IK + Panda's own PD position drive, so there are
    no Kp/KR/Dtr/null-space gains here -- only the outer cascade above it.
    """

    Ka_p: np.ndarray = field(default_factory=lambda: np.diag([100.0, 100.0, 100.0]))
    Ka_r: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 10.0]))
    Ba: np.ndarray = field(default_factory=lambda: np.diag([25.0] * 3 + [2.5] * 3))
    Ma: np.ndarray = field(default_factory=lambda: np.diag([3.0] * 3 + [0.05] * 3))
    Br: np.ndarray = field(default_factory=lambda: np.diag([25.0] * 3 + [1.5] * 3))
    f_e_limit: float = 15.0


def build_pinocchio(urdf_path: str, robot) -> tuple[PinocchioModel, list[str]]:
    with open(urdf_path, "r") as f:
        urdf_str = f.read()
    pmodel = PinocchioModel(urdf_str, [0.0, 0.0, -9.81])
    joint_order = [j.name for j in robot.active_joints]
    link_order = [l.name for l in robot.get_links()]
    pmodel.set_joint_order(joint_order)
    pmodel.set_link_order(link_order)
    return pmodel, link_order


def master_command_trajectory(t: float):
    """Scripted joint-space motion for the 6-DOF master: rest -> lower -> slide.

    Stands in for "the human moving the master arm" since no physical/haptic
    device is attached in this simulation.
    """
    rest = np.array([0.0, -1.10912404, -0.09713439, 0.0, 1.20606723, 0.0])
    lowered = rest + np.array([0.0, 0.35, 0.15, 0.0, -0.35, 0.0])
    slid = lowered + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    t_approach, t_press, t_slide = 1.5, 3.5, 8.0
    if t <= t_approach:
        s = t / t_approach
        return rest + s * (lowered - rest)
    elif t <= t_press:
        return lowered.copy()
    elif t <= t_slide:
        s = (t - t_press) / (t_slide - t_press)
        return lowered + s * (slid - lowered)
    return slid.copy()


def run(args):
    env: BaseEnv = gym.make(
        "HapticTeleopDemo-v1",
        num_envs=1,
        sim_backend="cpu",
        render_mode=None,
    )
    env.reset(seed=0)
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    dt = 1.0 / unwrapped.sim_freq

    master, slave = unwrapped.agent.agents

    # --- slave: Panda's own PD position drive, at its default gains ---
    arm_joint_names = set(slave.arm_joint_names)
    slave_arm_joints = [j for j in slave.robot.active_joints if j.name in arm_joint_names]
    for j in slave_arm_joints:
        j.set_drive_properties(
            slave.arm_stiffness, slave.arm_damping, force_limit=slave.arm_force_limit
        )
    arm_idx = np.array(
        [i for i, j in enumerate(slave.robot.active_joints) if j.name in arm_joint_names]
    )

    pmodel, slave_link_order = build_pinocchio(slave.urdf_path, slave.robot)
    ee_link_index = slave_link_order.index(slave.ee_link_name)
    ik_qmask = np.zeros(len(slave.robot.active_joints), dtype=np.int32)
    ik_qmask[arm_idx] = 1

    gains = ControllerGains()

    # --- initial frames ---
    master_link = sapien_utils.get_obj_by_name(master.robot.get_links(), master.ee_link_name)
    slave_link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)

    def get_pose(link):
        pose = link.pose
        p = pose.p.cpu().numpy().reshape(3)
        q = pose.q.cpu().numpy().reshape(4)
        return quat_wxyz_to_R(q), p

    Rc0, pc0 = get_pose(master_link)
    Rc0_inv, pc0_inv = pose_inv(Rc0, pc0)

    Rr, pr = get_pose(slave_link)  # admittance reference starts at the slave's pose
    Rr0, pr0 = Rr, pr  # fixed anchor for mapping the master's relative motion
    Vr = np.zeros(6)

    n_steps = int(args.duration / dt)
    log = {k: [] for k in ["t", "pc", "pr", "ps", "fe", "fch", "track_err"]}

    prev_Rc, prev_pc = Rc0, pc0
    v_lin_world = np.zeros(3)  # admittance reference velocity, WORLD frame
    w_world = np.zeros(3)

    for step in range(n_steps):
        t = step * dt

        # ---------------- master: scripted kinematic motion -----------------
        q_master = master_command_trajectory(t)
        master.robot.set_qpos(torch.tensor(q_master, dtype=torch.float32).unsqueeze(0))
        master.robot.set_qvel(torch.zeros_like(master.robot.get_qvel()))

        Rc_world, pc_world = get_pose(master_link)
        # relative master motion since t=0, expressed directly in WORLD-frame
        # deltas (not re-expressed through either arm's own base orientation),
        # so "master moves +z in the world" maps onto "slave target moves +z
        # in the world" regardless of how the two arms are mounted.
        dR_world = Rc_world @ Rc0_inv
        dp_world = pc_world - pc0
        vc_world = (pc_world - prev_pc) / dt
        wc_world = log_SO3(Rc_world @ prev_Rc.T) / dt
        prev_Rc, prev_pc = Rc_world, pc_world

        # map master's relative command onto the slave-side admittance frame,
        # anchored at the admittance's FIXED initial pose (Rr0, pr0).
        Rc = dR_world @ Rr0
        pc = pr0 + dp_world

        # ---------------- virtual coupling + outer admittance -----------------
        # Simplified (WORLD-frame) version of the paper's Eq. 13/15 cascade:
        # rather than transporting the coupling wrench through the full SE(3)
        # adjoint (which requires gc and gr to stay close together or its
        # moment-arm term injects spurious torque -- see note in the module
        # docstring), position and orientation are each driven by their own
        # independent world-frame mass-spring-damper, with the true contact
        # force f_e (already reported in world coordinates) added directly to
        # the translational admittance. This keeps the same qualitative
        # behavior (a compliant reference that follows the master and yields
        # to contact) without the coordinate-frame fragility.
        contact = slave.robot.get_net_contact_forces(unwrapped.CONTACT_LINK_NAMES)[0].sum(axis=0)
        f_e_world = np.clip(contact.cpu().numpy(), -gains.f_e_limit, gains.f_e_limit)

        # Damping acts on velocity RELATIVE TO THE MASTER (e_a_lin/e_a_rot),
        # not on v_lin_world/w_world against the fixed world. Damping against
        # the world is what caused the steady-state tracking lag under
        # constant-velocity master motion: at a moving equilibrium the spring
        # term would otherwise have to permanently balance a nonzero "drag"
        # force, i.e. a nonzero position error, just to hold a constant
        # velocity. Damping the RELATIVE velocity instead means that drag
        # vanishes once the reference is actually keeping up with the master,
        # which is what a full velocity feedforward is meant to achieve. This
        # is a further deliberate deviation from the paper's strict tank
        # passivity accounting (Br is no longer purely dissipative against a
        # fixed frame) made for responsiveness, on top of this script's other
        # documented simplifications.
        e_p = pc - pr
        e_a_lin = vc_world - v_lin_world
        a_lin_world = np.linalg.solve(
            gains.Ma[:3, :3],
            gains.Ka_p @ e_p + (gains.Ba[:3, :3] + gains.Br[:3, :3]) @ e_a_lin + f_e_world,
        )
        v_lin_world = v_lin_world + a_lin_world * dt
        pr = pr + v_lin_world * dt

        e_R = log_SO3(Rc @ Rr.T)
        e_a_rot = wc_world - w_world
        a_rot_world = np.linalg.solve(
            gains.Ma[3:, 3:],
            gains.Ka_r @ e_R + (gains.Ba[3:, 3:] + gains.Br[3:, 3:]) @ e_a_rot,
        )
        w_world = w_world + a_rot_world * dt
        Rr = R.from_rotvec(w_world * dt).as_matrix() @ Rr

        # body twist of the (Rr, pr) frame, for the inner controller below
        Vr = np.concatenate([Rr.T @ v_lin_world, Rr.T @ w_world])
        f_ch = np.concatenate([gains.Ka_p @ e_p + gains.Ba[:3, :3] @ e_a_lin, np.zeros(3)])
        f_e = np.concatenate([f_e_world, np.zeros(3)])

        # ---------------- inner control: IK + Panda's own PD position drive -----
        Rgs, pgs = get_pose(slave_link)
        qpos_full = slave.robot.get_qpos()[0].cpu().numpy()
        target_pose = sapien.Pose(p=pr, q=R_to_quat_wxyz(Rr))
        q_ik, ik_ok, ik_err = pmodel.compute_inverse_kinematics(
            ee_link_index,
            target_pose,
            initial_qpos=qpos_full,
            active_qmask=ik_qmask,
            max_iterations=20,  # warm-started from the previous qpos each step
        )
        q_target = q_ik[arm_idx]
        slave.robot.set_joint_drive_targets(q_target[None, :], joints=slave_arm_joints)

        if args.debug and step % 100 == 0:
            print(
                f"t={t:5.2f} q_arm={np.round(qpos_full[arm_idx], 3)} pgs={np.round(pgs, 3)} "
                f"pr={np.round(pr, 3)} ik_ok={bool(ik_ok)}"
            )

        scene.step()

        if step % max(1, int(round(1.0 / (dt * args.log_hz)))) == 0:
            log["t"].append(t)
            log["pc"].append(pc.copy())
            log["pr"].append(pr.copy())
            log["ps"].append(pgs.copy())
            log["fe"].append(f_e[:3].copy())
            log["fch"].append(f_ch[:3].copy())
            log["track_err"].append(float(np.linalg.norm(pr - pgs)))

    env.close()
    return log


def plot_log(log, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(log["t"])
    pc = np.array(log["pc"])
    pr = np.array(log["pr"])
    ps = np.array(log["ps"])
    fe = np.array(log["fe"])
    fch = np.array(log["fch"])
    track_err = np.array(log["track_err"])

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    labels = ["x", "y", "z"]
    colors = ["tab:blue", "tab:green", "tab:purple"]
    for i, lab in enumerate(labels):
        axes[0].plot(t, pc[:, i], ":", color=colors[i], label=f"master command {lab}", alpha=0.6)
        axes[0].plot(t, pr[:, i], "--", color=colors[i], label=f"admittance ref {lab}", alpha=0.8)
        axes[0].plot(t, ps[:, i], "-", color=colors[i], label=f"slave actual {lab}")
    axes[0].set_ylabel("position (m)")
    axes[0].set_title("Master command -> admittance reference -> slave end-effector")
    axes[0].legend(fontsize=6, ncol=3)

    for i, lab in enumerate(labels):
        axes[1].plot(t, fe[:, i], label=f"f_e {lab}")
    axes[1].set_ylabel("contact force (N)")
    axes[1].set_title("Slave-on-environment contact force f_e")
    axes[1].legend(fontsize=8, ncol=3)

    fch_norm = np.linalg.norm(fch, axis=1)
    axes[2].plot(t, fch_norm, color="tab:red")
    axes[2].set_ylabel("|f_ch| (N)")
    axes[2].set_title("Reflected wrench magnitude at the virtual coupling\n"
                       "(the signal that would be rendered to the human -- plotted, not rendered)")

    axes[3].plot(t, track_err, color="tab:purple")
    axes[3].set_ylabel("|pr - pgs| (m)")
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("Admittance-reference vs. slave tracking error")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--log-hz", type=float, default=100.0)
    parser.add_argument("--out", type=str, default="haptic_teleop_result.png")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log = run(args)
    plot_log(log, args.out)
