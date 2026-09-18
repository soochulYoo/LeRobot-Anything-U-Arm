"""Interactive, real-time version of haptic_teleop_fr3_demo.py.

Drive the 6-DOF master (xArm6, standing in for the real uarm hardware this
repo builds -- see the module docstring in haptic_teleop_fr3_demo.py for why
xArm6/Panda are used as ManiSkill-available stand-ins) with the keyboard,
watch the Panda slave follow it through the virtual coupling + outer
admittance (the slave itself is held by IK + Panda's own PD position drive,
not a hand-written torque law -- see haptic_teleop_fr3_demo.py's docstring
for why) in the 3D viewer, and watch the resulting contact/reflected force
live in a separate matplotlib window.

Controls
--------
TAB          toggle between EE-translation mode and joint mode
ESC          quit

EE-translation mode (moves the master's end effector in world XYZ):
  Z / X      +X / -X
  C / V      -Y / +Y
  B / N      +Z / -Z
(WASD/QE are avoided: SAPIEN's viewer binds those to its own free-fly camera,
so they move the camera instead of the arm.)

Joint mode (moves one master joint at a time):
  1 2 3 4 5 6   joints 1-6, positive direction
  hold SHIFT    reverses the direction of the above

This script cannot be verified visually in a headless environment -- it was
checked for API correctness (env/viewer construction, keyboard state queries)
but its actual on-screen behavior should be confirmed by running it locally.

Usage:
    python haptic_teleop_fr3_interactive.py
"""

from __future__ import annotations

import time

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

import mani_skill.envs  # noqa: F401
from mani_skill.utils import sapien_utils

from haptic_teleop_fr3_demo import (
    ControllerGains,
    HapticTeleopDemoEnv,  # noqa: F401  (registers "HapticTeleopDemo-v1")
    R_to_quat_wxyz,
    build_pinocchio,
    log_SO3,
    quat_wxyz_to_R,
    vee,
)

EE_STEP = 0.4  # m/s, while a translation key is held
JOINT_STEP = 2.0  # rad/s, while a joint key is held

EE_KEYS = {"z": (0, +1), "x": (0, -1), "c": (1, -1), "v": (1, +1), "b": (2, +1), "n": (2, -1)}
JOINT_KEYS = ["1", "2", "3", "4", "5", "6"]  # one per master joint


def main():
    env = gym.make(
        "HapticTeleopDemo-v1",
        num_envs=1,
        sim_backend="cpu",
        render_mode="human",
    )
    env.reset(seed=0)
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    dt = 1.0 / unwrapped.sim_freq
    unwrapped.render_human()
    viewer = unwrapped.viewer

    master, slave = unwrapped.agent.agents

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
    master_pmodel, master_link_order = build_pinocchio(master.urdf_path, master.robot)
    master_ee_index = master_link_order.index(master.ee_link_name)

    gains = ControllerGains()

    master_link = sapien_utils.get_obj_by_name(master.robot.get_links(), master.ee_link_name)
    slave_link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)

    def get_pose(link):
        pose = link.pose
        p = pose.p.cpu().numpy().reshape(3)
        q = pose.q.cpu().numpy().reshape(4)
        return quat_wxyz_to_R(q), p

    Rc0, pc0 = get_pose(master_link)
    Rc0_inv = Rc0.T

    Rr, pr = get_pose(slave_link)
    Rr0, pr0 = Rr, pr

    v_lin_world = np.zeros(3)
    w_world = np.zeros(3)

    q_master = master.robot.get_qpos()[0].cpu().numpy().copy()
    prev_Rc, prev_pc = Rc0, pc0

    # -------- live wrench display (separate matplotlib window) --------
    matplotlib.rcParams["toolbar"] = "none"
    plt.ion()
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(9, 4))
    bar = ax_bar.bar(["x", "y", "z"], [0, 0, 0], color=["tab:blue", "tab:orange", "tab:green"])
    ax_bar.set_ylim(-gains.f_e_limit * 1.2, gains.f_e_limit * 1.2)
    ax_bar.set_title("Contact force f_e (N)")
    hist_len = 300
    fch_hist = np.zeros(hist_len)
    (line,) = ax_line.plot(fch_hist)
    ax_line.set_ylim(0, 20)
    ax_line.set_title("Reflected |f_ch| (N), recent history")
    fig.tight_layout()
    fig.show()

    mode = "ee"  # or "joint"
    tab_was_down = False
    print(__doc__)
    print(f"Starting in '{mode}' mode.")

    step = 0
    fps_t0 = time.time()
    fps_count = 0
    last_time = time.time()
    accumulator = 0.0
    plot_t0 = time.time()
    f_e_world = np.zeros(3)
    f_ch = np.zeros(6)
    MAX_SUBSTEPS = 50  # cap catch-up if a frame stalls, to avoid a "spiral of death"
    try:
        while not viewer.window.should_close:
            # ---------------- fixed-timestep accumulator ----------------
            # Rendering (and on this machine, even physics) can run much
            # slower than sim_freq in wall-clock time. Without this, a key
            # held for "1 real second" might only advance a few ms of
            # simulated time, making motion look nearly frozen. Instead we
            # measure real elapsed time each frame and run as many dt-sized
            # physics/controller steps as needed to catch simulated time up
            # to it, so motion speed is tied to the wall clock, not to how
            # fast rendering happens to be.
            now = time.time()
            accumulator += min(now - last_time, 0.1)
            last_time = now

            # ---------------- keyboard: mode toggle ----------------
            tab_down = viewer.window.key_down("tab")
            if tab_down and not tab_was_down:
                mode = "joint" if mode == "ee" else "ee"
                print(f"[mode] switched to '{mode}'")
            tab_was_down = tab_down
            if viewer.window.key_down("esc"):
                break

            n_substeps = 0
            while accumulator >= dt and n_substeps < MAX_SUBSTEPS:
                accumulator -= dt
                n_substeps += 1

                # ---------------- keyboard: master motion ----------------
                if mode == "ee":
                    v_cmd_world = np.zeros(3)
                    for key, (axis, sign) in EE_KEYS.items():
                        if viewer.window.key_down(key):
                            v_cmd_world[axis] += sign * EE_STEP
                    Rc_now, _ = get_pose(master_link)
                    master_pmodel.compute_forward_kinematics(q_master)
                    J_master = master_pmodel.compute_single_link_local_jacobian(
                        q_master, master_ee_index
                    )
                    v_body = Rc_now.T @ v_cmd_world
                    twist_cmd = np.concatenate([v_body, np.zeros(3)])
                    damping = 1e-3 * np.eye(6)
                    dq = J_master.T @ np.linalg.solve(
                        J_master @ J_master.T + damping, twist_cmd
                    )
                    q_master = q_master + dq * dt
                else:
                    for i, key in enumerate(JOINT_KEYS):
                        if viewer.window.key_down(key):
                            sign = -1.0 if viewer.window.shift else 1.0
                            q_master[i] += sign * JOINT_STEP * dt
                master.robot.set_qpos(torch.tensor(q_master, dtype=torch.float32).unsqueeze(0))
                master.robot.set_qvel(torch.zeros_like(master.robot.get_qvel()))

                # ---------------- master relative motion -> coupling target ----------------
                Rc_world, pc_world = get_pose(master_link)
                dR_world = Rc_world @ Rc0_inv
                dp_world = pc_world - pc0
                vc_world = (pc_world - prev_pc) / dt
                wc_world = log_SO3(Rc_world @ prev_Rc.T) / dt
                prev_Rc, prev_pc = Rc_world, pc_world

                Rc = dR_world @ Rr0
                pc = pr0 + dp_world

                # ---------------- contact force ----------------
                contact = slave.robot.get_net_contact_forces(
                    unwrapped.CONTACT_LINK_NAMES
                )[0].sum(axis=0)
                f_e_world = np.clip(contact.cpu().numpy(), -gains.f_e_limit, gains.f_e_limit)

                # ---------------- outer admittance (world-frame, simplified) ----------------
                # Damping acts on velocity relative to the master (e_a_lin/
                # e_a_rot), not against the fixed world -- this is what
                # removes the steady-state tracking lag under continuous
                # master motion (see haptic_teleop_fr3_demo.py for the full
                # explanation). It's a further deviation from strict tank
                # passivity, traded for responsiveness.
                e_p = pc - pr
                e_a_lin = vc_world - v_lin_world
                a_lin_world = np.linalg.solve(
                    gains.Ma[:3, :3],
                    gains.Ka_p @ e_p
                    + (gains.Ba[:3, :3] + gains.Br[:3, :3]) @ e_a_lin
                    + f_e_world,
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

                Vr = np.concatenate([Rr.T @ v_lin_world, Rr.T @ w_world])
                f_ch = np.concatenate(
                    [gains.Ka_p @ e_p + gains.Ba[:3, :3] @ e_a_lin, np.zeros(3)]
                )

                # ---------------- inner control: IK + Panda's own PD position drive ---
                Rgs, pgs = get_pose(slave_link)
                qpos_full = slave.robot.get_qpos()[0].cpu().numpy()
                target_pose = sapien.Pose(p=pr, q=R_to_quat_wxyz(Rr))
                q_ik, ik_ok, ik_err = pmodel.compute_inverse_kinematics(
                    ee_link_index,
                    target_pose,
                    initial_qpos=qpos_full,
                    active_qmask=ik_qmask,
                    max_iterations=20,
                )
                q_target = q_ik[arm_idx]
                slave.robot.set_joint_drive_targets(q_target[None, :], joints=slave_arm_joints)

                scene.step()
                step += 1
                fps_count += 1

            # ---------------- rendering and live plot: once per outer frame ----------------
            unwrapped.render_human()

            if now - plot_t0 >= 0.1:  # throttle the plot to ~10 Hz regardless of sim rate
                for rect, val in zip(bar, f_e_world):
                    rect.set_height(val)
                fch_hist = np.roll(fch_hist, -1)
                fch_hist[-1] = np.linalg.norm(f_ch[:3])
                line.set_ydata(fch_hist)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plot_t0 = now

            if now - fps_t0 >= 1.0:
                sim_hz = fps_count / (now - fps_t0)
                print(
                    f"[perf] {sim_hz:6.1f} physics steps/s "
                    f"({sim_hz / unwrapped.sim_freq * 100:5.1f}% of real time) "
                    f"q_master={np.round(q_master, 2)}"
                )
                fps_t0 = now
                fps_count = 0
    finally:
        plt.close(fig)
        env.close()


if __name__ == "__main__":
    main()
