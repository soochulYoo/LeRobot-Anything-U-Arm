"""Compare slave arms of different DOF on the SAME cascade.

The point is the inner loop: a 7-DOF arm is redundant and has a null space to
spend on posture, a 6-DOF arm has none, so any difference in how well the inner
impedance holds the admittance reference is attributable to redundancy.

The previous version of this comparison was degenerate: CONTACT_LINK_NAMES is
hard-coded to the Panda's fingers on the env class, so a non-Panda slave
reported zero contact force and never actually engaged the task.  Contact links
are selected per robot here.

Usage:  python3 dof_compare.py [--duration 10] [--out dof_compare.png]
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import sapien

import haptic_teleop_fr3_bilateral as B
from haptic_teleop_fr3_demo import (  # noqa: F401  (import registers the env)
    HapticTeleopDemoEnv, R_to_quat_wxyz, build_pinocchio,
)

# The task is a pure -z push onto a box fixed at x = 0.615, which sits directly
# under the PANDA's rest pose.  The xarm6's rest pose puts its end effector at
# x = 0.193, so on the original comparison it pressed down into empty air for the
# whole run -- it can reach 0.753 m horizontally, it was simply never sent there.
# Both arms are therefore started from a common task pose, or the two curves are
# not measuring the same thing.
COMMON_START_P = np.array([0.615, 0.0, 0.17])

SLAVES = {
    "panda": dict(dof=7, contact=["panda_leftfinger", "panda_rightfinger"]),
    "xarm6_nogripper": dict(dof=6, contact=["link6"]),
}
SLOT = ["#2a78d6", "#eb6834"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"


def run_slave(uid: str, args) -> dict:
    """Run the cascade with `uid` as the slave, patching the env's robot pair,
    contact links and slave start pose for the duration of the run."""
    old_init = HapticTeleopDemoEnv.__init__
    old_episode = HapticTeleopDemoEnv._initialize_episode
    old_contact = HapticTeleopDemoEnv.CONTACT_LINK_NAMES
    HapticTeleopDemoEnv.CONTACT_LINK_NAMES = SLAVES[uid]["contact"]

    def patched_init(self, *a, **kw):
        kw["robot_uids"] = ("xarm6_nogripper", uid)
        super(HapticTeleopDemoEnv, self).__init__(*a, **kw)

    def patched_episode(self, env_idx, options):
        master, slave = self.agent.agents
        master.robot.set_qpos(master.keyframes["rest"].qpos)
        slave.robot.set_qpos(slave.keyframes["rest"].qpos)
        # Move the slave to the common task pose, keeping its own rest
        # orientation: only the position has to agree for a -z push.
        pmodel, link_order = build_pinocchio(slave.urdf_path, slave.robot)
        ee_idx = link_order.index(slave.ee_link_name)
        link = [l for l in slave.robot.get_links() if l.name == slave.ee_link_name][0]
        R_rest, _ = B.get_pose(link)
        names = set(slave.arm_joint_names)
        qmask = np.array([1 if j.name in names else 0
                          for j in slave.robot.active_joints], dtype=np.int32)
        q0 = slave.robot.get_qpos()[0].cpu().numpy()
        q_ik, ok, err = pmodel.compute_inverse_kinematics(
            ee_idx, sapien.Pose(p=COMMON_START_P, q=R_to_quat_wxyz(R_rest)),
            initial_qpos=q0, active_qmask=qmask, max_iterations=200,
        )
        if not ok:
            print(f"  [warn] {uid}: IK to the common start pose did not converge "
                  f"(residual {np.linalg.norm(err):.4f}); using the best solution found")
        slave.robot.set_qpos(q_ik[None, :])

    HapticTeleopDemoEnv.__init__ = patched_init
    HapticTeleopDemoEnv._initialize_episode = patched_episode
    try:
        log = B.run(argparse.Namespace(**vars(args)))
    finally:
        HapticTeleopDemoEnv.__init__ = old_init
        HapticTeleopDemoEnv._initialize_episode = old_episode
        HapticTeleopDemoEnv.CONTACT_LINK_NAMES = old_contact
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--human-reach", type=float, default=-0.16)
    ap.add_argument("--Kh", type=float, default=400.0)
    ap.add_argument("--Bh", type=float, default=20.0)
    ap.add_argument("--channel-mode", choices=["wave", "direct"], default="direct")
    ap.add_argument("--Tf", type=float, default=0.0)
    ap.add_argument("--Tb", type=float, default=0.0)
    ap.add_argument("--log-hz", type=float, default=200.0)
    ap.add_argument("--Ki", type=float, default=2000.0)
    ap.add_argument("--Di", type=float, default=120.0)
    ap.add_argument("--inner", choices=["ik", "cartesian", "geometric"], default="geometric")
    ap.add_argument("--null-gain", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="dof_compare.png")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--realtime", action="store_false")
    args = ap.parse_args()
    args.realtime = False

    logs = {}
    for uid in SLAVES:
        print(f"\n=== slave = {uid} ({SLAVES[uid]['dof']} DOF), contact links "
              f"{SLAVES[uid]['contact']} ===")
        logs[uid] = run_slave(uid, args)

    print("\n" + "=" * 78)
    print(f"  {'slave':<20}{'DOF':>5}{'|pr-ps| (mm)':>16}{'steady f_e (N)':>17}{'reached box?':>15}")
    print("  " + "-" * 70)
    for uid, log in logs.items():
        t = np.array(log["t"]); tail = t >= t[-1] - 1.5
        pr = np.array(log["pr"]); ps = np.array(log["ps"])
        err = np.linalg.norm(pr - ps, axis=1)[tail].mean()
        fe = np.array([np.linalg.norm(v) for v in log["fe"]])[tail].mean()
        print(f"  {uid:<20}{SLAVES[uid]['dof']:>5}{1000*err:>15.2f}{fe:>16.2f}"
              f"{'yes' if fe > 0.5 else 'NO':>15}")
    plot(logs, args.out)


def plot(logs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    for i, (uid, log) in enumerate(logs.items()):
        t = np.array(log["t"])
        pr, ps = np.array(log["pr"]), np.array(log["ps"])
        fe = np.array([np.linalg.norm(v) for v in log["fe"]])
        lbl = f"{uid} ({SLAVES[uid]['dof']} DOF)"
        ax[0].plot(t, 1000 * np.linalg.norm(pr - ps, axis=1), color=SLOT[i], lw=2.0, label=lbl)
        ax[1].plot(t, ps[:, 2], color=SLOT[i], lw=2.0, label=f"{lbl} actual z")
        if i == 0:
            ax[1].plot(t, pr[:, 2], color=INK3, lw=1.6, ls="--", label="admittance ref z")
        ax[2].plot(t, fe, color=SLOT[i], lw=2.0, label=lbl)
    ax[0].set_ylabel("|$p_r - p_s$|  (mm)")
    ax[0].set_title("Inner-loop tracking error: how far the arm lags the admittance reference",
                    color=INK, loc="left")
    ax[1].set_ylabel("z (m)"); ax[1].set_title("z tracking", color=INK, loc="left")
    ax[2].set_ylabel("|$f_e$|  (N)"); ax[2].set_xlabel("time (s)")
    ax[2].set_title("Contact force (zero here means the arm never reached the box)",
                    color=INK, loc="left")
    for a in ax:
        a.grid(True, alpha=0.9); a.set_axisbelow(True); a.legend(fontsize=8, loc="best")
    fig.suptitle("Slave DOF comparison on an identical cascade (geometric inner impedance)",
                 color=INK, fontsize=11, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_path, dpi=150)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
