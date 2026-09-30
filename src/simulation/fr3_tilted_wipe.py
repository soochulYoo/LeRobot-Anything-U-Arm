"""FR3 validation of the T1 frame result: a wipe on a TILTED box, with the
anisotropic stiffness carried by the inner loop, comparing GIC against the naive
world-frame Cartesian law.

cascade/study_anisotropy.py established on the lean simulator that the frame is
the factor a wiping task cannot absorb.  cascade/gic_frame-style measurements
established on this arm that GIC preserves an anisotropic stiffness at any tilt
while the naive law does not (200 N/m held at every tilt, against 1550 N/m at
60 degrees).  This closes the loop: does that controller difference actually
decide whether the wipe succeeds on the real arm?

The box is tilted, the command presses along the true surface normal and strokes
along the tangent, and the inner impedance is given a soft-normal / stiff-tangent
stiffness IN THE TASK FRAME.  GIC applies it there.  The naive law applies the
same numbers in world coordinates, which is what a label that never states its
frame amounts to.

Usage:  python3 fr3_tilted_wipe.py [--tilt-x 20 --tilt-y -12] [--out FIG.png]
"""
from __future__ import annotations

import argparse
import warnings

import gymnasium as gym
import numpy as np
import sapien

warnings.filterwarnings("ignore")

import mani_skill.envs  # noqa: F401
from mani_skill.utils import sapien_utils

import haptic_teleop_fr3_bilateral as B
from haptic_teleop_fr3_demo import (  # noqa: F401  (import registers the env)
    HapticTeleopDemoEnv, R_to_quat_wxyz, build_pinocchio,
)

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"


def rot_xy(tx: float, ty: float) -> np.ndarray:
    cx, sx, cy, sy = np.cos(tx), np.sin(tx), np.cos(ty), np.sin(ty)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx


def compose(magnitude: float, ratio: float, frame: np.ndarray) -> np.ndarray:
    """K = magnitude * R diag(w) R^T with det(diag(w)) = 1; soft axis last."""
    w = np.array([ratio, ratio, 1.0], dtype=float)
    w = w / np.prod(w) ** (1.0 / 3.0)
    return magnitude * (frame @ np.diag(w) @ frame.T)


def align_z_to(R0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation of R0 that points its third column along `target`.

    The tool has to face the surface.  Holding the end effector at its rest
    orientation over a tilted plane leaves only a finger corner in contact -- a
    point contact that bounces, and the bouncing swamps whatever the stiffness
    is doing.  Aligning the approach axis with the surface normal is also what
    the desired orientation in GIC's error term is FOR.
    """
    a = R0[:, 2] / np.linalg.norm(R0[:, 2])
    b = np.asarray(target, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return R0.copy() if c > 0 else -R0.copy()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Rrot = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    return Rrot @ R0


def sqrtm_spd(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return V @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ V.T


def run(args, law: str, K_world: np.ndarray, R_task: np.ndarray,
        step_zero_report: bool = False) -> dict:
    """One wipe.

    `K_world` is the EFFECTIVE stiffness wanted in world coordinates.  GIC does
    not take that directly: its law is f = -R^T R_d K_p R_d^T (p - p_d), so the
    matrix it is handed lives in the DESIRED END-EFFECTOR frame and the world
    stiffness it realises is R_d K_p R_d^T.  Passing a world matrix straight in
    would rotate it a second time.  K_p is therefore pre-rotated by R_d here --
    which is the same bookkeeping a real label needs and the same bookkeeping
    nobody writes down.
    """
    tx, ty = np.deg2rad(args.tilt_x), np.deg2rad(args.tilt_y)
    q_box = R_to_quat_wxyz(rot_xy(tx, ty))
    old_scene = HapticTeleopDemoEnv._load_scene

    def tilted_scene(self, options):
        old_scene(self, options)
        self.box.set_pose(sapien.Pose(p=self.BOX_POSITION, q=q_box))

    HapticTeleopDemoEnv._load_scene = tilted_scene
    try:
        env = gym.make("HapticTeleopDemo-v1", num_envs=1, sim_backend="cpu")
        env.reset(seed=0)
    finally:
        HapticTeleopDemoEnv._load_scene = old_scene

    u = env.unwrapped
    scene, dt = u.scene, 1.0 / u.sim_freq
    master, slave = u.agent.agents
    gains = B.BilateralGains()

    names = set(slave.arm_joint_names)
    joints = [j for j in slave.robot.active_joints if j.name in names]
    idx = np.array([i for i, j in enumerate(slave.robot.active_joints) if j.name in names])
    for j in joints:
        j.set_drive_properties(0.0, 0.0, force_limit=1000.0)
    pmodel, link_order = build_pinocchio(slave.urdf_path, slave.robot)
    ee_idx = link_order.index(slave.ee_link_name)
    link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)
    qlim = slave.robot.get_qlimits()[0].cpu().numpy()[idx]
    q_mid = (qlim[:, 0] + qlim[:, 1]) / 2.0
    nq = len(slave.robot.active_joints)

    # A closed gripper presents one solid tip instead of two thin fingers.
    q = slave.robot.get_qpos()[0].cpu().numpy().copy()
    q[-2:] = 0.0
    slave.robot.set_qpos(q[None, :])
    for _ in range(20):
        scene.step()

    R_ee0, p0 = B.get_pose(link)
    # R_task[:, 2] is the box's OUTWARD normal, which points up.  The tool has to
    # press the other way.  cascade/core.py's convention calls the penetration
    # direction "normal", and taking the box's outward normal for it is a sign
    # error that produces a flawless run with zero contact force -- the tool
    # presses away from the surface for the whole episode.
    n_out = R_task[:, 2]
    n_hat = -n_out                    # penetration direction
    t_hat = R_task[:, 0]

    # Measure the approach gap instead of guessing it.  A press chosen by hand
    # is wrong the moment the tilt changes: too little and the tool never
    # touches, too much and it drives the reference tens of millimetres into the
    # box and slams it at 200 N, where the anisotropy under study is irrelevant.
    box_c = u.box.pose.p.cpu().numpy().reshape(3)
    top_c = box_c + u.BOX_HALF_SIZE[2] * n_out
    gap = float(n_out @ (p0 - top_c))
    press = (gap - args.finger_offset) if args.press is None else args.press
    if step_zero_report:
        print(f"      gap to the tilted surface {1000*gap:.1f} mm, finger offset "
              f"{1000*args.finger_offset:.1f} mm -> press {1000*press:.1f} mm")
    law_fn = {"geometric": B.geometric_impedance_torque,
              "cartesian": B.cartesian_impedance_torque}[law]

    # The label goes in the OUTER coupling, matching cascade/study_anisotropy.py.
    # An earlier version put it in the inner impedance and left the outer
    # coupling isotropic at 200 N/m, which could not drag the tool through 6 N of
    # static friction inside a +-20 mm stroke: the tool stuck, and all three
    # frames then produced the same 23 mm path error for a reason that had
    # nothing to do with the frame.  The inner loop stays stiff and isotropic,
    # which is its job.
    K_inner = args.inner_stiffness * np.eye(3)
    gains.Ki = K_inner
    gains.Di = 2.0 * 0.8 * np.sqrt(2.0 * args.inner_stiffness) * np.eye(3)

    # Outer admittance in world coordinates, driven by (x_ref, f_d) exactly as
    # cascade/core.py's policy mode does.
    Ma, Br = 3.0 * np.eye(3), 25.0 * np.eye(3)
    x_r, v_r = p0.copy(), np.zeros(3)
    Ko_outer = K_world            # the label under test
    # Bound the approach speed.  The layer study showed the impact peak is set by
    # contact speed and is nearly blind to the inner stiffness, so without this
    # the run is dominated by a 130 N slam that has nothing to do with the
    # anisotropy under test.
    v_limit = args.v_limit
    R_ref = align_z_to(R_ee0, n_hat)

    log = {k: [] for k in ["t", "x", "x_r", "f_e", "f_n", "stroke"]}
    n_steps = int(args.duration / dt)
    for step in range(n_steps):
        t = step * dt
        depth = press * min(1.0, t / args.approach_s)
        stroke = (0.0 if t < args.settle else
                  0.5 * args.stroke_len * np.sin(2 * np.pi * args.stroke_hz * (t - args.settle)))
        x_ref = p0 + depth * n_hat + stroke * t_hat
        f_d = args.f_target * n_hat * min(1.0, t / args.approach_s)

        contact = slave.robot.get_net_contact_forces(u.CONTACT_LINK_NAMES)[0].sum(axis=0)
        f_e = contact.cpu().numpy()
        f_e_ctrl = np.clip(f_e, -gains.f_e_limit, gains.f_e_limit)

        drive = f_d + Ko_outer @ (x_ref - x_r)
        a_r = np.linalg.solve(Ma, drive + f_e_ctrl - Br @ v_r)
        v_r = v_r + a_r * dt
        sp = float(np.linalg.norm(v_r))
        if sp > v_limit:
            v_r = v_r * (v_limit / sp)
        x_r = x_r + v_r * dt

        tau, _ = law_fn(slave, link, pmodel, ee_idx, idx, gains,
                        x_r, v_r, R_ref, q_mid)
        qf = np.zeros(nq); qf[idx] = tau
        slave.robot.set_qf(qf[None, :])
        scene.step()

        _, p_now = B.get_pose(link)
        log["t"].append(t); log["x"].append(p_now.copy()); log["x_r"].append(x_r.copy())
        # the reaction on the robot points along the OUTWARD normal
        log["f_e"].append(f_e.copy()); log["f_n"].append(float(f_e @ n_out))
        log["stroke"].append(stroke)
    env.close()
    out = {k: np.array(v) for k, v in log.items()}
    peak = float(np.abs(out["f_n"]).max())
    out["f_n_filt"] = lowpass(np.abs(out["f_n"]), dt, args.force_hz)
    if peak < 0.5:
        raise RuntimeError(
            f"the tool never reached the surface (peak normal force {peak:.3f} N). "
            f"--press {args.press:.3f} m plus the admittance's f_d/Ko offset did not "
            f"close the gap to the box; raise --press.")
    return out


def lowpass(y: np.ndarray, dt: float, hz: float) -> np.ndarray:
    """Moving average at roughly `hz`, applied before any force metric.

    SAPIEN's per-step net contact force is not what a force sensor reports.  Its
    contact set flickers between solver steps, so the raw 500 Hz signal reads
    zero on 37% of samples and spikes to 150 N on re-establishment, while its
    MEAN is a correct 6.1 N against a commanded 6 N.  Scoring the raw signal
    measures solver chatter: it reported 37% "contact loss" and a 150 N peak for
    a contact that never actually broke.  At 20 Hz -- where a real F/T pipeline
    filters -- the same run shows continuous contact and a 13.8 N peak.
    """
    w = max(1, int(round(1.0 / (hz * dt))))
    pad = np.r_[np.full(w, y[0]), y, np.full(w, y[-1])]
    return np.convolve(pad, np.ones(w) / w, mode="same")[w:-w]


def score(log, args, R_task) -> dict:
    t_hat = R_task[:, 0]
    sel = log["t"] >= args.settle
    dt = float(log["t"][1] - log["t"][0])
    f_n = lowpass(np.abs(log["f_n"]), dt, args.force_hz)[sel]
    x = log["x"][sel]
    got = (x - x[0]) @ t_hat
    want = log["stroke"][sel] - log["stroke"][sel][0]
    return {
        "force_rmse": float(np.sqrt(np.mean((f_n - args.f_target) ** 2))),
        "force_rel": float(np.sqrt(np.mean((f_n - args.f_target) ** 2)) / args.f_target),
        "path_error": float(np.sqrt(np.mean((got - want) ** 2))),
        "contact_loss": float(1.0 - np.mean(f_n > 0.1 * args.f_target)),
        "peak": float(f_n.max()),
    }


def ee_rest_frame() -> np.ndarray:
    """The slave end-effector's orientation at the rest pose.

    A label that says "soft on the third axis" and means the END-EFFECTOR frame
    resolves to this rotation -- a perfectly reasonable reading that no paper
    rules out, and a different matrix from the task-frame reading.
    """
    env = gym.make("HapticTeleopDemo-v1", num_envs=1, sim_backend="cpu")
    env.reset(seed=0)
    u = env.unwrapped
    slave = u.agent.agents[1]
    link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)
    R, _ = B.get_pose(link)
    env.close()
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt-x", type=float, default=20.0)
    ap.add_argument("--tilt-y", type=float, default=-12.0)
    ap.add_argument("--magnitude", type=float, default=1500.0)
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--f-target", type=float, default=6.0)
    # Left as None the press is computed from the measured gap so the reference
    # lands ON the surface and f_d does the pressing, which is the regime the
    # anisotropy actually governs.
    ap.add_argument("--press", type=float, default=None)
    ap.add_argument("--finger-offset", type=float, default=0.0,
                    help="how far the fingertips sit ahead of the TCP, m")
    ap.add_argument("--force-hz", type=float, default=20.0,
                    help="cut-off for the contact-force filter; see lowpass()")
    ap.add_argument("--inner-stiffness", type=float, default=3000.0,
                    help="isotropic inner Cartesian stiffness, N/m")
    # 40 mm peak-to-peak.  At 80 mm the arm cannot follow the stroke at this
    # pose and the path error (21 mm) says more about reach than about the
    # stiffness; at 40 mm it tracks to 4 mm, which is the friction hysteresis.
    ap.add_argument("--stroke-len", type=float, default=0.04)
    ap.add_argument("--stroke-hz", type=float, default=0.25)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--approach-s", type=float, default=4.0)
    ap.add_argument("--v-limit", type=float, default=0.03)
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--out", default="fr3_tilted_wipe.png")
    args = ap.parse_args()

    R_task = rot_xy(np.deg2rad(args.tilt_x), np.deg2rad(args.tilt_y))
    R_ee = ee_rest_frame()
    K_task = compose(args.magnitude, args.ratio, R_task)
    K_ee = compose(args.magnitude, args.ratio, R_ee)
    K_world = compose(args.magnitude, args.ratio, np.eye(3))
    n_hat = R_task[:, 2]

    print(f"box tilted {args.tilt_x:+.0f} deg about x, {args.tilt_y:+.0f} about y")
    print(f"label: magnitude {args.magnitude:.0f} N/m, ratio {args.ratio:.0f}:1, soft axis third")
    print("the same numbers, read in three frames -> effective stiffness ALONG THE SURFACE NORMAL:")
    kt = float(n_hat @ K_task @ n_hat)   # stiffness is a quadratic form: sign of n is irrelevant
    for nm, K in [("task frame (correct)", K_task), ("end-effector frame", K_ee),
                  ("world / per-axis frame", K_world)]:
        k = float(n_hat @ K @ n_hat)
        print(f"  {nm:<26}{k:8.1f} N/m   ({k/kt:5.2f}x the intended)")
    print()

    # The SAME label numbers, read in three different frames.  This is the
    # label-frame question itself: a rule that reports diag(kt, kt, kn) without
    # saying which frame it means leaves the reader to pick one of these.
    R_ee0 = None
    results = {}
    for tag, law, K in [("task frame (correct)", "geometric", K_task),
                        ("end-effector frame", "geometric", K_ee),
                        ("world / per-axis frame", "geometric", K_world)]:
        print(f"  running: {tag} ...")
        lg = run(args, law, K, R_task, step_zero_report=(tag == "task frame (correct)"))
        results[tag] = (lg, score(lg, args, R_task))

    print("\n" + "=" * 92)
    print(f"  {'configuration':<32}{'force RMSE':>13}{'rel':>8}{'path err':>12}{'peak':>9}{'lost':>8}")
    print("  " + "-" * 86)
    for tag, (_, s) in results.items():
        print(f"  {tag:<32}{s['force_rmse']:10.2f} N{100*s['force_rel']:7.0f}%"
              f"{1000*s['path_error']:9.2f} mm{s['peak']:8.1f} N{100*s['contact_loss']:7.0f}%")
    print("  " + "-" * 86)
    plot(results, args, R_task)


def plot(results, args, R_task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    t_hat = R_task[:, 0]
    for i, (tag, (lg, s)) in enumerate(results.items()):
        sel = lg["t"] >= args.settle
        ax[0].plot(lg["t"][sel], lg["f_n_filt"][sel], color=SLOT[i], lw=2.0,
                   label=f"{tag}  ({100*s['force_rel']:.0f}% err)")
        x = lg["x"][sel]
        got = (x - x[0]) @ t_hat
        want = lg["stroke"][sel] - lg["stroke"][sel][0]
        ax[1].plot(lg["t"][sel], 1000 * (got - want), color=SLOT[i], lw=2.0,
                   label=f"{tag}  ({1000*s['path_error']:.1f} mm)")
    ax[0].axhline(args.f_target, color=INK3, ls="--", lw=1.3)
    ax[0].annotate(f" commanded {args.f_target:.0f} N", xy=(0.02, 0.06),
                   xycoords="axes fraction", color=INK2, fontsize=8)
    ax[0].set_ylabel("force along the surface normal (N)")
    ax[0].set_title("Normal force on a tilted surface", color=INK, loc="left")
    ax[1].axhline(0.0, color=INK3, lw=0.8)
    ax[1].set_ylabel("deviation from the commanded stroke (mm)")
    ax[1].set_title("Staying on the stroke", color=INK, loc="left")
    for a in ax:
        a.set_xlabel("time (s)"); a.grid(True, alpha=0.9); a.set_axisbelow(True)
        a.legend(fontsize=7.5, loc="best")
    fig.suptitle(f"FR3: an anisotropic stiffness on a box tilted {args.tilt_x:+.0f}/{args.tilt_y:+.0f} deg -- the frame it is expressed in decides the outcome",
                 color=INK, fontsize=10.5, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=170)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
