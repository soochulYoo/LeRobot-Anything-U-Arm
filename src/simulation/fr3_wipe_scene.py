"""A scene that can actually be wiped: a large tilted plate and a spherical pad
welded to the hand.

The earlier FR3 wipe (fr3_tilted_wipe.py) confirmed the direction of the frame
effect but only by 1.3x, because the contact was two thin Panda fingers grazing
a 240x300 mm box.  Pressing worked perfectly (100% contact, 6.1 N peak); the
moment any tangential motion started the contact collapsed to 60% and spiked to
150 N, and box friction made no difference at all -- the failure was geometric,
not frictional.

So the tool and the surface are replaced here:

  a SPHERE welded rigidly to panda_hand -- a sphere on a plane is the canonical
  sliding contact, with no edge or corner to catch, and welding removes the
  grasp slip that moved the peg 44 mm in the insertion study;
  a LARGE plate, wide enough for the whole stroke plus margin, so the pad never
  approaches an edge.

Usage:  python3 fr3_wipe_scene.py [--check] [--frames]
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
from mani_skill.utils.building import actors

import haptic_teleop_fr3_bilateral as B
from haptic_teleop_fr3_demo import HapticTeleopDemoEnv, R_to_quat_wxyz, build_pinocchio
from fr3_tilted_wipe import align_z_to, compose, lowpass, rot_xy, sqrtm_spd

PLATE_HALF = [0.16, 0.16, 0.015]
PLATE_POS = [0.60, 0.0, 0.05]


def build_scene(tilt_x: float, tilt_y: float, pad_radius: float,
                render_mode: str | None = None, cam: dict | None = None,
                n_bumps: int = 0, bump_radius: float = 0.010,
                bump_proud: float = 0.0015, bump_seed: int = 0):
    """Env with the contact box swapped for a large tilted plate.

    `render_mode="rgb_array"` also installs a render camera, which the demo env
    does not provide: its _default_human_render_camera_configs computes a pose
    and then returns an empty dict, so env.render() comes back with nothing.
    """
    R_task = rot_xy(tilt_x, tilt_y)
    q_plate = R_to_quat_wxyz(R_task)
    old_scene = HapticTeleopDemoEnv._load_scene

    def patched(self, options):
        old_scene(self, options)
        # remove the original small box from the contact path by dropping it far
        # below the workspace; rebuilding the whole scene loader would duplicate
        # a lot of the demo env for no benefit
        self.box.set_pose(sapien.Pose(p=[0.0, 0.0, -5.0]))
        self.plate = actors.build_box(
            self.scene, half_sizes=PLATE_HALF, color=[0.55, 0.55, 0.6, 1.0],
            name="wipe_plate", body_type="static",
            initial_pose=sapien.Pose(p=PLATE_POS, q=q_plate),
        )
        # Bumps sunk into the plate so only `bump_proud` stands above it.  A
        # perfectly flat plate is a disturbance-free surface, and the inner
        # layer's whole job is the fast disturbances the outer layer cannot
        # reach -- without something to ride over, a cascade and a stiff
        # position servo look identical.
        self.bumps = []
        if n_bumps > 0:
            rng = np.random.default_rng(bump_seed)
            n_out_l, t1, t2 = R_task[:, 2], R_task[:, 0], R_task[:, 1]
            top = np.array(PLATE_POS) + PLATE_HALF[2] * n_out_l
            for i in range(n_bumps):
                a = rng.uniform(0.020, 0.105)
                th = rng.uniform(0, 2 * np.pi)
                c = top + a * (np.cos(th) * t1 + np.sin(th) * t2) \
                    - (bump_radius - bump_proud) * n_out_l
                self.bumps.append(actors.build_sphere(
                    self.scene, radius=bump_radius, color=[0.45, 0.45, 0.5, 1.0],
                    name=f"bump_{i}", body_type="static",
                    initial_pose=sapien.Pose(p=c)))

    old_cam = HapticTeleopDemoEnv._default_human_render_camera_configs
    if render_mode is not None:
        from mani_skill.sensors.camera import CameraConfig
        c = cam or {}
        eye = c.get("eye", [1.15, -0.85, 0.75])
        at = c.get("at", [0.60, 0.0, 0.10])
        w, h = c.get("size", (960, 720))

        @property
        def cam_cfg(self):
            return CameraConfig("render_camera", sapien_utils.look_at(eye, at),
                                w, h, 1.0, 0.01, 100)

        HapticTeleopDemoEnv._default_human_render_camera_configs = cam_cfg

    HapticTeleopDemoEnv._load_scene = patched
    try:
        kw = dict(num_envs=1, sim_backend="cpu")
        if render_mode is not None:
            kw["render_mode"] = render_mode
        env = gym.make("HapticTeleopDemo-v1", **kw)
        env.reset(seed=0)
    finally:
        HapticTeleopDemoEnv._load_scene = old_scene
        HapticTeleopDemoEnv._default_human_render_camera_configs = old_cam
    return env, R_task


def weld_pad(u, radius: float, ahead: float):
    """Rigidly attach a sphere to panda_hand, `ahead` metres along the hand's
    approach axis so it, not the fingers, is what touches."""
    slave = u.agent.agents[1]
    hand = sapien_utils.get_obj_by_name(slave.robot.get_links(), "panda_hand")
    tcp = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)
    R_h = tcp.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    p_tcp = tcp.pose.p[0].cpu().numpy()
    pad_p = p_tcp + ahead * R_h[:, 2]

    builder = u.scene.create_actor_builder()
    builder.add_sphere_collision(radius=radius, density=500.0)
    builder.add_sphere_visual(radius=radius,
                              material=sapien.render.RenderMaterial(base_color=[0.9, 0.3, 0.2, 1]))
    builder.initial_pose = sapien.Pose(p=pad_p, q=R_to_quat_wxyz(R_h))
    pad = builder.build(name="wipe_pad")

    # A fixed drive with every limit locked at zero is a weld.  Gripping the pad
    # instead would reintroduce the slip that moved the peg 44 mm.
    hand_pose = hand.pose
    R_hand = hand_pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    rel = R_hand.T @ (pad_p - hand_pose.p[0].cpu().numpy())
    drive = u.scene.create_drive(hand, sapien.Pose(p=rel), pad, sapien.Pose())
    # ManiSkill's Drive wrapper exposes only the linear limits, so the ANGULAR
    # ones are set on the underlying PhysxDriveComponent.  Without them the pad
    # is free to spin and tumble on contact.
    for comp in getattr(drive, "_objs", [drive]):
        for ax in ("x", "y", "z"):
            getattr(comp, f"set_limit_{ax}")(0.0, 0.0)
        comp.set_limit_cone(0.0, 0.0)
        comp.set_limit_twist(0.0, 0.0)
    return pad


def tangential(t: float, args, t1: np.ndarray, t2: np.ndarray):
    """Where the tool should be in the surface plane, and a scalar for logging.

    "stroke" is a back-and-forth line; "spiral" winds outward from r0 to r_max,
    which covers area instead of retracing one line and reads far better on
    video.  The spiral's rim speed is r_max * 2 pi turns / path_s, so the outer
    layer's v_limit has to sit above it -- an earlier run blamed a 24 mm path
    error on the arm's reach when it was the clamp binding at 0.03 m/s.
    """
    if t < args.settle:
        return np.zeros(3), 0.0
    if getattr(args, "path", "stroke") == "spiral":
        u = min(1.0, (t - args.settle) / max(args.path_s, 1e-9))
        r = args.r0 + (args.r_max - args.r0) * u
        th = 2.0 * np.pi * args.turns * u
        return r * (np.cos(th) * t1 + np.sin(th) * t2), r
    sc = 0.5 * args.stroke_len * np.sin(2 * np.pi * args.stroke_hz * (t - args.settle))
    return sc * t1, sc


def rollout(u, pad, R_task, K_world, args, report=False) -> dict:
    """One wipe with outer coupling `K_world`, inner GIC isotropic and stiff."""
    slave = u.agent.agents[1]
    robot = slave.robot
    dt = 1.0 / u.sim_freq
    names = set(slave.arm_joint_names)
    arm = [j for j in robot.active_joints if j.name in names]
    idx = np.array([i for i, j in enumerate(robot.active_joints) if j.name in names])
    for j in arm:
        j.set_drive_properties(0.0, 0.0, force_limit=1000.0)
    pmodel, order = build_pinocchio(slave.urdf_path, robot)
    ee_idx = order.index(slave.ee_link_name)
    link = sapien_utils.get_obj_by_name(robot.get_links(), slave.ee_link_name)
    qlim = robot.get_qlimits()[0].cpu().numpy()[idx]
    q_mid = (qlim[:, 0] + qlim[:, 1]) / 2.0
    nq = len(robot.active_joints)

    n_out = R_task[:, 2]
    n_hat = -n_out                       # penetration direction
    t_hat, t2_hat = R_task[:, 0], R_task[:, 1]
    R_ee0, p0 = B.get_pose(link)
    R_ref = align_z_to(R_ee0, n_hat)

    plate_top = np.array(PLATE_POS) + PLATE_HALF[2] * n_out
    gap = float(n_out @ (p0 - plate_top))
    # the contact point sits pad_ahead + pad_radius beyond the TCP once the tool
    # faces the surface, so that is what the commanded press must not include
    # The commanded surface height is wrong for EVERY controller, not only the
    # open-loop one: the comparison is about which of them can absorb that, so
    # they must all be given the same wrong belief.
    press = gap - (args.pad_ahead + args.pad_radius) + getattr(args, "height_error", 0.0)
    if getattr(args, "outer", "admittance") == "rigid":
        # With no force feedback the command has to decide how deep to press.
        # The fair choice is the penetration that yields f_target on a surface
        # believed to be exactly where it is: f / series(Ki, ke).  Leaving it at
        # the surface makes the tool merely kiss it (1% contact) and the
        # comparison measures the missing offset instead of the missing layer.
        k_series = 1.0 / (1.0 / args.inner_stiffness + 1.0 / 5000.0)
        press += args.f_target / k_series
    if report:
        print(f"      gap {1000*gap:.1f} mm, tool reach {1000*(args.pad_ahead+args.pad_radius):.1f} mm"
              f" -> press {1000*press:.1f} mm")

    gains = B.BilateralGains()
    gains.Ki = args.inner_stiffness * np.eye(3)
    gains.Di = 2.0 * 0.8 * np.sqrt(2.0 * args.inner_stiffness) * np.eye(3)
    gains.Kr = args.rot_stiffness
    gains.Dr = 2.0 * 0.8 * np.sqrt(args.wrist_inertia * args.rot_stiffness)

    Ma, Br = 3.0 * np.eye(3), 25.0 * np.eye(3)
    x_r, v_r = p0.copy(), np.zeros(3)
    log = {k: [] for k in ["t", "f_n", "x", "stroke"]}
    for step in range(int(args.duration / dt)):
        t = step * dt
        depth = press * min(1.0, t / args.approach_s)
        tang, stroke = tangential(t, args, t_hat, t2_hat)
        x_ref = p0 + depth * n_hat + tang
        f_d = args.f_target * n_hat * min(1.0, t / args.approach_s)

        f_e = u.scene.get_pairwise_contact_forces(pad, u.plate)[0].cpu().numpy()
        if getattr(args, "outer", "admittance") == "rigid":
            # No outer layer: the command goes straight to the inner impedance,
            # so nothing responds to the measured force at all.
            v_r = (x_ref - x_r) / dt
            x_r = x_ref.copy()
        else:
            drive = f_d + K_world @ (x_ref - x_r)
            a_r = np.linalg.solve(Ma, drive + np.clip(f_e, -args.f_limit, args.f_limit) - Br @ v_r)
            v_r = v_r + a_r * dt
            sp = float(np.linalg.norm(v_r))
            if sp > args.v_limit:
                v_r *= args.v_limit / sp
            x_r = x_r + v_r * dt

        tau, _ = B.geometric_impedance_torque(slave, link, pmodel, ee_idx, idx, gains,
                                              x_r, v_r, R_ref, q_mid)
        qf = np.zeros(nq); qf[idx] = tau
        robot.set_qf(qf[None, :])
        u.scene.step()

        log["t"].append(t)
        log["f_n"].append(float(f_e @ n_out))
        log["x"].append(B.get_pose(link)[1].copy())
        log["stroke"].append(stroke)
    out = {k: np.array(v) for k, v in log.items()}
    out["f_filt"] = lowpass(np.abs(out["f_n"]), dt, args.force_hz)
    sel = out["t"] >= args.settle
    f = out["f_filt"][sel]
    x = out["x"][sel]
    if getattr(args, "path", "stroke") == "spiral":
        # path error is the in-plane distance from the commanded spiral
        cmd = np.array([tangential(float(tt), args, t_hat, t2_hat)[0] for tt in out["t"][sel]])
        d = (x - x[0]) - (cmd - cmd[0])
        err = np.linalg.norm(d - np.outer(d @ n_out, n_out), axis=1)
        got = want = None
    else:
        got = (x - x[0]) @ t_hat
        want = out["stroke"][sel] - out["stroke"][sel][0]
        err = got - want
    out["force_rmse"] = float(np.sqrt(np.mean((f - args.f_target) ** 2)))
    out["force_rel"] = out["force_rmse"] / args.f_target
    out["path_error"] = float(np.sqrt(np.mean(err ** 2)))
    out["contact"] = float(np.mean(f > 0.1 * args.f_target))
    out["peak"] = float(f.max())
    return out


def frame_from_soft_axis(u_soft: np.ndarray) -> np.ndarray:
    """Frame whose THIRD column is `u_soft`, matching compose()'s convention
    that the soft axis comes last."""
    u = np.asarray(u_soft, float); u = u / np.linalg.norm(u)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, a); v /= np.linalg.norm(v)
    R = np.column_stack([v, np.cross(u, v), u])
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    return R


def frame_study(args) -> None:
    """The same label numbers read in four frames, on a scene that can be wiped.

    The motion-aligned reading is the one that matters: on a wipe the stroke is
    tangential, so a rule that puts its soft axis along the motion puts it 90
    degrees from the surface normal.  That is the case the lean simulator says
    the task cannot absorb, and this is it on the arm.
    """
    env, R_task = build_scene(np.deg2rad(args.tilt_x), np.deg2rad(args.tilt_y), args.pad_radius)
    u = env.unwrapped
    slave = u.agent.agents[1]
    link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)
    R_ee_rest, _ = B.get_pose(link)
    env.close()

    n_out, t_hat = R_task[:, 2], R_task[:, 0]
    frames = {
        "task frame (soft on the normal)": R_task,
        "world axes (soft on world z)": np.eye(3),
        "end-effector rest axes": R_ee_rest,
        "motion-aligned (soft on the stroke)": frame_from_soft_axis(t_hat),
    }
    rows = []
    for name, F in frames.items():
        env, R_task = build_scene(np.deg2rad(args.tilt_x), np.deg2rad(args.tilt_y), args.pad_radius)
        u = env.unwrapped
        pad = weld_pad(u, args.pad_radius, args.pad_ahead)
        K = compose(args.magnitude, args.ratio, F)
        r = rollout(u, pad, R_task, K, args)
        k_n = float(n_out @ K @ n_out)
        ang = np.rad2deg(np.arccos(np.clip(abs(float(F[:, 2] @ n_out)), 0, 1)))
        rows.append((name, r, k_n, ang))
        print(f"  {name} done")
        env.close()

    k_ref = rows[0][2]
    print("\n" + "=" * 100)
    print(f"FRAME COMPARISON on a wipeable scene  (magnitude {args.magnitude:.0f}, "
          f"ratio {args.ratio:.0f}:1, stroke {1000*args.stroke_len:.0f} mm)")
    print("=" * 100)
    print(f"  {'label frame':<36}{'soft axis':>11}{'K normal':>11}{'contact':>9}"
          f"{'force':>15}{'path':>10}{'peak':>8}")
    print("  " + "-" * 96)
    for name, r, k_n, ang in rows:
        print(f"  {name:<36}{ang:8.0f} deg{k_n:11.0f}{100*r['contact']:8.0f}%"
              f"{r['force_rmse']:9.2f} N ({100*r['force_rel']:3.0f}%){1000*r['path_error']:8.2f} mm"
              f"{r['peak']:7.1f}")
    print("  " + "-" * 96)
    print("  'world axes' and 'end-effector rest axes' are the SAME matrix here: the rest")
    print("  TCP's z lies along -world z and the two stiff axes are degenerate.")
    print(f"  The motion-aligned reading puts the soft axis {rows[-1][3]:.0f} deg from the normal and")
    print(f"  makes the surface {rows[-1][2]/k_ref:.1f}x stiffer than intended along it.")
    k_t = float(t_hat @ compose(args.magnitude, args.ratio, R_task) @ t_hat)
    print(f"\n  measured normal stiffness against the quadratic form "
          f"Kn cos^2(th) + Kt sin^2(th), Kn={k_ref:.0f} Kt={k_t:.0f}:")
    for name, r, k_n, ang in rows:
        th = np.deg2rad(ang)
        pred = k_ref * np.cos(th) ** 2 + k_t * np.sin(th) ** 2
        print(f"    {ang:5.0f} deg  measured {k_n:7.0f}  predicted {pred:7.0f}"
              f"  ({100*abs(k_n-pred)/pred:4.1f}% off)")
    plot_frames(rows, args, k_ref, k_t)


def plot_frames(rows, args, k_ref, k_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8))

    for i, (name, r, k_n, ang) in enumerate(rows):
        sel = r["t"] >= args.settle
        ax[0].plot(r["t"][sel], r["f_filt"][sel], color=SLOT[i], lw=2.0,
                   label=f"{name.split(' (')[0]}  ({100*r['force_rel']:.0f}% err)")
    ax[0].axhline(args.f_target, color=INK3, ls="--", lw=1.3)
    ax[0].annotate(f" commanded {args.f_target:.0f} N", xy=(0.02, 0.06),
                   xycoords="axes fraction", color=INK2, fontsize=8)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("force along the surface normal (N)")
    ax[0].set_title("Holding force while wiping a tilted plate", color=INK, loc="left")
    ax[0].grid(True, alpha=0.9); ax[0].set_axisbelow(True)
    ax[0].legend(fontsize=7.5, loc="upper right")

    th = np.linspace(0, np.pi / 2, 100)
    ax[1].plot(np.rad2deg(th), k_ref * np.cos(th) ** 2 + k_t * np.sin(th) ** 2,
               color=INK3, lw=1.6, ls="--",
               label=r"$K_n\cos^2\theta + K_t\sin^2\theta$")
    seen: dict = {}
    for i, (name, r, k_n, ang) in enumerate(rows):
        ax[1].plot([ang], [k_n], "o", color=SLOT[i], ms=9, zorder=3,
                   markeredgecolor=SURFACE, markeredgewidth=1.2)
        # Frames that resolve to the same matrix land on the same point; stack
        # their labels rather than drawing them on top of one another.
        seen[round(ang)] = seen.get(round(ang), 0) + 1
        ax[1].annotate(f"  {name.split(' (')[0]}\n  {100*r['force_rel']:.0f}% force error",
                       (ang, k_n), textcoords="offset points",
                       xytext=(8, -4 - 24 * (seen[round(ang)] - 1)),
                       color=SLOT[i], fontsize=7.5, va="center")
    ax[1].set_xlabel(r"angle between the label's soft axis and the surface normal (deg)")
    ax[1].set_ylabel("effective stiffness along the normal (N/m)")
    ax[1].set_title("The frame error is a stiffness error, exactly as predicted",
                    color=INK, loc="left")
    ax[1].set_xlim(-6, 118)
    ax[1].grid(True, alpha=0.9); ax[1].set_axisbelow(True)
    ax[1].legend(fontsize=8, loc="upper left")

    fig.suptitle("FR3, wipeable scene: a label that never states its frame can put the soft axis 90 degrees from the surface normal",
                 color=INK, fontsize=10.5, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=170)
    print(f"\n  wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt-x", type=float, default=18.0)
    ap.add_argument("--tilt-y", type=float, default=-10.0)
    ap.add_argument("--pad-radius", type=float, default=0.015)
    ap.add_argument("--pad-ahead", type=float, default=0.012)
    ap.add_argument("--magnitude", type=float, default=1200.0)
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--inner-stiffness", type=float, default=3000.0)
    ap.add_argument("--rot-stiffness", type=float, default=200.0)
    ap.add_argument("--wrist-inertia", type=float, default=0.004)
    ap.add_argument("--f-target", type=float, default=8.0)
    ap.add_argument("--f-limit", type=float, default=30.0)
    ap.add_argument("--v-limit", type=float, default=0.03)
    # 50 mm peak-to-peak.  An earlier note here blamed the growing path error of
    # longer strokes on the arm's reach; it was the outer layer's v_limit.  At
    # 0.03 m/s a 90 mm stroke (0.071 m/s peak) tracks to 24 mm; raising the clamp
    # to 0.10 m/s takes the same stroke to 0.97 mm with contact held throughout.
    ap.add_argument("--stroke-len", type=float, default=0.05)
    ap.add_argument("--stroke-hz", type=float, default=0.25)
    ap.add_argument("--approach-s", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--force-hz", type=float, default=20.0)
    ap.add_argument("--path", choices=["stroke", "spiral"], default="stroke")
    ap.add_argument("--outer", choices=["admittance", "rigid"], default="admittance")
    ap.add_argument("--height-error", type=float, default=0.0,
                    help="how wrong the commanded surface height is, m")
    ap.add_argument("--n-bumps", type=int, default=0)
    ap.add_argument("--bump-radius", type=float, default=0.010)
    ap.add_argument("--bump-proud", type=float, default=0.0015)
    # The spiral starts at the centre, where the tool already is after pressing
    # in.  Starting at a finite radius asks it to jump there, and the path error
    # then sits at exactly r0 for the whole run whatever else is changed.
    ap.add_argument("--r0", type=float, default=0.0)
    ap.add_argument("--r-max", type=float, default=0.070)
    ap.add_argument("--turns", type=float, default=3.0)
    ap.add_argument("--path-s", type=float, default=14.0)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--out", default="fr3_wipe_frames.png")
    args = ap.parse_args()

    if args.frames:
        frame_study(args)
        return

    env, R_task = build_scene(np.deg2rad(args.tilt_x), np.deg2rad(args.tilt_y), args.pad_radius)
    u = env.unwrapped
    pad = weld_pad(u, args.pad_radius, args.pad_ahead)
    slave = u.agent.agents[1]
    link = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)

    n_out = R_task[:, 2]
    plate_top = np.array(PLATE_POS) + PLATE_HALF[2] * n_out
    _, p_tcp = B.get_pose(link)
    print(f"plate {2*PLATE_HALF[0]*1000:.0f} x {2*PLATE_HALF[1]*1000:.0f} mm at "
          f"{args.tilt_x:+.0f}/{args.tilt_y:+.0f} deg, top centre {plate_top.round(4)}")
    print(f"pad radius {1000*args.pad_radius:.0f} mm, {1000*args.pad_ahead:.0f} mm ahead of the TCP")
    print(f"TCP {p_tcp.round(4)}   gap to the plate along the normal "
          f"{1000*float(n_out @ (p_tcp - plate_top)):.1f} mm")

    # hold everything still and confirm the weld survives
    robot = slave.robot
    q = robot.get_qpos()[0].cpu().numpy()
    for i, j in enumerate(robot.active_joints):
        j.set_drive_target(float(q[i]))
    p0 = pad.pose.p[0].cpu().numpy().copy()
    for _ in range(500):
        u.scene.step()
    drift = 1000 * np.linalg.norm(pad.pose.p[0].cpu().numpy() - p0)
    print(f"weld holds: pad drifted {drift:.3f} mm in 1 s\n")

    K = compose(args.magnitude, args.ratio, R_task)   # soft along the normal
    r = rollout(u, pad, R_task, K, args, report=True)
    print(f"  nominal wipe, task-frame stiffness: contact {100*r['contact']:.0f}%  "
          f"force {r['force_rmse']:.2f} N ({100*r['force_rel']:.0f}%)  "
          f"path {1000*r['path_error']:.2f} mm  peak {r['peak']:.1f} N")
    env.close()


if __name__ == "__main__":
    main()
