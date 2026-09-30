"""Does rotational stiffness matter when the surface is curved?

The claim under test: on a surface whose normal turns as the tool travels, a
flat pad held by a STIFF rotational impedance cannot stay flush.  It keeps the
orientation it was commanded, meets the surface on one edge, and the contact
degrades.  A soft rotational impedance lets the contact moment reorient the pad,
so it lies flat and the force stays where it was asked to be.

Two things have to be true for the question to be non-vacuous, and both are
arranged deliberately here:

  THE TOOL MUST HAVE EXTENT.  The earlier wipe scene welded a SPHERE to the
  hand.  A sphere touches a surface at one point whatever its orientation, so it
  transmits no moment and K_R has nothing to resist.  This one welds a flat box.

  THE SURFACE MUST HAVE CURVATURE, not slope.  A tilted plane has a constant
  normal; align once and a rigid tool stays aligned forever.  surface.py builds
  a random Gaussian-bump field whose normal turns 6-10 deg as the tool travels
  and swings ~5 deg across the width of the pad itself.

Everything else is held fixed between the two conditions -- same surface, same
path, same translational gains, same outer admittance regulating normal force.
Only K_R changes, so anything that differs is rotational stiffness.

Usage:  python3 wipe.py --check           scene sanity, one short run
        python3 wipe.py --sweep           K_R sweep over several surfaces
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

import gymnasium as gym
import numpy as np
import sapien

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import mani_skill.envs  # noqa: F401
from mani_skill.utils import sapien_utils

import haptic_teleop_fr3_bilateral as B
from haptic_teleop_fr3_demo import HapticTeleopDemoEnv, R_to_quat_wxyz, build_pinocchio
from fr3_tilted_wipe import align_z_to, lowpass, sqrtm_spd

from surface import CurvedSurface

SURF_POS = np.array([0.58, 0.0, 0.05])
SCRATCH = pathlib.Path(os.environ.get("CLAUDE_SCRATCH", "/tmp")) / "curved_meshes"


def build_scene(surf: CurvedSurface, render_mode: str | None = None,
                cam: dict | None = None):
    """The demo env with its small box swapped for the curved slab."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"surface_{surf.key()}.obj"
    if not path.exists():
        surf.mesh().export(path)

    old_scene = HapticTeleopDemoEnv._load_scene
    old_cam = HapticTeleopDemoEnv._default_human_render_camera_configs

    def patched(self, options):
        old_scene(self, options)
        self.box.set_pose(sapien.Pose(p=[0.0, 0.0, -5.0]))
        builder = self.scene.create_actor_builder()
        # Static + nonconvex: a convex hull of this mesh would be a dome and
        # would erase every dent, which is half of what makes it curved.
        builder.add_nonconvex_collision_from_file(str(path))
        builder.add_visual_from_file(str(path))
        builder.initial_pose = sapien.Pose(p=SURF_POS)
        self.surface = builder.build_static(name="curved_surface")

    if render_mode is not None:
        from mani_skill.sensors.camera import CameraConfig
        c = cam or {}
        eye, at = c.get("eye", [1.10, -0.80, 0.70]), c.get("at", [0.58, 0.0, 0.08])
        w, h = c.get("size", (960, 720))
        fov = c.get("fov", 1.0)

        @property
        def cam_cfg(self):
            return CameraConfig("render_camera", sapien_utils.look_at(eye, at),
                                w, h, fov, 0.01, 100)

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
    return env


def weld_pad(u, half_w: float, thick: float, ahead: float):
    """A FLAT box welded to panda_hand: the face, not a point, does the wiping."""
    slave = u.agent.agents[1]
    hand = sapien_utils.get_obj_by_name(slave.robot.get_links(), "panda_hand")
    tcp = sapien_utils.get_obj_by_name(slave.robot.get_links(), slave.ee_link_name)
    R_h = tcp.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    pad_p = tcp.pose.p[0].cpu().numpy() + (ahead + thick) * R_h[:, 2]

    builder = u.scene.create_actor_builder()
    hs = (half_w, half_w, thick)
    builder.add_box_collision(half_size=hs, density=8000.0)   # ~160 g pad
    builder.add_box_visual(half_size=hs, material=sapien.render.RenderMaterial(
        base_color=[0.9, 0.35, 0.2, 1]))
    builder.initial_pose = sapien.Pose(p=pad_p, q=R_to_quat_wxyz(R_h))
    pad = builder.build(name="wipe_pad")

    hand_pose = hand.pose
    R_hand = hand_pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    rel = R_hand.T @ (pad_p - hand_pose.p[0].cpu().numpy())
    drive = u.scene.create_drive(hand, sapien.Pose(p=rel), pad, sapien.Pose())
    for comp in getattr(drive, "_objs", [drive]):
        for ax in ("x", "y", "z"):
            getattr(comp, f"set_limit_{ax}")(0.0, 0.0)
        comp.set_limit_cone(0.0, 0.0)
        comp.set_limit_twist(0.0, 0.0)
    return pad


def op_space_rot_inertia(robot, pmodel, ee_idx, idx, tool_offset) -> np.ndarray:
    """Rotational block of Lambda = (J M^-1 J^T)^-1, in the BODY frame.

    The GIC's damping acts on the body twist error, and SAPIEN's
    compute_single_link_local_jacobian already returns the body Jacobian, so
    this is the inertia the rotational damping actually sees.  Evaluated once at
    the starting configuration: it varies with q, but holding it is a far
    smaller error than the 27x a hand-guessed scalar was making.
    """
    q = robot.get_qpos()[0].cpu().numpy()
    J = pmodel.compute_single_link_local_jacobian(q, ee_idx)[:, idx]
    a = np.asarray(tool_offset, dtype=float).reshape(3)
    if np.any(a):
        J = np.block([[np.eye(3), -B.skew(a)], [np.zeros((3, 3)), np.eye(3)]]) @ J
    M = pmodel.compute_generalized_mass_matrix(q)[np.ix_(idx, idx)]
    return np.linalg.inv(J @ np.linalg.solve(M, J.T))[3:, 3:]


def rollout(u, pad, surf: CurvedSurface, Kr: float, args, report=False,
            on_frame=None, every: int = 1) -> dict:
    """One traverse across the curved surface at rotational stiffness `Kr`.

    Translational control is IDENTICAL in every condition: the same outer
    admittance regulating the same target normal force, the same inner Kp.  The
    orientation command is a constant flat pose -- the controller is never told
    the surface curves, which is the whole point.  Only Kr changes.
    """
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
    q_mid = robot.get_qlimits()[0].cpu().numpy()[idx].mean(axis=1)
    nq = len(robot.active_joints)

    n_up = np.array([0.0, 0.0, 1.0])        # the slab is not tilted; +z is out
    n_hat = -n_up                            # press direction
    t_hat, t2_hat = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    R_ee0, p0 = B.get_pose(link)
    R_ref = align_z_to(R_ee0, n_hat)         # commanded orientation: FLAT, fixed

    # WITH A COMPLIANCE CENTRE, EVERY REFERENCE IS THE OFFSET POINT'S, not the
    # TCP's.  Anchoring x_ref at p0 while the controller tracks the pad centre
    # puts the whole command 34 mm too high and the pad stops short of the
    # surface with 0 N for the entire run.
    p_ctrl0 = p0 + (args.pad_ahead + args.pad_thick) * R_ee0[:, 2]
    # Where the mean surface sits under the starting point, from the closed-form
    # height rather than from a probe: the controller is allowed to know the
    # nominal plane, just not the curvature.
    loc0 = p_ctrl0 - SURF_POS
    z_top = SURF_POS[2] + float(surf.height(loc0[0], loc0[1]))
    # DISTANCE TO TRAVEL, positive: x_ref = p0 + press*n_hat drives downward.
    # Writing it the other way round sends the arm up and away, and the run
    # reports 0 N for the whole traverse.
    press = (p_ctrl0[2] - z_top) - args.pad_thick
    assert press > 0.0, (f"pad starts below the surface: pad {1000*p_ctrl0[2]:.0f} mm,"
                         f" surface {1000*z_top:.0f} mm")
    if report:
        print(f"      pad centre {1000*p_ctrl0[2]:.0f} mm, surface {1000*z_top:.0f} mm"
              f" -> press {1000*press:.1f} mm down")

    gains = B.BilateralGains()
    gains.Ki = args.lin_stiffness * np.eye(3)
    gains.Di = 2.0 * 0.8 * np.sqrt(2.0 * args.lin_stiffness) * np.eye(3)
    gains.Kr = Kr
    # Dr = 2 zeta sqrt(I*Kr) needs the TRUE rotational inertia.  Guessing a
    # scalar 5e-3 put the real damping ratio at 0.15 on every axis, and the
    # wrist rang through the whole traverse -- which showed up as an apparent
    # Kr effect even on a FLAT slab, where the normal never turns.  The
    # operational-space inertia is measured here instead, and it is strongly
    # anisotropic (eigenvalues spread ~40x), so Dr must be a MATRIX.
    # Compliance centre at the pad, so K_R acts about the contact and not about
    # a wrist 70 mm away -- otherwise the lever arm, not Kr, sets the result.
    gains.tool_offset = np.array([0.0, 0.0, args.pad_ahead + args.pad_thick])
    Irot = op_space_rot_inertia(robot, pmodel, ee_idx, idx, gains.tool_offset)
    gains.Dr = 2.0 * args.zeta * sqrtm_spd(Irot * Kr)
    if report:
        print(f"      I_rot eigenvalues {np.array2string(np.linalg.eigvalsh(Irot), precision=4)}"
              f" kg m^2  ->  Dr diag {np.array2string(np.diag(gains.Dr), precision=3)}")
    # Compliance centre at the pad, so K_R acts about the contact and not about
    # a wrist 70 mm away -- otherwise the lever arm, not Kr, sets the result.


    Ma, Br = 3.0 * np.eye(3), 25.0 * np.eye(3)
    K_world = args.outer_stiffness * np.eye(3)
    # The raw 500 Hz pairwise force reads 0 on a large fraction of samples and
    # spikes to tens of N while its MEAN is correct -- a box sliding over a
    # triangle mesh re-seats its contact points every step.  Fed straight into
    # the admittance that noise becomes reference motion and the pad bounces
    # (measured: 0 -> 44 -> 3 -> 0 N within 1.2 s, 51% contact).  Every real
    # admittance filters its wrench; this is that filter, and it is identical
    # in all conditions so it cannot create the effect under test.
    alpha_f = float(np.exp(-2.0 * np.pi * args.sense_hz * dt))
    f_s = np.zeros(3)
    x_r, v_r = p_ctrl0.copy(), np.zeros(3)
    log = {k: [] for k in ["t", "f_n", "align", "tilt", "x", "cmd"]}

    for step in range(int(args.duration / dt)):
        t = step * dt
        ramp = min(1.0, t / args.approach_s)
        depth = press * ramp
        # A single slow traverse: long enough to cross several bumps.
        s = 0.0 if t < args.settle else min(1.0, (t - args.settle) / args.path_s)
        tang = (-0.5 + s) * args.stroke_len * t_hat + args.y_offset * t2_hat
        x_ref = p_ctrl0 + depth * n_hat + tang
        f_d = args.f_target * n_hat * ramp

        f_e = u.scene.get_pairwise_contact_forces(pad, u.surface)[0].cpu().numpy()
        f_s = alpha_f * f_s + (1.0 - alpha_f) * np.clip(f_e, -args.f_limit, args.f_limit)
        drive = f_d + K_world @ (x_ref - x_r)
        a_r = np.linalg.solve(Ma, drive + f_s - Br @ v_r)
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

        # How flush is the pad?  Angle between the face it presses with and the
        # TRUE local normal under the pad centre.
        T_pad = pad.pose.to_transformation_matrix()[0].cpu().numpy()
        face = -T_pad[:3, :3][:, 2]                  # outward from the wiping face
        c = T_pad[:3, 3] - SURF_POS
        n_loc = surf.normal(c[0], c[1])
        log["t"].append(t)
        log["f_n"].append(float(f_e @ n_up))
        log["align"].append(float(np.degrees(np.arccos(
            np.clip(face @ n_loc, -1.0, 1.0)))))
        log["tilt"].append(float(np.degrees(np.arccos(np.clip(n_loc @ n_up, -1, 1)))))
        log["x"].append(B.get_pose(link)[1].copy())
        log["cmd"].append(x_ref.copy())
        if on_frame is not None and step % every == 0:
            on_frame(t, float(f_e @ n_up), log["align"][-1], log["tilt"][-1])

    out = {k: np.array(v) for k, v in log.items()}
    # Raw 500 Hz contact force reads 0 on a large fraction of samples and spikes
    # while its mean is correct; every force statistic uses the 20 Hz signal.
    out["f_filt"] = lowpass(np.abs(out["f_n"]), dt, args.force_hz)
    out["peak_raw"] = float(np.max(np.abs(out["f_n"])))
    sel = out["t"] >= args.settle + 0.2
    f, a = out["f_filt"][sel], out["align"][sel]
    # A diverged run must never be averaged in: one flat-control run reached
    # 840 N peak and 542 N RMSE, which on its own moved that whole arm of the
    # sweep.  Flagged rather than raised, so one bad cell does not lose the run.
    out["Kr"] = Kr
    out["diverged"] = bool(out["peak_raw"] > args.diverge_n
                           or not np.all(np.isfinite(out["f_filt"])))
    out["align_mean"] = float(a.mean())
    out["align_p95"] = float(np.percentile(a, 95))
    out["tilt_mean"] = float(out["tilt"][sel].mean())
    out["force_rmse"] = float(np.sqrt(np.mean((f - args.f_target) ** 2)))
    out["f_mean"] = float(f.mean())
    out["contact"] = float(np.mean(f > 0.1 * args.f_target))
    out["peak"] = float(f.max())
    return out


def run_one(surf: CurvedSurface, Kr: float, args, report=False,
            render_mode=None, cam=None, on_frame=None, every: int = 1) -> dict:
    """A fresh scene per condition, so no run inherits the previous one's pose."""
    env = build_scene(surf, render_mode=render_mode, cam=cam)
    try:
        u = env.unwrapped
        pad = weld_pad(u, args.pad_half, args.pad_thick, args.pad_ahead)
        if on_frame is not None:
            on_frame = (lambda cb: lambda *a: cb(env, *a))(on_frame)
        return rollout(u, pad, surf, Kr, args, report=report,
                       on_frame=on_frame, every=every)
    finally:
        env.close()


def add_args(p):
    p.add_argument("--f-target", type=float, default=8.0)
    p.add_argument("--f-limit", type=float, default=60.0)
    p.add_argument("--lin-stiffness", type=float, default=1200.0)
    p.add_argument("--outer-stiffness", type=float, default=600.0)
    p.add_argument("--zeta", type=float, default=1.0)
    p.add_argument("--pad-half", type=float, default=0.025)   # 50 mm square face
    p.add_argument("--pad-thick", type=float, default=0.004)
    p.add_argument("--pad-ahead", type=float, default=0.030)
    p.add_argument("--stroke-len", type=float, default=0.16)
    p.add_argument("--y-offset", type=float, default=0.0)
    p.add_argument("--approach-s", type=float, default=1.2)
    p.add_argument("--settle", type=float, default=1.6)
    p.add_argument("--path-s", type=float, default=6.0)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--v-limit", type=float, default=0.10)
    p.add_argument("--force-hz", type=float, default=20.0)   # reporting filter
    p.add_argument("--sense-hz", type=float, default=15.0)   # control-side filter
    p.add_argument("--diverge-n", type=float, default=300.0)
    return p


def main() -> None:
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.check:
        surf = CurvedSurface(seed=args.seed)
        st = surf.stats(args.pad_half)
        print(f"surface seed {args.seed}: height range {1000*st['h_ptp']:.0f} mm,"
              f" tilt {st['tilt_mean']:.1f} deg mean / {st['tilt_p95']:.1f} p95,"
              f" normal swing across pad {st['swing_mean']:.1f} deg\n")
        for Kr in (60.0, 3.0):
            r = run_one(surf, Kr, args, report=True)
            print(f"   Kr={Kr:5.1f}  align {r['align_mean']:5.2f} deg"
                  f"  force {r['f_mean']:5.2f} N (rmse {r['force_rmse']:4.2f})"
                  f"  peak {r['peak']:5.1f} N  contact {100*r['contact']:3.0f}%")


if __name__ == "__main__":
    main()
