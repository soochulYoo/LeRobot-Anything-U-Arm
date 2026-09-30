"""Record the demonstrations as video.

Two recordings, each chosen to show something the numbers alone do not:

  wipe       the SAME label numbers read in two frames, side by side and stepped
             in lockstep -- left, the task frame, soft along the surface normal;
             right, the motion-aligned reading, whose soft axis is 90 degrees
             away.  The arms look alike; the force traces do not.
  insertion  the contrasting task, where motion and force share an axis and the
             frame question cannot arise.

Usage:  python3 record_demo.py --what wipe       [--out wipe_demo]
        python3 record_demo.py --what insertion  [--out insertion_demo]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

import haptic_teleop_fr3_bilateral as B
import fr3_wipe_scene as W
from mani_skill.utils import sapien_utils
from haptic_teleop_fr3_demo import build_pinocchio


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_of(env) -> np.ndarray:
    img = env.render()
    a = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    return a[0] if a.ndim == 4 else a


def annotate(img: np.ndarray, title: str, lines: list[str], colour=(255, 255, 255)) -> np.ndarray:
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, im.width, 34 + 22 * len(lines)], fill=(0, 0, 0, 150))
    d.text((12, 6), title, font=_font(22), fill=colour)
    for i, ln in enumerate(lines):
        d.text((12, 34 + 22 * i), ln, font=_font(17), fill=(235, 235, 235))
    return np.asarray(im)


def save(frames: list[np.ndarray], out: str, fps: int) -> None:
    imageio.mimsave(f"{out}.mp4", frames, fps=fps, quality=8, macro_block_size=1)
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2)))
             for f in frames[::2]]
    imageio.mimsave(f"{out}.gif", small, duration=2.0 / fps, loop=0)
    print(f"  wrote {out}.mp4 ({len(frames)} frames) and {out}.gif")


# --------------------------------------------------------------------------- #
def record_wipe(args) -> None:
    """Two runs stepped in lockstep so the frames line up for a split screen."""
    tx, ty = np.deg2rad(args.tilt_x), np.deg2rad(args.tilt_y)
    # Framed on the contact: the plate fills the pane and the pad is visible
    # touching it.  A wider view put half the frame on empty floor.
    cam = dict(eye=[0.88, -0.42, 0.34], at=[0.60, -0.01, 0.07],
               size=(960, 720) if args.single else (720, 720))

    labels = ("task frame",) if args.single else ("task frame", "motion-aligned")
    cases = []
    for label in labels:
        env, R_task = W.build_scene(tx, ty, args.pad_radius, render_mode="rgb_array", cam=cam)
        u = env.unwrapped
        pad = W.weld_pad(u, args.pad_radius, args.pad_ahead)
        F = R_task if label == "task frame" else W.frame_from_soft_axis(R_task[:, 0])
        cases.append(dict(env=env, u=u, pad=pad, R=R_task, label=label,
                          K=W.compose(args.magnitude, args.ratio, F)))

    # per-case controller state
    for c in cases:
        u, R_task = c["u"], c["R"]
        slave = u.agent.agents[1]
        robot = slave.robot
        names = set(slave.arm_joint_names)
        idx = np.array([i for i, j in enumerate(robot.active_joints) if j.name in names])
        for j in [j for j in robot.active_joints if j.name in names]:
            j.set_drive_properties(0.0, 0.0, force_limit=1000.0)
        pmodel, order = build_pinocchio(slave.urdf_path, robot)
        link = sapien_utils.get_obj_by_name(robot.get_links(), slave.ee_link_name)
        qlim = robot.get_qlimits()[0].cpu().numpy()[idx]
        R_ee0, p0 = B.get_pose(link)
        n_out = R_task[:, 2]
        g = B.BilateralGains()
        g.Ki = args.inner_stiffness * np.eye(3)
        g.Di = 2.0 * 0.8 * np.sqrt(2.0 * args.inner_stiffness) * np.eye(3)
        g.Kr = args.rot_stiffness
        g.Dr = 2.0 * 0.8 * np.sqrt(args.wrist_inertia * args.rot_stiffness)
        plate_top = np.array(W.PLATE_POS) + W.PLATE_HALF[2] * n_out
        c.update(robot=robot, idx=idx, pmodel=pmodel, ee_idx=order.index(slave.ee_link_name),
                 link=link, q_mid=(qlim[:, 0] + qlim[:, 1]) / 2.0, slave=slave,
                 nq=len(robot.active_joints), gains=g, n_out=n_out, n_hat=-n_out,
                 t_hat=R_task[:, 0], t2_hat=R_task[:, 1],
                 R_ref=W.align_z_to(R_ee0, -n_out), p0=p0,
                 x_r=p0.copy(), v_r=np.zeros(3), fhist=[],
                 press=float(n_out @ (p0 - plate_top)) - (args.pad_ahead + args.pad_radius))

    dt = 1.0 / cases[0]["u"].sim_freq
    every = max(1, int(round(1.0 / (args.fps * dt))))
    frames = []
    for step in range(int(args.duration / dt)):
        t = step * dt
        for c in cases:
            depth = c["press"] * min(1.0, t / args.approach_s)
            tang, _ = W.tangential(t, args, c["t_hat"], c["t2_hat"])
            x_ref = c["p0"] + depth * c["n_hat"] + tang
            f_d = args.f_target * c["n_hat"] * min(1.0, t / args.approach_s)
            f_e = c["u"].scene.get_pairwise_contact_forces(c["pad"], c["u"].plate)[0].cpu().numpy()
            drive = f_d + c["K"] @ (x_ref - c["x_r"])
            a_r = np.linalg.solve(3.0 * np.eye(3),
                                  drive + np.clip(f_e, -args.f_limit, args.f_limit)
                                  - 25.0 * np.eye(3) @ c["v_r"])
            c["v_r"] = c["v_r"] + a_r * dt
            sp = float(np.linalg.norm(c["v_r"]))
            if sp > args.v_limit:
                c["v_r"] *= args.v_limit / sp
            c["x_r"] = c["x_r"] + c["v_r"] * dt
            tau, _ = B.geometric_impedance_torque(
                c["slave"], c["link"], c["pmodel"], c["ee_idx"], c["idx"], c["gains"],
                c["x_r"], c["v_r"], c["R_ref"], c["q_mid"])
            qf = np.zeros(c["nq"]); qf[c["idx"]] = tau
            c["robot"].set_qf(qf[None, :])
            c["u"].scene.step()
            c["fhist"].append(float(f_e @ c["n_out"]))

        if step % every == 0 and t >= args.skip:
            panes = []
            for c in cases:
                n = max(1, int(round(1.0 / (args.force_hz * dt))))
                f = float(np.mean(np.abs(c["fhist"][-n:])))
                k_n = float(c["n_out"] @ c["K"] @ c["n_out"])
                r_now = float(np.linalg.norm(W.tangential(t, args, c["t_hat"], c["t2_hat"])[0]))
                extra = ([f"outward spiral, radius {1000*r_now:5.1f} mm"]
                         if args.path == "spiral" else [])
                panes.append(annotate(
                    frame_of(c["env"]),
                    f"{c['label']}",
                    extra + [f"stiffness along the surface normal: {k_n:.0f} N/m",
                             f"commanded {args.f_target:.0f} N    measured {f:5.2f} N",
                             f"t = {t:5.2f} s"],
                    colour=(120, 200, 255) if c["label"] == "task frame" else (255, 190, 90)))
            frames.append(np.concatenate(panes, axis=1))
    for c in cases:
        c["env"].close()
    save(frames, args.out or "wipe_demo", args.fps)


# --------------------------------------------------------------------------- #
def record_insertion(args) -> None:
    import peg_insertion_cascade as P

    s = P.setup(args.seed, 0.040, render_mode="rgb_array")
    u = s["u"]
    Rh, hole = P.hole_frame(u)
    a = argparse.Namespace(
        seed=args.seed, check=False, frames=False, episodes=1, magnitude=1200.0, ratio=10.0,
        inner_stiffness=3000.0, rot_stiffness=400.0, wrist_inertia=0.005, grip_back=0.040,
        f_push=12.0, f_limit=30.0, v_limit=0.06, standoff_margin=0.035, overdrive=0.010,
        align_s=4.0, insert_s=4.0, duration=args.duration, lateral_mm=0.0, retarget=True)

    dt = 1.0 / u.sim_freq
    every = max(1, int(round(1.0 / (args.fps * dt))))
    frames = []
    r = P.rollout(s, P.compose_axial(1200.0, 10.0, Rh), np.zeros(3), a,
                  on_frame=lambda t, depth, f: frames.append(annotate(
                               frame_of(s["env"]),
                               "peg insertion",
                               ["motion and force share the hole axis",
                                f"insertion depth {1000*depth:+7.1f} mm   (success at -15 mm)",
                                f"contact force {f:5.1f} N",
                                f"t = {t:5.2f} s"],
                               colour=(150, 230, 150))),
                  every=every)
    print(f"  success: {r['success']}")
    s["env"].close()
    save(frames, args.out or "insertion_demo", args.fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["wipe", "insertion"], default="wipe")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=float, default=2.0, help="seconds to skip at the start")
    # wipe parameters, matching fr3_wipe_scene's defaults
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
    ap.add_argument("--stroke-len", type=float, default=0.05)
    ap.add_argument("--stroke-hz", type=float, default=0.25)
    ap.add_argument("--approach-s", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--force-hz", type=float, default=20.0)
    ap.add_argument("--single", action="store_true", help="one pane, task frame only")
    ap.add_argument("--path", choices=["stroke", "spiral"], default="stroke")
    ap.add_argument("--r0", type=float, default=0.0)
    ap.add_argument("--r-max", type=float, default=0.10)
    ap.add_argument("--turns", type=float, default=3.5)
    ap.add_argument("--path-s", type=float, default=18.0)
    args = ap.parse_args()
    (record_wipe if args.what == "wipe" else record_insertion)(args)


if __name__ == "__main__":
    main()
