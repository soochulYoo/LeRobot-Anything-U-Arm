"""PegInsertionSide under the cascade, starting from a pre-grasped peg.

The wiping study (cascade/study_anisotropy.py, cascade/study_transplant.py) found
the FRAME of a stiffness label to be the factor the task cannot absorb, because
on a wipe the motion is tangential and the contact force is normal -- so a rule
that aligns its axes with motion and a rule that aligns them with force end up
orthogonal.

Insertion is the opposite case, and that is why it is worth running.  Here the
motion and the contact force BOTH lie along the hole axis, so the motion-aligned
and force-aligned frames should coincide with the correct one and the frame
choice should stop mattering.  If that holds, the design rule is specific rather
than universal: frame ambiguity bites when motion and force are orthogonal, and
an insertion-heavy literature would never have noticed it.

The grasp is skipped -- the peg starts between the closed fingers -- because a
pick phase is not part of the hypothesis and only adds failures of its own.

Usage:  python3 peg_insertion_cascade.py [--episodes 5] [--check]
"""
from __future__ import annotations

import argparse
import warnings

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings("ignore")

import mani_skill.envs  # noqa: F401
from mani_skill.utils.structs.pose import Pose
from transforms3d.quaternions import mat2quat

import haptic_teleop_fr3_bilateral as B
from haptic_teleop_fr3_demo import build_pinocchio


def compose_axial(magnitude: float, ratio: float, frame: np.ndarray) -> np.ndarray:
    """K with the FIRST column stiff and the other two soft: stiff along the
    insertion axis to push, compliant laterally to accommodate misalignment.

    This is the mirror image of the wipe's anisotropy, where one axis was soft
    and two were stiff.  det(diag(w)) = 1 keeps magnitude and ratio separable.
    """
    w = np.array([ratio, 1.0, 1.0], dtype=float)
    w = w / np.prod(w) ** (1.0 / 3.0)
    return magnitude * (frame @ np.diag(w) @ frame.T)


def frame_from_axis(u: np.ndarray) -> np.ndarray:
    """Orthonormal frame whose FIRST column is `u`."""
    u = np.asarray(u, float); u = u / max(np.linalg.norm(u), 1e-12)
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, a); v /= max(np.linalg.norm(v), 1e-12)
    R = np.column_stack([u, v, np.cross(u, v)])
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def setup(seed: int, grip_back: float = 0.040, render_mode: str | None = None,
          cam: dict | None = None):
    """Env with the peg already gripped, the arm torque-controlled, and every
    handle the controller needs."""
    # PegInsertionSide's default SimConfig runs at 100 Hz.  An inner Cartesian
    # impedance cannot live there: with a rotational damping matched to
    # Kr = 120 Nm/rad against the wrist's ~0.01 kg m^2, the explicit stability
    # limit is 2 I / D = 1.1 ms, a tenth of the 10 ms step, and the arm explodes
    # inside 50 ms.  cascade/study_layers.py put the inner loop's requirement at
    # 200 Hz minimum for exactly this reason; 500 Hz matches the rest of this
    # project and the gains validated on it.
    from mani_skill.utils.structs.types import SimConfig
    # Plain "panda", not the env's default "panda_wristcam".  The wrist camera's
    # collision body reaches past the fingers and hits the box at 168 N while the
    # peg is still 35 mm short of the success threshold -- the arm then stalls
    # with a 62 mm inner-loop deflection and no peg-box contact to explain it.
    # The cameras are not used here.
    kw = dict(num_envs=1, sim_backend="cpu", robot_uids="panda",
              sim_config=SimConfig(sim_freq=500, control_freq=500))
    old_cam = None
    if render_mode is not None:
        kw["render_mode"] = render_mode
        # The task's own render camera sits far back; frame the hole instead.
        from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils as _su
        old_cam = PegInsertionSideEnv._default_human_render_camera_configs
        c = cam or {}
        eye = c.get("eye", [0.50, 0.02, 0.34])
        at = c.get("at", [0.02, 0.23, 0.12])
        w, h = c.get("size", (720, 540))

        @property
        def _cam(self):
            return CameraConfig("render_camera", _su.look_at(eye, at), w, h, 1.0, 0.01, 100)

        PegInsertionSideEnv._default_human_render_camera_configs = _cam
    try:
        env = gym.make("PegInsertionSide-v1", **kw)
        # reset() is what builds the cameras, so the override has to survive
        # until after it -- restoring at the end of gym.make left the task's own
        # 512x512 camera in place and the reframing silently did nothing.
        env.reset(seed=seed)
    finally:
        if old_cam is not None:
            from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
            PegInsertionSideEnv._default_human_render_camera_configs = old_cam
    u = env.unwrapped
    robot, tcp = u.agent.robot, u.agent.tcp

    Rt = tcp.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    pt = tcp.pose.p[0].cpu().numpy()
    r = float(u.peg_half_sizes[0, 1])

    # Peg axis along TCP-x, which is across the finger closing direction (TCP-y).
    Rp = np.column_stack([Rt[:, 0], Rt[:, 2], np.cross(Rt[:, 0], Rt[:, 2])])
    if np.linalg.det(Rp) < 0:
        Rp[:, 2] *= -1.0
    # Grip the peg BEHIND its centre.  Held at the centre, the hand body reaches
    # about 30 mm past the TCP along the peg axis and strikes the box face at
    # 107 N while the head is still 8 mm short of the success threshold -- a
    # geometric limit of the grasp, not of the controller.  Offsetting the grip
    # toward the tail buys exactly that clearance.
    peg_centre = pt + grip_back * Rp[:, 0]
    u.peg.set_pose(Pose.create_from_pq(
        torch.tensor(peg_centre, dtype=torch.float32)[None, :],
        torch.tensor(mat2quat(Rp), dtype=torch.float32)[None, :]))

    q = robot.get_qpos()[0].cpu().numpy().copy()
    q[-2:] = r + 0.004
    robot.set_qpos(q[None, :])
    robot.set_qvel(np.zeros((1, len(robot.active_joints))))
    u.peg.set_linear_velocity(torch.zeros(1, 3))
    u.peg.set_angular_velocity(torch.zeros(1, 3))

    # Every joint must be told to hold where it is.  Stepping the scene directly
    # bypasses the env's controller, and the drive targets then default to zero,
    # which drives the whole arm to its zero configuration and throws the peg.
    for i, j in enumerate(robot.active_joints):
        j.set_drive_target(0.0 if "finger" in j.name else float(q[i]))
    for _ in range(400):
        u.scene.step()                      # let the grasp settle

    names = set(u.agent.arm_joint_names)
    arm = [j for j in robot.active_joints if j.name in names]
    idx = np.array([i for i, j in enumerate(robot.active_joints) if j.name in names])
    for j in arm:                            # arm goes to torque control
        j.set_drive_properties(0.0, 0.0, force_limit=1000.0)

    pmodel, order = build_pinocchio(u.agent.urdf_path, robot)
    ee_idx = order.index(u.agent.ee_link_name)
    link = [l for l in robot.get_links() if l.name == u.agent.ee_link_name][0]
    qlim = robot.get_qlimits()[0].cpu().numpy()[idx]
    return dict(env=env, u=u, robot=robot, tcp=tcp, link=link, idx=idx,
                pmodel=pmodel, ee_idx=ee_idx, q_mid=(qlim[:, 0] + qlim[:, 1]) / 2.0,
                nq=len(robot.active_joints),
                tcp_to_peg=u.peg.pose.p[0].cpu().numpy() - tcp.pose.p[0].cpu().numpy())


def hole_frame(u) -> tuple[np.ndarray, np.ndarray]:
    hp = u.box_hole_pose
    R = hp.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    return R, hp.p[0].cpu().numpy()


def align_axis(R0: np.ndarray, a_from: np.ndarray, a_to: np.ndarray) -> np.ndarray:
    """Rotate R0 by the smallest rotation carrying `a_from` onto `a_to`.

    The peg is rigidly gripped, so aiming it at the hole is a wrist rotation, not
    a translation: at the grasp it sits 4.5 degrees off the hole axis and would
    simply jam.
    """
    a = np.asarray(a_from, float); a /= np.linalg.norm(a)
    b = np.asarray(a_to, float); b /= np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return R0.copy() if c > 0 else -R0.copy()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + vx + vx @ vx / (1.0 + c)) @ R0


def rollout(s, Ko: np.ndarray, lateral_err: np.ndarray, args,
            on_frame=None, every: int = 0) -> dict:
    """One insertion attempt with outer coupling stiffness `Ko` (world 3x3).

    Phase 1 aligns the peg and carries it to a stand-off; phase 2 advances along
    the hole axis with a commanded force.  `lateral_err` is added to the aim, so
    the lateral compliance has something to accommodate -- with a perfect aim the
    peg goes in whatever the stiffness is and the experiment measures nothing.
    """
    u, robot, tcp, link = s["u"], s["robot"], s["tcp"], s["link"]
    idx, pmodel, ee_idx, q_mid, nq = s["idx"], s["pmodel"], s["ee_idx"], s["q_mid"], s["nq"]
    dt = 1.0 / u.sim_freq
    Rh, _ = hole_frame(u)
    axis = Rh[:, 0]
    goal_peg = u.goal_pose.p[0].cpu().numpy()

    R_ee0, p_tcp0 = B.get_pose(link)
    peg_R0 = u.peg.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
    peg_p0 = u.peg.pose.p[0].cpu().numpy()
    peg_axis0 = peg_R0[:, 0]
    if peg_axis0 @ axis < 0:
        peg_axis0 = -peg_axis0
    R_ref = align_axis(R_ee0, peg_axis0, axis)
    # the grip offset travels with the wrist
    offset_ee = R_ee0.T @ (peg_p0 - p_tcp0)

    # The stand-off has to clear the box, whose depth equals the peg's
    # half-length and therefore changes every episode.  A fixed 140 mm left the
    # longest pegs (123.7 mm) only 16 mm of clearance and they never started.
    half_len = float(u.peg_half_sizes[0, 0])
    standoff = half_len + args.standoff_margin

    gains = B.BilateralGains()
    gains.Ki = args.inner_stiffness * np.eye(3)
    gains.Di = 2.0 * 0.8 * np.sqrt(2.0 * args.inner_stiffness) * np.eye(3)
    # Rotational damping must be matched to the WRIST's inertia, ~0.005 kg m^2
    # with the peg gripped, not to 1 kg m^2.  The earlier 2 zeta sqrt(Kr) made
    # that mistake and over-damped by 14x, which at a 2 ms step put the loop past
    # the explicit stability limit 2I/D: the arm held still for 0.24 s and then
    # locked into a step-to-step sign flip at 142 rad/s.
    # Rotational stiffness in the DESIRED end-effector frame, whose x axis is the
    # hole axis after alignment.  rot_ratio > 1 means stiff in roll about the
    # hole and SOFT in pitch/yaw, so the peg can pivot into the hole -- the RCC
    # idea, expressed as an anisotropy instead of an offset.  The geometric mean
    # is held at rot_stiffness so the ratio does not smuggle in a magnitude.
    w = np.array([args.rot_ratio, 1.0, 1.0], dtype=float)
    w = w / np.prod(w) ** (1.0 / 3.0)
    K_R = args.rot_stiffness * np.diag(w)
    gains.Kr = K_R
    gains.Dr = 2.0 * 0.8 * np.diag(np.sqrt(args.wrist_inertia * np.diag(K_R)))

    # Compliance centre: 0 leaves it at the wrist, 1 puts it at the peg tip.
    # The peg is gripped grip_back behind its centre and the head is half_len
    # ahead of that, so the tip is (half_len + grip_back) along the tool axis,
    # which after the grasp is the end effector's own x axis.
    reach = half_len + args.grip_back
    gains.tool_offset = np.array([args.cc_frac * reach, 0.0, 0.0])

    Ma, Br = 3.0 * np.eye(3), 25.0 * np.eye(3)
    x_r, v_r = p_tcp0.copy(), np.zeros(3)

    # ---- optional SEARCH phase -------------------------------------------
    # A real insertion demonstration does not aim and push: it puts the peg on
    # the face near the hole, presses, and scrubs until the head drops into the
    # aperture.  That is the motion worth having as data, because it is where
    # lateral compliance earns its keep -- a stiff lateral wrist skates over the
    # edge instead of being drawn in.
    # search_s = 0 keeps the original aim-and-push behaviour, so the frame study
    # above is unaffected.
    #
    # Geometry, which the first version of this got wrong.  hole_frame's origin
    # is the box CENTRE, and the box is as deep as the peg is long, so the
    # entrance face sits at x = -half_len and the goal puts the peg centre
    # exactly on that face (its head at the centre).  `at_hole[0]` is the head's
    # x there, so depth past the face is at_hole[0] + half_len -- the earlier
    # catch test (at_hole[0] > 6 mm) asked for 6 mm past the hole CENTRE, which
    # the scrub command could never reach, and the scrub target was 4 mm short
    # of full insertion, i.e. pressing the head through 80 mm of solid wall.
    # There is also no chamfer: the aperture is a square of half-width
    # radius + 3 mm cut by four boxes, so the edge is sharp and the whole
    # tolerance is that 3 mm of clearance.
    search_s = float(getattr(args, "search_s", 0.0))
    e1, e2 = Rh[:, 1], Rh[:, 2]                      # the hole's lateral axes
    search = dict(found=None, catch_p=None,
                  r0=float(getattr(args, "search_r0", 0.001)),
                  growth=float(getattr(args, "search_growth", 0.012)),
                  turns=float(getattr(args, "search_turns", 5.0)),
                  press=float(getattr(args, "search_press", 8.0)),
                  depth=float(getattr(args, "search_depth", 0.003)),
                  catch=float(getattr(args, "search_catch", 0.005)))
    # peg centre that puts the head `depth` past the entrance face
    face_peg = goal_peg - (half_len - search["depth"]) * axis

    pre_peg = goal_peg - standoff * axis + lateral_err
    log = {k: [] for k in ["t", "f", "f_vec", "depth", "peg", "tcp"]}
    n = int(args.duration / dt)
    for step in range(n):
        t = step * dt
        if t < args.align_s:
            a = t / args.align_s
            peg_target = peg_p0 + a * (pre_peg - peg_p0)
            f_d = np.zeros(3)
        elif search_s > 0.0 and search["found"] is None and t < args.align_s + search_s:
            # press the head onto the face and spiral outwards from the believed
            # hole centre until it drops in
            a = (t - args.align_s) / search_s
            rad = search["r0"] + search["growth"] * a
            th = 2.0 * np.pi * search["turns"] * a
            lat = rad * (np.cos(th) * e1 + np.sin(th) * e2)
            peg_target = face_peg + lateral_err + lat
            f_d = search["press"] * axis
        elif search_s > 0.0:
            # push in from wherever the scrub left the peg: if it caught, that
            # position already carries the lateral correction the search found,
            # and if it timed out this pushes blindly and fails, which is the
            # honest outcome.
            if search["found"] is None:
                search["found"] = t
                search["catch_p"] = u.peg.pose.p[0].cpu().numpy().copy()
            a = min(1.0, (t - search["found"]) / args.insert_s)
            p0 = search["catch_p"]
            travel = float((goal_peg - p0) @ axis) + args.overdrive
            peg_target = p0 + a * travel * axis
            f_d = args.f_push * axis * a
        else:
            a = min(1.0, (t - args.align_s) / args.insert_s)
            peg_target = pre_peg + a * ((goal_peg + lateral_err) - pre_peg) + a * args.overdrive * axis
            f_d = args.f_push * axis * a

        # Aim the HEAD, not the centre.  goal_pose gives the peg-centre position
        # that puts the head in the hole ONLY if the peg is perfectly aligned;
        # any residual angle displaces the head by L sin(theta), and at
        # L = 124 mm a 12 degree error is 27 mm -- against 3 mm of clearance.
        # Measured head errors at the box face ran from 1.5 mm (which inserted)
        # to 27 mm (which did not), tracking exactly that.  Correcting with the
        # CURRENT peg axis turns an open-loop aim into a closed-loop one.
        if args.retarget:
            peg_R = u.peg.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
            pa = peg_R[:, 0]
            if pa @ axis < 0:
                pa = -pa
            head_target = peg_target + half_len * axis
            peg_target = head_target - half_len * pa
            R_now, p_now = B.get_pose(link)
            offset_ee = R_now.T @ (u.peg.pose.p[0].cpu().numpy() - p_now)
            R_ref = align_axis(R_ref, R_ref @ (R_ee0.T @ pa), axis)
        # The reference must name the CONTROLLED point, which the compliance
        # centre has just moved.
        x_ref = peg_target - R_ref @ offset_ee + R_ref @ gains.tool_offset

        # PAIRWISE, not net.  The peg's net contact force includes the gripper
        # squeezing it -- tens of newtons that have nothing to do with the task.
        # Feeding the grasp force into the admittance drove the arm to 8 m/s
        # within 10 ms.  Only the peg-box interaction belongs in this loop.
        f_e = u.scene.get_pairwise_contact_forces(u.peg, u.box)[0].cpu().numpy()
        f_e_ctrl = np.clip(f_e, -args.f_limit, args.f_limit)
        if getattr(args, "outer", "admittance") == "rigid":
            # No outer layer: the pose command goes straight to the inner
            # impedance and nothing responds to the measured force.
            v_r = (x_ref - x_r) / dt
            x_r = x_ref.copy()
        else:
            drive = f_d + Ko @ (x_ref - x_r)
            a_r = np.linalg.solve(Ma, drive + f_e_ctrl - Br @ v_r)
            v_r = v_r + a_r * dt
            sp = float(np.linalg.norm(v_r))
            if sp > args.v_limit:
                v_r *= args.v_limit / sp
            x_r = x_r + v_r * dt

        tau, _ = B.geometric_impedance_torque(u.agent, link, pmodel, ee_idx, idx,
                                              gains, x_r, v_r, R_ref, q_mid)
        qf = np.zeros(nq); qf[idx] = tau
        robot.set_qf(qf[None, :])
        u.scene.step()

        ok, at_hole = u.has_peg_inserted()
        if search_s > 0.0 and search["found"] is None and t > args.align_s:
            # caught when the head sits deeper than the entrance face
            if float(at_hole[0, 0]) + half_len > search["catch"]:
                search["found"] = t
                search["catch_p"] = u.peg.pose.p[0].cpu().numpy().copy()
        if on_frame is not None and every > 0 and step % every == 0:
            on_frame(t, float(at_hole[0, 0]), float(np.linalg.norm(f_e)))
        log["t"].append(t)
        log["f"].append(np.linalg.norm(f_e))
        log["f_vec"].append(f_e.copy())
        log["depth"].append(float(at_hole[0, 0]))
        log["peg"].append(u.peg.pose.p[0].cpu().numpy().copy())
        log["tcp"].append(B.get_pose(link)[1].copy())
    out = {k: np.array(v) for k, v in log.items()}
    ok, at_hole = u.has_peg_inserted()
    out["success"] = bool(ok[0])
    out["final_at_hole"] = at_hole[0].cpu().numpy()
    out["search_found_s"] = search["found"] if search_s > 0.0 else None
    out["search_caught"] = bool(search_s > 0.0 and search["catch_p"] is not None
                               and search["found"] < args.align_s + search_s)
    out["face_depth"] = out["depth"] + half_len     # head depth past the face
    out["half_len"] = half_len
    out["hole_half_w"] = float(u.box_hole_radii[0])
    return out


def frame_study(args) -> None:
    """The same label numbers read in different frames, across hole orientations.

    On a wipe this is decisive: the motion is tangential and the force is normal,
    so a motion-aligned rule and a force-aligned rule end up orthogonal, and the
    task tolerates only 25 degrees of frame error.  Insertion is the opposite
    case by construction -- during the push, the motion and the contact force
    BOTH lie along the hole axis -- so those two rules and the correct one
    collapse onto the same frame and only the frame-free conventions (world
    axes, end-effector axes) can be wrong.
    """
    rows = []
    for ep in range(args.episodes):
        s = setup(ep, args.grip_back)
        u = s["u"]
        Rh, _ = hole_frame(u)
        axis = Rh[:, 0]
        R_tcp = u.agent.tcp.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
        lat = args.lateral_mm / 1000.0 * Rh[:, 1]
        frames = {
            "hole axis (= motion = force)": Rh,
            "world axes": np.eye(3),
            "end-effector axes": R_tcp,
        }
        rec = {"ep": ep, "axis": axis}
        for name, F in frames.items():
            K = compose_axial(args.magnitude, args.ratio, F)
            k_ax = float(axis @ K @ axis)
            ang = np.rad2deg(np.arccos(np.clip(abs(float(F[:, 0] @ axis)), 0, 1)))
            r = rollout(s, K, lat, args)
            rec[name] = (r["success"], 1000 * r["depth"][-1], k_ax, ang,
                         float(np.max(r["f"])))
            # restore the start state for the next frame on the same episode
            s["env"].close()
            s = setup(ep, args.grip_back)
            u = s["u"]
        s["env"].close()
        rows.append(rec)
        print(f"  episode {ep+1}/{args.episodes} done")

    names = list(rows[0].keys())[2:]
    print("\n" + "=" * 96)
    print(f"FRAME COMPARISON  (magnitude {args.magnitude:.0f}, ratio {args.ratio:.0f}:1 "
          f"stiff along the first axis, lateral aim error {args.lateral_mm:.1f} mm)")
    print("=" * 96)
    print(f"  {'label frame':<32}{'success':>9}{'axial K':>11}{'stiff axis vs hole':>21}{'peak N':>9}")
    print("  " + "-" * 82)
    for n in names:
        ok = 100.0 * np.mean([r[n][0] for r in rows])
        kax = np.mean([r[n][2] for r in rows])
        ang = np.mean([r[n][3] for r in rows])
        pk = np.mean([r[n][4] for r in rows])
        print(f"  {n:<32}{ok:7.0f}% {kax:10.0f} {ang:16.0f} deg {pk:8.1f}")
    print("  " + "-" * 82)
    print("  On this task the motion-aligned and force-aligned frames ARE the hole frame:")
    print("  the peg moves along the hole axis and the reaction opposes it along the same")
    print("  axis.  The frame ambiguity that decided the wipe cannot arise here.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="one nominal run, verbose")
    ap.add_argument("--frames", action="store_true", help="the frame comparison across seeds")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--magnitude", type=float, default=1200.0)
    ap.add_argument("--ratio", type=float, default=10.0)
    ap.add_argument("--inner-stiffness", type=float, default=3000.0)
    # 120 Nm/rad leaves 5-6 degrees of orientation error, which at a 124 mm peg
    # displaces the head by more than the 3 mm clearance.  400 with the
    # head-aiming correction inserts on every episode tried.
    ap.add_argument("--rot-stiffness", type=float, default=400.0)
    ap.add_argument("--rot-ratio", type=float, default=1.0,
                    help="roll / pivot rotational stiffness; >1 = free to pivot")
    ap.add_argument("--cc-frac", type=float, default=0.0,
                    help="compliance centre, 0 = wrist, 1 = peg tip")
    ap.add_argument("--wrist-inertia", type=float, default=0.005,
                    help="rotational inertia used to match Dr, kg m^2")
    ap.add_argument("--grip-back", type=float, default=0.040,
                    help="how far behind the peg centre the gripper holds it, m")
    ap.add_argument("--f-push", type=float, default=12.0)
    ap.add_argument("--f-limit", type=float, default=30.0)
    ap.add_argument("--v-limit", type=float, default=0.06)
    # The box spans +-length along the hole axis (depth = length in
    # _build_box_with_hole), so its near face is one peg half-length before the
    # hole centre.  A 55 mm stand-off put the peg head INSIDE the box: the align
    # phase drove it into the face at 53 N and knocked the peg 44 mm out of the
    # grasp before insertion had even started.
    ap.add_argument("--standoff-margin", type=float, default=0.035,
                    help="clearance ahead of the box face; the stand-off is "
                         "peg_half_length + this")
    ap.add_argument("--retarget", action="store_true", default=True)
    ap.add_argument("--no-retarget", dest="retarget", action="store_false")
    ap.add_argument("--overdrive", type=float, default=0.010)
    ap.add_argument("--align-s", type=float, default=4.0)
    ap.add_argument("--insert-s", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--lateral-mm", type=float, default=0.0)
    ap.add_argument("--search-s", type=float, default=0.0,
                    help="seconds of spiral search on the face before pushing "
                         "(0 = the original aim-and-push)")
    ap.add_argument("--search-r0", type=float, default=0.001, help="m, start radius")
    ap.add_argument("--search-growth", type=float, default=0.012, help="m added over the search")
    ap.add_argument("--search-turns", type=float, default=5.0)
    ap.add_argument("--search-press", type=float, default=8.0, help="N onto the face")
    ap.add_argument("--search-depth", type=float, default=0.003,
                    help="m the aim sits past the face, so the peg stays pressed")
    ap.add_argument("--search-catch", type=float, default=0.005,
                    help="m past the entrance face that counts as having dropped in")
    args = ap.parse_args()

    s = setup(args.seed, args.grip_back)
    u = s["u"]
    Rh, hole_p = hole_frame(u)
    goal_peg = u.goal_pose.p[0].cpu().numpy()
    peg_p = u.peg.pose.p[0].cpu().numpy()
    print(f"hole centre {hole_p.round(4)}   insertion axis {Rh[:, 0].round(3)}")
    print(f"peg (gripped) {peg_p.round(4)}   goal peg centre {goal_peg.round(4)}")
    print(f"distance to go {1000*np.linalg.norm(goal_peg - peg_p):.1f} mm, "
          f"hole clearance {1000*u._clearance:.1f} mm")
    print(f"peg axis vs insertion axis: "
          f"{np.rad2deg(np.arccos(abs(float(u.peg.pose.to_transformation_matrix()[0,:3,0].cpu().numpy() @ Rh[:,0])))):.1f} deg")
    if args.frames:
        frame_study(args); s["env"].close(); return
    Rt2 = Rh
    lat = args.lateral_mm / 1000.0 * Rh[:, 1]
    Ko = compose_axial(args.magnitude, args.ratio, Rh)
    print(f"\nnominal run: hole frame, magnitude {args.magnitude:.0f}, ratio {args.ratio:.0f}:1, "
          f"lateral aim error {args.lateral_mm:.1f} mm")
    r = rollout(s, Ko, lat, args)
    d = r["final_at_hole"]
    print(f"  success: {r['success']}   peg head at hole (x,y,z) = "
          f"{(1000*d).round(1)} mm   (x >= -15 mm and |y|,|z| <= "
          f"{1000*float(u.box_hole_radii[0]):.1f} mm)")
    print(f"  peak contact force {r['f'].max():.1f} N, mean during insert "
          f"{r['f'][r['t'] >= args.align_s].mean():.1f} N")
    print(f"  insertion depth: {1000*r['depth'][-1]:.1f} mm "
          f"(started {1000*r['depth'][0]:.1f} mm)")
    s["env"].close()


if __name__ == "__main__":
    main()
