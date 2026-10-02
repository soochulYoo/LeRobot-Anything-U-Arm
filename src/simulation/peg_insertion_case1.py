"""PegInsertionSide under the Case 1 controller, with a scrubbing search.

The cascade version of this study (peg_insertion_cascade.py) is kept for later.
This one runs the same insertion under the controller the rest of the project
uses -- geometric impedance, energy tank, common power gate -- so the peg task
can be compared with writing and wiping on equal terms.

Three things change, and they are not cosmetic:

  * Case 1's proposal is (Vd, Up, Ur) and has NO active-wrench slot, by design.
    The cascade pushed with a commanded f_d; here the press has to come from
    geometry, by sinking x_d past the entrance face and letting K convert the
    gap into force.  k_axial * search_depth IS the press, so the two knobs are
    no longer independent.
  * K is bounded (k_lo..k_hi) and must stay SPD, so the axial:lateral
    anisotropy is capped: the cascade's magnitude 1200 / ratio 10 wants a 5565
    N/m eigenvalue, past k_hi.  Stiffness is named by its eigenvalues here
    instead, in the hole frame.
  * Nothing reads the measured force.  There is no admittance, so the 30 N
    clip the cascade applied to its force feedback is gone and the reported
    contact force is the real one.  The peg pose IS read, for the head
    conversion and the sag calibration -- that is kinematics, not force
    feedback, and a policy would get it from vision.

What the gate does to a search is the interesting part.  Scrubbing drags the
reference against a contact for seconds at a time, every bit of which is
metered as active power, so the tank is a budget for searching.  alpha and E
are logged for that reason.  Measured: a 13 s episode at an 8 N press draws
about 5 J of the 20 J tank and alpha never leaves 1.0, so searching is
affordable -- the tank is not what limits it.

Four things the first version of this got wrong, all of them specific to
replacing the cascade with a pure impedance:

  * THE PEG'S OWN WEIGHT.  ManiSkill disables link gravity for a fixed-base
    arm, so the arm is weightless, but the grasped peg is not: ~0.54 kg hanging
    off the hand sags it 13.8 mm on a 300 N/m lateral axis, against 3 mm of
    clearance.  The cascade hid this behind a 3000 N/m isotropic inner loop.
    Stiffening it away would throw out the lateral compliance the task needs,
    so the sag is CALIBRATED instead: the reference holds at the stand-off with
    the search stiffness already applied and integrates the head error into a
    frozen bias.  That correction is a reference motion, so it travels in Vd
    and the gate meters it like everything else.
  * A PHASE BOUNDARY IS A STEP.  Jumping x_d 32 mm from the stand-off to the
    face in one 2 ms tick asks for Vd = 16 m/s, and the damping feed-forward
    u = D Vd then demands 2129 Nm against an 87 Nm limit and slams the peg into
    the face at 65 N.  Every target now ramps, and |Vd| is clipped.
  * A PURE IMPEDANCE CANNOT PASS ITS OWN REFERENCE.  The catch test asked for
    5 mm of insertion while the reference only sank 3 mm past the face, so it
    could never fire.  The press is k_axial * depth, so the search needs a SOFT
    axial spring with a DEEP reference (800 N/m x 10 mm = 8 N) and the
    insertion a stiff one.  That is a per-phase stiffness, which is what this
    task is supposed to be about.
  * AIMING IS NOT COMMANDING.  x_d names the TCP; success is judged on the peg
    HEAD, 107 mm away through a grasp that sits 4.5 degrees off the hole axis.
    The schedule is written in head coordinates and converted through the
    MEASURED peg axis each step.

One honest exception: the proposal has no rotational reference velocity, so
there is no gated way to turn the peg from its grasp attitude (4.5 degrees off
the hole axis) onto the hole.  R_d is slerped directly during the approach,
which does unmetered work on the rotational spring.  It finishes in free space
before any contact, and R_d is then frozen for the whole scrub and insertion,
so the contact phase is strictly Case 1.

Usage:
    python3 peg_insertion_case1.py --episodes 4 --search-s 4
    python3 peg_insertion_case1.py --check          # one verbose episode
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import peg_insertion_cascade as P        # env build + grasp, reused as-is
import writing.controller as C


# --------------------------------------------------------------------------- #
def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation, as the log map on SO(3)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2.0 * np.sin(th)) * C.vee(R - R.T)


def expm_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def slerp_R(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    return R0 @ expm_so3(a * rotvec(R0.T @ R1))


def smooth(a: float) -> float:
    """Cosine ease on [0, 1], so a phase boundary is not a step in x_d."""
    a = min(1.0, max(0.0, a))
    return 0.5 - 0.5 * np.cos(np.pi * a)


def hole_K(k_axial: float, k_lat: float, Rh: np.ndarray) -> np.ndarray:
    """World K with the hole axis stiff and the two lateral axes soft.

    Named by eigenvalues, not by a magnitude and a ratio: Case 1 clips K's
    eigenvalues into [k_lo, k_hi] and a geometric-mean parameterisation walks
    straight out of that box.
    """
    return Rh @ np.diag([k_axial, k_lat, k_lat]) @ Rh.T


# --------------------------------------------------------------------------- #
def setup(seed: int, args, render_mode: str | None = None, cam: dict | None = None):
    """The cascade's env and grasp, with a Case 1 controller on top."""
    cl = float(getattr(args, "clearance_mm", 0.0))
    if cl > 0.0:
        # The task fixes the gap at 3 mm, which on a 15-25 mm half-width peg is
        # 12-20% -- loose by the standards of precision assembly.  The class
        # attribute is read during _load_scene, so it has to be set before the
        # env is built.
        from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
        PegInsertionSideEnv._clearance = cl * 1e-3
    s = P.setup(seed, grip_back=args.grip_back, render_mode=render_mode, cam=cam)
    u, robot = s["u"], s["robot"]
    nq = s["nq"]

    g = C.Case1Gains()
    g.Kr = args.kr
    g.zeta = args.zeta
    g.tank = not args.no_tank
    g.E0 = args.E0
    g.k_hi = max(g.k_hi, args.k_axial)
    g.k_lo = min(g.k_lo, args.k_lat)
    # The fingers are not part of the impedance.  compute() applies set_qf to
    # every active joint, and a zero limit on the two finger entries is the
    # cleanest way to keep the grasp's own PD drive the only thing holding the
    # peg -- P.setup leaves those drives on deliberately, which is also why
    # disable_joint_drives() must not be called here: it would open the hand.
    g.tau_limit = np.concatenate([C.PANDA_TAU_LIMIT, np.zeros(nq - 7)])
    ctl = C.Case1Controller(robot, g, u.agent.urdf_path, tcp_name=u.agent.tcp.name)

    s["ctl"] = ctl
    s["dt"] = 1.0 / u.sim_freq
    return s


PHASES = ("travel", "calibrate", "touch", "search", "insert")


def levels_for(phase: str, args) -> tuple[float, float, float]:
    """(k_axial, k_lateral, K_R) for a phase.  This is the protocol table that
    ../writing/protocol.py and ../wiping/protocol.py express as LOW/MID/HIGH;
    here it is still a scripted default, because the numbers that belong in it
    have not been measured yet -- see PEG_DEMO_PLAN.md."""
    if getattr(args, "no_schedule", False):
        # ONE stiffness for the whole episode.  Without this arm the sweeps
        # compare levels OF a schedule against each other and can say nothing
        # about whether the schedule is needed -- which is the question the
        # plan's gate actually asks.
        return args.k_axial_search, args.k_lat, args.kr_search
    if phase == "travel":
        return args.k_axial, args.k_lat_hold, args.kr_hold
    if phase in ("calibrate", "touch", "search"):
        return args.k_axial_search, args.k_lat, args.kr_search
    return args.k_axial, args.k_lat, args.kr_search      # insert


def rollout(s, args, lateral_err: np.ndarray, on_frame=None, every: int = 0) -> dict:
    """One attempt: travel, calibrate the sag, touch the face, scrub, push home.

    `lateral_err` is what the script BELIEVES about the hole, added to the aim,
    so the search has something to find -- with a perfect aim the peg drops in
    whatever the motion is.  Nothing in the schedule reads the true hole
    position; `has_peg_inserted` is only the scorer.
    """
    u, ctl, dt = s["u"], s["ctl"], s["dt"]
    Rh, hole_p = P.hole_frame(u)
    axis = Rh[:, 0]
    e1, e2 = Rh[:, 1], Rh[:, 2]
    half_len = float(u.peg_half_sizes[0, 0])

    R_tcp0, p_tcp0, _, _ = ctl.tip_state()
    peg_p0 = u.peg.pose.p[0].cpu().numpy()
    peg_in_tcp = R_tcp0.T @ (peg_p0 - p_tcp0)
    R_aim = P.align_axis(R_tcp0, R_tcp0[:, 0], axis)

    def peg_axis() -> np.ndarray:
        R = u.peg.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
        a = R[:, 0]
        return a if a @ axis > 0 else -a

    def head_now() -> np.ndarray:
        return u.peg.pose.p[0].cpu().numpy() + half_len * peg_axis()

    # The schedule lives in HEAD coordinates, measured from where the script
    # believes the hole is.  hole_frame's origin is the box centre and the box
    # is as deep as the peg is long, so the entrance face is half_len in front
    # of it and success wants the head within 15 mm of it.
    believed = hole_p + lateral_err
    head0 = head_now()
    head_face = believed - half_len * axis
    head_pre = head_face - args.standoff_margin * axis
    head_press = head_face + args.search_depth * axis
    head_goal = believed + args.overdrive * axis

    K_tgt = hole_K(*levels_for("travel", args)[:2], Rh)
    kr_tgt = levels_for("travel", args)[2]
    ctl.reset_state(p_tcp0, R_tcp0, K_tgt, q_rest=s["robot"].get_qpos()[0].cpu().numpy(),
                    kr=kr_tgt)

    t_cal = args.align_s
    t_touch = t_cal + args.cal_s
    t_scrub = t_touch + args.touch_s
    t_push = t_scrub + args.search_s

    bias = np.zeros(3)
    search = dict(found=None, catch_p=None)
    log = {k: [] for k in ["t", "f", "depth", "alpha", "E", "kr", "k_ax",
                           "gap", "dtau", "vd"]}
    n = int(args.duration / dt)
    phase = "travel"
    for step in range(n):
        t = step * dt
        if t < t_cal:
            phase = "travel"
            a = min(1.0, t / (0.7 * args.align_s))
            ctl.R_d = slerp_R(R_tcp0, R_aim, a)       # free space, see docstring
            head_t = head0 + smooth(t / args.align_s) * (head_pre - head0)
        elif t < t_touch:
            # hold at the stand-off with the SEARCH stiffness already applied
            # and learn how far the peg hangs below where it is commanded.
            phase = "calibrate"
            head_t = head_pre
            # The error to integrate is DESIRED minus ACTUAL.  Including `bias`
            # in it integrates the tracking error instead, which stays equal to
            # the sag however large the bias grows, so the loop never closes --
            # it wound straight to its clip at 60 mm.
            bias = bias + args.cal_gain * (head_t - head_now()) * dt
            bias = np.clip(bias, -args.bias_max, args.bias_max)
        elif t < t_scrub:
            phase = "touch"
            head_t = head_pre + smooth((t - t_touch) / args.touch_s) * (head_press - head_pre)
        elif args.search_s > 0.0 and search["found"] is None and t < t_push:
            phase = "search"
            a = (t - t_scrub) / args.search_s
            rad = args.search_r0 + args.search_growth * a
            th = 2.0 * np.pi * args.search_turns * a
            head_t = head_press + rad * (np.cos(th) * e1 + np.sin(th) * e2)
        else:
            phase = "insert"
            if search["found"] is None:                 # timed out, or no search
                search["found"] = t
                search["catch_p"] = head_now()
            a = min(1.0, (t - search["found"]) / args.insert_s)
            p0 = search["catch_p"]
            head_t = p0 + a * float((head_goal - p0) @ axis) * axis

        # ---- stiffness as a RATE, ramped: this is the Case 1 action ----------
        k_ax, k_lat, kr = levels_for(phase, args)
        Up = (hole_K(k_ax, k_lat, Rh) - ctl.K) / args.ramp
        Ur = (kr - ctl.kr) / args.ramp

        # ---- head command -> TCP reference ----------------------------------
        centre_t = head_t + bias - half_len * peg_axis()
        x_t = centre_t - ctl.R_d @ peg_in_tcp
        Vd = (x_t - ctl.x_d) / dt
        sp = float(np.linalg.norm(Vd))
        if sp > args.v_limit:                 # a demo does not ask for 16 m/s
            Vd *= args.v_limit / sp

        rec = ctl.compute(C.Case1Proposal(Vd, Up=Up, Ur=Ur), dt)
        u.scene.step()
        ctl.advance()

        f_e = u.scene.get_pairwise_contact_forces(u.peg, u.box)[0].cpu().numpy()
        ok, at_hole = u.has_peg_inserted()
        face_d = float(at_hole[0, 0]) + half_len
        if args.search_s > 0.0 and search["found"] is None and t > t_scrub:
            if face_d > args.search_catch:
                search["found"] = t
                search["catch_p"] = head_now()
        if on_frame is not None and every > 0 and step % every == 0:
            on_frame(t, face_d, float(np.linalg.norm(f_e)))
        log["t"].append(t)
        log["f"].append(float(np.linalg.norm(f_e)))
        log["depth"].append(face_d)
        log["alpha"].append(rec["alpha"])
        log["E"].append(rec["E"])
        log["kr"].append(rec["kr"])
        log["k_ax"].append(float(axis @ ctl.K @ axis))
        log["gap"].append(float(np.linalg.norm(rec["p"] - rec["x_d"])))
        log["dtau"].append(float(np.abs(rec["d_tau"]).max()))
        log["vd"].append(sp)

    out = {k: np.array(v) for k, v in log.items()}
    ok, at_hole = u.has_peg_inserted()
    out["success"] = bool(ok[0])
    out["caught"] = bool(args.search_s > 0.0 and search["catch_p"] is not None
                         and search["found"] < t_push)
    out["found_s"] = None if search["found"] is None else search["found"] - t_scrub
    out["half_len"] = half_len
    out["hole_half_w"] = float(u.box_hole_radii[0])
    out["clearance"] = out["hole_half_w"] - float(u.peg_half_sizes[0, 1])
    out["deepest"] = float(out["depth"].max())
    out["bias_mm"] = float(np.linalg.norm(bias) * 1e3)
    out["bias_clipped"] = bool(np.any(np.abs(bias) >= args.bias_max - 1e-9))
    return out


def episode(seed: int, err_mm: float, args) -> dict:
    s = setup(seed, args)
    Rh, _ = P.hole_frame(s["u"])
    rng = np.random.default_rng(1000 + seed)
    d = rng.normal(size=2)
    d /= max(np.linalg.norm(d), 1e-9)
    lat = 1e-3 * err_mm * (d[0] * Rh[:, 1] + d[1] * Rh[:, 2])
    r = rollout(s, args, lat)
    s["env"].close()
    return r


# --------------------------------------------------------------------------- #
def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--errors", default="0,2,4,6,8,10", help="mm of lateral aim error")
    ap.add_argument("--check", action="store_true", help="one verbose episode")
    ap.add_argument("--per-seed", action="store_true",
                    help="print every episode's outcome, to find a contrasting seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--err", type=float, default=4.0, help="--check aim error, mm")
    # stiffness, as eigenvalues in the hole frame
    ap.add_argument("--k-axial", type=float, default=3000.0,
                    help="N/m along the hole, for the push")
    ap.add_argument("--k-axial-search", type=float, default=800.0,
                    help="N/m along the hole while scrubbing; times --search-depth "
                         "this IS the press, and it must be soft enough that the "
                         "reference can sit deeper than --search-catch")
    ap.add_argument("--k-lat", type=float, default=300.0, help="N/m across the hole")
    ap.add_argument("--k-lat-hold", type=float, default=1500.0,
                    help="N/m across it while travelling, to hold the aim")
    ap.add_argument("--kr", type=float, default=80.0, help="Nm/rad, the GIC gain K_R")
    ap.add_argument("--kr-hold", type=float, default=80.0, help="Nm/rad while travelling")
    ap.add_argument("--kr-search", type=float, default=20.0,
                    help="Nm/rad once in contact, so the peg can pivot in")
    ap.add_argument("--no-schedule", action="store_true",
                    help="hold one (k_axial_search, k_lat, kr_search) for every "
                         "phase, instead of switching per phase")
    ap.add_argument("--ramp", type=float, default=0.15,
                    help="s, stiffness-rate time constant; the tank pays for it")
    ap.add_argument("--v-limit", type=float, default=0.10, help="m/s on |Vd|")
    ap.add_argument("--cal-s", type=float, default=1.5,
                    help="s held at the stand-off to calibrate the peg's sag")
    ap.add_argument("--cal-gain", type=float, default=4.0, help="1/s")
    ap.add_argument("--bias-max", type=float, default=0.030,
                    help="m, clip on the sag correction; hitting it means the "
                         "lateral stiffness is too soft for the payload")
    ap.add_argument("--touch-s", type=float, default=3.0,
                    help="s ramping onto the face.  writing/protocol.py found a "
                         "15-20 mm/s descent lands hard enough to matter and "
                         "slowed to 10 mm/s; 45 mm in 3 s is the same lesson")
    ap.add_argument("--zeta", type=float, default=0.8)
    # energy tank
    ap.add_argument("--E0", type=float, default=20.0, help="J")
    ap.add_argument("--no-tank", action="store_true", help="ungate, for comparison")
    # schedule
    ap.add_argument("--align-s", type=float, default=3.0)
    ap.add_argument("--search-s", type=float, default=4.0, help="0 = aim and push")
    ap.add_argument("--insert-s", type=float, default=4.0)
    ap.add_argument("--duration", type=float, default=19.0)
    ap.add_argument("--grip-back", type=float, default=0.040)
    ap.add_argument("--clearance-mm", type=float, default=0.0,
                    help="override the task's 3 mm gap; 0 keeps it")
    ap.add_argument("--standoff-margin", type=float, default=0.035)
    ap.add_argument("--overdrive", type=float, default=0.010)
    # the spiral
    ap.add_argument("--search-r0", type=float, default=0.001)
    ap.add_argument("--search-growth", type=float, default=0.012)
    ap.add_argument("--search-turns", type=float, default=5.0)
    ap.add_argument("--search-depth", type=float, default=0.010,
                    help="m the reference sinks past the face; times "
                         "--k-axial-search this IS the press")
    ap.add_argument("--search-catch", type=float, default=0.005)
    # Step 2 needs to vary the axial stiffness without varying the press, and
    # the press IS k_axial_search * search_depth, so one of the two has to be
    # derived.  --press does that, and --catch-frac keeps the catch threshold
    # inside the commanded depth as the depth shrinks: a pure impedance cannot
    # pass its own reference, so a fixed 5 mm catch becomes unreachable the
    # moment the press is held and the stiffness is raised (8 N at 3000 N/m is
    # 2.7 mm of depth).
    ap.add_argument("--press", type=float, default=0.0,
                    help="N; >0 derives --search-depth as press/k_axial_search")
    ap.add_argument("--catch-frac", type=float, default=0.0,
                    help=">0 derives --search-catch as this fraction of the depth")


def resolve(args) -> None:
    """Apply the derived knobs, so every caller sees one consistent set."""
    if args.press > 0.0:
        args.search_depth = args.press / args.k_axial_search
    if args.catch_frac > 0.0:
        args.search_catch = args.catch_frac * args.search_depth


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    args = ap.parse_args()
    resolve(args)

    press = args.k_axial_search * args.search_depth
    print(f"Case 1 | search K diag({args.k_axial_search:.0f}, {args.k_lat:.0f}) "
          f"-> push K diag({args.k_axial:.0f}, {args.k_lat:.0f}) N/m, K_R "
          f"{args.kr_hold:.0f} -> {args.kr_search:.0f} Nm/rad, ramp "
          f"{args.ramp:.2f} s | tank {'off' if args.no_tank else f'{args.E0:.0f} J'}"
          f" | press {press:.1f} N over {args.search_depth*1e3:.1f} mm, catch "
          f"{args.search_catch*1e3:.1f} mm | spiral {args.search_r0*1e3:.0f}.."
          f"{(args.search_r0 + args.search_growth)*1e3:.0f} mm, "
          f"{args.search_turns:.0f} turns in {args.search_s:.0f} s")

    if args.check:
        r = episode(args.seed, args.err, args)
        print(f"peg half-length {r['half_len']*1e3:.1f} mm, aperture half-width "
              f"{r['hole_half_w']*1e3:.1f} mm (3 mm clearance, no chamfer)")
        print(f"aim error {args.err:.1f} mm -> caught={r['caught']} at "
              f"{r['found_s']}, success={r['success']}, deepest "
              f"{r['deepest']*1e3:.1f} mm, sag bias {r['bias_mm']:.1f} mm"
              f"{' (CLIPPED)' if r['bias_clipped'] else ''}")
        t = r["t"]
        print(f"{'t':>8} {'depth':>9} {'|f|':>12} {'gap':>8} {'k_ax':>6} "
              f"{'kr':>5} {'alpha':>6} {'E':>6}")
        for lo in range(int(t[-1]) + 1):
            m = (t >= lo) & (t < lo + 1)
            if not m.any():
                continue
            print(f"{lo:4d}-{lo+1:<3d} {r['depth'][m].max()*1e3:8.1f}mm "
                  f"{r['f'][m].mean():6.1f}/{r['f'][m].max():<5.1f}N "
                  f"{r['gap'][m].max()*1e3:6.1f}mm {r['k_ax'][m].mean():6.0f} "
                  f"{r['kr'][m].mean():5.0f} {r['alpha'][m].min():6.2f} "
                  f"{r['E'][m].min():6.2f}")
        if r["dtau"].max() > 1e-6:
            print(f"torque saturated: max |d_tau| {r['dtau'].max():.1f} Nm")
        return

    n = args.episodes
    probe = episode(0, 0.0, argparse.Namespace(**{**vars(args), "duration": 0.02}))
    print(f"realised clearance {probe['clearance']*1e3:.2f} mm, spiral pitch "
          f"{1e3*args.search_growth/max(args.search_turns, 1e-9):.2f} mm/turn "
          f"({'inside' if args.search_growth / max(args.search_turns, 1e-9) <= probe['clearance'] else 'WIDER THAN'} "
          f"the capture window)")
    print(f"\n{n} seeds per cell. in = inserted, jam = past the face but stuck, "
          f"miss = never found it\n")
    print(f"{'aim err':>8} {'in':>5} {'jam':>5} {'miss':>5} {'Fpk':>7} "
          f"{'caught':>14} {'alpha_min':>10} {'E_end':>7}")
    for err in [float(x) for x in args.errors.split(",")]:
        ins = jam = miss = 0
        fpk, found, amin, eend = [], [], [], []
        for seed in range(n):
            r = episode(seed, err, args)
            if args.per_seed:
                how = ("in" if r["success"]
                       else "jam" if r["deepest"] > 0.005 else "miss")
                when = "-" if not r["caught"] else f"{r['found_s']:.1f} s"
                print(f"  err {err:4.0f} mm  seed {seed}  search_s "
                      f"{args.search_s:.0f}  -> {how:4s}  deepest "
                      f"{r['deepest'] * 1e3:6.1f} mm  Fpk {r['f'].max():5.1f} N"
                      f"  caught {when}", flush=True)
            fpk.append(r["f"].max())
            amin.append(r["alpha"].min())
            eend.append(r["E"][-1])
            if r["success"]:
                ins += 1
            elif r["deepest"] > 0.005:
                jam += 1
            else:
                miss += 1
            if r["caught"]:
                found.append(r["found_s"])
        when = (f"{np.mean(found):5.1f} s {len(found)}/{n}" if found
                else f"{'none':>9}")
        print(f"{err:7.0f} {ins:5d} {jam:5d} {miss:5d} {np.mean(fpk):6.1f}N "
              f"{when:>14} {np.mean(amin):10.2f} {np.mean(eend):6.1f}J",
              flush=True)


if __name__ == "__main__":
    main()
