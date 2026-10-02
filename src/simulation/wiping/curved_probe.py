"""Step 1 of CURVED_BOARD.md: can OUR controller yield to K_R at all?

The wiping K_R comparison on a tilted flat board measured 9.9 deg of pad
misalignment at K_R = 60 and 8.8 deg at K_R = 0.3.  Two readings of that:

  THE SURFACE.  A tilted plane has a constant normal, so rotational compliance
  never has to work (../curved/surface.py).  Then the fix is a curved board.

  THE CONTROLLER.  Even a constant 8 deg wedge is something a compliant wrist
  should lay flat on, and ours did not: 8.8 deg of misalignment at K_R = 0.3
  means the pad kept the orientation it was commanded.  Then no amount of
  surface work will fix anything.

This file separates them before any of the canvas rework in CURVED_BOARD.md
gets written.  The pad is ours (`wipe_scene.PandaEraser`), the controller is
ours (`writing/controller.Case1Controller`, tank, null-space and torque
saturation included), the slab is `../curved/surface.CurvedSurface`, and the
motion is ONE STRAIGHT TRAVERSE.  No glyph, no marks, no erasure model: the
only outputs are how flush the pad lay and how far it moved off its command.

Deliberately standalone rather than a `WipingSim` subclass.  Everything
`WipingSim` adds over this -- the canvas box, the ink dots, the marks, the work
integral, the scoring -- is exactly what step 1 is supposed to remove, and the
pieces that do matter (the two force filters, the pen-down convention, the
start pose) are small enough to restate here with the original in view.

THE COMMANDED ORIENTATION IS ALWAYS VERTICAL (`R_PEN_DOWN`).  The controller is
never told the board tilts or curves; that is the whole point.  Translation is
given the surface height as an oracle (`--press-mode follow`) so a height error
cannot masquerade as a rotation effect: on this probe K_R is the ONLY thing
between the pad and the surface.  `--press-mode flat` commands a constant
height instead, which is what the real task faces.

WHAT SETS THE SCALE OF K_R.  A pad of half-width r pressing with f N on a
wedge can generate at most about f*r of contact moment, so an impedance of
K_R Nm/rad yields f*r/K_R rad and no more.  Our eraser is r = 15 mm at ~3.5 N:
0.05 Nm, so K_R has to be below ~0.3 Nm/rad before 9 deg of yield is even
available.  ../curved/wipe.py had a 25 mm pad at 8 N -- 0.2 Nm, four times more
-- which is why K_R = 3 was already at the knee there and is rigid here.  The
sweep therefore runs K_R down to 0.03, and each row prints the yield this
estimate predicts next to the one that was measured.

WHAT IT FOUND (tables and the revised plan in CURVED_BOARD.md).  Both readings
above were wrong, and so was the mis-set-up they rested on.  Two bugs: the
eraser agent shipped no SRDF, so an unfilterable panda_hand/panda_link7
self-collision pinned the wrist, and a sharp-edged box pad catches on the
internal edges of a triangle mesh, which stalls the traverse outright.  With
those fixed, K_R separates on BOTH surfaces -- 8.8 -> 4.0 deg of residual
misalignment on the curve, 8.0 -> 4.1 on a plain 8 deg tilt -- and it separates
only below K_R = 1, where f*r/K_R says it should.  The tank, the null-space
term and the torque limits were ablated and change nothing (alpha = 1.00 and
0% saturation in every run).  So a tilted plane DOES exercise rotational
compliance; what it cannot do is ask the pad to keep turning, or to lie flush
on a normal that swings across its own width.

Usage:
    python3 curved_probe.py                       # curved slab, K_R sweep
    python3 curved_probe.py --amp 0 --tilt 8      # the flat tilted control
    python3 curved_probe.py --press-mode flat     # what the task commands today
    python3 curved_probe.py --no-tank --null-kp 0 --null-kd 0 --tau-scale 100
                                                  # suspects switched off
"""
from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))
# APPEND for the sibling study: `src/simulation` holds a vendored `mani_skill/`
# that would shadow the installed one if it reached the front of sys.path
# (../curved/record.py was bitten by exactly this).
sys.path.append(str(HERE.parent / "curved"))

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
import torch  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from transforms3d.quaternions import mat2quat  # noqa: E402

import controller as C  # noqa: E402  writing/controller.py
import scene as W  # noqa: E402  writing/scene.py, for CANVAS_AXES / rot_xy
import wipe_scene as SC  # noqa: E402  our eraser agent and env
from surface import CurvedSurface  # noqa: E402  ../curved/surface.py

Array = np.ndarray

# pen straight down: tool z along -world z.  Same constant as writing/sim.py,
# restated rather than imported so this probe does not pull in the ink rule.
R_PEN_DOWN = np.diag([1.0, -1.0, -1.0])
# The slab's top surface replaces the paper, at the paper's nominal height.
SURF_POS = W.TeleopWritingEnv.CANVAS_CENTER.copy()
# Shared with ../curved/wipe.py, so a mesh exported there is not re-exported.
SCRATCH = pathlib.Path(os.environ.get("CLAUDE_SCRATCH", "/tmp")) / "curved_meshes"
# The two filters writing/sim.py reads force through: the 25 Hz one is what a
# sensor reports, the 5 Hz one is the sustained pressure that task decides on.
SENSOR_HZ, PRESSURE_HZ = 25.0, 5.0


class Slab:
    """The curved slab placed in the world: a pose plus the closed-form field.

    Height and normal come from `CurvedSurface`, never from the exported mesh:
    alignment is measured against the true surface, so a facet of the triangle
    mesh cannot show up as a tilt the pad failed to follow.
    """

    def __init__(self, surf: CurvedSurface, R: Array):
        self.surf, self.R = surf, np.asarray(R, dtype=float)

    def point(self, a: float, b: float, lift: float = 0.0) -> Array:
        """World point `lift` above the surface at slab coordinates (a, b)."""
        h = float(self.surf.height(a, b))
        return SURF_POS + self.R @ np.array([a, b, h + lift])

    def local(self, p: Array) -> Array:
        return self.R.T @ (np.asarray(p, dtype=float) - SURF_POS)

    def normal_at(self, p: Array) -> Array:
        """Outward world normal of the surface under the world point `p`."""
        l = self.local(p)
        return self.R @ self.surf.normal(l[0], l[1])

    def swing_at(self, p: Array, r: float) -> float:
        """How much the normal turns across a pad of half-width r, degrees.

        The quantity that decides whether a RIGID pad can lie flat here at all.
        """
        l = self.local(p)
        n0 = self.surf.normal(l[0], l[1])
        worst = 0.0
        for dx, dy in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
            nk = self.surf.normal(l[0] + dx, l[1] + dy)
            worst = max(worst, float(np.degrees(np.arccos(np.clip(n0 @ nk, -1, 1)))))
        return worst


def demand(slab: Slab, a0: float, b0: float, stroke: float, pad_r: float,
           n: int = 241) -> dict:
    """What the COMMANDED PATH asks of the pad, before anything is simulated.

    `surf.stats` averages over the whole patch, which says nothing about the
    one line the pad actually drives: the first traverse tried here crossed a
    flat valley and asked for 2 deg where the patch average was 6.  A sweep
    over a path with no demand cannot separate on K_R no matter what K_R does,
    so the demand is printed with the result.
    """
    b = b0 + np.linspace(0.0, stroke, n)
    up = np.array([0.0, 0.0, 1.0])
    ask, swing = [], []
    for bb in b:
        p = slab.point(a0, bb)
        nl = slab.normal_at(p)
        ask.append(np.degrees(np.arccos(np.clip(nl @ up, -1, 1))))
        swing.append(slab.swing_at(p, pad_r))
    ask, swing = np.asarray(ask), np.asarray(swing)
    return {"ask_mean": ask.mean(), "ask_p95": np.percentile(ask, 95),
            "ask_max": ask.max(), "swing_mean": swing.mean(),
            "swing_p95": np.percentile(swing, 95),
            "h_ptp": 1000 * float(np.ptp(
                [slab.local(slab.point(a0, bb))[2] for bb in b]))}


def build_scene(surf: CurvedSurface, R_s: Array, friction: float,
                render_mode: str | None = None):
    """The wiping scene with the paper swapped for the slab.

    The canvas box is not deleted but parked below the floor: it is kinematic
    and `place_canvas` is never called here, so moving it is enough, and the
    env keeps working for anything that still expects `u.canvas` to exist.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"surface_{surf.key()}.obj"
    if not path.exists():
        surf.mesh().export(path)

    old = SC.TeleopWipingEnv._load_scene

    def patched(self, options):
        old(self, options)
        mat = sapien.physx.PhysxMaterial(friction, friction, 0.0)
        b = self.scene.create_actor_builder()
        # Static + nonconvex, as ../curved/wipe.py:build_scene: a convex hull of
        # this mesh is a dome and erases every dent, which is half the geometry.
        b.add_nonconvex_collision_from_file(str(path), material=mat)
        b.add_visual_from_file(str(path))
        b.initial_pose = sapien.Pose(p=SURF_POS, q=mat2quat(np.asarray(R_s, float)))
        self.surface = b.build_static(name="curved_surface")

    SC.TeleopWipingEnv._load_scene = patched
    try:
        env = gym.make("TeleopWiping-v1", num_envs=1, sim_backend="cpu",
                       obs_mode="state", wrist_camera=False, render_mode=render_mode)
        env.reset(seed=0)
    finally:
        SC.TeleopWipingEnv._load_scene = old

    u = env.unwrapped
    u.canvas.set_pose(sapien.Pose(p=[0.0, 0.0, -5.0]))
    u.hide_all_dots()
    return env


def rot_inertia(ctl: C.Case1Controller) -> Array:
    """Rotational block of the operational-space inertia at the tip, BODY frame.

    The frame matters: the controller damps `V_b[3:]`, the body angular
    velocity, so this is the inertia its Dr actually sees.  Compared against
    `Case1Gains.wrist_inertia`, which is a guess, it says what the rotational
    damping ratio really is -- ../curved/wipe.py found a hand-guessed scalar
    put it at 0.15 and the wrist rang through the whole traverse.
    """
    q = ctl.robot.get_qpos()[0].cpu().numpy()
    J_b = ctl.pm.compute_single_link_local_jacobian(q, ctl.ee)
    M = ctl.pm.compute_generalized_mass_matrix(q)
    L6 = np.linalg.inv(J_b @ np.linalg.solve(M, J_b.T))
    return 0.5 * (L6[3:, 3:] + L6[3:, 3:].T)


def run(env, slab: Slab, Kr: float, a, report: bool = False) -> dict:
    """One straight traverse at rotational stiffness `Kr`.

    Everything except Kr is identical between conditions: the same start pose,
    the same translational stiffness, the same commanded path, the same
    vertical orientation command.
    """
    u = env.unwrapped
    robot = u.agent.robot
    dt = 1.0 / u.sim_freq
    # The pad is its own link, so pad/slab contact is told apart from the hand
    # hitting the slab -- the same reason the pen is its own link.
    pad = sapien_utils.get_obj_by_name(robot.get_links(), "eraser")

    gains = C.Case1Gains(Kr=float(Kr), tank=not a.no_tank,
                         null_kp=a.null_kp, null_kd=a.null_kd,
                         wrist_inertia=a.wrist_inertia, zeta=a.zeta)
    gains.tau_limit = gains.tau_limit * a.tau_scale
    ctl = C.Case1Controller(robot, gains, SC.PandaEraser.urdf_path)

    # Stiffness along the writing axes, as every other driver of this arm does:
    # WritingSim's belief frame is the untilted canvas, so W is CANVAS_AXES.
    K = C.k_world(np.array([a.k_lat, a.k_lat, a.k_n]), W.CANVAS_AXES)

    b0 = -0.5 * a.stroke
    p0 = slab.point(a.a0, b0, a.hover)
    q, ok = ctl.ik(p0, R_PEN_DOWN, u.agent.keyframes["rest"].qpos)
    if not ok:
        raise RuntimeError(f"no IK solution for the start pose {p0}")
    robot.set_qpos(torch.tensor(q[None], dtype=torch.float32))
    robot.set_qvel(torch.zeros((1, len(q))))
    ctl.disable_joint_drives()
    # Seed the reference where the pad face actually is, not at the IK target:
    # the residual would otherwise read as the arm settling (writing/sim.py).
    _, p_act, _, _ = ctl.tip_state()
    ctl.reset_state(p_act, R_PEN_DOWN, K, q_rest=q)

    Irot = rot_inertia(ctl)
    lam = np.linalg.eigvalsh(Irot)
    if gains.wrist_inertia is None:
        # Dr = 2 zeta sqrt(kr) Lambda_r^1/2 -> zeta on every axis, by construction
        Dr = 2.0 * gains.zeta * np.sqrt(gains.Kr) * np.sqrt(np.clip(lam, 0.0, None))
        zeta_true = np.full_like(lam, gains.zeta)
    else:
        Dr = np.full_like(lam, 2.0 * gains.zeta * np.sqrt(gains.wrist_inertia * gains.Kr))
        zeta_true = Dr / (2.0 * np.sqrt(np.clip(lam, 1e-12, None) * max(gains.Kr, 1e-12)))
    if report:
        print(f"      I_rot eigenvalues {np.array2string(lam, precision=4)} kg m^2"
              + ("  (Dr is built from them)" if gains.wrist_inertia is None
                 else f"  (the controller assumes {gains.wrist_inertia})"))
        print(f"      Dr eigenvalues {np.array2string(Dr, precision=3)} Nms/rad  ->"
              f"  true damping ratio per axis {np.array2string(zeta_true, precision=2)}"
              f"  (asked for {gains.zeta})")

    axis_cmd = R_PEN_DOWN[:, 2]              # the orientation the pad is told to hold
    keys = ("t", "f_raw_n", "f_sensor", "f_n", "align", "ask", "yielded",
            "alpha", "E", "sat", "sat_wrist", "swing", "clear", "p", "p_ref")
    log = {k: [] for k in keys}
    f25 = np.zeros(3)
    f5 = 0.0
    a25 = dt / (dt + 1.0 / (2 * np.pi * SENSOR_HZ))
    a5 = dt / (dt + 1.0 / (2 * np.pi * PRESSURE_HZ))

    for i in range(int(round(a.duration / dt))):
        t = i * dt
        # Land first, then traverse: `ramp` lowers the command from hover to
        # press depth, `s` carries it along the path once it has settled.
        ramp = min(1.0, t / a.approach_s)
        s = 0.0 if t < a.settle else min(1.0, (t - a.settle) / a.path_s)
        b = b0 + s * a.stroke
        lift = a.hover * (1.0 - ramp) - a.press * ramp
        if a.press_mode == "flat":
            # Constant height: the controller knows the nominal plane only.
            p_ref = slab.point(a.a0, b0, lift) + slab.R @ np.array([0.0, b - b0, 0.0])
        else:
            # Oracle height: translation is solved, so only K_R is left.
            p_ref = slab.point(a.a0, b, lift)

        rec = ctl.compute(C.Case1Proposal((p_ref - ctl.x_d) / dt), dt)
        u.scene.step()
        ctl.advance()

        f_raw = u.scene.get_pairwise_contact_forces(pad, u.surface)[0].cpu().numpy()
        f25 = f25 + a25 * (f_raw - f25)
        n_loc = slab.normal_at(rec["p"])
        f5 = f5 + a5 * (float(f_raw @ n_loc) - f5)

        axis = rec["R"][:, 2]                 # tool z, pointing into the board
        log["t"].append(t)
        log["f_raw_n"].append(float(f_raw @ n_loc))
        log["f_sensor"].append(float(f25 @ n_loc))
        log["f_n"].append(f5)
        # How flush the pad lies: its outward face against the TRUE local normal.
        log["align"].append(float(np.degrees(np.arccos(np.clip(-axis @ n_loc, -1, 1)))))
        # What the surface asks for: the rotation that WOULD make it flush.
        log["ask"].append(float(np.degrees(np.arccos(np.clip(-axis_cmd @ n_loc, -1, 1)))))
        # What the pad actually gave: how far it moved off its command.
        log["yielded"].append(float(np.degrees(np.arccos(np.clip(axis @ axis_cmd, -1, 1)))))
        log["alpha"].append(float(rec["alpha"]))
        log["E"].append(float(rec["E"]))
        log["sat"].append(float(np.max(np.abs(rec["d_tau"]))))
        log["sat_wrist"].append(float(np.max(np.abs(rec["d_tau"][4:]))))
        log["swing"].append(slab.swing_at(rec["p"], a.pad_r))
        log["clear"].append(1000.0 * (slab.local(rec["p"])[2]
                                      - float(slab.surf.height(*slab.local(rec["p"])[:2]))))
        log["p"].append(rec["p"].copy())
        log["p_ref"].append(p_ref.copy())

    out = {k: np.asarray(v) for k, v in log.items()}
    sel = out["t"] >= a.settle + 0.2                 # the traverse, not the landing
    on = sel & (out["f_n"] > a.contact_n)            # ... and only while touching
    f = out["f_n"][sel]
    out.update(Kr=float(Kr), I_rot=lam, zeta_true=zeta_true, Dr=Dr,
               contact=float(np.mean(out["f_n"][sel] > a.contact_n)),
               f_mean=float(f[f > a.contact_n].mean()) if on.any() else 0.0,
               f_peak=float(f.max()), f_peak_raw=float(np.abs(out["f_raw_n"]).max()),
               in_band=float(np.mean((f >= 1.0) & (f <= 6.0))),
               align_mean=float(out["align"][on].mean()) if on.any() else float("nan"),
               align_p95=float(np.percentile(out["align"][on], 95)) if on.any() else float("nan"),
               ask_mean=float(out["ask"][on].mean()) if on.any() else float("nan"),
               yield_mean=float(out["yielded"][on].mean()) if on.any() else float("nan"),
               yield_max=float(out["yielded"][sel].max()),
               swing_mean=float(out["swing"][on].mean()) if on.any() else float("nan"),
               sat_frac=float(np.mean(out["sat"][sel] > 1e-9)),
               sat_wrist_frac=float(np.mean(out["sat_wrist"][sel] > 1e-9)),
               alpha_min=float(out["alpha"][sel].min()), E_end=float(out["E"][-1]))
    # The yield a contact moment of f*r can buy against this K_R, for the row to
    # be read against: below the surface's demand, K_R is simply too stiff to
    # matter and nothing about the surface can change that.
    out["yield_pred"] = float(np.degrees(out["f_mean"] * a.pad_r / max(Kr, 1e-9)))
    # What the compliance BOUGHT.  A rigid pad is misaligned by whatever the
    # surface asked for, so align/ask ~ 1 means K_R did nothing and < 1 means
    # the pad came flush.  Yielding is not the same as yielding the RIGHT WAY:
    # a wrist that rings yields plenty and still lands at align > ask.
    out["left"] = float(out["align_mean"] / max(out["ask_mean"], 1e-6))
    return out


def one(surf, R_s, Kr, a, report=False) -> dict:
    """A fresh scene per condition, so no run inherits the previous one's pose."""
    env = build_scene(surf, R_s, a.friction)
    try:
        return run(env, Slab(surf, R_s), Kr, a, report=report)
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kr", default="60,10,3,1,0.3,0.1,0.03",
                   help="K_R values to compare, Nm/rad")
    p.add_argument("--amp", type=float, default=0.010, help="bump amplitude, m (0 = flat)")
    # 35 mm bumps, not ../curved's 50: a bump of amplitude a and width s has
    # radius of curvature ~ s^2/a, and what matters is how much the normal
    # turns ACROSS THE TOOL.  Our pad is 30 mm where ../curved/wipe.py welded a
    # 50 mm one, so the same demand needs a proportionally sharper surface.
    # Seed 3 at a0 = 0 puts the traverse through a valley: the along-path slope
    # runs -7 deg -> 0 -> +9.5, so the normal turns under the pad rather than
    # holding one slope the way a tilted plane does.
    p.add_argument("--sigma", type=float, default=0.035, help="bump width, m")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--tilt", type=float, default=0.0, help="slab tilt about x, deg")
    p.add_argument("--press-mode", choices=("follow", "flat"), default="follow")
    p.add_argument("--press", type=float, default=0.004, help="commanded depth, m")
    p.add_argument("--hover", type=float, default=0.020)
    p.add_argument("--stroke", type=float, default=0.14, help="traverse length, m")
    p.add_argument("--a0", type=float, default=0.0, help="slab x of the traverse")
    p.add_argument("--k-lat", type=float, default=1000.0)
    p.add_argument("--k-n", type=float, default=600.0)
    p.add_argument("--zeta", type=float, default=0.8)
    p.add_argument("--wrist-inertia", type=float, default=None,
                   help="restore the OLD guessed-scalar Dr (0.01 was the stock guess)")
    p.add_argument("--null-kp", type=float, default=5.0)
    p.add_argument("--null-kd", type=float, default=1.0)
    p.add_argument("--tau-scale", type=float, default=1.0,
                   help="multiplier on the joint torque limits (100 = no saturation)")
    p.add_argument("--no-tank", action="store_true")
    p.add_argument("--friction", type=float, default=0.3)
    p.add_argument("--approach-s", type=float, default=1.0)
    p.add_argument("--settle", type=float, default=1.4)
    p.add_argument("--path-s", type=float, default=6.0)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--contact-n", type=float, default=0.2, help="N, pad-down threshold")
    p.add_argument("--pad-r", type=float, default=SC.ERASER_R, help="pad half-width, m")
    p.add_argument("--save", default="", help="pickle the logs here")
    a = p.parse_args()

    surf = CurvedSurface(amp=a.amp, sigma=a.sigma, seed=a.seed)
    R_s = W.rot_xy(np.deg2rad(a.tilt), 0.0)
    st = surf.stats(a.pad_r)
    print(f"slab: amp {1000*a.amp:.0f} mm, sigma {1000*a.sigma:.0f} mm, seed {a.seed},"
          f" tilted {a.tilt:.0f} deg")
    print(f"      height range {1000*st['h_ptp']:.0f} mm, local tilt {st['tilt_mean']:.1f} deg"
          f" mean / {st['tilt_p95']:.1f} p95, normal swing across the"
          f" {1000*2*a.pad_r:.0f} mm pad {st['swing_mean']:.1f} deg")
    print(f"pad: half-width {1000*a.pad_r:.0f} mm, commanded depth {1000*a.press:.1f} mm,"
          f" K_n {a.k_n:.0f} N/m, press mode {a.press_mode}\n")
    d = demand(Slab(surf, R_s), a.a0, -0.5 * a.stroke, a.stroke, a.pad_r)
    print(f"path: asks for {d['ask_mean']:.1f} deg of tilt mean / {d['ask_p95']:.1f} p95"
          f" / {d['ask_max']:.1f} max, normal swing across the pad"
          f" {d['swing_mean']:.1f} deg mean / {d['swing_p95']:.1f} p95,"
          f" climbs {d['h_ptp']:.0f} mm\n")
    print("  K_R    ask  yield  pred   align  a_p95  left     f    peak  cont  band"
          "   sat  satW  alpha")

    runs = []
    for i, Kr in enumerate([float(x) for x in a.kr.split(",")]):
        r = one(surf, R_s, Kr, a, report=(i == 0))
        runs.append(r)
        print(f"{Kr:6g}  {r['ask_mean']:5.1f}  {r['yield_mean']:5.1f}"
              f"  {r['yield_pred']:5.1f}   {r['align_mean']:5.1f}  {r['align_p95']:5.1f}"
              f"  {r['left']:4.2f}  {r['f_mean']:5.2f}  {r['f_peak']:5.1f}  {100*r['contact']:3.0f}%"
              f"  {100*r['in_band']:3.0f}%  {100*r['sat_frac']:3.0f}%"
              f"  {100*r['sat_wrist_frac']:3.0f}%  {r['alpha_min']:5.2f}")

    print("\n  ask    what the surface asks for: the rotation that would lie flush")
    print("  yield  what the pad gave: how far it moved off its vertical command")
    print("  pred   f*r/K_R, the yield the contact moment can buy at this K_R")
    print("  align  what is left: the pad's face against the true local normal")
    print("  left   align/ask: 1 means K_R bought nothing, 0 means it lay flush")
    if a.save:
        with open(a.save, "wb") as f:
            pickle.dump({"args": vars(a), "runs": runs}, f)
        print(f"\nwrote {a.save}")


if __name__ == "__main__":
    main()
