"""A scripted eraser, to check the task does what it says.

Lands the pad on the board, rasters across the glyph's bounding box at a fixed
penetration, and reports what came off.  Not a policy -- the point is that the
mechanics are sound: hovering erases nothing, pressing and rubbing does.

    python3 smoke.py                 flat board
    python3 smoke.py --curved        the same glyph on a curved board
    python3 smoke.py --curved --no-follow    ... commanded as if it were flat

THE COMMAND IS INTERPOLATED IN BOARD COORDINATES, not in world space.  On a
flat board the two are the same thing, because the board frame is affine; on a
curved one a world-space straight line between two waypoints 80 mm apart cuts
a chord through every bump between them, and the pad either loses the surface
or digs in.  Going through `frame.to_world` each step instead makes "press in
by `penetration`" mean the same on both boards, which is what `--no-follow`
turns off: it commands the mean plane, and `curved_probe.py --press-mode flat`
measures what that costs (55-63% of the time in the force band, against 100%).

A scripted demonstrator is allowed to know the surface -- an operator would see
it.  What stays hidden is the pad's ORIENTATION problem: the controller is told
one vertical pose and never told the board turns under it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import controller as C          # noqa: E402  writing/controller.py
import sim as WSM               # noqa: E402  writing/sim.py, for TaskSpec
import wipe_scene as SC         # noqa: E402
from wipe_sim import CurvedWipingSim, WipeCriteria, WipingSim  # noqa: E402


def run(text="S", penetration=0.004, k_lat=1000.0, k_n=600.0, speed=0.05,
        hover_s=1.0, curved=False, follow=True, tilt=0.0, seed=0, verbose=True,
        kr=None, every=10, erase_work=None, wrist_inertia=None, f_target=None,
        on_sample=None, render_mode=None, kr_fn=None, cameras=False,
        letter_height=0.035, offset_uv=(0.0, 0.0), time_limit=120.0,
        pad_give=None):
    # `wrist_inertia` is the guess Dr is built from; the true operational-space
    # rotational inertia at the pad is [0.005, 0.18, 0.38] kg m^2, so the stock
    # 0.01 puts the real rotational damping ratio at 0.13-0.19 instead of 0.8.
    # Exposed here because a raster REVERSES, and a reversal is where an
    # underdamped wrist shows up (see CURVED_BOARD.md).
    gains = (None if (kr is None and wrist_inertia is None) else C.Case1Gains(
        **({} if kr is None else dict(Kr=float(kr)))
        | ({} if wrist_inertia is None else dict(wrist_inertia=float(wrist_inertia)))))
    # `pad_give` is how far the felt squashes, and it sets the misalignment the
    # erasure model is BLIND to: a tilt whose height change across the pad is
    # under it is absorbed, so it costs nothing.  atan(give / 2r) is 7.6 deg at
    # the 15 mm pad -- more than a soft wrist's whole error -- which is why
    # rotational stiffness had nothing to win on coverage.  A harder pad (a
    # squeegee, not a sponge) lowers that threshold.
    crit = (None if (erase_work is None and pad_give is None) else WipeCriteria(
        **({} if erase_work is None else dict(erase_work=float(erase_work)))
        | ({} if pad_give is None else dict(pad_give=float(pad_give)))))
    s = (CurvedWipingSim if curved else WipingSim)(cameras=cameras, gains=gains,
                                                  criteria=crit,
                                                  render_mode=render_mode)
    # On the patchy board the glyph has to be big enough that the raster
    # crosses a band: the bands are 170 mm of flat and 170 mm of curved, so a
    # 35 mm glyph sits inside one of them and the board might as well be
    # uniform -- which is exactly what the first run measured.
    # offset_uv moves the glyph on the board.  Without it a seed changes
    # nothing here: TaskSpec is built directly rather than through
    # TaskSpec.sample(), so every randomisable field keeps its default and the
    # seed only reaches env.reset(), which this scene does not use.
    # `time_limit` cuts the raster off (`drive` returns False), so a slow
    # traverse needs it raised: corr_gate.py runs at 0.02 m/s where the same
    # path takes 2.5x longer than at the shipped 0.05.
    spec = WSM.TaskSpec(text=text, letter_height=float(letter_height),
                        time_limit=float(time_limit), seed=seed,
                        offset_uv=tuple(float(x) for x in offset_uv),
                        tilt_x=float(np.deg2rad(tilt)))
    s.reset(spec)
    if verbose:
        print(f"marks on the board: {len(s.marks_uv)}"
              + (f", board height under them {1000*np.ptp(s.frame.height(s.marks_uv)):.0f}"
                 f" mm peak to peak" if curved else ""))

    lo, hi = s.marks_uv.min(0), s.marks_uv.max(0)
    pad = SC.ERASER_R
    rows = np.arange(lo[1] - pad / 2, hi[1] + pad, pad)      # overlap the passes
    way = []
    for i, v in enumerate(rows):                            # boustrophedon raster
        us = [lo[0] - pad, hi[0] + pad][:: 1 if i % 2 == 0 else -1]
        way += [np.array([us[0], v]), np.array([us[1], v])]

    K = C.k_world(np.array([k_lat, k_lat, k_n]), s.W)
    # Where "pressed in by `penetration`" is, as a world point.  On the curved
    # board with --no-follow this is the mean plane, i.e. what the flat task
    # commands; otherwise it is the local surface.
    plane = s.belief if (curved and not follow) else s.frame
    normal = (plane.normal_at if hasattr(plane, "normal_at")
              else (lambda uv: plane.normal))

    # ---- the normal-force loop -----------------------------------------
    # WITHOUT IT, K_R LEAKS INTO THE FORCE.  A fixed penetration is a position
    # command, and how much force it produces depends on how the pad is lying:
    # measured on the flat board, 2.51 N at K_R = 60 and 3.73 N at K_R = 0.3,
    # a 1.5x difference from rotational stiffness alone.  Erasure is work, so
    # that difference erases 1.5x faster and the comparison then reports a
    # force effect under a rotational name (it did: 58% vs 100% of the glyph
    # off at erase_work 0.12).
    #
    # One integrator along the local normal, riding on the surface-following
    # path, which is what holds f at the target in EVERY condition:
    #
    #     depth_ddot = (f_target - f_sensed - b depth_dot) / m
    #
    # No outer position spring in the normal direction, deliberately.  With one
    # (../curved/wipe.py's admittance) the steady state splits the target
    # between the two springs -- f = f_target K_n/(K_n + K_outer), half of it
    # here -- and only an integrator removes that.  In-plane the path is still
    # commanded by position: this is force control along the normal and
    # position control across it, the usual split for a contact task.
    #
    # m = 3 kg and b = 60 Ns/m against the board's 600 N/m put the loop at
    # 2.2 Hz and zeta = 0.7, well inside the 25 Hz force sensor it reads.
    depth, v_n = penetration, 0.0

    def press(uv):
        nonlocal depth, v_n
        if f_target is None:
            return plane.to_world(np.r_[uv, -penetration])
        n = normal(uv)
        err = f_target - float(s.f_filt @ n)      # f_filt is the force ON the pad
        v_n = float(np.clip(v_n + (err - 60.0 * v_n) / 3.0 * s.dt, -0.05, 0.05))
        depth = float(np.clip(depth + v_n * s.dt, -0.002, 0.020))
        return plane.to_world(np.r_[uv, -depth])

    # How flush the pad lies, step by step.  Against the TRUE LOCAL normal on
    # the curved board and the board normal on the flat one, which is the same
    # quantity in both cases -- the angle between the face that does the
    # erasing and the surface it is supposed to be erasing.
    trace = {k: [] for k in ("t", "f", "erased", "mis", "yld", "depth", "u")}

    def sample():
        uvh = s.last["contact_uvh"]
        n = (s.frame.normal_at(uvh[:2]) if hasattr(s.frame, "normal_at")
             else s.frame.normal)
        axis = s.last["R"][:, 2]                      # tool z, into the board
        trace["t"].append(s.t)
        trace["f"].append(float(s.last["f_n"]))
        trace["erased"].append(float(s.gone.mean()))
        trace["mis"].append(float(np.degrees(np.arccos(np.clip(-axis @ n, -1, 1)))))
        # How far the pad moved off its COMMAND, which is what tells a wrist
        # that stayed put from one that wandered: the misalignment above is an
        # unsigned angle, so 8 deg of "held the vertical it was told" and 8 deg
        # of "drifted somewhere else" read the same.  On a raster they are the
        # two different failures, and they are what separates K_R.
        trace["yld"].append(float(np.degrees(np.arccos(
            np.clip(axis @ WSM.R_PEN_DOWN[:, 2], -1, 1)))))
        trace["depth"].append(1000.0 * depth)
        # where on the board the pad is, so a score can ask what THIS band wants
        uvh = (s.last or {}).get("contact_uvh")
        trace["u"].append(float(uvh[0]) if uvh is not None else float("nan"))
        if on_sample is not None:          # a renderer hangs its frame grab here
            on_sample(s)

    def drive(path, n_steps) -> bool:
        """Step `n_steps`, taking the command from `path(frac)`.  False at the
        time limit."""
        for i in range(1, n_steps + 1):
            nxt = path(i / n_steps)
            # K_R is integrated state, not a parameter: `advance` steps it by
            # Ur*dt and the gate scales Ur like any other proposal, so a
            # time-varying rotational stiffness is REQUESTED, not assigned.
            # (Assigning g.Kr does nothing -- reset_state copied it into
            # ctl.kr, which is what compute uses.)
            ur = None if kr_fn is None else (float(kr_fn(s)) - s.ctl.kr) / s.dt
            s.step(C.Case1Proposal((nxt - s.ctl.x_d) / s.dt,
                                   (K - s.ctl.K) / s.dt, Ur=ur))
            if s.step_i % every == 0:
                sample()
            if s.t >= spec.time_limit:
                return False
        return True

    x0 = s.ctl.x_d.copy()
    down = plane.to_world(np.r_[way[0], -penetration])   # land in position mode
    ok = drive(lambda f: x0 + (down - x0) * f, int(hover_s / s.dt))
    for a, b in zip(way, way[1:]):
        if not ok:
            break
        n_steps = max(1, int(np.linalg.norm(b - a) / speed / s.dt))
        ok = drive(lambda f, a=a, b=b: press(a + (b - a) * f), n_steps)

    sample()
    r = s.score()
    r["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    down = r["trace"]["f"] > 0.8
    r["mis_mean"] = float(r["trace"]["mis"][down].mean()) if down.any() else float("nan")
    r["yld_mean"] = float(r["trace"]["yld"][down].mean()) if down.any() else float("nan")
    # When the glyph was gone, in pad-down seconds: the thing a stiff pad is
    # slower at, and a measure that does not saturate the way "erased 100%"
    # does once the raster has covered the glyph twice over.
    hit = np.nonzero(r["trace"]["erased"] >= 0.9)[0]
    r["t90"] = float(r["trace"]["t"][hit[0]]) if len(hit) else float("nan")
    # What the board asks of the pad over the glyph: the rotation that would lie
    # flush, averaged over the marks.  The board tilt on a flat board, and the
    # thing to compare `mis` against on a curved one.
    up = np.array([0.0, 0.0, 1.0])
    n = np.stack([normal(uv) for uv in s.marks_uv])
    r["ask_mean"] = float(np.degrees(np.arccos(np.clip(n @ up, -1, 1))).mean())
    if verbose:
        print(f"erased {100 * r['erased']:.1f}%  ({r['n_marks'] - r['n_left']}/{r['n_marks']})  "
              f"in-band {100 * r['in_band']:.1f}%  peak {r['peak_force']:.1f} N  "
              f"pad-down {r['pad_down_s']:.1f}s  t {r['t']:.1f}s")
        print(f"success: {r['success']}" + ("" if r["success"] else f"   ({r['fail_reason']})"))
    s.close()
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default="S")
    p.add_argument("--curved", action="store_true", help="curved board")
    p.add_argument("--no-follow", dest="follow", action="store_false",
                   help="on a curved board, command the mean plane anyway")
    p.add_argument("--tilt", type=float, default=0.0, help="board tilt, deg")
    p.add_argument("--penetration", type=float, default=0.004)
    p.add_argument("--k-lat", type=float, default=1000.0)
    p.add_argument("--k-n", type=float, default=600.0)
    p.add_argument("--speed", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kr", type=float, default=None, help="K_R, Nm/rad")
    p.add_argument("--wrist-inertia", type=float, default=None)
    p.add_argument("--force", type=float, default=None,
                   help="N: regulate the normal force instead of the depth")
    a = p.parse_args()
    r = run(a.text, penetration=a.penetration, k_lat=a.k_lat, k_n=a.k_n, speed=a.speed,
            curved=a.curved, follow=a.follow, tilt=a.tilt, seed=a.seed, kr=a.kr,
            wrist_inertia=a.wrist_inertia, f_target=a.force)
    print(f"pad misalignment {r['mis_mean']:.1f} deg mean while down,"
          f" glyph gone at t = {r['t90']:.1f} s")


if __name__ == "__main__":
    main()
