"""Write by hand: keyboard FORCE on the master, live K_p, optional recording.

The keys do not move anything.  They apply a force to the virtual master, the
master moves against its own damping and whatever the slave pushes back with,
and the slave's Case 1 equilibrium follows the master.  Hold "press" over the
paper and the pen goes down until the paper pushes back as hard as you push --
at rest the force you apply IS the writing force, which is the point of a
force-input teleoperator.  Stiffness is yours to set while you write.

Controls (SAPIEN's viewer owns WASD/QE for its camera, so they are avoided):
    J / L      push left / right along the writing line
    I / K      push up / down the letter (away from / toward you)
    U          press the pen into the paper (3.5 N at rest)
    O          lift
    SHIFT      faster along the paper and up -- never harder into it
    1 / 2      in-plane stiffness down / up  (x 1.25)
    3 / 4      normal stiffness down / up    (x 1.25)
    R          finish: score, save if successful (--record), next task
    N          discard, next task
    T          discard, try the same task again
    P          print the score so far
    ESC        quit; an unfinished attempt is discarded

See README.md, "Collecting demonstrations with the keyboard".

A real haptic device (the U-Arm master of this repository, or any other) plugs
in where KeyboardHuman is: anything with
    wrench(t, x_m, v_m, f_fb) -> (f_h, k_target)
drives the same TeleopSession, and f_fb is the force to render on the device.

This script needs a display.  The force/stiffness logic is exercised headless
by tests.py; the viewer loop itself was only smoke-tested (it opens, steps in
real time and quits), not driven by a person.

Usage:  python3 interactive.py --record demos/keyboard [--text HELLO] [--keep-failed]
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim as SM  # noqa: E402
import teleop as T  # noqa: E402


class KeyboardHuman:
    """f_h from held keys, plus the damping of the arm resting on the handle
    (without it a held key accelerates the master without bound).

    The damping is ANISOTROPIC, like a careful hand: loose along the paper,
    heavy into it.  With one isotropic 40 Ns/m, holding "press" from hover
    landed the pen at 72 mm/s and tore the paper every time (15-16 N sustained);
    at 230 Ns/m along the normal it lands at ~15 mm/s, the synthetic writer's
    descent speed.  Damping sets only the approach speed: at rest the paper
    still pushes back with exactly the pressed force.

    SHIFT doubles the in-plane and lift forces, never the press: 2 x 3.5 N would
    sit outside the 1-6 N force band the task is scored on.

    With neither U nor O held, a relaxed hand lifts slightly (F_IDLE).  Without
    it, releasing U left the pen resting on the paper -- the reflected force
    pushes the master back only until it balances -- and travelling to the next
    stroke dragged 8 dots of stray ink.  With it, release means pen-up.
    """
    F_PLANE = 2.0      # N  -> ~5 cm/s along the paper
    F_NORMAL = 3.5     # N  -> the writing force at rest
    B_PLANE = 40.0     # Ns/m
    B_NORMAL = 230.0   # Ns/m -> ~15 mm/s toward the paper
    F_IDLE = 0.5       # N up when no normal key is held -> ~2 mm/s off the paper

    def __init__(self, window, W: np.ndarray, k0):
        self.win, self.W = window, W
        self.k = np.asarray(k0, dtype=float).copy()
        self.B = W @ np.diag([self.B_PLANE, self.B_PLANE, self.B_NORMAL]) @ W.T

    def wrench(self, t, x_m, v_m, f_fb):
        w = self.win
        g = 2.0 if w.shift else 1.0
        f = np.zeros(3)                         # in the writing frame (u, v, n)
        f[0] = g * self.F_PLANE * (w.key_down("l") - w.key_down("j"))
        f[1] = g * self.F_PLANE * (w.key_down("i") - w.key_down("k"))
        up, down = w.key_down("o"), w.key_down("u")
        f[2] = self.F_NORMAL * (g * up - down) if (up or down) else self.F_IDLE
        return self.W @ f - self.B @ v_m, self.k.copy()

    def stiffness_keys(self) -> bool:
        w, changed = self.win, False
        for key, idx, fac in (("1", (0, 1), 0.8), ("2", (0, 1), 1.25),
                              ("3", (2,), 0.8), ("4", (2,), 1.25)):
            if w.key_press(key):
                for i in idx:
                    self.k[i] = float(np.clip(self.k[i] * fac, 100.0, 4000.0))
                changed = True
        return changed


CONTROLS = """
  J / L  left / right      I / K  up / down the letter      U  press      O  lift
  SHIFT  faster (not harder)      1 / 2  in-plane K down / up      3 / 4  normal K down / up
  R  finish: score, save if successful, next task      N  discard, next task
  T  discard, try the same task again      P  score so far      ESC  quit (unfinished attempt is discarded)
"""


def _resume(out: pathlib.Path, first_seed: int) -> int:
    """Next unused seed in `out`, so a new session never repeats a task or
    overwrites a file.  Discarded attempts count too: they are in the log."""
    import json
    seeds = [int(p.stem.split("_")[1]) for p in out.glob("ep_*.h5")]
    log = out / "attempts.jsonl"
    if log.exists():
        seeds += [json.loads(l)["seed"] for l in log.read_text().splitlines() if l.strip()]
    return max([first_seed - 1] + seeds) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="write this every time (paper is still random); "
                                                 "random words and shapes if omitted")
    ap.add_argument("--seed", type=int, default=50_000,
                    help="first task seed; synthetic demos use 0.., evaluation 10000..")
    ap.add_argument("--record", default=None, help="directory to save demonstrations into")
    ap.add_argument("--keep-failed", action="store_true", help="save unsuccessful attempts too")
    ap.add_argument("--no-template", action="store_true")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--duration", type=float, default=0.0, help="quit after N wall seconds (0 = never)")
    args = ap.parse_args()

    import json

    out = pathlib.Path(args.record) if args.record else None
    sim = SM.WritingSim(cameras=out is not None, wrist_camera=not args.no_wrist,
                        image_size=args.image_size, render_mode="human")
    viewer = sim.u.render_human()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        from collect import Recorder
    seed = _resume(out, args.seed) if out else args.seed
    kept = 0

    def new_episode(seed):
        spec = SM.TaskSpec.sample(seed)
        if args.text:
            spec = dataclasses.replace(spec, text=args.text)
        spec = dataclasses.replace(spec, show_template=not args.no_template, time_limit=1e9)
        sim.reset(spec, hover=0.015)
        sess = T.TeleopSession(sim)
        human = KeyboardHuman(viewer.window, sim.W, [1500.0, 1500.0, 400.0])
        rec = Recorder(sim) if out else None
        print(f"\n[task {seed}] write {spec.text!r}   (paper height and tilt are hidden)")
        return spec, sess, human, rec

    def finish(spec, rec, keep: bool) -> None:
        """Score the attempt; save it if asked and allowed; log it either way."""
        nonlocal kept
        res = sim.score()
        path = None
        if out is not None and keep and sim.ink_uv and (res["success"] or args.keep_failed):
            path = out / f"ep_{spec.seed:05d}.h5"
            metrics = {k: v for k, v in res.items() if k != "checks"}
            rec.save(path, dict(
                text=spec.text, success=bool(res["success"]), source="keyboard",
                spec=dataclasses.asdict(spec), metrics=metrics,
                master=dataclasses.asdict(sess.mp), criteria=dataclasses.asdict(sim.crit),
                gains={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in dataclasses.asdict(sim.ctl.g).items()},
                phases=T.PHASES, sim_dt=sim.dt, log_hz=100.0, policy_hz=10.0,
                writing_frame=sim.W.tolist()))
            kept += int(res["success"])
        verdict = "SUCCESS" if res["success"] else f"failed: {res['fail_reason']}"
        print(f"\n[{verdict}] coverage {res['coverage']:.2f} precision {res['precision']:.2f} "
              f"in-band {res['in_band']:.2f} peak {res['peak_force']:.1f} N  ->  "
              f"{path if path else 'not saved'}   ({kept} successful saved this session)")
        if out is not None:
            with open(out / "attempts.jsonl", "a") as f:
                f.write(json.dumps(dict(seed=spec.seed, text=spec.text, saved=str(path) if path else None,
                                        success=bool(res["success"]), reason=res["fail_reason"],
                                        coverage=res["coverage"], precision=res["precision"],
                                        in_band=res["in_band"], peak_force=res["peak_force"],
                                        t=res["t"], kept_on="R" if keep else "discard")) + "\n")

    print(CONTROLS)
    spec, sess, human, rec = new_episode(seed)
    dt = sim.dt
    last, acc, last_print, i = time.time(), 0.0, 0.0, 0
    t_start = last
    while not viewer.window.should_close:
        now = time.time()
        acc += min(now - last, 0.1)
        last = now
        w = viewer.window
        if w.key_down("esc") or (args.duration and now - t_start > args.duration):
            break
        for key, keep, advance in (("r", True, True), ("n", False, True), ("t", False, False)):
            if w.key_press(key):
                finish(spec, rec, keep)
                seed += int(advance)
                spec, sess, human, rec = new_episode(seed)
                i = 0
                break
        if w.key_press("p"):
            r = sim.score()
            print(f"\n[score] success={r['success']} coverage {r['coverage']:.2f} precision "
                  f"{r['precision']:.2f} in-band {r['in_band']:.2f} peak {r['peak_force']:.1f} N "
                  f"{r['fail_reason']}")
        if human.stiffness_keys():
            print(f"\n[K] u,v {human.k[0]:.0f}  n {human.k[2]:.0f} N/m")
        n = 0
        while acc >= dt and n < 50:            # fixed-step catch-up to wall time
            acc -= dt
            n += 1
            f_h, k = human.wrench(sim.t, sess.x_m, sess.v_m, sess.f_fb)
            r = sess.step(f_h, k)
            if rec is not None:
                rec.on_step(i, r, sess)
            i += 1
        sim.env.render_human()
        if sim.t - last_print > 0.5:
            last_print = sim.t
            felt = float(sess.f_fb @ sim.belief.normal)
            lo, hi = sim.crit.force_band
            print(f"\r t {sim.t:6.1f}s  felt {felt:5.2f} N  paper {sim.last['f_n']:5.2f} N "
                  f"(band {lo:.0f}-{hi:.0f})  ink {len(sim.ink_uv):4d}  "
                  f"K u,v {human.k[0]:.0f} n {human.k[2]:.0f}   ", end="", flush=True)
    print("\n[quit] unfinished attempt discarded")
    sim.close()


if __name__ == "__main__":
    main()
