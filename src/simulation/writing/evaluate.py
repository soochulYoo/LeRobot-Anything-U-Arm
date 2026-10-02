"""Evaluate a policy on the writing task, in the scene the demonstrations came from.

A policy acts at `control_hz` and outputs the Case 1 action the demonstrations
are labelled with:

    action = {"x_d": (3,) world, m,  "k_diag": (3,) N/m along the paper's u, v, n}
             or a flat (6,) array [x_d, k_diag];   None ends the episode

WritingPolicyEnv holds the action as the set-point to reach by the next frame:
between frames x_d and K are interpolated linearly and fed to the controller as
deadbeat proposals, through the same energy-tank gate the teleoperator used.
The score is sim.WritingSim.score(), unchanged.

Policies here are references, not learned ones:

  replay     the stored actions of a demonstration, open loop, on its own
             paper -- the ceiling a perfect imitator reaches, and a check that
             the 10 Hz labels survive the trip
  scripted   a privileged writer: knows the strokes, NOT the paper's pose.
             It feels for the paper, then presses by setting the depth to
             f_target / k_n below where it touched, and writes.  Its three
             stiffnesses are arguments, which makes it the probe for whether
             the task rewards choosing K:
                 --k-t 2000 --k-n 400    stiff along the paper, soft off it
                 --k-t 3000 --k-n 3000   stiff everywhere
                 --k-t 400  --k-n 400    soft everywhere

Plug in a learned policy with --policy my_module:make_policy, where
make_policy(env) returns an object with reset(goal, obs) and act(obs).

Usage:
    python3 evaluate.py --policy scripted --k-t 2000 --k-n 400 --episodes 24
    python3 evaluate.py --policy replay --demos demos/writing_v1 --episodes 10
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import controller as C  # noqa: E402
import sim as SM  # noqa: E402

Array = np.ndarray

# --------------------------------------------------------------------------- #
# THE TWO ACTION LAYOUTS
#
# GIC puts a spring between the tool and a target pose, so three quantities --
# the target, the stiffness and the force -- are tied by one equation,
# F = K (x_d - x).  Which two of them a policy names decides what a stiffness
# error costs.
#
#   "legacy" (6,)  {"x_d", "k_diag"}            the labels cases a-d train on
#       The policy names the TARGET.  In contact the force is K times the gap
#       to the surface, so a K that comes out 20% high presses 20% harder: the
#       stiffness error IS a force error.  Worse, the demonstration cannot tell
#       the two apart -- 10 N is "K = 1000 N/m, 10 mm in" and "K = 250 N/m,
#       40 mm in" and the recorded (x, F) is identical, which is why K is not
#       identifiable from the log at all.
#
#   "spring" (9,)  {"x_ref", "f_d", "k_diag"}   SPRING-CONSISTENT
#       The policy names where it expects the TOOL to be (x_ref, world), the
#       wrench it intends to apply there (f_d, along the paper's u, v, n,
#       pointing tool -> environment) and the stiffness.  The CONTROLLER places
#       the target:
#                       x_d = x_ref + K^-1 f_d
#       Now at x = x_ref the force is f_d for ANY K, so a wrong K no longer
#       moves the force; it only changes how the force responds when the
#       surface is not where x_ref said it was.  Both new labels are DIRECT
#       MEASUREMENTS (the tip pose and the contact wrench) and neither depends
#       on how K was chosen, so they are available on demonstrations already
#       recorded and stay fixed if the stiffness rule changes.
#
#       The future wrench is then part of the action instead of a second head
#       that can contradict it:
#                   F(t+j) = f_d(t+j) + K (x_ref(t+j) - x(t+j))
#       which on the demonstration (x = x_ref) is f_d exactly.  That is the
#       whole of "future_ft derived": cases b, c and d exist to reconcile an
#       action head with a force head, and this layout leaves them nothing to
#       reconcile.
#
#       It also isolates the part that CANNOT be supervised.  Because x = x_ref
#       on every training sample, the spring term is identically zero there and
#       K receives no gradient from imitation at all.  That is the point, not a
#       defect: in the legacy layout K soaks up gradient meant for the force.
#       K has to come from somewhere else -- a human channel, or the robustness
#       term in train.py that asks what the force does when the surface moves.
def parse_action(action, W: Array) -> tuple[Array, Array | None, Array]:
    """-> (x, f_d or None, k_diag).  `f_d` is None for the legacy layout."""
    if isinstance(action, dict):
        k = np.asarray(action["k_diag"], dtype=float).reshape(3)
        if "f_d" in action:
            return (np.asarray(action["x_ref"], float).reshape(3),
                    np.asarray(action["f_d"], float).reshape(3), k)
        return np.asarray(action["x_d"], float).reshape(3), None, k
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size == 9:
        return a[:3], a[3:6], a[6:]
    return a[:3], None, a[3:]


class WritingPolicyEnv:
    def __init__(self, control_hz: float = 10.0, image_size: int = 128,
                 wrist_camera: bool = True, cameras: bool = True,
                 render_mode: str | None = None, criteria: SM.Criteria | None = None,
                 rel_anchor: bool = False):
        self.sim = SM.WritingSim(image_size=image_size, wrist_camera=wrist_camera,
                                 cameras=cameras, render_mode=render_mode, criteria=criteria)
        self.control_hz = control_hz
        self.n_sub = max(1, int(round(1.0 / (control_hz * self.sim.dt))))
        self.frames: list[np.ndarray] = []
        self.record_video = False
        self.rel_anchor = bool(rel_anchor)

    def reset(self, spec: SM.TaskSpec) -> dict:
        self.spec = spec
        self.frames = []
        return self.sim.reset(spec)

    def goal(self) -> dict:
        return self.sim.goal()

    @property
    def time_up(self) -> bool:
        return self.sim.t >= self.spec.time_limit

    def step(self, action) -> dict:
        """One policy frame, interpolated down to the controller rate.

        x_ref is used OPEN LOOP on purpose.  Re-anchoring it on the measured
        tip every substep would hold the force at f_d whatever the surface did
        -- that is force control, and it would make K do nothing.  The 10 Hz
        replan is the feedback, and between replans a deviation from x_ref
        turns into a force change through K, which is exactly the behaviour K
        is supposed to be choosing.
        """
        sim = self.sim
        x_goal, f_goal, k_goal = parse_action(action, sim.W)
        x0, K0 = sim.ctl.x_d.copy(), sim.ctl.K.copy()
        # spring_rel: the caller passes x_ref as a DELTA from the measured tip
        # (data.py builds the label the same way), so the anchor is re-read every
        # policy frame.  Only the spring OFFSET K^-1 f_d stays open loop, which is
        # what keeps the force K-independent and K responsible for the response to
        # surface deviation.  Position error no longer accumulates across a chunk.
        if f_goal is not None and self.rel_anchor:
            _, p_meas, _, _ = sim.ctl.tip_state()
            x_goal = p_meas + x_goal
        if f_goal is None:
            for i in range(1, self.n_sub + 1):
                w = i / self.n_sub
                x_next = x0 + w * (x_goal - x0)
                K_next = K0 + w * (C.k_world(k_goal, sim.W) - K0)
                sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt,
                                         (K_next - sim.ctl.K) / sim.dt))
        else:
            # Interpolate the three quantities the policy named and let the
            # spring law place the target at every substep, rather than drawing
            # a straight line between two of its answers.  Stiffness moves in
            # LOG space, so a 3000 -> 300 N/m ramp is geometric and never
            # crosses a decade in one substep.
            # DIVIDE BY THE STIFFNESS THE CONTROLLER WILL ACTUALLY APPLY.
            # k_goal is exp() of a network output and therefore unbounded, while
            # Case1Controller.advance projects K onto [k_lo, k_hi].  Dividing by
            # the REQUESTED value let one outlier sample with k -> 0 put the
            # target at infinity; the force went with it and the controller's
            # eigh raised "Eigenvalues did not converge" from the resulting NaN,
            # which killed 3 of 32 spring evaluations outright.  Clamping here is
            # also the physically honest decode: the spring that gets applied is
            # the clamped one, so that is the K the target must be placed against.
            g = sim.ctl.g
            k_goal = np.clip(k_goal, g.k_lo, g.k_hi)
            _, p_now, _, _ = sim.ctl.tip_state()
            k0 = sim.k_diag(K0)
            # the wrench the spring is applying right now, so w = 0 reproduces
            # the current target exactly and the policy frame joins smoothly
            f0 = sim.W.T @ (K0 @ (x0 - p_now))
            for i in range(1, self.n_sub + 1):
                w = i / self.n_sub
                x_ref = p_now + w * (x_goal - p_now)
                f_d = f0 + w * (f_goal - f0)
                k = np.exp(np.log(k0) + w * (np.log(k_goal) - np.log(k0)))
                x_next = x_ref + sim.W @ (f_d / k)
                K_next = C.k_world(k, sim.W)
                sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt,
                                         (K_next - sim.ctl.K) / sim.dt))
        if self.record_video:
            r = sim.env.render()
            r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
            self.frames.append(r[0] if r.ndim == 4 else r)
        return sim.observe()

    def score(self) -> dict:
        return self.sim.score()


# --------------------------------------------------------------------------- #
# reference policies
# --------------------------------------------------------------------------- #
class ReplayPolicy:
    """Open-loop replay of a demonstration's policy-rate actions."""

    def __init__(self, path):
        import h5py
        with h5py.File(path) as f:
            self.x_d = f["action/x_d"][:].astype(float)
            self.k = f["action/k_diag"][:].astype(float)
            self.spec = SM.TaskSpec(**json.loads(f.attrs["spec"]))
            self.hz = float(f.attrs["policy_hz"])

    def reset(self, goal, obs) -> None:
        self.i = 0

    def act(self, obs):
        if self.i >= len(self.x_d):
            return None
        a = {"x_d": self.x_d[self.i], "k_diag": self.k[self.i]}
        self.i += 1
        return a


class ScriptedPolicy:
    """Privileged strokes, unknown paper.  Constant stiffness while in contact
    (k_t along the paper, k_n off it), k_travel elsewhere."""

    def __init__(self, env: WritingPolicyEnv, k_t: float = 2000.0, k_n: float = 400.0,
                 k_travel: float = 800.0, f_target: float = 3.0, v_write: float = 0.03,
                 v_travel: float = 0.06, v_desc: float = 0.015, hover: float = 0.012,
                 f_touch: float = 0.5, spring: bool = False, k_err: float = 1.0):
        self.env = env
        # `spring`: emit the spring-consistent layout instead of a target pose.
        # The press is then named as a force rather than as a depth, which is
        # the same command -- this writer already computed the depth AS
        # f_target / k_n, so --spring is the identical controller input with the
        # division moved from the policy to the controller.  Its one deliberate
        # difference is that f_d is RAMPED at both ends.  Stepping it is what a
        # depth command never has to worry about and a force command does: the
        # target moves by f_d / k_n, so at a soft k_n = 120 N/m a 3 N step is a
        # 25 mm jump in the reference and the pen arrives at travel speed.
        # Measured on 6 episodes at k_err = 0.3, stepping f_d peaked at 13.3 N
        # and tore the paper every time, against 7.0 N once it is ramped.  The
        # release is ramped for the mirror-image reason: lifting a loaded spring
        # flicks the pen up.
        self.spring = spring
        self.f_press = 0.0
        # WHAT A STIFFNESS ERROR COSTS, the two layouts side by side.
        # k_err is the ratio of the k_n this writer COMMANDS to the k_n it
        # reasoned with -- i.e. what a policy whose stiffness head is wrong by
        # k_err while its pose head reproduced the demonstration does.
        #   legacy  depth was f_target / k_n, commanded stiffness is k_err k_n,
        #           so the force comes out k_err * f_target: the stiffness error
        #           IS a force error, one for one.
        #   spring  the controller divides f_d by the stiffness it was actually
        #           given, so the force is f_target whatever k_err is.  A wrong
        #           K now only changes the response to surface deviation.
        self.k_err = float(k_err)
        self.t_ramp = 0.2             # s, over which f_d is taken up and let go
        self.k = dict(t=k_t, n=k_n, travel=k_travel)
        self.f_target, self.v_write, self.v_travel = f_target, v_write, v_travel
        self.v_desc, self.hover, self.f_touch = v_desc, hover, f_touch
        self.dt = 1.0 / env.control_hz

    def reset(self, goal, obs) -> None:
        B = self.env.sim.belief
        self.B = B
        self.strokes = [s[m].astype(float) for s, m in zip(goal["strokes_uv"], goal["mask"]) if m.any()]
        self.plan = B.to_canvas(obs["x_d"]).astype(float)
        self.k_idx, self.phase, self.s, self.t_phase = 0, "travel", 0.0, 0.0
        self.h_touch = 0.0
        self._from = self.plan.copy()

    def _act(self, k):
        k = np.asarray(k, dtype=float)
        k = k * np.array([1.0, 1.0, self.k_err])
        if not self.spring:
            return {"x_d": self.B.to_world(self.plan), "k_diag": k}
        # x_ref is where the TOOL is expected to be -- the surface it touched,
        # not a point below it -- and the press is f_d along -n.
        return {"x_ref": self.B.to_world(self.plan), "k_diag": k,
                "f_d": np.array([0.0, 0.0, -self.f_press])}

    def act(self, obs):
        dt, k = self.dt, self.k
        self.t_phase += dt
        f_n = float(np.asarray(obs["f_contact"]) @ self.B.normal)
        if self.k_idx >= len(self.strokes):
            if self.phase != "done":
                self.phase, self.t_phase = "done", 0.0
            self.plan[2] = min(self.plan[2] + 0.04 * dt, 0.04)
            return None if self.t_phase > 1.0 else self._act([k["travel"]] * 3)
        P = self.strokes[self.k_idx]
        if self.phase == "travel":
            goal = np.r_[P[0], self.hover]
            T = max(0.3, 1.5 * np.linalg.norm(goal - self._from) / self.v_travel)
            tau = min(1.0, self.t_phase / T)
            self.plan = self._from + (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5) * (goal - self._from)
            if tau >= 1.0:
                self.phase, self.t_phase = "descend", 0.0
            return self._act([k["travel"]] * 3)
        if self.phase == "descend":
            if f_n > self.f_touch and self.t_phase > 0.3:
                self.h_touch = self.plan[2]
                if self.spring:
                    self.f_press = 0.0            # ramped in `stroke`, not stepped
                else:
                    # press: the depth that gives f_target through k_n
                    self.plan[2] = self.h_touch - self.f_target / k["n"]
                self.phase, self.t_phase, self.s = "stroke", 0.0, 0.0
            else:
                self.plan[2] -= self.v_desc * dt
            return self._act([k["t"], k["t"], k["n"]])
        if self.phase == "stroke":
            if self.spring:
                self.f_press = min(self.f_target,
                                   self.f_press + self.f_target * dt / self.t_ramp)
            if self.t_phase > 0.3:                  # let the press settle first
                self.s += self.v_write * dt
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            s = min(self.s, cum[-1])
            self.plan[:2] = [np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])]
            if self.s >= cum[-1] + 0.3 * self.v_write:
                self.phase, self.t_phase = "lift", 0.0
            return self._act([k["t"], k["t"], k["n"]])
        if self.phase == "lift":
            if self.spring and self.f_press > 0.0:
                # spend the spring first: lifting a loaded spring flicks the pen
                self.f_press = max(0.0, self.f_press - self.f_target * dt / self.t_ramp)
                return self._act([k["t"], k["t"], k["travel"]])
            self.plan[2] += 0.04 * dt
            if self.plan[2] >= self.h_touch + self.hover:
                self.k_idx += 1
                self.phase, self.t_phase = "travel", 0.0
                self._from = self.plan.copy()
            return self._act([k["t"], k["t"], k["travel"]])
        return None


def load_policy(name: str, env, args):
    if name in ("scripted", "scripted_spring"):
        return ScriptedPolicy(env, k_t=args.k_t, k_n=args.k_n, k_travel=args.k_travel,
                              f_target=args.f_target, spring=name.endswith("spring"),
                              k_err=args.k_err)
    mod, fn = name.split(":")
    return getattr(importlib.import_module(mod), fn)(env)


def run_episode(env: WritingPolicyEnv, policy, spec: SM.TaskSpec) -> dict:
    obs = env.reset(spec)
    policy.reset(env.goal(), obs)
    while not env.time_up:
        a = policy.act(obs)
        if a is None:
            break
        obs = env.step(a)
    res = env.score()
    res["text"] = spec.text
    res["seed"] = spec.seed
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="scripted",
                    help="scripted | scripted_spring | replay | module:factory")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=10_000,
                    help="evaluation seeds; demos use 0.. so keep these disjoint")
    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--demos", default=None, help="for --policy replay")
    ap.add_argument("--k-t", type=float, default=2000.0)
    ap.add_argument("--k-n", type=float, default=400.0)
    ap.add_argument("--k-travel", type=float, default=800.0)
    ap.add_argument("--f-target", type=float, default=3.0)
    ap.add_argument("--k-err", type=float, default=1.0,
                    help="commanded k_n / reasoned k_n: a stiffness head that is wrong")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--video", type=int, default=0, help="save an mp4 for the first N episodes")
    ap.add_argument("--out", default=None, help="write per-episode results as JSON")
    args = ap.parse_args()

    env = WritingPolicyEnv(control_hz=args.control_hz, cameras=not args.no_cameras,
                           render_mode="rgb_array" if args.video else None)
    rows = []
    if args.policy == "replay":
        files = sorted(pathlib.Path(args.demos).glob("ep_*.h5"))[:args.episodes]
        runs = [(ReplayPolicy(p), None) for p in files]
        runs = [(p, p.spec) for p, _ in runs]
        env.n_sub = max(1, int(round(1.0 / (runs[0][0].hz * env.sim.dt)))) if runs else env.n_sub
    else:
        pol = load_policy(args.policy, env, args)
        runs = [(pol, SM.TaskSpec.sample(args.seed_offset + i)) for i in range(args.episodes)]
    for i, (pol, spec) in enumerate(runs):
        env.record_video = i < args.video
        r = run_episode(env, pol, spec)
        rows.append({k: v for k, v in r.items() if k != "checks"})
        print(f"  {spec.seed:6d} {spec.text:10s} ok={int(r['success'])} cov={r['coverage']:.2f} "
              f"prec={r['precision']:.2f} band={r['in_band']:.2f} peak={r['peak_force']:.1f} N "
              f"t={r['t']:.1f} s {r['fail_reason']}")
        if env.frames:
            import imageio.v2 as imageio
            imageio.mimsave(f"eval_{spec.seed:05d}.mp4", env.frames, fps=int(args.control_hz),
                            quality=7, macro_block_size=1)
    ok = np.array([r["success"] for r in rows])
    print(f"\nsuccess {ok.sum()}/{len(ok)} ({100 * ok.mean():.0f}%)")
    for key in ("coverage", "precision", "in_band", "contact_force", "peak_force", "chamfer_mm"):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            print(f"  {key:<12} {v.mean():7.3f} +- {v.std():6.3f}")
    from collections import Counter
    why = Counter(r for row in rows for r in row["fail_reason"].split(",") if r)
    if why:
        print("  failures: " + ", ".join(f"{k} x{v}" for k, v in why.most_common()))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
