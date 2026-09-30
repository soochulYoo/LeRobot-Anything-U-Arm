"""The spiral wiping task, driven by all three executors: cascade, Case 1, Case 2.

WHY THIS IS THE RIGHT COMPARISON
---------------------------------
analytic.py compares the three on a quasistatic push, where the answer is a
series-stiffness count and Case 1 and Case 2 are indistinguishable (both
1 + K/ke).  That comparison cannot separate them, because nothing in it moves.

The spiral wipe separates them, for two reasons the static test has no access to:

  - The surface is a random height field the operator cannot see, so the
    follower is continuously disturbed.  How a follower REJECTS that
    disturbance is a property of its structure, not of its steady state.
  - The operator is itself a delayed feedback controller (200 ms visual,
    60 ms haptic).  A follower that fights the surface hands the operator a
    correction problem they are too slow to solve, so the cost of a bad
    follower is amplified rather than averaged away.

Every run shares the SAME operator (spiral_task.make_operator_target), the same
surface seed, the same wall, the same dt and the same commanded force.  The only
thing that changes is what sits between the operator's command and the tool:

    cascade   coupling Ka -> outer admittance (Ma, Br) -> inner impedance Ki
    Case 1    the command IS the GIC equilibrium x_d;  Kp faces the surface
              directly.  No admittance, no inner/outer split.
    Case 2    coupling Ka -> outer admittance (Ma, Br) -> first-order lag servo
              Ts with velocity/acceleration limits.  No inner law at all.

GAIN MATCHING, and why it is the honest choice
-----------------------------------------------
Ka = Kp = 300 N/m in all three: "the gain that faces the environment" is held
fixed, and the cascade's extra inner stiffness Ki = 4000 is the thing under
test.  Case 2 inherits the cascade's admittance (Ma, Ba, Br) unchanged, so
cascade-vs-Case-2 isolates the inner loop and Case-1-vs-cascade isolates the
outer loop.  Any other matching would confound the two.

The operator's haptic loop closes on force, so all three converge to roughly the
intended 8 N mean -- that is the operator doing its job, not the followers being
equivalent.  The comparison therefore lives in the VARIABILITY and the PATH,
which is exactly what study_cascade_spiral.py measures for its own ablation.

THE TANK IS OFF BY DEFAULT HERE.  Case 1's gate is its distinguishing feature,
but leaving it armed would throttle the reference and confound a
disturbance-rejection comparison with a reference-tracking one.  `tank_E0`
exposes it; study_case_spiral.py runs it as a separate panel.
"""
from __future__ import annotations

import dataclasses

import numpy as np

import case1 as C1
import case2 as C2
import core as C
import spiral_task as S

Array = np.ndarray

CONTROLLERS = ["cascade", "Case 1", "Case 2"]


def _env_and_surface(task: S.SpiralTask, surface: S.RandomSurface):
    """The shared environment: tilted plate, friction, random height field."""
    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]

    def slope(_t, x):
        g1, g2 = surface.grad(float(x @ t1), float(x @ t2))
        return g1 * t1 + g2 * t2

    env = C.EnvParams(ke=task.ke, de=task.de, x_wall=task.wall, normal=n,
                      mu=task.mu, slope_fn=slope)
    offset = lambda t, x: surface.height(float(x @ t1), float(x @ t2))
    return env, offset, (t1, t2, n)


def common_dt(task: S.SpiralTask) -> float:
    """One dt for all three executors.

    Not a detail: the operator's haptic loop INTEGRATES a rate, so a different
    step would make it a different operator.  The cascade has the stiffest
    element (Ki = 4000), so its safe step is the binding one and the two cases
    inherit it well inside their own limits.
    """
    base = C.CascadeParams(
        n=3, coupling=C.CouplingParams(Ka=task.Ka, Ba=task.Ba),
        outer=C.OuterParams(Ma=task.Ma, Br=task.Br),
        inner=C.InnerParams(Ki=task.Ki, Di=task.Di, Ms=task.Ms),
        env=C.EnvParams(ke=task.ke, de=task.de, x_wall=task.wall),
        human=C.HumanParams(Kh=900.0, Bh=25.0), master=C.MasterParams(Mm=2.0, Bm=10.0))
    return C.safe_dt(base)


# --------------------------------------------------------------------------- #
# Case 1: the operator's command IS the GIC equilibrium
# --------------------------------------------------------------------------- #
def demonstrate_case1(task, intent, op, surface, dt=None, tank_E0=1e6, zeta=0.8) -> dict:
    env, offset, (t1, t2, n) = _env_and_surface(task, surface)
    dt = common_dt(task) if dt is None else dt
    Kp = task.Ka                                   # the gain facing the surface
    D = 2.0 * zeta * np.sqrt(Kp * task.Ms)         # critically-ish damped
    p = C1.Case1Params(Ms=task.Ms, Kp=Kp, D=D, tank=True, E0=tank_E0, Ec=1.0,
                       env=env, n=3, dt=dt)
    sim = C1.Case1Sim(p, wall_offset_fn=offset)
    sim.x = task.wall * n.copy()
    sim.x_d = task.wall * n.copy()

    target, dt_holder = S.make_operator_target(task, intent, op)
    dt_holder["dt"] = dt
    vis = C.DelayLine(op.visual_delay, dt, 3)
    hap = C.DelayLine(op.haptic_delay, dt, 3)
    rng = np.random.default_rng(op.seed)

    recs = []
    n_steps = int(round((task.settle + intent.duration + 1.0) / dt))
    f_e = np.zeros(3)
    for _ in range(n_steps):
        # Delayed observations, built exactly as core.CascadeSim builds them.
        obs = {"x": vis.push_and_get(sim.x), "f": hap.push_and_get(f_e),
               "t": max(0.0, sim.t - op.visual_delay)}
        tgt = np.asarray(target(sim.t, obs), dtype=float).reshape(3)
        # Signal-dependent motor noise, on the COMMAND -- the operator's own
        # tremor, the same mechanism core.py applies to f_h.
        if op.motor_noise:
            tgt = tgt + rng.normal(0.0, op.motor_noise * 0.001 * float(np.linalg.norm(tgt)), 3)
        # Deadbeat reference decoder: V_d^prop carries the command into the
        # applied equilibrium in one step, so that WITH the gate open x_d == tgt
        # and with the gate closed the shortfall is visible rather than hidden.
        Vd = (tgt - sim.x_d) / dt
        rec = sim.step(C1.Case1Proposal(np.zeros(3), Vd))
        f_e = rec["f_e"]
        recs.append({"t": rec["t"], "x": rec["x"], "x_cmd": rec["x_d"],
                     "f_normal": rec["f_normal"], "f_cmd_n": float(-rec["f_G"] @ n),
                     "alpha": rec["alpha"], "E": rec["E"]})
    return _pack(recs, task, intent, "Case 1")


# --------------------------------------------------------------------------- #
# Case 2: the operator's command IS the outer coupling command x_c
# --------------------------------------------------------------------------- #
def demonstrate_case2(task, intent, op, surface, dt=None, Ts=0.020,
                      v_limit=0.10, a_limit=0.50, delay_ticks=0) -> dict:
    env, offset, (t1, t2, n) = _env_and_surface(task, surface)
    dt = common_dt(task) if dt is None else dt
    p = C2.Case2Params(Ma=task.Ma, Ka=task.Ka, Ba=task.Ba, Br=task.Br, Ts=Ts,
                       v_limit=v_limit, a_limit=a_limit, delay_ticks=delay_ticks,
                       env=env, n=3, dt=dt)
    sim = C2.Case2Sim(p, wall_offset_fn=offset)
    sim.x_r = task.wall * n.copy()
    sim.x_s = task.wall * n.copy()

    target, dt_holder = S.make_operator_target(task, intent, op)
    dt_holder["dt"] = dt
    vis = C.DelayLine(op.visual_delay, dt, 3)
    hap = C.DelayLine(op.haptic_delay, dt, 3)
    rng = np.random.default_rng(op.seed)

    recs = []
    n_steps = int(round((task.settle + intent.duration + 1.0) / dt))
    f_e = np.zeros(3)
    x_c_prev = task.wall * n.copy()
    for _ in range(n_steps):
        obs = {"x": vis.push_and_get(sim.x_s), "f": hap.push_and_get(f_e),
               "t": max(0.0, sim.t - op.visual_delay)}
        tgt = np.asarray(target(sim.t, obs), dtype=float).reshape(3)
        if op.motor_noise:
            tgt = tgt + rng.normal(0.0, op.motor_noise * 0.001 * float(np.linalg.norm(tgt)), 3)
        v_c = (tgt - x_c_prev) / dt
        rec = sim.step(tgt, v_c)
        x_c_prev = tgt
        f_e = rec["f_e"]
        recs.append({"t": rec["t"], "x": rec["x_s"], "x_cmd": rec["x_c"],
                     "f_normal": rec["f_normal"], "f_cmd_n": float(rec["f_ch"] @ n),
                     "alpha": 1.0, "E": np.nan, "r_M": rec["r_M"],
                     "clipped": rec["clipped"], "c_inv": float(rec["c_inv"] @ n)})
    return _pack(recs, task, intent, "Case 2")


# --------------------------------------------------------------------------- #
# cascade, via the existing driver, remapped onto the same keys
# --------------------------------------------------------------------------- #
def demonstrate_cascade(task, intent, op, surface, dt=None) -> dict:
    log = S.demonstrate(task, intent, op, surface, "cascade")
    n = task.R[:, 2]
    recs = [{"t": float(t), "x": x, "x_cmd": xc, "f_normal": float(fn),
             "f_cmd_n": float(fch @ n), "alpha": 1.0, "E": np.nan}
            for t, x, xc, fn, fch in zip(log["t"], log["x"], log["x_c"],
                                         np.asarray(log["f_normal"])[:, 0], log["f_ch"])]
    out = _pack(recs, task, intent, "cascade")
    out["_dt"] = log["_dt"]
    return out


def demonstrate(task, intent, op, surface, controller: str, **kw) -> dict:
    if controller == "cascade":
        return demonstrate_cascade(task, intent, op, surface, **kw)
    if controller == "Case 1":
        return demonstrate_case1(task, intent, op, surface, **kw)
    if controller == "Case 2":
        return demonstrate_case2(task, intent, op, surface, **kw)
    raise ValueError(f"unknown controller {controller!r}")


def _pack(recs: list[dict], task, intent, controller: str) -> dict:
    out = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    out["_task"], out["_intent"], out["_controller"] = task, intent, controller
    return out


# --------------------------------------------------------------------------- #
# metrics, identical for all three
# --------------------------------------------------------------------------- #
def gap(log: dict) -> dict:
    """Intent-vs-result on the unified log.  Same four quantities and the same
    nearest-point-on-curve path metric spiral_task.gap uses, so numbers here are
    directly comparable to the numbers that package already reports."""
    task, intent = log["_task"], log["_intent"]
    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
    t = log["t"]
    sel = t >= task.settle
    curve = intent.curve(task.settle, t1, t2)

    def path_rms(pos: Array, chunk: int = 512) -> float:
        """RMS distance to the nearest point on the intended spiral.

        CHUNKED deliberately.  The obvious one-liner allocates an
        (N_samples, N_curve, 3) tensor -- at 19k samples against a 3k-point
        curve that is 1.4 GB for a single call, which is fine once and an
        out-of-memory kill the moment a few episodes run in parallel.  Chunking
        bounds the peak at (chunk, N_curve) and changes no result.
        """
        p = pos - task.wall * n
        p = p - np.outer(p @ n, n)
        best = np.empty(len(p))
        for i in range(0, len(p), chunk):
            blk = p[i:i + chunk]
            d = np.linalg.norm(blk[:, None, :] - curve[None, :, :], axis=2)
            best[i:i + chunk] = d.min(axis=1)
        return float(np.sqrt(np.mean(best ** 2)))

    f = np.asarray(log["f_normal"])[sel]
    return {
        "path_result": path_rms(np.asarray(log["x"])[sel]),
        "path_command": path_rms(np.asarray(log["x_cmd"])[sel]),
        "force_result": float(np.sqrt(np.mean((f - intent.f_normal) ** 2))),
        "force_command": float(np.sqrt(np.mean(
            (np.asarray(log["f_cmd_n"])[sel] - intent.f_normal) ** 2))),
        "contact": float(np.mean(f > 0.1 * intent.f_normal)),
        "f_mean": float(np.mean(f)),
        "f_std": float(np.std(f)),
        "f_p95": float(np.percentile(f, 95)),
    }
