"""Validation suite for the writing environment.

Run:  python3 tests.py     (plain asserts; pytest not required, ~1 min)

Same contract as ../cascade/tests.py: every check prints both numbers, so a
failure states the actual discrepancy.  The point-mass Case 1 is verified there
against closed forms; what is checked here is that the SAME law on the arm, the
ink rule, the success criterion and the teleoperation loop do what the rest of
this package assumes they do.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import controller as C  # noqa: E402
import glyphs as G  # noqa: E402
import sim as SM  # noqa: E402
import teleop as T  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name, got, want, tol, note=""):
    got_f, want_f = float(got), float(want)
    err = abs(got_f - want_f)
    rel = err / max(abs(want_f), 1e-9)
    ok = bool(err <= tol or rel <= tol)
    line = f"  {'PASS' if ok else 'FAIL'}  {name:58s} got={got_f: .5f} want={want_f: .5f} err={err:.2e}"
    if note:
        line += f"  [{note}]"
    print(line)
    (PASS if ok else FAIL).append(name)


def check_true(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:58s} {note}")
    (PASS if bool(cond) else FAIL).append(name)


def hold(sim, sec, x_target=None, K_target=None, v_max=0.02):
    """Drive x_d toward x_target at <= v_max, and K toward K_target in 0.2 s."""
    rec = None
    for _ in range(int(round(sec / sim.dt))):
        Vd = np.zeros(3)
        if x_target is not None:
            Vd = np.clip((x_target - sim.ctl.x_d) / sim.dt, -v_max, v_max)
        Up = None if K_target is None else (K_target - sim.ctl.K) / 0.2
        rec = sim.step(C.Case1Proposal(Vd, Up))
    return rec


# ========================================================================== #
print("\n--- glyphs ----------------------------------------------------------")
inside = all(np.all((np.concatenate(G.layout(c).strokes) >= -0.03) &
                    (np.concatenate(G.layout(c).strokes) <= 0.03)) for c in G.available())
check_true("every glyph lays out inside its box", inside, f"{len(G.available())} glyphs")
lo, hi = G.layout("HELLO").bbox()
check("layout is centred on the canvas origin", float(np.abs(lo + hi).max()), 0.0, 1e-12)
try:
    G.tokenize("A~")
    check_true("an unknown character raises", False)
except KeyError:
    check_true("an unknown character raises", True)

sim = SM.WritingSim(cameras=False)
W = sim.W

# ========================================================================== #
print("\n--- Case 1 on the arm -----------------------------------------------")
sim.reset(SM.TaskSpec(text="I"))
p0 = sim.last["p"].copy()
r = hold(sim, 1.0)
check("holds still: tip drift over 1 s (mm)", 1000 * np.linalg.norm(r["p"] - p0), 0.0, 0.02,
      "the reference is seeded at the measured tip, not the IK target")

for kn in (300.0, 1500.0):
    sim.reset(SM.TaskSpec(text="I"))
    sim.ctl.K = C.k_world([2000.0, 2000.0, kn], W)
    depth = 0.004
    hold(sim, 3.0, sim.frame.to_world([0.0, 0.0, -depth]))
    f = np.mean([hold(sim, 0.002)["f_n"] for _ in range(200)])
    check(f"static force = k_n x depth at k_n = {kn:.0f}", f, kn * depth, 0.05,
          "the paper is rigid, so K_p is the only compliance")

# stiffening a stretched spring costs exactly 1/2 p^T dK p
sim.reset(SM.TaskSpec(text="I"))
sim.ctl.K = C.k_world([2000.0, 2000.0, 400.0], W)
hold(sim, 2.5, sim.frame.to_world([0, 0, -0.005]))
hold(sim, 0.5)
E0 = sim.ctl.E
p_de = sim.last["p"] - sim.ctl.x_d
K0 = sim.ctl.K.copy()
Kt = C.k_world([2000.0, 2000.0, 1600.0], W)
hold(sim, 1.0, K_target=Kt)
want = 0.5 * float(p_de @ (sim.ctl.K - K0) @ p_de)
check("tank pays 1/2 p_de^T dK p_de for a stiffness change (J)", E0 - sim.ctl.E, want, 0.1,
      "a variable K_p is an energy source without the tank")

# the gate throttles, and requested != applied
sim.reset(SM.TaskSpec(text="I"))
sim.ctl.E = 0.2 * sim.ctl.g.Ec
x_req = sim.ctl.x_d + np.array([0.0, 0.0, -0.05])      # a 50 mm plunge into the paper
Es, alphas = [], []
for _ in range(1500):
    rec = sim.step(C.Case1Proposal((x_req - sim.ctl.x_d) / 0.25))
    Es.append(rec["E"]); alphas.append(rec["alpha"])
check_true("gate throttles once the tank runs low", min(alphas) < 0.2, f"alpha min {min(alphas):.3f}")
check_true("tank energy never goes negative", min(Es) >= -1e-9, f"E min {min(Es):.2e} J")
short = float(np.linalg.norm(x_req - sim.ctl.x_d))
check_true("applied x_d falls short of the request", short > 0.005, f"{1000 * short:.1f} mm short")

# stiffness bounds
sim.reset(SM.TaskSpec(text="I"))
hold(sim, 0.5, K_target=C.k_world([1e5, 1e5, 1e5], W))
check("K_p is clipped at k_hi", float(np.max(np.linalg.eigvalsh(sim.ctl.K))), sim.ctl.g.k_hi, 1e-6)

# ========================================================================== #
print("\n--- ink, tearing, scoring -------------------------------------------")
sim.reset(SM.TaskSpec(text="I"))
sim.ctl.K = C.k_world([2000.0, 2000.0, 300.0], W)
hold(sim, 2.5, sim.frame.to_world([0, 0, 0.003]))
check("hovering 3 mm up: no ink", len(sim.ink_uv), 0, 0)
r = hold(sim, 3.5, sim.frame.to_world([0, 0, -0.0015]), v_max=0.002)   # 300 x 1.5 mm = 0.45 N
check_true("pressing at 0.45 N (< ink_force): touching but no ink",
           0.2 < r["f_n"] < sim.crit.ink_force and len(sim.ink_uv) == 0,
           f"f_n {r['f_n']:.2f} N, {len(sim.ink_uv)} dots")
r = hold(sim, 1.0, sim.frame.to_world([0, 0, -0.010]))                 # 3 N
check_true("pressing at 3 N: ink", len(sim.ink_uv) > 0, f"f_n {r['f_n']:.2f} N, {len(sim.ink_uv)} dots")

sim.reset(SM.TaskSpec(text="I"))
sim.ctl.K = C.k_world([2000.0, 2000.0, 300.0], W)
hold(sim, 2.5, sim.frame.to_world([0, 0, 0.010]))
hold(sim, 2.0, sim.frame.to_world([0, 0, -0.010]), v_max=0.015)       # the writer's descent
check_true("landing at 15 mm/s is an impact, not a tear", not sim.torn,
           f"pressure peak {sim.peak:.1f} N, sensor peak {sim.peak_fast:.1f} N")
hold(sim, 2.0, sim.frame.to_world([0, 0, -0.050]))                     # 15 N sustained
check_true("sustained overload tears the paper", sim.torn, f"pressure peak {sim.peak:.1f} N")

sim.reset(SM.TaskSpec(text="HI"))
sim.ink_uv = list(sim.target_pts)
sim.pen_down_steps, sim.in_band_steps = 100, 100
check_true("perfect ink scores as success", sim.score()["success"])
sim.ink_uv = list(sim.target_pts + [0.005, 0.0])
s = sim.score()
check_true("ink 5 mm off the target fails", not s["success"],
           f"coverage {s['coverage']:.2f} precision {s['precision']:.2f}")

# ========================================================================== #
print("\n--- teleoperation ---------------------------------------------------")
import interactive as I  # noqa: E402


class FakeWindow:
    def __init__(self):
        self.down, self.shift = set(), False     # KeyboardHuman reads only these

    def key_down(self, k):
        return k in self.down

    def key_press(self, k):
        return False


sim.reset(SM.TaskSpec(text="I", canvas_dz=0.003, tilt_x=0.05))
sess = T.TeleopSession(sim)
win = FakeWindow()
hum = I.KeyboardHuman(win, W, [1500.0, 1500.0, 400.0])
win.down = {"u"}
for _ in range(int(4.0 / sim.dt)):
    f_h, k = hum.wrench(sim.t, sess.x_m, sess.v_m, sess.f_fb)
    r = sess.step(f_h, k)
check("force input: paper force at rest = the pushed force", r["f_n"], I.KeyboardHuman.F_NORMAL, 0.08,
      "friction and tilt take the rest")
check_true("keyboard: holding press from hover lands without tearing", not sim.torn,
           f"pressure peak {sim.peak:.1f} N (isotropic arm damping tore it at 15-16 N)")
for keys, sec in (({"u", "l"}, 1.0), (set(), 0.5)):
    win.down = keys
    for _ in range(int(sec / sim.dt)):
        f_h, k = hum.wrench(sim.t, sess.x_m, sess.v_m, sess.f_fb)
        sess.step(f_h, k)
n0 = len(sim.ink_uv)
win.down = {"i"}
for _ in range(int(1.0 / sim.dt)):
    f_h, k = hum.wrench(sim.t, sess.x_m, sess.v_m, sess.f_fb)
    r = sess.step(f_h, k)
check_true("keyboard: releasing press lifts the pen (no stray ink)", len(sim.ink_uv) == n0,
           f"{len(sim.ink_uv) - n0} dots while travelling 0.5 s after release, "
           f"tip {1000 * sim.frame.to_canvas(r['p'])[2]:.1f} mm up")

# one synthetic episode, recorded, then replayed from its labels
import collect  # noqa: E402

spec = SM.TaskSpec(text="L", canvas_dz=-0.002, tilt_x=0.04, tilt_y=-0.03, friction=0.4)
rc = collect.Recorder(sim, images=False)
gap = [0.0]


def hook(i, rec, sess, wr):
    rc.on_step(i, rec, sess, wr)
    gap[0] = max(gap[0], float(np.abs(rec["x_d_next"] - rec["x_d_req"]).max()))


res = T.run_synthetic(sim, spec, T.WriterStyle(), on_step=hook)
check_true("synthetic writer writes 'L' successfully", res["success"],
           f"coverage {res['coverage']:.2f} precision {res['precision']:.2f} "
           f"in-band {res['in_band']:.2f} peak {res['peak_force']:.1f} N")
check("gate open: applied x_d == requested x_d, every step (m)", gap[0], 0.0, 1e-9,
      "so the stored applied label IS the operator's request")
kd = np.asarray(rc.full["k_diag"])
check_true("K_p actually varied during the demonstration",
           kd[:, 2].max() / kd[:, 2].min() > 2.0 and kd[:, 0].max() / kd[:, 0].min() > 1.5,
           f"k_n {kd[:, 2].min():.0f}-{kd[:, 2].max():.0f}, k_t {kd[:, 0].min():.0f}-{kd[:, 0].max():.0f} N/m")
with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / "ep.h5"
    rc.save(path, dict(spec=json.dumps(dataclasses.asdict(spec))))
    rp = collect.replay(sim, path, "recorded")
check_true("replaying the stored (x_d, K) reproduces the demonstration",
           rp["success"] and rp["force_rmse"] < 0.5,
           f"force RMSE {rp['force_rmse']:.2f} N, coverage {rp['coverage']:.2f}")

# ========================================================================== #
print("\n--- stiffness protocol ----------------------------------------------")
import types  # noqa: E402

import protocol as P  # noqa: E402

pargs = types.SimpleNamespace(dwell=1.0, ramp=0.15, grace=0.6, v_desc=0.010, v_travel=0.030)
rc = collect.Recorder(sim, images=False)
spec_p, style_p, seed_p = P.episode_spec("7", 1, 0, 60_000)
res = P.run_episode(sim, spec_p, style_p, P.AutoUser(seed_p), P.Levels(), pargs, recorder=rc, cues=False)
comp = res["compliance"]
check_true("protocol: a person following it writes '7' successfully", res["success"],
           f"coverage {res['coverage']:.2f} in-band {res['in_band']:.2f} peak {res['peak_force']:.1f} N")
check("protocol: compliance of a person who follows it", comp["overall"], 1.0, 1e-9)
kl, kd, ph = (np.asarray(rc.full[k]) for k in ("k_level", "k_diag", "phase"))
contact = np.isin(ph, [T.PHASES.index(p) for p in ("stroke",)])
check("protocol: K along the paper while writing = xy LOW (N/m)", kd[contact, 0].mean(), P.Levels().xy[P.LOW], 0.01)
check("protocol: K into the paper while writing = z MID (N/m)", kd[contact, 2].mean(), P.Levels().z[P.MID], 0.01)
check_true("protocol: the discrete levels are recorded", set(map(tuple, kl.tolist())) >= {(1, 1), (1, 2), (0, 1)},
           f"levels seen {sorted(set(map(tuple, kl.tolist())))}")

sim.close()
print(f"\n{'=' * 100}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES: " + ", ".join(FAIL))
    raise SystemExit(1)
print("all checks passed\n")
