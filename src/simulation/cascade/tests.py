"""Validation suite: core.py's integrator against analytic.py's closed forms,
plus the invariants that must hold for the physics to be right.

Run:  python3 tests.py        (plain asserts; pytest not required)

Every check prints both numbers, so a failure states the actual discrepancy.
Tolerances are tight on purpose: the steady state of semi-implicit Euler is an
EXACT fixed point of the continuous system (x_{k+1}=x_k forces v*=0, which
forces a(x*,0)=0), so a loose tolerance here would hide a real sign or
step-ordering error rather than a discretisation artefact.
"""
from __future__ import annotations

import dataclasses

import numpy as np

import analytic as A
import core as C

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, got, want, tol: float, note: str = "") -> None:
    got_f = float(np.asarray(got).ravel()[0])
    want_f = float(np.asarray(want).ravel()[0])
    err = abs(got_f - want_f)
    rel = err / max(abs(want_f), 1e-9)
    ok = bool(err <= tol or rel <= tol)
    line = f"  {'PASS' if ok else 'FAIL'}  {name:50s} got={got_f: .6f} want={want_f: .6f} err={err:.2e}"
    if note:
        line += f"  [{note}]"
    print(line)
    (PASS if ok else FAIL).append(name)


def check_true(name: str, cond: bool, note: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name:50s} {note}")
    (PASS if bool(cond) else FAIL).append(name)


def make(dt=None, **kw) -> C.CascadeParams:
    """Build params and pick a STABLE dt automatically, so changing a gain in a
    test can never cross the stability boundary and report an integrator
    failure as if it were a physics failure."""
    base = C.CascadeParams(**kw)
    return dataclasses.replace(base, dt=C.safe_dt(base) if dt is None else dt)


def settle(sim: C.CascadeSim, duration: float, action_fn=None, window: float = 1.0):
    log = sim.run(duration, action_fn)
    return {k: C.steady_state_of(log, k, window, sim.dt) for k in log}, log


def const(x_ref: float, f_d: float, Ko: float):
    act = C.Action(np.array([x_ref]), np.array([f_d]), np.array([Ko]))
    return act, (lambda t, last: act)


# ========================================================================== #
print("\n[1] TELEOP steady state vs analytic.contact_force_teleop")
print("    full chain: human -> master -> coupling -> admittance -> inner -> slave -> wall")
for Kh, Ka, Ki, ke, reach, wall in [
    (400, 100, 2000, 5000, 0.16, 0.10),
    (400, 300, 2000, 5000, 0.16, 0.10),    # stiffer coupling
    (800, 100, 2000, 5000, 0.20, 0.08),    # stiffer human, deeper reach
    (200, 100, 8000, 20000, 0.14, 0.10),   # stiff inner + stiff wall
    (400, 100, 2000, 5000, 0.30, 0.05),    # deep push
]:
    p = make(human=C.HumanParams(Kh=Kh, reach=reach),
             coupling=C.CouplingParams(Ka=Ka),
             inner=C.InnerParams(Ki=Ki),
             env=C.EnvParams(ke=ke, x_wall=wall))
    ss, _ = settle(C.CascadeSim(p, "teleop"), 25.0)
    want = A.contact_force_teleop(Kh, reach, Ka, Ki, ke, wall)
    check(f"Kh={Kh} Ka={Ka} Ki={Ki} ke={ke} reach={reach}", ss["f_e"], want, 2e-3)

# ========================================================================== #
print("\n[2] POLICY steady state vs analytic.contact_force_policy")
print("    the Ko/Ki term is the one a single-layer (impedance-only) analysis has no place for")
for f_d, x_ref, Ko, Ki, Di, ke, wall in [
    (10.0, 0.10, 1000, 2000, 80, 5000, 0.10),
    (10.0, 0.10, 200, 2000, 80, 5000, 0.10),        # soft outer -> bias shrinks
    (10.0, 0.10, 1000, 2000, 80, 50000, 0.10),      # stiff wall
    (5.0, 0.12, 400, 4000, 110, 5000, 0.10),
    (10.0, 0.10, 1000, 100000, 600, 5000, 0.10),    # near-rigid inner
]:
    p = make(inner=C.InnerParams(Ki=Ki, Di=Di), env=C.EnvParams(ke=ke, x_wall=wall))
    _, fn = const(x_ref, f_d, float(Ko))
    ss, _ = settle(C.CascadeSim(p, "policy"), 20.0, fn)
    want = A.contact_force_policy(f_d, x_ref, Ko, Ki, ke, wall)
    check(f"f_d={f_d} Ko={Ko} Ki={Ki:.0e} ke={ke} (dt={p.dt:.0e})", ss["f_e"], want, 2e-3)

rigid = A.contact_force_policy(10.0, 0.10, 1000, np.inf, 5000, 0.10)
near = A.contact_force_policy(10.0, 0.10, 1000, 100000, 5000, 0.10)
check("near-rigid inner -> single-layer limit", near, rigid, 1.5e-2,
      f"residual Ko/Ki bias {100*(1-near/rigid):.2f}%")
soft_inner = A.contact_force_policy(10.0, 0.10, 1000, 2000, 5000, 0.10)
print("    DESIGN CONSEQUENCE: at Ki=2000, Ko=1000 the inner loop adds Ko/Ki=0.50 of")
print(f"    series compliance against the wall's Ko/ke=0.20, so a commanded 10 N lands")
print(f"    at {soft_inner:.3f} N. Ki >> Ko is a requirement, not a preference.")

# ========================================================================== #
print("\n[3] EP and force parameterisations are the SAME controller")
print("    x_vt = x_ref + f_d/Ko must give a bit-for-bit identical trajectory")
p = make(env=C.EnvParams(x_wall=0.10))
Ko, f_d, x_ref = 1000.0, 10.0, 0.10
_, fn_force = const(x_ref, f_d, Ko)
_, fn_ep = const(x_ref + f_d / Ko, 0.0, Ko)
log_f = C.CascadeSim(p, "policy").run(5.0, fn_force)
log_e = C.CascadeSim(p, "policy").run(5.0, fn_ep)
dev = float(np.max(np.abs(log_f["f_e"] - log_e["f_e"])))
check_true("identical f_e trajectory", dev < 1e-12, f"max|df| = {dev:.2e} N")

# ========================================================================== #
print("\n[4] GAUGE: the label split must not double-count the coupling force")
print("    demo labels f_d := f_ch;  x_ref := x_r (correct) vs x_ref := x_c (buggy)")
p = make(human=C.HumanParams(Kh=400, reach=0.16),
         coupling=C.CouplingParams(Ka=100),
         inner=C.InnerParams(Ki=2000),
         env=C.EnvParams(ke=5000, x_wall=0.10))
ss_demo, _ = settle(C.CascadeSim(p, "teleop"), 25.0)
f_demo = float(ss_demo["f_e"][0])
f_ch_demo = float(ss_demo["f_ch"][0])
x_r_demo = float(ss_demo["x_r"][0])
x_c_demo = float(ss_demo["x_c"][0])
print(f"    demo: f_e={f_demo:.4f} N  f_ch={f_ch_demo:.4f} N  x_r={x_r_demo:.6f}  x_c={x_c_demo:.6f}")
check("demo quasi-static: f_ch == f_e", f_ch_demo, f_demo, 5e-3, "series chain at rest")

for label, x_lab in [("x_ref := x_r  (correct gauge)", x_r_demo),
                     ("x_ref := x_c  (double count)", x_c_demo)]:
    act, fn = const(x_lab, f_ch_demo, 100.0)
    ss_rep, _ = settle(C.CascadeSim(p, "policy"), 25.0, fn)
    want = A.contact_force_policy(f_ch_demo, x_lab, 100.0, 2000.0, 5000.0, 0.10)
    check(label, ss_rep["f_e"], want, 2e-3, f"replay/demo = {float(ss_rep['f_e'][0])/f_demo:.2f}x")

act_ok, fn_ok = const(x_r_demo, f_ch_demo, 100.0)
ss_ok, _ = settle(C.CascadeSim(p, "policy"), 25.0, fn_ok)
check("correct gauge reproduces the demo force", ss_ok["f_e"], f_demo, 5e-3)
share = float(act_ok.spring_share(ss_ok["x_r"])[0])
check_true("gauge diagnostic ~0 at steady state", share < 0.02, f"spring_share = {share:.4f}")

# ========================================================================== #
print("\n[5] CONTACT INVARIANTS")
p = make(human=C.HumanParams(reach=0.30), env=C.EnvParams(x_wall=0.05))
log = C.CascadeSim(p, "teleop").run(12.0)
check_true("no adhesion: f_e >= 0 everywhere", bool(np.all(log["f_e"] >= 0.0)),
           f"min f_e = {log['f_e'].min():.3e} N")
check_true("f_e == 0 whenever not penetrating",
           bool(np.all(log["f_e"][log["x"] <= p.env.x_wall] == 0.0)))
p_free = make(human=C.HumanParams(reach=0.02), env=C.EnvParams(x_wall=0.50))
check_true("free space: never any contact force",
           bool(np.all(C.CascadeSim(p_free, "teleop").run(6.0)["f_e"] == 0.0)))

p = make(inner=C.InnerParams(Ki=2000), env=C.EnvParams(x_wall=0.10))
_, fn = const(0.11, 8.0, 500.0)
ss, _ = settle(C.CascadeSim(p, "policy"), 25.0, fn)
check("inner loop at rest: f_cmd == f_e", ss["f_cmd"], ss["f_e"], 2e-3)
check("inner deflection: f_e == Ki*(x_r - x)", 2000.0 * (ss["x_r"] - ss["x"]), ss["f_e"], 2e-3)

# ========================================================================== #
print("\n[6] FREE-SPACE RUNAWAY and the velocity clamp")
print("    a commanded force with nothing to push against accelerates without bound;")
print("    OuterParams.v_limit bounds approach speed regardless of policy error")
p_run = make(env=C.EnvParams(x_wall=10.0))              # wall effectively absent
_, fn = const(0.0, 10.0, 0.0)                            # pure force, Ko = 0
log_run = C.CascadeSim(p_run, "policy").run(4.0, fn)
v_free = float(np.abs(log_run["v_r"]).max())
check("runaway speed ~ f_d/Br at 4 s", v_free, 10.0 / 25.0, 5e-2,
      "asymptote f_d/Br = 0.40 m/s")
check_true("runaway is monotone (no bound of its own)",
           bool(np.all(np.diff(log_run["v_r"].ravel()) >= -1e-12)),
           f"reached {v_free:.3f} m/s")

p_clamp = make(outer=C.OuterParams(v_limit=0.05), env=C.EnvParams(x_wall=10.0))
log_clamp = C.CascadeSim(p_clamp, "policy").run(4.0, fn)
v_clamped = float(np.abs(log_clamp["v_r"]).max())
check_true("v_limit bounds it", v_clamped <= 0.05 + 1e-12, f"max |v_r| = {v_clamped:.4f} m/s")
check_true("clamp actually binds here", v_clamped < v_free, f"{v_clamped:.3f} < {v_free:.3f} m/s")

# ========================================================================== #
print("\n[7] DELAY changes the transient, not the steady state")
base = dict(human=C.HumanParams(Kh=400, reach=0.16), env=C.EnvParams(x_wall=0.10))
ss0, _ = settle(C.CascadeSim(make(**base, channel=C.ChannelParams(0.00, 0.00)), "teleop"), 30.0)
ssd, logd = settle(C.CascadeSim(make(**base, channel=C.ChannelParams(0.05, 0.05)), "teleop"), 30.0)
check("50+50 ms delay: same steady-state force", ssd["f_e"], ss0["f_e"], 5e-3)
check_true("delay does perturb the transient",
           float(logd["f_e"].max()) > float(ss0["f_e"][0]) * 1.02,
           f"peak {float(logd['f_e'].max()):.3f} N vs ss {float(ss0['f_e'][0]):.3f} N")

# ========================================================================== #
print("\n[8] dt REFINEMENT: steady state dt-independent, transient converges")
peaks: dict[float, float] = {}
for dt in (2e-3, 1e-3, 5e-4, 2.5e-4):
    p = make(dt=dt, human=C.HumanParams(reach=0.16), env=C.EnvParams(x_wall=0.10))
    ss, log = settle(C.CascadeSim(p, "teleop"), 25.0)
    peaks[dt] = float(log["f_e"].max())
    check(f"dt={dt:.1e} steady state", ss["f_e"],
          A.contact_force_teleop(400, 0.16, 100, 2000, 5000, 0.10), 3e-3)
d = [abs(peaks[1e-3] - peaks[2e-3]), abs(peaks[5e-4] - peaks[1e-3]), abs(peaks[2.5e-4] - peaks[5e-4])]
print(f"    peak-force successive differences: {d[0]:.3e} -> {d[1]:.3e} -> {d[2]:.3e} N")
check_true("transient differences shrink monotonically", d[0] > d[1] > d[2],
           f"ratios {d[0]/max(d[1],1e-15):.2f}, {d[1]/max(d[2],1e-15):.2f} (~2 = first order)")

# ========================================================================== #
print("\n[9] GUARDS fail loudly")
guards = [
    ("unstable dt", lambda: C.CascadeSim(dataclasses.replace(C.CascadeParams(), dt=0.05), "teleop")),
    ("policy mode without action", lambda: C.CascadeSim(make(), "policy").step(None)),
    ("teleop mode given an action",
     lambda: C.CascadeSim(make(), "teleop").step(C.Action(np.zeros(1), np.zeros(1), np.zeros(1)))),
    ("bad gain shape", lambda: C._vec([1.0, 2.0], 1, "Ko")),
    ("negative delay", lambda: C.DelayLine(-0.1, 1e-3, 1)),
    ("duration shorter than dt", lambda: C.CascadeSim(make(), "teleop").run(1e-9)),
]
for desc, fn in guards:
    try:
        fn()
        check_true(f"{desc} is rejected", False, "NO exception raised")
    except ValueError as e:
        check_true(f"{desc} is rejected", True, str(e)[:60])

# ========================================================================== #
print("\n[10] LOG CONSISTENCY: every logged force reproducible from the logged state")
print("     a log whose force and pose are one step apart silently biases any")
print("     regression-based label rule, so this is a permanent check, not a one-off")
import labels as L  # noqa: E402  (imported here so [1]-[9] run even if labels.py breaks)

pc = make(human=C.HumanParams(Kh=400, reach=0.16, reach_amp=0.03, reach_hz=0.2),
          master=C.MasterParams(probe_amp=3.0, probe_hz=3.0),
          coupling=C.CouplingParams(Ka=100), inner=C.InnerParams(Ki=2000, Di=80),
          env=C.EnvParams(ke=5000, de=10, x_wall=0.10))
lg = C.CascadeSim(pc, "teleop").run(20.0)
tgt = (pc.human.reach + pc.human.reach_amp * np.sin(2 * np.pi * pc.human.reach_hz * lg["t"]))[:, None]
identities = {
    "f_h  == Kh(target-x_m) - Bh v_m": lg["f_h"] - (400 * (tgt - lg["x_m"]) - 20 * lg["v_m"]),
    "f_ch == Ka(x_c-x_r) + Ba(v_c-v_r)": lg["f_ch"] - (100 * (lg["x_c"] - lg["x_r"]) + 25 * (lg["v_c"] - lg["v_r"])),
    "f_cmd == Ki(x_r-x) + Di(v_r-v)": lg["f_cmd"] - (2000 * (lg["x_r"] - lg["x"]) + 80 * (lg["v_r"] - lg["v"])),
    "f_e  == max(0, ke*pen + de*v)": lg["f_e"] - np.maximum(
        0.0, np.where(lg["x"] > 0.10, 5000 * (lg["x"] - 0.10) + 10 * lg["v"], 0.0)),
}
for name, resid in identities.items():
    r = float(np.abs(resid).max())
    check_true(name, r < 1e-12, f"max residual {r:.2e}")

# ========================================================================== #
print("\n[11] LABEL RULES return the quantity the analysis predicts")
print("     the negative result is the point: R3/R4 track controller gains, not Kh")
est_by_Kh: dict[float, dict[str, float]] = {}
for Kh in (200.0, 400.0, 1200.0):
    pp = make(human=C.HumanParams(Kh=Kh, reach=0.16, reach_amp=0.03, reach_hz=0.2),
              master=C.MasterParams(probe_amp=3.0, probe_hz=3.0),
              coupling=C.CouplingParams(Ka=100), inner=C.InnerParams(Ki=2000),
              env=C.EnvParams(ke=5000, x_wall=0.10))
    dd = L.Demo(log=C.CascadeSim(pp, "teleop").run(60.0), dt=pp.dt, Kh_true=Kh,
                Ka=100, Ki=2000, ke=5000, settle_s=20.0)
    got = {}
    for tag, fn in [("R3", L.rule_impact_controller_rule),
                    ("R4", L.rule_compliance_for_free),
                    ("R4b", lambda d: L.rule_compliance_for_free(d, against="x_r")),
                    ("R6", L.rule_probe_identification)]:
        e = fn(dd)
        got[tag] = e.value
        check(f"Kh={Kh:.0f}  {tag} -> {e.recovers}", e.value, e.predicted, 2e-2)
    est_by_Kh[Kh] = got

for tag, what in [("R3", "Ki"), ("R4", "series(Ka,Ki)"), ("R4b", "Ka")]:
    vals = np.array([est_by_Kh[k][tag] for k in est_by_Kh])
    spread = float(vals.max() / vals.min() - 1.0)
    check_true(f"{tag} is INDEPENDENT of Kh (returns {what})", spread < 0.02,
               f"6x change in Kh moves it by {100*spread:.2f}%")
r6 = np.array([est_by_Kh[k]["R6"] for k in est_by_Kh])
khs = np.array(list(est_by_Kh.keys()))
check_true("R6 DOES track Kh", bool(np.allclose(r6, khs, rtol=2e-2)),
           f"est {np.round(r6,1)} vs true {khs}")

# ========================================================================== #
print("\n[12] EXCITATION is necessary: no probe -> no information about Kh")
for tag, amp in [("no probe", 0.0), ("probe", 3.0)]:
    pp = make(human=C.HumanParams(Kh=400, reach=0.16, reach_amp=0.03, reach_hz=0.2),
              master=C.MasterParams(probe_amp=amp, probe_hz=3.0), env=C.EnvParams(x_wall=0.10))
    dd = L.Demo(log=C.CascadeSim(pp, "teleop").run(60.0), dt=pp.dt, Kh_true=400,
                Ka=100, Ki=2000, ke=5000, settle_s=20.0)
    e = L.rule_probe_identification(dd)
    exc = e.diagnostics.get("excitation_amp_m", np.nan)
    if amp == 0.0:
        check_true("no probe -> R6 refuses to answer", not np.isfinite(e.value),
                   f"excitation {exc:.1e} m, returned {e.value}")
    else:
        check(f"probe -> R6 recovers Kh exactly", e.value, 400.0, 5e-3, f"excitation {exc:.1e} m")
        check("probe -> R6 recovers Bh too", e.diagnostics["Bh_hat"], 20.0, 5e-3)

# R7 works only because a constant intent fits in one intercept.
for tag, ra, want_ok in [("constant intent", 0.0, True), ("varying intent", 0.03, False)]:
    pp = make(human=C.HumanParams(Kh=400, reach=0.16, reach_amp=ra, reach_hz=0.2),
              master=C.MasterParams(probe_amp=3.0, probe_hz=3.0), env=C.EnvParams(x_wall=0.10))
    dd = L.Demo(log=C.CascadeSim(pp, "teleop").run(60.0), dt=pp.dt, Kh_true=400,
                Ka=100, Ki=2000, ke=5000, settle_s=20.0)
    v = L.rule_naive_regression_with_intercept(dd).value
    ok = abs(v - 400.0) / 400.0 < 0.02
    check_true(f"R7 naive regression, {tag}: {'exact' if want_ok else 'FAILS as predicted'}",
               ok == want_ok, f"estimate {v:.1f} vs true 400")

# ========================================================================== #
print("\n[13] 3-D: tilted surface, matrix gains, friction")
print("     the anisotropy study lives or dies on these, so they are checked here")


def rot_y(th):
    c, s_ = np.cos(th), np.sin(th)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def make3(Ko, Ki, normal, wall, mu=0.0, Di=80.0, dt=None):
    base = C.CascadeParams(
        n=3,
        outer=C.OuterParams(Ma=3.0, Br=25.0),
        inner=C.InnerParams(Ki=Ki, Di=Di, Ms=2.0),
        env=C.EnvParams(ke=5000.0, de=10.0, x_wall=wall, normal=normal, mu=mu),
    )
    return dataclasses.replace(base, dt=C.safe_dt(base) if dt is None else dt), Ko


def settle3(prm, Ko, f_d_vec, x_ref_vec, duration=25.0):
    sim = C.CascadeSim(prm, "policy")
    act = C.Action(np.asarray(x_ref_vec, float), np.asarray(f_d_vec, float), Ko)
    log = sim.run(duration, lambda t, last: act)
    k = max(1, int(1.0 / prm.dt))
    return {key: np.asarray(log[key])[-k:].mean(axis=0) for key in log}, sim


# --- axis-aligned 3-D must reproduce the 1-D answer exactly ---
n_ax = np.array([1.0, 0.0, 0.0])
prm, Ko = make3(1000.0, 2000.0, n_ax, 0.10)
ss, sim = settle3(prm, 1000.0 * np.eye(3), [10.0, 0, 0], [0.10, 0, 0])
want = A.contact_force_policy(10.0, 0.10, 1000.0, 2000.0, 5000.0, 0.10)
check("3-D along x == the 1-D result", ss["f_normal"][0], want, 2e-3)

# --- tilted plane, isotropic gains: normal direction still obeys the 1-D formula ---
for deg in (0, 20, 40, 60):
    th = np.deg2rad(deg)
    n_t = rot_y(th) @ np.array([0.0, 0.0, 1.0])
    wall = 0.10
    prm, _ = make3(1000.0, 2000.0, n_t, wall)
    f_d = 10.0 * n_t                      # push along the normal
    x_ref = wall * n_t                    # reference on the surface
    ss, _ = settle3(prm, 1000.0 * np.eye(3), f_d, x_ref)
    check(f"tilt {deg:2d} deg, isotropic K", ss["f_normal"][0], want, 3e-3)

# --- anisotropic stiffness ROTATED into the task frame (what GIC guarantees) ---
# Kp is soft along the task normal; the effective normal stiffness must be the
# soft value at EVERY tilt.  Rotating it is the lean-sim equivalent of GIC;
# leaving it in world coordinates is the naive Cartesian law.
KN, KT = 200.0, 2000.0
K_task = np.diag([KT, KT, KN])
print(f"     anisotropic Kp: {KN:.0f} N/m along the task normal, {KT:.0f} tangential")
for deg in (0, 30, 60):
    th = np.deg2rad(deg)
    R = rot_y(th)
    n_t = R @ np.array([0.0, 0.0, 1.0])
    wall = 0.10
    Ko_gic = R @ K_task @ R.T                      # task-frame stiffness (GIC)
    prm, _ = make3(Ko_gic, 2000.0, n_t, wall)
    ss, _ = settle3(prm, Ko_gic, 10.0 * n_t, wall * n_t)
    want_soft = A.contact_force_policy(10.0, 0.10, KN, 2000.0, 5000.0, 0.10)
    check(f"tilt {deg:2d} deg, K rotated into task frame", ss["f_normal"][0], want_soft, 5e-3)
    # the naive law keeps K in world coordinates: effective normal stiffness is
    # the quadratic form n^T K n, which grows with tilt
    k_naive = float(n_t @ K_task @ n_t)
    want_naive = KN * np.cos(th) ** 2 + KT * np.sin(th) ** 2
    check(f"tilt {deg:2d} deg, naive n^T K n", k_naive, want_naive, 1e-6,
          f"{k_naive/KN:.1f}x too stiff" if deg else "equal at zero tilt")

# --- Coulomb friction ---
MU = 0.3
th = np.deg2rad(25.0)
n_t = rot_y(th) @ np.array([0.0, 0.0, 1.0])
tang = rot_y(th) @ np.array([1.0, 0.0, 0.0])
prm, _ = make3(1000.0, 2000.0, n_t, 0.10, mu=MU)
sim = C.CascadeSim(prm, "policy")
act = C.Action(0.10 * n_t + 0.05 * tang, 10.0 * n_t, 1000.0 * np.eye(3))
log = sim.run(6.0, lambda t, last: act)
fe = np.asarray(log["f_e"]); fn = np.asarray(log["f_normal"])[:, 0]
f_t = fe - fn[:, None] * n_t
ratio = np.linalg.norm(f_t, axis=1) / np.maximum(fn, 1e-9)
moving = fn > 0.5
check_true("friction obeys |f_t| <= mu |f_n|",
           bool(np.all(ratio[moving] <= MU + 1e-9)),
           f"max ratio {ratio[moving].max():.4f} vs mu = {MU}")
check_true("friction is actually engaged (a wipe costs something)",
           bool(np.median(ratio[moving]) > 0.5 * MU),
           f"median ratio {np.median(ratio[moving]):.4f}")
prm0, _ = make3(1000.0, 2000.0, n_t, 0.10, mu=0.0)
sim0 = C.CascadeSim(prm0, "policy")
log0 = sim0.run(6.0, lambda t, last: act)
fe0 = np.asarray(log0["f_e"]); fn0 = np.asarray(log0["f_normal"])[:, 0]
f_t0 = np.linalg.norm(fe0 - fn0[:, None] * n_t, axis=1)
check_true("mu = 0 gives no tangential force at all", bool(f_t0.max() < 1e-9),
           f"max |f_t| = {f_t0.max():.2e} N")

# --- gain spellings must agree ---
prm, _ = make3(1000.0, 2000.0, n_ax, 0.10)
outs = []
for spelling in (1000.0, np.full(3, 1000.0), 1000.0 * np.eye(3)):
    ss, _ = settle3(prm, spelling, [10.0, 0, 0], [0.10, 0, 0])
    outs.append(float(ss["f_normal"][0]))
check_true("scalar, diagonal and matrix gains agree",
           bool(max(outs) - min(outs) < 1e-9), f"spread {max(outs)-min(outs):.2e} N")

# ========================================================================== #
print("\n[14] WIPE TASK calibration -- the anisotropy study needs all three factors live")
import wipe as W  # noqa: E402

task = W.WipeTask(tilt_x=np.deg2rad(20), tilt_y=np.deg2rad(-12))
m_, r_, F_ = W.decompose(W.compose(700.0, 10.0, task.R_belief))
check("compose/decompose round trip: magnitude", m_, 700.0, 1e-6)
check("compose/decompose round trip: ratio", r_, 10.0, 1e-6)
check_true("decompose puts the soft axis last",
           bool(abs(float(F_[:, 2] @ task.normal_belief)) > 1 - 1e-9),
           f"|dot with believed normal| = {abs(float(F_[:, 2] @ task.normal_belief)):.9f}")

ex = W.rollout(task, W.expert_stiffness(task))
check_true("expert succeeds with margin on BOTH metrics",
           ex.success and ex.force_rel < 0.8 * 0.30 and ex.path_error < 0.8 * 0.005,
           f"force {100*ex.force_rel:.1f}% of {30}%, path {1000*ex.path_error:.2f} of 5.00 mm")

stiff_n = W.rollout(task, W.compose(700.0, 0.3, task.R_belief))
soft_n = W.rollout(task, W.compose(700.0, 100.0, task.R_belief))
check_true("force error is sensitive to the NORMAL stiffness",
           stiff_n.force_rmse / soft_n.force_rmse > 3.0,
           f"{stiff_n.force_rmse:.2f} -> {soft_n.force_rmse:.2f} N "
           f"({stiff_n.force_rmse/soft_n.force_rmse:.1f}x)")
check_true("path error is sensitive to the TANGENTIAL stiffness",
           stiff_n.path_error / soft_n.path_error > 3.0,
           f"{1000*stiff_n.path_error:.2f} -> {1000*soft_n.path_error:.2f} mm "
           f"({stiff_n.path_error/soft_n.path_error:.1f}x)")

inverted = W.rollout(task, W.compose(700.0, 10.0, task.R_belief @ W.rot_xy(0.0, np.pi / 2)))
check_true("a 90 deg frame error fails the task", not inverted.success,
           f"force {100*inverted.force_rel:.0f}%, path {1000*inverted.path_error:.1f} mm")
frictionless = W.rollout(dataclasses.replace(task, mu=0.0),
                         W.compose(700.0, 10.0, task.R_belief))
check_true("friction is the largest contributor to path error",
           ex.path_error > 2.0 * frictionless.path_error,
           f"{1000*frictionless.path_error:.2f} mm at mu=0 -> {1000*ex.path_error:.2f} mm at "
           f"mu={task.mu} ({ex.path_error/frictionless.path_error:.1f}x); the remainder is "
           f"stroke dynamics, so the tangential axis still matters without friction")

# ========================================================================== #
print(f"\n{'='*94}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES: " + ", ".join(FAIL))
    raise SystemExit(1)
print("all checks passed\n")
