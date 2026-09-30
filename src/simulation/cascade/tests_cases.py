"""Validation suite for the two single-interface execution cases.

Run:  python3 tests_cases.py     (plain asserts; pytest not required)

Same contract as tests.py: every check prints both numbers, so a failure states
the actual discrepancy.  Where a quantity is only first-order accurate the test
checks the CONVERGENCE RATE under step refinement rather than a fixed
tolerance -- a fixed tolerance on an O(dt) quantity passes for the wrong reason
as soon as someone lowers dt.
"""
from __future__ import annotations

import dataclasses

import numpy as np

import analytic as A
import case1 as C1
import case2 as C2

PASS: list[str] = []
FAIL: list[str] = []


def check(name, got, want, tol, note=""):
    got_f = float(np.asarray(got).ravel()[0]); want_f = float(np.asarray(want).ravel()[0])
    err = abs(got_f - want_f); rel = err / max(abs(want_f), 1e-9)
    ok = bool(err <= tol or rel <= tol)
    line = f"  {'PASS' if ok else 'FAIL'}  {name:54s} got={got_f: .6f} want={want_f: .6f} err={err:.2e}"
    if note:
        line += f"  [{note}]"
    print(line)
    (PASS if ok else FAIL).append(name)


def check_true(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:54s} {note}")
    (PASS if bool(cond) else FAIL).append(name)


def c1_run(duration=6.0, reach=0.035, **kw):
    p = dataclasses.replace(C1.Case1Params(), **kw)
    sim = C1.Case1Sim(p)
    log = sim.run(duration, lambda t: C1.Case1Proposal(
        np.zeros(p.n), np.array([C1.quintic_vel(t, 0.5, 2.0, reach)])))
    return sim, log, p


# ========================================================================== #
print("\n--- Case 1: direct GIC, steady state --------------------------------")
sim, log, p = c1_run()
fe = log["f_e"][-2000:, 0].mean()
check("Case 1 steady force matches closed form",
      fe, A.contact_force_case1(0.0, 0.035, p.Kp, p.env.ke, p.env.x_wall), 1e-3,
      "one series compliance: 1 + Kp/ke")
check("Case 1 applied reference reaches the requested 35 mm",
      log["x_d"][-1, 0], 0.035, 1e-6)

# The gauge trap: with a small tank the gate throttles V_d, so the run tracks a
# DIFFERENT reference than the one requested.  The closed form still holds --
# but only against the APPLIED x_d.  This is the requested/applied distinction
# the recording spec insists on, made numerical.
sim_g, log_g, p_g = c1_run(E0=1.0)
xd_applied = log_g["x_d"][-1, 0]
check_true("gated run tracks LESS than requested", xd_applied < 0.0349,
           f"applied x_d = {1000*xd_applied:.2f} mm vs 35.00 requested, alpha_min={log_g['alpha'].min():.4f}")
check("Case 1 closed form holds against the APPLIED reference",
      log_g["f_e"][-2000:, 0].mean(),
      A.contact_force_case1(0.0, xd_applied, p_g.Kp, p_g.env.ke, p_g.env.x_wall), 1e-3,
      "labelling the REQUESTED reference would be wrong by 3.6%")

print("\n--- Case 1: Theorem 1 energy identity -------------------------------")
res = {}
for dt in (2e-3, 1e-3, 5e-4, 2.5e-4):
    s, lg, _ = c1_run(dt=dt)
    res[dt] = (np.abs(lg["resid"]).max(), abs(s.resid_int))
dts = sorted(res, reverse=True)
for a, b in zip(dts[:-1], dts[1:]):
    check(f"residual is first order in dt ({a:.1e} -> {b:.1e})",
          res[a][0] / res[b][0], 2.0, 0.05)
check_true("cumulative residual is small in absolute terms",
           res[2.5e-4][1] < 1e-4, f"{1e3*res[2.5e-4][1]:.4f} mJ over 6 s")

print("\n--- Case 1: the tank actually enforces something --------------------")
# An aggressive ACTIVE TORQUE proposal, which is pure energy injection.
def aggressive(t):
    return C1.Case1Proposal(np.array([200.0]), np.array([C1.quintic_vel(t, 0.5, 2.0, 0.035)]))

on = C1.Case1Sim(dataclasses.replace(C1.Case1Params(), tank=True, E0=1.0, Ec=1.0))
off = C1.Case1Sim(dataclasses.replace(C1.Case1Params(), tank=False))
lon, loff = on.run(6.0, aggressive), off.run(6.0, aggressive)
W_on = float(np.sum(lon["p_T"]) * on.dt)
W_off = float(np.sum(loff["p_T"]) * off.dt)
check_true("tank keeps E >= 0 at all times", bool(lon["E"].min() >= 0.0),
           f"min E = {lon['E'].min():.3e} J")
check_true("gated active work is bounded by the tank", W_on <= 1.0 + 1e-6,
           f"{W_on:.4f} J admitted of E0 = 1.0 J")
check_true("ungated active work exceeds the tank budget", W_off > 1.0,
           f"{W_off:.4f} J injected with the gate off ({W_off/max(W_on,1e-9):.1f}x the gated run)")
check_true("gate closes under the aggressive proposal", lon["alpha"].min() < 0.01,
           f"alpha fell to {lon['alpha'].min():.4f}")

# The passivity statement, stated correctly.  Theorem 1 gives
#     Hdot_T + Edot <= -f_e.V_s - D_T,
# NOT monotonicity of H_T + E.  The distinction is not pedantic: -f_e.V_s is
# POSITIVE whenever the tool retreats, because the compressed wall returns the
# energy it stored, so H_T + E genuinely rises during unloading and a
# monotonicity test fails on correct code.  What must hold is the DISSIPATION
# inequality -- storage minus supply <= 0 -- and it does, to O(dt).
exc = {}
for dt in (1e-3, 5e-4, 2.5e-4, 1.25e-4):
    s_ = C1.Case1Sim(dataclasses.replace(C1.Case1Params(), dt=dt, E0=1.0))
    lg_ = s_.run(6.0, aggressive)
    supply = float(np.sum(-(lg_["f_e"][:, 0] * lg_["v"][:, 0]) - lg_["D_T"]) * dt)
    exc[dt] = float(lg_["HE"][-1] - lg_["HE"][0]) - supply
de = sorted(exc, reverse=True)
for a, b in zip(de[:-1], de[1:]):
    check(f"dissipation inequality closes first order ({a:.1e} -> {b:.1e})",
          exc[a] / exc[b], 2.0, 0.05)
check_true("excess storage vanishes under refinement", exc[1.25e-4] < 1e-2,
           f"{1e3*exc[1.25e-4]:.3f} mJ at dt=1.25e-4 over 6 s, falling as O(dt)")

print("\n--- Case 1: the held-torque defect is real --------------------------")
held = C1.Case1Sim(dataclasses.replace(C1.Case1Params(), hold_steps=4))
lh = held.run(6.0, lambda t: C1.Case1Proposal(np.zeros(1), np.array([C1.quintic_vel(t, 0.5, 2.0, 0.035)])))
check_true("zero-order hold produces a nonzero delta_tau",
           float(np.abs(lh["d_tau"]).max()) > 1e-6,
           f"max |delta_tau| = {np.abs(lh['d_tau']).max():.3e} N")
check_true("delta_tau is accounted, so the identity still closes",
           float(np.abs(lh["resid"]).max()) < 5e-2,
           f"max |resid| = {np.abs(lh['resid']).max():.3e} W")

# ========================================================================== #
print("\n--- Case 2: admittance + motion servo, steady state -----------------")
s2 = C2.Case2Sim(C2.Case2Params()); l2 = s2.run(6.0, C2.wall_command)
c_inv = float(l2["c_inv"][-1, 0])
check("Case 2 steady force matches closed form (with the servo invariant c)",
      l2["f_e"][-1500:, 0].mean(),
      A.contact_force_case2(0.035, 300.0, 1000.0, 0.012, c=c_inv), 1e-3,
      "one series compliance: 1 + Ka/ke")
check_true("clipping moved the equilibrium leaf (c != 0)",
           abs(c_inv) > 1e-5 and int(l2["clipped"].sum()) > 0,
           f"c = {1e3*c_inv:.4f} mm after {int(l2['clipped'].sum())} clipped steps; "
           f"the c=0 prediction would be {A.contact_force_case2(0.035,300.,1000.,0.012):.4f} N")

print("\n--- Case 2: Prop. 1 port-defect identity ----------------------------")
res2 = {}
for dt in (2e-3, 1e-3, 5e-4, 2.5e-4):
    # Ba != 0 and a moving command so the Ba(v_c - v_r) term is exercised.
    pp = dataclasses.replace(C2.Case2Params(), dt=dt, Ba=10.0, Br=20.0)
    ss = C2.Case2Sim(pp)
    lg = ss.run(6.0, C2.wall_command)
    res2[dt] = np.abs(lg["resid"]).max()
d2 = sorted(res2, reverse=True)
for a, b in zip(d2[:-1], d2[1:]):
    check(f"Prop. 1 residual is first order in dt ({a:.1e} -> {b:.1e})",
          res2[a] / res2[b], 2.0, 0.12)

# The one configuration where the audit becomes a guarantee.
ideal = dataclasses.replace(C2.Case2Params(), Ts=0.0, f_filter_tau=0.0, f_bias=0.0,
                            v_limit=np.inf, a_limit=np.inf, delay_ticks=0)
li = C2.Case2Sim(ideal).run(6.0, C2.wall_command)
check_true("r_M -> 0 under exact tracking and sensing",
           float(np.abs(li["r_M"]).max()) < 1e-12,
           f"max |r_M| = {np.abs(li['r_M']).max():.3e} W; virtual storage then certifies the real port")
check_true("r_M is NOT zero for the realistic servo",
           float(np.abs(l2["r_M"]).max()) > 1e-3,
           f"max |r_M| = {np.abs(l2['r_M']).max():.3e} W -- retrospective, nothing reacts to it")

print("\n--- Case 2: Prop. 2 Routh condition vs actual poles -----------------")
rng = np.random.default_rng(0)
agree = margin_skip = 0
for _ in range(250):
    ma = 10 ** rng.uniform(-1, 1); da = 10 ** rng.uniform(0, 2.5)
    ka = 10 ** rng.uniform(1, 3.5); ke = 10 ** rng.uniform(2, 4.5)
    Ts = 10 ** rng.uniform(-3, -0.7)
    ok, margin = A.routh_case2(ma, da, ka, ke, Ts)
    if abs(margin) < 1e-6 * max(1.0, abs(margin)):
        margin_skip += 1
        continue
    poles = C2.reduced_poles(ma, da, ka, ke, Ts)
    agree += int(ok == bool(np.max(poles.real) < 0.0))
check("Routh criterion agrees with the reduced cubic's poles",
      agree, 250 - margin_skip, 0.0, f"{agree}/{250-margin_skip} randomised parameter sets")

print("\n--- Case 2: manuscript Table I --------------------------------------")
ROWS = [(0.020, 1000, 30, 0, 5.47, 0.00, 0.65), (0.080, 1000, 30, 0, 5.60, 0.00, 1.31),
        (0.020, 10000, 30, 0, 31.40, 21.29, 79.69), (0.020, 1000, 30, 10, 5.64, 0.44, 1.43),
        (0.020, 10000, 120, 0, 25.06, 21.43, 44.38)]
for Ts, ke, da, d, f_want, sd_want, rms_want in ROWS:
    pp = dataclasses.replace(C2.Case2Params(), Ts=Ts, Br=da, delay_ticks=d,
                             env=dataclasses.replace(C2.Case2Params().env, ke=ke))
    lg = C2.Case2Sim(pp).run(6.0, C2.wall_command)
    m = lg["t"] >= 3.0                      # force stats: 3-6 s, per the caption
    fe = lg["f_e"][m, 0]
    # RMS: the FULL 0-6 s run.  The caption states the window for the force
    # column only; 3-6 s does not reproduce the RMS column and 0-6 s does, on
    # all five rows.  Worth stating in the caption.
    err = (lg["x_s"] - lg["x_r"])[:, 0]
    rms = 1000.0 * np.sqrt((err ** 2).mean())
    tag = f"Ts={1000*Ts:.0f}ms ke={ke} da={da} d={d}"
    check(f"Table I force   [{tag}]", fe.mean(), f_want, 0.02)
    check(f"Table I std     [{tag}]", fe.std(), sd_want, 0.02)
    check(f"Table I RMS     [{tag}]", rms, rms_want, 0.02)

# ========================================================================== #
print("\n--- Cross-case: the series-compliance count -------------------------")
# The single number this whole comparison exists to produce.  Same wall, same
# commanded offset, same stiffness in the role of "the gain that faces the
# environment" -- the only difference is how many springs sit in series.
ke, xw, reach, K = 1000.0, 0.012, 0.035, 300.0
f1 = A.contact_force_case1(0.0, reach, K, ke, xw)
f2 = A.contact_force_case2(reach, K, ke, xw)
fc = A.contact_force_policy(0.0, reach, K, 2000.0, ke, xw)      # cascade, Ki = 2000
fc_soft = A.contact_force_policy(0.0, reach, K, 300.0, ke, xw)  # cascade, Ki = Ko
check("Case 1 and Case 2 deliver the same force (one compliance each)", f1, f2, 1e-9)
check_true("the cascade delivers LESS for the same command", fc < f1,
           f"Case1/2 {f1:.3f} N -> cascade {fc:.3f} N at Ki=2000 "
           f"({100*(1-fc/f1):.1f}% lost to the inner series compliance)")
check_true("a soft inner loop costs much more", fc_soft < 0.75 * fc,
           f"Ki=2000 -> {fc:.3f} N, Ki=Ko=300 -> {fc_soft:.3f} N "
           f"({100*(1-fc_soft/f1):.0f}% lost); this is the cost neither single-interface case pays")

# ========================================================================== #
print(f"\n{'='*100}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES: " + ", ".join(FAIL))
    raise SystemExit(1)
print("all checks passed\n")
