"""Study: what does each layer of the cascade actually do?

All experiments run in policy mode with a constant action, so every difference
comes from the controller and not from a changing command.

  IMPACT      step approach into a wall.  Which layer sets the peak force?
  SLOW DRIFT  the surface rises 20 mm over 6 s.  Which layer absorbs it, and
              what force error is left over?
  BANDWIDTH   the surface oscillates at a swept frequency; the force ripple,
              with and without the outer layer, locates the crossover.

Headline result, which contradicts the usual assumption: the inner impedance
does NOT own the impact.  Changing Ki by 16x moves the peak by 4%, while an
outer-layer velocity clamp cuts it 3.6x.  By the time contact happens the arm's
momentum is already committed, and that momentum was set by the outer layer
beforehand.  The inner layer owns the in-contact behaviour instead.

Usage:  python3 study_layers.py [--quick] [--out FIG.png]
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

import core as C
from labels import _tone_fit

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

KO, F_D, KI, DI, KE, WALL = 200.0, 6.0, 2000.0, 80.0, 5000.0, 0.10
MS, MA, BR = 2.0, 3.0, 25.0
RISE, T0, T1 = 0.020, 3.0, 9.0


def params(Ki=KI, Di=DI, Br=BR, v_limit=np.inf, rigid=False, dt=None) -> C.CascadeParams:
    p = C.CascadeParams(
        outer=C.OuterParams(Ma=MA, Br=Br, v_limit=v_limit, rigid=rigid),
        inner=C.InnerParams(Ki=Ki, Di=Di, Ms=MS),
        env=C.EnvParams(ke=KE, de=10.0, x_wall=WALL),
    )
    return dataclasses.replace(p, dt=C.safe_dt(p) if dt is None else dt)


def act_fn(x_ref, f_d, Ko):
    a = C.Action(np.array([x_ref]), np.array([f_d]), np.array([Ko]))
    return lambda t, last: a


def onset(log, thresh=0.5) -> int:
    """Index of first contact.  Displacements must be referenced to HERE, not to
    t=0, or the approach travel is mistaken for the outer layer retreating."""
    f = log["f_e"].ravel()
    i = int(np.argmax(f > thresh))
    return i if f[i] > thresh else 0


def bandwidths() -> dict[str, float]:
    """Each layer's natural frequency, Hz.

    The inner resonance RISES on contact: the mass then sits between the
    impedance spring and the environment spring and feels Ki + ke, not Ki.  It is
    the in-contact value that a contact-rich design has to respect.
    """
    return {
        "inner, free space": np.sqrt(KI / MS) / (2 * np.pi),
        "inner, in contact": np.sqrt((KI + KE) / MS) / (2 * np.pi),
        "outer Br/Ma": (BR / MA) / (2 * np.pi),
        "outer sqrt(Ko/Ma)": np.sqrt(KO / MA) / (2 * np.pi),
        "inner damping ratio": DI / (2 * np.sqrt(KI * MS)),
    }


def run_impact(duration=2.5, **kw):
    p = params(**kw)
    s = C.CascadeSim(p, "policy")
    s.x_r = np.array([WALL - 0.03]); s.x = np.array([WALL - 0.03]); s._x_r0 = s.x_r.copy()
    log = s.run(duration, act_fn(WALL + 0.02, F_D, KO))
    f = log["f_e"].ravel(); k = onset(log)
    return p, log, dict(peak=float(f.max()), v_contact=abs(float(log["v"].ravel()[k])),
                        steady=float(f[-200:].mean()))


def run_drift(Ko=KO, duration=12.0):
    p = params()
    wall = lambda t, x: RISE * float(np.clip((t - T0) / (T1 - T0), 0.0, 1.0))
    s = C.CascadeSim(p, "policy", wall_offset_fn=wall)
    s.x_r = np.array([WALL]); s.x = np.array([WALL]); s._x_r0 = s.x_r.copy()
    return p, s.run(duration, act_fn(WALL + 0.02, F_D, Ko))


def run_bandwidth(freqs, amp=0.002, cycles=14.0, settle_cycles=5.0):
    out = {"cascade": [], "rigid outer": []}
    for tag, rigid in [("cascade", False), ("rigid outer", True)]:
        for f in freqs:
            p = params(rigid=rigid, dt=min(C.safe_dt(params(rigid=rigid)), 1.0 / (60.0 * f)))
            s = C.CascadeSim(p, "policy",
                             wall_offset_fn=lambda t, x, f=f: amp * np.sin(2 * np.pi * f * t))
            s.x_r = np.array([WALL]); s.x = np.array([WALL]); s._x_r0 = s.x_r.copy()
            log = s.run(cycles / f, act_fn(WALL + 0.02, F_D, KO))
            k = int(settle_cycles / f / p.dt)
            amps, _, _ = _tone_fit({"f": log["f_e"][k:].ravel()}, log["t"][k:], f, f_cut_ratio=0.4)
            out[tag].append(abs(amps["f"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="layer_roles.png")
    args = ap.parse_args()

    bw = bandwidths()
    print("Layer natural frequencies")
    print("=" * 78)
    for k, v in bw.items():
        print(f"  {k:22s} {v:8.3f}" + ("" if "ratio" in k else " Hz"))
    print(f"  separation in contact: inner / outer = {bw['inner, in contact']/bw['outer Br/Ma']:.1f}x")

    # ---------------- impact ----------------
    INNER_VAR = [("Ki = 500", dict(Ki=500.0, Di=40.0)),
                 ("Ki = 2000 (base)", dict()),
                 ("Ki = 8000", dict(Ki=8000.0, Di=160.0))]
    # v_limit only changes the approach speed, so it sweeps one variable cleanly.
    # Br is kept separate: it also changes how fast the outer layer backs off
    # AFTER impact, so it does not belong on a speed-only curve.
    OUTER_VAR = [("v_limit 0.02", dict(v_limit=0.02)), ("v_limit 0.05", dict(v_limit=0.05)),
                 ("v_limit 0.12", dict(v_limit=0.12)), ("no limit (base)", dict())]
    OTHER_VAR = [("Br = 100", dict(Br=100.0))]
    inner_pts, outer_pts, other_pts = [], [], []
    p_i, log_i, m_i = run_impact()
    _, log_clamp, m_clamp = run_impact(v_limit=0.05)
    print("\nIMPACT: which layer sets the peak?")
    print("=" * 78)
    print(f"  {'variation':<24}{'peak (N)':>10}{'v at contact':>15}{'peak/steady':>13}")
    print("  " + "-" * 60)
    for tag, kw in INNER_VAR:
        _, _, m = run_impact(**kw)
        inner_pts.append((m["v_contact"], m["peak"]))
        print(f"  INNER  {tag:<17}{m['peak']:10.2f}{m['v_contact']:13.3f} m/s"
              f"{m['peak']/m['steady']:12.2f}x")
    for tag, kw in OUTER_VAR:
        _, _, m = run_impact(**kw)
        outer_pts.append((m["v_contact"], m["peak"]))
        print(f"  OUTER  {tag:<17}{m['peak']:10.2f}{m['v_contact']:13.3f} m/s"
              f"{m['peak']/m['steady']:12.2f}x")
    for tag, kw in OTHER_VAR:
        _, _, m = run_impact(**kw)
        other_pts.append((m["v_contact"], m["peak"]))
        print(f"  OUTER  {tag:<17}{m['peak']:10.2f}{m['v_contact']:13.3f} m/s"
              f"{m['peak']/m['steady']:12.2f}x   (also changes post-impact damping)")
    ip = np.array(inner_pts); op = np.array(outer_pts); qp = np.array(other_pts)
    print("  " + "-" * 60)
    print(f"  16x of INNER stiffness moves the peak {100*(ip[:,1].max()/ip[:,1].min()-1):.0f}%;")
    print(f"  the OUTER layer's approach speed moves it {op[:,1].max()/op[:,1].min():.1f}x.")
    print("  The peak tracks contact SPEED and is nearly blind to inner stiffness.")
    print("  The peak is a function of contact SPEED, which the outer layer fixes before")
    print("  the event. A 30 ms impact is over long before the outer loop's 120 ms time")
    print("  constant can act -- but it never needed to act during the event.")

    # ---------------- drift ----------------
    KOS = [50.0, 200.0, 800.0]
    drifts = {Ko: run_drift(Ko) for Ko in KOS}
    p_d, log_d = drifts[KO]
    print("\nSLOW DRIFT: surface rises 20.00 mm over 6 s")
    print("=" * 78)
    print(f"  {'Ko (N/m)':>9}{'outer absorbed':>17}{'inner change':>15}{'force change':>15}{'predicted':>12}")
    print("  " + "-" * 66)
    for Ko in KOS:
        pp, lg = drifts[Ko]
        fe, do, di = (lg[k].ravel() for k in ("f_e", "defl_outer", "defl_inner"))
        i0, i1 = int(T0 / pp.dt), int(T1 / pp.dt)
        pred = -Ko * RISE / (1 + Ko / KE + Ko / KI)
        print(f"  {Ko:9.0f}{1000*(do[i1]-do[i0]):14.2f} mm{1000*(di[i1]-di[i0]):12.2f} mm"
              f"{fe[i1]-fe[i0]:12.2f} N{pred:11.2f} N")
    print("  " + "-" * 66)
    print("  The outer layer absorbs the DISPLACEMENT; the residual force error is set by")
    print("  Ko alone, as Ko*drift/(1+Ko/ke+Ko/Ki). Soft outer coupling is what buys")
    print("  robustness to unknown surface height -- the 1-DOF analysis, on the cascade.")

    # ---------------- bandwidth ----------------
    freqs = np.geomspace(0.05, 20.0, 9 if args.quick else 15)
    print(f"\nBANDWIDTH: surface oscillating at 2 mm, {len(freqs)} frequencies")
    print("=" * 78)
    bwr = run_bandwidth(freqs)
    print(f"  {'freq (Hz)':>10}{'cascade (N)':>14}{'rigid outer (N)':>18}{'rejection':>12}")
    print("  " + "-" * 56)
    for f, a, b in zip(freqs, bwr["cascade"], bwr["rigid outer"]):
        print(f"  {f:10.3f}{a:14.3f}{b:18.3f}{b/max(a,1e-9):11.1f}x")
    rej = np.array(bwr["rigid outer"]) / np.maximum(np.array(bwr["cascade"]), 1e-9)
    cross = float(np.interp(2.0, rej[::-1], freqs[::-1])) if rej.min() < 2 < rej.max() else np.nan
    print("  " + "-" * 56)
    print(f"  outer layer stops earning its place (rejection < 2x) above ~{cross:.2f} Hz")

    plot(log_i, log_clamp, m_i, m_clamp, ip, op, qp, drifts, KOS, p_d,
         freqs, bwr, rej, bw, cross, args.out)


def plot(log_i, log_clamp, m_i, m_clamp, ip, op, qp, drifts, KOS, p_d,
         freqs, bwr, rej, bw, cross, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(2, 3, figsize=(15.8, 7.6))

    # ---- col 0 top: impact traces ----
    a = ax[0, 0]
    for log, m, c, lbl in [(log_i, m_i, SLOT[1], "outer speed unbounded"),
                           (log_clamp, m_clamp, SLOT[0], "outer $v_{limit}$ = 0.05 m/s")]:
        k = onset(log)
        a.plot(log["t"] - log["t"][k], log["f_e"].ravel(), color=c, lw=2.0,
               label=f"{lbl}   peak {m['peak']:.1f} N")
    a.axhline(F_D, color=INK3, ls="--", lw=1.2)
    a.text(-0.08, F_D * 1.35, f" commanded $f_d$ = {F_D:.0f} N", color=INK2, fontsize=7.5)
    a.set_xlim(-0.1, 0.8); a.set_ylabel("contact force (N)")
    a.set_title("Impact: one outer-layer clamp cuts the peak 3.6x", color=INK, loc="left")
    a.grid(True, alpha=0.9); a.set_axisbelow(True); a.legend(fontsize=7.5, loc="upper right")

    # ---- col 0 bottom: the peak collapses onto contact speed ----
    b = ax[1, 0]
    o = op[np.argsort(op[:, 0])]
    b.plot(o[:, 0], o[:, 1], "-o", color=SLOT[0], lw=2.0, ms=7, zorder=3,
           markeredgecolor=SURFACE, markeredgewidth=1.2,
           label=r"outer $v_{limit}$ swept (approach speed)")
    b.plot(qp[:, 0], qp[:, 1], "^", color=SLOT[2], ms=9, zorder=4,
           markeredgecolor=SURFACE, markeredgewidth=1.2,
           label=r"outer $B_r$ raised (speed + post-impact damping)")
    b.plot(ip[:, 0], ip[:, 1], "s", color=SLOT[1], ms=8, zorder=5,
           markeredgecolor=SURFACE, markeredgewidth=1.2, label="inner stiffness varied 16x")
    b.annotate(f"$K_i$ 500 to 8000 N/m\nall land here: {ip[:,1].min():.0f}-{ip[:,1].max():.0f} N",
               (ip[:, 0].mean(), ip[:, 1].mean()), textcoords="offset points",
               xytext=(-14, -42), color=SLOT[1], fontsize=7.5, ha="right")
    b.set_xlabel("end-effector speed at contact (m/s)")
    b.set_ylabel("peak contact force (N)")
    b.set_title("The peak tracks contact speed, not inner stiffness", color=INK, loc="left")
    b.grid(True, alpha=0.9); b.set_axisbelow(True); b.legend(fontsize=7.5, loc="upper left")

    # ---- col 1 top: drift displacements ----
    pp, lg = drifts[KO]
    k = onset(lg); t = lg["t"] - lg["t"][k]
    di = 1000 * (lg["defl_inner"].ravel() - lg["defl_inner"].ravel()[k])
    do = 1000 * (lg["defl_outer"].ravel() - lg["defl_outer"].ravel()[k])
    dw = 1000 * (lg["x_wall"].ravel() - lg["x_wall"].ravel()[k])
    a = ax[0, 1]
    a.plot(t, dw, color=INK3, lw=1.8, ls="--", label="surface rises 20 mm")
    a.plot(t, do, color=SLOT[1], lw=2.0, label="outer retreat  $x_r$")
    a.plot(t, di, color=SLOT[0], lw=2.0, label="inner deflection  $x_r - x$")
    a.set_ylabel("displacement from contact (mm)"); a.set_xlim(0, t[-1])
    a.set_title("Slow drift: the outer layer absorbs the motion", color=INK, loc="left")
    a.grid(True, alpha=0.9); a.set_axisbelow(True); a.legend(fontsize=7.5, loc="best")

    # ---- col 1 bottom: residual force error vs Ko ----
    b = ax[1, 1]
    for i, Ko in enumerate(KOS):
        _, l2 = drifts[Ko]
        k2 = onset(l2)
        b.plot(l2["t"] - l2["t"][k2], l2["f_e"].ravel(), color=SLOT[i], lw=2.0,
               label=rf"$K_o$ = {Ko:.0f} N/m")
    b.axhline(F_D, color=INK3, ls="--", lw=1.2)
    b.set_xlim(0, t[-1]); b.set_xlabel("time from contact (s)"); b.set_ylabel("contact force (N)")
    b.set_title(r"Residual force error is set by $K_o$ alone", color=INK, loc="left")
    b.grid(True, alpha=0.9); b.set_axisbelow(True); b.legend(fontsize=7.5, loc="best")

    # ---- col 2: bandwidth ----
    a = ax[0, 2]
    a.plot(freqs, bwr["rigid outer"], "-o", color=SLOT[1], lw=2.0, ms=5,
           markeredgecolor=SURFACE, markeredgewidth=1.0, label="outer layer removed")
    a.plot(freqs, bwr["cascade"], "-o", color=SLOT[0], lw=2.0, ms=5,
           markeredgecolor=SURFACE, markeredgewidth=1.0, label="full cascade")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylabel("force ripple amplitude (N)")
    a.set_title("Rejecting a moving surface", color=INK, loc="left")
    ymax = a.get_ylim()[1]
    for key, c, lbl in [("outer Br/Ma", SLOT[1], "outer $B_r/M_a$"),
                        ("inner, in contact", SLOT[0], r"inner $\sqrt{(K_i+k_e)/M_s}$")]:
        a.axvline(bw[key], color=c, ls=":", lw=1.4)
        a.text(bw[key], ymax, f" {lbl}\n {bw[key]:.2f} Hz", color=c, fontsize=6.8, va="top")
    a.grid(True, which="major", alpha=0.9); a.set_axisbelow(True)
    a.legend(fontsize=7.5, loc="lower right")

    b = ax[1, 2]
    b.plot(freqs, rej, "-o", color=SLOT[3], lw=2.0, ms=5,
           markeredgecolor=SURFACE, markeredgewidth=1.0)
    b.axhline(1.0, color=INK3, lw=0.8)
    b.axhline(2.0, color=INK3, ls="--", lw=1.2)
    b.text(freqs[0], 2.2, " 2x", color=INK2, fontsize=7.5)
    if np.isfinite(cross):
        b.axvline(cross, color=SLOT[3], ls=":", lw=1.4)
        b.text(cross, b.get_ylim()[1], f" crossover {cross:.2f} Hz", color=SLOT[3],
               fontsize=7.5, va="top")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("surface oscillation frequency (Hz)")
    b.set_ylabel("ripple without outer / with outer")
    b.set_title("Where the outer layer stops helping", color=INK, loc="left")
    b.grid(True, which="major", alpha=0.9); b.set_axisbelow(True)

    fig.suptitle("Division of labour: the outer admittance owns approach speed and slow disturbances; the inner impedance owns in-contact behaviour",
                 color=INK, fontsize=11, x=0.006, ha="left", y=0.988)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=170)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
