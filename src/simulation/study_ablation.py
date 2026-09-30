"""What each layer of the cascade buys, over many episodes.

Three controllers, identical in every other respect:

  cascade      outer admittance (force feedback) + compliant inner impedance
  outer only   outer admittance + a STIFF inner, so the arm tracks its reference
               rigidly and all the compliance sits in the outer loop
  inner only   compliant inner impedance driven straight by the pose command,
               with no force feedback anywhere

on two tasks that stress different layers: a spiral wipe over an uneven tilted
plate (a fast disturbance the outer layer cannot reach) and a peg insertion (a
slow push that needs force feedback to happen at all).  Each condition is run
over many randomised episodes -- a different surface, a different peg and hole --
and reported as a mean with its spread, because a single episode of either task
says as much about that episode's geometry as about the controller.

Usage:  python3 study_ablation.py [--episodes 50] [--workers 12]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

CFG = {"cascade": dict(inner_stiffness=3000.0, outer="admittance"),
       "outer only": dict(inner_stiffness=20000.0, outer="admittance"),
       "inner only": dict(inner_stiffness=3000.0, outer="rigid")}
NAMES = list(CFG)

WIPE = dict(pad_radius=0.015, pad_ahead=0.012, magnitude=1200.0, ratio=10.0,
            rot_stiffness=200.0, wrist_inertia=0.004, f_target=8.0, f_limit=30.0,
            v_limit=0.20, stroke_len=0.05, stroke_hz=0.25, approach_s=4.0, settle=6.0,
            duration=26.0, force_hz=20.0, path="spiral", r0=0.0, r_max=0.10,
            turns=3.5, path_s=18.0, n_bumps=8)

PEG = dict(check=False, frames=False, episodes=1, magnitude=1200.0, ratio=10.0,
           rot_stiffness=400.0, wrist_inertia=0.005, grip_back=0.040, f_push=12.0,
           f_limit=30.0, v_limit=0.06, standoff_margin=0.035, overdrive=0.010,
           align_s=4.0, insert_s=4.0, duration=10.0, retarget=True)

TRACE_N = 400        # points kept per force trace, so 300 episodes stay small


def wipe_job(spec):
    """One wipe episode.  Imports inside the worker so each process builds its
    own SAPIEN scene; sharing one across processes does not work."""
    import warnings as _w
    _w.filterwarnings("ignore")
    import numpy as _np
    import argparse as _ap
    import fr3_wipe_scene as W

    name, he, seed = spec
    rng = _np.random.default_rng(1000 + seed)
    tx, ty = rng.uniform(12.0, 24.0), rng.uniform(-16.0, -4.0)
    env, R = W.build_scene(_np.deg2rad(tx), _np.deg2rad(ty), WIPE["pad_radius"],
                           n_bumps=WIPE["n_bumps"], bump_radius=0.020,
                           bump_proud=0.0012, bump_seed=seed)
    u = env.unwrapped
    pad = W.weld_pad(u, WIPE["pad_radius"], WIPE["pad_ahead"])
    a = _ap.Namespace(**{**WIPE, **CFG[name], "tilt_x": tx, "tilt_y": ty,
                         "height_error": he})
    try:
        r = W.rollout(u, pad, R, W.compose(WIPE["magnitude"], WIPE["ratio"], R), a)
    finally:
        env.close()
    sel = (r["t"] >= WIPE["settle"]) & (r["t"] <= WIPE["settle"] + WIPE["path_s"])
    f = r["f_filt"][sel]
    k = max(1, len(f) // TRACE_N)
    return dict(name=name, he=he, seed=seed, contact=r["contact"],
                mean=float(f.mean()), rmse=r["force_rmse"], peak=r["peak"],
                path=r["path_error"], trace=f[:TRACE_N * k].reshape(-1, k).mean(axis=1))


def peg_job(spec):
    import warnings as _w
    _w.filterwarnings("ignore")
    import numpy as _np
    import argparse as _ap
    import peg_insertion_cascade as P

    name, lat, seed = spec
    s = P.setup(seed, PEG["grip_back"])
    Rh, _ = P.hole_frame(s["u"])
    a = _ap.Namespace(**{**PEG, **CFG[name]}, seed=seed, lateral_mm=lat)
    try:
        r = P.rollout(s, P.compose_axial(PEG["magnitude"], PEG["ratio"], Rh),
                      lat / 1000.0 * Rh[:, 1], a)
    finally:
        s["env"].close()
    f = r["f"][r["t"] >= a.align_s]
    return dict(name=name, lat=lat, seed=seed, success=float(r["success"]),
                mean=float(f.mean()), peak=float(f.max()),
                depth=1000.0 * float(r["depth"][-1]))


def run_all(jobs, fn, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        out = []
        for i, res in enumerate(pool.imap_unordered(fn, jobs), 1):
            out.append(res)
            if i % max(1, len(jobs) // 20) == 0:
                print(f"    {i}/{len(jobs)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="ablation.png")
    args = ap.parse_args()
    n = args.episodes
    h_errs, lats = [0.0, 0.003], [0.0, 3.0]

    wjobs = [(nm, he, sd) for nm in NAMES for he in h_errs for sd in range(n)]
    pjobs = [(nm, lat, sd) for nm in NAMES for lat in lats for sd in range(n)]
    print(f"wipe: {len(wjobs)} episodes on {args.workers} workers")
    wipe = run_all(wjobs, wipe_job, args.workers)
    print(f"peg:  {len(pjobs)} episodes")
    peg = run_all(pjobs, peg_job, args.workers)

    def W_(nm, he, f):
        return np.array([r[f] for r in wipe if r["name"] == nm and r["he"] == he])

    def P_(nm, lat, f):
        return np.array([r[f] for r in peg if r["name"] == nm and r["lat"] == lat])

    print("\n" + "=" * 96)
    print(f"WIPE -- spiral over an uneven tilted plate, {n} episodes each, commanded 8 N")
    print("=" * 96)
    print(f"  {'controller':<13}{'height err':>11}{'contact':>11}{'mean force':>16}"
          f"{'force RMSE':>15}{'peak':>14}")
    print("  " + "-" * 82)
    for nm in NAMES:
        for he in h_errs:
            c, m, e, pk = (W_(nm, he, k) for k in ("contact", "mean", "rmse", "peak"))
            print(f"  {nm:<13}{1000*he:8.0f} mm{100*c.mean():9.0f}%"
                  f"{m.mean():9.2f}+-{m.std():4.2f} N{e.mean():8.2f}+-{e.std():4.2f} N"
                  f"{pk.mean():8.0f}+-{pk.std():4.0f} N")
    print("\n" + "=" * 96)
    print(f"PEG INSERTION -- {n} episodes each")
    print("=" * 96)
    print(f"  {'controller':<13}{'lateral err':>12}{'success':>12}{'mean force':>18}{'peak':>14}")
    print("  " + "-" * 74)
    for nm in NAMES:
        for lat in lats:
            sc, m, pk = (P_(nm, lat, k) for k in ("success", "mean", "peak"))
            sem = 100 * sc.std() / np.sqrt(len(sc))
            print(f"  {nm:<13}{lat:9.0f} mm{100*sc.mean():8.0f}+-{sem:3.0f}%"
                  f"{m.mean():11.2f}+-{m.std():5.2f} N{pk.mean():8.0f}+-{pk.std():4.0f} N")
    print("  " + "-" * 74)
    plot(wipe, peg, W_, P_, h_errs, lats, n, args.out)


def plot(wipe, peg, W_, P_, h_errs, lats, n, out_path):
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
    fig, ax = plt.subplots(1, 3, figsize=(15.8, 5.0))

    # --- A: mean force trace with spread across episodes ---
    a = ax[0]
    for i, nm in reversed(list(enumerate(NAMES))):
        tr = np.vstack([r["trace"] for r in wipe if r["name"] == nm and r["he"] == 0.0])
        t = np.linspace(0, WIPE["path_s"], tr.shape[1])
        mu, sd = tr.mean(axis=0), tr.std(axis=0)
        a.fill_between(t, mu - sd, mu + sd, color=SLOT[i], alpha=0.22, lw=0, zorder=2 + i)
        a.plot(t, mu, color=SLOT[i], lw=2.0, zorder=5 + i,
               label=f"{nm}   mean {mu.mean():.1f} N")
    a.axhline(WIPE["f_target"], color=INK, ls="--", lw=1.4, zorder=9)
    a.annotate(f" commanded {WIPE['f_target']:.0f} N", xy=(0.02, 0.06),
               xycoords="axes fraction", color=INK, fontsize=8)
    a.set_xlabel("time into the spiral (s)")
    a.set_ylabel("force along the surface normal (N)")
    a.set_title(f"Wipe: mean over {n} surfaces, band = 1 s.d.", color=INK, loc="left")
    a.set_ylim(0, 45)
    a.grid(True, alpha=0.9); a.set_axisbelow(True)
    h, l = a.get_legend_handles_labels()
    a.legend(h[::-1], l[::-1], fontsize=7.5, loc="upper right")

    # --- B: delivered force vs how wrong the commanded height is ---
    b = ax[1]
    x = np.arange(len(h_errs)); w = 0.26
    for i, nm in enumerate(NAMES):
        mu = [W_(nm, he, "mean").mean() for he in h_errs]
        sd = [W_(nm, he, "mean").std() for he in h_errs]
        b.bar(x + (i - 1) * w, mu, yerr=sd, width=w, color=SLOT[i], label=nm, zorder=3,
              error_kw=dict(ecolor=INK2, lw=1.1, capsize=3))
        for xi, v in zip(x + (i - 1) * w, mu):
            b.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
    b.axhline(WIPE["f_target"], color=INK, ls="--", lw=1.3)
    b.annotate(f" commanded {WIPE['f_target']:.0f} N", xy=(0.02, 0.05),
               xycoords="axes fraction", color=INK, fontsize=8)
    b.set_xticks(x); b.set_xticklabels([f"{1000*h:.0f} mm" for h in h_errs])
    b.set_xlabel("how wrong the commanded surface height is")
    b.set_ylabel("mean force delivered (N)")
    b.set_title("Only force feedback ignores where the surface is", color=INK, loc="left")
    b.grid(True, axis="y", alpha=0.9); b.set_axisbelow(True)
    b.legend(fontsize=7.5, loc="upper left")

    # --- C: insertion ---
    c = ax[2]
    for i, nm in enumerate(NAMES):
        sc = [100 * P_(nm, lat, "success").mean() for lat in lats]
        se = [100 * P_(nm, lat, "success").std() / np.sqrt(n) for lat in lats]
        c.bar(x + (i - 1) * w, sc, yerr=se, width=w, color=SLOT[i], label=nm, zorder=3,
              error_kw=dict(ecolor=INK2, lw=1.1, capsize=3))
        for xi, lat in zip(x + (i - 1) * w, lats):
            c.text(xi, 3, f"{P_(nm, lat, 'mean').mean():.0f} N", ha="center",
                   fontsize=7.5, color=SURFACE, rotation=90, va="bottom")
    c.set_xticks(x); c.set_xticklabels([f"{l:.0f} mm" for l in lats])
    c.set_xlabel("lateral aim error")
    c.set_ylabel("insertion success rate (%)")
    c.set_title("Insertion: bar = success, label = mean contact force", color=INK, loc="left")
    c.set_ylim(0, 108)
    c.grid(True, axis="y", alpha=0.9); c.set_axisbelow(True)
    c.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(f"What each layer buys, over {n} randomised episodes per condition: remove the inner compliance and the tool cannot ride a surface; remove the force feedback and it cannot find one.",
                 color=INK, fontsize=10.5, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out_path, dpi=170)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
