"""The gate: on a corrugated board, does knowing the surface beat every constant?

This is the experiment the whole direction rests on, with the five defects of the
earlier ones fixed.  It reports three numbers on held-out seeds:

    best CONSTANT, re-tuned on THIS board          <- what we would ship
    best constant PER EPISODE                      <- ceiling for per-episode (RMA-style) adaptation
    within-episode ORACLE, handed the true surface  <- ceiling for any schedule

and the rule is decided before it runs: the direction is alive only if the
oracle beats the best constant by >= 5 points of coverage AND by more than twice
the seed standard deviation.  Anything less and no estimator built on top can be
expected to matter, because an estimator cannot beat the oracle it approximates.

What is different from the dome and patchy runs, and why:

  THE BOARD.  Flat facets of alternating tilt, the only family that satisfies
    residual <= pad_give < 2 r tan(tilt) -- see CorrugatedSurface and
    gate_feasible.py.  Bumps and domes fail it on both sides, which is why a
    constant won all 40 dome cells.
  THE ORACLE'S SIGNAL.  K_R = f_n r / theta with theta the tilt of the best-fit
    plane under the pad, not the normal SWING across it.  Swing is the part of
    the surface a rotation cannot fix at any stiffness; steering stiffness by it
    was asking the controller to chase an unreachable target.  Note this rule
    also goes STIFF on the plateaus for free (theta -> 0 clips to kr_hi), which
    is the half the swing rule could never express.
  THE SPEED.  Slow enough that the wrist can acquire a flank's tilt while on it
    (t90 < flank/v) and the plateau is short enough that it cannot recover
    (plateau/v < 2 t90).  At the shipped 0.05 m/s neither holds and the best any
    controller can do IS a constant.
  THE METRIC.  Coverage of the marks, with erase_work set so nothing saturates;
    force-band compliance is a FILTER, not a tiebreak, so the oracle cannot win
    by pressing harder.
  THE SEEDS.  The glyph is moved per seed, fit and held-out seeds are separate,
    and the spread is reported.

    WIPE_SURFACE=corr WIPE_CORR_FLANK=0.10 python3 corr_gate.py --seed 0
    python3 corr_gate.py --report logs/corr_gate_*.log
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

KR_LO, KR_HI = 0.3, 60.0


def demand_tilt(frame, uv, r_pad: float, n: int = 7) -> float:
    """Tilt of the best-fit plane under the pad, in radians: the rotation the
    wrist would have to acquire.  Fitted on the TRUE surface -- this is the
    oracle's privilege, and the only one it gets."""
    g = np.linspace(-r_pad, r_pad, n)
    dx, dy = np.meshgrid(g, g, indexing="ij")
    keep = (dx ** 2 + dy ** 2) <= r_pad ** 2 + 1e-12
    dx, dy = dx[keep], dy[keep]
    h = frame.height(np.column_stack([uv[0] + dx, uv[1] + dy]))
    A = np.column_stack([dx, dy, np.ones_like(dx)])
    c = np.linalg.lstsq(A, np.asarray(h, float).ravel(), rcond=None)[0]
    return float(np.arctan(np.hypot(c[0], c[1])))


def oracle(r_pad: float, f_fallback: float):
    """K_R = f_n r / theta, clipped.  Soft on a flank, stiff on a plateau."""
    def f(s):
        last = getattr(s, "last", {}) or {}
        f_n = float(last.get("f_n", 0.0))
        uvh = last.get("contact_uvh")
        if uvh is None or f_n < 0.8:
            return KR_HI                      # in the air there is no moment to yield to
        th = demand_tilt(s.frame, np.asarray(uvh[:2], float), r_pad)
        return float(np.clip(max(f_n, f_fallback) * r_pad / max(th, 1e-4), KR_LO, KR_HI))
    return f


def board_from_env():
    """The same corrugation the scene built, for splitting a trace by band.
    `attitude_task.py` does this too -- the surface is cheap and closed-form, so
    re-deriving it beats threading a handle out of the sim."""
    from patchy_surface import CorrugatedSurface
    return CorrugatedSurface(
        half=0.16, seed=3,
        plateau=float(os.environ.get("WIPE_CORR_PLATEAU", 0.030)),
        flank=float(os.environ.get("WIPE_CORR_FLANK", 0.100)),
        slope_deg=float(os.environ.get("WIPE_CORR_SLOPE", 12.0)))


def episode(a, seed: int, kr: float, kr_fn=None) -> dict:
    import smoke
    import wipe_scene as SC
    rng = np.random.default_rng(90_000 + seed)
    th = rng.uniform(0, 2 * np.pi)
    rad = 1e-3 * a.jitter_mm * np.sqrt(rng.uniform())
    r = smoke.run(text=a.text, curved=True, kr=kr, letter_height=a.letter,
                  seed=seed, verbose=False, speed=a.speed, f_target=a.f,
                  erase_work=a.erase_work, kr_fn=kr_fn, time_limit=a.time_limit,
                  pad_give=a.pad_give,
                  offset_uv=(rad * np.cos(th), rad * np.sin(th)))
    t = r["trace"]
    down = t["f"] > 0.8
    # Per band, because the whole design is that the two bands want opposite
    # things: the flank wants the pad to acquire its tilt (low `mis`), the
    # plateau wants it to have let go of it again (low `yld`).  An average over
    # both hides exactly the trade-off under test.
    surf = board_from_env()
    ok = down & np.isfinite(t["u"])
    b = surf.band_at(t["u"][ok]) if ok.any() else np.zeros(0)
    fl, pl = b > 0.8, b < 0.2

    def m(arr, sel):
        v = arr[ok][sel]
        return float(v.mean()) if v.size else None

    return dict(pad=SC.ERASER_R, seed=seed, kr=(None if kr_fn else kr),
                oracle=bool(kr_fn), coverage=float(r["erased"]),
                in_band=float(r["in_band"]), peak=float(r["peak_force"]),
                mis=float(r["mis_mean"]), yld=float(r["yld_mean"]),
                mis_flank=m(t["mis"], fl), mis_plat=m(t["mis"], pl),
                yld_flank=m(t["yld"], fl), yld_plat=m(t["yld"], pl),
                frac_flank=(float(fl.mean()) if b.size else None),
                f_rmse=float(np.sqrt(np.mean((t["f"][down] - a.f) ** 2))) if down.any() else None,
                speed=a.speed, f=a.f, erase_work=a.erase_work, pad_give=a.pad_give,
                blind_deg=float(np.degrees(np.arctan(
                    (a.pad_give if a.pad_give else 0.004) / (2 * SC.ERASER_R)))),
                board=dict(plateau=os.environ.get("WIPE_CORR_PLATEAU"),
                           flank=os.environ.get("WIPE_CORR_FLANK"),
                           slope=os.environ.get("WIPE_CORR_SLOPE")))


def report(paths, min_band: float):
    rows = []
    for p in paths:
        for L in open(p):
            if L.startswith("{"):
                rows.append(json.loads(L))
    if not rows:
        print("no rows"); return
    seeds = sorted({r["seed"] for r in rows})
    krs = sorted({r["kr"] for r in rows if r["kr"] is not None}, reverse=True)
    ban = [r for r in rows if r["in_band"] < min_band]
    print(f"{len(rows)} episodes, seeds {seeds}, K_R {krs}")
    print(f"force-band filter (in_band >= {min_band}): {len(ban)} episodes dropped\n")
    keep = [r for r in rows if r["in_band"] >= min_band]

    print(f"  {'condition':>12} {'coverage':>18} {'in_band':>8} "
          f"{'mis_fl':>7} {'yld_pl':>7} {'mis':>6} {'yld':>6} {'f_rmse':>7}")
    tab = {}
    for kr in krs + ["ORACLE"]:
        sub = [r for r in keep if (r["oracle"] if kr == "ORACLE" else r["kr"] == kr)]
        if not sub:
            continue
        cov = np.array([r["coverage"] for r in sub])
        tab[kr] = cov
        def col(k):
            v = [r[k] for r in sub if r.get(k) is not None]
            return float(np.mean(v)) if v else float("nan")
        print(f"  {str(kr):>12} {100*cov.mean():8.1f}% +- {100*cov.std():4.1f} ({len(sub):2d})"
              f" {np.mean([r['in_band'] for r in sub]):8.2f}"
              f" {col('mis_flank'):7.1f} {col('yld_plat'):7.1f}"
              f" {np.mean([r['mis'] for r in sub]):6.1f} {np.mean([r['yld'] for r in sub]):6.1f}"
              f" {np.mean([r['f_rmse'] for r in sub if r['f_rmse'] is not None]):7.2f}")

    const = {k: v for k, v in tab.items() if k != "ORACLE"}
    if not const:
        print("\nno surviving constant"); return
    best = max(const, key=lambda k: const[k].mean())
    print(f"\nsaturation check: best constant coverage {100*const[best].mean():.1f}% "
          f"-- {'SATURATED, raise erase_work' if const[best].mean() > 0.95 else 'ok'}")

    # per-episode ceiling: each episode's own best constant, averaged
    per_seed: dict[int, list[float]] = {}
    for r in keep:
        if r["kr"] is not None:
            per_seed.setdefault(r["seed"], []).append(r["coverage"])
    pe = [max(v) for v in per_seed.values()]
    print(f"\n  best CONSTANT       K_R = {best:<6} {100*const[best].mean():5.1f}%"
          f"  (seed sd {100*const[best].std():.1f})")
    if pe:
        print(f"  best PER EPISODE             {100*np.mean(pe):5.1f}%"
              f"  gap {100*(np.mean(pe) - const[best].mean()):+5.1f} pp")
    if "ORACLE" in tab:
        gap = 100 * (tab["ORACLE"].mean() - const[best].mean())
        thr = max(5.0, 2 * 100 * const[best].std())
        print(f"  within-episode ORACLE        {100*tab['ORACLE'].mean():5.1f}%"
              f"  gap {gap:+5.1f} pp   (threshold {thr:.1f} pp)")
        print(f"\nVERDICT: {'ALIVE -- collect data and train' if gap >= thr else 'DEAD at this cell -- no estimator can beat this oracle'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kr", default="60,30,10,3,1,0.3")
    ap.add_argument("--speed", type=float, default=0.02)
    ap.add_argument("--f", type=float, default=4.0)
    ap.add_argument("--erase-work", type=float, default=0.12)
    ap.add_argument("--text", default="S")
    ap.add_argument("--letter", type=float, default=0.15)
    ap.add_argument("--jitter-mm", type=float, default=20.0)
    ap.add_argument("--time-limit", type=float, default=400.0)
    ap.add_argument("--pad-give", type=float, default=None,
                    help="m the felt squashes; sets the misalignment coverage is blind to")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--report", nargs="*", default=None)
    ap.add_argument("--min-band", type=float, default=0.90)
    a = ap.parse_args()

    if a.report is not None:
        paths = [p for g in (a.report or ["logs/corr_gate_*.log"]) for p in glob.glob(g)]
        report(paths, a.min_band)
        return

    os.environ.setdefault("WIPE_SURFACE", "corr")
    import wipe_scene as SC
    for kr in [float(x) for x in a.kr.split(",")]:
        print(json.dumps(episode(a, a.seed, kr)), flush=True)
    if not a.no_oracle:
        print(json.dumps(episode(a, a.seed, KR_HI,
                                 kr_fn=oracle(SC.ERASER_R, a.f))), flush=True)


if __name__ == "__main__":
    main()
