"""Collect spiral-wipe demonstrations from the synthetic operator.

Each episode: a different random surface, a different tilt, a different
operator, the SAME kind of intent -- an Archimedean spiral of fixed pitch
pressed at a constant normal force.  That structure is the point.  The intent is
shared and known, the disturbance is not, so a policy trained across episodes is
being asked to recover the intent rather than to copy any one execution, and can
be scored against the intent rather than against its teacher.

WHAT IS LOGGED

  observation   what a policy could see: tool pose and velocity, the contact
                wrench, and the CONTROLLER STATE (inner deflection, outer
                reference velocity).  The deflection is a clean force proxy that
                is a function of the commanded history rather than of the
                measured force, so it does not offer the same shortcut.
  action        the unified force-impedance triple under the FIXED GAUGE:
                    x_ref := x_r     (NOT x_c -- see below)
                    f_d   := f_ch
                    K     := the coupling stiffness actually applied
  intent        the ideal spiral and the intended force: ground truth, available
                only because this is a simulator, and the reason the scenario is
                worth running.
  privileged    the surface height under the tool and the operator's true arm
                stiffness, for studies that need an answer key.

THE GAUGE.  (x_ref, f_d, K) is over-parameterised: the drive f_d + K(x_ref - x_r)
is set by two numbers, not three.  Taking f_d := f_ch AND x_ref := x_c
double-counts the coupling force -- replay gives 1.93x the demonstrated force
(cascade/tests.py section 4).  Fixing x_ref := x_r makes the spring term vanish
and f_d carry all of the contact intent, which is why it is done here and not
left to the reader.

FORCES ARE FILTERED.  A raw per-step contact force is not what a sensor reports;
in the SAPIEN scenes the raw signal reads zero on a third of its samples while
its mean is correct.  Both the full-rate and the policy-rate views are filtered,
and the policy-rate view also keeps the peak inside each step, because a policy
that only sees means cannot know it is hammering.

Usage:  python3 collect_demos.py --episodes 200 --workers 14 --out demos/spiral_v1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# success thresholds, relative to the intent
MAX_PATH = 0.008          # m RMS from the ideal spiral (pitch is ~22 mm)
MAX_FORCE_REL = 0.25      # of the intended normal force
MIN_CONTACT = 0.95


def sample_episode(seed: int) -> dict:
    """Everything that varies between demonstrations."""
    rng = np.random.default_rng(10_000 + seed)
    return dict(
        seed=seed,
        tilt_x=float(rng.uniform(0.17, 0.44)),          # 10-25 deg
        tilt_y=float(rng.uniform(-0.31, -0.07)),        # -18 to -4 deg
        surf_amp=float(rng.uniform(0.0015, 0.0035)),
        surf_len=float(rng.uniform(0.012, 0.025)),
        op_magnitude=float(rng.uniform(600.0, 1400.0)),
        op_ratio=float(rng.uniform(5.0, 12.0)),
        op_visual_gain=float(rng.uniform(0.40, 0.80)),
        op_visual_delay=float(rng.uniform(0.15, 0.28)),
        op_noise=float(rng.uniform(0.010, 0.035)),
        pitch=float(rng.uniform(0.018, 0.028)),
        f_normal=float(rng.uniform(6.0, 10.0)),
    )


def lowpass(y: np.ndarray, dt: float, hz: float) -> np.ndarray:
    w = max(1, int(round(1.0 / (hz * dt))))
    if y.ndim == 1:
        pad = np.r_[np.full(w, y[0]), y, np.full(w, y[-1])]
        return np.convolve(pad, np.ones(w) / w, mode="same")[w:-w]
    return np.column_stack([lowpass(y[:, i], dt, hz) for i in range(y.shape[1])])


def run_episode(job) -> dict:
    """One demonstration.  Imports live here so workers start clean."""
    import warnings as _w
    _w.filterwarnings("ignore")
    import numpy as _np
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import spiral_task as S

    spec, out_dir, policy_hz, force_hz, log_hz = job
    task = S.SpiralTask(tilt_x=spec["tilt_x"], tilt_y=spec["tilt_y"])
    intent = S.SpiralIntent(pitch=spec["pitch"], f_normal=spec["f_normal"])
    op = S.Operator(magnitude=spec["op_magnitude"], ratio=spec["op_ratio"],
                    visual_gain=spec["op_visual_gain"], visual_delay=spec["op_visual_delay"],
                    motor_noise=spec["op_noise"], seed=spec["seed"])
    surface = S.RandomSurface(amp=spec["surf_amp"], length_scale=spec["surf_len"],
                              seed=spec["seed"])
    try:
        log = S.demonstrate(task, intent, op, surface)
    except Exception as exc:                       # a diverged episode is a result
        return dict(**spec, ok=False, reason=f"sim failed: {type(exc).__name__}")

    g = S.gap(log)
    dt = float(log["_dt"])
    R = task.R
    t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
    t = log["t"]
    keep = t >= task.settle

    f_e = lowpass(_np.asarray(log["f_e"]), dt, force_hz)[keep]
    f_n = lowpass(_np.asarray(log["f_normal"])[:, 0], dt, force_hz)[keep]
    x = _np.asarray(log["x"])[keep]
    v = _np.asarray(log["v"])[keep]
    x_r = _np.asarray(log["x_r"])[keep]
    v_r = _np.asarray(log["v_r"])[keep]
    f_ch = _np.asarray(log["f_ch"])[keep]
    tt = t[keep]

    intent_xy = _np.array([intent.offset(float(s), task.settle, t1, t2) for s in tt])
    surf_h = _np.array([surface.height(float(p @ t1), float(p @ t2)) for p in x])
    x_c = _np.asarray(log["x_c"])[keep]

    # --- policy-rate view -------------------------------------------------- #
    step = max(1, int(round(1.0 / (policy_hz * dt))))
    m = (len(tt) // step) * step
    def block(a):
        return a[:m].reshape(m // step, step, *a.shape[1:])
    p_t = block(tt).mean(axis=1)
    p_obs_x = block(x).mean(axis=1)
    p_obs_v = block(v).mean(axis=1)
    p_obs_f = block(f_e).mean(axis=1)
    p_obs_fpeak = _np.abs(block(f_n)).max(axis=1)      # a mean hides hammering
    p_obs_defl = block(x_r - x).mean(axis=1)
    p_act_xref = block(x_r).mean(axis=1)               # gauge: x_ref := x_r
    p_act_fd = block(f_ch).mean(axis=1)                # gauge: f_d   := f_ch

    # Decimate the full-rate record.  The sim runs at 1 kHz and storing that
    # costs ~2 MB an episode; the forces are already low-passed at force_hz, so
    # dropping to log_hz aliases nothing and still leaves an order of magnitude
    # over the fastest thing in the system (the in-contact resonance, ~9 Hz).
    dec = max(1, int(round(1.0 / (log_hz * dt))))
    sl = slice(None, None, dec)

    ok = (g["contact"] >= MIN_CONTACT
          and g["path_result"] <= MAX_PATH
          and abs(g["f_mean"] - intent.f_normal) <= MAX_FORCE_REL * intent.f_normal)
    reason = "" if ok else (
        f"contact {100*g['contact']:.0f}%" if g["contact"] < MIN_CONTACT else
        f"path {1000*g['path_result']:.1f} mm" if g["path_result"] > MAX_PATH else
        f"force {g['f_mean']:.1f} vs {intent.f_normal:.1f} N")

    if ok:
        _np.savez_compressed(
            pathlib.Path(out_dir) / f"ep_{spec['seed']:05d}.npz",
            t=tt[sl].astype(_np.float32),
            obs_x=x[sl].astype(_np.float32), obs_v=v[sl].astype(_np.float32),
            obs_f=f_e[sl].astype(_np.float32), obs_f_normal=f_n[sl].astype(_np.float32),
            obs_defl_inner=(x_r - x)[sl].astype(_np.float32),
            obs_v_r=v_r[sl].astype(_np.float32),
            act_x_ref=x_r[sl].astype(_np.float32), act_f_d=f_ch[sl].astype(_np.float32),
            # the decoded command, stored ONLY so the verifier can replay the
            # wrong gauge and show that the check is able to fail
            diag_x_c=x_c[sl].astype(_np.float32),
            act_K=_np.full((3, 3), 0.0, _np.float32) + task.Ka * _np.eye(3, dtype=_np.float32),
            intent_xy=intent_xy[sl].astype(_np.float32),
            intent_f=_np.float32(intent.f_normal),
            priv_surface_h=surf_h[sl].astype(_np.float32),
            priv_Kh=_np.asarray(log["_Kh_true"], dtype=_np.float32),
            task_frame=R.astype(_np.float32),
            p_t=p_t.astype(_np.float32), p_obs_x=p_obs_x.astype(_np.float32),
            p_obs_v=p_obs_v.astype(_np.float32), p_obs_f=p_obs_f.astype(_np.float32),
            p_obs_f_peak=p_obs_fpeak.astype(_np.float32),
            p_obs_defl=p_obs_defl.astype(_np.float32),
            p_act_x_ref=p_act_xref.astype(_np.float32),
            p_act_f_d=p_act_fd.astype(_np.float32),
        )
    return dict(**spec, ok=bool(ok), reason=reason, dt=dt * dec, sim_dt=dt,
                steps=int(len(tt[sl])),
                policy_steps=int(len(p_t)),
                path_mm=1000 * g["path_result"], path_cmd_mm=1000 * g["path_command"],
                force_rmse=g["force_result"], f_mean=g["f_mean"], contact=g["contact"])


def verify(out_dir: pathlib.Path, n_check: int = 3) -> None:
    """Replay the stored labels through the cascade and compare to the demo.

    This is the check that matters.  Everything else -- shapes, field names, a
    gauge identity that is true by construction -- can pass while the labels
    still fail to reproduce what the operator did.  Replaying (x_ref, f_d, K)
    into POLICY mode on the same surface and asking whether the same force comes
    out is the only test that would catch it.
    """
    import dataclasses
    import core as C
    import spiral_task as S

    meta = json.loads((out_dir / "index.json").read_text())
    kept = [e for e in meta["episodes"] if e["ok"]][:n_check]
    print(f"\n  replaying {len(kept)} episodes from their stored labels:")
    for spec in kept:
        d = np.load(out_dir / f"ep_{spec['seed']:05d}.npz")
        task = S.SpiralTask(tilt_x=spec["tilt_x"], tilt_y=spec["tilt_y"])
        surface = S.RandomSurface(amp=spec["surf_amp"], length_scale=spec["surf_len"],
                                  seed=spec["seed"])
        R = task.R
        t1, t2, n = R[:, 0], R[:, 1], R[:, 2]
        base = C.CascadeParams(
            n=3, outer=C.OuterParams(Ma=task.Ma, Br=task.Br),
            inner=C.InnerParams(Ki=task.Ki, Di=task.Di, Ms=task.Ms),
            env=C.EnvParams(ke=task.ke, de=task.de, x_wall=task.wall, normal=n,
                            mu=task.mu,
                            slope_fn=lambda _t, x: (lambda g: g[0] * t1 + g[1] * t2)(
                                surface.grad(float(x @ t1), float(x @ t2)))))
        # Replay at the SIMULATION step, holding each stored action for the
        # decimation factor.  Stepping at the stored rate instead adds its own
        # integration error (0.5 N on one episode) and would be mistaken for a
        # labelling error.
        sim_dt = float(spec.get("sim_dt", spec["dt"]))
        dec = max(1, int(round(float(spec["dt"]) / sim_dt)))
        prm = dataclasses.replace(base, dt=sim_dt)
        xr, fd, K = d["act_x_ref"], d["act_f_d"], d["act_K"].astype(float)
        idx = {"i": 0}

        def replay(x_ref_seq):
            idx["i"] = 0
            sim = C.CascadeSim(prm, "policy",
                               wall_offset_fn=lambda _t, x: surface.height(
                                   float(x @ t1), float(x @ t2)))
            sim.x_r = d["act_x_ref"][0].astype(float).copy()
            sim.x = d["obs_x"][0].astype(float).copy()
            sim._x_r0 = sim.x_r.copy()

            def act(_t, _last):
                i = min(idx["i"] // dec, len(x_ref_seq) - 1)
                idx["i"] += 1
                return C.Action(x_ref_seq[i].astype(float), fd[i].astype(float), K)

            lg = sim.run(len(x_ref_seq) * dec * prm.dt, act)
            return lowpass(np.asarray(lg["f_normal"])[:, 0], prm.dt, meta["force_hz"])

        ref = float(np.mean(d["obs_f_normal"]))
        good, bad = replay(xr)[::dec], replay(d["diag_x_c"])[::dec]
        m = min(len(good), len(d["obs_f_normal"]))
        e_g = float(np.sqrt(np.mean((good[:m] - d["obs_f_normal"][:m]) ** 2)))
        mb = min(len(bad), len(d["obs_f_normal"]))
        print(f"    ep {spec['seed']:>5}  demo {ref:5.2f} N | x_ref:=x_r {np.mean(good[:m]):6.2f} N"
              f" RMSE {e_g:4.2f} | x_ref:=x_c {np.mean(bad[:mb]):7.2f} N"
              f" ({np.mean(bad[:mb])/max(ref,1e-9):5.2f}x the demo)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="demos/spiral_v1")
    ap.add_argument("--policy-hz", type=float, default=20.0)
    ap.add_argument("--force-hz", type=float, default=20.0)
    ap.add_argument("--log-hz", type=float, default=100.0,
                    help="rate of the full-rate record after decimation")
    ap.add_argument("--no-verify", dest="verify", action="store_false", default=True,
                    help="skip the label-replay check")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(sample_episode(i), str(out), args.policy_hz, args.force_hz, args.log_hz)
            for i in range(args.episodes)]

    print(f"collecting {args.episodes} episodes into {out} on {args.workers} workers")
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            rows.append(r)
            if i % max(1, args.episodes // 20) == 0:
                print(f"  {i}/{args.episodes}  kept {sum(x['ok'] for x in rows)}")

    rows.sort(key=lambda r: r["seed"])
    kept = [r for r in rows if r["ok"]]
    meta = dict(n_requested=args.episodes, n_kept=len(kept),
                policy_hz=args.policy_hz, force_hz=args.force_hz, log_hz=args.log_hz,
                thresholds=dict(max_path_m=MAX_PATH, max_force_rel=MAX_FORCE_REL,
                                min_contact=MIN_CONTACT),
                gauge="x_ref := x_r, f_d := f_ch", episodes=rows)
    (out / "index.json").write_text(json.dumps(meta, indent=1))

    print("\n" + "=" * 84)
    print(f"kept {len(kept)}/{args.episodes} episodes "
          f"({100*len(kept)/args.episodes:.0f}%)")
    if kept:
        for f, lab, sc in (("path_mm", "path error", 1.0), ("force_rmse", "force RMSE", 1.0),
                           ("contact", "contact held", 100.0)):
            v = np.array([r[f] for r in kept]) * sc
            print(f"  {lab:<14} {v.mean():7.2f} +- {v.std():5.2f}   "
                  f"[{v.min():.2f}, {v.max():.2f}]")
        steps = np.array([r["policy_steps"] for r in kept])
        print(f"  {'policy steps':<14} {steps.mean():7.0f} per episode at "
              f"{args.policy_hz:.0f} Hz  ({steps.sum()} total)")
    rej = [r for r in rows if not r["ok"]]
    if rej:
        from collections import Counter
        why = Counter(r["reason"].split()[0] for r in rej)
        print(f"  rejected {len(rej)}: " + ", ".join(f"{k} x{v}" for k, v in why.most_common()))
    print(f"  index written to {out/'index.json'}")
    if kept and args.verify:
        verify(out, n_check=min(3, len(kept)))


if __name__ == "__main__":
    main()
