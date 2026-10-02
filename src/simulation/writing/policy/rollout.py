"""Closed-loop evaluation of a trained flow policy in the writing simulator.

The policy runs at 10 Hz in evaluate.WritingPolicyEnv -- the same scene,
Case 1 controller, energy tank, ink rule and score the demonstrations came
from.  Every 0.1 s it sees both cameras, its pose, its own stiffness and the
last second of contact force; every `n_exec` frames it samples a 1 s chunk of
(x_d, K) and executes the first `n_exec` set-points.

Evaluation papers are NEW randomizations (attempt numbers 500+ of the same
seed scheme; demonstrations used 0-49): unseen height, tilt, friction, letter
size, slant and placement.

Besides the task score it reports what the policy did with STIFFNESS, measured
the same way on the demonstrations so the two can be compared:
    contact   mean K (u, n) while the pressure is above the ink threshold
    approach  mean K_n while moving in the air (> 4 mm up, > 8 mm/s in plane)
    hover     mean K (u, n) when hovering low (< 15 mm up, < 3 mm/s) before
              the first contact

Usage (from src/simulation/writing):
    python3 policy/rollout.py --ckpt policy/runs/cross_cond/model.pkl --episodes 20 --workers 12
"""
from __future__ import annotations

import os

# Inference on the CPU, one XLA thread per process: evaluation runs many
# processes side by side, and on a GPU node each would otherwise preallocate
# 75% of the GPU that SAPIEN needs for rendering.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import deque  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import data as DATA  # noqa: E402


class FlowPolicy:
    """evaluate.py's policy interface: reset(goal, obs) and act(obs)."""

    def __init__(self, ckpt, n_exec: int = 3, seed: int = 0, d_lead: int | None = None):
        import jax
        from flax import nnx

        import train as TR
        self.jax = jax
        self.model, self.cfg, self.stats, self.extra = TR.load_checkpoint(ckpt)
        # SAMPLING-ONLY OVERRIDE.  d_lead changes (d)'s inference schedule and
        # nothing in its loss, so the honest way to measure it is to run ONE set
        # of weights both ways.  Retraining with D_LEAD=3 instead gave different
        # weights at every seed -- GPU training is not bit-reproducible across
        # nodes -- so that comparison mixed the schedule with retraining noise
        # (+2.3 +- 6.0 pp, t = 1.07, better 4/8 and worse 2/8: indistinguishable).
        if d_lead is not None:
            self.cfg = dataclasses.replace(self.cfg, d_lead=int(d_lead))
            self.model.cfg = self.cfg        # CrossCond.sample reads the model's
        self._sample = nnx.jit(lambda m, b, k: m.sample(b, k))
        self.n_exec = n_exec
        self.key = jax.random.key(seed)
        self.plans: list[dict] = []
        # A policy trained with --surprise reads nine channels
        # [measured | expected | z], and the expected half needs the expectation
        # model AT INFERENCE.  Its path comes from the checkpoint's own args, so
        # nothing has to be passed in and a surprise policy cannot be rolled out
        # against the wrong one.
        self.sur = None
        if self.cfg.ft_in == DATA.FT_SUR:
            import surprise as SUR
            npz = (self.extra.get("args") or {}).get("surprise")
            if not npz:
                raise ValueError(
                    f"{ckpt} was trained with nine force channels but its args carry no "
                    "--surprise path, so the expectation model cannot be located")
            dp = SUR.deploy_path(npz)
            if not pathlib.Path(dp).exists():
                raise FileNotFoundError(
                    f"{dp} missing: the folds were written without a deploy model. "
                    "Run  python3 policy/surprise.py --deploy-only --data <demos>  "
                    "to fit it without touching the npz the policy was trained on.")
            self.sur = SUR.Deploy(dp)

    def reset(self, goal, obs) -> None:
        self.gp = DATA.goal_points(goal["strokes_world"], goal["mask"])
        self.origin = np.asarray(goal["belief_origin"], dtype=np.float64)
        self.W = np.asarray(goal["belief_R"], dtype=np.float64)   # paper u, v, n
        self.hist = deque([np.asarray(obs["f_contact"], np.float32)] * DATA.HIST, maxlen=DATA.HIST)
        # mu/sigma histories, filled per frame beside the measured one so the
        # three windows stay aligned exactly as data.attach_surprise aligns them
        self.mu_hist = deque(maxlen=DATA.HIST)
        self.sd_hist = deque(maxlen=DATA.HIST)
        self.queue: list[dict] = []
        self.plans = []

    def _plan(self, obs) -> None:
        jnp = self.jax.numpy
        x_d = np.asarray(obs["x_d"], np.float64)
        raw = dict(
            top=DATA._down(np.asarray(obs["rgb_top_camera"])[None]),
            wrist=DATA._down(np.asarray(obs["rgb_wrist_camera"])[None]),
            state=DATA.state_vec(x_d, obs["tcp_pos"], obs["tcp_vel"], obs["k_diag"],
                                 obs["qpos"], self.origin)[None].astype(np.float32),
            goal=(self.gp - x_d[:2]).reshape(1, -1).astype(np.float32),
            ft_hist=np.stack(self.hist)[None])
        # force=self.cfg.force, so a policy trained force-blind stays blind here
        if self.sur is not None:
            # expectation for THIS frame, then the nine channels through the one
            # shared assembler data.attach_surprise also uses
            mu, sd = self.sur.predict(raw["top"], raw["wrist"], raw["state"], raw["goal"])
            if not self.mu_hist:
                for _i in range(DATA.HIST):
                    self.mu_hist.append(mu[0])
                    self.sd_hist.append(sd[0])
            else:
                self.mu_hist.append(mu[0])
                self.sd_hist.append(sd[0])
            raw["ft_hist"] = DATA.surprise_channels(
                raw["ft_hist"], np.stack(self.mu_hist)[None], np.stack(self.sd_hist)[None])
        b = {k: jnp.asarray(v) for k, v in DATA.features(
            **raw, stats=self.stats, force=getattr(self.cfg, "force", True)).items()}
        self.key, k = self.jax.random.split(self.key)
        a, z = self._sample(self.model, b, k)
        a = self.stats.denorm("act", np.asarray(a)[0])
        if DATA.is_spring(self.cfg.layout):
            # x_ref + f_d + log K.  The controller does the K^-1 f_d, so a
            # stiffness error does not become a force error here either --
            # evaluate.WritingPolicyEnv.step.
            # spring: absolute, anchored on x_d(t).  spring_rel: pass the DELTA
            # through and let the env add the measured tip (evaluate.py).
            x_ref = (a[:, :3] if self.cfg.layout == "spring_rel" else x_d + a[:, :3])
            f_d = a[:, 3:6]
            k_diag = np.exp(a[:, 6:9])
            self.queue = [{"x_ref": x_ref[j], "f_d": f_d[j], "k_diag": k_diag[j]}
                          for j in range(self.n_exec)]
            # the future wrench this action implies, in the same world frame and
            # sign as a predicted one, so the two are directly comparable
            f_impl = -(f_d @ np.asarray(self.W).T)
        else:
            targets = x_d + a[:, :3]
            k_diag = np.exp(a[:, 3:])
            self.queue = [{"x_d": targets[j], "k_diag": k_diag[j]} for j in range(self.n_exec)]
            f_impl = None
        self.plans.append(dict(t=float(obs["t"]), k=k_diag, f_implied=f_impl,
                               f_pred=None if z is None else self.stats.denorm("fut_ft", np.asarray(z)[0])))

    def act(self, obs):
        self.hist.append(np.asarray(obs["f_contact"], np.float32))
        if not self.queue:
            self._plan(obs)
        return self.queue.pop(0)


# --------------------------------------------------------------------------- #
def stiffness_summary(t, k, pressure, height, v_plane, ink_force=0.8) -> dict:
    """What the stiffness was in contact, approaching and hovering -- from
    kinematics alone, so demos and rollouts are measured identically."""
    contact = pressure >= ink_force
    first = np.argmax(contact) if contact.any() else len(t)
    air_move = (height > 0.004) & (v_plane > 0.008) & ~contact
    hover = (height < 0.015) & (height > 0.0005) & (v_plane < 0.003) & ~contact & (np.arange(len(t)) < first)
    m = lambda sel, i: float(k[sel, i].mean()) if sel.any() else float("nan")
    return dict(contact_f=float(pressure[contact].mean()) if contact.any() else float("nan"),
                contact_ku=m(contact, 0), contact_kn=m(contact, 2),
                approach_kn=m(air_move, 2), approach_ku=m(air_move, 0),
                hover_ku=m(hover, 0), hover_kn=m(hover, 2))


def run_episode(env, policy, spec) -> dict:
    sim = env.sim
    obs = env.reset(spec)
    policy.reset(env.goal(), obs)
    rows = []
    while not env.time_up:
        obs = env.step(policy.act(obs))
        p = np.asarray(obs["tcp_pos"], np.float64)
        uvh = sim.frame.to_canvas(p)
        rows.append((sim.t, *sim.k_diag(), sim.last["f_n"], uvh[2],
                     float(np.linalg.norm(np.asarray(obs["tcp_vel"])[:2]))))
    res = sim.score()
    L = np.array(rows)
    res.update(stiffness_summary(L[:, 0], L[:, 1:4], L[:, 4], L[:, 5], L[:, 6]))
    res["trace"] = dict(t=L[:, 0].tolist(), k=L[:, 1:4].tolist(), f=L[:, 4].tolist(), h=L[:, 5].tolist())
    res["ink"] = np.asarray(sim.ink_uv).tolist()
    res["target"] = [s.tolist() for s in sim.target.strokes]
    return res


def demo_reference(root) -> dict:
    """The same stiffness summary on the demonstrations, per case."""
    import h5py
    out = {}
    for p in sorted(pathlib.Path(root).glob("*/ep_*.h5")):
        with h5py.File(p) as f:
            case = int(f.attrs["case"])
            full = f["full"]
            R, o = f["privileged/canvas_R"][:], f["privileged/canvas_origin"][:]
            h = (full["tcp_pos"][:] - o) @ R[:, 2]
            s = stiffness_summary(full["t"][:], full["k_diag"][:], full["f_n"][:], h,
                                  np.linalg.norm(full["tcp_vel"][:, :2], axis=1))
        out.setdefault(case, []).append(s)
    return {c: {k: float(np.nanmean([r[k] for r in v])) for k in v[0]} for c, v in out.items()}


# --------------------------------------------------------------------------- #
_ENV = _POL = None
_HARD: dict = {}


def _init(ckpt, n_exec, hard=None, d_lead=None):
    global _ENV, _POL, _HARD
    from evaluate import WritingPolicyEnv
    # The policy is loaded FIRST so the env can be built with the anchor its
    # layout implies.  Getting this wrong is silent: a spring_rel policy decoded
    # with spring's anchor is a different controller, and nothing raises.
    _POL = FlowPolicy(ckpt, n_exec=n_exec, d_lead=d_lead)
    _ENV = WritingPolicyEnv(control_hz=10.0, cameras=True,
                            rel_anchor=(_POL.cfg.layout == "spring_rel"))
    _HARD = dict(hard or {})


def _job(job):
    import protocol as P
    c, text, attempt, duration = job
    spec, _, seed = P.episode_spec(text, c, attempt, 60_000, **_HARD)
    spec = dataclasses.replace(spec, time_limit=duration)
    t0 = time.time()
    res = run_episode(_ENV, _POL, spec)
    res.update(case=c, text=text, seed=seed, wall_s=time.time() - t0)
    res.pop("checks", None)
    return res


def plot(results, ref, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = sorted({r["case"] for r in results})
    fig, ax = plt.subplots(3, len(cases), figsize=(4.6 * len(cases), 9.5), squeeze=False)
    for j, c in enumerate(cases):
        r = next(r for r in results if r["case"] == c)
        a = ax[0, j]
        for s in r["target"]:
            s = np.asarray(s)
            a.plot(1000 * s[:, 0], 1000 * s[:, 1], color="#8a8880", lw=7, alpha=0.35, solid_capstyle="round")
        ink = np.asarray(r["ink"]).reshape(-1, 2)
        if len(ink):
            a.plot(1000 * ink[:, 0], 1000 * ink[:, 1], ".", color="#2a78d6", ms=2.5)
        a.set_aspect("equal")
        a.set_title(f"{r['text']!r} seed {r['seed']}: {'SUCCESS' if r['success'] else r['fail_reason']}\n"
                    f"coverage {100 * r['coverage']:.0f}% precision {100 * r['precision']:.0f}%", fontsize=9)
        tr = r["trace"]
        t, k, f = np.asarray(tr["t"]), np.asarray(tr["k"]), np.asarray(tr["f"])
        ax[1, j].plot(t, f, color="#2a78d6")
        ax[1, j].axhspan(1, 6, color="#1baf7a", alpha=0.08)
        ax[1, j].set_ylabel("pressure (N)")
        ax[2, j].plot(t, k[:, 0], color="#1baf7a", label="K_u (= K_v)")
        ax[2, j].plot(t, k[:, 2], color="#eb6834", label="K_n")
        for lv in (500, 1000, 3000):
            ax[2, j].axhline(lv, color="#8a8880", lw=0.6, ls=":")
        ax[2, j].set_ylabel("stiffness (N/m)")
        ax[2, j].set_xlabel("time (s)")
        ax[2, j].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="demos/protocol_v1", help="for the demo reference and durations")
    ap.add_argument("--texts", nargs="+", default=["S", "7", "<star>"])
    ap.add_argument("--episodes", type=int, default=20, help="per case")
    ap.add_argument("--first-attempt", type=int, default=500, help="demos used 0..49; keep these unseen")
    ap.add_argument("--n-exec", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    # (d) only, SAMPLING-ONLY: re-evaluate an existing checkpoint with the wrench
    # field N steps ahead.  Same weights, different schedule -- the comparison
    # retraining cannot give.
    ap.add_argument("--d-lead", type=int, default=None)
    # THE PAPER'S HIDDEN POSE RANGE AT EVALUATION TIME.
    # The demonstrations were collected at dz = 4 mm and tilt = 5 deg, and the
    # standard evaluation uses the same numbers -- which is why every structure
    # reproduces the demonstrated stiffness to 0.1% and the stiffness axis of
    # the comparison is saturated (policy/runs/COMPARISON.md).  A stiffness only
    # earns its keep when the surface is NOT where the policy expected it, so
    # raise the range and the comparison has somewhere to go.  Nothing is
    # re-collected: this is test-time shift, and the demos stay as they are.
    ap.add_argument("--dz", type=float, default=0.004, help="m, paper height error (demos: 0.004)")
    ap.add_argument("--tilt-deg", type=float, default=5.0, help="deg, paper tilt (demos: 5.0)")
    ap.add_argument("--dz-sweep", default=None,
                    help="comma-separated dz values in mm -- one evaluation per value")
    # TILT IS THE STRESSOR, NOT dz, and this was measured the hard way.
    # dz is a CONSTANT height offset, and the policy absorbs it exactly as the
    # operator does: it descends until it feels contact and then presses.  Swept
    # to 20 mm, act_ft_input/seed0 still succeeded and in-band moved 99.8 ->
    # 96.6%.  A flat curve, and six GPU-hours to learn it.
    # Tilt is deviation DURING contact.  Over a 37 mm glyph the surface height
    # ranges w tan(theta), so at k_n = 1000 N/m the force swings by that many
    # newtons and the sizing rule k_n <~ (f - F_lo)/delta caps k_n at:
    #     tilt   5 deg -> 3.2 mm swing, k_n <~ 1236   (the demos: k_n = 1000, at its limit)
    #            8 deg -> 5.2 mm,       k_n <~  769
    #           12 deg -> 7.9 mm,       k_n <~  509
    #           20 deg -> 13.5 mm,      k_n <~  297   (band is 1-6 N, tear 12 N)
    # So a policy that memorised k_n = 1000 should start losing the band around
    # 8-12 deg, and one trained at a larger --robust-delta should command a softer
    # k_n and outlive it.  That crossover is the experiment.
    ap.add_argument("--tilt-sweep", default=None,
                    help="comma-separated tilt values in deg, e.g. 5,8,12,16,20 -- "
                         "the axis that actually loads the stiffness")
    args = ap.parse_args()

    import multiprocessing as mp
    ckpt = pathlib.Path(args.ckpt)
    out = pathlib.Path(args.out) if args.out else ckpt.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)

    # episode length: 1.3 x the longest demonstration of that case
    rows = [json.loads(l) for l in (pathlib.Path(args.data) / "attempts.jsonl").read_text().splitlines()]
    dur = {t: 1.3 * max(r["t"] for r in rows if r["text"] == t) for t in args.texts}
    jobs = [(c, t, args.first_attempt + i, dur[t]) for c, t in enumerate(args.texts) for i in range(args.episodes)]
    t0 = time.time()

    def run_at(hard: dict) -> list:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                          initargs=(str(ckpt), args.n_exec, hard,
                                                    args.d_lead)) as pool:
            return list(pool.imap_unordered(_job, jobs))

    # ---- the stress sweep -------------------------------------------------
    # One evaluation per paper-height range, nominal first.  This is the curve
    # the stiffness axis of COMPARISON.md cannot show: at the demos' own 4 mm
    # every structure succeeds and commands the demonstrated K to 0.1%, so the
    # question "did it choose a stiffness or memorise one" has no room to be
    # asked.  Where each policy's success falls off IS the answer.
    if args.dz_sweep or args.tilt_sweep:
        if args.tilt_sweep:
            axis, unit = "tilt_deg", "deg"
            vals = [float(v) for v in args.tilt_sweep.split(",")]
            fixed = dict(dz=args.dz)
            held = f"dz {1000 * args.dz:.0f} mm"
        else:
            axis, unit = "dz", "mm"
            vals = [float(v) / 1000.0 for v in args.dz_sweep.split(",")]
            fixed = dict(tilt_deg=args.tilt_deg)
            held = f"tilt {args.tilt_deg} deg"
        shown = (lambda v: 1000 * v) if axis == "dz" else (lambda v: v)
        print(f"{ckpt}\n  {axis} sweep {args.tilt_sweep or args.dz_sweep} {unit}, "
              f"{held} held, {len(jobs)} episodes each")
        print(f"  {axis + ' ' + unit:>9}{'success':>9}{'coverage':>10}{'in band':>9}"
              f"{'contact N':>11}{'peak N':>8}{'K_n':>7}{'K_u':>7}  top failures")
        curve = []
        for v in vals:
            rs = run_at({axis: v, **fixed})
            m = lambda k: float(np.nanmean([r[k] for r in rs]))
            from collections import Counter
            why = Counter(x for r in rs for x in r["fail_reason"].split(",") if x)
            row = dict(axis=axis, x=shown(v),
                       dz_mm=1000 * (v if axis == "dz" else args.dz),
                       tilt_deg=(v if axis == "tilt_deg" else args.tilt_deg),
                       success=float(np.mean([r["success"] for r in rs])),
                       **{k: m(k) for k in ("coverage", "precision", "in_band", "contact_f",
                                            "peak_force", "contact_kn", "contact_ku")},
                       failures=dict(why))
            curve.append(row)
            print(f"  {shown(v):9.0f}{100 * row['success']:8.0f}%{100 * row['coverage']:9.1f}%"
                  f"{100 * row['in_band']:8.1f}%{row['contact_f']:10.2f} {row['peak_force']:7.1f}"
                  f"{row['contact_kn']:7.0f}{row['contact_ku']:7.0f}  "
                  + ", ".join(f"{k} x{v}" for k, v in why.most_common(3)), flush=True)
        name = "tilt_sweep.json" if axis == "tilt_deg" else "dz_sweep.json"
        (out / name).write_text(json.dumps(dict(
            ckpt=str(ckpt), axis=axis, held=held, curve=curve), indent=1))
        print(f"\n  wrote {out / name}  ({(time.time() - t0) / 60:.1f} min)")
        return

    with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                      initargs=(str(ckpt), args.n_exec,
                                                dict(dz=args.dz, tilt_deg=args.tilt_deg),
                                                args.d_lead)) as pool:
        results = []
        for r in pool.imap_unordered(_job, jobs):
            results.append(r)
            print(f"  {r['text']:8s} seed {r['seed']}  {'SUCCESS' if r['success'] else 'fail: ' + r['fail_reason']:<32}"
                  f" cov {r['coverage']:.2f} prec {r['precision']:.2f} band {r['in_band']:.2f} "
                  f"peak {r['peak_force']:.1f} N | K contact u {r['contact_ku']:.0f} n {r['contact_kn']:.0f}"
                  f"  approach n {r['approach_kn']:.0f}", flush=True)
    ref = demo_reference(args.data)
    summary = {}
    print(f"\n{ckpt}  ({(time.time() - t0) / 60:.1f} min, n_exec {args.n_exec})")
    print(f"  {'case':<9}{'success':>9}{'coverage':>10}{'precision':>10}{'in band':>9}{'peak':>7}"
          f"  | K contact u/n  approach n  hover u/n   (demos)")
    for c, t in enumerate(args.texts):
        rs = [r for r in results if r["case"] == c]
        m = lambda k: float(np.nanmean([r[k] for r in rs]))
        s = dict(success=float(np.mean([r["success"] for r in rs])), **{k: m(k) for k in (
            "coverage", "precision", "in_band", "peak_force", "contact_f", "contact_ku",
            "contact_kn", "approach_kn", "hover_ku", "hover_kn")})
        summary[t] = s
        d = ref.get(c, {})
        print(f"  {t:<9}{100 * s['success']:8.0f}%{100 * s['coverage']:9.1f}%{100 * s['precision']:9.1f}%"
              f"{100 * s['in_band']:8.1f}%{s['peak_force']:6.1f}N"
              f"  | {s['contact_ku']:5.0f}/{s['contact_kn']:<5.0f}  {s['approach_kn']:6.0f}    "
              f"{s['hover_ku']:5.0f}/{s['hover_kn']:<5.0f} "
              f"({d.get('contact_ku', np.nan):.0f}/{d.get('contact_kn', np.nan):.0f}, "
              f"{d.get('approach_kn', np.nan):.0f}, {d.get('hover_ku', np.nan):.0f}/{d.get('hover_kn', np.nan):.0f})")
    from collections import Counter
    why = Counter(x for r in results for x in r["fail_reason"].split(",") if x)
    if why:
        print("  failures: " + ", ".join(f"{k} x{v}" for k, v in why.most_common()))
    slim = [{k: v for k, v in r.items() if k not in ("trace", "ink", "target")} for r in results]
    (out / "results.json").write_text(json.dumps(dict(summary=summary, demo_reference=ref,
                                                      episodes=slim), indent=1))
    # Every episode's ink and its own target, for policy/plot_overlay.py: the
    # paper is randomized per episode, so each drawing only means anything
    # beside the target it was aiming at.
    traces = {}
    for r_ in results:
        tag = f"c{r_['case']}_s{r_['seed']}"
        traces[f"{tag}_ink"] = np.asarray(r_["ink"], np.float32).reshape(-1, 2)
        traces[f"{tag}_tgt"] = np.concatenate([np.asarray(s, np.float32) for s in r_["target"]])
        traces[f"{tag}_ok"] = np.asarray([r_["success"]])
        # the force and stiffness the episode actually ran at, at policy rate,
        # for policy/plot_force.py -- results.json drops the trace as too bulky
        tr = r_["trace"]
        traces[f"{tag}_t"] = np.asarray(tr["t"], np.float32)
        traces[f"{tag}_f"] = np.asarray(tr["f"], np.float32)
        traces[f"{tag}_k"] = np.asarray(tr["k"], np.float32)
    np.savez_compressed(out / "traces.npz", **traces)
    plot(sorted(results, key=lambda r: r["seed"]), ref, out / "rollouts.png",
         f"{ckpt.parent.name}: first unseen randomization per case")
    print(f"  wrote {out / 'results.json'}, {out / 'rollouts.png'} and {out / 'traces.npz'}")


if __name__ == "__main__":
    main()
