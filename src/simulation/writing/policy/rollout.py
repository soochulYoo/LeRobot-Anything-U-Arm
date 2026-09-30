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

    def __init__(self, ckpt, n_exec: int = 3, seed: int = 0):
        import jax
        from flax import nnx

        import train as TR
        self.jax = jax
        self.model, self.cfg, self.stats, _ = TR.load_checkpoint(ckpt)
        self._sample = nnx.jit(lambda m, b, k: m.sample(b, k))
        self.n_exec = n_exec
        self.key = jax.random.key(seed)
        self.plans: list[dict] = []

    def reset(self, goal, obs) -> None:
        self.gp = DATA.goal_points(goal["strokes_world"], goal["mask"])
        self.origin = np.asarray(goal["belief_origin"], dtype=np.float64)
        self.hist = deque([np.asarray(obs["f_contact"], np.float32)] * DATA.HIST, maxlen=DATA.HIST)
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
        b = {k: jnp.asarray(v) for k, v in DATA.features(**raw, stats=self.stats).items()}
        self.key, k = self.jax.random.split(self.key)
        a, z = self._sample(self.model, b, k)
        a = self.stats.denorm("act", np.asarray(a)[0])
        targets = x_d + a[:, :3]
        k_diag = np.exp(a[:, 3:])
        self.queue = [{"x_d": targets[j], "k_diag": k_diag[j]} for j in range(self.n_exec)]
        self.plans.append(dict(t=float(obs["t"]), k=k_diag,
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
    return dict(contact_ku=m(contact, 0), contact_kn=m(contact, 2),
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


def _init(ckpt, n_exec):
    global _ENV, _POL
    from evaluate import WritingPolicyEnv
    _ENV = WritingPolicyEnv(control_hz=10.0, cameras=True)
    _POL = FlowPolicy(ckpt, n_exec=n_exec)


def _job(job):
    import protocol as P
    c, text, attempt, duration = job
    spec, _, seed = P.episode_spec(text, c, attempt, 60_000)
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
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(ckpt), args.n_exec)) as pool:
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
            "coverage", "precision", "in_band", "peak_force", "contact_ku", "contact_kn",
            "approach_kn", "hover_ku", "hover_kn")})
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
    plot(sorted(results, key=lambda r: r["seed"]), ref, out / "rollouts.png",
         f"{ckpt.parent.name}: first unseen randomization per case")
    print(f"  wrote {out / 'results.json'} and {out / 'rollouts.png'}")


if __name__ == "__main__":
    main()
