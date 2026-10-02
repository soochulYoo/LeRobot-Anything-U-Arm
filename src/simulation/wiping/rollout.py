"""Closed-loop evaluation of a trained wiping policy, K_R included.

`../writing/policy/rollout.py` does this for the pen.  Everything structural is
the same -- the policy runs at 10 Hz in the scene, controller, energy tank and
score the demonstrations came from, sampling a 1 s chunk every `n_exec` frames
and executing the first three set-points -- and two things differ, both of them
the reason this task exists:

  THE ACTION CARRIES A ROTATIONAL STIFFNESS.  The chunk is (10, 7), and the
  seventh column is log K_R, applied through the same gated rate the operator
  used (`Case1Proposal.Ur`).  A policy that places the pad correctly and holds
  the wrong K_R has learned the motions of wiping without the compliance that
  makes it work, and only a closed loop can tell those apart: open loop they
  differ by 0.1 in a log.

  THE SCORE IS WHAT CAME OFF THE BOARD.  Not coverage and precision of ink --
  the fraction of the glyph erased, the force band, and how flush the pad lay
  against the TRUE local normal while it was wiping.

Evaluation boards are NEW randomizations: attempt numbers 500+ of the seed
scheme `protocol.episode_spec` uses, where the demonstrations took 0-60.  The
board's SHAPE is the one the policy trained on (it is baked into the scene);
its height, tilt, friction and the glyph's size and placement are all unseen.

Every stiffness number is measured on the demonstrations the same way, from
kinematics alone, so the two columns can be read against each other.

    python3 rollout.py --ckpt runs/act_film/seed0/model.pkl --episodes 20 --workers 10
    python3 rollout.py --ckpt runs/act_film/seed0/model.pkl --demo-only
"""
from __future__ import annotations

import os

# Inference on the CPU, one XLA thread per process: evaluation runs many
# processes side by side, and on a GPU node each would otherwise preallocate
# 75% of the GPU that SAPIEN needs for rendering.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse  # noqa: E402
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
sys.path.insert(0, str(HERE.parent / "writing"))
sys.path.insert(0, str(HERE.parent / "writing" / "policy"))

import controller as C  # noqa: E402
import data as DATA  # noqa: E402
import protocol as P  # noqa: E402  the wiping one: episode_spec and the seed scheme
from wipe_sim import CurvedWipingSim, WipingSim  # noqa: E402

INK = 0.8          # N, the pressure above which the pad counts as down


# --------------------------------------------------------------------------- #
class WipePolicyEnv:
    """The wiping simulator at the policy's rate.  One action is (x_d, K, K_R)
    reached by the next policy frame, interpolated over the physics steps and
    fed through Case 1 as rates, exactly as the teleoperated session did."""

    def __init__(self, board: str = "curved", control_hz: float = 10.0,
                 image_size: int = 128, cameras: bool = True,
                 render_mode: str | None = None):
        self.sim = (CurvedWipingSim if board == "curved" else WipingSim)(
            image_size=image_size, cameras=cameras, render_mode=render_mode)
        self.n_sub = max(1, int(round(1.0 / (control_hz * self.sim.dt))))
        self.frames: list[np.ndarray] = []
        self.record_video = False

    def reset(self, spec) -> dict:
        self.spec = spec
        self.frames = []
        return self.sim.reset(spec)

    def goal(self) -> dict:
        return self.sim.goal()

    @property
    def time_up(self) -> bool:
        return self.sim.t >= self.spec.time_limit

    def step(self, action) -> dict:
        sim = self.sim
        x_goal = np.asarray(action["x_d"], dtype=float).reshape(3)
        k_goal = np.asarray(action["k_diag"], dtype=float).reshape(3)
        kr_goal = float(action.get("kr", sim.ctl.kr))
        K_goal = C.k_world(k_goal, sim.W)
        x0, K0, kr0 = sim.ctl.x_d.copy(), sim.ctl.K.copy(), float(sim.ctl.kr)
        for i in range(1, self.n_sub + 1):
            w = i / self.n_sub
            x_next = x0 + w * (x_goal - x0)
            K_next = K0 + w * (K_goal - K0)
            kr_next = kr0 + w * (kr_goal - kr0)
            sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt,
                                     (K_next - sim.ctl.K) / sim.dt,
                                     Ur=(kr_next - sim.ctl.kr) / sim.dt))
        if self.record_video:
            r = sim.env.render()
            r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
            self.frames.append(r[0] if r.ndim == 4 else r)
        return sim.observe()


class WipePolicy:
    """A trained checkpoint behind reset(goal, obs) / act(obs)."""

    def __init__(self, ckpt, n_exec: int = 3, seed: int = 0):
        import jax
        from flax import nnx

        import train as TR
        self.jax = jax
        self.model, self.cfg, self.stats, _ = TR.load_checkpoint(ckpt)
        self._sample = nnx.jit(lambda m, b, k: m.sample(b, k))
        self.n_exec = n_exec
        self.key = jax.random.key(seed)

    def reset(self, goal, obs) -> None:
        self.gp = DATA.goal_points(goal["strokes_world"], goal["mask"])
        self.origin = np.asarray(goal["belief_origin"], dtype=np.float64)
        self.hist = deque([np.asarray(obs["f_contact"], np.float32)] * DATA.HIST,
                          maxlen=DATA.HIST)
        self.queue: list[dict] = []
        self.plans: list[dict] = []

    def _plan(self, obs) -> None:
        jnp = self.jax.numpy
        x_d = np.asarray(obs["x_d"], np.float64)
        # kr enters the state only on a checkpoint that was trained with it,
        # which is exactly the checkpoints whose action is 7 wide.
        kr = float(obs["kr"]) if self.cfg.A > 6 else None
        raw = dict(
            top=DATA._down(np.asarray(obs["rgb_top_camera"])[None]),
            wrist=DATA._down(np.asarray(obs["rgb_wrist_camera"])[None]),
            state=DATA.state_vec(x_d, obs["tcp_pos"], obs["tcp_vel"], obs["k_diag"],
                                 obs["qpos"], self.origin, kr)[None].astype(np.float32),
            goal=(self.gp - x_d[:2]).reshape(1, -1).astype(np.float32),
            ft_hist=np.stack(self.hist)[None])
        b = {k: jnp.asarray(v) for k, v in DATA.features(**raw, stats=self.stats).items()}
        self.key, k = self.jax.random.split(self.key)
        a, z = self._sample(self.model, b, k)
        a = self.stats.denorm("act", np.asarray(a)[0])
        targets = x_d + a[:, :3]
        k_diag = np.exp(a[:, 3:6])
        krs = np.exp(a[:, 6]) if a.shape[-1] > 6 else np.full(len(a), np.nan)
        self.queue = [{"x_d": targets[j], "k_diag": k_diag[j],
                       **({} if np.isnan(krs[j]) else {"kr": float(krs[j])})}
                      for j in range(self.n_exec)]
        self.plans.append(dict(t=float(obs["t"]), kr=krs.tolist()))

    def act(self, obs):
        self.hist.append(np.asarray(obs["f_contact"], np.float32))
        if not self.queue:
            self._plan(obs)
        return self.queue.pop(0)


# --------------------------------------------------------------------------- #
def regimes(pressure, height, v_plane):
    """The three kinematic windows, identical for demos and rollouts.

    Defined on where the pad IS and how fast it moves, never on a phase label:
    a policy has no phases, and a comparison that used the demonstrator's would
    be scoring the two on different clocks.
    """
    contact = pressure >= INK
    first = int(np.argmax(contact)) if contact.any() else len(pressure)
    return dict(
        contact=contact,
        # WIPING is contact that is MOVING.  Pressing down is contact too, and
        # the protocol is still ramping K_R through it: averaged over all
        # contact the demonstrations read 7.4 Nm/rad, over the moving part
        # 0.44, and only the second is the stiffness the pad wipes with.  A
        # kinematic window, not a phase, because a policy has no phases.
        wipe=contact & (v_plane > 0.010),
        approach=(height > 0.004) & (v_plane > 0.008) & ~contact,
        hover=((height < 0.015) & (height > 0.0005) & (v_plane < 0.003)
               & ~contact & (np.arange(len(pressure)) < first)))


def summarize(pressure, height, v_plane, k, kr, mis, ask) -> dict:
    """What the stiffness and the pad's alignment were, per regime."""
    sel = regimes(pressure, height, v_plane)
    m = lambda s, v: float(np.nanmean(v[s])) if s.any() else float("nan")
    out = {"contact_f": m(sel["contact"], pressure)}
    for name, s in sel.items():
        out[f"{name}_ku"] = m(s, k[:, 0])
        out[f"{name}_kn"] = m(s, k[:, 2])
        out[f"{name}_kr"] = m(s, kr)
    for w in ("contact", "wipe"):
        out[f"{w}_mis"] = m(sel[w], mis)
        out[f"{w}_ask"] = m(sel[w], ask)
        out[f"{w}_left"] = out[f"{w}_mis"] / max(out[f"{w}_ask"], 1e-6)
    return out


def run_episode(env: WipePolicyEnv, policy, spec) -> dict:
    sim = env.sim
    obs = env.reset(spec)
    policy.reset(env.goal(), obs)
    up = np.array([0.0, 0.0, 1.0])
    rows = []
    while not env.time_up:
        obs = env.step(policy.act(obs))
        rec = sim.last
        uvh = rec["contact_uvh"]
        n = (sim.frame.normal_at(uvh[:2]) if hasattr(sim.frame, "normal_at")
             else sim.frame.normal)
        rows.append((sim.t, *sim.k_diag(), float(sim.ctl.kr), rec["f_n"], uvh[2],
                     float(np.linalg.norm(np.asarray(obs["tcp_vel"])[:2])),
                     np.degrees(np.arccos(np.clip(-rec["R"][:, 2] @ n, -1.0, 1.0))),
                     np.degrees(np.arccos(np.clip(n @ up, -1.0, 1.0)))))
    res = sim.score()
    L = np.array(rows)
    res.update(summarize(L[:, 5], L[:, 6], L[:, 7], L[:, 1:4], L[:, 4], L[:, 8], L[:, 9]))
    res["trace"] = dict(t=L[:, 0].tolist(), kr=L[:, 4].tolist(), f=L[:, 5].tolist(),
                        mis=L[:, 8].tolist())
    return res


def demo_reference(root) -> dict:
    """The same summary on the demonstrations, per case.  Their `mis` and `ask`
    are recorded per step, so nothing has to be re-simulated."""
    import h5py
    out: dict[int, list] = {}
    for p in sorted(pathlib.Path(root).glob("*/ep_*.h5")):
        with h5py.File(p) as f:
            case = int(f.attrs["case"])
            full = f["full"]
            R, o = f["privileged/canvas_R"][:], f["privileged/canvas_origin"][:]
            h = (full["tcp_pos"][:] - o) @ R[:, 2]
            s = summarize(full["f_n"][:], h, np.linalg.norm(full["tcp_vel"][:, :2], axis=1),
                          full["k_diag"][:], full["kr"][:], full["mis"][:], full["ask"][:])
            s["erased"] = float(json.loads(f.attrs["metrics"])["erased"])
            s["in_band"] = float(json.loads(f.attrs["metrics"])["in_band"])
        out.setdefault(case, []).append(s)
    return {c: {k: float(np.nanmean([r[k] for r in v])) for k in v[0]} for c, v in out.items()}


# --------------------------------------------------------------------------- #
_ENV = _POLICY = None


def _init(board, ckpt, n_exec, image_size):
    global _ENV, _POLICY
    _ENV = WipePolicyEnv(board=board, image_size=image_size)
    _POLICY = WipePolicy(ckpt, n_exec=n_exec)


def _job(job):
    c, text, attempt, base = job
    spec, _, seed = P.episode_spec(text, c, attempt, base)
    r = run_episode(_ENV, _POLICY, spec)
    r.update(case=c, text=text, seed=seed)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--board", choices=("curved", "flat"), default="curved")
    ap.add_argument("--texts", nargs="+", default=["S", "7", "<star>"])
    ap.add_argument("--episodes", type=int, default=10, help="per text")
    ap.add_argument("--attempt-base", type=int, default=500,
                    help="unseen randomizations: the demos used 0-60")
    ap.add_argument("--seed-base", type=int, default=70_000)
    ap.add_argument("--n-exec", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--demos", default="demos/wipe_curved",
                    help="the demonstrations to measure the same way")
    ap.add_argument("--demo-only", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ref = demo_reference(a.demos) if pathlib.Path(a.demos).exists() else {}
    if a.demo_only:
        for c, r in sorted(ref.items()):
            print(f"  case {c}: erased {100*r['erased']:.1f}%  in-band {100*r['in_band']:.1f}%"
                  f"  K_R wiping {r['wipe_kr']:.2f} approach {r['approach_kr']:.1f}"
                  f"  misaligned {r['wipe_mis']:.1f} of {r['wipe_ask']:.1f} deg")
        return

    jobs = [(c, t, a.attempt_base + i, a.seed_base)
            for c, t in enumerate(a.texts) for i in range(a.episodes)]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(
                a.workers, initializer=_init,
                initargs=(a.board, a.ckpt, a.n_exec, a.image_size)) as pool:
            results = list(pool.imap_unordered(_job, jobs))
    else:
        _init(a.board, a.ckpt, a.n_exec, a.image_size)
        results = [_job(j) for j in jobs]
    print(f"  {len(results)} episodes in {(time.time() - t0)/60:.1f} min\n")

    def col(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None and not np.isnan(float(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    print("  text      n   success   erased   in-band   peak     K_R wiping / approach"
          "    misaligned / asked")
    for c, t in enumerate(a.texts):
        rs = [r for r in results if r["case"] == c]
        if not rs:
            continue
        d = ref.get(c, {})
        print(f"  {t!r:8s} {len(rs):3d}  {100*np.mean([r['success'] for r in rs]):6.0f}%"
              f"  {100*col(rs,'erased'):6.1f}%  {100*col(rs,'in_band'):6.1f}%"
              f"  {col(rs,'peak_force'):5.1f} N"
              f"   {col(rs,'wipe_kr'):6.2f} / {col(rs,'approach_kr'):6.2f}"
              f"      {col(rs,'wipe_mis'):5.1f} / {col(rs,'wipe_ask'):4.1f} deg")
        if d:
            print(f"  {'  demos':8s}      {'':7s}  {100*d['erased']:6.1f}%"
                  f"  {100*d['in_band']:6.1f}%  {'':7s}"
                  f"   {d['wipe_kr']:6.2f} / {d['approach_kr']:6.2f}"
                  f"      {d['wipe_mis']:5.1f} / {d['wipe_ask']:4.1f} deg")
    print(f"\n  overall success {100*np.mean([r['success'] for r in results]):.0f}%"
          f"  erased {100*col(results,'erased'):.1f}%"
          f"  K_R while wiping {col(results,'wipe_kr'):.2f} Nm/rad")
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(
            {"episodes": [{k: v for k, v in r.items() if k != "trace"} for r in results],
             "demo_reference": {str(k): v for k, v in ref.items()},
             "ckpt": a.ckpt, "board": a.board}, indent=1))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
