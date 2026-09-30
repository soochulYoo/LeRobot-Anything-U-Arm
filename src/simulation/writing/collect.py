"""Collect teleoperated writing demonstrations.

Each episode: a random word, digit string or shape, a random paper pose and
friction (hidden), a random writer, the same bilateral Case 1 teleoperation.
Only successful episodes are kept, and every episode -- kept or not -- is
listed in index.json with its metrics and the reason it failed.

ONE HDF5 FILE PER EPISODE

  attrs         text, spec / style / master / gains / criteria (JSON), metrics
  /full/*       100 Hz (log_hz), everything, for analysis and label checks:
                  robot     qpos qvel tau tcp_pos tcp_quat tcp_vel tcp_angvel
                  force     f_contact (sensor, 25 Hz bandwidth)  f_raw_mean (the
                            raw PhysX force averaged over each 10 ms window --
                            its impulse, unbiased by the filter)  f_n (the 5 Hz
                            pressure that inks and tears)  pen_down
                  Case 1    x_d_req x_d  k_req k_diag  K (3x3 world)  D  alpha  E
                            (state at the start of each 2 ms step; x_d_req is
                            the request FOR the next step, so with the gate
                            open it equals x_d 2 ms later)
                  master    x_m v_m f_h f_fb
                  operator  phase stroke          (see attrs["phases"])
  /obs/*        policy rate (policy_hz): rgb_top_camera rgb_wrist_camera (uint8)
                and the same proprioceptive/force fields a policy may use
  /action/*     policy rate: x_d (3, world) and k_diag (3, N/m along the paper
                axes u, v and the normal n) -- the APPLIED values at the NEXT
                policy step, i.e. the set-point to reach by the next frame
  /goal/*       strokes_uv (32, 128, 2) + mask, strokes_world, belief frame
  /privileged/* the true paper frame, friction, the ink laid down

THE LABEL IS (x_d, K), APPLIED, NOT REQUESTED.  They coincide while the tank
gate is open (it was, in every episode collected so far: `alpha_min` in the
index), and the verifier below replays them to prove it.  If the gate ever
closes, the requested values are the wrong label and only /full/x_d is right.

Usage:
    python3 collect.py --episodes 200 --workers 8 --out demos/writing_v1
    python3 collect.py --episodes 4 --workers 2 --out demos/smoke --video 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FULL_KEYS = ["t", "qpos", "qvel", "tau", "tcp_pos", "tcp_quat", "tcp_vel", "tcp_angvel",
             "f_contact", "f_raw_mean", "f_n", "pen_down",
             "x_d_req", "x_d", "k_req", "k_diag", "K", "D", "alpha", "E",
             "x_m", "v_m", "f_h", "f_fb", "phase", "stroke", "n_ink", "k_level"]
OBS_KEYS = ["t", "qpos", "qvel", "tau", "tcp_pos", "tcp_quat", "tcp_vel", "tcp_angvel",
            "f_contact", "x_d", "k_diag", "tank_E", "k_level"]
# The discrete stiffness command (xy, z) as 0 = low, 1 = mid, 2 = high, when a
# person sets stiffness by level (protocol.py); -1 when K is continuous.
NO_LEVEL = np.array([-1, -1])


class Recorder:
    """Hooks into teleop.run_synthetic (or the interactive loop) via on_step."""

    def __init__(self, sim, log_hz: float = 100.0, policy_hz: float = 10.0,
                 images: bool = True, video_every: int = 0):
        self.sim = sim
        self.every = max(1, int(round(1.0 / (log_hz * sim.dt))))
        self.p_every = max(1, int(round(1.0 / (policy_hz * sim.dt))))
        self.images = images and sim.cameras
        self.video_every = video_every
        self.full = {k: [] for k in FULL_KEYS}
        self.obs: dict[str, list] = {}
        self.video: list[np.ndarray] = []
        self._f_acc = np.zeros(3)
        self._n_acc = 0

    def on_step(self, i, rec, sess, wr=None) -> None:
        sim = self.sim
        self._f_acc += rec["f_raw"]
        self._n_acc += 1
        if i % self.every == 0:
            q = sim.ctl.tcp.pose.q[0].cpu().numpy()
            row = dict(
                t=rec["t"], qpos=rec["q"], qvel=rec["qd"], tau=rec["tau"],
                tcp_pos=rec["p"], tcp_quat=q, tcp_vel=rec["v"], tcp_angvel=rec["w"],
                f_contact=rec["f_filt"], f_raw_mean=self._f_acc / max(1, self._n_acc),
                f_n=rec["f_n"], pen_down=rec["pen_down"],
                x_d_req=rec["x_d_req"], x_d=rec["x_d"], k_req=rec["k_req"],
                k_diag=sim.k_diag(rec["K"]), K=rec["K"], D=rec["D"],
                alpha=rec["alpha"], E=rec["E"],
                x_m=rec["x_m"], v_m=rec["v_m"], f_h=rec["f_h"], f_fb=rec["f_fb"],
                phase=rec.get("phase", -1), stroke=rec.get("stroke", -1), n_ink=rec["n_ink"],
                k_level=rec.get("k_level", NO_LEVEL))
            for k in FULL_KEYS:
                self.full[k].append(np.asarray(row[k]))
            self._f_acc[:] = 0.0
            self._n_acc = 0
        if i % self.p_every == 0:
            o = sim.observe(images=self.images)
            o["k_level"] = rec.get("k_level", NO_LEVEL)
            for k, v in o.items():
                if k in OBS_KEYS or k.startswith("rgb_"):
                    self.obs.setdefault(k, []).append(np.asarray(v))
            if self.video_every and (i // self.p_every) % self.video_every == 0:
                r = sim.env.render()
                r = r.cpu().numpy() if hasattr(r, "cpu") else np.asarray(r)
                self.video.append(r[0] if r.ndim == 4 else r)

    # ------------------------------------------------------------------ #
    def actions(self) -> dict:
        """Policy-rate labels: the applied (x_d, k_diag) at the NEXT frame.
        The last frame repeats its own set-point."""
        nxt = lambda a: np.r_[a[1:], a[-1:]]
        return {"x_d": nxt(np.asarray(self.obs["x_d"])),
                "k_diag": nxt(np.asarray(self.obs["k_diag"])),
                "k_level": nxt(np.asarray(self.obs["k_level"]))}

    def save(self, path, attrs: dict) -> None:
        import h5py
        with h5py.File(path, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v if isinstance(v, (int, float, str, bool, np.number)) else json.dumps(v)
            g = f.create_group("full")
            for k, v in self.full.items():
                arr = np.asarray(v)
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                elif arr.dtype == np.int64:
                    arr = arr.astype(np.int32)
                g.create_dataset(k, data=arr, compression="gzip", compression_opts=4)
            g = f.create_group("obs")
            for k, v in self.obs.items():
                arr = np.asarray(v)
                if k.startswith("rgb_"):
                    g.create_dataset(k, data=arr, compression="gzip", compression_opts=4,
                                     chunks=(1,) + arr.shape[1:])
                else:
                    g.create_dataset(k, data=arr.astype(np.int32 if k == "k_level" else np.float32))
            g = f.create_group("action")
            for k, v in self.actions().items():
                g.create_dataset(k, data=v.astype(np.int32 if k == "k_level" else np.float32))
            goal = self.sim.goal()
            g = f.create_group("goal")
            for k, v in goal.items():
                if k != "text":
                    g.create_dataset(k, data=np.asarray(v))
            g = f.create_group("privileged")
            for k, v in self.sim.privileged().items():
                g.create_dataset(k, data=np.asarray(v, dtype=np.float64))
            g.create_dataset("ink_uv", data=np.asarray(self.sim.ink_uv, dtype=np.float32).reshape(-1, 2))


# --------------------------------------------------------------------------- #
_SIM = None


def _worker_init(image_size: int, wrist: bool, cameras: bool, video: bool) -> None:
    global _SIM
    import sim as SM
    _SIM = SM.WritingSim(image_size=image_size, wrist_camera=wrist, cameras=cameras,
                         render_mode="rgb_array" if video else None)


def run_one(job) -> dict:
    import sim as SM
    import teleop as T
    seed, out_dir, a = job
    sim = _SIM
    spec = SM.TaskSpec.sample(seed, max_len=a["max_len"], dz=a["dz"], tilt_deg=a["tilt_deg"])
    style = T.WriterStyle.sample(seed)
    mp_ = T.MasterParams()
    want_video = seed < a["video"]
    rec = Recorder(sim, a["log_hz"], a["policy_hz"], images=a["cameras"],
                   video_every=2 if want_video else 0)
    t0 = time.time()
    try:
        res = T.run_synthetic(sim, spec, style, mp_, on_step=rec.on_step)
    except Exception as exc:                           # a diverged episode is a result
        return dict(seed=seed, text=spec.text, ok=False, reason=f"sim failed: {type(exc).__name__}: {exc}")
    full_alpha = np.asarray(rec.full["alpha"])
    row = dict(seed=seed, text=spec.text, ok=bool(res["success"]), reason=res["fail_reason"],
               wall_s=round(time.time() - t0, 1), sim_s=round(res["t"], 2),
               frames=len(rec.obs.get("t", [])),
               alpha_min=float(full_alpha.min()) if len(full_alpha) else 1.0,
               **{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in res.items() if k not in ("checks", "success", "fail_reason")})
    if res["success"] or a["keep_failed"]:
        attrs = dict(text=spec.text, success=bool(res["success"]), source="synthetic",
                     spec=dataclasses.asdict(spec), style=dataclasses.asdict(style),
                     master=dataclasses.asdict(mp_),
                     gains={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in dataclasses.asdict(sim.ctl.g).items()},
                     criteria=dataclasses.asdict(sim.crit), metrics=row,
                     phases=T.PHASES, sim_dt=sim.dt, log_hz=a["log_hz"], policy_hz=a["policy_hz"],
                     writing_frame=sim.W.tolist())
        rec.save(pathlib.Path(out_dir) / f"ep_{seed:05d}.h5", attrs)
    if rec.video:
        import imageio.v2 as imageio
        imageio.mimsave(pathlib.Path(out_dir) / f"ep_{seed:05d}.mp4", rec.video,
                        fps=int(a["policy_hz"] / 2), quality=7, macro_block_size=1)
    return row


# --------------------------------------------------------------------------- #
def replay(sim, path, k_mode: str = "recorded") -> dict:
    """Drive the recorded APPLIED labels (x_d, K) back through the controller
    on the same paper, at the physics rate, interpolating the 100 Hz record.

    k_mode="constant" replays the same x_d with K frozen at its episode mean --
    the check that the stiffness label carries information.  If it did not,
    the two replays would agree.
    """
    import h5py
    import controller as C
    import sim as SM
    with h5py.File(path) as f:
        spec = SM.TaskSpec(**json.loads(f.attrs["spec"]))
        t = f["full/t"][:].astype(float)
        xd = f["full/x_d"][:].astype(float)
        K = f["full/K"][:].astype(float)
        f_demo = f["full/f_n"][:].astype(float)
    sim.reset(spec)
    sim.ctl.x_d = xd[0].copy()
    if k_mode == "constant":
        K = np.repeat(K.mean(axis=0, keepdims=True), len(K), axis=0)
    sim.ctl.K = K[0].copy()
    f_rep = []
    j = 0
    while sim.t < t[-1]:
        tn = sim.t + sim.dt
        while j + 1 < len(t) - 1 and t[j + 1] <= tn:
            j += 1
        w = np.clip((tn - t[j]) / max(t[j + 1] - t[j], 1e-9), 0.0, 1.0)
        x_next = xd[j] + w * (xd[j + 1] - xd[j])
        K_next = K[j] + w * (K[j + 1] - K[j])
        rec = sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt,
                                       (K_next - sim.ctl.K) / sim.dt))
        f_rep.append((rec["t"], rec["f_n"]))
    f_rep = np.array(f_rep)
    f_on_demo = np.interp(t, f_rep[:, 0], f_rep[:, 1])
    res = sim.score()
    res["force_rmse"] = float(np.sqrt(np.mean((f_on_demo - f_demo) ** 2)))
    res["force_mean"] = float(np.mean(f_on_demo[f_demo > 0.8])) if np.any(f_demo > 0.8) else 0.0
    res["force_mean_demo"] = float(np.mean(f_demo[f_demo > 0.8])) if np.any(f_demo > 0.8) else 0.0
    return res


def verify(out_dir: pathlib.Path, n_check: int = 3) -> None:
    import sim as SM
    files = sorted(out_dir.glob("ep_*.h5"))[:n_check]
    if not files:
        return
    sim = SM.WritingSim(cameras=False)
    print(f"\n  replaying {len(files)} episodes from their stored labels (x_d, K):")
    for p in files:
        a = replay(sim, p, "recorded")
        b = replay(sim, p, "constant")
        print(f"    {p.name}  recorded K: success={int(a['success'])} cov {a['coverage']:.2f} "
              f"prec {a['precision']:.2f} force RMSE {a['force_rmse']:.2f} N "
              f"(mean {a['force_mean']:.2f} vs demo {a['force_mean_demo']:.2f})"
              f"  |  K frozen at its mean: success={int(b['success'])} force RMSE {b['force_rmse']:.2f} N"
              f" peak {b['peak_force']:.1f} N {b['fail_reason']}")
    sim.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="demos/writing_v1")
    ap.add_argument("--log-hz", type=float, default=100.0)
    ap.add_argument("--policy-hz", type=float, default=10.0)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--no-cameras", action="store_true", help="proprio/force only, much faster")
    ap.add_argument("--max-len", type=int, default=3, help="characters per word")
    ap.add_argument("--dz", type=float, default=0.004, help="paper height error, +- m")
    ap.add_argument("--tilt-deg", type=float, default=5.0, help="paper tilt, +- deg per axis")
    ap.add_argument("--video", type=int, default=0, help="save an mp4 for the first N seeds")
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = dict(log_hz=args.log_hz, policy_hz=args.policy_hz, cameras=not args.no_cameras,
             max_len=args.max_len, dz=args.dz, tilt_deg=args.tilt_deg,
             video=args.seed_offset + args.video, keep_failed=args.keep_failed)
    jobs = [(args.seed_offset + i, str(out), a) for i in range(args.episodes)]
    print(f"collecting {args.episodes} episodes into {out} on {args.workers} workers")
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(args.image_size, not args.no_wrist, not args.no_cameras,
                            args.video > 0)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(r)
            if i % max(1, args.episodes // 20) == 0 or i == args.episodes:
                print(f"  {i}/{args.episodes}  kept {sum(x['ok'] for x in rows)}")

    rows.sort(key=lambda r: r["seed"])
    kept = [r for r in rows if r["ok"]]
    meta = dict(n_requested=args.episodes, n_kept=len(kept), log_hz=args.log_hz,
                policy_hz=args.policy_hz, image_size=args.image_size,
                cameras=not args.no_cameras, wrist_camera=not args.no_wrist,
                label="action = applied (x_d world, k_diag along paper u, v, n) at the next frame",
                episodes=rows)
    (out / "index.json").write_text(json.dumps(meta, indent=1))

    print("\n" + "=" * 84)
    print(f"kept {len(kept)}/{args.episodes} ({100 * len(kept) / max(1, args.episodes):.0f}%)")
    if kept:
        for f_, lab, sc in (("coverage", "coverage", 100), ("precision", "precision", 100),
                            ("in_band", "force in band", 100), ("peak_force", "peak force N", 1),
                            ("chamfer_mm", "chamfer mm", 1), ("sim_s", "duration s", 1),
                            ("alpha_min", "min gate alpha", 1)):
            v = np.array([r[f_] for r in kept], dtype=float) * sc
            print(f"  {lab:<16} {v.mean():8.2f} +- {v.std():6.2f}   [{v.min():.2f}, {v.max():.2f}]")
    rej = [r for r in rows if not r["ok"]]
    if rej:
        from collections import Counter
        why = Counter(r["reason"] for r in rej)
        print(f"  rejected {len(rej)}: " + ", ".join(f"{k or '?'} x{v}" for k, v in why.most_common()))
    print(f"  index written to {out / 'index.json'}")
    if kept and args.verify:
        verify(out, n_check=min(3, len(kept)))


if __name__ == "__main__":
    main()
