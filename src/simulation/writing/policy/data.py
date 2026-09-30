"""Demonstrations -> training arrays for the force-aware flow policies.

One SAMPLE is one policy frame t (10 Hz) of one episode:

  observation
    top, wrist   (64, 64, 3) uint8   both cameras, 2x2 box-downsampled from 128
    state        (S,)   x_d relative to the believed paper origin, the spring
                        deflection tcp - x_d, tcp velocity, log K (u, v, n), qpos
    goal         (G*2,) the target path resampled to G points, xy, RELATIVE to
                        the current x_d (so the net sees "where to go", not
                        "where the world is")
    ft_hist      (HIST, 3)  contact force, the last HIST frames up to and
                        including t -- the only place the paper's hidden height
                        and tilt show up, exactly as mu does in the toy task
  action chunk   (H, 6)  for j = 0..H-1, the set-point to reach at frame t+1+j:
                        [x_d(t+1+j) - x_d(t)  (3, m),  log k_diag(t+1+j)  (3)]
                        -- STIFFNESS IS PART OF THE ACTION
  future F/T     (H, 3)  contact force at frames t+1 .. t+H, aligned with the chunk

Chunks and histories are clamped at the episode ends (the last value repeats).
Normalization is per (horizon step, dim) for the chunks, per dim for the rest.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib

import h5py
import numpy as np

IMG = 64
H = 10          # action / future-force horizon, frames (1 s at 10 Hz)
HIST = 10       # force history, frames
G = 32          # goal path points
FT = 3          # contact force (the pen tip carries no torque)
A = 6           # x_d delta (3) + log stiffness (3)
S = 19


def _down(img: np.ndarray) -> np.ndarray:
    """(T, 128, 128, 3) -> (T, 64, 64, 3) by 2x2 box average."""
    T, h, w, c = img.shape
    f = h // IMG
    return img.reshape(T, IMG, f, IMG, f, c).mean(axis=(2, 4)).astype(np.uint8)


def goal_points(strokes_world: np.ndarray, mask: np.ndarray, n: int = G) -> np.ndarray:
    """All valid stroke points in writing order, resampled to n points (xy)."""
    pts = np.concatenate([s[m] for s, m in zip(strokes_world, mask) if m.any()], axis=0)[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    q = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])])


def state_vec(x_d, tcp_pos, tcp_vel, k_diag, qpos, origin) -> np.ndarray:
    return np.concatenate([x_d - origin, tcp_pos - x_d, tcp_vel, np.log(k_diag), qpos], axis=-1)


def episode_samples(path) -> dict:
    with h5py.File(path) as f:
        o = f["obs"]
        x_d, tcp, vel = o["x_d"][:].astype(np.float64), o["tcp_pos"][:], o["tcp_vel"][:]
        k, q, ft = o["k_diag"][:], o["qpos"][:], o["f_contact"][:]
        a_x, a_k = f["action/x_d"][:].astype(np.float64), f["action/k_diag"][:]
        k_level = f["action/k_level"][:] if "k_level" in f["action"] else -np.ones((len(x_d), 2), int)
        top, wrist = _down(o["rgb_top_camera"][:]), _down(o["rgb_wrist_camera"][:])
        gp = goal_points(f["goal/strokes_world"][:], f["goal/mask"][:])
        origin = f["goal/belief_origin"][:]
        case = int(f.attrs.get("case", 0))
    T = len(x_d)
    t = np.arange(T)
    fut = np.minimum(t[:, None] + np.arange(H)[None], T - 1)             # action index t+j
    fut_ft = np.minimum(t[:, None] + 1 + np.arange(H)[None], T - 1)      # force at t+1+j
    hist = np.maximum(t[:, None] - HIST + 1 + np.arange(HIST)[None], 0)
    act = np.concatenate([a_x[fut] - x_d[:, None], np.log(a_k[fut])], axis=-1)
    return dict(
        top=top, wrist=wrist,
        state=state_vec(x_d, tcp, vel, k, q, origin).astype(np.float32),
        goal=(gp[None] - x_d[:, None, :2]).reshape(T, -1).astype(np.float32),
        ft_hist=ft[hist].astype(np.float32),
        act=act.astype(np.float32), fut_ft=ft[fut_ft].astype(np.float32),
        k_level=k_level[fut].astype(np.int8), case=np.full(T, case, np.int16))


@dataclasses.dataclass
class Stats:
    mean: dict
    std: dict

    def norm(self, key, x):
        return (x - self.mean[key]) / self.std[key]

    def denorm(self, key, x):
        return x * self.std[key] + self.mean[key]

    def to_json(self):
        return {k: {"mean": np.asarray(self.mean[k]).tolist(), "std": np.asarray(self.std[k]).tolist()}
                for k in self.mean}

    @staticmethod
    def from_json(d):
        return Stats({k: np.asarray(v["mean"], np.float32) for k, v in d.items()},
                     {k: np.asarray(v["std"], np.float32) for k, v in d.items()})


NORM_KEYS = {"state": (0,), "goal": (0,), "ft_hist": (0, 1), "act": (0,), "fut_ft": (0,)}


def compute_stats(d: dict) -> Stats:
    mean, std = {}, {}
    for k, axes in NORM_KEYS.items():
        mean[k] = d[k].mean(axis=axes).astype(np.float32)
        std[k] = np.maximum(d[k].std(axis=axes), 1e-3).astype(np.float32)
    # one scale for all goal points, so relative geometry is not distorted
    mean["goal"][:] = 0.0
    std["goal"][:] = float(np.sqrt(np.mean(d["goal"] ** 2)))
    return Stats(mean, std)


def load(root, val_per_case: int = 5, cache: bool = True) -> tuple[dict, dict, Stats]:
    """All episodes under root/*/ep_*.h5 -> (train, val, stats).  The last
    `val_per_case` episodes of each case are held out for open-loop checks."""
    root = pathlib.Path(root)
    files = sorted(root.glob("*/ep_*.h5"))
    cache_path = root / f"_policy_cache_{IMG}.npz"
    if cache and cache_path.exists() and cache_path.stat().st_mtime > max(p.stat().st_mtime for p in files):
        z = np.load(cache_path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    else:
        parts = []
        for i, p in enumerate(files):
            e = episode_samples(p)
            e["ep"] = np.full(len(e["state"]), i, np.int32)
            parts.append(e)
        d = {k: np.concatenate([e[k] for e in parts]) for k in parts[0]}
        if cache:
            # Atomic: on a cluster several jobs can build the cache at once, and
            # a reader must never see a half-written file.
            tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, **d)
            os.replace(tmp, cache_path)
    val_eps = set()
    for c in np.unique(d["case"]):
        eps = np.unique(d["ep"][d["case"] == c])
        val_eps |= set(eps[-val_per_case:].tolist()) if val_per_case else set()
    is_val = np.isin(d["ep"], list(val_eps))
    train = {k: v[~is_val] for k, v in d.items()}
    val = {k: v[is_val] for k, v in d.items()}
    return train, val, compute_stats(train)


def shift_aug(img: np.ndarray, rng, pad: int = 3) -> np.ndarray:
    """Random translation by up to `pad` pixels, edge-replicated (DrQ-style)."""
    B = len(img)
    p = np.pad(img, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    dx, dy = rng.integers(0, 2 * pad + 1, B), rng.integers(0, 2 * pad + 1, B)
    return np.stack([p[i, dy[i]:dy[i] + IMG, dx[i]:dx[i] + IMG] for i in range(B)])


def features(top, wrist, state, goal, ft_hist, stats: Stats) -> dict:
    """Raw arrays -> normalized network inputs.  Shared by training and the
    closed-loop rollout, so the two cannot drift apart."""
    return dict(top=top.astype(np.float32) / 255.0, wrist=wrist.astype(np.float32) / 255.0,
                state=stats.norm("state", state), goal=stats.norm("goal", goal),
                ft_hist=stats.norm("ft_hist", ft_hist))


def batch(d: dict, stats: Stats, idx: np.ndarray, rng=None) -> dict:
    top, wrist = d["top"][idx], d["wrist"][idx]
    if rng is not None:
        top, wrist = shift_aug(top, rng), shift_aug(wrist, rng)
    b = features(top, wrist, d["state"][idx], d["goal"][idx], d["ft_hist"][idx], stats)
    b.update(action=stats.norm("act", d["act"][idx]), future_ft=stats.norm("fut_ft", d["fut_ft"][idx]))
    return b
