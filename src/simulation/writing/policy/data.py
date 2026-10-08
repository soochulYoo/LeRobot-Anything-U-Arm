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
  action chunk   (H, A)  TWO LAYOUTS, selected by `layout` -- see LAYOUTS below.
                 legacy:  for j = 0..H-1, the set-point to reach at frame t+1+j:
                        [x_d(t+1+j) - x_d(t)  (3, m),  log k_diag(t+1+j)  (3)]
                        -- STIFFNESS IS PART OF THE ACTION -- and, on a dataset
                        that carries one, log K_R(t+1+j) (1), so A = 7.  A flat
                        pad on a board whose normal turns has a ROTATIONAL
                        stiffness to choose too, and the wiping protocol
                        schedules it per phase (../../wiping/CURVED_BOARD.md);
                        a ball-point pen never did, so a writing dataset has no
                        `action/kr` and A stays 6.  The dimension is read from
                        the data, never assumed: `dims()` below.
  future F/T     (H, 3)  contact force at frames t+1 .. t+H, aligned with the chunk

Chunks and histories are clamped at the episode ends (the last value repeats).
Normalization is per (horizon step, dim) for the chunks, per dim for the rest.

LAYOUTS
-------
`layout="legacy"` is the action above: a TARGET pose and a stiffness.  The two
are tied to the force by GIC's spring law F = K (x_d - x), and naming the target
puts the whole stiffness error into the force -- a K that is 2x high presses 2x
harder.  Measured on the scripted writer (`evaluate.py --k-err`), 6 episodes:

    k_err      0.3      1.0      2.0
    legacy    1.26 N   3.14 N   6.02 N     success 0/6, 6/6, 0/6
    spring    3.27 N   3.03 N   3.01 N     success 6/6, 6/6, 5/6

`layout="spring"` is the other side of the same equation:

    [ x_ref(t+1+j) - x_d(t)   (3, m, world)
      f_d(t+1+j)              (3, N, along the paper u, v, n, tool -> env)
      log k_diag(t+1+j)       (3) ]                              A = 9

  x_ref  where the TOOL is expected to be      label: the measured tip pose
  f_d    the wrench to apply there             label: the measured contact
                                                      wrench, negated
Neither label depends on how K was chosen, so both are already in every episode
recorded -- `obs/tcp_pos` and `obs/f_contact` -- and no re-collection is needed.
The controller places the target at x_ref + K^-1 f_d (`evaluate.py`).

Two consequences worth stating, because they are the reason for the layout:

  * THE FUTURE WRENCH IS NOW PART OF THE ACTION.  On the demonstration x = x_ref,
    so F(t+1+j) = f_d(t+1+j) exactly: `fut_ft` and the f_d columns are the SAME
    numbers in two frames (the assertion in `episode_samples`).  Cases b, c and
    d exist to reconcile an action head with a future-force head; this layout
    leaves them nothing to reconcile, which is why case `s` has no force field.
  * K GETS NO GRADIENT FROM IMITATION.  x = x_ref on every training sample makes
    the spring term identically zero there, so nothing in the behaviour-cloning
    loss prefers one K over another.  That is the honest state of the problem --
    K is not identifiable from (x, F) and the legacy layout only appeared to
    learn it because K was absorbing gradient meant for the force.  K has to
    come from a human channel or from `train.py`'s robustness term, which asks
    what the force does when the surface is NOT where x_ref said.
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
FT_SUR = 9      # ft_hist channels once surprise is attached: measured | expected | z
INK_FORCE = 0.8 # N, sim.Criteria.ink_force -- the contact gate for the residual
A = 6           # legacy: x_d delta (3) + log stiffness (3); 7 with log K_R
A_SPRING = 9    # spring: x_ref delta (3) + f_d (3) + log stiffness (3); 10 with log K_R
S = 19          # 20 with log K_R
# "spring_rel" is the repair for spring's 32/32 failure.  Spring anchors x_ref on
# x_d(t) and uses it OPEN LOOP, which protects K's meaning (re-anchoring every
# substep would make the force exact and K idle) but passes the learned x_ref
# error straight into position: coverage halved, 97 -> 64%, while in_band held.
# spring_rel anchors the POSITION on the measured tip and leaves only the spring
# OFFSET open loop:
#     x_d = p_measured + delta_x_ref + K^-1 f_d
# The offset is still open loop, so the force stays K-independent and K still
# governs the response to surface deviation -- the property worth keeping -- while
# position error stops accumulating across a chunk.
LAYOUTS = ("legacy", "spring", "spring_rel")


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


def state_vec(x_d, tcp_pos, tcp_vel, k_diag, qpos, origin, kr=None) -> np.ndarray:
    """The proprioceptive vector.  `kr` is the CURRENT rotational stiffness, on
    a task that has one: a policy that commands K_R has to see where in its own
    schedule it is, exactly as it sees log K for the translational axes."""
    v = [x_d - origin, tcp_pos - x_d, tcp_vel, np.log(k_diag), qpos]
    if kr is not None:
        v.append(np.log(np.asarray(kr, dtype=np.float64))[..., None])
    return np.concatenate(v, axis=-1)


def is_spring(layout: str) -> bool:
    """Does this layout carry f_d (i.e. is it one of the spring-consistent ones)?

    EVERY site that asks that question must come through here.  Three separate
    bugs came from testing `layout == "spring"` after `spring_rel` was added, and
    the worst of them sent spring_rel policies down the LEGACY decode in
    rollout._plan -- `k_diag = exp(a[:, 3:])` is 6 wide for a 9-wide action, so
    parse_action raised "cannot reshape array of size 6 into shape (3,)" and took
    out 32 evaluations.  `data.py:178` and `rollout.py:143` deliberately
    distinguish spring from spring_rel and must NOT use this.
    """
    return layout.startswith("spring")


def dims(root, layout: str = "legacy") -> tuple[int, int]:
    """(A, S) for the dataset under `root`, from one file's keys.

    Read before the model is built, so the action head is the width of the
    labels rather than of a constant that was right for the other task.
    """
    assert layout in LAYOUTS, layout
    files = sorted(pathlib.Path(root).glob("*/ep_*.h5"))
    if not files:
        raise FileNotFoundError(f"no */ep_*.h5 under {root}")
    with h5py.File(files[0]) as f:
        kr = "kr" in f["action"]
    a = A_SPRING if layout.startswith("spring") else A
    return (a + 1, S + 1) if kr else (a, S)


def episode_samples(path, layout: str = "legacy") -> dict:
    assert layout in LAYOUTS, layout
    with h5py.File(path) as f:
        o = f["obs"]
        x_d, tcp, vel = o["x_d"][:].astype(np.float64), o["tcp_pos"][:], o["tcp_vel"][:]
        k, q, ft = o["k_diag"][:], o["qpos"][:], o["f_contact"][:]
        a_x, a_k = f["action/x_d"][:].astype(np.float64), f["action/k_diag"][:]
        a_kr = f["action/kr"][:] if "kr" in f["action"] else None
        kr = o["kr"][:] if "kr" in o else None
        k_level = f["action/k_level"][:] if "k_level" in f["action"] else -np.ones((len(x_d), 2), int)
        top, wrist = _down(o["rgb_top_camera"][:]), _down(o["rgb_wrist_camera"][:])
        # A TASK WITHOUT A GLYPH still has a goal, it just is not a path: flipping a box
        # and opening a door record `goal/belief_origin` and `goal/belief_R` and no
        # strokes.  Zeros keep the channel's width so one `Config` serves every task,
        # and `--no-goal` is what actually removes it from the model -- which is the
        # honest way to train these two, since a constant channel teaches nothing.
        gp = (goal_points(f["goal/strokes_world"][:], f["goal/mask"][:])
              if "strokes_world" in f["goal"] else np.zeros((G, 2)))
        origin = f["goal/belief_origin"][:]
        W = f["goal/belief_R"][:].astype(np.float64)   # the paper's u, v, n
        case = int(f.attrs.get("case", 0))
    T = len(x_d)
    t = np.arange(T)
    fut = np.minimum(t[:, None] + np.arange(H)[None], T - 1)             # action index t+j
    fut_ft = np.minimum(t[:, None] + 1 + np.arange(H)[None], T - 1)      # force at t+1+j
    hist = np.maximum(t[:, None] - HIST + 1 + np.arange(HIST)[None], 0)
    if layout == "legacy":
        cols = [a_x[fut] - x_d[:, None], np.log(a_k[fut])]
    else:
        # Both spring labels are measurements, read off the same episode: where
        # the tip WAS, and the wrench it was applying there (tool -> env, so the
        # negated sensor reading) resolved along the paper's u, v, n.
        nxt = lambda a: np.r_[a[1:], a[-1:]]
        x_ref, f_d = nxt(tcp.astype(np.float64)), -(nxt(ft.astype(np.float64)) @ W)
        # The ANCHOR is the whole difference between the two spring layouts, and
        # it is a LABEL change, not just a decode change: "spring" predicts where
        # the tip will be relative to the current reference x_d(t), "spring_rel"
        # relative to where the tip actually IS.  The second is what lets the
        # rollout re-anchor on measurement without the label meaning something
        # else than it did in training.
        anchor = x_d if layout == "spring" else tcp.astype(np.float64)
        cols = [x_ref[fut] - anchor[:, None], f_d[fut], np.log(a_k[fut])]
    if a_kr is not None:
        cols.append(np.log(a_kr[fut].astype(np.float64))[..., None])
    act = np.concatenate(cols, axis=-1)
    return dict(
        top=top, wrist=wrist,
        state=state_vec(x_d, tcp, vel, k, q, origin, kr).astype(np.float32),
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


def load(root, val_per_case: int = 5, cache: bool = True,
         layout: str = "legacy", surprise=None) -> tuple[dict, dict, Stats]:
    """All episodes under root/*/ep_*.h5 -> (train, val, stats).  The last
    `val_per_case` episodes of each case are held out for open-loop checks.

    The layout is part of the cache name: the two layouts are different labels
    over the same episodes, and sharing one cache between them would silently
    train a 9-wide head on 6-wide targets."""
    assert layout in LAYOUTS, layout
    root = pathlib.Path(root)
    files = sorted(root.glob("*/ep_*.h5"))
    cache_path = root / f"_policy_cache_{IMG}_{layout}.npz"
    if cache and cache_path.exists() and cache_path.stat().st_mtime > max(p.stat().st_mtime for p in files):
        z = np.load(cache_path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
    else:
        parts = []
        for i, p in enumerate(files):
            e = episode_samples(p, layout=layout)
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
    if surprise is not None:
        # attached BEFORE the split, so the train/val normalization sees the
        # widened ft_hist and the two halves are standardized the same way
        attach_surprise(d, surprise)
    is_val = np.isin(d["ep"], list(val_eps))
    train = {k: v[~is_val] for k, v in d.items()}
    val = {k: v[is_val] for k, v in d.items()}
    return train, val, compute_stats(train)


def surprise_channels(ft_h, mu_h, sd_h, standardize: bool = False):
    """[measured | expected | residual] from aligned (HIST, 3) windows.

    THE RESIDUAL IS IN NEWTONS, NOT STANDARDIZED, and that is the fix for the
    -24 pp result.  Dividing by sigma is what made the channel unusable:
    measured between the fold models (which produced the training targets) and
    the deploy model (which serves the rollout),

        RMS(mu_deploy - mu_fold)  =  0.52-0.77 sigma
        RMS(z_deploy  - z_fold)   =  2.00 on the normal axis

    against a training-time |z| of 0.66.  The ratio amplifies: sigma disagrees
    between the two predictors as well as mu, and wherever sigma is small the
    quotient explodes.  In newtons the same disagreement is 0.05-0.19 N on a
    signal of several N -- a perturbation instead of a different variable.

    `standardize=True` restores the old channel for reproducing that result.

    THE ONE PLACE the nine channels are assembled, called by both
    `attach_surprise` (offline, cross-fitted) and policy/rollout.py (online,
    from the deploy expectation model).  They must not drift: a policy trained
    on one convention and rolled out on another is being fed a different input
    than it was fit to, and nothing would raise.

    PHASE GATE.  Out of contact the wrench is ~0, mu is ~0 and sigma is small,
    so z is whatever ratio the noise makes -- a large number meaning nothing.
    z is zeroed below the ink threshold, the same contact test the simulator
    scores with.  mu is NOT gated: "I expect to feel nothing yet" is a real
    statement about the approach, and the camera is what can make it.
    """
    ft_h = np.asarray(ft_h, np.float32)
    res = (ft_h - mu_h) / np.maximum(sd_h, 1e-3) if standardize else (ft_h - mu_h)
    gate = (np.linalg.norm(ft_h, axis=-1, keepdims=True) > INK_FORCE).astype(np.float32)
    return np.concatenate([ft_h, mu_h, res * gate], -1).astype(np.float32)


def attach_surprise(d: dict, path) -> dict:
    """ft_hist (HIST, 3) -> (HIST, 9) = [measured | expected | z], in place.

    `path` is a `_surprise_*.npz` from policy/surprise.py: out-of-fold (mu,
    sigma) for every sample, so z is the part of the wrench the scene did NOT
    predict on an episode the expectation model never saw.

    PHASE GATE.  Out of contact the wrench is zero, mu is near zero and sigma is
    small, so z is whatever ratio the noise happens to make -- a large number
    that means nothing.  z is zeroed wherever the measured wrench is below the
    ink threshold, which is the same contact test the simulator scores with.
    The expectation mu is NOT gated: "I expect to feel nothing yet" is a real
    statement about the approach, and it is the one the camera can make.
    """
    z = np.load(path)
    mu, sd = z["mu"].astype(np.float32), z["sigma"].astype(np.float32)
    if len(mu) != len(d["ep"]):
        raise ValueError(f"{path}: {len(mu)} rows for {len(d['ep'])} samples -- "
                         "it was built from a different dataset or IMG")
    if not np.array_equal(z["ep"], d["ep"]):
        raise ValueError(f"{path}: episode ids do not line up with the samples; "
                         "rebuild it with the same --data")
    # Per-frame (mu, sigma) -> the same HIST window ft_hist already uses.  Each
    # episode's samples are one contiguous, frame-ordered block (see `load`), so
    # the window is clamped at the episode's own start, never the previous one's.
    N = len(d["ep"])
    hist_idx = np.empty((N, HIST), np.int64)
    for e in np.unique(d["ep"]):
        g = np.flatnonzero(d["ep"] == e)
        r = np.arange(len(g))
        hist_idx[g] = g[np.maximum(r[:, None] - HIST + 1 + np.arange(HIST)[None], 0)]
    d["ft_hist"] = surprise_channels(d["ft_hist"], mu[hist_idx], sd[hist_idx])
    return d


def shift_aug(img: np.ndarray, rng, pad: int = 3) -> np.ndarray:
    """Random translation by up to `pad` pixels, edge-replicated (DrQ-style)."""
    B = len(img)
    p = np.pad(img, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    dx, dy = rng.integers(0, 2 * pad + 1, B), rng.integers(0, 2 * pad + 1, B)
    return np.stack([p[i, dy[i]:dy[i] + IMG, dx[i]:dx[i] + IMG] for i in range(B)])


def features(top, wrist, state, goal, ft_hist, stats: Stats, force: bool = True) -> dict:
    """Raw arrays -> normalized network inputs.  Shared by training and the
    closed-loop rollout, so the two cannot drift apart.

    `force=False` is the CONTROL for the whole eight-variant comparison.  The
    ranking in policy/runs/COMPARISON.md (flow a 93.8% > b 86.2 > d 70.4 > c 52.1)
    tracks pos1_mm (0.35 -> 0.45) and the action loss (0.086 -> 0.131) while K
    error is 0.0-0.7% for every variant -- so it may be measuring OPTIMISATION
    DIFFICULTY rather than how well each structure uses force.  Zeroing the force
    input decides it: if the ranking survives with no force to condition on, the
    four structures were never being separated by their force handling.
    Zeroed AFTER normalization, so "no force" is the mean of the training force
    rather than an out-of-distribution constant, and done HERE so the rollout
    cannot accidentally feed real force to a force-blind policy.
    """
    ft = stats.norm("ft_hist", ft_hist)
    return dict(top=top.astype(np.float32) / 255.0, wrist=wrist.astype(np.float32) / 255.0,
                state=stats.norm("state", state), goal=stats.norm("goal", goal),
                ft_hist=ft if force else np.zeros_like(ft))


def batch(d: dict, stats: Stats, idx: np.ndarray, rng=None,
          force: bool = True, force_target: bool = True) -> dict:
    """`force` removes the force INPUT, `force_target` the future-force
    SUPERVISION.  They are separate because they ablate different things:
    without the input, b/c/d still get a force auxiliary task; without the
    target as well, their extra machinery has nothing to predict and the
    comparison is purely architectural.  Run both to tell those apart."""
    top, wrist = d["top"][idx], d["wrist"][idx]
    if rng is not None:
        top, wrist = shift_aug(top, rng), shift_aug(wrist, rng)
    b = features(top, wrist, d["state"][idx], d["goal"][idx], d["ft_hist"][idx], stats,
                 force=force)
    fut = stats.norm("fut_ft", d["fut_ft"][idx])
    b.update(action=stats.norm("act", d["act"][idx]),
             future_ft=fut if force_target else np.zeros_like(fut))
    return b
