"""Force-aware flow policies (CoFA, Fig. 1) for compliant writing, in Flax NNX.

The four structures of the toy script, unchanged in how the flows couple; what
changes is what flows:

    action    (H, 6)  x_d offsets (3) + log stiffness (3) -- a COMPLIANCE action
    future    (H, 3)  the contact force the chunk will produce

  a  ft_input    current force only an input to the action flow
  b  uni_dir     a future-force predictor feeds the action flow
  c  unified     one flow over [action | future force], one time
  d  cross_cond  separate action and future-force flows, independent times; at
                 every Euler step each field sees the other's current state

(d) is the one this package is built around.  Its action field sees a force
trajectory that is itself conditioned on the action being generated, so the
stiffness and the set-point are chosen against the force they will produce --
which is what compliance control is.

Rectified flow everywhere: x_tau = (1 - tau) noise + tau data, target velocity
data - noise, Euler from tau = 0 to 1.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

import data as DATA


@dataclasses.dataclass
class Config:
    case: str = "d"
    H: int = DATA.H
    A: int = DATA.A
    FT: int = DATA.FT
    HIST: int = DATA.HIST
    S: int = DATA.S
    G: int = DATA.G
    D: int = 256
    n_steps: int = 10        # Euler steps at inference
    w_f: float = 0.5         # future-force loss weight (CoFA uses 0.5)
    vision: bool = True
    goal: bool = True


# ------------------------------------------------------------------ blocks
def time_emb(t, dim=32):
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), dim // 2))
    ang = t[:, None] * freqs[None]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], -1)


def flat(x):
    return x.reshape(x.shape[0], -1)


def mse(a, b):
    return jnp.mean((a - b) ** 2)


def noisy(key, x1, tau):
    """Point on the straight path and its target velocity."""
    x0 = jax.random.normal(key, x1.shape)
    t = tau[:, None, None]
    return (1 - t) * x0 + t * x1, x1 - x0


def full(b, value):
    return jnp.full((b,), value)


class MLP(nnx.Module):
    def __init__(self, din, dout, D, rngs):
        self.l1 = nnx.Linear(din, D, rngs=rngs)
        self.l2 = nnx.Linear(D, D, rngs=rngs)
        self.l3 = nnx.Linear(D, dout, rngs=rngs)

    def __call__(self, x):
        return self.l3(nnx.gelu(self.l2(nnx.gelu(self.l1(x)))))


class CNN(nnx.Module):
    """64x64 RGB -> D.  Small on purpose: three strided convs train at ~25 ms a
    step on CPU with two cameras."""

    def __init__(self, D, rngs):
        self.c1 = nnx.Conv(3, 16, (8, 8), strides=(4, 4), rngs=rngs)
        self.c2 = nnx.Conv(16, 32, (4, 4), strides=(2, 2), rngs=rngs)
        self.c3 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), rngs=rngs)
        self.out = nnx.Linear(4 * 4 * 64, D, rngs=rngs)

    def __call__(self, x):
        for c in (self.c1, self.c2, self.c3):
            x = nnx.gelu(c(x))
        return self.out(flat(x))


class ObsEncoder(nnx.Module):
    """h: cameras + state + goal.  f: current force feature (GRU over history)."""

    def __init__(self, cfg: Config, rngs):
        D = cfg.D
        self.cfg = cfg
        if cfg.vision:
            self.top = CNN(D, rngs)
            self.wrist = CNN(D, rngs)
        self.state = nnx.Linear(cfg.S, D, rngs=rngs)
        if cfg.goal:
            self.goal = nnx.Linear(2 * cfg.G, D, rngs=rngs)
        self.mix = nnx.Linear(D, D, rngs=rngs)
        self.gru = nnx.RNN(nnx.GRUCell(cfg.FT, D, rngs=rngs))

    def __call__(self, b):
        h = self.state(b["state"])
        if self.cfg.vision:
            h = h + self.top(b["top"]) + self.wrist(b["wrist"])
        if self.cfg.goal:
            h = h + self.goal(b["goal"])
        h = self.mix(nnx.gelu(h))
        f = self.gru(b["ft_hist"])[:, -1]
        return jnp.concatenate([h, f], -1)          # (B, 2D)


class Field(nnx.Module):
    """Velocity field v(x_tau, times | cond).  x: (B, T, dim)."""

    def __init__(self, x_shape, cond_dim, n_times, D, rngs):
        size = x_shape[0] * x_shape[1]
        self.net = MLP(size + cond_dim + 32 * n_times, size, D, rngs)

    def __call__(self, x, times, cond):
        t = jnp.concatenate([time_emb(ti) for ti in times], -1)
        return self.net(jnp.concatenate([flat(x), cond, t], -1)).reshape(x.shape)


# ------------------------------------------------------------------ (a)
class FTInput(nnx.Module):
    """Current force is an input.  No future-force prediction."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.enc = ObsEncoder(cfg, rngs)
        self.vA = Field((cfg.H, cfg.A), 2 * cfg.D, 1, cfg.D, rngs)

    def loss(self, b, key):
        c = self.enc(b)
        k1, k2 = jax.random.split(key)
        tau = jax.random.uniform(k1, (len(c),))
        a_t, u = noisy(k2, b["action"], tau)
        return mse(self.vA(a_t, [tau], c), u), {}

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        a = jax.random.normal(key, (len(c), cfg.H, cfg.A))
        for k in range(cfg.n_steps):
            a = a + self.vA(a, [full(len(c), k / cfg.n_steps)], c) / cfg.n_steps
        return a, None


# ------------------------------------------------------------------ (b)
class UniDir(nnx.Module):
    """Force predictor (from the observation only) -> policy."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.enc = ObsEncoder(cfg, rngs)
        self.pred = MLP(2 * cfg.D, cfg.H * cfg.FT, cfg.D, rngs)
        self.vA = Field((cfg.H, cfg.A), 2 * cfg.D + cfg.H * cfg.FT, 1, cfg.D, rngs)

    def predict_ft(self, c):
        return self.pred(c).reshape(len(c), self.cfg.H, self.cfg.FT)

    def loss(self, b, key):
        c = self.enc(b)
        z = self.predict_ft(c)
        l_f = mse(z, b["future_ft"])
        cond = jnp.concatenate([c, flat(jax.lax.stop_gradient(z))], -1)
        k1, k2 = jax.random.split(key)
        tau = jax.random.uniform(k1, (len(c),))
        a_t, u = noisy(k2, b["action"], tau)
        l_a = mse(self.vA(a_t, [tau], cond), u)
        return l_a + self.cfg.w_f * l_f, {"l_a": l_a, "l_f": l_f}

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        z = self.predict_ft(c)
        cond = jnp.concatenate([c, flat(z)], -1)
        a = jax.random.normal(key, (len(c), cfg.H, cfg.A))
        for k in range(cfg.n_steps):
            a = a + self.vA(a, [full(len(c), k / cfg.n_steps)], cond) / cfg.n_steps
        return a, z


# ------------------------------------------------------------------ (c)
class Unified(nnx.Module):
    """One field, one time, over [action | future force]."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.enc = ObsEncoder(cfg, rngs)
        self.v = Field((cfg.H, cfg.A + cfg.FT), 2 * cfg.D, 1, cfg.D, rngs)

    def loss(self, b, key):
        A = self.cfg.A
        c = self.enc(b)
        x1 = jnp.concatenate([b["action"], b["future_ft"]], -1)
        k1, k2 = jax.random.split(key)
        tau = jax.random.uniform(k1, (len(c),))
        x_t, u = noisy(k2, x1, tau)
        v = self.v(x_t, [tau], c)
        l_a, l_f = mse(v[..., :A], u[..., :A]), mse(v[..., A:], u[..., A:])
        return l_a + self.cfg.w_f * l_f, {"l_a": l_a, "l_f": l_f}

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        x = jax.random.normal(key, (len(c), cfg.H, cfg.A + cfg.FT))
        for k in range(cfg.n_steps):
            x = x + self.v(x, [full(len(c), k / cfg.n_steps)], c) / cfg.n_steps
        return x[..., :cfg.A], x[..., cfg.A:]


# ------------------------------------------------------------------ (d)
class CrossCond(nnx.Module):
    """Separate fields; each sees the other's evolving state and flow time.

        vF = vF(Z_lamF, lamF | c, A_lamA, lamA)   action-conditioned future force
        vA = vA(A_lamA, lamA | c, Z_lamF, lamF)   force-guided compliance action

    Independent flow times in training make each field robust to the other
    being anywhere between noise and data; at inference both run on one clock.
    CoFA extras omitted, as in the toy: the EMA-encoder latent as the force
    target, and the force-conditioned mixture-of-experts action field.
    """

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        H, A, FT, D = cfg.H, cfg.A, cfg.FT, cfg.D
        self.enc = ObsEncoder(cfg, rngs)
        self.vA = Field((H, A), 2 * D + H * FT, 2, D, rngs)
        self.vF = Field((H, FT), 2 * D + H * A, 2, D, rngs)

    def loss(self, b, key):
        c = self.enc(b)
        kA, kF, ka, kf = jax.random.split(key, 4)
        lamA = jax.random.uniform(kA, (len(c),))
        lamF = jax.random.uniform(kF, (len(c),))
        a_t, uA = noisy(ka, b["action"], lamA)
        z_t, uF = noisy(kf, b["future_ft"], lamF)
        vA = self.vA(a_t, [lamA, lamF], jnp.concatenate([c, flat(z_t)], -1))
        vF = self.vF(z_t, [lamF, lamA], jnp.concatenate([c, flat(a_t)], -1))
        l_a, l_f = mse(vA, uA), mse(vF, uF)
        return l_a + self.cfg.w_f * l_f, {"l_a": l_a, "l_f": l_f}

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        kA, kF = jax.random.split(key)
        a = jax.random.normal(kA, (len(c), cfg.H, cfg.A))
        z = jax.random.normal(kF, (len(c), cfg.H, cfg.FT))
        for k in range(cfg.n_steps):
            t = full(len(c), k / cfg.n_steps)
            vA = self.vA(a, [t, t], jnp.concatenate([c, flat(z)], -1))
            vF = self.vF(z, [t, t], jnp.concatenate([c, flat(a)], -1))
            a, z = a + vA / cfg.n_steps, z + vF / cfg.n_steps     # synchronous
        return a, z


CASES = {"a": FTInput, "b": UniDir, "c": Unified, "d": CrossCond}
NAMES = {"a": "ft_input", "b": "uni_dir", "c": "unified", "d": "cross_cond"}


def build(cfg: Config, seed: int = 0):
    return CASES[cfg.case](cfg, nnx.Rngs(seed))
