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
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import data as DATA


@dataclasses.dataclass
class Config:
    case: str = "d"
    H: int = DATA.H
    A: int = DATA.A
    FT: int = DATA.FT        # the future-force OUTPUT width
    # The force INPUT width, which is no longer the same number: with
    # --surprise, ft_hist carries [measured | expected | z] = 9 channels, and
    # only the first 3 of them are a wrench the policy could also be asked to
    # predict.  See policy/surprise.py and data.attach_surprise.
    ft_in: int = DATA.FT
    HIST: int = DATA.HIST
    S: int = DATA.S
    G: int = DATA.G
    D: int = 256
    n_steps: int = 10        # Euler steps at inference
    w_f: float = 0.5         # future-force loss weight (CoFA uses 0.5)
    vision: bool = True
    goal: bool = True        # the 32 target-path points: the TASK COMMAND (which
                             # glyph, where), not the answer.  Without it the task
                             # is under-resolved rather than merely harder -- the
                             # glyph spans 8x8 px of the 64x64 input, so the 3 mm
                             # scoring tolerance is 0.67 px and every structure
                             # scores ~0% (see policy/runs_nogoal).  --no-goal
                             # reproduces that ablation.

    # arch picks the policy family that the four structures are built on:
    #   "flow"  rectified-flow fields (CoFA), the rest of this file
    #   "act"   Comp-ACT's CVAE transformer, act_model.py
    arch: str = "flow"
    hidden: int = 512        # Comp-ACT's published sizes, unchanged
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    ff: int = 3200
    latent: int = 32         # CVAE latent, as in ACT
    kl_weight: float = 100.0 # Comp-ACT's value
    # how force reaches the policy, beyond the four CoFA structures:
    film: bool = False       # (e) adaLN-Zero: the wrench scales and shifts the
                             #     decoder's features -- gain scheduling K(F)
    ft_tokens: bool = False  # the measured history enters as HIST tokens with
                             #     frame positions, not one GRU summary, so the
                             #     action queries can re-read a moment of it
    n_refine: int = 4        # (d) mutual-refinement passes, the ACT analogue
                             # of the flow's integration steps

    # ---- (d)'s TRAIN/INFERENCE MISMATCH, and the fix -------------------
    # (d) trains with INDEPENDENT flow times lamA, lamF drawn uniformly, so each
    # field learns to read the other anywhere between noise and data.  It then
    # samples with lamA = lamF at every step -- the DIAGONAL of a square the
    # training distribution covers uniformly, i.e. a measure-zero slice.  The
    # model is asked at inference for the one configuration it almost never saw.
    #
    # Diffusion Forcing (arXiv:2407.01392), which is where the independent
    # per-token noise levels come from, samples on a 2-D schedule matrix and
    # explicitly AVOIDS a shared clock: its "zig-zag" keeps some tokens fully
    # denoised while others stay noisy.  It also confirms uniform independent
    # sampling at training is the right choice (which (d) already does), so the
    # mismatch is entirely on the sampling side.
    #
    # `d_lead` is that schedule for two modalities: the WRENCH field runs
    # `d_lead` steps ahead, so the action field always reads a force estimate
    # that is cleaner than itself -- which is the direction CoFA's figure
    # intends ("the stiffness and the set-point are chosen against the force
    # they will produce").  d_lead = 0 reproduces the synchronous clock exactly.
    d_lead: int = 0

    # The two pieces of the CoFA figure the first version left out:
    moe: int = 2             # softly mixed experts in the action field (CoFA: 2)
    ft_latent: int = 16      # per-frame width of the future-F/T latent the wrench
                             # flow runs on (0 = flow over raw newtons)
    ema: float = 0.99        # decay of the target encoder that supplies that latent
    backbone: str = "cnn"    # or "resnet18", as drawn in the figure
    pretrained: bool = True  # ImageNet weights for the resnet18 backbone, as
                             # Comp-ACT uses -- its lr_backbone 1e-5 is a
                             # fine-tuning rate and means little from scratch

    # ---- the action layout, ORTHOGONAL to the four structures ----
    # "legacy" (A = 6): the policy names the target pose and the stiffness.
    # "spring" (A = 9): it names where the tool will be, the wrench to apply
    #     there and the stiffness, and the CONTROLLER places the target at
    #     x_ref + K^-1 f_d (data.py LAYOUTS, evaluate.py parse_action).
    # Crossing the two axes is the point: under "spring" the future wrench is
    # already inside the action, so b, c and d have nothing left to reconcile
    # and the grid measures whether their extra machinery was buying anything
    # or only repairing a layout.
    layout: str = "legacy"

    # ---- the control for the eight-variant ranking (data.features) ----
    # force=False zeroes the force INPUT, force_target=False the future-force
    # SUPERVISION.  Stored on the config so load_checkpoint rebuilds a
    # force-blind policy force-blind, and rollout.py reads it when it builds
    # features -- otherwise a blind policy would be fed real force at test time
    # and the control would silently not be a control.
    force: bool = True
    force_target: bool = True

    # ---- K's only gradient, and why it needs one -------------------------
    # Under "spring" the labels put x = x_ref on every training sample, so the
    # spring term K (x_ref - x) is identically zero there and imitation cannot
    # prefer one stiffness over another.  That is not a defect of the layout --
    # it is the identifiability problem stated honestly: K is not in (x, F).
    # The legacy layout only LOOKED like it learned K because K was absorbing
    # gradient that belonged to the force.
    #
    # So K is given a gradient from the one thing that is known without any
    # human label: the surface is not where x_ref said it was.  Both terms are
    # the sizing rules, not a regulariser, and they pull in OPPOSITE directions:
    #
    #   soft enough   |f_d| +/- k_n delta must stay inside the force band
    #                 -> k_n <~ (f_mag - F_lo) / delta
    #   stiff enough  friction mu |f_d| must not drag the tip off the line by
    #                 more than the scoring tolerance
    #                 -> k_t >~ mu f_mag / tol
    #
    # Calibration, which is the reason to believe them: at f_mag = 3 N,
    # band (1, 6) N, delta = 2 mm, mu = 0.5, tol = 3 mm they give
    # k_n <~ 1000 N/m and k_t >~ 500 N/m -- which are exactly the MID and LOW
    # levels the demonstration protocol uses (tests.py).  The term explains the
    # demonstrated stiffnesses rather than fighting them.
    #
    # delta is a REQUIREMENT, not an estimate: "keep the force in band when the
    # surface is up to delta from where the policy expected it".  2 mm is half
    # the +/- 4 mm canvas height error the task randomises.  Sweep it.
    w_robust: float = 0.0        # 0 disables; spring layout only
    robust_delta: float = 0.002  # m, surface deviation to stay in band under
    robust_mu: float = 0.5       # the top of the task's friction range
    robust_tol: float = 0.003    # m, sim.Criteria.tol
    robust_band: tuple = (1.0, 6.0)   # N, sim.Criteria.force_band
    robust_ink: float = 0.8      # N, below this the chunk frame is not in contact


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


def robust_loss(a_phys, a_label_phys, cfg: Config):
    """The sizing rules of Config.w_robust, in physical units.

    `a_phys` is the model's own action estimate and `a_label_phys` the label,
    used only for the contact mask so the gate cannot be gamed by predicting
    no contact.  Both terms are made dimensionless before they are added, so
    `w_robust` is one number rather than a units conversion.
    """
    lo, hi = cfg.robust_band
    f_d, k = a_phys[..., 3:6], jnp.exp(a_phys[..., 6:9])
    f_mag = -f_d[..., 2]                       # pressing along -n is positive
    k_t, k_n = k[..., 0], k[..., 2]
    swing = k_n * cfg.robust_delta             # N the surface deviation adds
    over = jax.nn.relu(f_mag + swing - hi)     # too stiff: the band's top
    under = jax.nn.relu(lo - (f_mag - swing))  # too stiff: loses contact
    drift = jax.nn.relu(cfg.robust_mu * f_mag / k_t - cfg.robust_tol)  # too soft
    mask = (-a_label_phys[..., 5] > cfg.robust_ink).astype(a_phys.dtype)
    span = max(hi - lo, 1e-6)
    per = (over / span) ** 2 + (under / span) ** 2 + (drift / cfg.robust_tol) ** 2
    return jnp.sum(mask * per) / jnp.maximum(jnp.sum(mask), 1.0)


def robust_term(model, a_hat, b):
    """`w_robust` x robust_loss, from the NORMALIZED one-step action estimate.

    A rectified-flow field predicts x1 - x0, so the data point it implies at
    flow time tau is a_tau + (1 - tau) v -- the model's own action, available
    without integrating, and differentiable.  Returns (term, dict) so the parts
    show up in the training log.
    """
    cfg = model.cfg
    if not DATA.is_spring(cfg.layout) or cfg.w_robust <= 0.0:
        return 0.0, {}
    m, sd = model.act_mean, model.act_std
    l_k = robust_loss(a_hat * sd + m, b["action"] * sd + m, cfg)
    return cfg.w_robust * l_k, {"l_k": l_k}


def one_step(a_t, tau, v):
    """The data point a rectified-flow velocity implies at flow time tau."""
    return a_t + (1.0 - tau)[:, None, None] * v


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


class Frozen(nnx.Variable):
    """Buffers the optimizer must not touch: ImageNet batch-norm statistics,
    folded into a per-channel affine.  Comp-ACT's FrozenBatchNorm2d, which keeps
    the statistics fixed while the convolutions fine-tune."""


class FrozenBN(nnx.Module):
    def __init__(self, c, scale=None, shift=None):
        self.scale = Frozen(jnp.ones((c,)) if scale is None else jnp.asarray(scale))
        self.shift = Frozen(jnp.zeros((c,)) if shift is None else jnp.asarray(shift))

    def __call__(self, x):
        return x * self.scale.value + self.shift.value


IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])
WEIGHTS = pathlib.Path(__file__).resolve().parent / "resnet18_imagenet.npz"


def imagenet_weights():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"{WEIGHTS} is missing -- rebuild it with policy/convert_resnet18.py, "
            "or pass --no-pretrained to train from scratch")
    return dict(np.load(WEIGHTS))


class ResBlock(nnx.Module):
    def __init__(self, cin, cout, stride, rngs, w=None, i=None):
        pre = w is not None
        P3 = ((1, 1), (1, 1))          # torch's padding=1 for a 3x3, any stride
        self.c1 = nnx.Conv(cin, cout, (3, 3), strides=(stride, stride), padding=P3,
                           use_bias=not pre, rngs=rngs)
        self.c2 = nnx.Conv(cout, cout, (3, 3), padding=P3, use_bias=not pre, rngs=rngs)
        self.skip = (nnx.Conv(cin, cout, (1, 1), strides=(stride, stride), padding="VALID",
                              use_bias=not pre, rngs=rngs)
                     if stride != 1 or cin != cout else None)
        if pre:
            self.c1.kernel.value = jnp.asarray(w[f"b{i}_c1"])
            self.c2.kernel.value = jnp.asarray(w[f"b{i}_c2"])
            self.n1 = FrozenBN(cout, w[f"b{i}_s1"], w[f"b{i}_h1"])
            self.n2 = FrozenBN(cout, w[f"b{i}_s2"], w[f"b{i}_h2"])
            if self.skip is not None:
                self.skip.kernel.value = jnp.asarray(w[f"b{i}_sk"])
                self.ns = FrozenBN(cout, w[f"b{i}_sks"], w[f"b{i}_skh"])
        else:
            self.n1 = nnx.GroupNorm(cout, num_groups=8, rngs=rngs)
            self.n2 = nnx.GroupNorm(cout, num_groups=8, rngs=rngs)
            self.ns = None
        if not pre or self.skip is None:
            self.ns = getattr(self, "ns", None)

    def __call__(self, x):
        y = nnx.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        if self.skip is not None:
            sk = self.skip(x)
            x = self.ns(sk) if self.ns is not None else sk
        return nnx.relu(y + x)


class ResNet18(nnx.Module):
    """The figure's camera encoder.  Group norm rather than batch norm: batches
    here are small and the flow already sees each frame at a random flow time,
    so batch statistics would be one more thing moving under training."""

    def __init__(self, D, rngs, width=64, pretrained=False):
        w = width
        p = imagenet_weights() if pretrained else None
        self.pretrained = pretrained
        self.stem = nnx.Conv(3, w, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)),
                             use_bias=not pretrained, rngs=rngs)
        if pretrained:
            self.stem.kernel.value = jnp.asarray(p["stem_kernel"])
            self.norm = FrozenBN(w, p["stem_scale"], p["stem_shift"])
        else:
            self.norm = nnx.GroupNorm(w, num_groups=8, rngs=rngs)
        spec = [(w, w, 1), (w, w, 1), (w, 2 * w, 2), (2 * w, 2 * w, 1),
                (2 * w, 4 * w, 2), (4 * w, 4 * w, 1), (4 * w, 8 * w, 2), (8 * w, 8 * w, 1)]
        self.stages = nnx.List([ResBlock(a, b, st, rngs, p, i)
                                for i, (a, b, st) in enumerate(spec)])
        self.out = nnx.Linear(8 * w, D, rngs=rngs)

    def normalize(self, x):
        """ImageNet statistics, as Comp-ACT applies before its backbone."""
        return (x - IMAGENET_MEAN) / IMAGENET_STD if self.pretrained else x

    def features(self, x):
        """The last feature map, which ACT flattens into encoder tokens."""
        x = nnx.relu(self.norm(self.stem(self.normalize(x))))
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        for b in self.stages:
            x = b(x)
        return x

    def __call__(self, x):
        return self.out(jnp.mean(self.features(x), axis=(1, 2)))   # global average pool


class ObsEncoder(nnx.Module):
    """h: cameras + state + goal.  f: current force feature (GRU over history)."""

    def __init__(self, cfg: Config, rngs):
        D = cfg.D
        self.cfg = cfg
        if cfg.vision:
            cam = ((lambda: ResNet18(D, rngs, pretrained=cfg.pretrained))
                   if cfg.backbone == "resnet18" else (lambda: CNN(D, rngs)))
            self.top, self.wrist = cam(), cam()
        self.state = nnx.Linear(cfg.S, D, rngs=rngs)
        if cfg.goal:
            self.goal = nnx.Linear(2 * cfg.G, D, rngs=rngs)
        self.mix = nnx.Linear(D, D, rngs=rngs)
        self.gru = nnx.RNN(nnx.GRUCell(cfg.ft_in, D, rngs=rngs))

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

    def __call__(self, x, times, cond, route=None):
        t = jnp.concatenate([time_emb(ti) for ti in times], -1)
        return self.net(jnp.concatenate([flat(x), cond, t], -1)).reshape(x.shape)


class MoEField(nnx.Module):
    """The figure's *Future F/T Conditioned Action MoE*: a router reads the same
    input the experts do -- the wrench state Z, the current-force feature, the
    rest of the condition and the action state -- and mixes their velocities.

    Compliance is regime-switched: approaching in free space, landing, and
    writing in contact want different gains from the same observation, and one
    MLP has to average them.  A router lets the field keep them apart.
    """

    def __init__(self, x_shape, cond_dim, n_times, D, n_experts, rngs, route_dim=None):
        size = x_shape[0] * x_shape[1]
        din = size + cond_dim + 32 * n_times
        self.experts = nnx.List([MLP(din, size, D, rngs) for _ in range(n_experts)])
        # CoFA routes on the measured F/T feature, the future-F/T latent and both
        # flow times -- not on the action state or the visual context.
        self.router = nnx.Linear((route_dim or din - size - 32 * n_times) + 32 * n_times,
                                 n_experts, rngs=rngs)

    def __call__(self, x, times, cond, route=None):
        t = jnp.concatenate([time_emb(ti) for ti in times], -1)
        inp = jnp.concatenate([flat(x), cond, t], -1)
        w = nnx.softmax(self.router(jnp.concatenate([route, t], -1)), -1)   # (B, M)
        v = jnp.stack([e(inp) for e in self.experts], 1)                    # (B, M, size)
        return jnp.einsum("bm,bms->bs", w, v).reshape(x.shape)


def make_field(x_shape, cond_dim, n_times, cfg: Config, rngs, moe=True, route_dim=None):
    if moe and cfg.moe > 0:
        return MoEField(x_shape, cond_dim, n_times, cfg.D, cfg.moe, rngs, route_dim)
    return Field(x_shape, cond_dim, n_times, cfg.D, rngs)


class FTCodec(nnx.Module):
    """The figure's *Future F/T latent*.  The wrench flow runs on a latent, not on
    newtons, and its target comes from an EMA copy of the encoder -- a moving
    target that the flow cannot collapse by shrinking the encoder.  A decoder
    brings a sampled latent back to newtons so the rollout can still be scored.
    """

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        n_in, n_out = cfg.H * cfg.FT, cfg.H * cfg.ft_latent
        self.enc = MLP(n_in, n_out, cfg.D, rngs)
        self.target = MLP(n_in, n_out, cfg.D, rngs)     # EMA copy, never gets gradients
        self.dec = MLP(n_out, n_in, cfg.D, rngs)

    def shape(self):
        return (self.cfg.H, self.cfg.ft_latent)

    def encode_target(self, ft):
        """The flow's data: stop-grad through the EMA encoder."""
        z = jax.lax.stop_gradient(self.target(flat(ft)))
        return z.reshape(len(ft), *self.shape())

    def decode(self, z):
        return self.dec(flat(z)).reshape(len(z), self.cfg.H, self.cfg.FT)

    def recon(self, ft):
        z = self.enc(flat(ft)).reshape(len(ft), *self.shape())
        return mse(self.decode(z), ft)


def ema_update(model, decay: float) -> None:
    """Pull the codec's target encoder towards its online encoder.  Called inside
    the jitted step after the optimizer, so the target always trails by one
    update.  The target takes no gradient (stop_gradient in encode_target), so
    this is the only thing that moves it."""
    codec = getattr(model, "codec", None)
    if codec is None:
        return
    online = nnx.state(codec.enc, nnx.Param)
    target = nnx.state(codec.target, nnx.Param)
    nnx.update(codec.target,
               jax.tree.map(lambda t, o: decay * t + (1 - decay) * o, target, online))


# ------------------------------------------------------------------ (a)
class FTInput(nnx.Module):
    """Current force is an input.  No future-force prediction."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.enc = ObsEncoder(cfg, rngs)
        self.vA = make_field((cfg.H, cfg.A), 2 * cfg.D, 1, cfg, rngs, route_dim=cfg.D)

    def loss(self, b, key):
        c = self.enc(b)
        k1, k2 = jax.random.split(key)
        tau = jax.random.uniform(k1, (len(c),))
        a_t, u = noisy(k2, b["action"], tau)
        v = self.vA(a_t, [tau], c, c[:, self.cfg.D:])
        l_a = mse(v, u)
        l_k, parts = robust_term(self, one_step(a_t, tau, v), b)
        return l_a + l_k, {"l_a": l_a, **parts}

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        a = jax.random.normal(key, (len(c), cfg.H, cfg.A))
        for k in range(cfg.n_steps):
            a = a + self.vA(a, [full(len(c), k / cfg.n_steps)], c, c[:, cfg.D:]) / cfg.n_steps
        return a, None


# ------------------------------------------------------------------ (b)
class UniDir(nnx.Module):
    """Force predictor (from the observation only) -> policy."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.enc = ObsEncoder(cfg, rngs)
        self.codec = FTCodec(cfg, rngs) if cfg.ft_latent else None
        fd = cfg.ft_latent or cfg.FT
        self.pred = MLP(2 * cfg.D, cfg.H * fd, cfg.D, rngs)
        self.vA = make_field((cfg.H, cfg.A), 2 * cfg.D + cfg.H * fd, 1, cfg, rngs,
                             route_dim=cfg.D + cfg.H * fd)

    def predict_ft(self, c):
        fd = self.cfg.ft_latent or self.cfg.FT
        return self.pred(c).reshape(len(c), self.cfg.H, fd)

    def loss(self, b, key):
        c = self.enc(b)
        z = self.predict_ft(c)
        tgt = self.codec.encode_target(b["future_ft"]) if self.codec else b["future_ft"]
        l_f = mse(z, tgt)
        # CoFA states L = L_A + 0.5 L_Z and never decodes Z.  We decode it only to
        # report a force error in newtons, so the reconstruction that keeps the
        # latent informative is its own term, outside the stated weighting.
        l_rec = self.codec.recon(b["future_ft"]) if self.codec else 0.0
        cond = jnp.concatenate([c, flat(jax.lax.stop_gradient(z))], -1)
        k1, k2 = jax.random.split(key)
        tau = jax.random.uniform(k1, (len(c),))
        a_t, u = noisy(k2, b["action"], tau)
        v = self.vA(a_t, [tau], cond, cond[:, self.cfg.D:])
        l_a = mse(v, u)
        l_k, parts = robust_term(self, one_step(a_t, tau, v), b)
        return (l_a + self.cfg.w_f * l_f + l_rec + l_k,
                {"l_a": l_a, "l_f": l_f, "l_rec": l_rec, **parts})

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        z = self.predict_ft(c)
        cond = jnp.concatenate([c, flat(z)], -1)
        a = jax.random.normal(key, (len(c), cfg.H, cfg.A))
        for k in range(cfg.n_steps):
            a = a + self.vA(a, [full(len(c), k / cfg.n_steps)], cond, cond[:, cfg.D:]) / cfg.n_steps
        return a, self.codec.decode(z) if self.codec else z


# ------------------------------------------------------------------ (c)
class Unified(nnx.Module):
    """One field, one time, over [action | future force]."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.enc = ObsEncoder(cfg, rngs)
        self.codec = FTCodec(cfg, rngs) if cfg.ft_latent else None
        self.v = make_field((cfg.H, cfg.A + (cfg.ft_latent or cfg.FT)), 2 * cfg.D, 1, cfg, rngs,
                            route_dim=cfg.D)

    def loss(self, b, key):
        A = self.cfg.A
        c = self.enc(b)
        tgt = self.codec.encode_target(b["future_ft"]) if self.codec else b["future_ft"]
        x1 = jnp.concatenate([b["action"], tgt], -1)
        k1, k2 = jax.random.split(key)
        tau = jax.random.uniform(k1, (len(c),))
        x_t, u = noisy(k2, x1, tau)
        v = self.v(x_t, [tau], c, c[:, self.cfg.D:])
        l_a, l_f = mse(v[..., :A], u[..., :A]), mse(v[..., A:], u[..., A:])
        l_rec = self.codec.recon(b["future_ft"]) if self.codec else 0.0
        l_k, parts = robust_term(self, one_step(x_t[..., :A], tau, v[..., :A]), b)
        return (l_a + self.cfg.w_f * l_f + l_rec + l_k,
                {"l_a": l_a, "l_f": l_f, "l_rec": l_rec, **parts})

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        x = jax.random.normal(key, (len(c), cfg.H, cfg.A + (cfg.ft_latent or cfg.FT)))
        for k in range(cfg.n_steps):
            x = x + self.v(x, [full(len(c), k / cfg.n_steps)], c, c[:, cfg.D:]) / cfg.n_steps
        z = x[..., cfg.A:]
        return x[..., :cfg.A], self.codec.decode(z) if self.codec else z


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
        H, A, D = cfg.H, cfg.A, cfg.D
        FD = cfg.ft_latent or cfg.FT
        self.enc = ObsEncoder(cfg, rngs)
        self.codec = FTCodec(cfg, rngs) if cfg.ft_latent else None
        # the MoE sits in the ACTION field, as the figure draws it
        self.vA = make_field((H, A), 2 * D + H * FD, 2, cfg, rngs, route_dim=D + H * FD)
        self.vF = make_field((H, FD), 2 * D + H * A, 2, cfg, rngs, moe=False)

    def loss(self, b, key):
        c = self.enc(b)
        kA, kF, ka, kf = jax.random.split(key, 4)
        lamA = jax.random.uniform(kA, (len(c),))
        lamF = jax.random.uniform(kF, (len(c),))
        tgt = self.codec.encode_target(b["future_ft"]) if self.codec else b["future_ft"]
        a_t, uA = noisy(ka, b["action"], lamA)
        z_t, uF = noisy(kf, tgt, lamF)
        condA = jnp.concatenate([c, flat(z_t)], -1)
        vA = self.vA(a_t, [lamA, lamF], condA, condA[:, self.cfg.D:])
        vF = self.vF(z_t, [lamF, lamA], jnp.concatenate([c, flat(a_t)], -1))
        l_a, l_f = mse(vA, uA), mse(vF, uF)
        l_rec = self.codec.recon(b["future_ft"]) if self.codec else 0.0
        l_k, parts = robust_term(self, one_step(a_t, lamA, vA), b)
        return (l_a + self.cfg.w_f * l_f + l_rec + l_k,
                {"l_a": l_a, "l_f": l_f, "l_rec": l_rec, **parts})

    def sample(self, b, key):
        cfg = self.cfg
        c = self.enc(b)
        kA, kF = jax.random.split(key)
        a = jax.random.normal(kA, (len(c), cfg.H, cfg.A))
        z = jax.random.normal(kF, (len(c), cfg.H, cfg.ft_latent or cfg.FT))
        n, N, lead = len(c), cfg.n_steps, cfg.d_lead
        # head start: the wrench denoises while the action is still pure noise,
        # so its clock ends up `lead` steps ahead.  z takes lead + (N - lead)
        # updates in total and its time still runs 0 -> 1, so nothing is
        # under-integrated; the action takes N updates as before.
        for k in range(lead):
            tF, tA = full(n, k / N), full(n, 0.0)
            z = z + self.vF(z, [tF, tA], jnp.concatenate([c, flat(a)], -1)) / N
        for k in range(N):
            tA = full(n, k / N)
            fF, fF1 = min(1.0, (k + lead) / N), min(1.0, (k + 1 + lead) / N)
            tF = full(n, fF)
            condA = jnp.concatenate([c, flat(z)], -1)
            vA = self.vA(a, [tA, tF], condA, condA[:, cfg.D:])
            vF = self.vF(z, [tF, tA], jnp.concatenate([c, flat(a)], -1))
            a = a + vA / N
            z = z + vF * (fF1 - fF)        # 1/N when lead = 0: the old behaviour
        return a, self.codec.decode(z) if self.codec else z


CASES = {"a": FTInput, "b": UniDir, "c": Unified, "d": CrossCond}
NAMES = {"a": "ft_input", "b": "uni_dir", "c": "unified", "d": "cross_cond",
         "e": "film"}          # (e) is an act-only structure, see act_model.py


def match_width(cfg: Config) -> int:
    """Hidden width that brings this structure to (d)'s parameter count.

    (d) carries two fields and is the structure under test, so it is the anchor
    and the others are widened up to it: a ranking should not be able to hide a
    capacity gap.  Counts grow with the width, so a bisection finds it quickly.
    """
    import dataclasses

    def size(case, D):
        c = dataclasses.replace(cfg, case=case, D=D)
        shape = nnx.eval_shape(lambda: CASES[case](c, nnx.Rngs(0)))
        return int(sum(x.size for x in jax.tree.leaves(nnx.state(shape, nnx.Param))))

    if cfg.case == "d":
        return cfg.D
    target = size("d", cfg.D)
    lo, hi = cfg.D, 4 * cfg.D
    while hi - lo > 8:
        mid = (lo + hi) // 16 * 8
        if size(cfg.case, mid) < target:
            lo = mid
        else:
            hi = mid
    return min((lo, hi), key=lambda D: abs(size(cfg.case, D) - target))


def run_name(cfg: Config) -> str:
    """Checkpoint directory name: the two families never share one, and neither
    do the two action layouts -- they are different label sets over the same
    episodes, so a shared directory would overwrite one run with the other."""
    name = NAMES[cfg.case] if cfg.arch == "flow" else f"act_{NAMES[cfg.case]}"
    # spring_rel must NOT collapse onto spring_: they are different labels and
    # different decodes, and sharing a directory would overwrite the 32 spring
    # evaluations already measured.
    name = {"legacy": name, "spring": f"spring_{name}",
            "spring_rel": f"springrel_{name}"}[cfg.layout]
    if cfg.w_robust > 0.0:
        # delta is swept against the evaluation's --dz, so it has to be in the
        # name: two deltas are two different hypotheses about how far the surface
        # moves, and sharing a directory would overwrite one with the other.
        name = f"{name}_d{round(1000 * cfg.robust_delta)}"
    if cfg.d_lead:
        name = f"{name}_lead{cfg.d_lead}"
    if not cfg.force:
        name = f"{name}_nf" + ("t" if not cfg.force_target else "")
    elif not cfg.force_target:
        name = f"{name}_nft"
    return name if cfg.ft_in == DATA.FT else f"{name}_sur"


def build(cfg: Config, seed: int = 0, stats=None):
    if cfg.case not in CASES and cfg.arch != "act":
        raise ValueError(f"case {cfg.case!r} exists only for --arch act")
    if cfg.arch == "act":
        import act_model                     # imported here to keep the cycle out
        model = act_model.CompACT(cfg, nnx.Rngs(seed))
    else:
        model = CASES[cfg.case](cfg, nnx.Rngs(seed))
    # The robustness term is stated in newtons and metres, so it needs the
    # action denormalization.  Plain arrays, not Params: nothing trains them.
    if cfg.w_robust > 0.0:
        if stats is None:
            raise ValueError("w_robust > 0 needs `stats` so the term can be "
                             "evaluated in physical units")
        model.act_mean = jnp.asarray(stats.mean["act"])
        model.act_std = jnp.asarray(stats.std["act"])
    return model


def n_params(model) -> int:
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))


def describe(cfg: Config) -> str:
    """What this structure consumes and produces, in shapes.

    Printed at the start of every run: the four cases differ only in what
    conditions what, and that is exactly what is easy to get wrong.
    """
    if cfg.arch == "act":
        import act_model
        return act_model.describe(cfg)
    H, A, FT, D, G = cfg.H, cfg.A, cfg.FT, cfg.D, cfg.G
    obs = [f"top, wrist  (64, 64, 3)      {'both cameras' if cfg.vision else 'DISABLED (--no-vision)'}",
           f"state       ({cfg.S},)            x_d - origin, tcp - x_d, tcp velocity, log K, qpos",
           f"goal        ({2 * G},)            {G} path points (xy), relative to the current x_d"
           if cfg.goal else
           "goal        --                 not given: the target is read off the paper",
           f"ft_hist     ({cfg.HIST}, {cfg.ft_in})          "
           + ("contact force, last "
              f"{cfg.HIST} frames -> GRU" if cfg.ft_in == FT else
              "measured | EXPECTED | z-score (surprise), "
              f"last {cfg.HIST} frames -> GRU")]
    parts = "+".join(p for p, on in (("vision", cfg.vision), ("state", True), ("goal", cfg.goal)) if on)
    enc = f"observation encoder -> cond c  (2D = {2 * D},)   [{parts} | force GRU]"
    kr_note = " + log K_R (1)" if A in (7, 10) else ""
    cols = ("x_ref offset (3, m) + f_d u, v, n (3, N, tool -> env)"
            " + log stiffness (3)" if DATA.is_spring(cfg.layout) else
            "x_d offset (3, m) + log stiffness u, v, n (3)")
    act = f"action      ({H}, {A})         {cols}{kr_note}"
    fut = f"future_ft   ({H}, {FT})         contact force over the same {H} frames"

    fields = {
        "a": [f"vA  in ({H},{A}) + cond {2 * D} + 1 time -> ({H},{A})",
              "no future-force field: force enters only through the encoder's GRU"],
        "b": [f"pred  cond {2 * D} -> ({H},{FT})            deterministic, observation only",
              f"vA  in ({H},{A}) + cond {2 * D} + {H * FT} (predicted force, stop-grad) + 1 time -> ({H},{A})"],
        "c": [f"v   in ({H},{A + FT}) + cond {2 * D} + 1 time -> ({H},{A + FT})   one field, one time",
              "the action and the force share a noise vector, so they are sampled jointly"],
        "d": [f"vA  in ({H},{A}) + cond {2 * D} + {H * FT} (force state) + 2 times -> ({H},{A})",
              f"vF  in ({H},{FT}) + cond {2 * D} + {H * A} (action state) + 2 times -> ({H},{FT})",
              "independent flow times in training; one shared clock at inference"],
    }[cfg.case]

    out = {"a": "sample() -> action, None", "b": "sample() -> action, predicted force",
           "c": "sample() -> action, force (split from the joint vector)",
           "d": "sample() -> action, force (co-evolved)"}[cfg.case]

    lines = [f"structure ({cfg.case}) {NAMES[cfg.case]}   layout: {cfg.layout}"
             + (f"   w_robust {cfg.w_robust} (delta {1000 * cfg.robust_delta:.1f} mm)"
                if cfg.w_robust else "")
             + ("" if cfg.force and cfg.force_target else
                "   CONTROL: force input "
                + ("ZEROED" if not cfg.force else "on")
                + ", future-force target "
                + ("ZEROED" if not cfg.force_target else "on")), "  inputs:"]
    lines += [f"    {o}" for o in obs]
    lines += [f"  {enc}", "  targets:", f"    {act}"]
    if cfg.case != "a":
        lines += [f"    {fut}"]
        if DATA.is_spring(cfg.layout):
            lines += ["      ^ REDUNDANT under the spring layout: the future wrench is",
                      "        -W f_d, already in the action.  Keeping the field is the",
                      "        ablation that asks whether a separate force head bought",
                      "        anything once it could no longer disagree."]
    lines += ["  fields:"] + [f"    {f}" for f in fields]
    lines += [f"  {out}", f"  Euler steps at inference: {cfg.n_steps}   future-force loss weight: {cfg.w_f}"]
    return "\n".join(lines)
