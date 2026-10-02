"""Comp-ACT and its three force-aware variants, in Flax NNX.

Comp-ACT (Kamijo, Beltran-Hernandez and Hamaya, arXiv:2406.14990,
github.com/omron-sinicx/CompACT) is ACT -- a CVAE whose decoder emits a whole
action chunk in one shot -- with **stiffness inside the action**, so the policy
learns a reference trajectory and the time-varying compliance to follow it
with.  Its `DETRVAE(include_ft=True)` feeds the measured wrench to the decoder
as its own token.  That is exactly structure (a) of the CoFA figure, which is
why case a here *is* Comp-ACT and the other three are modifications of it:

  a  ft_input    Comp-ACT unchanged: the wrench is a decoder token, nothing
                 predicts the future
  b  uni_dir     a future-wrench decoder reads the observation alone; its
                 (detached) chunk becomes extra memory for the action decoder
  c  unified     one decoder, one query per frame, whose head emits the action
                 and the future wrench together
  d  cross_cond  an action decoder and a wrench decoder, each reading the
                 other's current estimate, unrolled for `n_refine` passes

(d) needs care.  In the flow family the two fields meet along the integration
steps, each seeing the other at its own flow time.  ACT has no such axis, so we
give it one: during training each stream is shown the other's target blended
with noise at an independently drawn level lambda (with lambda embedded as a
token, exactly the role the flow time plays), and at inference the pair is
unrolled from noise with lambda rising to 1.  Same mutual conditioning, on the
only clock a one-shot decoder can be given.

Deviations from the published configuration, and why: hidden 256 / 4 encoder /
4 decoder layers rather than 512 / 4 / 7, and a chunk of 10 rather than 100 --
this task predicts 1 s at 10 Hz from 135 demonstrations, where the paper's
capacity would be mostly free parameters.  Both are `Config` fields.  There is
no `is_pad` head because data.py clamps chunks at episode ends instead of
padding them.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

import data as DATA
from model import NAMES, Config, flat, robust_term, time_emb


def sinusoid(n, d):
    """ACT's fixed position table, used for the query and token positions."""
    pos = jnp.arange(n)[:, None]
    i = jnp.arange(d)[None]
    ang = pos / jnp.power(10000.0, 2 * (i // 2) / d)
    return jnp.where(i % 2 == 0, jnp.sin(ang), jnp.cos(ang))


class FF(nnx.Module):
    def __init__(self, h, ff, rngs):
        self.l1 = nnx.Linear(h, ff, rngs=rngs)
        self.l2 = nnx.Linear(ff, h, rngs=rngs)

    def __call__(self, x):
        return self.l2(nnx.gelu(self.l1(x)))


class EncLayer(nnx.Module):
    def __init__(self, cfg: Config, rngs):
        h = cfg.hidden
        self.n1, self.n2 = nnx.LayerNorm(h, rngs=rngs), nnx.LayerNorm(h, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(cfg.nheads, h, decode=False, rngs=rngs)
        self.ff = FF(h, cfg.ff, rngs)

    def __call__(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y)
        return x + self.ff(self.n2(x))


class DecLayer(nnx.Module):
    """A decoder block, optionally modulated by the wrench (adaLN-Zero).

    With `film`, a zero-initialised projection turns the force feature into a
    shift, a scale and a gate for each of the three sub-layers.  The gate enters
    as `1 + g`, so at initialisation scale = 1, shift = 0 and gate = 1: the block
    is *exactly* the unmodulated one and force modulation grows from zero.

    (DiT's adaLN-Zero opens with the gates shut, which suits a diffusion backbone
    whose signal arrives through the noisy latent.  Here the decoder is the only
    path from observation to action, so shutting its residual branches at step
    one would start the policy deaf to the memory as well as to the force.)
    """

    def __init__(self, cfg: Config, rngs):
        h = cfg.hidden
        self.film = cfg.film
        self.n1 = nnx.LayerNorm(h, rngs=rngs)
        self.n2 = nnx.LayerNorm(h, rngs=rngs)
        self.n3 = nnx.LayerNorm(h, rngs=rngs)
        self.self_attn = nnx.MultiHeadAttention(cfg.nheads, h, decode=False, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(cfg.nheads, h, decode=False, rngs=rngs)
        self.ff = FF(h, cfg.ff, rngs)
        if cfg.film:
            self.mod = nnx.Linear(h, 9 * h, rngs=rngs,
                                  kernel_init=nnx.initializers.zeros_init(),
                                  bias_init=nnx.initializers.zeros_init())

    def __call__(self, q, mem, f=None):
        if not self.film or f is None:
            y = self.n1(q)
            q = q + self.self_attn(y, y, y)
            y = self.n2(q)
            q = q + self.cross_attn(y, mem, mem)
            return q + self.ff(self.n3(q))
        p = jnp.split(self.mod(f)[:, None], 9, -1)          # (B, 1, h) each
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = p
        y = self.n1(q) * (1 + sc1) + sh1
        q = q + (1 + g1) * self.self_attn(y, y, y)
        y = self.n2(q) * (1 + sc2) + sh2
        q = q + (1 + g2) * self.cross_attn(y, mem, mem)
        y = self.n3(q) * (1 + sc3) + sh3
        return q + (1 + g3) * self.ff(y)


class ResNetTokens(nnx.Module):
    """Comp-ACT's camera encoder: a ResNet18 whose last feature map becomes the
    encoder tokens.  At 64x64 that map is 2x2, so 4 tokens a camera."""

    def __init__(self, cfg: Config, h, rngs):
        from model import ResNet18
        self.net = ResNet18(h, rngs, pretrained=cfg.pretrained)
        self.proj = nnx.Linear(512, h, rngs=rngs)

    def __call__(self, x):
        f = self.net.features(x)
        b, hh, ww, cc = f.shape
        return self.proj(f.reshape(b, hh * ww, cc))


class ConvTokens(nnx.Module):
    """One camera -> a 4x4 grid of tokens, ACT's backbone feature map in little."""

    def __init__(self, h, rngs):
        self.c1 = nnx.Conv(3, 16, (8, 8), strides=(4, 4), rngs=rngs)
        self.c2 = nnx.Conv(16, 32, (4, 4), strides=(2, 2), rngs=rngs)
        self.c3 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), rngs=rngs)
        self.proj = nnx.Linear(64, h, rngs=rngs)

    def __call__(self, x):
        for c in (self.c1, self.c2, self.c3):
            x = nnx.gelu(c(x))
        b, hh, ww, cc = x.shape
        return self.proj(x.reshape(b, hh * ww, cc))


class Memory(nnx.Module):
    """Observation -> transformer memory, and the CVAE latent token with it."""

    def __init__(self, cfg: Config, rngs):
        h = cfg.hidden
        self.cfg = cfg
        if cfg.vision:
            if cfg.backbone == "resnet18":
                self.top, self.wrist = ResNetTokens(cfg, h, rngs), ResNetTokens(cfg, h, rngs)
            else:
                self.top, self.wrist = ConvTokens(h, rngs), ConvTokens(h, rngs)
        self.state = nnx.Linear(cfg.S, h, rngs=rngs)
        if cfg.goal:
            self.goal = nnx.Linear(2 * cfg.G, h, rngs=rngs)
        # Comp-ACT's include_ft: the measured wrench is its own token.  With
        # ft_tokens it stays resolved in time -- one token per measured frame.
        self.gru = nnx.RNN(nnx.GRUCell(cfg.ft_in, h, rngs=rngs))
        if cfg.ft_tokens and cfg.case != "e":
            self.ft_proj = nnx.Linear(cfg.ft_in, h, rngs=rngs)
            self.ft_pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (cfg.HIST, h)))
        self.latent = nnx.Linear(cfg.latent, h, rngs=rngs)
        n_img = (8 if cfg.backbone == "resnet18" else 32) if cfg.vision else 0
        # case (e) isolates the mechanism: force reaches the policy ONLY through
        # FiLM, so it is not also a token competing for attention
        n_ft = 0 if cfg.case == "e" else (cfg.HIST if cfg.ft_tokens else 1)
        n = n_img + 2 + n_ft + (1 if cfg.goal else 0)   # state, latent, wrench
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (n, h)))
        self.layers = nnx.List([EncLayer(cfg, rngs) for _ in range(cfg.enc_layers)])

    def __call__(self, b, z):
        f = self.gru(b["ft_hist"])[:, -1]                    # force feature, for FiLM
        toks = [self.state(b["state"])[:, None], self.latent(z)[:, None]]
        if self.cfg.case != "e":
            toks.append(self.ft_proj(b["ft_hist"]) + self.ft_pos.value[None]
                        if self.cfg.ft_tokens else f[:, None])
        if self.cfg.goal:
            toks.append(self.goal(b["goal"])[:, None])
        if self.cfg.vision:
            toks += [self.top(b["top"]), self.wrist(b["wrist"])]
        x = jnp.concatenate(toks, 1) + self.pos.value[None]
        for l in self.layers:
            x = l(x)
        return x, f


class StyleEncoder(nnx.Module):
    """ACT's CVAE encoder: [cls, state, action chunk] -> mu, log var."""

    def __init__(self, cfg: Config, rngs):
        h = cfg.hidden
        self.cfg = cfg
        self.cls = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, h)))
        self.act = nnx.Linear(cfg.A, h, rngs=rngs)
        self.state = nnx.Linear(cfg.S, h, rngs=rngs)
        self.layers = nnx.List([EncLayer(cfg, rngs) for _ in range(2)])
        self.out = nnx.Linear(h, 2 * cfg.latent, rngs=rngs)

    def __call__(self, b):
        n = len(b["state"])
        x = jnp.concatenate([jnp.tile(self.cls.value[None], (n, 1, 1)),
                             self.state(b["state"])[:, None],
                             self.act(b["action"])], 1)
        x = x + sinusoid(x.shape[1], self.cfg.hidden)[None]
        for l in self.layers:
            x = l(x)
        mu, logvar = jnp.split(self.out(x[:, 0]), 2, -1)
        return mu, logvar


class Decoder(nnx.Module):
    """`n_out` values per chunk frame, from learned queries over the memory."""

    def __init__(self, cfg: Config, n_out, rngs):
        h = cfg.hidden
        self.cfg = cfg
        self.query = nnx.Param(0.02 * jax.random.normal(rngs.params(), (cfg.H, h)))
        self.layers = nnx.List([DecLayer(cfg, rngs) for _ in range(cfg.dec_layers)])
        self.norm = nnx.LayerNorm(h, rngs=rngs)
        self.head = nnx.Linear(h, n_out, rngs=rngs)

    def __call__(self, mem, extra=None, f=None):
        q = jnp.tile(self.query.value[None], (len(mem), 1, 1))
        m = mem if extra is None else jnp.concatenate([mem, extra], 1)
        for l in self.layers:
            q = l(q, m, f)
        return self.head(self.norm(q))


class Stream(nnx.Module):
    """Projects another stream's current estimate, with its fidelity lambda, to
    memory tokens -- the ACT stand-in for reading a field at its flow time.

    The frame position matters: the action at frame j has to be read against the
    force at frame j, and without a position embedding these tokens arrive as an
    unordered set and the decoder can only use their average."""

    def __init__(self, cfg: Config, dim, rngs):
        self.proj = nnx.Linear(dim, cfg.hidden, rngs=rngs)
        self.lam = nnx.Linear(32, cfg.hidden, rngs=rngs)
        self.pos = nnx.Param(0.02 * jax.random.normal(rngs.params(), (cfg.H, cfg.hidden)))

    def __call__(self, x, lam):
        return self.proj(x) + self.lam(time_emb(lam))[:, None] + self.pos.value[None]


def kl(mu, logvar):
    return jnp.mean(-0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar), -1))


def blend(key, x1, lam):
    """x1 mixed with noise at fidelity lam -- lam = 1 is the target, 0 is noise."""
    x0 = jax.random.normal(key, x1.shape)
    t = lam[:, None, None]
    return t * x1 + (1 - t) * x0


class CompACT(nnx.Module):
    """The four structures; `cfg.case` picks which heads exist."""

    def __init__(self, cfg: Config, rngs):
        self.cfg = cfg
        self.style = StyleEncoder(cfg, rngs)
        self.mem = Memory(cfg, rngs)
        if cfg.case == "c":
            self.dec = Decoder(cfg, cfg.A + cfg.FT, rngs)
        else:
            self.dec = Decoder(cfg, cfg.A, rngs)      # a, b, d, e
        if cfg.case in ("b", "d"):
            self.decF = Decoder(cfg, cfg.FT, rngs)
        if cfg.case == "b":
            self.fstream = Stream(cfg, cfg.FT, rngs)
        if cfg.case == "d":
            self.fstream = Stream(cfg, cfg.FT, rngs)
            self.astream = Stream(cfg, cfg.A, rngs)

    # ------------------------------------------------------------- training
    def loss(self, b, key):
        cfg = self.cfg
        mu, logvar = self.style(b)
        kz, k1, k2 = jax.random.split(key, 3)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(kz, mu.shape)
        mem, f = self.mem(b, z)
        f = f if cfg.film else None
        l_kl = kl(mu, logvar)
        l1 = lambda p, t: jnp.mean(jnp.abs(p - t))
        # The action is predicted directly here, so the robustness term reads
        # it straight off the decoder rather than through a flow's one-step
        # estimate.  See model.robust_term / Config.w_robust.
        robust = lambda a_p: robust_term(self, a_p, b)

        if cfg.case in ("a", "e"):
            a_p = self.dec(mem, None, f)
            l_a = l1(a_p, b["action"])
            l_k, parts = robust(a_p)
            return l_a + cfg.kl_weight * l_kl + l_k, {"l_a": l_a, "l_kl": l_kl, **parts}

        if cfg.case == "b":
            zf = self.decF(mem, None, f)
            l_f = l1(zf, b["future_ft"])
            # the action decoder reads the prediction, not the truth, so
            # training and inference see the same thing
            extra = self.fstream(jax.lax.stop_gradient(zf), jnp.ones(len(mem)))
            a_p = self.dec(mem, extra, f)
            l_a = l1(a_p, b["action"])
            l_k, parts = robust(a_p)
            return (l_a + cfg.w_f * l_f + cfg.kl_weight * l_kl + l_k,
                    {"l_a": l_a, "l_f": l_f, "l_kl": l_kl, **parts})

        if cfg.case == "c":
            out = self.dec(mem, None, f)
            l_a = l1(out[..., :cfg.A], b["action"])
            l_f = l1(out[..., cfg.A:], b["future_ft"])
            l_k, parts = robust(out[..., :cfg.A])
            return (l_a + cfg.w_f * l_f + cfg.kl_weight * l_kl + l_k,
                    {"l_a": l_a, "l_f": l_f, "l_kl": l_kl, **parts})

        # (d) each stream reads the other at an independently drawn fidelity
        ka, kf, ba, bf = jax.random.split(k1, 4)
        lamA = jax.random.uniform(ka, (len(mem),))
        lamF = jax.random.uniform(kf, (len(mem),))
        a_in = blend(ba, b["action"], lamA)
        f_in = blend(bf, b["future_ft"], lamF)
        a_p = self.dec(mem, self.fstream(f_in, lamF), f)
        l_a = l1(a_p, b["action"])
        l_f = l1(self.decF(mem, self.astream(a_in, lamA), f), b["future_ft"])
        l_k, parts = robust(a_p)
        return (l_a + cfg.w_f * l_f + cfg.kl_weight * l_kl + l_k,
                {"l_a": l_a, "l_f": l_f, "l_kl": l_kl, **parts})

    # ------------------------------------------------------------ inference
    def sample(self, b, key):
        """z = 0, the CVAE prior's mean, as ACT does at test time."""
        cfg = self.cfg
        n = len(b["state"])
        mem, f = self.mem(b, jnp.zeros((n, cfg.latent)))
        f = f if cfg.film else None

        if cfg.case in ("a", "e"):
            return self.dec(mem, None, f), None
        if cfg.case == "b":
            zf = self.decF(mem, None, f)
            return self.dec(mem, self.fstream(zf, jnp.ones(n)), f), zf
        if cfg.case == "c":
            out = self.dec(mem, None, f)
            return out[..., :cfg.A], out[..., cfg.A:]

        ka, kf = jax.random.split(key)
        a = jax.random.normal(ka, (n, cfg.H, cfg.A))
        z = jax.random.normal(kf, (n, cfg.H, cfg.FT))
        for k in range(cfg.n_refine):                    # unroll from noise
            # lambda must reach 1 on the last pass: by then the other stream's
            # estimate is nearly clean, and telling the decoder otherwise makes
            # it discount the very thing cross-conditioning is for.
            lam = jnp.full((n,), k / max(1, cfg.n_refine - 1))
            a_next = self.dec(mem, self.fstream(z, lam), f)
            z = self.decF(mem, self.astream(a, lam), f)
            a = a_next                                   # synchronous, as in (d)
        return a, z


def describe(cfg: Config) -> str:
    """What this Comp-ACT variant consumes and produces, in shapes."""
    H, A, FT, h, G = cfg.H, cfg.A, cfg.FT, cfg.hidden, cfg.G
    # FT is the future-force OUTPUT width; cfg.ft_in is what ft_hist carries,
    # which is 9 under --surprise.  Printing FT for both said "(10, 3)" on a
    # 9-channel input, so read the input width from the field that holds it.
    FI = cfg.ft_in
    ft_note = ("measured | EXPECTED | z-score (surprise)" if FI != FT
               else "contact force")
    cols = ("x_ref offset (3, m) + f_d u, v, n (3, N, tool -> env) + log stiffness (3)"
            if DATA.is_spring(cfg.layout) else
            "x_d offset (3, m) + log stiffness u, v, n (3)")
    cols += " + log K_R (1)" if A in (7, 10) else ""
    n_img = ((8 if cfg.backbone == "resnet18" else 32) if cfg.vision else 0)
    mem = n_img + 3 + (1 if cfg.goal else 0)
    lines = [f"structure ({cfg.case}) act_{NAMES[cfg.case]}   layout: {cfg.layout}   "
             f"[Comp-ACT, arXiv:2406.14990]"
             + ("  <- unmodified" if cfg.case == "a" and cfg.layout == "legacy" else "")
             + (f"   w_robust {cfg.w_robust}" if cfg.w_robust else ""),
             "  inputs:",
             f"    top, wrist  (64, 64, 3)      "
             + (f"2 x {n_img // 2} tokens ({cfg.backbone})" if cfg.vision else "DISABLED"),
             f"    state       ({cfg.S},)            1 token",
             (f"    goal        ({2 * G},)            1 token" if cfg.goal else
              "    goal        --                 not given: the target is read off the paper"),
             (f"    ft_hist     ({cfg.HIST}, {FI})          {ft_note}, "
              "GRU -> adaLN-Zero, not a token"
              if cfg.case == "e" else
              f"    ft_hist     ({cfg.HIST}, {FI})          {ft_note}, "
              + (f"{cfg.HIST} tokens, one per frame" if cfg.ft_tokens
                 else "GRU -> 1 token (Comp-ACT's include_ft)")),
             f"  CVAE: [cls, state, action chunk] -> z ({cfg.latent},), z = 0 at test time",
             f"  encoder memory: {mem} tokens of {h}, {cfg.enc_layers} layers",
             "  targets:",
             f"    action      ({H}, {A})         {cols}"]
    if cfg.case != "a":
        lines.append(f"    future_ft   ({H}, {FT})         contact force over the same {H} frames")
        if DATA.is_spring(cfg.layout):
            lines.append("      ^ REDUNDANT under the spring layout: it is -W f_d, "
                         "already in the action")
    dec = {
        "a": [f"action decoder  {H} queries, {cfg.dec_layers} layers -> ({H},{A})"],
        "b": [f"force decoder   {H} queries -> ({H},{FT}), observation only",
              f"action decoder  memory + {H} tokens of the predicted force (stop-grad) -> ({H},{A})"],
        "c": [f"one decoder     {H} queries -> ({H},{A + FT}), action and force from one head"],
        "d": [f"action decoder  memory + {H} tokens of the force estimate + its lambda -> ({H},{A})",
              f"force decoder   memory + {H} tokens of the action estimate + its lambda -> ({H},{FT})",
              f"independent lambda in training; {cfg.n_refine} synchronous passes from noise at test time"],
        "e": [f"action decoder  {H} queries, {cfg.dec_layers} layers -> ({H},{A})",
              "the wrench is NOT a token: it reaches the decoder only through adaLN-Zero,",
              "scaling and shifting the features of all three sub-layers (gain scheduling)"],
    }[cfg.case]
    lines += ["  decoders:"] + [f"    {d}" for d in dec]
    out = {"a": "sample() -> action, None", "b": "sample() -> action, predicted force",
           "c": "sample() -> action, force (one head)", "d": "sample() -> action, force (co-refined)",
           "e": "sample() -> action, None"}
    lines += [f"  {out[cfg.case]}",
              f"  L1 action loss + {cfg.w_f} x L1 force + {cfg.kl_weight} x KL"]
    return "\n".join(lines)


def match_hidden(cfg: Config) -> int:
    """Width that brings this variant UP to the largest structure's size.

    (b) and (d) carry a second decoder, so they are the big ones; narrowing them
    would take them off Comp-ACT's published dimensions.  Instead (a) and (c) are
    widened to meet them, which leaves every structure at least as large as the
    paper's and removes capacity as an explanation for the ranking.  Counts grow
    with the width, so a bisection finds it in a few traced constructions.
    """
    import dataclasses

    def size(case, h, **over):
        c = dataclasses.replace(cfg, case=case, hidden=h, **over)
        shape = nnx.eval_shape(lambda: CompACT(c, nnx.Rngs(0)))
        return int(sum(x.size for x in jax.tree.leaves(nnx.state(shape, nnx.Param))))

    # The reference is the four plain structures.  Without this the mechanism
    # under test inflates its own target: film adds a modulation projection to
    # every decoder layer, so matching against "a-d WITH film" sized act_film at
    # 160M against a 125M baseline.
    plain = dict(film=False, ft_tokens=False)
    target = max(size(c, cfg.hidden, **plain) for c in "abcd")
    if size(cfg.case, cfg.hidden) >= target:
        return cfg.hidden                      # already the largest: (b) and (d)
    lo, hi = cfg.hidden, 2 * cfg.hidden
    while hi - lo > cfg.nheads:
        mid = (lo + hi) // 2 // cfg.nheads * cfg.nheads
        if size(cfg.case, mid) < target:
            lo = mid
        else:
            hi = mid
    return min((lo, hi), key=lambda h: abs(size(cfg.case, h) - target))
