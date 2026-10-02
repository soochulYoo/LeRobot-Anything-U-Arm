"""Force as surprise: the wrench a policy could NOT have predicted.

Raw force answers three different questions at once, and only one of them is
about the environment.  Under a variable stiffness, 5 N means "the surface
stopped me 10 mm short" at K = 500 N/m and "2.5 mm short" at K = 2000 N/m, so
most of the signal is the policy's own commanded compliance coming back at it.
What vision cannot supply -- two blocks that look identical and one is sponge --
is the part of the wrench that the scene does not predict.

So train an EXPECTATION model

    g(images, where I am, where I am going)  ->  mu, sigma   of the wrench

and give the policy the expectation AND the standardized residual

    z = (F - mu) / sigma

Two identical boxes, one full of sand: expected 2 N, measured 20 N, and the
z of +18 N says "stiffen" before anything looks different.  Vision cannot
drown z, because z is defined as what vision got wrong.

THE LEAK THAT MAKES THIS FEATURE USELESS IF YOU MISS IT
-------------------------------------------------------
`data.state_vec` carries `tcp - x_d`, the spring deflection, and `log K`.  GIC's
spring law makes those two ALGEBRAICALLY the force: F = K (x_d - tcp).  An
expectation model given the full state therefore does not predict the wrench,
it COMPUTES it, mu comes out equal to F, and z is identically zero -- a feature
that looks implemented, trains without error and carries nothing.  `S_KEEP`
below is the force-free slice: where the reference is, and how fast the tip is
moving.  The deflection, log K and qpos are all excluded, qpos because forward
kinematics plus x_d reconstructs the deflection.

CROSS-FITTING, for the same reason in the other direction.  An expectation model
that saw an episode predicts that episode's wrench nearly exactly, so z would be
~0 on training data and large at rollout, and the policy would learn to ignore
the one input that matters.  Episodes are split into K folds and every sample's
(mu, sigma) comes from the fold model that never saw its episode.

A SECOND RESIDUAL, free, and not this file's job: `measured - K (x_d - tcp)` is
what the environment and the arm's own dynamics add on top of the spring, and it
needs no model at all -- both terms are already in the state.

    python3 policy/surprise.py --data demos/protocol_v1 --steps 4000
writes demos/protocol_v1/_surprise_64_k4.npz, which `data.load(surprise=...)`
turns into the extra six channels of `ft_hist`.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import pickle
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
import optax                    # noqa: E402
from flax import nnx            # noqa: E402

import data as DATA             # noqa: E402
import model as M               # noqa: E402

# The force-free slice of data.state_vec:
#   0:3   x_d - origin     where the reference is on the paper
#   6:9   tcp velocity     how fast the tip is moving
# EXCLUDED: 3:6 tcp - x_d (the deflection IS the force), 9:12 log K (with the
# deflection it is the force, and on its own it is the demonstrator's reaction
# to the force), 12:19 qpos (FK + x_d rebuilds the deflection).
S_KEEP = np.r_[0:3, 6:9]


class Expectation(nnx.Module):
    """(images, force-free state, goal) -> (mu, log sigma^2) of the wrench."""

    def __init__(self, cfg: M.Config, rngs):
        D = cfg.D
        self.cfg = cfg
        self.top, self.wrist = M.CNN(D, rngs), M.CNN(D, rngs)
        self.state = nnx.Linear(len(S_KEEP), D, rngs=rngs)
        self.goal = nnx.Linear(2 * cfg.G, D, rngs=rngs)
        self.head = M.MLP(D, 2 * cfg.FT, D, rngs)

    def __call__(self, b):
        h = (self.top(b["top"]) + self.wrist(b["wrist"])
             + self.state(b["sur_state"]) + self.goal(b["goal"]))
        out = self.head(nnx.gelu(h))
        mu, logvar = out[:, :self.cfg.FT], out[:, self.cfg.FT:]
        # sigma is bounded below: an unbounded one can drive the NLL to -inf on
        # the frames where the wrench is exactly zero (the pen in free space)
        return mu, jnp.clip(logvar, -8.0, 4.0)

    def loss(self, b):
        """Heteroscedastic Gaussian NLL.  sigma matters as much as mu here: the
        residual is divided by it, and a constant sigma would make quiet frames
        in free space look as surprising as a missed contact."""
        mu, logvar = self(b)
        return jnp.mean(0.5 * ((b["ft"] - mu) ** 2 * jnp.exp(-logvar) + logvar))


# --------------------------------------------------------------------------- #
# THE DEPLOY MODEL: why there has to be one, and why it is not a fold model.
#
# The folds exist to make the TRAINING targets honest -- every sample's
# (mu, sigma) comes from a model that never saw its episode.  The fold models
# were then thrown away, which left a policy that trains on nine channels and
# cannot be rolled out at all: at inference there is no mu, so `ft_hist` stays
# three channels wide and data.Stats.norm raises
#     ValueError: operands could not be broadcast together with shapes (1,10,3) (9,)
# which is exactly how all 32 *_sur evaluations died.
#
# So a separate model is fitted on ALL episodes and saved beside the npz.  State
# the mismatch plainly: a training sample saw mu from a model blind to its
# episode, while at rollout mu comes from a model fitted on every demonstration.
# For the evaluation episodes that is still out-of-sample -- they are unseen
# randomizations -- so the deploy model is the right analogue of the fold models
# rather than a leak.  It is a slightly DIFFERENT predictor though.
#
# Its own Stats travel with it: the folds are fitted with val_per_case=0
# statistics and a policy checkpoint carries train-split ones, and mixing them
# would quietly shift the expectation model's inputs.
def save_deploy(path, gs, stats) -> None:
    """`gs` is a model or a list of them; a list is saved as the ensemble."""
    gs = gs if isinstance(gs, (list, tuple)) else [gs]
    ps = [jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(g, nnx.Param))) for g in gs]
    with open(path, "wb") as f:
        pickle.dump(dict(params=ps if len(ps) > 1 else ps[0],
                         config=dataclasses.asdict(gs[0].cfg),
                         stats=stats.to_json()), f)


class Deploy:
    """The expectation model at inference: observation -> (mu, sigma), online.

    AN ENSEMBLE OF THE FOLD MODELS, not a separate all-data fit.  A training
    sample's mu came from one fold model; serving the rollout from a different
    single model made the two sides disagree by 0.52-0.77 sigma.  The mean of the
    K folds has about 1/sqrt(K) of a single model's variance, so it sits closer
    to every individual fold than two folds sit to each other -- which is the
    quantity that matters here.  It also never saw any episode more than K-1
    folds' worth, so nothing about the honest out-of-fold targets changes.

    `params` may therefore be a LIST of K parameter sets; a single set is still
    accepted so the older all-data pickles keep loading.
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        self.cfg = M.Config(**ck["config"])
        self.stats = DATA.Stats.from_json(ck["stats"])
        ps = ck["params"] if isinstance(ck["params"], list) else [ck["params"]]
        self.gs = []
        for i, pr in enumerate(ps):
            g = Expectation(self.cfg, nnx.Rngs(i))
            st = nnx.state(g, nnx.Param)
            nnx.replace_by_pure_dict(st, pr)
            nnx.update(g, st)
            self.gs.append(g)
        self.g = self.gs[0]          # for the in-sample R^2 printout
        self._fwd = nnx.jit(lambda m, b: m(b))

    def predict(self, top, wrist, state, goal):
        """RAW arrays for one frame, batched (1, ...) -> (mu, sigma) as (1, 3).

        mu is the ensemble mean.  sigma combines the members' own predicted
        spread with their DISAGREEMENT (total variance = mean aleatoric + spread
        of the means), which is the honest uncertainty of an ensemble.
        """
        b = dict(top=jnp.asarray(np.asarray(top, np.float32) / 255.0),
                 wrist=jnp.asarray(np.asarray(wrist, np.float32) / 255.0),
                 sur_state=jnp.asarray(self.stats.norm("state", state)[:, S_KEEP]),
                 goal=jnp.asarray(self.stats.norm("goal", goal)))
        mus, vrs = [], []
        for g in self.gs:
            mu, logvar = self._fwd(g, b)
            mus.append(np.asarray(mu))
            vrs.append(np.exp(np.asarray(logvar)))
        mus = np.stack(mus)
        mu = mus.mean(0)
        var = np.stack(vrs).mean(0) + mus.var(0)
        return mu, np.sqrt(var)


def deploy_path(npz_path) -> pathlib.Path:
    """<...>_surprise_64_k4.npz -> <...>_surprise_64_k4_deploy.pkl"""
    q = pathlib.Path(npz_path)
    return q.with_name(q.stem + "_deploy.pkl")


def batch(d, stats, idx, rng=None):
    top, wrist = d["top"][idx], d["wrist"][idx]
    if rng is not None:
        top, wrist = DATA.shift_aug(top, rng), DATA.shift_aug(wrist, rng)
    return dict(top=jnp.asarray(top.astype(np.float32) / 255.0),
                wrist=jnp.asarray(wrist.astype(np.float32) / 255.0),
                sur_state=jnp.asarray(stats.norm("state", d["state"][idx])[:, S_KEEP]),
                goal=jnp.asarray(stats.norm("goal", d["goal"][idx])),
                # the wrench at the CURRENT frame, the last of the history
                ft=jnp.asarray(d["ft_hist"][idx][:, -1, :]))


def fit_fold(d, stats, train_i, steps, lr, seed, batch_size):
    cfg = M.Config(D=128, G=DATA.G, FT=DATA.FT)
    g = Expectation(cfg, nnx.Rngs(seed))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, steps // 10), steps, lr * 0.05)
    opt = nnx.Optimizer(g, optax.adamw(sched, weight_decay=1e-4), wrt=nnx.Param)

    @nnx.jit
    def step(g, opt, b):
        loss, grads = nnx.value_and_grad(lambda m: m.loss(b))(g)
        opt.update(g, grads)
        return loss

    rng = np.random.default_rng(seed)
    for i in range(1, steps + 1):
        idx = train_i[rng.integers(0, len(train_i), batch_size)]
        loss = float(step(g, opt, batch(d, stats, idx, rng)))
        if i % max(1, steps // 4) == 0:
            print(f"    step {i:6d}  nll {loss:+.4f}", flush=True)
    return g


def predict(g, d, stats, idx, chunk=512):
    mu, sd = [], []
    fwd = nnx.jit(lambda m, b: m(b))
    for s in range(0, len(idx), chunk):
        b = batch(d, stats, idx[s:s + chunk])
        m_, lv = fwd(g, b)
        mu.append(np.asarray(m_))
        sd.append(np.exp(0.5 * np.asarray(lv)))
    return np.concatenate(mu), np.concatenate(sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deploy-only", action="store_true",
                    help="fit only the all-episode deploy model and write the pkl, "
                         "leaving an existing npz untouched -- use this when policies "
                         "have already been trained against that npz")
    args = ap.parse_args()

    # val_per_case=0: every episode is needed, because every sample has to get a
    # (mu, sigma) from a model that did not see it.  The folds ARE the holdout.
    d, _, stats = DATA.load(args.data, val_per_case=0)
    out = pathlib.Path(args.data) / f"_surprise_{DATA.IMG}_k{args.folds}.npz"

    if args.deploy_only:
        # One model on every episode, for inference.  The npz -- and so the
        # targets any already-trained policy was fitted to -- is NOT touched.
        print(f"  deploy model on all {len(np.unique(d['ep']))} episodes", flush=True)
        g = fit_fold(d, stats, np.arange(len(d["ep"])), args.steps, args.lr,
                     args.seed + 999, args.batch)
        dp = deploy_path(out)
        save_deploy(dp, g, stats)
        mu, sd = predict(g, d, stats, np.arange(len(d["ep"])))
        ft = d["ft_hist"][:, -1, :]
        r2 = 1.0 - np.mean((ft - mu) ** 2, axis=0) / np.maximum(np.var(ft, axis=0), 1e-12)
        print(f"\n  wrote {dp}")
        print(f"  IN-SAMPLE R^2 per axis: {np.round(r2, 3)}  -- the folds' "
              f"OUT-OF-FOLD numbers are the honest ones; this only confirms it fitted")
        return

    eps = np.unique(d["ep"])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(eps)
    folds = np.array_split(order, args.folds)
    mu = np.zeros((len(d["ep"]), DATA.FT), np.float32)
    sd = np.ones_like(mu)
    folds_fitted = []
    t0 = time.time()
    for f, held in enumerate(folds):
        is_held = np.isin(d["ep"], held)
        tr = np.flatnonzero(~is_held)
        te = np.flatnonzero(is_held)
        print(f"  fold {f + 1}/{args.folds}: {len(tr)} train / {len(te)} held-out frames "
              f"({len(eps) - len(held)} / {len(held)} episodes)", flush=True)
        g = fit_fold(d, stats, tr, args.steps, args.lr, args.seed + f, args.batch)
        mu[te], sd[te] = predict(g, d, stats, te)
        folds_fitted.append(g)

    ft = d["ft_hist"][:, -1, :]
    z = (ft - mu) / np.maximum(sd, 1e-3)
    np.savez(out, mu=mu, sigma=sd, ep=d["ep"], case=d["case"])
    # ... and the deploy ENSEMBLE: the same K fold models that produced the
    # targets above, averaged.  No extra fit, and no new data seen.
    save_deploy(deploy_path(out), folds_fitted, stats)
    # The number that says whether this is worth feeding to a policy: how much
    # of the wrench the scene explains, OUT OF SAMPLE.  Near 1.0 and the
    # residual is noise; near 0 and the expectation model is not working.
    var = np.var(ft, axis=0)
    r2 = 1.0 - np.mean((ft - mu) ** 2, axis=0) / np.maximum(var, 1e-12)
    print(f"\nwrote {out}  ({(time.time() - t0) / 60:.1f} min)")
    print(f"  out-of-fold R^2 per axis (u/v/n in world xyz): {np.round(r2, 3)}")
    print(f"  residual |z| mean {np.abs(z).mean():.2f}  p95 {np.percentile(np.abs(z), 95):.2f}")
    print(f"  sigma   mean {sd.mean(0).round(3)}  N  (bigger = the scene is less sure)")


if __name__ == "__main__":
    main()
