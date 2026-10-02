"""Train a force-aware flow policy on the writing demonstrations.

Usage (from src/simulation/writing):
    python3 policy/train.py --case d --data demos/protocol_v1 --out policy/runs/cross_cond
    python3 policy/train.py --case a --data demos/protocol_v1 --out policy/runs/ft_input

Validation is OPEN LOOP on held-out demonstrations, in physical units:
    pos1 / posH   x_d error at the first / last step of the chunk (mm)
    level acc     predicted stiffness rounded to the nearest protocol level
                  (500 / 1000 / 3000 N/m), against the recorded level, xy and z
    ft            future contact-force error (N), where the model predicts it
    val           the flow loss on those same held-out frames, next to the
                  training loss -- 150 episodes is little data, so the gap matters
Every --log-every steps a cheaper line reports the loss and the GRADIENT NORM
(mean and max since the last line, before any clipping), which is where an
unstable structure shows itself first.
Closed-loop success is policy/rollout.py's job.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import pickle
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402

import data as DATA  # noqa: E402
import model as M  # noqa: E402

LEVELS = np.array([500.0, 1000.0, 3000.0])     # writing's, for both axis groups


def dataset_levels(root) -> dict:
    """The protocol's stiffness levels, from the demonstrations themselves.

    The level-accuracy metric below asks which of three levels the policy
    commanded, and the three are not the same numbers on every task: wiping
    presses with a softer normal (300/600/1500) and adds a rotational group
    (0.3/3/30).  Scoring a wiping policy against writing's ladder reports a
    miss every time it is right.  Every episode stores its own `protocol`.
    """
    import h5py
    lv = {"xy": LEVELS, "z": LEVELS, "kr": None}
    files = sorted(pathlib.Path(root).glob("*/ep_*.h5"))
    if files:
        with h5py.File(files[0]) as f:
            p = json.loads(f.attrs["protocol"]) if "protocol" in f.attrs else {}
        for k, v in (p.get("levels") or {}).items():
            if k in lv:
                lv[k] = np.asarray(v, dtype=float)
    return lv


def save(path: pathlib.Path, model, cfg: M.Config, stats: DATA.Stats, extra: dict) -> None:
    state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    with open(path, "wb") as f:
        pickle.dump(dict(params=jax.tree.map(np.asarray, state), config=dataclasses.asdict(cfg),
                         stats=stats.to_json(), extra=extra), f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    cfg = M.Config(**ck["config"])
    # the stats go in because the robustness term is stated in physical units,
    # so a model trained with one cannot be rebuilt without them
    stats = DATA.Stats.from_json(ck["stats"])
    model = M.build(cfg, stats=stats)
    state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(state, ck["params"])
    nnx.update(model, state)
    return model, cfg, stats, ck.get("extra", {})


def evaluate_open_loop(sample_fn, model, val: dict, stats: DATA.Stats, n: int = 1024,
                       seed: int = 0, levels: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val["state"]), size=min(n, len(val["state"])), replace=False)
    b = {k: jnp.asarray(v) for k, v in DATA.batch(
        val, stats, idx, force=getattr(model.cfg, "force", True),
        force_target=getattr(model.cfg, "force_target", True)).items()}
    a, z = sample_fn(model, b, jax.random.key(seed))
    a = stats.denorm("act", np.asarray(a))
    true = val["act"][idx]
    pos_err = 1000 * np.linalg.norm(a[..., :3] - true[..., :3], axis=-1)
    # Where the stiffness columns start: 3 under the legacy layout, 6 under
    # spring, where f_d sits in between.  Reading this off the layout rather
    # than assuming 3 is what keeps acc_z from silently scoring the wrong
    # column -- it read 0.0% on the first spring run because of exactly that.
    k0 = 6 if getattr(model.cfg, "layout", "legacy") == "spring" else 3
    L = levels or {"xy": LEVELS, "z": LEVELS, "kr": None}
    lab = val["k_level"][idx]                                                # (B, H, 2 or 3)
    ok = lab[..., 0] >= 0

    def level_of(col, ladder):
        """Which of the three levels the commanded stiffness is nearest, in log."""
        return np.abs(np.log(np.exp(a[..., col])[..., None]) - np.log(ladder)).argmin(-1)

    def acc(col, ladder, j):
        if ladder is None or lab.shape[-1] <= j or not ok.any():
            return float("nan")
        return float((level_of(col, ladder) == lab[..., j])[ok].mean())

    out = dict(pos1_mm=float(pos_err[:, 0].mean()), posH_mm=float(pos_err[:, -1].mean()),
               logk_err=float(np.abs(a[..., k0:k0 + 3] - true[..., k0:k0 + 3]).mean()),
               acc_xy=acc(k0, L["xy"], 0), acc_z=acc(k0 + 2, L["z"], 1))
    if k0 == 6:
        # The headline number for this layout: f_d is the force the policy
        # COMMANDS, in newtons, so its error is a force error directly rather
        # than a stiffness error that has to be multiplied by a gap to become
        # one.  Comparable with ft_N, which is the same quantity predicted by a
        # separate head.
        out["fd_N"] = float(np.sqrt(np.mean((a[..., 3:6] - true[..., 3:6]) ** 2)))
        out["fd_n_N"] = float(np.sqrt(np.mean((a[..., 5] - true[..., 5]) ** 2)))
    if a.shape[-1] > k0 + 3:     # the rotational stiffness is part of the action
        out["logkr_err"] = float(np.abs(a[..., k0 + 3] - true[..., k0 + 3]).mean())
        out["acc_kr"] = acc(k0 + 3, L["kr"], 2)
    if z is not None:
        zf = stats.denorm("fut_ft", np.asarray(z))
        out["ft_N"] = float(np.sqrt(np.mean((zf - val["fut_ft"][idx]) ** 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(M.NAMES), default="d",
                    help="a-d are the CoFA structures; e (film) is act-only, "
                         "where the wrench reaches the decoder through adaLN-Zero "
                         "instead of as a token")
    ap.add_argument("--arch", choices=["flow", "act"], default="flow",
                    help="flow: the CoFA rectified-flow fields.  "
                         "act: Comp-ACT's CVAE transformer (case a is Comp-ACT itself)")
    ap.add_argument("--match-params", dest="match", action="store_true", default=None,
                    help="equalise parameter counts across the structures of a family")
    ap.add_argument("--no-match-params", dest="match", action="store_false",
                    help="leave every structure at its natural size")
    ap.add_argument("--hidden", type=int, default=512, help="act transformer width")
    ap.add_argument("--backbone", choices=["cnn", "resnet18"], default=None,
                    help="camera encoder for the flow family.  resnet18 is what the "
                         "CoFA figure draws; it costs 22M more parameters, which on "
                         "135 demonstrations is a lot of room to memorise in")
    ap.add_argument("--film", action="store_true",
                    help="adaLN-Zero: the wrench scales and shifts the decoder features")
    ap.add_argument("--ft-tokens", action="store_true",
                    help="the measured F/T history enters as 10 tokens with frame "
                         "positions instead of one GRU summary")
    ap.add_argument("--moe", type=int, default=None, help="experts in the action field (0 = off)")
    ap.add_argument("--ft-latent", type=int, default=None,
                    help="width of the future-F/T latent the wrench flow runs on (0 = raw newtons)")
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 3e-4 for flow, 1e-5 for act (Comp-ACT's value)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vision", action="store_true")
    # THE CONTROL FOR THE EIGHT-VARIANT RANKING.  See data.features.
    #   --no-force         zero the force INPUT (b/c/d keep their force target)
    #   --no-force-target  zero the future-force SUPERVISION too, so the extra
    #                      machinery of b/c/d has nothing to predict and the
    #                      comparison is purely architectural
    # If flow a still beats c by ~40 pp with both off, the ranking in
    # policy/runs/COMPARISON.md was never about force handling.
    ap.add_argument("--no-force", action="store_true")
    # (d) only: steps the wrench field runs ahead of the action field at
    # inference.  0 is the synchronous clock the first runs used, which is the
    # measure-zero diagonal of (d)'s own training distribution -- see
    # model.Config.d_lead.
    ap.add_argument("--d-lead", type=int, default=0)
    ap.add_argument("--no-force-target", action="store_true")
    ap.add_argument("--no-goal", action="store_true",
                    help="drop the 32-point target path.  The cameras alone cannot carry "
                         "it: the glyph is 8x8 px at 64x64, so the 3 mm tolerance is "
                         "0.67 px and every structure scores ~0% (policy/runs_nogoal)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="linear warmup steps; default 1000 for act "
                         "(Comp-ACT's lr_warmup_steps), 500 for flow")
    ap.add_argument("--val-per-case", type=int, default=5)
    # The action layout, orthogonal to --case (data.py LAYOUTS, model.Config):
    #   legacy  x_d + log K          -- the labels the first eight runs used
    #   spring  x_ref + f_d + log K  -- the controller places the target, so a
    #                                   stiffness error stops being a force error
    ap.add_argument("--layout", choices=list(DATA.LAYOUTS), default="legacy")
    # K's only gradient under --layout spring; 0 leaves K unsupervised, which is
    # the honest baseline to compare against.  See model.Config.w_robust.
    ap.add_argument("--w-robust", type=float, default=0.0)
    ap.add_argument("--robust-delta", type=float, default=0.002,
                    help="m of surface deviation the force must stay in band under")
    # Force as surprise: a _surprise_*.npz from policy/surprise.py widens
    # ft_hist to [measured | expected | z], so the policy sees the part of the
    # wrench the scene did not predict rather than only the raw newtons.
    ap.add_argument("--surprise", default=None,
                    help="path to a _surprise_*.npz (policy/surprise.py)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--clip", type=float, default=None,
                    help="clip gradients to this global norm (0 = off); "
                         "default 0 for flow, 10 for act, which is Comp-ACT's value")
    args = ap.parse_args()
    # Comp-ACT trains at a lower learning rate and clips; the flow family does not.
    if args.lr is None:
        args.lr = 1e-5 if args.arch == "act" else 3e-4      # Comp-ACT's value
    if args.clip is None:
        args.clip = 10.0 if args.arch == "act" else 0.0
    if args.warmup is None:
        args.warmup = 1000 if args.arch == "act" else 500
    if args.backbone is None:
        args.backbone = "resnet18" if args.arch == "act" else "cnn"
    if args.match is None:
        # the act family stays at Comp-ACT's published dimensions, so only the
        # flow family gets widths equalised across structures
        args.match = args.arch == "flow"

    dev = jax.devices()
    print(f"jax {jax.__version__} on {dev}")
    if dev[0].platform != "gpu":
        print("  NOTE: no GPU visible to JAX -- this runs on CPU (~40 ms/step). "
              "pip install -U 'jax[cuda12]' in this environment for the GPU.")
    # The action and state widths come from the DATA, not from data.py's
    # constants: a wiping dataset carries a rotational stiffness in both, and
    # a writing one does not.  Read before the model is built, and before
    # --match-params equalises the budgets against them.
    a_dim, s_dim = DATA.dims(args.data, args.layout)
    cfg = M.Config(case=args.case, arch=args.arch, hidden=args.hidden, A=a_dim, S=s_dim,
                   backbone=args.backbone, vision=not args.no_vision, goal=not args.no_goal,
                   layout=args.layout, w_robust=args.w_robust,
                   robust_delta=args.robust_delta,
                   force=not args.no_force, force_target=not args.no_force_target,
                   d_lead=args.d_lead,
                   ft_in=DATA.FT_SUR if args.surprise else DATA.FT)
    if args.no_force and args.surprise:
        raise SystemExit("--no-force with --surprise is contradictory: the surprise "
                         "channels ARE force.  Drop one.")
    if args.w_robust > 0.0 and not DATA.is_spring(args.layout):
        raise SystemExit("--w-robust needs --layout spring: the term is stated in "
                         "f_d and K, and the legacy layout has no f_d")
    if args.film or args.case == "e":
        cfg = dataclasses.replace(cfg, film=True)
    if args.ft_tokens:
        cfg = dataclasses.replace(cfg, ft_tokens=True)
    if args.moe is not None:
        cfg = dataclasses.replace(cfg, moe=args.moe)
    if args.ft_latent is not None:
        cfg = dataclasses.replace(cfg, ft_latent=args.ft_latent)
    if args.match:
        # every structure of a family gets the same parameter budget, so a
        # ranking cannot be a capacity gap in disguise
        if args.arch == "act":
            import act_model as AM
            cfg = dataclasses.replace(cfg, hidden=AM.match_hidden(cfg))
        else:
            cfg = dataclasses.replace(cfg, D=M.match_width(cfg))
    out = pathlib.Path(args.out or f"policy/runs/{M.run_name(cfg)}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    levels = dataset_levels(args.data)
    train, val, stats = DATA.load(args.data, val_per_case=args.val_per_case,
                                  layout=args.layout, surprise=args.surprise)
    print(f"data: {len(train['state'])} train / {len(val['state'])} val frames "
          f"({len(np.unique(train['ep']))} / {len(np.unique(val['ep']))} episodes) "
          f"in {time.time() - t0:.0f}s")

    model = M.build(cfg, args.seed, stats=stats)
    warm = min(args.warmup, args.steps // 10)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warm, args.steps, args.lr * 0.05)
    tx = optax.adamw(sched, weight_decay=1e-4)
    if args.clip > 0:
        tx = optax.chain(optax.clip_by_global_norm(args.clip), tx)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
    print(f"{M.n_params(model) / 1e6:.2f}M parameters")
    try:
        print(M.describe(cfg), flush=True)
    except Exception as e:          # a printout must never cost a training run
        print(f"  (could not describe this configuration: {e!r})", flush=True)

    @nnx.jit
    def train_step(model, opt, b, key):
        (loss, parts), grads = nnx.value_and_grad(lambda m: m.loss(b, key), has_aux=True)(model)
        gnorm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
        opt.update(model, grads)
        M.ema_update(model, cfg.ema)     # the future-F/T target encoder trails by one step
        return loss, parts, gnorm

    @nnx.jit
    def val_loss(model, b, key):
        return model.loss(b, key)[0]

    @nnx.jit
    def sample(model, b, key):
        return model.sample(b, key)

    # A fixed held-out batch (no augmentation) so the validation loss is
    # comparable across steps, structures and seeds.
    vrng = np.random.default_rng(0)
    vidx = vrng.choice(len(val["state"]), size=min(1024, len(val["state"])), replace=False)
    vb = {k: jnp.asarray(v) for k, v in DATA.batch(
        val, stats, vidx, force=cfg.force, force_target=cfg.force_target).items()}

    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    log = []
    t0 = time.time()
    run_loss, gn = None, []
    for i in range(1, args.steps + 1):
        idx = rng.integers(0, len(train["state"]), args.batch)
        b = {k: jnp.asarray(v) for k, v in DATA.batch(
            train, stats, idx, rng, force=cfg.force,
            force_target=cfg.force_target).items()}
        key, kl = jax.random.split(key)
        loss, parts, gnorm = train_step(model, opt, b, kl)
        loss, gnorm = float(loss), float(gnorm)
        if not np.isfinite(loss) or not np.isfinite(gnorm):
            print(f"  [{M.run_name(cfg)}] step {i}: loss {loss} grad norm {gnorm} -- stopping. "
                  f"Last finite loss {run_loss}. Try --clip 1.0 or a smaller --lr.", flush=True)
            raise SystemExit(2)
        run_loss = loss if run_loss is None else 0.98 * run_loss + 0.02 * loss
        gn.append(gnorm)
        if i % args.log_every == 0:
            print(f"  [{M.run_name(cfg)}] {i:6d}  loss {run_loss:.4f}  "
                  f"grad norm mean {np.mean(gn):.3f} max {np.max(gn):.3f}  "
                  f"{1000 * (time.time() - t0) / i:.0f} ms/step", flush=True)
            gn.clear()
        if i % args.eval_every == 0 or i == args.steps:
            ev = evaluate_open_loop(sample, model, val, stats, levels=levels)
            row = dict(step=i, loss=run_loss, val_loss=float(val_loss(model, vb, jax.random.key(0))),
                       grad_norm=gnorm, **{k: float(v) for k, v in parts.items()}, **ev,
                       minutes=(time.time() - t0) / 60)
            log.append(row)
            print(f"  [{M.run_name(cfg)}] {i:6d}  loss {run_loss:.4f}  "
                  f"pos1 {ev['pos1_mm']:.2f} mm  posH {ev['posH_mm']:.2f} mm  "
                  f"level acc xy {100 * ev['acc_xy']:.1f}% z {100 * ev['acc_z']:.1f}%"
                  + (f" K_R {100 * ev['acc_kr']:.1f}%" if "acc_kr" in ev else "")
                  + (f"  f_d {ev['fd_n_N']:.2f} N" if "fd_n_N" in ev else "")
                  + (f"  ft {ev['ft_N']:.2f} N" if "ft_N" in ev else "")
                  + f"  val {row['val_loss']:.4f}  ({row['minutes']:.1f} min)", flush=True)
            save(out / "model.pkl", model, cfg, stats, dict(log=log, args=vars(args)))
            # written every eval, so a dependent job or a watcher can read progress
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out / 'model.pkl'}")


if __name__ == "__main__":
    main()
