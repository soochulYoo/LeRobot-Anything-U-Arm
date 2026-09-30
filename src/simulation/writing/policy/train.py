"""Train a force-aware flow policy on the writing demonstrations.

Usage (from src/simulation/writing):
    python3 policy/train.py --case d --data demos/protocol_v1 --out policy/runs/cross_cond
    python3 policy/train.py --case a --data demos/protocol_v1 --out policy/runs/ft_input

Validation is OPEN LOOP on held-out demonstrations, in physical units:
    pos1 / posH   x_d error at the first / last step of the chunk (mm)
    level acc     predicted stiffness rounded to the nearest protocol level
                  (500 / 1000 / 3000 N/m), against the recorded level, xy and z
    ft            future contact-force error (N), where the model predicts it
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

LEVELS = np.array([500.0, 1000.0, 3000.0])


def save(path: pathlib.Path, model, cfg: M.Config, stats: DATA.Stats, extra: dict) -> None:
    state = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    with open(path, "wb") as f:
        pickle.dump(dict(params=jax.tree.map(np.asarray, state), config=dataclasses.asdict(cfg),
                         stats=stats.to_json(), extra=extra), f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    cfg = M.Config(**ck["config"])
    model = M.build(cfg)
    state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(state, ck["params"])
    nnx.update(model, state)
    return model, cfg, DATA.Stats.from_json(ck["stats"]), ck.get("extra", {})


def evaluate_open_loop(sample_fn, model, val: dict, stats: DATA.Stats, n: int = 1024, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(val["state"]), size=min(n, len(val["state"])), replace=False)
    b = {k: jnp.asarray(v) for k, v in DATA.batch(val, stats, idx).items()}
    a, z = sample_fn(model, b, jax.random.key(seed))
    a = stats.denorm("act", np.asarray(a))
    true = val["act"][idx]
    pos_err = 1000 * np.linalg.norm(a[..., :3] - true[..., :3], axis=-1)
    k_pred = np.exp(a[..., 3:])
    lv = np.abs(np.log(k_pred[..., None]) - np.log(LEVELS)).argmin(-1)      # (B, H, 3)
    lab = val["k_level"][idx]                                                # (B, H, 2)
    ok = lab[..., 0] >= 0
    out = dict(pos1_mm=float(pos_err[:, 0].mean()), posH_mm=float(pos_err[:, -1].mean()),
               logk_err=float(np.abs(a[..., 3:] - true[..., 3:]).mean()),
               acc_xy=float((lv[..., 0] == lab[..., 0])[ok].mean()) if ok.any() else float("nan"),
               acc_z=float((lv[..., 2] == lab[..., 1])[ok].mean()) if ok.any() else float("nan"))
    if z is not None:
        zf = stats.denorm("fut_ft", np.asarray(z))
        out["ft_N"] = float(np.sqrt(np.mean((zf - val["fut_ft"][idx]) ** 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=list(M.CASES), default="d")
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-goal", action="store_true")
    ap.add_argument("--val-per-case", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2000)
    args = ap.parse_args()

    dev = jax.devices()
    print(f"jax {jax.__version__} on {dev}")
    if dev[0].platform != "gpu":
        print("  NOTE: no GPU visible to JAX -- this runs on CPU (~40 ms/step). "
              "pip install -U 'jax[cuda12]' in this environment for the GPU.")
    out = pathlib.Path(args.out or f"policy/runs/{M.NAMES[args.case]}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, val, stats = DATA.load(args.data, val_per_case=args.val_per_case)
    print(f"data: {len(train['state'])} train / {len(val['state'])} val frames "
          f"({len(np.unique(train['ep']))} / {len(np.unique(val['ep']))} episodes) "
          f"in {time.time() - t0:.0f}s")

    cfg = M.Config(case=args.case, vision=not args.no_vision, goal=not args.no_goal)
    model = M.build(cfg, args.seed)
    warm = min(500, args.steps // 10)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warm, args.steps, args.lr * 0.05)
    opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=1e-4), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, opt, b, key):
        (loss, parts), grads = nnx.value_and_grad(lambda m: m.loss(b, key), has_aux=True)(model)
        opt.update(model, grads)
        return loss, parts

    @nnx.jit
    def sample(model, b, key):
        return model.sample(b, key)

    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    log = []
    t0 = time.time()
    run_loss = None
    for i in range(1, args.steps + 1):
        idx = rng.integers(0, len(train["state"]), args.batch)
        b = {k: jnp.asarray(v) for k, v in DATA.batch(train, stats, idx, rng).items()}
        key, kl = jax.random.split(key)
        loss, parts = train_step(model, opt, b, kl)
        run_loss = float(loss) if run_loss is None else 0.98 * run_loss + 0.02 * float(loss)
        if i % args.eval_every == 0 or i == args.steps:
            ev = evaluate_open_loop(sample, model, val, stats)
            row = dict(step=i, loss=run_loss, **{k: float(v) for k, v in parts.items()}, **ev,
                       minutes=(time.time() - t0) / 60)
            log.append(row)
            print(f"  [{M.NAMES[args.case]}] {i:6d}  loss {run_loss:.4f}  "
                  f"pos1 {ev['pos1_mm']:.2f} mm  posH {ev['posH_mm']:.2f} mm  "
                  f"level acc xy {100 * ev['acc_xy']:.1f}% z {100 * ev['acc_z']:.1f}%"
                  + (f"  ft {ev['ft_N']:.2f} N" if "ft_N" in ev else "")
                  + f"  ({row['minutes']:.1f} min)", flush=True)
            save(out / "model.pkl", model, cfg, stats, dict(log=log, args=vars(args)))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {out / 'model.pkl'}")


if __name__ == "__main__":
    main()
