"""Step 2: regress the local surface swing from what the robot can see.

The oracle sets K_R from the true swing; this asks how much of that an
estimator recovers from observations alone.  Target is in DEGREES, so the error
is reported in the same unit the oracle rule consumes and can be compared with
the only baseline that matters:

    predict the training mean  ->  2.49 deg rmse

Inputs are everything the recordings carry: both cameras, the contact force,
the joint torques and proprioception.  There is no six-axis F/T here, so the
contact moment -- the most direct evidence of how the pad is sitting -- reaches
the model only through `tau`.

Episodes are split, not frames: consecutive frames of one wipe are nearly
identical, so a frame split would let the model memorise and report an error it
has not earned.

    python3 train_surface.py data/surface.npz --epochs 30
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_vec, width=128):
        super().__init__()
        def cnn():
            return nn.Sequential(
                nn.Conv2d(3, 16, 8, 4), nn.GELU(),
                nn.Conv2d(16, 32, 4, 2), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2), nn.GELU(),
                nn.Flatten(), nn.LazyLinear(width), nn.GELU())
        self.wrist, self.top = cnn(), cnn()
        self.vec = nn.Sequential(nn.Linear(n_vec, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                  nn.Linear(width, 1))

    def forward(self, w, t, v):
        return self.head(torch.cat([self.wrist(w), self.top(t), self.vec(v)], -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/surface_reg.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    z = np.load(a.data)
    down = z["down"]
    ep = z["ep"][down]
    y = z["swing"][down].astype(np.float32)
    # 128 -> 64 by decimation: the policy pipeline sees 64, and it halves memory
    wr = z["wrist"][down][:, ::2, ::2].astype(np.float32) / 255.0
    tp = z["top"][down][:, ::2, ::2].astype(np.float32) / 255.0
    vec = np.concatenate([z[k][down].reshape(down.sum(), -1).astype(np.float32)
                          for k in ("f", "tau", "qpos", "qvel", "k", "tcp", "quat")]
                         + [z["kr"][down][:, None].astype(np.float32)], axis=1)

    eps = np.unique(ep)
    rng = np.random.default_rng(a.seed)
    val_eps = set(rng.choice(eps, size=max(1, int(a.val_frac * len(eps))), replace=False).tolist())
    is_val = np.isin(ep, list(val_eps))
    mu, sd = vec[~is_val].mean(0), vec[~is_val].std(0) + 1e-6
    vec = (vec - mu) / sd

    base = float(np.sqrt(np.mean((y[is_val] - y[~is_val].mean()) ** 2)))
    print(f"{len(y)} contact frames, {len(eps)} episodes "
          f"({len(val_eps)} held out, {is_val.sum()} frames)")
    print(f"predict-the-mean baseline on val: {base:.2f} deg rmse")

    T = lambda x: torch.as_tensor(x)
    tr = [T(wr[~is_val]).permute(0, 3, 1, 2), T(tp[~is_val]).permute(0, 3, 1, 2),
          T(vec[~is_val]), T(y[~is_val])]
    va = [T(wr[is_val]).permute(0, 3, 1, 2), T(tp[is_val]).permute(0, 3, 1, 2),
          T(vec[is_val]), T(y[is_val])]

    net = Net(vec.shape[1]).to(dev)
    net(tr[0][:2].to(dev), tr[1][:2].to(dev), tr[2][:2].to(dev))     # build lazy layers
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    n = len(tr[3])
    best = float("inf")
    for e in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            j = perm[i:i + a.batch]
            w, t, v, yy = (x[j].to(dev, non_blocking=True) for x in tr)
            loss = nn.functional.mse_loss(net(w, t, v), yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        sched.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(*(x[i:i + 512].to(dev) for x in va[:3]))
                              for i in range(0, len(va[3]), 512)]).cpu()
        rmse = float(torch.sqrt(torch.mean((pred - va[3]) ** 2)))
        mae = float(torch.mean(torch.abs(pred - va[3])))
        best = min(best, rmse)
        print(f"  epoch {e:3d}  train mse {tot/n:7.3f}   val rmse {rmse:5.2f} deg"
              f"   mae {mae:5.2f}   (baseline {base:.2f})", flush=True)
    print(f"\nbest val rmse {best:.2f} deg against a {base:.2f} deg baseline"
          f"  ->  {100*(1-best/base):.0f}% of the variance in the target explained")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=net.state_dict(), mu=mu, sd=sd, val_eps=sorted(val_eps)), out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
