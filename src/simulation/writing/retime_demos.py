"""Build a RATE-AUGMENTED demonstration set by replaying the originals faster.

S1 gives an exact relabeling for free motion -- resample the chunk in phase by
c and add 2 log c to log K -- and S3 says exactly where it stops being true: on
a rigid surface the contact force is k_n times the commanded depth, so scaling K
multiplies the force by c^2 and tears the paper, and S4 says the same scaling
also corrupts the policy's own observation of contact.  So the labels are NOT
relabeled here.

They do not have to be.  S3's retimer already replays a recorded demonstration
through the real controller on the real paper at rate c, and the contact force
that comes back is measured, not assumed.  This turns that into a data
generator: every demonstration is written again at several rates, with the
stiffness left exactly as demonstrated, and only the replays that still pass the
simulator's own ink, force-band and tear criteria are kept.

What that gives the policy is the one thing a relabeling cannot: episodes whose
forces, twists and spring deflections are what they really would be at that
speed.  The known bias is that only successful retimings survive, which is the
behaviour we want imitated rather than a hazard.

Each episode is written in the demonstration format (collect.Recorder), with
`retime_c` in the attributes and the policy-rate grid set to 10c Hz so one
action frame is still one DEMONSTRATION frame -- a retimed episode is, in phase,
the same length as the original.

Usage (one shard per array task):
    python3 retime_demos.py --data demos/protocol_v1 --out demos/protocol_v1_retimed \
        --c 1 1.5 2 2.5 3 4 --shard 0 --shards 16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def retime_one(sim, path, c, out_dir, log_hz=100.0, policy_hz=10.0) -> dict:
    """Replay one demonstration at rate c and write it as a new episode.

    The applied labels (x_d, K) are driven back through the controller exactly
    as collect.replay does, but the 100 Hz record is consumed c times faster.
    Interpolation is in the demonstration's OWN time, so the phase grid is
    untouched and only the clock that walks it changes.
    """
    import h5py
    import collect as CO
    import controller as C
    import sim as SM

    with h5py.File(path) as f:
        attrs = dict(f.attrs)
        spec = SM.TaskSpec(**json.loads(f.attrs["spec"]))
        t = f["full/t"][:].astype(float)
        xd = f["full/x_d"][:].astype(float)
        K = f["full/K"][:].astype(float)

    sim.reset(spec)
    sim.ctl.x_d = xd[0].copy()
    sim.ctl.K = K[0].copy()
    # the recorder samples on the PHYSICAL grid, so its rates carry the clock:
    # at c the same phase grid arrives c times sooner
    rec_er = CO.Recorder(sim, log_hz=log_hz * c, policy_hz=policy_hz * c, images=True)
    j, i = 0, 0
    while sim.t < (t[-1] - t[0]) / c:
        s = t[0] + (sim.t + sim.dt) * c              # demonstration phase
        while j + 1 < len(t) - 1 and t[j + 1] <= s:
            j += 1
        w = np.clip((s - t[j]) / max(t[j + 1] - t[j], 1e-9), 0.0, 1.0)
        x_next = xd[j] + w * (xd[j + 1] - xd[j])
        K_next = K[j] + w * (K[j + 1] - K[j])
        r = sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt,
                                     (K_next - sim.ctl.K) / sim.dt))
        # the teleoperator's own channels have no meaning in a replay; they are
        # filled so the file keeps the demonstration schema and nothing reads
        # them as if a hand had been there
        r.update(x_d_req=x_next, k_req=sim.k_diag(K_next), x_m=r["x_d"],
                 v_m=np.zeros(3), f_h=np.zeros(3), f_fb=np.zeros(3),
                 phase=-1, stroke=-1, k_level=CO.NO_LEVEL)
        rec_er.on_step(i, r, None)
        i += 1
    res = sim.score()
    if not res["success"]:
        return dict(ok=False, c=c, src=pathlib.Path(path).name, **{
            k: res[k] for k in ("coverage", "precision", "in_band", "peak_force",
                                "torn", "fail_reason")})
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{pathlib.Path(path).stem}_c{str(c).replace('.', 'p')}.h5"
    a = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in attrs.items()}
    a.update(retime_c=float(c), policy_hz=float(policy_hz * c), log_hz=float(log_hz * c),
             source=f"retimed from {pathlib.Path(path).name}", success=True,
             metrics=json.dumps({k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                                     float(v) if isinstance(v, (int, float, np.number)) else v)
                                 for k, v in res.items() if k != "checks"}))
    rec_er.save(out_dir / name, a)
    return dict(ok=True, c=c, src=pathlib.Path(path).name, out=name, **{
        k: res[k] for k in ("coverage", "precision", "in_band", "peak_force", "torn")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--out", default="demos/protocol_v1_retimed")
    ap.add_argument("--c", nargs="*", type=float, default=[1, 1.5, 2, 2.5, 3, 4])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()

    import sim as SM
    files = sorted(pathlib.Path(a.data).glob("*/ep_*.h5"))
    mine = files[a.shard::a.shards]
    print(f"shard {a.shard}/{a.shards}: {len(mine)} of {len(files)} demonstrations "
          f"x {len(a.c)} rates", flush=True)
    sim = SM.WritingSim(cameras=True)
    rows = []
    for p in mine:
        case = p.parent.name
        for c in a.c:
            r = retime_one(sim, p, float(c), pathlib.Path(a.out) / case)
            rows.append(r)
            print(f"  {p.stem:>10} c={c:<4g} {'kept' if r['ok'] else 'dropped'} "
                  f"cov {r['coverage']:.3f} band {r['in_band']:.3f} "
                  f"peak {r['peak_force']:5.2f} N"
                  + ("" if r["ok"] else f"  ({r['fail_reason']})"), flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"_manifest_{a.shard:03d}.json").write_text(json.dumps(rows, indent=1))
    kept = sum(r["ok"] for r in rows)
    print(f"\nkept {kept}/{len(rows)}")
    for c in a.c:
        g = [r for r in rows if r["c"] == c]
        print(f"  c={c:<5g} {sum(r['ok'] for r in g):>3}/{len(g)}")
    sim.close()


if __name__ == "__main__":
    main()
