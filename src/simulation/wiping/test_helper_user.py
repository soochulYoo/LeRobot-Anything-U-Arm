"""Drive `helper_user.HelperUser` and `protocol.Arms` without SAPIEN.

    python3 test_helper_user.py

WHY THIS FILE EXISTS.  The helper path can only be exercised on a machine with the
simulator, which is the local one, which is where a mistake costs a collection session.
The last one cost a session: the loop handed the helper `sim.last`, the leader quantities
live on the TeleopSession instead, and `x_m` was missing -- a KeyError on the first step
of the first episode.  Everything between the record and the model is plain numpy, so a
stub `sim` with the four methods HelperUser touches runs the whole path here: observe ->
model -> snap to levels -> the axes the helper owns replaced in the table's triple.

What it does NOT cover: the record really having these keys (`protocol.py` builds it) and
the viewer.  The record is checked against the simulator's own source by the attribute
scan in the session notes; the viewer is checked by running one episode.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LADDERS = dict(xy=(500.0, 1000.0, 3000.0), z=(300.0, 600.0, 1500.0), kr=(0.3, 3.0, 30.0))


class Levels:
    xy, z, kr = LADDERS["xy"], LADDERS["z"], LADDERS["kr"]


class Table:
    """KeyboardLevels' three methods, with the levels under test's control."""

    def __init__(self, level=(1, 1, 1)):
        self.level = list(level)
        self.polls = 0

    def reset(self):
        self.level = [1, 1, 1]

    def poll(self, t, group):
        self.polls += 1
        return False


class StubSim:
    """The four things HelperUser asks of a simulator."""

    W = np.eye(3)

    def observe(self, images: bool = True) -> dict:
        return {"rgb_top_camera": np.zeros((128, 128, 3), np.uint8),
                "rgb_wrist_camera": np.zeros((128, 128, 3), np.uint8)}

    @staticmethod
    def k_diag(K) -> np.ndarray:
        return np.diag(np.asarray(K)).copy()


def record(f_z: float = 3.0, kr: float = 3.0) -> dict:
    """The keys HelperUser reads out of the step record, and nothing else."""
    return dict(p=np.array([0.1, 0.2, 0.3]), R=np.eye(3), v=np.array([0.02, 0.0, 0.0]),
                x_d=np.array([0.1, 0.2, 0.3]), x_m=np.array([0.0, 0.0, 0.0]),
                f_filt=np.array([0.3, 0.1, f_z]), kr=np.float32(kr),
                K=np.diag([1000.0, 1000.0, 600.0]))


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def main() -> int:
    import helper_user as HU
    bad = 0

    # ---- v0 is a rule and loads with no checkpoint ---------------------------
    tab = Table()
    u = HU.HelperUser(StubSim(), None, Levels, tab, version="v0", axes=("r",))
    bad += not check("v0 loads without a checkpoint", u.version == "v0", f"ckpt={u.ckpt}")
    u.reset()
    bad += not check("reset resets the table too", tab.level == [1, 1, 1])

    # ---- the whole observe -> poll path, which is where the KeyError was -----
    try:
        u.observe(0.0, record(f_z=0.0))          # in the air
        u.poll(0.0, "approach")
        air = list(u.level)
        u.observe(1.0, record(f_z=3.0))          # in contact
        u.poll(1.0, "contact")
        con = list(u.level)
        ok = True
    except Exception as e:                        # the bug this file exists for
        ok, air, con = False, None, repr(e)
    bad += not check("observe -> poll runs on a step record", ok, f"air {air} contact {con}")
    if ok:
        bad += not check("only the owned axis moves",
                         air[:2] == [1, 1] and con[:2] == [1, 1],
                         "xy and z stayed with the table")
        bad += not check("v0's K_R is a near-constant on this rig",
                         con[2] == 0,
                         "no moment channel -> the rotational rule pins low; "
                         "a v0 arm is a CONSTANT baseline, not a reacting rule")
        bad += not check("the record it logs says which helper drove it",
                         u.record()["arm" if "arm" in u.record() else "helper"] == "v0"
                         and u.record()["axes"] == ["r"])

    # ---- the helper runs at its own rate, not the loop's --------------------
    u.reset()
    n0 = u.helper.n_calls
    for i in range(100):                          # 100 control steps at 2 ms = 0.2 s
        u.observe(i * 0.002, record())
    rate = u.helper.n_calls - n0
    bad += not check("10 Hz over 0.2 s of control steps", rate in (2, 3), f"{rate} calls")

    # ---- the real checkpoints, if they are here -------------------------------
    # The pre-flight for a session: both generations load through this path and they
    # actually command DIFFERENT K_R.  Two arms that agree everywhere are not an
    # experiment, and finding that out after an hour of hand-collection is expensive.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-v0", default=None)
    ap.add_argument("--ckpt-v1", default=None)
    a = ap.parse_args()
    ck = {k: v for k, v in (("v0", a.ckpt_v0), ("v1", a.ckpt_v1)) if v}
    if ck:
        traces = {}
        for name, path in ck.items():
            h = HU.HelperUser(StubSim(), path, Levels, Table(), version=name, axes=("r",))
            h.reset()
            tr = []
            for j, fz in enumerate((0.0, 0.0, 1.0, 3.0, 3.0, 5.0, 3.0, 0.5)):
                h.observe(j * 0.2, record(f_z=fz, kr=Levels.kr[h.level[2]]))
                h.poll(j * 0.2, "contact" if fz > 1.0 else "approach")
                tr.append(h.level[2])
            traces[name] = tr
            bad += not check(f"{name} checkpoint loads and drives K_R", True,
                             f"levels {tr}")
        if len(traces) == 2:
            t0, t1 = traces["v0"], traces["v1"]
            bad += not check("the two arms are distinguishable", t0 != t1,
                             "v0 and v1 command the SAME K_R on this trace -- an arm "
                             "comparison would measure nothing" if t0 == t1
                             else f"differ on {sum(x != y for x, y in zip(t0, t1))}/8 steps")
    else:
        print("  --    no checkpoints given; pass --ckpt-v0/--ckpt-v1 to pre-flight them")

    # ---- the arm schedule ---------------------------------------------------
    src = (HERE / "protocol.py").read_text()
    ns = {"np": np}
    exec(src[src.index("class Arms:"):src.index("_HELPERS: dict")], ns)
    A = ns["Arms"](["manual", "v0", "v1"], 0)
    seq = [A.next() for _ in range(9)]
    bad += not check("balanced in blocks",
                     all(sorted(seq[3 * b:3 * b + 3]) == ["manual", "v0", "v1"]
                         for b in range(3)), " ".join(seq))
    A2 = ns["Arms"](["manual", "v0", "v1"], 0)
    a = A2.next()
    A2.undo(a)
    bad += not check("an abort gives the arm back", A2.next() == a)
    bad += not check("the schedule is reproducible",
                     [ns["Arms"](["manual", "v0", "v1"], 0).next() for _ in range(3)]
                     == [seq[0]] * 3)

    print(f"\n  {'all checks passed' if not bad else f'{bad} FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
