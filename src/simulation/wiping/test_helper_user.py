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

import ast
import pathlib
import time
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

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

    # ---- the console tap: what the SIMULATOR sends to the page -----------------
    # The mapping the operator actually watches -- sim camera to ui camera, sim force to
    # ui force, sim stiffness to ui stiffness -- and it cannot be tried here because the
    # simulator needs SAPIEN.  So `Console` is driven against a stub sim instead, which
    # is enough: everything between `sim.last` and the wire is plain numpy.
    import io
    import json as _json

    class TelSim:
        """The six things protocol.Console asks of a simulator."""

        class _Ctl:
            kr = 3.0
            K = np.diag([1000.0, 1000.0, 600.0])

        class _Crit:
            force_band = (1.0, 6.0)

        t = 1.234
        ctl, crit = _Ctl(), _Crit()
        gone = np.zeros(14, dtype=bool)
        last = dict(f_filt=np.array([0.3, 0.4, 3.0]), f_sensor_n=3.0, f_n=2.8,
                    pen_down=True)

        def k_diag(self, K=None):
            return np.array([1000.0, 1000.0, 600.0])

        def observe(self, images=True):
            # a float frame scaled 0-1, the case a straight cast would turn black
            return {"rgb_top_camera": np.ones((1, 32, 32, 3), np.float32) * 0.5,
                    "rgb_wrist_camera": np.zeros((32, 32, 3), np.uint8)}

    class _Lvl:
        level = [1, 1, 0]

    src_c = (HERE / "protocol.py").read_text()
    ns_c = {"np": np, "SM": None}
    WHO_SRC = src_c[src_c.index("WHO = ("):src_c.index("class WipeRecorder(")]
    exec(WHO_SRC, ns_c)
    exec(src_c[src_c.index("class Console:"):src_c.index("class Arms:")], ns_c)
    out = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(out):
        con = ns_c["Console"].__new__(ns_c["Console"])
        from stiffness_helper.console import wire as W
        con.w, con.cmds = W, None
        con.new_episode()
        con.tel(TelSim(), _Lvl(), np.array([3000.0, 3000.0, 1500.0]), 30.0, 0.98)
        con.frame(TelSim())
    evs = [_json.loads(l) for l in out.getvalue().splitlines() if l.startswith("{")]
    tel = next(e for e in evs if e["ev"] == "tel")
    frames = [e for e in evs if e["ev"] == "frame"]
    # |f_t| = sqrt(|f|^2 - f_n^2) = sqrt(0.3^2 + 0.4^2) = 0.5 for this wrench
    bad += not check("sim force -> ui force", abs(tel["f_n"] - 2.8) < 1e-6
                     and abs(tel["f_t"] - 0.5) < 1e-3,
                     f"normal {tel['f_n']} N, sliding {tel['f_t']} N")
    bad += not check("sim stiffness -> ui stiffness is the APPLIED one",
                     tel["k"] == [1000.0, 600.0, 3.0],
                     "k_diag and ctl.kr, after the ramp and the tank")
    bad += not check("and what was ASKED for goes too, so the gap is visible",
                     tel["k_req"] == [3000.0, 1500.0, 30.0],
                     "the tank refusing a stiffening is the thing worth seeing")
    bad += not check("and whose level each axis has, with where the person's stick stands",
                     tel.get("who") == dict(human=[-1, -1, -1], table=[1, 1, 0],
                                            source=[0, 0, 0]), f"{tel.get('who')}")
    bad += not check("contact and progress reach the page",
                     tel["down"] is True and tel["left"] == 14)
    bad += not check("sim camera -> ui camera, both of them",
                     [f["cam"] for f in frames] == ["top", "side"])
    bad += not check("  ... and a float frame is not cast to black",
                     all(len(f["jpg"]) > 100 for f in frames)
                     and __import__("base64").b64decode(frames[0]["jpg"])[:3].hex()
                     == "ffd8ff")

    # ---- arbitration on a contested axis ---------------------------------------
    # Only reachable with a window, so it is driven here instead.  The rule under test
    # is NOT about accuracy: K is not identifiable from (x, F), so at the instant of a
    # disagreement nothing decides it.  The person wins, the model's opinion is kept
    # beside theirs, and the hindsight labeler scores it later.
    class _Lv:
        def __init__(self, lv): self.level = list(lv)
        def reset(self): pass
        def poll(self, t, g): return False

    class _Tab:
        axes = (2,)
        def __init__(self):
            self.own, self.auto = _Lv([1, 1, 1]), _Lv([1, 1, 1])
            self.own_pressed, self.level = False, [1, 1, 1]
        def reset(self): pass
        def poll(self, t, g):
            self.level = list(self.auto.level)
            self.level[2] = self.own.level[2]
            return False

    tab = _Tab()
    arb = HU.HelperUser(StubSim(), None, Levels, tab, version="v0", axes=("r",))
    arb.reset()
    arb._own_prev = list(tab.own.level)
    arb.helper.level = [1, 1, 0]                 # the model wants K_R low
    for t in (0.0, 0.2, 0.4, 0.6):
        arb.poll(t, "contact")
    bad += not check("the model drives the axis it owns", arb.level[2] == 0)
    # WHO WANTED WHAT, beside the level that won.  The applied level alone cannot be
    # taken apart afterwards, so each step records all three and whose was applied.
    o = arb.opinions()
    bad += not check("each step records whose level an axis has",
                     list(o["k_source"]) == [arb.TABLE, arb.TABLE, arb.MODEL],
                     f"{list(o['k_source'])}")
    bad += not check("  ... with the model's level, the person's and the table's",
                     list(o["k_level_model"]) == [1, 1, 0]
                     and list(o["k_level_human"]) == [1, 1, 1]
                     and list(o["k_level_table"]) == [1, 1, 1])
    bad += not check("  ... and a model that has not run has said nothing",
                     bool(np.isnan(o["k_norm_model"]).all()),
                     "its default rung is not an output")
    bad += not check("a standing split with the table is one event, not one per tick",
                     sum(e["code"] == "shadow_disagree" for e in arb.events) == 1)
    tab.own.level[2] = 2                          # the person presses HIGH
    arb.poll(1.0, "contact")
    bad += not check("a press takes the axis back", arb.level[2] == 2)
    o = arb.opinions()
    bad += not check("  ... which the step records as the person's, at their level",
                     o["k_source"][2] == arb.PERSON and o["k_level_human"][2] == 2)
    bad += not check("  ... with what the model went on wanting beside it",
                     o["k_level_model"][2] == 0,
                     "the event says when they parted; this says for how long")
    tk = [e for e in arb.events if e["code"] == "takeover"]
    bad += not check("the takeover keeps what the MODEL wanted beside it",
                     len(tk) == 1 and tk[0]["human"] == 2 and tk[0]["model"] == 0
                     and "k_norm" in tk[0],
                     "it says the helper was wrong, not what right would have been")
    arb.poll(2.0, "contact")
    bad += not check("  ... and holds it while they are still working", arb.level[2] == 2)
    arb.poll(1.0 + arb.HOLD + 0.1, "contact")
    bad += not check(f"  ... and gives it back after {arb.HOLD} s of silence",
                     arb.level[2] == 0)
    o = arb.opinions()
    bad += not check("  ... when the step is the model's again, and the person's level "
                     "is where they left it",
                     o["k_source"][2] == arb.MODEL and o["k_level_human"][2] == 2)
    r = arb.record()
    bad += not check("the counts reach the episode record",
                     r["takeovers"] == 1 and r["disagreements"] == 1
                     and len(r["events"]) >= 2)

    # ---- the output applied as it is, not snapped --------------------------------
    # `--helper-output continuous`.  The rung is what every arm shares, so it stays the
    # default; this is the other thing a helper's number can be used for, and it only
    # ever applies to an axis the MODEL is driving at that step.
    class _Lad:
        xy, z, kr = LADDERS["xy"], LADDERS["z"], LADDERS["kr"]
        def k(self, lv): return np.array([self.xy[lv[0]], self.xy[lv[0]], self.z[lv[1]]])
        def k_r(self, lv): return float(self.kr[lv[2]])

    def _driven(output):
        t3 = _Tab()
        u = HU.HelperUser(StubSim(), None, Levels, t3, version="v0",
                          axes=("t", "n", "r"), output=output)
        u.reset()
        u._own_prev = list(t3.own.level)
        return u, t3

    cont, tab3 = _driven("continuous")
    cont.poll(0.0, "contact")
    k0, kr0 = cont.stiffness(_Lad())
    bad += not check("a model that has not run yet is given the rung, not a guess",
                     np.allclose(k0, [1000.0, 1000.0, 600.0]) and kr0 == 3.0,
                     f"{k0} {kr0}")
    cont.helper.k_norm, cont.helper.level = np.array([0.30, 0.60, 0.45]), [0, 1, 1]
    cont.helper.n_calls = 1
    cont.poll(0.1, "contact")
    want = cont.helper.ks.expand(cont.helper.k_norm)
    k1, kr1 = cont.stiffness(_Lad())
    bad += not check("continuous: the robot is given the model's own number",
                     np.allclose(k1, [want[0], want[0], want[1]]) and abs(kr1 - want[2]) < 1e-9
                     and not np.isclose(k1[0], 500.0) and not np.isclose(k1[2], 600.0),
                     f"asked {np.round(k1, 1)} / {kr1:.2f}, rungs would be 500, 600, 3")
    bad += not check("  ... while the level it reports is still the nearest rung",
                     cont.level == [0, 1, 1] and cont.record()["output"] == "continuous")
    tab3.own.level[2] = 2                         # the person presses K_R high
    cont.poll(0.2, "contact")
    k2, kr2 = cont.stiffness(_Lad())
    bad += not check("  ... and an axis the person holds is theirs, on the ladder",
                     kr2 == 30.0 and np.allclose(k2, k1), f"K_R {kr2}")
    snap, _ = _driven("level")
    snap.helper.k_norm, snap.helper.level = np.array([0.30, 0.60, 0.45]), [0, 1, 1]
    snap.helper.n_calls = 1
    snap.poll(0.1, "contact")
    k3, kr3 = snap.stiffness(_Lad())
    bad += not check("level, the default: the same output is given as its rung",
                     np.allclose(k3, [500.0, 500.0, 600.0]) and kr3 == 3.0
                     and snap.record()["output"] == "level", f"{k3} {kr3}")

    # ---- the latched hand ------------------------------------------------------
    # The state machine a person drives the pad with.  It cannot be tried here (it needs
    # a window), and it is the thing they fight if it is wrong, so it is checked against
    # a fake window instead.  The property that matters is the LAST one: a latch holds
    # after the key is gone, which is the whole reason this hand exists.
    # interactive.py imports the whole simulator at module level, so the two classes
    # are taken out of the source, like Arms below.  They need only numpy.
    i_src = (HERE.parent / "writing" / "interactive.py").read_text()
    I = type(sys)("I")
    exec(i_src[i_src.index("class KeyboardHuman:"):i_src.index('CONTROLS = """')],
         {"np": np}, I.__dict__)

    class FakeWin:
        shift = False

        def __init__(self):
            self.queue = []

        def tap(self, *keys):
            self.queue += list(keys)
            return self

        def key_press(self, k):
            if k in self.queue:
                self.queue.remove(k)
                return True
            return False

        def key_down(self, k):
            return False

    w = FakeWin()
    hand = I.LatchedHand(w, np.eye(3), [1000.0] * 3)
    F, N, IDLE = hand.F_PLANE, hand.F_NORMAL, hand.F_IDLE

    def wrench(*keys):
        w.tap(*keys)
        return hand.wrench(0.0, np.zeros(3), np.zeros(3), np.zeros(3))[0]

    bad += not check("a tap sets a direction", abs(wrench("l")[0] - F) < 1e-9)
    bad += not check("the same tap again clears it", abs(wrench("l")[0]) < 1e-9)
    bad += not check("the opposite tap reverses it", abs(wrench("j")[0] + F) < 1e-9)
    bad += not check("press is a tap, not a hold",
                     abs(wrench("u")[2] + N) < 1e-9 and hand.press and not hand.lift)
    bad += not check("pressing while already sweeping needs no second key down",
                     abs(wrench("i")[1] - F) < 1e-9 and abs(hand.wrench(
                         0.0, np.zeros(3), np.zeros(3), np.zeros(3))[0][2] + N) < 1e-9,
                     "sweep and press are two separate taps, both latched")
    bad += not check("tapping press again relaxes", abs(wrench("u")[2] - IDLE) < 1e-9)
    bad += not check("lift is its own latch",
                     abs(wrench("o")[2] - N) < 1e-9 and hand.lift and not hand.press)
    bad += not check("F doubles the plane, never the press",
                     abs(wrench("f", "u")[2] + N) < 1e-9 and hand.fast)
    bad += not check("X clears everything",
                     np.allclose(wrench("x"), [0.0, 0.0, IDLE]))
    f1 = wrench("l")
    f2 = hand.wrench(0.0, np.zeros(3), np.zeros(3), np.zeros(3))[0]
    bad += not check("a latch survives the key being gone", np.allclose(f1, f2),
                     "the one property a held key does not have")

    hold = I.KeyboardHuman(FakeWin(), np.eye(3), [1000.0] * 3)
    bad += not check("the held hand still answers press/lift from the keys",
                     hold.press is False and hold.lift is False)

    # ---- SplitLevels: two owners, one triple ----------------------------------
    src_p = (HERE / "protocol.py").read_text()
    ns_s = {"np": np, "MID": 1}
    exec(src_p[src_p.index("class SplitLevels:"):src_p.index("_HELPERS: dict")], ns_s)
    own, auto = Table([2, 2, 2]), Table([0, 0, 0])
    sp = ns_s["SplitLevels"](own, auto, (2,))
    sp.reset()
    own.level, auto.level = [2, 2, 2], [0, 0, 0]
    sp.poll(0.0, "contact")
    bad += not check("the table owns its axes and the person owns theirs",
                     sp.level == [0, 0, 2], f"{sp.level}")
    # with no model at all the same five series are written, so generation 0 reads the
    # way every later one does
    ns_w = {"np": np}
    exec(WHO_SRC, ns_w)
    o = ns_w["opinions"](sp)
    bad += not check("without a model a step still says who set each axis",
                     list(o["k_source"]) == [0, 0, 1]
                     and list(o["k_level_human"]) == [2, 2, 2]
                     and list(o["k_level_table"]) == [0, 0, 0]
                     and list(o["k_level_model"]) == [-1, -1, -1]
                     and set(o) == set(ns_w["WHO"]))
    o = ns_w["opinions"](auto)
    bad += not check("  ... and a table alone is the table, with nobody at the sticks",
                     list(o["k_source"]) == [0, 0, 0]
                     and list(o["k_level_human"]) == [-1, -1, -1])
    row = ns_w["who_row"]({})
    bad += not check("a loop that never asked records that nobody said anything",
                     set(row) == set(ns_w["WHO"]) and list(row["k_source"]) == [-1] * 3
                     and bool(np.isnan(row["k_norm_model"]).all()))

    # ---- lint: the paths this suite cannot run --------------------------------
    # The viewer path is one of them, and that is where a key map referenced the axis
    # ownership computed below it -- a NameError on the first line of a session, found
    # only by a person running the real thing.  Code that cannot be executed here can
    # still be read.  Only names used before they are bound are gated; a noisy gate
    # gets switched off.
    try:
        import io as _io

        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
        _o, _e = _io.StringIO(), _io.StringIO()
        _rep = Reporter(_o, _e)
        for _f in sorted(HERE.glob("*.py")) + [HERE.parent / "writing" / n for n in
                                               ("protocol.py", "interactive.py",
                                                "teleop.py")]:
            checkPath(str(_f), _rep)
        _hits = [ln for ln in _o.getvalue().splitlines()
                 if "undefined name" in ln or "referenced before assignment" in ln]
        bad += not check("no name is used before it is bound", not _hits,
                         "; ".join(h.split("/")[-1] for h in _hits[:3]) if _hits
                         else "pyflakes, wiping + the writing modules it builds on")
    except ImportError:
        print("  --    pyflakes not installed; the lint gate is skipped")

    # ---- the VR hand against a sticky mailbox and a reloaded page -------------
    # Both are wire-level, so the mapper's own tests cannot see them: the console hands
    # back the LAST pose for ever, and the browser's clock restarts at zero on a reload.
    src_i = (HERE.parent / "writing" / "interactive.py").read_text()
    ns_v = {"np": np}
    exec(src_i[src_i.index("class VRHand:"):src_i.index('CONTROLS = """')], ns_v)

    class _Sim0:
        W = np.eye(3)
        K0 = np.array([1000.0, 1000.0, 1000.0])
        last = {"p": np.array([0.0, 0.0, 0.30])}

    def _m(t, y):
        return {"t": t, "pos": [0, y, -0.4], "quat": [1, 0, 0, 0], "trigger": 0.0,
                "ok": True, "hand": "right", "sticks": {}, "buttons": {}}

    vh = ns_v["VRHand"](_Sim0())
    vh.feed(_m(10.0, 1.20))
    vh.map.target = np.array([0.2, 0.0, 0.30])
    vh.map.last_t = time.time() - 5.0
    for _ in range(40):
        vh.feed(_m(10.0, 1.20))              # the loop polls; the mailbox never empties
    bad += not check("the dead-man elapses on a mailbox that keeps repeating itself",
                     vh.map.stale and np.allclose(
                         vh.map.wrench([0, 0, 0.30], np.zeros(3)), 0),
                     "a headset that goes quiet stops pulling the tool")
    vh.feed(_m(11.0, 1.20))
    bad += not check("  ... and a fresh sample revives it", not vh.map.stale)

    vh2 = ns_v["VRHand"](_Sim0())
    for i in range(5):
        vh2.feed(_m(10.0 + 0.03 * i, 1.20 + 0.01 * i))
    _was = vh2.map.target.copy()
    vh2.feed(_m(0.02, 1.60))                 # a reload: the clock restarts, hand moved
    bad += not check("a reloaded page re-references instead of jumping the tool",
                     np.allclose(vh2.map.target, _was),
                     "the browser clock starts at zero on every load")
    vh2.feed(_m(0.05, 1.63))
    bad += not check("  ... and relative motion resumes from there",
                     abs(float(vh2.map.target[2] - _was[2]) - 0.03) < 1e-9)

    # ---- a person's level has somewhere to go, in EVERY arrangement -----------
    # It did not.  --motion vr owns all three axes, which made `auto_ax` empty, which
    # built a bare KeyboardLevels with no `.own`, so every thumbstick push was dropped
    # without a word -- and with a helper the table was a bare AutoUser, so the same.
    # The sticks never worked in any configuration they were built for.
    src_p2 = (HERE / "protocol.py").read_text()

    def _grab(a, b):
        i = src_p2.index(a)
        return src_p2[i:src_p2.index(b, i)]

    class _KL:                       # stands in for KeyboardLevels and AutoUser alike
        def __init__(self, *a):
            self.level = [1, 1, 1]

        def reset(self):
            self.level = [1, 1, 1]

        def poll(self, t, g):
            return False

    ns_t = {"np": np, "MID": 1, "KeyboardLevels": _KL, "AutoUser": _KL}
    exec(_grab("class SplitLevels:", "def make_table"), ns_t)
    exec(_grab("def make_table", "def helper_table"), ns_t)
    exec(_grab("def helper_table", "_HELPERS: dict"), ns_t)

    class _Args:
        seed_base, reaction = 70000, [0.2, 0.45]

        def __init__(self, motion, helper, person):
            self.motion, self.auto_user = motion, helper
            self._person_ax = person
            self._viewer = type("V", (), {"window": None})()

    _lands = {}
    for _name, _a, _p in (("vr", _Args("vr", False, (0, 1, 2)), (0, 1, 2)),
                          ("vr+helper", _Args("vr", True, (0, 1, 2)), (0, 1, 2)),
                          ("human", _Args("human", False, (2,)), (2,)),
                          ("human+helper", _Args("human", True, (2,)), (2,))):
        tbl = ns_t["make_table"](_a, _a._viewer, _p)
        usr = (type("H", (), {"_table": ns_t["helper_table"](_a, 0)})()
               if _a.auto_user else tbl)
        _lands[_name] = ns_t["human_levels"](usr) is not None
    bad += not check("a person's stiffness has somewhere to go in every arrangement",
                     all(_lands.values()), str(_lands))

    # ---- and the write is read as a takeover, exactly like a keypress ----------
    class _Sp:
        axes = (0, 1, 2)

        def __init__(self):
            self.own, self.auto = _KL(), _KL()
            self.level, self.own_pressed = [1, 1, 1], False

        def reset(self):
            pass

        def poll(self, t, g):
            self.level = list(self.own.level)
            return False

    from stiffness_helper.console.vr import StickLevels
    _tab = _Sp()
    _hu = HU.HelperUser(StubSim(), None, Levels, _tab, version="v0", axes=("r",))
    _hu.reset()
    _hu._own_prev = list(_tab.own.level)
    _hu.helper.level = [1, 1, 0]
    _hu.poll(0.0, "contact")
    _drove = _hu.level[2] == 0
    _st = StickLevels()
    _st.feed("drive", [0, -0.9])
    for _ax, _sg in _st.drain():
        _tab.own.level[_ax] = int(np.clip(_tab.own.level[_ax] + _sg, 0, 2))
    _hu.poll(0.1, "contact")
    bad += not check("a stick push is a takeover, exactly as a keypress is",
                     _drove and _hu.level[2] == 2
                     and any(e["code"] == "takeover" and e["model"] == 0
                             for e in _hu.events),
                     "the model's own choice is kept beside the person's")
    _hu.poll(0.1 + _hu.HOLD + 0.1, "contact")
    bad += not check("  ... and the model has the axis back after the hold",
                     _hu.level[2] == 0)
    _st.feed("off", [0, -0.9])
    _st.feed("off", [0, 0])
    _st.feed("off", [0.9, 0])
    for _ax, _sg in _st.drain():
        _tab.own.level[_ax] = int(np.clip(_tab.own.level[_ax] + _sg, 0, 2))
    _hu.poll(5.0, "contact")
    bad += not check("K_p is the person's and uncontested; only K_R is argued over",
                     _hu.level[0] == 2 and _hu.level[1] == 2 and _hu.level[2] == 0)

    # ---- a walrus loop variable reassigned inside its own loop ---------------
    # No linter has this one: pyflakes sees a valid rebinding and pylint's
    # redefined-loop-name only covers `for` targets.  It cost a session -- the console's
    # command was bound to `c`, the case index of `while (c := next_case())`, so a demo
    # started from the browser was saved under case "collect" and then raised KeyError.
    # The rule is narrow and sound: the condition recomputes the name every iteration,
    # so assigning it in the body is either dead or wrong.
    def walrus_clobbers(path):
        tree = ast.parse(pathlib.Path(path).read_text())
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.While):
                continue
            names = {w.target.id for w in ast.walk(node.test)
                     if isinstance(w, ast.NamedExpr) and isinstance(w.target, ast.Name)}
            if not names:
                continue
            for stmt in node.body:
                for n in ast.walk(stmt):
                    tgt = ([n.target] if isinstance(n, (ast.AugAssign, ast.NamedExpr))
                           else n.targets if isinstance(n, ast.Assign) else [])
                    for t in tgt:
                        if isinstance(t, ast.Name) and t.id in names:
                            out.append(f"{pathlib.Path(path).name}:{t.lineno} "
                                       f"reassigns `{t.id}` inside `while ({t.id} := ...)`")
        return out

    _clob = []
    for _f in sorted(HERE.glob("*.py")) + sorted((HERE.parent / "writing").glob("*.py")):
        _clob += walrus_clobbers(_f)
    bad += not check("no walrus loop variable is reassigned inside its own loop",
                     not _clob, "; ".join(_clob[:3]) if _clob
                     else "wiping + writing, every while-walrus")

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
