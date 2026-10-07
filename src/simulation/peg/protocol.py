"""Stiffness-protocol collection for peg insertion: you move the peg, and the stiffness
is a level per axis.

`../wiping/protocol.py` does this for the pad.  Here the tool is a square peg in the
gripper and the work is a hole it has to find and go into, so the three axes are

                lateral (across the hole)   axial (along it)   K_R (the wrist)
    low               300 N/m                 400 N/m            5 Nm/rad
    mid               700 N/m                1200 N/m           20 Nm/rad
    high             1500 N/m                3000 N/m           80 Nm/rad

Why these numbers.  Every one of them is a value the scripted study of this scene ran
(../peg_insertion_case1.py, measurements in ../PEG_DEMO_PLAN.md): it carries the peg at
1500 / 3000 / 80, searches at 300 / 400 / 20 and pushes home at 300 / 3000 / 20, and
that schedule seated 8 of 8 at 4 mm and at 8 mm of aim error.  The low wrist rung and
the two mids are the only ones it did not use.

THE PROTOCOL, read off the hand:

    0. start of episode      lateral mid,   axial mid,   K_R mid
    1. APPROACH              lateral HIGH,  axial HIGH,  K_R HIGH   the peg is carried
    2. PRE-CONTACT           lateral LOW,   axial LOW,   K_R mid    it comes onto the face
    3. CONTACT               lateral LOW,   axial LOW,   K_R mid    it rubs, looking for the hole
    4. INSERT                lateral LOW,   axial HIGH,  K_R mid    the head is in: push home

BE CLEAR ABOUT WHAT THIS TABLE IS.  That same study closed its own gate against it: one
constant stiffness -- 300 / 400 / 20, i.e. rows 2 and 3 held throughout -- seated as
many as the schedule at every aim error and at every clearance down to 1 mm.  So the
table is the schedule that was measured to WORK, not one measured to be NEEDED, and a
session collected here is evidence about a person doing the task, not about switching.

WHICH STEP IT IS.  Pulling the trigger (or holding press) in the air is PRE-CONTACT,
the peg touching the box is CONTACT, and the head 5 mm past the entrance face is INSERT;
anything else is APPROACH.  Nothing is scripted once the episode starts.

WHAT IS DONE FOR YOU, before it starts: the peg is turned onto the hole's axis and
carried to 35 mm in front of the face, up to 12 mm off the hole.  See peg_sim.py.

WHERE YOU STAND.  The window looks at the entrance from 65 degrees round to the side,
because from behind the peg the hand hides the hole.  A VR controller follows the
window -- away from you is into the screen -- so going in is mostly a sideways push, and
each episode prints which way.  The trigger presses along the hole whatever the hand does.

Keys (SAPIEN's viewer owns WASD/QE for its camera):
    1 / 2 / 3     lateral  low / mid / high
    8 / 9 / 0     axial    low / mid / high
    4 / 5 / 6     K_R      low / mid / high
    M             everything to mid
    G             start the next episode
    R             this demo is finished
    N             abort this episode, retry the same randomization
    ESC           quit (the episode in progress is discarded)
  with --motion human, the other hand:
    J / L         across the hole, left / right       I / K   up / down
    U             press along the hole                O       let go

Usage:
    # nobody at the controls: a scripted hand and the table, for a dry run
    python3 protocol.py --auto-user --headless --per-case 8 --out demos/peg_script
    # a person with a VR controller, driven from stiffness_helper.console
    python3 protocol.py --console --motion vr --keep-failed --min-compliance 0 \
                        --out demos/console/g0
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import pathlib
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "writing"))

import collect as CO  # noqa: E402  ../writing/collect.py, for the Recorder
import peg_sim as PS  # noqa: E402
import teleop as T  # noqa: E402


def _wiping_protocol():
    """`../wiping/protocol.py`, loaded under a name of its own.

    It is `protocol` there and this file is `protocol` here, and the console tap, the
    level sources and the table/person split are that file's -- copies would drift.
    """
    spec = importlib.util.spec_from_file_location(
        "wiping_protocol", HERE.parent / "wiping" / "protocol.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["wiping_protocol"] = m
    spec.loader.exec_module(m)
    return m


WPR = _wiping_protocol()
LOW, MID, HIGH = WPR.LOW, WPR.MID, WPR.HIGH
LEVEL_NAMES = WPR.LEVEL_NAMES

PHASES = ("travel", "touch", "search", "insert")
PHASE_GROUP = {"travel": "approach", "touch": "pre-contact",
               "search": "contact", "insert": "insert"}
GROUPS = ["approach", "pre-contact", "contact", "insert"]
# protocol phase -> (lateral level, axial level, K_R level)
EXPECTED = {"approach": (HIGH, HIGH, HIGH), "pre-contact": (LOW, LOW, MID),
            "contact": (LOW, LOW, MID), "insert": (LOW, HIGH, MID)}
AXIS_OF = {"t": 0, "n": 1, "r": 2}        # --auto-axes -> (lateral, axial, K_R)
AX = ("lateral", "axial", "kr")
# The handle's travel.  Writing's 0.12 m is a letter; seating a peg is the stand-off
# plus most of its own length, and a wall at 0.12 would stop it short of home.
WORKSPACE = 0.30
MIN_DEMO = 0.2      # s: an episode shorter than this was ended by a queued command


# --------------------------------------------------------------------------- #
class AutoUser:
    """Someone who follows the table perfectly, a reaction time late."""

    def __init__(self, seed: int, reaction=(0.20, 0.45)):
        self.rng = np.random.default_rng(90_000 + seed)
        self.reaction = reaction
        self.reset()

    def reset(self) -> None:
        self.level = [MID, MID, MID]
        self._group = None
        self._due = None

    def poll(self, t: float, group: str) -> bool:
        if group != self._group:
            self._group = group
            self._due = t + float(self.rng.uniform(*self.reaction))
        if self._due is not None and t >= self._due:
            self._due = None
            new = list(EXPECTED[group])
            if new != self.level:
                self.level = new
                return True
        return False


class HoldUser:
    """One set of levels for the whole episode.  The control: PEG_DEMO_PLAN.md found a
    constant seated as many pegs as the schedule, so anything that sets levels here has
    that to beat."""

    def __init__(self, level):
        self._hold = [int(v) for v in level]
        self.level = list(self._hold)

    def reset(self) -> None:
        self.level = list(self._hold)

    def poll(self, t: float, group: str) -> bool:
        return False


class Compliance:
    """Fraction of time the levels matched the table, per protocol phase and per axis,
    not counting `grace` seconds after each phase change."""

    def __init__(self, grace: float):
        self.grace = grace
        self.t_change = 0.0
        self.group = None
        self.hit = {g: 0 for g in GROUPS}
        self.n = {g: 0 for g in GROUPS}
        self.ax_hit = {a: 0 for a in AX}

    def step(self, t: float, group: str, level) -> None:
        if group != self.group:
            self.group, self.t_change = group, t
        if t - self.t_change >= self.grace:
            self.n[group] += 1
            self.hit[group] += int(tuple(level) == EXPECTED[group])
            for i, a in enumerate(AX):
                self.ax_hit[a] += int(level[i] == EXPECTED[group][i])

    def summary(self) -> dict:
        tot = sum(self.n.values())
        per = {g: (self.hit[g] / self.n[g] if self.n[g] else float("nan")) for g in GROUPS}
        per["overall"] = sum(self.hit.values()) / tot if tot else float("nan")
        for a in AX:
            per[f"axis_{a}"] = self.ax_hit[a] / tot if tot else float("nan")
        return per


# --------------------------------------------------------------------------- #
class ScriptHand:
    """A hand nobody is holding: carry the peg to the face, press, rub toward where the
    hole is, push home.  It offers what `interactive.VRHand` offers -- `wrench`, `press`,
    `lift` -- and pulls the handle through the same kind of spring, so a dry run takes
    the path a person's demo takes.

    IT IS TOLD WHERE THE HOLE IS, which a person is not: it rubs straight at it.  That
    makes it a check that the task can be done through this loop, not a search study --
    the search study is ../peg_insertion_case1.py.

    It keeps pushing past the hole until the head drops, as a person does.  Aimed
    exactly at the hole it stalls on the face: the lateral spring at its low rung makes
    3.6 N over the largest offset, and friction under a 3 N press holds that.
    """
    K, B, REACH = 300.0, 30.0, 0.040      # N/m, Ns/m, m the target may lead the handle
    PRESS = 4.0                           # N: the press the scripted study found best
    V_CARRY, V_RUB, V_PUSH = 0.020, 0.010, 0.025   # m/s

    def __init__(self, sim, spec):
        self.sim, self.spec = sim, spec
        self.state, self.t0 = "carry", 0.0
        self.press, self.done = False, False
        self.home = None
        self._seated = None

    @property
    def lift(self) -> bool:
        return not self.press

    def wrench(self, t, x_m, v_m, f_fb):
        s, sp = self.sim, self.spec
        if self.home is None:
            self.home = np.asarray(x_m, float).copy()
        e1, e2, axis = s.W[:, 0], s.W[:, 1], s.axis
        face = self.home + (sp.standoff + 0.004) * axis       # 4 mm into the face
        off = sp.offset[0] * e1 + sp.offset[1] * e2
        r = float(np.linalg.norm(off))
        if self.state == "carry":
            tgt = self.home + min(sp.standoff + 0.004, self.V_CARRY * t) * axis
            if s.last["pen_down"] or self.V_CARRY * t > sp.standoff + 0.010:
                self.state, self.t0, self.press = "rub", t, True
        elif self.state == "rub":
            go = min(r + 0.020, self.V_RUB * (t - self.t0))
            tgt = face - go * off / max(r, 1e-9)
            self._rub = tgt
            if s.last["in_hole"]:
                self.state, self.t0 = "push", t
        else:
            tgt = self._rub + self.V_PUSH * (t - self.t0) * axis
            if s.last["seated"]:
                self._seated = t if self._seated is None else self._seated
                self.done = t - self._seated > 0.5
        d = tgt - np.asarray(x_m, float)
        n = float(np.linalg.norm(d))
        if n > self.REACH:
            d *= self.REACH / n
        f = self.K * d - self.B * np.asarray(v_m, float)
        if self.press:
            f = f - self.PRESS * s.W[:, 2]
        return f, None


class PegHand:
    """The phase, read off the hand and the peg.  Nobody plans the motion here.

        travel -> touch     press is held
        touch  -> search    the peg is on the box
        touch  -> travel    press let go before it got there
        search -> insert    the head is past the entrance face
        search -> travel    press let go and the peg came off
        insert -> search    it was pulled back out

    "Let go" is RELEASE seconds without press, as in `writing/protocol.KeyboardWriter`:
    a finger re-seating itself on the trigger is not a retreat.  Once the head is in the
    hole the press no longer matters to the phase -- the hole is holding the peg.
    """
    RELEASE = 0.15     # s

    def __init__(self, sim, hand):
        self.sim, self.hand = sim, hand
        self.phase, self.done = "travel", False
        self._idle = 0.0

    def act(self, t: float, x_m, v_m, f_fb) -> np.ndarray:
        f_h, _ = self.hand.wrench(t, x_m, v_m, f_fb)
        press = bool(self.hand.press)
        self._idle = 0.0 if press else self._idle + self.sim.dt
        let_go = self._idle > self.RELEASE
        last = self.sim.last
        if self.phase == "travel":
            if press:
                self.phase = "touch"
        elif self.phase == "touch":
            if last["in_hole"]:
                self.phase = "insert"
            elif last["pen_down"]:
                self.phase = "search"
            elif let_go:
                self.phase = "travel"
        elif self.phase == "search":
            if last["in_hole"]:
                self.phase = "insert"
            elif let_go and not last["pen_down"]:
                self.phase = "travel"
        elif not last["in_hole"]:
            self.phase = "search" if last["pen_down"] else "travel"
        return f_h


# --------------------------------------------------------------------------- #
class PegRecorder(CO.Recorder):
    """`../writing/collect.py`'s Recorder, plus what only this task has: K_R as a logged
    state and a label, as wiping records it, and how far in the head is."""
    EXTRA_FULL = ("kr", "kr_req", "depth", "in_hole") + WPR.WHO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in self.EXTRA_FULL:
            self.full[k] = []

    def on_step(self, i, rec, sess, wr=None) -> None:
        super().on_step(i, rec, sess, wr)
        if i % self.every == 0:
            # the person's level, the model's and the table's, beside the one applied
            for k, v in WPR.who_row(rec).items():
                self.full[k].append(v)
            self.full["kr"].append(np.float32(rec["kr"]))
            self.full["kr_req"].append(np.float32(rec.get("kr_req", rec["kr"])))
            self.full["depth"].append(np.float32(rec["depth"]))
            self.full["in_hole"].append(bool(rec["in_hole"]))
        if i % self.p_every == 0:
            self.obs.setdefault("kr", []).append(np.float32(rec["kr"]))

    def actions(self) -> dict:
        a = super().actions()
        kr = np.asarray(self.obs["kr"])
        a["kr"] = np.r_[kr[1:], kr[-1:]]
        return a

    def save(self, path, attrs: dict) -> None:
        import h5py
        super().save(path, attrs)
        with h5py.File(path, "a") as f:
            del f["privileged/ink_uv"]          # writing's; a peg lays none


class PegConsole(WPR.Console):
    """The wiping collector's console tap, saying what THIS task has to say."""

    def __init__(self, sim):
        # `helper_user` finds the stiffness_helper checkout and puts it on sys.path --
        # see the wiping collector, which is where this order was learned.
        import helper_user  # noqa: F401
        from stiffness_helper.console import wire
        self.w = wire
        self.cmds = wire.Commands()
        self.frames = self.gated = 0
        wire.emit("ready", task="peg", keys="G start / R done / N pass",
                  band=[float(v) for v in sim.crit.force_band],
                  ladders=None, cams=[n for n, _ in self.CAMS],
                  note="the keyboard still works; the console is a second source")

    def tel(self, sim, user, k_cmd, kr_cmd, rtf):
        """`k` is what the robot HAS, `k_req` what was asked -- see the wiping tap."""
        fn = float(sim.last["f_sensor_n"])
        f = float(np.linalg.norm(sim.last["f_filt"]))
        kd = sim.k_diag()
        alpha = float(sim.last.get("alpha", 1.0))
        self.frames += 1
        self.gated += int(alpha < self.GATE_SHUT)
        # `left` is what is still to do: millimetres short of seated.
        left = max(0.0, sim.half_len - 0.015 - float(sim.last["depth"]))
        self.w.emit("tel", t=round(float(sim.t), 3),
                    f_n=round(float(sim.last["f_n"]), 3),
                    f_t=round(float(max(0.0, f * f - fn * fn) ** 0.5), 3),
                    lvl=[int(v) for v in user.level],
                    k=[float(kd[0]), float(kd[2]), float(sim.ctl.kr)],
                    k_req=[float(k_cmd[0]), float(k_cmd[2]), float(kr_cmd)],
                    rtf=round(float(rtf), 2), left=int(round(1000.0 * left)),
                    down=bool(sim.last["pen_down"]), helper=self.helper_out(user),
                    who=self.who(user), alpha=round(alpha, 3),
                    gated=round(self.gated / max(1, self.frames), 4))


# --------------------------------------------------------------------------- #
def run_episode(sim, spec, user, levels, args, recorder=None, viewer=None,
                con=None) -> dict | None:
    """One episode.  None means it was thrown away, by N, by PASS, or by ending before
    a single step had been recorded."""
    sim.reset(spec)
    if viewer is not None:
        sim.aim_viewer(viewer)
    b = sim.hole_bearing
    print(f"    the hole runs {abs(b):.0f} deg to the {'right' if b > 0 else 'left'} of "
          f"straight ahead in the window")
    sess = T.TeleopSession(sim, T.MasterParams(workspace=WORKSPACE))
    vr = None
    if args.motion == "vr":
        import interactive as I
        vr = hand = I.VRHand(sim)
        vr.map.R = sim.vr_axes                 # the hand follows the window; see peg_sim
    elif args.motion == "human":
        import interactive as I
        hand = (I.LatchedHand if args.keys == "latch" else I.KeyboardHuman)(
            viewer.window, sim.W, sim.K0)
    else:
        hand = ScriptHand(sim, spec)
    wr = PegHand(sim, hand)
    names = ("lateral", "axial", "K_R")

    user.reset()
    comp = Compliance(args.grace)
    k_cmd = levels.k(user.level)
    kr_cmd = levels.k_r(user.level)
    dt = sim.dt
    i = 0
    last_group = None
    last, acc = time.time(), 0.0
    t_wall0, t_sim0 = time.time(), sim.t
    last_frame, last_tel, aborted = -1e9, -1e9, False
    if con is not None:
        con.new_episode()
    while not wr.done and sim.t < spec.time_limit:
        n_steps = 1
        if viewer is not None:
            w = viewer.window
            if w.should_close or w.key_down("esc"):
                raise KeyboardInterrupt
            if w.key_press("n"):
                print("\n[aborted] same randomization again")
                return None
            if w.key_press("r"):
                print("\n[finished by hand]")
                wr.done = True
        if vr is not None and con is not None:
            # The newest pose, every iteration, and a stick push is one rung -- both as
            # the wiping collector does them, for the reasons it gives.
            vr.feed(con.cmds.vr())
            own = WPR.human_levels(user)
            for ax, sign in vr.levels():
                if own is None:
                    break
                own.level[ax] = int(np.clip(own.level[ax] + sign, 0, 2))
        if con is not None:
            cmd = con.poll()
            if cmd == "done":
                print("\n[finished from the console]")
                wr.done = True
            elif cmd == "pass":
                print("\n[passed from the console]")
                aborted = True
                break
            elif cmd == "stop":
                raise KeyboardInterrupt
        if viewer is not None:
            now = time.time()
            acc += min(now - last, 0.1)
            last = now
            n_steps = min(int(acc / dt), 50)
            acc -= n_steps * dt
        for _ in range(n_steps):
            group = PHASE_GROUP[wr.phase]
            if user.poll(sim.t, group) and viewer is not None:
                print("\n    " + "  ".join(f"{n} {LEVEL_NAMES[v]}"
                                           for n, v in zip(names, user.level)))
            if group != last_group and viewer is not None:
                print(f"\n  [{sim.t:5.1f}s] {group.upper():<12} -> " + ", ".join(
                    f"{n} {LEVEL_NAMES[v]}" for n, v in zip(names, EXPECTED[group])))
            last_group = group
            f_h = wr.act(sim.t, sess.x_m, sess.v_m, sess.f_fb)
            a_ramp = min(1.0, dt / args.ramp)
            k_want, kr_want = WPR.stiffness(user, levels)
            k_cmd = k_cmd + (k_want - k_cmd) * a_ramp
            kr_cmd = kr_cmd + (kr_want - kr_cmd) * a_ramp
            comp.step(sim.t, group, user.level)
            rec = sess.step(f_h, k_cmd, kr_cmd)
            rec.update(phase=PHASES.index(wr.phase), stroke=0,
                       k_level=np.array(user.level), **WPR.opinions(user))
            # A helper sees exactly the record the recorder logs, so what it is given at
            # run time cannot drift from what it was trained on.
            if hasattr(user, "observe"):
                user.observe(sim.t, rec)
            if recorder is not None:
                recorder.on_step(i, rec, sess, wr)
            i += 1
            if getattr(hand, "done", False):          # the script ends its own episode
                wr.done = True
            if wr.done:
                break
        if viewer is not None:
            sim.env.render_human()
        rtf = (sim.t - t_sim0) / max(1e-6, time.time() - t_wall0)
        if con is not None and sim.t - last_tel > 0.5:    # 2 Hz: a screen, not a loop
            last_tel = sim.t
            con.tel(sim, user, k_cmd, kr_cmd, rtf)
            if sim.t - last_frame > 0.5:
                last_frame = sim.t
                con.frame(sim)
    if aborted:
        return None
    if sim.t < MIN_DEMO:
        # DONE BEFORE ANYTHING HAPPENED IS A PASS.  The wiping collector learned this
        # from a DONE queued behind a COLLECT while the simulator was still loading:
        # with a window the first pass steps zero times and saving raised; without one
        # it steps once and saved a one-frame demo.  Neither is a demonstration.
        print("\n[done before anything was recorded: discarded]")
        return None
    res = sim.score()
    res["finished"] = bool(wr.done)
    if not wr.done:
        res["success"] = False
        res["fail_reason"] = ",".join(filter(None, [res["fail_reason"], "timeout"]))
    res["compliance"] = comp.summary()
    return res


def save_episode(sim, rec, res, spec, levels, args, seed: int, source: str,
                 out: pathlib.Path, operator=None):
    comp = res["compliance"]
    ok_comp = comp["overall"] >= args.min_compliance
    keep = bool(res["success"] and ok_comp)
    path = None
    if keep or args.keep_failed:
        d = out / "peg"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ep_{seed:05d}.h5"
        rec.save(path, dict(
            task="peg", text="peg", case=0, board="box", success=bool(res["success"]),
            compliant=bool(ok_comp), source=source, spec=dataclasses.asdict(spec),
            protocol=dict(levels=dataclasses.asdict(levels), expected=EXPECTED,
                          phase_group=PHASE_GROUP, ramp=args.ramp, grace=args.grace,
                          reaction=list(args.reaction), level_names=LEVEL_NAMES,
                          axes=list(AX), auto_axes=list(args.auto_axes)),
            compliance=comp, operator=operator,
            master=dataclasses.asdict(T.MasterParams(workspace=WORKSPACE)),
            criteria=dataclasses.asdict(sim.crit),
            gains={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in dataclasses.asdict(sim.ctl.g).items()},
            metrics={k: v for k, v in res.items() if k not in ("checks", "compliance")},
            peg=dict(half_len=sim.half_len, half_w=sim.half_w,
                     hole_half_w=sim.hole_half_w),
            phases=list(PHASES), sim_dt=sim.dt, log_hz=100.0, policy_hz=10.0,
            writing_frame=sim.W.tolist()))
    reason = ", ".join(filter(None, [res["fail_reason"], "" if ok_comp else "protocol"]))
    # `kept` is "a file exists" and `scored` is "it met the criteria" -- two words for
    # two ideas, as the wiping collector learned.  `erased` is this task's progress,
    # under the name the console's table already has a column for.
    row = dict(case=0, text="peg", board="box", seed=seed, kept=bool(path),
               scored=bool(keep), saved=str(path) if path else None,
               success=bool(res["success"]), reason=reason, compliance=comp,
               erased=res["progress"], inserted=bool(res["inserted"]),
               depth_mm=res["depth_mm"], in_band=res["in_band"],
               peak_force=res["peak_force"], t=res["t"])
    return keep, path, row


def verdict_line(row: dict, done: int, per_case: int) -> str:
    comp = row["compliance"]
    v = ("GOOD" if row["scored"] else
         f"saved, did not score: {row['reason']}" if row["kept"] else
         f"not kept: {row['reason']}")
    return (f"[{v}] {'seated' if row['inserted'] else 'not seated'} at "
            f"{row['depth_mm']:.0f} mm, in-band {row['in_band']:.2f} "
            f"peak {row['peak_force']:.1f} N | protocol "
            + " ".join(f"{g} {100 * comp[g]:.0f}%" for g in GROUPS if comp[g] == comp[g])
            + f" | {done}/{per_case}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", type=int, default=50, help="demos to collect")
    ap.add_argument("--out", default="demos/peg")
    ap.add_argument("--seed-base", type=int, default=80_000)
    ap.add_argument("--k-lat", type=float, nargs=3, default=[300.0, 700.0, 1500.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-axial", type=float, nargs=3, default=[400.0, 1200.0, 3000.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-r", type=float, nargs=3, default=[5.0, 20.0, 80.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--ramp", type=float, default=0.15)
    ap.add_argument("--grace", type=float, default=0.6)
    ap.add_argument("--min-compliance", type=float, default=0.8)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--attempts", type=int, default=None,
                    help="stop after this many attempts, kept or not")
    ap.add_argument("--auto-user", action="store_true",
                    help="the table sets every level; nobody presses anything")
    ap.add_argument("--reaction", type=float, nargs=2, default=[0.20, 0.45],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--motion", choices=("script", "human", "vr"), default="script",
                    help="who moves the peg: a scripted hand, the keyboard, or a VR "
                         "controller through the console")
    ap.add_argument("--console", action="store_true",
                    help="driven by stiffness_helper.console over a pipe")
    ap.add_argument("--keys", choices=("latch", "hold"), default="latch")
    ap.add_argument("--auto-axes", nargs="+", default=None,
                    help="axes the table drives so the person does not: t (lateral) "
                         "n (axial) r (K_R), or none.  Default `t n` with a person "
                         "at the controls, leaving them K_R")
    ap.add_argument("--helper", default=None, metavar="CKPT",
                    help="a trained stiffness_helper checkpoint sets the levels on "
                         "--helper-axes; the table and the person keep the rest, and "
                         "a key or a stick push takes an axis back")
    ap.add_argument("--helper-version", default=None,
                    help="what to call it in the log; the checkpoint's folder by default")
    ap.add_argument("--helper-axes", nargs="+", default=["t", "n", "r"],
                    choices=("t", "n", "r"))
    ap.add_argument("--helper-output", choices=("level", "continuous"), default="level",
                    help="what the robot is given on an axis the helper drives: its "
                         "output snapped to the nearest level, or the output itself")
    ap.add_argument("--hold", type=int, nargs=3, default=None, metavar=("LAT", "AX", "KR"),
                    help="hold these three levels (0 low, 1 mid, 2 high) for the whole "
                         "episode: the constant-stiffness control a schedule has to beat")
    ap.add_argument("--max-offset-mm", type=float, default=12.0,
                    help="how far off the hole the peg may start")
    ap.add_argument("--clearance-mm", type=float, default=0.0,
                    help="0 keeps the task's own 3 mm per side")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="override the spec's 90 s")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()

    person = args.motion in ("human", "vr")
    if args.auto_axes is None:
        args.auto_axes = ["t", "n"] if person else ["none"]
    if args.auto_axes != ["none"] and not set(args.auto_axes) <= set(AXIS_OF):
        ap.error(f"--auto-axes takes {' '.join(AXIS_OF)} or none")
    if person and args.headless:
        ap.error(f"--motion {args.motion} needs a window")
    if args.motion == "vr" and not args.console:
        ap.error("--motion vr needs --console: the controller samples arrive on that pipe")
    if args.headless and not args.auto_user:
        ap.error("--headless needs --auto-user: nobody can press keys without a window")

    levels = WPR.Levels(xy=tuple(args.k_lat), z=tuple(args.k_axial), kr=tuple(args.k_r))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "attempts.jsonl"
    done = len(list((out / "peg").glob("ep_*.h5"))) if (out / "peg").exists() else 0
    attempt = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            attempt = max(attempt, json.loads(line)["attempt"] + 1)

    def log(row: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print("  table   " + "  ".join(
        f"{g}:" + "/".join(LEVEL_NAMES[i][0] for i in EXPECTED[g]) for g in GROUPS))
    print("  levels  lateral " + " / ".join(f"{k:.0f}" for k in levels.xy)
          + "   axial " + " / ".join(f"{k:.0f}" for k in levels.z) + " N/m"
          + "   K_R " + " / ".join(f"{k:g}" for k in levels.kr) + " Nm/rad")
    print(f"  peg     {done}/{args.per_case} collected")

    # One env for the session, and one peg: see peg_sim on why, and on its size.
    sim = PS.PegSim(cameras=not args.no_cameras, image_size=args.image_size,
                    render_mode=None if args.headless else "human",
                    build_seed=args.seed_base, clearance_mm=args.clearance_mm)
    viewer = None if args.headless else sim.u.render_human()
    print(f"  the peg: {2000 * sim.half_len:.0f} mm long, "
          f"{2000 * sim.half_w:.0f} mm across, in a {2000 * sim.hole_half_w:.0f} mm hole")

    auto_ax = () if args.auto_axes == ["none"] else tuple(AXIS_OF[a] for a in args.auto_axes)
    person_ax = tuple(i for i in range(3) if i not in auto_ax)
    if args.hold is not None:
        user = HoldUser(args.hold)
    elif args.auto_user or viewer is None:
        user = AutoUser(args.seed_base, tuple(args.reaction))
    else:
        # ALWAYS A SplitLevels WHEN A PERSON MAY PRESS SOMETHING, even with every axis
        # theirs: that is what gives a thumbstick push somewhere to land.
        user = WPR.SplitLevels(WPR.KeyboardLevels(viewer.window),
                               AutoUser(args.seed_base, tuple(args.reaction)), person_ax)
    if args.helper:
        # The model takes its axes and wraps whoever had them: the table stays as the
        # standing second opinion, and a person's key or stick takes an axis back for a
        # few seconds.  All of that is the wiping collector's `HelperUser`, with this
        # task's envelope.
        import helper_user as HU
        from stiffness_helper.adapters.peg import PEG_KSPEC
        version = args.helper_version or pathlib.Path(args.helper).resolve().parent.name
        user = HU.HelperUser(sim, args.helper, levels, user, version=version,
                             axes=tuple(args.helper_axes), kspec=PEG_KSPEC,
                             output=args.helper_output)
        print(f"  helper  {version} on {' '.join(args.helper_axes)}, "
              f"{args.helper_output} output  ({args.helper})")
    if viewer is not None:
        own = [("lateral", "1/2/3"), ("axial", "8/9/0"), ("K_R", "4/5/6")]
        print("  levels  " + "   ".join(f"{n} {k}" for i, (n, k) in enumerate(own)
                                        if i in person_ax)
              + ("   (the table drives " + "/".join(own[i][0] for i in range(3)
                                                    if i not in person_ax) + ")"
                 if len(person_ax) < 3 else ""))
    con = PegConsole(sim) if args.console else None
    source = {"script": "protocol-script", "human": "protocol-human",
              "vr": "protocol-vr"}[args.motion]

    try:
        while done < args.per_case and (args.attempts is None or attempt < args.attempts):
            seed = args.seed_base + attempt
            spec = PS.PegSpec.sample(seed, 1e-3 * args.max_offset_mm)
            if args.time_limit:
                spec = dataclasses.replace(spec, time_limit=args.time_limit)
            print(f"\n=== demo {done + 1}/{args.per_case}  (seed {seed})")
            if viewer is not None or args.console:
                print("    press G to start" + ("  (or COLLECT in the console)"
                                                if args.console else ""))
                sim.reset(spec)
                if viewer is not None:
                    sim.aim_viewer(viewer)
                while True:
                    if viewer is not None:
                        if viewer.window.key_press("g"):
                            break
                        if viewer.window.should_close or viewer.window.key_down("esc"):
                            raise KeyboardInterrupt
                        sim.env.render_human()
                    if con is not None:
                        cmd = con.poll()
                        if cmd == "collect":
                            break
                        if cmd == "stop":
                            raise KeyboardInterrupt
                    if viewer is None:
                        time.sleep(0.02)
            rec = PegRecorder(sim, images=not args.no_cameras)
            res = run_episode(sim, spec, user, levels, args, recorder=rec,
                              viewer=viewer, con=con)
            if res is None:
                if con is not None:
                    # A discarded episode is still an event, or the console sits in
                    # "recording" for the rest of the session.
                    con.episode(dict(case=0, text="peg", board="box", seed=seed,
                                     attempt=attempt, kept=False, scored=False,
                                     success=False, reason="passed", arm=None,
                                     motion=args.motion))
                continue
            operator = user.record() if hasattr(user, "record") else None
            keep, path, row = save_episode(sim, rec, res, spec, levels, args, seed,
                                           source, out, operator)
            row["attempt"] = attempt
            row["arm"] = None
            row["calls"] = (operator or {}).get("calls")
            row["helper"] = (operator or {}).get("helper")
            row["helper_output"] = (operator or {}).get("output")
            row["motion"] = args.motion
            done += int(bool(path))
            attempt += 1
            log(row)
            if con is not None:
                # After save_episode, so the event cannot arrive before the file exists.
                con.episode(dict(row, compliance=row["compliance"]["overall"],
                                 gated=round(con.gated / max(1, con.frames), 4),
                                 takeovers=(operator or {}).get("takeovers"),
                                 disagreements=(operator or {}).get("disagreements")))
            print("\n" + verdict_line(row, done, args.per_case))
    except KeyboardInterrupt:
        print("\n[quit] episode in progress discarded")
    print(f"\npeg: {done}/{args.per_case}")
    sim.close()


if __name__ == "__main__":
    main()
