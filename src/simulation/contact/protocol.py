"""Stiffness-protocol collection for two tasks whose stiff direction TURNS: flipping a
box up against a wall, and pushing a door open -- both with a stick.

`../wiping/protocol.py` does this for the pad on a board and `../peg/protocol.py` for a
peg in a hole.  Here the tool is a ball-tipped stick pointing down, and the three axes
are named relative to the way the tip is MOVING, because that is what turns with the
work (see contact_tasks.py):

                across the motion (K_t)   along it (K_n)    the wrist (K_R)
    low               300 N/m               300 N/m           0.3 Nm/rad
    mid               800 N/m              1000 N/m           3   Nm/rad
    high             2000 N/m              3000 N/m          30   Nm/rad

THE PROTOCOL is the same three rows for both tasks, read off the hand:

    0. start of episode   across mid,  along mid,  K_R mid
    1. APPROACH           across HIGH, along HIGH, K_R HIGH   nothing is touching: go
                                                              where you are pointed
    2. PRE-CONTACT        across LOW,  along LOW,  K_R LOW    about to touch, and not
                                                              told exactly when
    3. WORK               across LOW,  along HIGH, K_R LOW    FLIP: carry the end of the
                                                              box up and over.
                                                              DOOR: push it open.

Row 3 is the one these tasks are for: SOFT PUSH, HARD MOVE.  ALONG the motion is stiff
because that is where the work is done -- the end of the box has to go up the arc it
is sent on, the door has to come off its latch and go against its closer.  ACROSS it
is soft because that is where the tip presses the box against the wall, at a force that
must stay moderate whatever the hand has wrong about the arc; and on the door it is the
way the spot under the ball slides off sideways as the door swings, which the ball can
only follow if the arm lets it.

WHICH ROW IT IS.  Holding press (the trigger) is PRE-CONTACT, and it stays PRE-CONTACT
for a moment after the tip has arrived.  WORK begins, on the BOX, when the tip has
TURNED -- when it is travelling across the way it came in; and on the DOOR, where the
push goes on the way the tip was already going and there is no turn to see, when the
landing is over (`ContactCriteria.land_window` after the first touch).  Anything else
is APPROACH.

  (Why the box's WORK waits for the turn.  Arriving, the tip is pressing ALONG its own
  motion, and "along" is the axis the third row makes stiff.  Called WORK at the first
  touch, the table stiffened the press itself: the tip leant ten times harder than it
  meant to, backed off, and lost the box in half the episodes -- while every CONSTANT
  stiffness flipped it every time.  On the door that same stiffening is the point: the
  press IS the push.)

BE CLEAR ABOUT WHAT THIS TABLE IS: a schedule reasoned out, not one measured to be
needed.  `stiffness_helper.envs.task_fit` is the measurement -- it holds one stiffness
for the whole episode and it moves this table's switches earlier and later, and says
whether either fails.  Read its verdict before collecting on either task.

Keys (SAPIEN's viewer owns WASD/QE for its camera):
    1 / 2 / 3     across   low / mid / high
    8 / 9 / 0     along    low / mid / high
    4 / 5 / 6     K_R      low / mid / high
    M             everything to mid
    G  start the next episode     R  this demo is finished     N  abort, retry
    ESC           quit (the episode in progress is discarded)

Usage:
    # nobody at the controls: a scripted hand and the table, for a dry run
    python3 protocol.py --task flip --auto-user --headless --per-case 8 --out demos/flip
    # the two checks a task has to pass, on this one
    python3 -m stiffness_helper.envs.task_fit --task door --out runs/task-fit/door
"""
from __future__ import annotations

import argparse
import collections
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
import contact_tasks as CT  # noqa: E402
import teleop as T  # noqa: E402


def _wiping_protocol():
    """`../wiping/protocol.py`, loaded under a name of its own -- as the peg collector
    does, and for its reason: the console tap, the level sources, the table/person
    split and the two task checks are that file's, and copies would drift."""
    spec = importlib.util.spec_from_file_location(
        "wiping_protocol", HERE.parent / "wiping" / "protocol.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["wiping_protocol"] = m
    spec.loader.exec_module(m)
    return m


WPR = _wiping_protocol()
LOW, MID, HIGH = WPR.LOW, WPR.MID, WPR.HIGH
LEVEL_NAMES = WPR.LEVEL_NAMES

# "seated" is the DRAWER's: against its stop.  Appended, never inserted, so the
# index every past log wrote still means what it meant.
PHASES = ("reach", "touch", "work", "seated")
PHASE_GROUP = {"reach": "approach", "touch": "pre-contact", "work": "work",
               "seated": "stop"}
# set per task by `main` from that task's table, as EXPECTED is: a task without a
# stop must not be scored on a row it does not have
GROUPS = ["approach", "pre-contact", "work"]
# protocol phase -> (across level, along level, K_R level), per task.  `EXPECTED` is the
# one in force: `main` sets it from the task, and everything below reads it.
TABLES = {
    "flip": {"approach": (HIGH, HIGH, HIGH), "pre-contact": (LOW, LOW, LOW),
             "work": (LOW, HIGH, LOW)},
    "door": {"approach": (HIGH, HIGH, HIGH), "pre-contact": (LOW, LOW, LOW),
             "work": (LOW, HIGH, LOW)},
    # THE DRAWER HAS A FOURTH ROW, and it is the reason the task exists.  The handle is
    # held from the first frame, so there is no approach and no pre-contact: the episode
    # opens pulling.  ALONG is HIGH to break the detent and draw the drawer out, and LOW
    # once it is against its stop, where a stiff axis is only pulling on something that
    # cannot move.  No constant does both -- which flipping a box and opening a door
    # cannot show, because neither of them ends.
    "drawer": {"work": (LOW, HIGH, LOW), "stop": (LOW, LOW, LOW)},
}
EXPECTED = dict(TABLES["flip"])
AXIS_OF = {"t": 0, "n": 1, "r": 2}        # --auto-axes -> (across, along, K_R)
AX = ("across", "along", "kr")
WORKSPACE = 0.45    # m: a door's handle travels a quarter of a metre, and the hand leads it
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
class Script:
    """A hand nobody is holding.  It offers what `interactive.VRHand` offers -- `wrench`,
    `press`, `lift` -- and pulls the handle through the same kind of spring, so a dry
    run takes the path a person's demo takes.

    WHAT IT KNOWS is what a person at the window knows: where it was TOLD the work is
    (which is wrong by a centimetre or so, so it does not know when it will touch);
    which way the work faces, as it was `SEE` seconds ago, because that is how old what
    an eye reports is; and what it feels through the handle, `FEEL` seconds late.

    WHAT IT DOES WITH THAT is the least a hand could: it moves along the way the work
    was last seen to face, and on the box it leans in as hard as it means to by feel.
    It does not steer round the arc.  Whatever that gets wrong lands ACROSS the motion,
    which is exactly the error the protocol's third row says to be soft against -- so a
    stiffness that is wrong shows up in the force, and is not quietly steered away by a
    clever hand.
    """
    # N/m, Ns/m, and how far the target may lead the handle: 30 N at full stretch.  The
    # peg's hand stops at 12 N, which is a press; a latch here holds up to 11 N, and at
    # 12 the hand hung on a door it could not open whatever the stiffness was.
    K, B, REACH = 300.0, 30.0, 0.100
    V_REACH, V_TOUCH, V_WORK = 0.060, 0.025, 0.040      # m/s
    PRE = 0.020        # m short of where the work was told to be: press from here
    SEE = 0.20         # s
    FEEL = 0.06        # s
    GIVE_UP = 0.060    # m past where it was told, with nothing there
    APPROACH = CT.FWD  # the way it comes at the work

    def __init__(self, sim, spec):
        self.sim, self.spec = sim, spec
        self.state, self.t0 = "reach", 0.0
        self.press, self.done = False, False
        self.tgt = None
        self._past: dict = {}

    @property
    def lift(self) -> bool:
        return not self.press

    def _old(self, name: str, t: float, value, age: float) -> np.ndarray:
        """`value` as it was `age` seconds ago."""
        q = self._past.setdefault(name, collections.deque())
        q.append((t, np.asarray(value, float).copy()))
        while len(q) > 1 and q[1][0] <= t - age:
            q.popleft()
        return q[0][1]

    def _work(self, t: float, f_fb) -> None:
        raise NotImplementedError

    def _contact(self) -> bool:
        return bool(self.sim.last["pen_down"])

    def wrench(self, t, x_m, v_m, f_fb):
        s, dt, A = self.sim, self.sim.dt, self.APPROACH
        if self.tgt is None:
            self.tgt = np.asarray(x_m, float).copy()
        if self.state == "reach":
            goal = s.told - self.PRE * A
            d = goal - self.tgt
            n = float(np.linalg.norm(d))
            self.tgt = goal if n <= self.V_REACH * dt else self.tgt + self.V_REACH * dt * d / n
            if n <= 1e-3:
                self.state, self.t0, self.press = "touch", t, True
        elif self.state == "touch":
            self.tgt = self.tgt + self.V_TOUCH * dt * A
            if self._contact():
                self.state, self.t0 = "work", t
            elif (self.tgt - s.told) @ A > self.GIVE_UP:
                self.done = True                       # it was never there
        elif self.state == "work":
            self._work(t, np.asarray(f_fb, float))
        else:                                          # "rest": stay, then say so
            self.done = t - self.t0 > 0.6
        d = self.tgt - np.asarray(x_m, float)
        n = float(np.linalg.norm(d))
        if n > self.REACH:
            d *= self.REACH / n
        return self.K * d - self.B * np.asarray(v_m, float), None


class FlipScript(Script):
    """Lean the tip on the near end until the box is against the wall, then carry that
    end up the way its face runs, leaning in by feel; let go when the box is standing."""
    PRESS = 5.0                    # N: how hard it means to lean
    HAPTIC = 0.010                 # m of press per N of felt error, per second: the
                                   # writing hand's own number (teleop.WriterStyle)
    LET_GO = np.radians(84.0)      # standing, near enough: the wall has it

    def __init__(self, sim, spec):
        super().__init__(sim, spec)
        self.lifting = False

    def _work(self, t: float, f_fb) -> None:
        s, dt = self.sim, self.sim.dt
        into = self._old("into", t, s.face_inward(), self.SEE)
        up = self._old("up", t, s.face_up(), self.SEE)
        if getattr(self.sim, "STEER", False):
            return self._work_steered(t, f_fb, into, up)
        # what comes back through the handle is the spring the tool is stretching:
        # along `into` it is how hard the tip is leaning on the box
        lean = -float(self._old("felt", t, f_fb, self.FEEL) @ into)
        if not self.lifting:
            self.tgt = self.tgt + self.V_TOUCH * dt * into
            self.lifting = lean >= self.PRESS
        else:
            self.tgt = (self.tgt + self.V_WORK * dt * up
                        + self.HAPTIC * (self.PRESS - lean) * dt * into)
        if s.rise() >= self.LET_GO:
            self.state, self.t0, self.press = "rest", t, False

    def _work_steered(self, t: float, f_fb, into, up) -> None:
        """The same hand, but it KNOWS THE ARC the near end travels.

        The default hand carries its target along the box's `up` as it last saw it --
        the instantaneous tangent, SEE seconds stale.  Over a 90 degree lift that
        tangent turns the whole way, so a straight step leaves the arc and the error
        lands ACROSS the motion.  That is deliberate (see `Script`): it is what makes
        a wrong stiffness show up in the force instead of being steered away.

        This one carries the target ROUND the arc.  The near end pivots about the
        box's far bottom edge, one box-length away along `into`, so the step is a
        rotation of v dt / L about that point rather than a chord off it.

        OFF by default, and here to be COMPARED against rather than to replace: a hand
        that steers may hide the very thing the protocol is built to measure, and
        whether it does is itself the question.
        """
        s, dt = self.sim, self.sim.dt
        lean = -float(self._old("felt", t, f_fb, self.FEEL) @ into)
        if not self.lifting:
            self.tgt = self.tgt + self.V_TOUCH * dt * into
            self.lifting = lean >= self.PRESS
        else:
            L = float(2 * s.u.BOX_HALF[0])               # the near end's radius
            pivot = self.tgt + L * into
            r = self.tgt - pivot
            dth = self.V_WORK * dt / max(L, 1e-6)
            # cross(UP, INTO), not the other way round: with into = x and up = z,
            # cross(into, up) is -y and Rodrigues then carries the target DOWN the
            # arc.  Measured with the sign wrong: 0/12 success, every episode timing
            # out at 30 s with the peak force half again the plain hand's.
            axis = np.cross(up, into)                    # the box turns about this
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                r = (r * np.cos(dth) + np.cross(axis, r) * np.sin(dth)
                     + axis * float(axis @ r) * (1.0 - np.cos(dth)))
                self.tgt = pivot + r
            else:
                self.tgt = self.tgt + self.V_WORK * dt * up
            self.tgt = self.tgt + self.HAPTIC * (self.PRESS - lean) * dt * into
        if s.rise() >= self.LET_GO:
            self.state, self.t0, self.press = "rest", t, False


class DoorScript(Script):
    """Come up to the door's face, push along the way the door faces, and stop and hold
    when it is open."""

    def _work(self, t: float, f_fb) -> None:
        s = self.sim
        into = self._old("into", t, s.into(), self.SEE)    # into the door, as last seen
        self.tgt = self.tgt + self.V_WORK * s.dt * into
        if s.angle() >= self.spec.target:
            self.state, self.t0 = "rest", t


class DrawerScript(Script):
    """Pull straight out, and stop when the drawer is out far enough.

    It opens in `work` because the handle is already held -- there is nothing to reach
    for -- and it pulls along the way the drawer comes out, as last SEEN.  It does not
    feel for the stop and it does not ease off before it: whatever being still stiff
    at the stop costs has to land in the force, or the task is not measuring anything.
    """
    V_PULL = 0.045                 # m/s
    # A STRONGER HAND THAN THE DOOR'S.  `Script.K` 300 over `REACH` 0.1 is 30 N, which
    # the door's comment calls generous against an 11 N latch.  A drawer needs the pull
    # AND the impedance behind it: measured at 30 N, the detent took 10-11 s of a 30 s
    # episode to break, and the drawer then came out in 1.5.  Nothing was wrong with
    # the drawer -- the operator could not pull.
    # 96 N opened one seed in 4.0 s and shook two others off the handle; 50 N is enough
    # for the detent the drawer actually has now, and does not snatch.
    K, REACH = 500.0, 0.10

    def __init__(self, sim, spec):
        super().__init__(sim, spec)
        self.state, self.press = "work", True

    def _work(self, t: float, f_fb) -> None:
        s = self.sim
        out = self._old("pull", t, s.pull(), self.SEE)
        self.tgt = self.tgt + self.V_PULL * s.dt * out
        # pull ON to the stop: stopping at `target` means the drawer never reaches its
        # end, and the end is the whole point of the task
        if s.seated():
            self.state, self.t0 = "rest", t


SCRIPTS = {"flip": FlipScript, "door": DoorScript, "drawer": DrawerScript}


class ContactHand:
    """The phase, read off the hand and the work.  Nobody plans the motion here.

        reach -> touch    press is held
        touch -> work     the tip has met the work, AND -- the box -- has turned, is
                          travelling across the way it came (TURN); or -- the door --
                          the landing is over (see ContactSim.WORK_ON)
        touch -> reach    press let go before it got there
        work  -> reach    press let go

    "Let go" is RELEASE seconds without press, as everywhere else here.
    """
    RELEASE = 0.15     # s
    # Travelling within 20 degrees of square to the way it came.  At 45 the turn was only
    # half made, half of the "along" axis was still the press, and stiffening it there
    # did what stiffening the press does.
    TURN = np.cos(np.radians(70.0))

    def __init__(self, sim, hand):
        self.sim, self.hand = sim, hand
        # A GRASPED task opens already working: there is nothing to reach for and
        # nothing to decide about when it was met.
        self.grasped = bool(getattr(sim, "GRASPED", False))
        self.phase, self.done = ("work" if self.grasped else "reach"), False
        self._idle = 0.0
        self._met = False

    def _working(self) -> bool:
        s = self.sim
        self._met = self._met or bool(s.last["pen_down"])
        if not self._met:
            return False
        if s.WORK_ON == "dwell":
            return s.t - s.t_touch >= s.crit.land_window
        return abs(float(s.n @ s.W0[:, 2])) < self.TURN

    def act(self, t: float, x_m, v_m, f_fb) -> np.ndarray:
        f_h, _ = self.hand.wrench(t, x_m, v_m, f_fb)
        press = bool(self.hand.press)
        self._idle = 0.0 if press else self._idle + self.sim.dt
        let_go = self._idle > self.RELEASE
        if self.grasped:
            # held, so `let_go` means nothing; the only move is onto the stop
            if self.phase == "work" and self.sim.seated():
                self.phase = "seated"
            return f_h
        if self.phase == "reach":
            self._met = False
            if press:
                self.phase = "touch"
        elif self.phase == "touch":
            if self._working():
                self.phase = "work"
            elif let_go:
                self.phase = "reach"
        elif let_go:
            self.phase = "reach"
        return f_h


# --------------------------------------------------------------------------- #
class ContactRecorder(CO.Recorder):
    """`../writing/collect.py`'s Recorder, plus what these tasks have: K_R as a logged
    state, how far the work has got, and THE FRAME AT EVERY STEP -- because here the
    stiffness triple is named along and across the motion, and a log of the triple
    without the frame it was in says nothing about which way the tool was stiff."""
    EXTRA_FULL = ("kr", "kr_req", "progress", "engaged", "across", "moment",
                  "frame") + WPR.WHO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in self.EXTRA_FULL:
            self.full[k] = []

    def on_step(self, i, rec, sess, wr=None) -> None:
        super().on_step(i, rec, sess, wr)
        if i % self.every == 0:
            for k, v in WPR.who_row(rec).items():
                self.full[k].append(v)
            self.full["kr"].append(np.float32(rec["kr"]))
            self.full["kr_req"].append(np.float32(rec.get("kr_req", rec["kr"])))
            self.full["progress"].append(np.float32(rec["progress"]))
            self.full["engaged"].append(bool(rec["engaged"]))
            self.full["across"].append(np.float32(rec["across"]))
            self.full["moment"].append(np.float32(rec["moment"]))
            self.full["frame"].append(np.asarray(rec["W"], np.float32).reshape(9))
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
            del f["privileged/ink_uv"]          # writing's; nothing is laid here


class ContactConsole(WPR.Console):
    """The wiping collector's console tap, saying what THESE tasks have to say."""

    def __init__(self, sim):
        import helper_user  # noqa: F401  finds the stiffness_helper checkout
        from stiffness_helper.console import wire
        self.w = wire
        self.cmds = wire.Commands()
        self.frames = self.gated = 0
        wire.emit("ready", task=sim.TASK, keys="G start / R done / N pass",
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
        self.w.emit("tel", t=round(float(sim.t), 3),
                    f_n=round(float(sim.last["f_n"]), 3),
                    f_t=round(float(max(0.0, f * f - fn * fn) ** 0.5), 3),
                    lvl=[int(v) for v in user.level],
                    k=[float(kd[0]), float(kd[2]), float(sim.ctl.kr)],
                    k_req=[float(k_cmd[0]), float(k_cmd[2]), float(kr_cmd)],
                    rtf=round(float(rtf), 2),
                    # what is still to do, in per cent of the way
                    left=int(round(100.0 * (1.0 - float(sim.last["progress"])))),
                    down=bool(sim.last["pen_down"]), helper=self.helper_out(user),
                    who=self.who(user), alpha=round(alpha, 3),
                    gated=round(self.gated / max(1, self.frames), 4))


class _HandView:
    """The simulator as `interactive.VRHand` sees it.  That hand takes `sim.W`'s third
    column for the work's outward normal and pushes against it while the trigger is
    held; here `W` follows the motion, so it is handed the hand's own frame instead."""

    def __init__(self, sim):
        self._sim = sim

    @property
    def W(self):
        return self._sim.HAND_AXES

    def __getattr__(self, name):
        return getattr(self._sim, name)


# --------------------------------------------------------------------------- #
def run_episode(sim, spec, user, levels, args, recorder=None, viewer=None,
                con=None) -> dict | None:
    """One episode.  None means it was thrown away, by N, by PASS, or by ending before
    a single step had been recorded."""
    sim.reset(spec)
    sess = T.TeleopSession(sim, T.MasterParams(workspace=WORKSPACE))
    vr = None
    if args.motion == "vr":
        import interactive as I
        vr = hand = I.VRHand(_HandView(sim))
        vr.map.R = CT.VR_AXES           # behind the arm: away from you is forward
        if getattr(sim, "GRASPED", False):
            # Held from the first frame, so there is nothing to press -- and the press
            # is FORWARD, the stick's push, which on a drawer is into the chest.  A hand
            # squeezing the trigger to hold the handle pushed against its own pull.
            vr.map.p.press_force = 0.0
    elif args.motion == "human":
        import interactive as I
        hand = (I.LatchedHand if args.keys == "latch" else I.KeyboardHuman)(
            viewer.window, sim.HAND_AXES, sim.K0)
    else:
        hand = SCRIPTS[sim.TASK](sim, spec)
    wr = ContactHand(sim, hand)
    names = ("across", "along", "K_R")

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
        d = out / sim.TASK
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ep_{seed:05d}.h5"
        rec.save(path, dict(
            task=sim.TASK, text=sim.TASK, case=0, board="motion-frame",
            success=bool(res["success"]), compliant=bool(ok_comp), source=source,
            spec=dataclasses.asdict(spec),
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
            phases=list(PHASES), sim_dt=sim.dt, log_hz=100.0, policy_hz=10.0,
            # the frame at the START; `full/frame` is the frame at every step
            writing_frame=sim.W0.tolist()))
    reason = ", ".join(filter(None, [res["fail_reason"], "" if ok_comp else "protocol"]))
    # `kept` is "a file exists" and `scored` is "it met the criteria"; `erased` is the
    # task's progress, under the name the console's table already has a column for.
    row = dict(case=0, text=sim.TASK, board="motion-frame", seed=seed, kept=bool(path),
               scored=bool(keep), saved=str(path) if path else None,
               success=bool(res["success"]), reason=reason, compliance=comp,
               erased=res["progress"], in_band=res["in_band"],
               peak_force=res["peak_force"], land_peak=res["land_peak"],
               peak_across=res["peak_across"], peak_moment=res["peak_moment"],
               t=res["t"])
    return keep, path, row


def verdict_line(row: dict, done: int, per_case: int) -> str:
    comp = row["compliance"]
    v = ("GOOD" if row["scored"] else
         f"saved, did not score: {row['reason']}" if row["kept"] else
         f"not kept: {row['reason']}")
    return (f"[{v}] {100 * row['erased']:.0f}% of the way, landed at "
            f"{row['land_peak']:.1f} N, across {row['peak_across']:.1f} N, wrist "
            f"{row['peak_moment']:.2f} Nm, {row['t']:.1f} s | protocol "
            + " ".join(f"{g} {100 * comp[g]:.0f}%" for g in GROUPS if comp[g] == comp[g])
            + f" | {done}/{per_case}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(CT.SIMS))
    ap.add_argument("--per-case", type=int, default=50, help="demos to collect")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-base", type=int, default=85_000)
    ap.add_argument("--k-across", type=float, nargs=3, default=[300.0, 800.0, 2000.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-along", type=float, nargs=3, default=[300.0, 1000.0, 3000.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-r", type=float, nargs=3, default=[0.3, 3.0, 30.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--ramp", type=float, default=0.15)
    ap.add_argument("--grace", type=float, default=0.6)
    ap.add_argument("--min-compliance", type=float, default=0.8)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--attempts", type=int, default=None,
                    help="stop after this many attempts, kept or not")
    ap.add_argument("--auto-user", action="store_true",
                    help="the table sets every level; nobody presses anything")
    ap.add_argument("--person-rate", type=float, default=None, metavar="P",
                    help="a scripted person who acts on a fraction P of the switches "
                         "the table asks for, so a helper can be taken over without a "
                         "human in the room.  The fraction of FRAMES they end up owning "
                         "is an outcome, recorded in `helper/axis_source`, not P")
    ap.add_argument("--reaction", type=float, nargs=2, default=[0.20, 0.45],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--motion", choices=("script", "human", "vr"), default="script",
                    help="who moves the tool: a scripted hand, the keyboard, or a VR "
                         "controller through the console")
    ap.add_argument("--console", action="store_true",
                    help="driven by stiffness_helper.console over a pipe")
    ap.add_argument("--keys", choices=("latch", "hold"), default="latch")
    ap.add_argument("--auto-axes", nargs="+", default=None,
                    help="axes the table drives so the person does not: t (across) "
                         "n (along) r (K_R), or none.  Default `t n` with a person "
                         "at the controls, leaving them K_R")
    ap.add_argument("--helper", default=None, metavar="CKPT",
                    help="a trained stiffness_helper checkpoint sets the levels on "
                         "--helper-axes; the table and the person keep the rest")
    ap.add_argument("--helper-version", default=None)
    ap.add_argument("--helper-axes", nargs="+", default=["t", "n", "r"],
                    choices=("t", "n", "r"))
    ap.add_argument("--helper-output", choices=("level", "level5", "continuous"),
                    default="level")
    ap.add_argument("--hold", type=int, nargs=3, default=None,
                    metavar=("ACROSS", "ALONG", "KR"),
                    help="hold these three levels (0 low, 1 mid, 2 high) for the whole "
                         "episode: the MAGNITUDE check")
    ap.add_argument("--pin", nargs="+", default=None, metavar="AXIS=LEVEL",
                    help="hold an axis at one level (0 low, 1 mid, 2 high) whatever the "
                         "table says, e.g. `--pin r=0`.  The unpinned axes keep "
                         "following the schedule: the PER-AXIS ablation --hold cannot do")
    ap.add_argument("--steer", action="store_true",
                    help="the scripted hand follows the ARC the work travels instead "
                         "of the tangent it last saw.  OFF by default on purpose: the "
                         "default hand's steering error is what makes a wrong "
                         "stiffness show up in the force rather than be quietly "
                         "corrected, so turning this on changes what the task "
                         "measures.  It is here to be compared against")
    ap.add_argument("--shift-axes", nargs="+", default=None, choices=("t", "n", "r"),
                    help="with --shift: only these axes are moved; the rest switch on "
                         "time.  Default: all three.  Moving one axis is how you ask "
                         "WHICH axis's timing matters -- `--pin` answers only whether "
                         "its LEVEL does, and a pinned axis has no transitions at all, "
                         "so the two questions are not separable without this")
    ap.add_argument("--shift", type=float, default=None, metavar="S",
                    help="the table's own switches, S seconds late (negative: early), "
                         "replayed from a reference pass of the same episode: the "
                         "TIMING check")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="override the spec's 30 s")
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

    EXPECTED.clear()
    EXPECTED.update(TABLES[args.task])
    GROUPS[:] = [g for g in ("approach", "pre-contact", "work", "stop") if g in EXPECTED]
    levels = WPR.Levels(xy=tuple(args.k_across), z=tuple(args.k_along), kr=tuple(args.k_r))
    out = pathlib.Path(args.out or f"demos/{args.task}")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "attempts.jsonl"
    done = len(list((out / args.task).glob("ep_*.h5"))) if (out / args.task).exists() else 0
    attempt = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            attempt = max(attempt, json.loads(line)["attempt"] + 1)

    def log(row: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print("  table   " + "  ".join(
        f"{g}:" + "/".join(LEVEL_NAMES[i][0] for i in EXPECTED[g]) for g in GROUPS))
    print("  levels  across " + " / ".join(f"{k:.0f}" for k in levels.xy)
          + "   along " + " / ".join(f"{k:.0f}" for k in levels.z) + " N/m"
          + "   K_R " + " / ".join(f"{k:g}" for k in levels.kr) + " Nm/rad")
    print(f"  {args.task:<7} {done}/{args.per_case} collected")

    Sim, Spec = CT.SIMS[args.task]
    sim = Sim(cameras=not args.no_cameras, image_size=args.image_size,
              render_mode=None if args.headless else "human")
    # read by FlipScript._work; an attribute rather than a constructor argument so the
    # three sims keep one signature and the flag reaches only the hand that uses it
    sim.STEER = bool(getattr(args, "steer", False))
    viewer = None if args.headless else sim.u.render_human()

    auto_ax = () if args.auto_axes == ["none"] else tuple(AXIS_OF[a] for a in args.auto_axes)
    person_ax = tuple(i for i in range(3) if i not in auto_ax)
    _pin = WPR.pin_spec(args)
    if args.hold is not None:
        user = WPR.HoldUser(args.hold)
    elif args.person_rate is not None:
        # A SCRIPTED PERSON BESIDE THE TABLE, for the assisted and manual arms.
        # `HelperUser` reads `table.own` to see a takeover, and a bare `AutoUser` has no
        # `own`, so without this a headless session can never produce one.  Two
        # independent AutoUsers: the person and the standing table must be able to
        # disagree, which they cannot if they are the same object.
        from stiffness_helper.scripted_person import ScriptedPerson
        # ONLY THE AXES THE HELPER OWNS -- see wiping/protocol.py's `helper_table`:
        # giving the throttled person every axis leaves the uncontested ones stuck at
        # stale levels and the arm stops being about the contested one.
        own_ax = tuple(AXIS_OF[a] for a in args.helper_axes)
        user = WPR.SplitLevels(
            ScriptedPerson(AutoUser(args.seed_base, tuple(args.reaction)),
                           args.person_rate, args.seed_base),
            AutoUser(args.seed_base, tuple(args.reaction)), own_ax)
    elif args.auto_user or viewer is None:
        user = AutoUser(args.seed_base, tuple(args.reaction))
        if _pin:
            user = WPR.PinnedUser(user, _pin)
    else:
        user = WPR.SplitLevels(WPR.KeyboardLevels(viewer.window),
                               AutoUser(args.seed_base, tuple(args.reaction)), person_ax)
    if args.helper:
        import helper_user as HU
        # the envelope a helper's output expands into here, from where its training
        # logs were normalised: one definition, so the two cannot come apart
        from stiffness_helper.adapters.contact import CONTACT_KSPEC as kspec
        version = args.helper_version or pathlib.Path(args.helper).resolve().parent.name
        user = HU.HelperUser(sim, args.helper, levels, user, version=version,
                             axes=tuple(args.helper_axes), kspec=kspec,
                             output=args.helper_output)
        print(f"  helper  {version} on {' '.join(args.helper_axes)}, "
              f"{args.helper_output} output  ({args.helper})")
    con = ContactConsole(sim) if args.console else None
    source = {"script": "protocol-script", "human": "protocol-human",
              "vr": "protocol-vr"}[args.motion]

    try:
        while done < args.per_case and (args.attempts is None or attempt < args.attempts):
            seed = args.seed_base + attempt
            spec = Spec.sample(seed)
            if args.time_limit:
                spec = dataclasses.replace(spec, time_limit=args.time_limit)
            print(f"\n=== demo {done + 1}/{args.per_case}  (seed {seed})")
            if viewer is not None or args.console:
                print("    press G to start" + ("  (or COLLECT in the console)"
                                                if args.console else ""))
                sim.reset(spec)
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
            shifted = None
            if args.shift is not None:
                user, shifted = WPR.shifted_user(
                    lambda u: run_episode(sim, spec, u, levels, args), seed, args,
                    auto=AutoUser)
            rec = ContactRecorder(sim, images=not args.no_cameras)
            res = run_episode(sim, spec, user, levels, args, recorder=rec,
                              viewer=viewer, con=con)
            if res is None:
                if con is not None:
                    con.episode(dict(case=0, text=args.task, board="motion-frame", seed=seed,
                                     attempt=attempt, kept=False, scored=False,
                                     success=False, reason="passed", arm=None,
                                     motion=args.motion))
                continue
            operator = user.record() if hasattr(user, "record") else shifted
            if args.hold is not None:
                operator = dict(skill="fixed", hold=list(args.hold))
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
                con.episode(dict(row, compliance=row["compliance"]["overall"],
                                 gated=round(con.gated / max(1, con.frames), 4),
                                 takeovers=(operator or {}).get("takeovers"),
                                 disagreements=(operator or {}).get("disagreements")))
            print("\n" + verdict_line(row, done, args.per_case))
    except KeyboardInterrupt:
        print("\n[quit] episode in progress discarded")
    print(f"\n{args.task}: {done}/{args.per_case}")
    sim.close()


if __name__ == "__main__":
    main()
