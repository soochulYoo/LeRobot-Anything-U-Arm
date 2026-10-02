"""A scripted operator at the keys, as good or as bad as its two delays.

It does what a person does in `protocol.py --motion human`, through the same
key interface (key_down / key_press), so a scripted demo takes the code path a
person's takes: KeyboardWriter, the virtual master, the Case 1 slave, the
recorder.  It is NOT a person, and its files say so (source
"protocol-scripted").  Its skill is two delays and nothing else:

    motion_delay      s between what the pen does and the keys answering it.
                      The operator steers by where the pen WAS: the ink weaves
                      about the line by about v x delay, the stroke's start is
                      found by tap-and-look, and the lift comes that long after
                      the end was reached.
    stiffness_delay   s between a protocol step starting and its levels being
                      set.  The level commands are the protocol's, late.

Everything else is the same at every skill: keys change at most every DECIDE
seconds, an error under DEAD is left alone, and the aim is LOOK ahead on the
line.  SKILLS holds the three presets and how they were chosen.
"""
from __future__ import annotations

import collections
import dataclasses

import numpy as np

import glyphs as G
import interactive as I
import teleop as T

DECIDE = 0.05        # s between key changes while steering
DEAD = 0.0012        # m, an error not worth a key
LOOK = 0.003         # m ahead on the line the operator aims at
START_TOL = 0.0010   # m, per axis, from the stroke's start before pressing
SETTLE = 0.15        # s after a tap before looking at where the pen went
LIFT_TAP = 0.3       # s of the lift key at the end of a stroke (~5 mm up)
LIFT_PAUSE = 0.5     # s after the end of a stroke before moving on
TIME_LIMIT = 120.0   # s, an operator that has not finished by then never will

# what the hand's keys do with the pen up (interactive.KeyboardHuman)
V_FREE = I.KeyboardHuman.F_PLANE / (I.KeyboardHuman.B_PLANE + T.MasterParams().Bm)

XY_KEYS, Z_KEYS = "123", "zxc"
# protocol step -> (xy level, z level); protocol.EXPECTED, not imported: protocol imports this
LEVELS = {"approach": (1, 2), "pre-contact": (0, 1), "contact": (0, 1), "after-contact": (1, 1)}


@dataclasses.dataclass
class Skill:
    motion_delay: float
    stiffness_delay: float

    def sample(self, rng) -> "Skill":
        """One episode's operator: the preset, +-20% each."""
        return Skill(*(float(v * rng.uniform(0.8, 1.2)) for v in dataclasses.astuple(self)))


# How the presets were chosen: S, 7 and <star>, 5 randomizations each, one delay
# swept with the other held at the "good" value.
#
#   motion delay (s)      0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.50
#   success, of 15          15    15    15    15    15    15    13     7     2
#   chamfer to target, mm  0.54  0.53  0.63  0.74  0.96  1.06  1.20  1.40  1.81
#   worst precision, %      100   100   100   100    93    95    82    64    47
#
#   stiffness delay (s)    0.0   0.2   0.5   0.8   1.2   2.0   3.0   4.0
#   success, of 15          15    15    15    15    15    15    15    15
#   protocol compliance   1.00  1.00  1.00  0.94  0.85  0.72  0.62  0.52
#
# The motion delay decides how well the text is written.  The stiffness delay
# does not touch the writing at all -- a key's 3.5 N press sets the paper force
# whatever K is -- it makes the stiffness LABELS late.  The first descent takes
# about 3 s, so only a delay longer than that carries the approach's levels
# (xy mid, z high) into the contact itself.
SKILLS = {
    "good": Skill(motion_delay=0.10, stiffness_delay=0.2),
    "normal": Skill(motion_delay=0.25, stiffness_delay=1.0),
    "bad": Skill(motion_delay=0.45, stiffness_delay=4.0),
}


class ScriptedOperator:
    """Answers the viewer window's key questions from a script.  Make one per
    episode and hand it to protocol.run_episode as `window`."""
    should_close = False
    shift = False

    def __init__(self, sim, skill: Skill):
        self.sim, self.skill = sim, skill
        self.down: set = set()
        self.taps: set = set()
        self._step = -1
        self.hist = collections.deque()
        self.state, self.k, self.i = "start", 0, 0
        self.t_state = self.t_next = 0.0
        self.release: dict = {}                 # key -> time its tap ends
        self.due: list = []                     # (time, key) level changes on their way
        self.level = [1, 1]                     # the levels asked for so far (mid, mid)

    # ---- the window ----
    def key_down(self, k: str) -> bool:
        self._update()
        return k in self.down

    def key_press(self, k: str) -> bool:
        self._update()
        return k in self.taps

    # ---- the operator ----
    def _goto(self, state: str, step: str | None = None) -> None:
        self.state, self.t_state = state, self.sim.t
        if step is not None:                    # a protocol step begins: its levels, late
            for axis, (keys, lv) in enumerate(zip((XY_KEYS, Z_KEYS), LEVELS[step])):
                if lv != self.level[axis]:
                    self.level[axis] = lv
                    self.due.append((self.sim.t + self.skill.stiffness_delay, keys[lv]))

    @staticmethod
    def _steer(d, dead: float) -> set:
        return ({"l"} if d[0] > dead else {"j"} if d[0] < -dead else set()) | \
               ({"i"} if d[1] > dead else {"k"} if d[1] < -dead else set())

    def _update(self) -> None:
        sim = self.sim
        if sim.step_i == self._step:
            return
        self._step, t = sim.step_i, sim.t
        self.taps = {key for due, key in self.due if due <= t}
        self.due = [(due, key) for due, key in self.due if due > t]
        for key in [key for key, end in self.release.items() if end <= t]:
            self.down.discard(key)
            del self.release[key]

        # what the operator sees: the pen and the ink, motion_delay ago
        self.hist.append((t, sim.belief.to_canvas(sim.last["p"])[:2].copy(), bool(sim.last["pen_down"])))
        while len(self.hist) > 1 and self.hist[1][0] <= t - self.skill.motion_delay:
            self.hist.popleft()
        _, uv, inking = self.hist[0]

        if self.state == "start":
            self.paths = [G.resample(s, 0.001) for s in sim.target.strokes]
            self._goto("travel", "approach")
        if self.state == "finished":
            # R, held until the episode ends, and not before the last level change is made
            if t - self.t_state > 0.5 and not self.due:
                self.taps.add("r")
            return
        if t < self.t_next or self.release:
            return
        self.t_next = t + DECIDE

        if self.state == "travel":
            # tap-and-look: a push sized to the error seen, then wait to see where it went
            d = self.paths[self.k][0] - uv
            keys = self._steer(d, START_TOL)
            if not keys:
                self.i = 0
                self.down = {"u"}
                self._goto("press", "pre-contact")
            else:
                self.down = set(keys)
                for key in keys:
                    self.release[key] = t + max(0.02, abs(d[0 if key in "jl" else 1]) / V_FREE)
                self.t_next = max(self.release.values()) + self.skill.motion_delay + SETTLE
        elif self.state == "press":
            if inking:
                self._goto("write", "contact")
        elif self.state == "write":
            path = self.paths[self.k]
            self.i += int(np.argmin(np.linalg.norm(path[self.i:self.i + 12] - uv, axis=1)))
            keys, ahead = set(), int(LOOK / 0.001)
            while not keys and self.i + ahead < len(path) + 4:      # a corner can hide the aim point
                keys = self._steer(path[min(self.i + ahead, len(path) - 1)] - uv, DEAD)
                ahead += 1
            self.down = {"u"} | keys
            if self.i >= len(path) - 2 or not keys:
                # a tap of "lift": letting go alone leaves the pen 1 mm up after
                # LIFT_PAUSE, too low to cross a tilted paper to the next stroke
                self.down, self.release = {"o"}, {"o": t + LIFT_TAP}
                self._goto("lift", "after-contact")
        elif self.state == "lift":
            if t - self.t_state > LIFT_PAUSE and not inking:
                self.k += 1
                if self.k < len(self.paths):
                    self._goto("travel", "approach")
                else:
                    self._goto("finished")
