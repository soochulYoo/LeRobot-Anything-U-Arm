"""A scripted operator for wiping, as good or as bad as its two delays.

`../writing/scripted.py` does this for the pen by answering a viewer window's key
questions, so a scripted writing demo takes a person's code path exactly.  Wiping has no
keyboard-motion path -- its motion comes from `wipe_teleop.SyntheticWiper` -- so the two
delays are applied to the knobs that operator ALREADY has instead of to a keyboard:

    motion_delay     -> WiperStyle.visual_delay.  The operator keeps the pad on the row
                        it can only see this late, so the pad weaves about the row by
                        about v x delay and the row's start is found late.
    stiffness_delay  -> AutoUser's reaction.  The protocol's levels, applied that late.

The difference from the writing tiers is worth stating rather than glossing: writing's
`motion_delay` is how late a KEY answers, wiping's is how late the operator SEES.  Both
are "a scripted operator with two delays", and neither is a person -- the files say
`protocol-scripted` and carry the skill in their `operator` attribute.

Naming: `wipe_scripted` and not `scripted`, for the reason `wipe_sim.py` gives -- the
writing package uses flat imports, and a sibling called `scripted.py` would shadow
`../writing/scripted.py` depending on sys.path order.

THE PRESETS ARE MEASURED, by `tier_sweep.sbatch`, and the table below is what it said.
The short version: writing's numbers did not transfer and neither did its choice of axis.
The seeing delay turned out to be inert on this task and the level reaction turned out to
decide it outright, so the tiers move the level reaction alone.
"""
from __future__ import annotations

import dataclasses

import numpy as np

# MEASURED, 3 texts x 5 randomizations per point, curved board, 40 mm pad, one delay
# swept with the other held good (tier_sweep.sbatch):
#
#   seeing delay (s)      0.10  0.20  0.30  0.45  0.60  0.80
#   success, of 15          10    10    10    11    10    10
#   erased                0.95  0.95  0.95  0.95  0.95  0.95
#   in-band               0.97  0.97  0.97  0.97  0.97  0.97
#   peak force, N         7.72  7.92  7.76  7.59  7.59  7.91
#
#   level reaction (s)     0.0   0.2   0.5   1.0   2.0   4.0
#   success, of 15          14    10     6     0     0     1
#   compliance            1.00  1.00  1.00  0.75  0.34  0.08
#   erased                0.99  0.95  0.84  0.43  0.18  0.36
#   in-band               0.97  0.97  0.97  0.97  0.96  0.95
#
# TWO RESULTS, AND THEY INVERT THE WRITING TASK.
#
# The seeing delay is INERT.  Nothing moves between 0.10 s and 0.80 s -- not success, not
# erasure, not the force.  A 40 mm pad sweeping straight rows has none of the 3 mm
# tolerance a pen holds, so steering late costs nothing.  Writing's tiers made this their
# main axis (chamfer 0.53 -> 1.81 mm over the same range); here it is not an axis at all.
#
# The LEVEL REACTION decides the task.  Erasure falls 0.99 -> 0.43 by one second of delay
# and success 14/15 -> 0/15, while in-band holds at 0.97 and every failure is `erased` --
# so this is not a force-band problem, it is the pad meeting a curved board on an edge
# because its wrist is still rigid.  That is exactly the mechanism
# `ROTATIONAL_STIFFNESS.md` measured (K_R 60 -> 0.3 moved erasure 14.5% -> 93.5%), and it
# is the opposite of writing, where a stiffness delay "does not touch the writing at all".
#
# SO THE TIERS MOVE ONE KNOB.  The seeing delay is held at its good value in all three,
# because a tier axis that changes nothing would make the tiers unattributable -- which is
# the defect of the writing set, where both delays moved together and bad-vs-good cannot
# be assigned to either.  Holding it fixed makes these tiers mean exactly one thing:
# how late the stiffness arrives.
#
# The usable range is 0 to about 0.7 s.  Past 1 s nothing succeeds and past 2 s the
# compliance score itself collapses, so a tier out there is degenerate rather than bad.


@dataclasses.dataclass
class Skill:
    """An operator as four independent knobs, so a tier drop can be ATTRIBUTED.

    The one-knob tiers above answer "is late stiffness bad" and nothing else.  Two
    more questions need their own axes, and both come from what the collected data
    turned out to be:

      HOW LATE vs HOW ERRATIC.  `stiffness_delay` is the MEAN lateness and
      `delay_spread` is the per-change scatter around it, and they are separate
      because they predict different things.  A mean shift moves E[label | scene],
      so a model fitted to it learns to be late.  Scatter at a correct mean leaves
      E[label | scene] untouched -- the noise is unpredictable, so the best causal
      predictor of an erratic operator is the CORRECT timing, and the only cost is
      sample efficiency.  Conflating the two (which `reaction` did, taking the
      window as a fixed +-20% of the mean) makes that distinction unaskable.

      LATE vs WRONG.  Every tier so far eventually applies EXPECTED[group] exactly;
      only the moment moves.  But a novice does not set the right stiffness late,
      they set the wrong one -- and before contact, where the table says every axis
      must come DOWN, the characteristic error is to leave it where it is or push it
      up.  `p_hold` and `p_wrong` are those two, kept apart because they are not the
      same mistake: holding leaves a rigid wrist meeting a curved board, which is the
      mechanism the tier sweep measured as decisive, while moving the wrong way also
      spends the tank.

    `episode_jitter` is operator-to-operator variation, drawn once per episode.  It
    is a knob rather than the hard-coded +-20% it was, because a controlled sweep of
    `delay_spread` needs the other source of spread held at zero.
    """
    motion_delay: float          # s, WiperStyle.visual_delay
    stiffness_delay: float       # s, the AutoUser's MEAN reaction
    delay_spread: float = 0.2    # fraction of the mean, per level change
    p_hold: float = 0.0          # P(an axis that should move does not move at all)
    p_wrong: float = 0.0         # P(it moves one rung the WRONG way)
    episode_jitter: float = 0.2  # fraction, drawn once per episode on the delays

    def sample(self, rng) -> "Skill":
        """One episode's operator: the delays jittered, the probabilities not.

        Only the delays move.  Jittering a probability the way the old
        `astuple`-over-everything did would also have jittered `delay_spread`, so
        the knob whose whole job is to set the spread would have had a spread of its
        own -- and `p_hold` would differ per episode, which is an operator who gets
        better and worse at the task between demos rather than being one operator.
        """
        j = self.episode_jitter
        f = (lambda v: float(v * rng.uniform(1 - j, 1 + j))) if j > 0 else float
        return dataclasses.replace(self, motion_delay=f(self.motion_delay),
                                   stiffness_delay=f(self.stiffness_delay))

    def reaction(self, rng=None) -> tuple:
        """The (min, max) reaction window AutoUser draws each level change from.

        A floor of 10 ms, because the "good" tier is a reaction of ZERO and
        `rng.uniform(0, 0)` would make every level change land on the same step -- a
        reaction time no person has and a label with no ramp in it.
        """
        d, w = self.stiffness_delay, self.delay_spread
        return (max((1 - w) * d, 0.0), max((1 + w) * d, 0.01))

    def errors(self) -> tuple:
        return (self.p_hold, self.p_wrong)


# Interpolated from the measured curve to span it: erasure ~0.99 / 0.90 / 0.65 and
# success ~14 / 8 / 3 of 15.  A graded set with real failures in it, and every episode
# still completing -- which a 4 s tier does not.
SKILLS = {
    "good": Skill(motion_delay=0.10, stiffness_delay=0.0),
    "normal": Skill(motion_delay=0.10, stiffness_delay=0.35),
    "bad": Skill(motion_delay=0.10, stiffness_delay=0.70),

    # THE 2x2.  `good` and `novice` are the ends of the ladder above, so the one-knob
    # set is the diagonal of this one and nothing is lost by generating the square.
    # The off-diagonal cells are the point: a model trained on `late` and a model
    # trained on `fumbling` have been given the same number of demos by operators who
    # are bad in different ways, and a drop that appears in one and not the other is
    # attributable.  Moving both at once cannot do that, which is the defect this
    # file's header records in the writing tiers -- "both delays moved together and
    # bad-vs-good cannot be assigned to either".
    "late": Skill(motion_delay=0.10, stiffness_delay=0.70),
    "fumbling": Skill(motion_delay=0.10, stiffness_delay=0.0,
                      p_hold=0.30, p_wrong=0.15),
    "novice": Skill(motion_delay=0.10, stiffness_delay=0.70,
                    p_hold=0.30, p_wrong=0.15),

    # Matched mean, 4.5x the scatter, and no episode jitter so the scatter is the only
    # thing that differs from `normal`.  This is the one tier whose prediction is that
    # NOTHING changes in the asymptote: the mean is right, so the conditional mean the
    # model fits is right, and only the sample efficiency should suffer.  Worth having
    # precisely because a null result here is informative.
    "jittery": Skill(motion_delay=0.10, stiffness_delay=0.35, delay_spread=0.9,
                     episode_jitter=0.0),
}


def skill(spec) -> Skill:
    """`--scripted bad` -> the preset; `--scripted 0.6 4.0` -> a custom pair.

    A name is also allowed trailing overrides -- `--scripted late p_hold=0.5` -- so a
    sweep of one knob does not need a preset per point, and the baseline it is being
    compared against stays the named one rather than a second hand-typed tuple.
    """
    if not spec:
        raise SystemExit("--scripted takes a tier name or two delays")
    if spec[0] in SKILLS:
        sk = SKILLS[spec[0]]
        over = {}
        for kv in spec[1:]:
            if "=" not in kv:
                raise SystemExit(f"--scripted {spec[0]} {kv!r}: expected key=value")
            k, v = kv.split("=", 1)
            if k not in {f.name for f in dataclasses.fields(Skill)}:
                raise SystemExit(f"--scripted: no knob {k!r}; one of "
                                 f"{sorted(f.name for f in dataclasses.fields(Skill))}")
            over[k] = float(v)
        return dataclasses.replace(sk, **over) if over else sk
    if len(spec) == 2 and all(_isnum(x) for x in spec):
        return Skill(float(spec[0]), float(spec[1]))
    raise SystemExit(f"--scripted {spec[0]!r}: one of {sorted(SKILLS)} or two floats")


def _isnum(x) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def name(spec) -> str:
    """The tier's name, with any overrides kept in it.

    A sweep run as `--scripted late p_hold=0.5` is not the `late` tier and it is not
    `custom` either; calling it either one makes two different operators land in the
    data under one label, which is the thing the skill attribute exists to prevent.
    """
    if spec and spec[0] in SKILLS:
        return spec[0] if len(spec) == 1 else spec[0] + "+" + ",".join(spec[1:])
    return "custom"


def apply(style, sk: Skill, seed: int):
    """-> the style with this operator's seeing delay, and its reaction window."""
    s = sk.sample(np.random.default_rng(80_000 + seed))
    return dataclasses.replace(style, visual_delay=s.motion_delay), s
