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
    motion_delay: float          # s, WiperStyle.visual_delay
    stiffness_delay: float       # s, the AutoUser's reaction

    def sample(self, rng) -> "Skill":
        """One episode's operator: the preset, +-20% each, as writing's does."""
        return Skill(*(float(v * rng.uniform(0.8, 1.2))
                       for v in dataclasses.astuple(self)))

    def reaction(self, rng=None) -> tuple:
        """The (min, max) reaction window AutoUser draws each level change from.

        A floor of 10 ms, because the "good" tier is a reaction of ZERO and
        `rng.uniform(0, 0)` would make every level change land on the same step -- a
        reaction time no person has and a label with no ramp in it.
        """
        d = self.stiffness_delay
        return (max(0.8 * d, 0.0), max(1.2 * d, 0.01))


# Interpolated from the measured curve to span it: erasure ~0.99 / 0.90 / 0.65 and
# success ~14 / 8 / 3 of 15.  A graded set with real failures in it, and every episode
# still completing -- which a 4 s tier does not.
SKILLS = {
    "good": Skill(motion_delay=0.10, stiffness_delay=0.0),
    "normal": Skill(motion_delay=0.10, stiffness_delay=0.35),
    "bad": Skill(motion_delay=0.10, stiffness_delay=0.70),
}


def skill(spec) -> Skill:
    """`--scripted bad` -> the preset; `--scripted 0.6 4.0` -> a custom pair."""
    if len(spec) == 1:
        if spec[0] not in SKILLS:
            raise SystemExit(f"--scripted {spec[0]!r}: one of {sorted(SKILLS)} or two floats")
        return SKILLS[spec[0]]
    if len(spec) == 2:
        return Skill(float(spec[0]), float(spec[1]))
    raise SystemExit("--scripted takes a tier name or two delays")


def name(spec) -> str:
    return spec[0] if len(spec) == 1 else "custom"


def apply(style, sk: Skill, seed: int):
    """-> the style with this operator's seeing delay, and its reaction window."""
    s = sk.sample(np.random.default_rng(80_000 + seed))
    return dataclasses.replace(style, visual_delay=s.motion_delay), s
