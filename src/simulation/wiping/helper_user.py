"""A `user` for protocol.run_episode whose levels come from a trained stiffness helper.

    python3 protocol.py --board curved --helper /scratch2/soochul/TeleopHelper/runs/w40_v1/model.pt \
            --helper-version v1 --per-case 20 --attempts 20 --keep-failed --min-compliance 0 \
            --headless --workers 12 --out demos/helper/v1

It has the same three methods `AutoUser` and `KeyboardLevels` have -- `reset`,
`poll(t, group)` and a `.level` triple -- so `run_episode` does not change at all.  The
helper's continuous stiffness is SNAPPED to the protocol's own levels, which is the point:
the ramp, the energy tank, the compliance score and the recorder all stay on the path the
human and scripted arms took, so the only thing that differs between arms is who chooses
the level.  Commanding continuous K instead would change two things at once and the
comparison would not be attributable.

Everything model-side lives in `stiffness_helper.deploy`; this file only pulls arrays out
of what the loop already has.  Point `STIFFNESS_HELPER` at that repository, or let the
default below find it.

What it feeds the model, and why each is the same quantity the training logs carried:

    k_applied   (mean of the two in-plane stiffnesses, the normal one, K_R) read from
                `sim.k_diag(rec["K"])` and `rec["kr"]` -- the APPLIED values, post-ramp
                and post-tank, which is what the adapter wrote into `k_app`
    f_contact   `rec["f_filt"]`, the filtered contact force, which the adapter wrote into
                both `f_est` and `f_true`.  There is no contact moment in this rig, so the
                moment channel is zeros here exactly as it was in training
    images      `sim.observe(images=True)`'s two 128x128 cameras, box-averaged to 64
    quat_d      the writing frame, which is the orientation the controller is commanded to

WHICH AXES THE HELPER OWNS is `--helper-axes`, and the default is the rotational one
alone.  See the comment in `__init__`: handing it the axes it has no evidence about would
make the comparison measure the labeler's weakest rule instead of its strongest.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

_DEFAULT = "/scratch2/soochul/TeleopHelper"


def _find_helper_repo() -> str:
    """Where `stiffness_helper` lives, in order: $STIFFNESS_HELPER, an ancestor of this
    file, then one machine's path.

    The ancestor search is the one that matters.  That repository reaches this simulator
    as a SUBMODULE of itself -- external/lerobot-uarm/src/simulation/wiping is where this
    file then sits -- so walking up from here finds it with no configuration at all, on
    any machine and under any clone name.  Without it, running anything in this directory
    by hand (the pre-flight test, most of all) only worked on the machine _DEFAULT names,
    and failed as a bare ModuleNotFoundError that says nothing about what to set.
    """
    env = os.environ.get("STIFFNESS_HELPER")
    if env:
        return env
    for up in pathlib.Path(__file__).resolve().parents:
        if (up / "stiffness_helper" / "__init__.py").exists():
            return str(up)
    return _DEFAULT


_SH = _find_helper_repo()
if _SH not in sys.path:
    sys.path.insert(0, _SH)
if not (pathlib.Path(_SH) / "stiffness_helper" / "__init__.py").exists():
    print(f"[helper_user] no `stiffness_helper` package at {_SH}.\n"
          f"              set STIFFNESS_HELPER=/path/to/the/stiffness-helper/checkout",
          file=sys.stderr)


def _np(x):
    """Whatever the simulator hands back -> a numpy array."""
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x)


class HelperUser:
    """Levels from a helper.  Construct one per episode, or call `reset()`.

    `ckpt=None` with `version="v0"` is the RULE, loaded through the same `LevelHelper`
    as a checkpoint so this file needs no branch on the version.  It is the cold start --
    the version that exists before any data does -- and it is NOT what `--arms v0` means
    by default: v0 there is the model trained on the tier demos, `--ckpt-v0`.  Be aware
    of what the rule reduces to on THIS rig before using it as a bar: SAPIEN's
    `get_pairwise_contact_forces` returns a force and no moment, so the moment channel
    is zeros, and v0's rotational rule -- |m| / theta_ok, the one rule that reads the
    pad's tilt -- pins to the bottom of the K_R range the moment contact begins.  With
    `--helper-axes r` a v0 arm is therefore close to a CONSTANT: K_R mid in the air,
    low once down.  That is not a crippled baseline, it is the honest first version on a
    rig whose sensor gives force only, and it is the bar v1 has to clear; but it is a
    constant, so do not report it as a rule that reacts.
    """

    AXES = {"t": 0, "n": 1, "r": 2}

    def __init__(self, sim, ckpt: str | None, levels, table, version: str = "v1",
                 hz: float = 10.0, axes=("r",)):
        from stiffness_helper.adapters.wiping import WIPING_KSPEC
        from stiffness_helper.deploy import LevelHelper, downsample
        self.sim = sim
        self._down = downsample
        self.helper = LevelHelper(ckpt, kspec=WIPING_KSPEC, hz=hz, version=version,
                                 ladders=(tuple(levels.xy), tuple(levels.z),
                                          tuple(levels.kr)))
        self.ckpt = None if ckpt is None else str(ckpt)
        self.version, self.hz = self.helper.version, hz
        self.axes = tuple(self.AXES[a] for a in axes)
        # THE AXES THE HELPER DOES NOT OWN KEEP FOLLOWING THE TABLE.  Only the rotational
        # axis is the helper's by default, because that is the axis the tier sweep measured
        # the task to be decided by: one second of delay in the level commands took erasure
        # 0.99 -> 0.43 while the force stayed in band at 0.97 and every failure was
        # `erased`, which is a rigid wrist on a curved board, not a force problem.  The
        # other two the protocol chose for reasons it measured -- "a stiff normal turns the
        # curved board's ~10 mm height swing into force error" -- and a helper trained on
        # logs whose force never left the band has no evidence against that.  Handing it
        # axes it has no evidence about would make a closed-loop comparison measure the
        # labeler's weakest rule instead of its strongest.
        # The table is INJECTED, not imported: importing `protocol` from here would pull
        # the whole simulator stack in and make this file impossible to test on its own.
        self._table = table
        self._level = [1, 1, 1]
        self._held = {i: None for i in self.axes}     # when the person last took an axis
        self._own_prev = None
        self._dis_since = {i: None for i in self.axes}
        self._dis_open = {i: False for i in self.axes}
        self.events: list = []

    # ---- the interface run_episode uses --------------------------------------
    def reset(self) -> None:
        self.helper.reset()
        self._table.reset()
        self._level = list(self._table.level)
        self._held = {i: None for i in self.axes}
        self._own_prev = None
        self._dis_since = {i: None for i in self.axes}
        self._dis_open = {i: False for i in self.axes}
        self.events = []

    @property
    def level(self):
        return self._level

    def observe(self, t: float, rec) -> None:
        """The step record the recorder also gets: the model reads it and nothing else.

        Called AFTER the controller step, so `poll` on the next iteration uses an
        observation one control period (2 ms) old -- the same staleness a real rig has, and
        three orders of magnitude below the 10 Hz the model runs at.  Reading `sim.last`
        instead was wrong: the leader quantities (x_m, f_h, f_fb) are not in it, they come
        from the session, and `rec` is where the two are already merged -- which is also
        exactly the dict the adapter built the training set from.
        """
        if t < self.helper.t_next:
            return
        from stiffness_helper.geometry import R_to_quat
        o = self.sim.observe(images=True)
        top = self._down(_np(o["rgb_top_camera"]))
        side = self._down(_np(o["rgb_wrist_camera"]))
        kd = _np(self.sim.k_diag(rec["K"])).reshape(-1)
        k_applied = np.array([0.5 * (kd[0] + kd[1]), kd[2], float(_np(rec["kr"]))])
        self.helper.step(
            t, p=_np(rec["p"]), quat=R_to_quat(_np(rec["R"])), v=_np(rec["v"]),
            x_d=_np(rec["x_d"]), quat_d=R_to_quat(_np(self.sim.W)),
            x_m=_np(rec["x_m"]), f_contact=_np(rec["f_filt"]),
            k_applied=k_applied, top=top, side=side)

    # ---- arbitration on a contested axis -------------------------------------
    # WHO DRIVES IS NOT A JUDGEMENT ABOUT WHO IS RIGHT.  K is not identifiable from
    # (x, F), so at the instant of a disagreement there is no measurement that decides
    # it -- which is the constraint this whole project exists under.  So the rule here
    # is fixed and boring: the PERSON wins the moment they press, and keeps the axis
    # until they have been quiet for HOLD seconds.  Nothing about accuracy enters.
    #
    # What the disagreement is FOR is the record.  A takeover says the helper was wrong,
    # not what right would have been, so it is kept as an event -- with what the model
    # wanted at that instant -- and the thing that turns it into a label is the hindsight
    # labeler, offline.  Using takeovers directly as student targets is what keeps a rare
    # signal rare.
    HOLD = 3.0            # s of silence before the model gets a taken axis back
    DIS_HOLD = 0.3        # s a model/table split must persist before it is an event
    MAX_EVENTS = 400      # an episode's worth; a runaway log helps nobody

    def poll(self, t: float, group: str) -> bool:
        """-> True when the level changed.

        `group` reaches the TABLE, which owns the axes the helper does not, and never the
        helper: a helper is not told the task phase, which is the whole point -- a phase
        lookup is the baseline it has to beat, and the reference work measured eight
        architectures reproducing a phase-indexed stiffness table to 97.7-97.9% while
        learning nothing.
        """
        self._table.poll(t, group)
        own = getattr(self._table, "own", None)          # the person's own keys
        auto = getattr(self._table, "auto", None)        # the protocol table
        own_lvl = list(own.level) if own is not None else None
        new = list(self._table.level)

        for i in self.axes:
            model = int(self.helper.level[i])
            # (1) a PRESS on an axis the model owns: the person takes it, and the model's
            #     opinion at that instant is kept with it.
            if (own_lvl is not None and self._own_prev is not None
                    and own_lvl[i] != self._own_prev[i]):
                self._held[i] = t
                self._note(t, "takeover", i, human=own_lvl[i], model=model,
                           phase=group)
            held = self._held[i] is not None and t - self._held[i] < self.HOLD
            if held:
                new[i] = own_lvl[i]
            else:
                self._held[i] = None
                new[i] = model

            # (2) the standing second opinion is the PROTOCOL TABLE, not the person's
            #     last keypress: a level nobody chose is not an opinion.  A split that
            #     persists is an event, raised once per run rather than per tick.
            rival = int(auto.level[i]) if auto is not None else None
            if rival is None or rival == model:
                if self._dis_open[i]:
                    self._note(t, "agree_again", i, human=rival, model=model,
                               phase=group)
                self._dis_since[i], self._dis_open[i] = None, False
            else:
                if self._dis_since[i] is None:
                    self._dis_since[i] = t
                elif (not self._dis_open[i]
                      and t - self._dis_since[i] >= self.DIS_HOLD):
                    self._dis_open[i] = True
                    self._note(t, "shadow_disagree", i, human=rival, model=model,
                               phase=group, held=held)

        self._own_prev = own_lvl
        changed = new != self._level
        self._level = new
        return changed

    def _note(self, t: float, code: str, axis: int, **kw) -> None:
        if len(self.events) >= self.MAX_EVENTS:
            return
        self.events.append(dict(t=round(float(t), 3), code=code, axis=int(axis),
                                k_norm=round(float(self.helper.k_norm[axis]), 3), **kw))

    def record(self) -> dict:
        """What goes in the episode's `operator` attribute, so a log says what drove it."""
        ev = self.events
        return dict(skill=f"helper-{self.version}", helper=self.version,
                    ckpt=self.ckpt, hz=self.hz,
                    axes=[a for a, i in self.AXES.items() if i in self.axes],
                    calls=int(self.helper.n_calls),
                    takeovers=sum(e["code"] == "takeover" for e in ev),
                    disagreements=sum(e["code"] == "shadow_disagree" for e in ev),
                    events=list(ev))
