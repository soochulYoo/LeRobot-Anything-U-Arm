"""Stiffness-protocol collection for wiping: the pad wipes by itself, you set
the stiffness -- including the ROTATIONAL one, which is the whole point.

`../writing/protocol.py` does this for the pen with two axis groups, along the
paper and into it.  A flat pad on a board whose normal turns has a third
decision, and it is the one this task exists to teach:

                xy (along the board)   z (into it)    K_R (the wrist)
    low              500 N/m             300 N/m        0.3 Nm/rad
    mid             1000 N/m             600 N/m        3   Nm/rad
    high            3000 N/m            1500 N/m       30   Nm/rad

Why these numbers (CURVED_BOARD.md has the measurements):

    K_R.  A pad of half-width r pressing with f generates at most f*r of
    contact moment -- 0.05 Nm for this 30 mm pad at 3 N -- so the knee is near
    1 Nm/rad and 30 is rigid.  Measured on the curved board, with the force
    held and the damping fixed: K_R 30 -> 0.3 takes the glyph off from 50% to
    84% and the misalignment from 8.9 to 5.5 deg.  On a FLAT board the same
    change buys nothing (95 -> 98%) and every degree of misalignment it
    produces is wander, not compliance.  So K_R is not a knob to turn down: it
    is right when the board asks and wrong when it does not, and the board is
    randomised per episode.
    z.  Lower than writing's, because the curved board's height under the glyph
    swings ~10 mm and a stiff normal turns that into force error.
    xy.  Writing's, unchanged: the job of dragging the pad along a row against
    friction is the same job as dragging a pen along a stroke.

THE PROTOCOL, repeated for every row of the raster:

    0. start of episode      xy mid,  z mid,   K_R mid
    1. APPROACH              xy mid,  z HIGH,  K_R HIGH   the pad travels
    2. PRE-CONTACT           xy LOW,  z mid,   K_R HIGH   it hovers, then descends
    3. CONTACT               xy LOW,  z LOW,   K_R LOW    it presses and wipes
    4. AFTER CONTACT         xy mid,  z mid,   K_R HIGH   it lifts

K_R HIGH before contact was deliberate -- the thought was that a wrist which is
already soft lands on a corner of the pad instead of flat on its face -- and
300 episodes say it protects nothing.  The pad arrives at 9.0 deg of the 9.0
asked either way, and the force in the 0.2 s after touchdown is 2.5 N either
way, while landing soft takes the misalignment WHILE WIPING from 0.80 of the
demand to 0.62 and the drop rate from 10% to 1%.  Before contact there is no
contact moment, so a soft K_R has nothing to yield to.  `--kr-land LOW` is that
run; the default is still HIGH because the table above is the one that was
agreed, and CURVED_BOARD.md has the comparison.

A level change is not a step in K: it ramps with a `--ramp` time constant, like
a muscle, and the tank pays for it -- for K_R too, since
`Case1Controller` meters a rotational stiffness rate as
`U_r tr(I - R_d^T R)` and gates it with the same alpha.

Keys (SAPIEN's viewer owns WASD/QE for its camera):
    1 / 2 / 3     xy   low / mid / high
    8 / 9 / 0     z    low / mid / high
    4 / 5 / 6     K_R  low / mid / high
    M             everything to mid
    G             start the next episode
    N             abort this episode, retry the same randomization
    ESC           quit (the episode in progress is discarded)

Usage:
    python3 protocol.py --board curved --texts S 7 --per-case 50 --auto-user \
                        --headless --workers 12 --out demos/wipe_curved

A PERSON STEERING, WITH A HELPER ON K_R -- the session this was built for:

    python3 protocol.py --board curved --texts S --per-case 9 --motion human \
            --arms manual v0 v1 --keep-failed --min-compliance 0 \
            --ckpt-v0 runs/w40_v0/model.pt --ckpt-v1 runs/w40_v1/model.pt \
            --time-limit 180 --out demos/human

    IJKL  slide the pad        U  press down      O  lift        SHIFT  faster
    R     this demo is finished (nothing else ends a hand-driven episode)

  The arm is drawn from balanced shuffled blocks and is NOT announced.  What the
  blind can and cannot cover is in `Arms`: v0 against v1 is blind, manual against
  the two is not, because a person obviously knows whether they are pressing the
  K_R keys themselves.
    python3 protocol.py --board flat --texts S --per-case 4 --auto-user --headless \
                        --out demos/smoke --video 2

ONE BOARD PER RUN.  The height field is baked into a collision mesh when the
scene is built, so a worker holds one board; a flat and a curved dataset are
two runs, and they can share an --out directory because the case name carries
the board.
"""
from __future__ import annotations

import argparse
import dataclasses
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
import protocol as WP  # noqa: E402  ../writing/protocol.py, for what is shared
import sim as SM  # noqa: E402
import teleop as T  # noqa: E402
import wipe_scripted as WS  # noqa: E402  the good / normal / bad tiers
import wipe_teleop as WT  # noqa: E402
from wipe_sim import CurvedWipingSim, WipingSim  # noqa: E402

LOW, MID, HIGH = WP.LOW, WP.MID, WP.HIGH
LEVEL_NAMES = WP.LEVEL_NAMES
GROUPS = WP.GROUPS
PHASE_GROUP = WP.PHASE_GROUP          # the wiper's phases ARE the writer's

# protocol phase -> (xy level, z level, K_R level)
EXPECTED = {"approach": (MID, HIGH, HIGH), "pre-contact": (LOW, MID, HIGH),
            "contact": (LOW, LOW, LOW), "after-contact": (MID, MID, HIGH)}
NO_LEVEL = np.array([-1, -1, -1])


def apply_protocol(args) -> None:
    """Let `--kr-land` move ONE cell of the table, and nowhere else.

    The cell is the one piece of the protocol that was a hypothesis rather
    than a measurement: K_R HIGH while hovering and descending, so the pad
    lands flat on its face instead of on a corner.  The collected data says
    that cell is not free.  The wrist's rotational mode at K_R ~ 0.4 has a
    period of about 6 s against a raster leg of 1.2 s, so a pad that only
    starts yielding at contact spends the whole stroke still rotating: it ends
    up 0.80 of the way to where it started, where a continuous traverse with
    the same stiffness reaches 0.45.  Setting this to LOW is the A/B.

    Mutates the module global, which the gate and the compliance score both
    read -- and it has to be called again inside each worker, because `spawn`
    imports this module fresh and would otherwise score against the default.
    """
    EXPECTED["pre-contact"] = (LOW, MID, {"LOW": LOW, "MID": MID, "HIGH": HIGH}[args.kr_land])


@dataclasses.dataclass
class Levels:
    xy: tuple = (500.0, 1000.0, 3000.0)
    z: tuple = (300.0, 600.0, 1500.0)
    kr: tuple = (0.3, 3.0, 30.0)

    def k(self, level) -> np.ndarray:
        """Level indices -> stiffness along (u, v, n)."""
        return np.array([self.xy[level[0]], self.xy[level[0]], self.z[level[1]]])

    def k_r(self, level) -> float:
        return float(self.kr[level[2]])


# --------------------------------------------------------------------------- #
# who sets the levels.  Both are ../writing/protocol.py's, with the third axis
# and this module's table; everything else about them is inherited.
# --------------------------------------------------------------------------- #
class KeyboardLevels:
    KEYS = {"1": (0, LOW), "2": (0, MID), "3": (0, HIGH),
            "8": (1, LOW), "9": (1, MID), "0": (1, HIGH),
            "4": (2, LOW), "5": (2, MID), "6": (2, HIGH)}

    def __init__(self, window):
        self.win = window
        self.level = [MID, MID, MID]

    def reset(self) -> None:
        self.level = [MID, MID, MID]

    def poll(self, t: float, group: str) -> bool:
        changed = False
        for key, (axis, lv) in self.KEYS.items():
            if self.win.key_press(key):
                self.level[axis] = lv
                changed = True
        if self.win.key_press("m"):
            self.level = [MID, MID, MID]
            changed = True
        return changed


class AutoUser(WP.AutoUser):
    """`../writing/protocol.py`'s, following THIS table, on three axes."""

    def reset(self) -> None:
        super().reset()
        self.level = [MID, MID, MID]

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


class Compliance(WP.Compliance):
    """The inherited score, against THIS module's table, plus a PER-AXIS breakdown.

    (The step override is not redundant: `EXPECTED` resolves in the module a method was
    defined in, so the inherited one would score against writing's two-axis table.)

    Why the breakdown.  `overall` requires all three axes to match at the same instant,
    so in an arm where a helper owns K_R it scores the helper and the person TOGETHER and
    is not comparable across arms -- and `save_episode` filters on it, which is why
    --arms refuses a non-zero --min-compliance.  Per axis, `axis_xy` and `axis_z` are the
    person's own work in every arm, so they are comparable; `axis_kr` is whoever owned
    K_R that episode.  The mechanism a helper arm claims is precisely that the first two
    go UP once K_R is off the person's hands, and that claim needs its own column: an
    outcome that improves tells you the demo got better, not that attention moved.
    """

    AX = ("xy", "z", "kr")

    def __init__(self, grace: float):
        super().__init__(grace)
        self.ax_hit = {a: 0 for a in self.AX}
        self.ax_n = 0

    def step(self, t: float, group: str, level) -> None:
        if group != self.group:
            self.group, self.t_change = group, t
        if t - self.t_change >= self.grace:
            self.n[group] += 1
            self.hit[group] += int(tuple(level) == EXPECTED[group])
            self.ax_n += 1
            for i, a in enumerate(self.AX):
                self.ax_hit[a] += int(level[i] == EXPECTED[group][i])

    def summary(self) -> dict:
        s = super().summary()
        for a in self.AX:
            s[f"axis_{a}"] = (self.ax_hit[a] / self.ax_n if self.ax_n else float("nan"))
        return s


# --------------------------------------------------------------------------- #
class WipeRecorder(CO.Recorder):
    """`../writing/collect.py`'s Recorder, plus what only wiping has.

    K_R is a logged state and a label like K_p: `kr` at 100 Hz beside `k_diag`,
    at policy rate in the observation, and shifted by one frame in /action.
    `n_gone` replaces `n_ink` as the progress counter -- the policy does not see
    it, it sees the board.
    """

    EXTRA_FULL = ("kr", "kr_req", "n_gone", "mis", "ask")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in self.EXTRA_FULL:
            self.full[k] = []

    def on_step(self, i, rec, sess, wr=None) -> None:
        super().on_step(i, rec, sess, wr)
        if i % self.every == 0:
            self.full["kr"].append(np.float32(rec["kr"]))
            self.full["kr_req"].append(np.float32(rec.get("kr_req", rec["kr"])))
            self.full["n_gone"].append(np.int32(self.sim.gone.sum()))
            # How flush the pad lies, and what the board asked for under it.
            # Privileged (it needs the TRUE local normal) and recorded per step
            # rather than averaged, because the protocol holds K_R LOW only
            # while wiping: an average over all pad-down time is half landing,
            # where K_R is still HIGH on purpose, and reads as no effect.
            uv = rec["contact_uvh"][:2]
            n = (self.sim.frame.normal_at(uv) if hasattr(self.sim.frame, "normal_at")
                 else self.sim.frame.normal)
            self.full["mis"].append(np.float32(np.degrees(np.arccos(
                np.clip(-rec["R"][:, 2] @ n, -1.0, 1.0)))))
            self.full["ask"].append(np.float32(np.degrees(np.arccos(
                np.clip(n @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0)))))
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
            del f["privileged/ink_uv"]          # writing's; a wiper lays none
            g = f["privileged"]
            g.create_dataset("marks_uv", data=np.asarray(self.sim.marks_uv, np.float32))
            g.create_dataset("gone", data=np.asarray(self.sim.gone))
            g.create_dataset("wear", data=np.asarray(self.sim.wear, np.float32))


# --------------------------------------------------------------------------- #
def case_name(text: str, board: str) -> str:
    return f"{WP.case_name(text)}_{board}"


def episode_spec(text: str, case: int, attempt: int, base: int):
    """Randomization for one attempt: the board pose, the glyph's size and
    placement and the operator vary; the text and the board's SHAPE do not."""
    seed = base + 1000 * case + attempt
    spec = dataclasses.replace(SM.TaskSpec.sample(seed), text=text)
    return spec, WT.WiperStyle.sample(seed), seed


AXIS_OF = {"t": 0, "n": 1, "r": 2}        # --helper-axes -> this file's (xy, z, K_R)


class Console:
    """--console: this process driven by `stiffness_helper.console`, over a pipe.

    It is a SECOND source of the three keys the operator already has -- G to start, R to
    finish, N to abort -- and a telemetry tap.  The keyboard keeps working: a person at
    the rig should not lose control of the arm because a browser tab was closed, and
    during a demo the window is where their hands already are.

    Telemetry goes out at the rate the terminal line already updates, 2 Hz, and camera
    frames at 2 Hz: a screen refresh, not a control rate.  Nothing here may block -- a
    loop holding an impedance cannot wait for a reader -- which is why the commands
    arrive through a queue a thread fills.
    """

    def __init__(self, sim):
        # `helper_user` OWNS finding the stiffness_helper checkout -- $STIFFNESS_HELPER,
        # then an ancestor of itself -- and puts it on sys.path as it loads.  Importing
        # it first is what makes the next line work: the collector runs with the wiping
        # directory as its cwd and nothing else on the path, so a bare
        # `from stiffness_helper...` raised ModuleNotFoundError AFTER the viewer had
        # already opened, which looked exactly like the simulator crashing.  It costs
        # nothing: that module imports numpy and three standard-library names.
        import helper_user  # noqa: F401
        from stiffness_helper.console import wire
        self.w = wire
        self.cmds = wire.Commands()
        # The band comes off THIS simulator, not the base class: the page draws the
        # force trace against it, and a console that drew a hard-coded 1-6 N would be
        # lying the moment a rig or a task used a different one.
        wire.emit("ready", task="wiping", keys="G start / R done / N pass",
                  band=[float(v) for v in sim.crit.force_band],
                  ladders=None, cams=[n for n, _ in self.CAMS],
                  note="the keyboard still works; the console is a second source")

    def poll(self):
        m = self.cmds.poll()
        return m["cmd"] if m else None

    def tel(self, sim, user, k_cmd, kr_cmd, rtf):
        """What the screen shows is what the ROBOT HAS, not what it was asked for.

        `k_cmd`/`kr_cmd` are the ramp's target.  The applied stiffness is `sim.ctl`,
        after the ramp and after the energy tank -- and the tank can refuse a stiffening,
        which is precisely the moment an operator needs to see rather than a number that
        says the stiffness arrived.  Both go out: `k` is what is in effect, `k_req` what
        was asked, and the page shows the gap when there is one.
        """
        f = np.asarray(sim.last["f_filt"], float)
        fn = float(sim.last["f_sensor_n"])
        ft = float(max(0.0, float(np.linalg.norm(f)) ** 2 - fn ** 2) ** 0.5)
        kd = sim.k_diag()                       # applied, along (u, v, n)
        self.w.emit("tel", t=round(float(sim.t), 3),
                    f_n=round(float(sim.last["f_n"]), 3), f_t=round(ft, 3),
                    lvl=[int(v) for v in user.level],
                    k=[float(kd[0]), float(kd[2]), float(sim.ctl.kr)],
                    k_req=[float(k_cmd[0]), float(k_cmd[2]), float(kr_cmd)],
                    rtf=round(float(rtf), 2), left=int((~sim.gone).sum()),
                    down=bool(sim.last["pen_down"]))

    CAMS = (("top", "rgb_top_camera"), ("side", "rgb_wrist_camera"))

    def frame(self, sim):
        """Both cameras.  One `observe` renders them both anyway, so the second costs
        one more JPEG of a small image -- and the side view is where contact is visible,
        which is the thing being collected."""
        try:
            obs = sim.observe(images=True)
        except Exception as e:                                 # noqa: BLE001
            self.w.emit("log", msg=f"[console] no frame: {type(e).__name__}: {e}")
            return
        for name, key in self.CAMS:
            img = obs.get(key)
            if img is None:
                continue
            a = np.asarray(img.cpu() if hasattr(img, "cpu") else img)
            while a.ndim > 3:
                a = a[0]
            # ManiSkill gives uint8, but a float frame scaled 0-1 would be destroyed by
            # a straight cast, and the failure would look like a black camera.
            if a.dtype != np.uint8:
                a = (a * 255.0 if float(np.nanmax(a)) <= 1.0 else a)
                a = np.clip(a, 0, 255).astype(np.uint8)
            j = self.w.jpeg(a[..., :3])
            if j:
                self.w.emit("frame", cam=name, jpg=j)

    def episode(self, row):
        self.w.emit("episode", row=row)


class Arms:
    """Which arm the next episode runs.  Balanced shuffled blocks: every arm appears
    once per block and the order inside a block is random, so any prefix of the session
    is near-balanced and the operator cannot predict what comes next.

    WHY NOT JUST RUN ALL OF v0 THEN ALL OF v1.  A person's hand gets better over a
    session -- it is the premise of the whole project that practice moves demo quality --
    so a blocked order charges the operator's own learning curve to whichever arm ran
    last.  Interleaving is the only way the arm difference survives it.

    WHAT THE BLIND COVERS, AND WHAT IT CANNOT.  v0 against v1 is blind: in both the
    person owns xy and z, does the identical job, and never sees what K_R is doing, so
    nothing but the robot's behaviour distinguishes them.  Manual against the two is NOT
    blind and cannot be -- the operator plainly knows whether they are pressing the K_R
    keys themselves, and hiding it would mean asking them to choose K_R with no cue and
    no read-out, which is not a baseline, it is sabotage.  So the terminal says which
    axes the person owns this episode and never which helper is behind the ones they do
    not.  --show-arm turns the blind off for a rehearsal run.
    """

    def __init__(self, arms, seed: int):
        self.arms = list(arms)
        self.rng = np.random.default_rng(seed)
        self.queue: list[str] = []

    def next(self) -> str:
        if not self.queue:
            self.queue = [str(a) for a in self.rng.permutation(self.arms)]
        return self.queue.pop()

    def undo(self, arm: str) -> None:
        """Give an arm back: an episode the operator ABORTED never happened.

        Without this an abort silently eats one arm out of its block -- the block ends
        short, the balance the whole design rests on is gone, and nothing says so.  The
        randomization is retried too (`attempt` does not advance on an abort), so the
        arm has to come back with it.
        """
        self.queue.append(arm)


class SplitLevels:
    """Levels from two sources at once: `own` owns `axes`, the protocol table owns the
    rest.  Same three methods as KeyboardLevels, so run_episode cannot tell.

    WHY THE TABLE SHOULD DRIVE xy AND z.  Those two are a lookup on the task phase, and
    the reference work measured eight architectures reproducing a phase-indexed stiffness
    table to 97.7-97.9% -- a person hand-keying it adds reaction noise and no signal, on
    top of steering.  K_R is the axis the sweep measured the task to be decided by, so it
    is the axis worth a person's attention and a helper's.  One AutoUser instance is
    shared by every arm, so the two axes nobody is being compared on behave IDENTICALLY
    in all of them.
    """

    def __init__(self, own, auto, axes):
        self.own, self.auto, self.axes = own, auto, tuple(axes)
        self._level = [MID, MID, MID]

    @property
    def level(self):
        return self._level

    def _merge(self):
        new = list(self.auto.level)
        for i in self.axes:
            new[i] = self.own.level[i]
        return new

    def reset(self) -> None:
        self.own.reset()
        self.auto.reset()
        self._level = self._merge()

    def poll(self, t: float, group: str) -> bool:
        self.own.poll(t, group)
        self.auto.poll(t, group)
        new = self._merge()
        changed = new != self._level
        self._level = new
        return changed


_HELPERS: dict = {}           # arm -> HelperUser: torch.load is not cheap


def arm_user(sim, levels, args, arm: str, table):
    """-> (user, operator record) for one arm of a human session.

    The arms differ in exactly ONE thing: who sets the axes named by --helper-axes.
    The table the other axes follow, the ramp, the energy tank, the compliance score and
    the recorder are the same objects in every arm, so a difference between arms is
    attributable to the helper and not to the path around it.

    `table` -- the operator's keyboard -- is SHARED across arms on purpose.  In a helper
    arm the person still owns the axes the helper does not, which is what a helper is
    for: it removes one of three things to think about, not all three.  Handing the
    helper arms their own table instead would have reset the person's xy/z choices on
    every arm change.
    """
    if arm == "manual":
        return table, dict(skill="human", arm=arm, axes=[],
                           person_axes=sorted(getattr(table, "axes", (0, 1, 2))))
    if arm not in _HELPERS:
        import helper_user as HU
        # BOTH generations are checkpoints: v0 is the model trained on the tier demos and
        # v1 the model trained on what v0's own sessions produced, relabeled.  That is the
        # loop this experiment is about, so a v0 arm must load v0's weights.  --ckpt-v0 is
        # nonetheless optional: with it omitted v0 falls back to the RULE, which is the
        # genuine cold start -- the version that exists before any data does -- and the
        # two are not interchangeable.  See helper_user.HelperUser on what the rule
        # reduces to on this rig.
        ckpt = args.ckpt_v0 if arm == "v0" else args.ckpt_v1
        _HELPERS[arm] = HU.HelperUser(sim, ckpt, levels, table, version=arm,
                                      axes=tuple(args.helper_axes))
    u = _HELPERS[arm]
    u.reset()
    return u, dict(u.record(), arm=arm)


_HELPER_USER = None            # one per process: torch.load is not cheap


def tiered(sim, style, seed: int, levels, args):
    """-> (style, user, operator record): who sets the levels this episode.

    Three sources, and they differ in exactly one thing so that a comparison between them
    is attributable -- the ramp, the energy tank, the compliance score and the recorder are
    the same path in all three:

        --scripted TIER   a scripted operator: the protocol's levels, two delays late
        --helper CKPT     a trained stiffness helper, its continuous K snapped to these
                          same levels (helper_user.HelperUser)
        neither           AutoUser on --reaction, as before
    """
    if getattr(args, "helper", None):
        global _HELPER_USER
        if _HELPER_USER is None:
            import helper_user as HU
            _HELPER_USER = HU.HelperUser(sim, args.helper, levels,
                                         AutoUser(seed, tuple(args.reaction)),
                                         version=args.helper_version or "v1",
                                         axes=tuple(args.helper_axes))
        _HELPER_USER.reset()
        return style, _HELPER_USER, _HELPER_USER.record()
    if not getattr(args, "scripted", None):
        return style, AutoUser(seed, tuple(args.reaction)), None
    style, sk = WS.apply(style, WS.skill(args.scripted), seed)
    return style, AutoUser(seed, sk.reaction()), dict(
        skill=WS.name(args.scripted),
        preset=dataclasses.asdict(WS.skill(args.scripted)),
        motion_delay=float(sk.motion_delay), stiffness_delay=float(sk.stiffness_delay))


def run_episode(sim, spec, style, user, levels: Levels, args, recorder=None,
                viewer=None, cues: bool = True, hide=()) -> dict | None:
    """One episode with levels from `user`.  `hide` is the axes whose cue and read-out
    are suppressed, because the person does not own them this episode (see `Arms`)."""
    sim.reset(spec)
    sess = T.TeleopSession(sim)
    # --motion human puts a PERSON where the synthetic wiper was.  writing/protocol.py's
    # KeyboardWriter drops in unchanged: WipingSim subclasses WritingSim, so it has the
    # W frame, K0, target.strokes and the pen_down flag that class reads, and the loop
    # below already calls exactly the attributes it offers.  Nobody plans the strokes
    # now, so `wr.phase` is read off the hand and only R ends the episode.
    human = getattr(args, "motion", "wiper") == "human"
    if human:
        if viewer is None:
            raise RuntimeError("--motion human needs a window")
        wr = WP.KeyboardWriter(viewer.window, sim,
                               latch=getattr(args, "keys", "latch") == "latch")
    else:
        wr = WT.SyntheticWiper(sim, dataclasses.replace(style, hover_dwell=args.dwell,
                                                        v_desc=args.v_desc,
                                                        v_travel=args.v_travel))
    shown = [i for i in range(3) if i not in tuple(hide)]
    names = ("xy", "z", "K_R")

    def lv_str(lv, w=4):
        return "  ".join(f"{names[i]} {LEVEL_NAMES[lv[i]]:<{w}}" for i in shown)

    user.reset()
    comp = Compliance(args.grace)
    k_cmd = levels.k(user.level)
    kr_cmd = levels.k_r(user.level)
    dt = sim.dt
    i = 0
    last_group = None
    last, acc, last_print = time.time(), 0.0, -1.0
    # REAL-TIME FACTOR, shown to the operator.  A helper arm renders two cameras and runs
    # a forward pass ten times a second on top of the viewer, and if that does not fit in
    # wall time the pad answers the keys late -- which a person compensates for, so the
    # arm would be scored on the lag and not on the stiffness.  The loop already catches
    # up to 0.1 s per frame and silently falls behind past that, so it has to say so.
    t_wall0, t_sim0 = time.time(), sim.t
    con = getattr(args, "_console", None)
    last_frame, aborted = -1e9, False
    while not wr.done and sim.t < spec.time_limit:
        n_steps = 1
        if viewer is not None:
            w = viewer.window
            if w.should_close or w.key_down("esc"):
                raise KeyboardInterrupt
            if w.key_press("n"):
                print("\n[aborted] same randomization again")
                return None
            if human and w.key_press("r"):
                print("\n[finished by hand]")
                wr.done = True
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
            # Pacing to the wall clock belongs to the WINDOW: a person watching it needs
            # the simulation to run at their speed.  Inserting the console block above
            # had swallowed these five lines, which left a windowless console-driven
            # collector stepping once per iteration.
            now = time.time()
            acc += min(now - last, 0.1)
            last = now
            n_steps = min(int(acc / dt), 50)
            acc -= n_steps * dt
        for _ in range(n_steps):
            group = WP.group_of(wr)
            if user.poll(sim.t, group) and viewer is not None and shown:
                print(f"\n    {lv_str(user.level)}")
            if cues and group != last_group:
                want = EXPECTED[group]
                cue = ", ".join(f"{names[i]} {LEVEL_NAMES[want[i]]}" for i in shown)
                print(f"\n  [{sim.t:5.1f}s] row {min(wr.k + 1, len(wr.strokes))}/{len(wr.strokes)} "
                      f"{group.upper():<13}" + (f" -> {cue}" if cue else ""))
            last_group = group
            f_h, _ = wr.act(sim.t, sess.x_m, sess.v_m, sess.f_fb, sim.last["p"])
            # the level change ramps, like a muscle; the tank pays for both
            a_ramp = min(1.0, dt / args.ramp)
            k_cmd = k_cmd + (levels.k(user.level) - k_cmd) * a_ramp
            kr_cmd = kr_cmd + (levels.k_r(user.level) - kr_cmd) * a_ramp
            comp.step(sim.t, group, user.level)
            rec = sess.step(f_h, k_cmd, kr_cmd)
            rec.update(phase=T.PHASES.index(wr.phase), stroke=wr.k,
                       k_level=np.array(user.level))
            # A helper sees EXACTLY what the recorder logs -- the same `rec` -- so what it
            # is given at run time cannot drift from what it was trained on.  Anything
            # else is a `user` and ignores this.
            if hasattr(user, "observe"):
                user.observe(sim.t, rec)
            if recorder is not None:
                recorder.on_step(i, rec, sess, wr)
            i += 1
            if wr.done:
                break
        if viewer is not None:
            sim.env.render_human()
        if sim.t - last_print > 0.5:
            last_print = sim.t
            rtf = (sim.t - t_sim0) / max(1e-6, time.time() - t_wall0)
            # NOT WHILE THE CONSOLE IS ATTACHED.  This line ends with \r and no
            # newline, so the telemetry JSON written immediately after it landed on the
            # SAME line and stopped being JSON -- the page showed no force, no stiffness
            # and no camera while the demos themselves recorded perfectly.  The page
            # carries all of it anyway, so with a console there is nothing to print.
            if viewer is not None and con is None:
                print(f"\r   t {sim.t:5.1f}s  {lv_str(user.level)}"
                      f"  board {sim.last['f_n']:4.1f} N  left {(~sim.gone).sum():3d}"
                      f"  {rtf:4.2f}x{' LAGGING' if rtf < 0.9 else '        '}",
                      end="", flush=True)
            if con is not None:
                con.tel(sim, user, k_cmd, kr_cmd, rtf)
                if sim.t - last_frame > 0.5:
                    last_frame = sim.t
                    con.frame(sim)
    if aborted:
        return None
    res = sim.score()
    res["finished"] = bool(wr.done)
    if not wr.done:
        res["success"] = False
        res["fail_reason"] = ",".join(filter(None, [res["fail_reason"], "timeout"]))
    res["compliance"] = comp.summary()
    return res


def save_episode(sim, rec, res, spec, style, levels, args, text: str, c: int, seed: int,
                 source: str, out: pathlib.Path, operator=None):
    comp = res["compliance"]
    ok_comp = comp["overall"] >= args.min_compliance
    keep = bool(res["success"] and ok_comp)
    path = None
    if keep or args.keep_failed:
        d = out / case_name(text, args.board)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ep_{seed:05d}.h5"
        rec.save(path, dict(
            text=text, case=c, board=args.board, success=bool(res["success"]),
            compliant=bool(ok_comp), source=source,
            spec=dataclasses.asdict(spec), style=dataclasses.asdict(style),
            protocol=dict(levels=dataclasses.asdict(levels), expected=EXPECTED,
                          phase_group=PHASE_GROUP, dwell=args.dwell, ramp=args.ramp,
                          grace=args.grace, v_desc=args.v_desc, v_travel=args.v_travel,
                          reaction=list(args.reaction), level_names=LEVEL_NAMES,
                          axes=["xy", "z", "kr"]),
            compliance=comp, operator=operator,
            master=dataclasses.asdict(T.MasterParams()),
            criteria=dataclasses.asdict(sim.crit),
            gains={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in dataclasses.asdict(sim.ctl.g).items()},
            metrics={k: v for k, v in res.items() if k not in ("checks", "compliance")},
            phases=T.PHASES, sim_dt=sim.dt, log_hz=100.0, policy_hz=10.0,
            writing_frame=sim.W.tolist()))
        if rec.video:
            import imageio.v2 as imageio
            imageio.mimsave(path.with_suffix(".mp4"), rec.video, fps=5, quality=7,
                            macro_block_size=1)
    reason = ", ".join(filter(None, [res["fail_reason"], "" if ok_comp else "protocol"]))
    # `kept` IS "A FILE EXISTS" AND `scored` IS "IT MET THE CRITERIA".  One word for
    # two ideas is what made a saved demo read as discarded in the console and kept the
    # on-screen counter frozen for a whole session; the distinction is now in the name.
    row = dict(case=c, text=text, board=args.board, seed=seed,
               kept=bool(path), scored=bool(keep),
               saved=str(path) if path else None, success=bool(res["success"]),
               reason=reason, compliance=comp, erased=res["erased"],
               in_band=res["in_band"], peak_force=res["peak_force"], t=res["t"])
    return keep, path, row


def verdict_line(row: dict, done: int, per_case: int) -> str:
    comp = row["compliance"]
    v = ("GOOD" if row["scored"] else
         f"saved, did not score: {row['reason']}" if row["kept"] else
         f"not kept: {row['reason']}")
    return (f"[{v}] erased {row['erased']:.2f} in-band {row['in_band']:.2f} "
            f"peak {row['peak_force']:.1f} N | protocol "
            + " ".join(f"{g} {100 * comp[g]:.0f}%" for g in GROUPS if comp[g] == comp[g])
            + f" | {done}/{per_case}")


# --------------------------------------------------------------------------- #
_SIM = None


def _worker_init(board: str, image_size: int, cameras: bool, video: bool) -> None:
    global _SIM
    _SIM = (CurvedWipingSim if board == "curved" else WipingSim)(
        cameras=cameras, image_size=image_size,
        render_mode="rgb_array" if video else None)


def _run_job(job) -> dict:
    c, text, attempt, args = job
    apply_protocol(args)                  # `spawn` re-imported this module
    levels = Levels(xy=tuple(args.k_xy), z=tuple(args.k_z), kr=tuple(args.k_r))
    spec, style, seed = episode_spec(text, c, attempt, args.seed_base)
    style, user, operator = tiered(_SIM, style, seed, levels, args)
    rec = WipeRecorder(_SIM, images=not args.no_cameras,
                       video_every=2 if attempt < args.video else 0)
    res = run_episode(_SIM, spec, style, user, levels, args, recorder=rec, cues=False)
    _, _, row = save_episode(_SIM, rec, res, spec, style, levels, args, text, c, seed,
                             "protocol-scripted" if operator else "protocol-auto",
                             pathlib.Path(args.out), operator=operator)
    row["attempt"] = attempt
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("flat", "curved"), default="curved")
    ap.add_argument("--texts", nargs="+", default=["S", "7", "<star>"])
    ap.add_argument("--per-case", type=int, default=50)
    ap.add_argument("--out", default="demos/wiping")
    ap.add_argument("--seed-base", type=int, default=70_000)
    ap.add_argument("--k-xy", type=float, nargs=3, default=[500.0, 1000.0, 3000.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-z", type=float, nargs=3, default=[300.0, 600.0, 1500.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-r", type=float, nargs=3, default=[0.3, 3.0, 30.0],
                    metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--dwell", type=float, default=1.0)
    ap.add_argument("--v-desc", type=float, default=0.010)
    ap.add_argument("--v-travel", type=float, default=0.030)
    ap.add_argument("--ramp", type=float, default=0.15)
    ap.add_argument("--grace", type=float, default=0.6)
    ap.add_argument("--kr-land", choices=("LOW", "MID", "HIGH"), default="HIGH",
                    help="K_R while hovering and descending.  HIGH is the protocol "
                         "(land flat); LOW is the A/B -- see apply_protocol()")
    ap.add_argument("--min-compliance", type=float, default=0.8)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--auto-next", action="store_true")
    ap.add_argument("--auto-user", action="store_true")
    ap.add_argument("--helper", default=None, metavar="CKPT",
                    help="a trained stiffness_helper checkpoint sets the levels "
                         "(helper_user.py).  Implies --auto-user.  Point STIFFNESS_HELPER "
                         "at that repository if it is not in the default place")
    ap.add_argument("--helper-version", default=None,
                    help="what to call it in the logs, e.g. v0 or v1")
    ap.add_argument("--helper-axes", nargs="+", default=["r"], choices=("t", "n", "r"),
                    help="which axes the helper owns; the rest keep following the table. "
                         "Default r alone: that is the axis the tier sweep measured this "
                         "task to be decided by")
    ap.add_argument("--attempts", type=int, default=None,
                    help="stop a case after this many attempts, kept or not.  A tier "
                         "whose demos mostly FAIL would otherwise never finish: "
                         "--per-case counts kept demos, and that is the point of a bad "
                         "tier (writing's bad tier succeeded 27% of the time)")
    ap.add_argument("--scripted", nargs="+", default=None,
                    metavar="TIER|MOTION STIFFNESS",
                    help="a scripted operator with two delays: good / normal / bad from "
                         "wipe_scripted.SKILLS, or two floats (seeing delay s, level "
                         "reaction s).  Implies --auto-user; the files record the tier "
                         "in their `operator` attribute")
    ap.add_argument("--reaction", type=float, nargs=2, default=[0.20, 0.45],
                    metavar=("MIN", "MAX"))
    ap.add_argument("--motion", choices=("wiper", "human"), default="wiper",
                    help="who moves the pad: the synthetic wiper, or a person on "
                         "IJKL/U/O (writing/protocol.py's KeyboardWriter; R ends it)")
    ap.add_argument("--console", action="store_true",
                    help="driven by stiffness_helper.console over the pipe: telemetry "
                         "out, COLLECT/DONE/PASS in, alongside the keyboard")
    ap.add_argument("--keys", choices=("latch", "hold"), default="latch",
                    help="--motion human: `latch` taps to set a direction and needs no "
                         "two keys at once; `hold` is the original held-key hand")
    ap.add_argument("--auto-axes", nargs="+", default=None,
                    help="axes the protocol table drives in EVERY arm, so nobody is "
                         "compared on them.  Default `t n` for a human session and "
                         "`none` otherwise, which leaves existing collection untouched")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=("manual", "v0", "v1"),
                    help="interleave these arms, one draw per episode, not announced: "
                         "manual = the person sets every axis, v0/v1 = a helper owns "
                         "--helper-axes and the person owns the rest")
    ap.add_argument("--ckpt-v0", default=None, metavar="CKPT",
                    help="the v0 checkpoint for --arms: the model trained on the tier "
                         "demos.  Omit it to run v0 as the RULE instead -- the cold "
                         "start, which on this rig is close to a constant K_R")
    ap.add_argument("--ckpt-v1", default=None, metavar="CKPT",
                    help="the v1 checkpoint for --arms: trained on what v0 collected")
    ap.add_argument("--arm-seed", type=int, default=0,
                    help="the arm schedule, so a session is reproducible and auditable")
    ap.add_argument("--show-arm", action="store_true",
                    help="announce the arm: for a rehearsal, never for a real session")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="override the spec's 90 s; a hand is slower than the wiper")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--video", type=int, default=0)
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()
    if args.scripted or args.helper:
        args.auto_user = True        # the tier or the helper sets the levels, not a keyboard
    if args.scripted and args.helper:
        ap.error("--scripted and --helper both set the levels; pick one")
    if args.arms:
        if args.auto_user or args.scripted or args.helper:
            ap.error("--arms chooses who sets the levels; drop --auto-user/--scripted/--helper")
        if args.headless:
            ap.error("--arms is a human session: it needs a window")
        if "v1" in args.arms and not args.ckpt_v1:
            ap.error("--arms v1 needs --ckpt-v1")
        if "v0" in args.arms and not args.ckpt_v0:
            print("  [arms] no --ckpt-v0: v0 runs as the RULE, not the first-generation "
                  "model.  On this rig that is close to a constant K_R (the F/T reading "
                  "carries no moment), so it is a cold-start bar, not a v0-vs-v1 "
                  "comparison.  Pass --ckpt-v0 runs/w40_v0/model.pt for that.")
        if len(set(args.arms)) < 2:
            ap.error("--arms wants at least two arms to compare")
        if args.min_compliance > 0 or not args.keep_failed:
            ap.error("--arms needs --min-compliance 0 --keep-failed.  Compliance scores "
                     "the axes the HELPER owns too, so keeping only compliant episodes "
                     "drops them by an arm-dependent rule -- the helper arms would be "
                     "filtered on the helper's own output and the comparison would be "
                     "biased before it started.  Filter afterwards, in the report.")
        args.arms = sorted(set(args.arms))
    if args.auto_axes is None:
        args.auto_axes = (["t", "n"] if (args.motion == "human" or args.arms)
                          else ["none"])
    if args.auto_axes != ["none"] and not set(args.auto_axes) <= set(AXIS_OF):
        ap.error(f"--auto-axes takes {' '.join(AXIS_OF)} or none")
    if args.motion == "human" and args.headless:
        ap.error("--motion human needs a window: nobody can press keys without one")
    if args.console and args.workers > 1:
        ap.error("--console drives one episode at a time; --workers is for a batch")
    if args.headless and not args.auto_user:
        ap.error("--headless needs --auto-user: nobody can press keys without a window")
    if (args.workers > 1 or args.video) and not args.headless:
        ap.error("--workers and --video need --headless")

    apply_protocol(args)
    levels = Levels(xy=tuple(args.k_xy), z=tuple(args.k_z), kr=tuple(args.k_r))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "attempts.jsonl"
    done = {c: len(list((out / case_name(t, args.board)).glob("ep_*.h5")))
            if (out / case_name(t, args.board)).exists() else 0
            for c, t in enumerate(args.texts)}
    attempt = {c: 0 for c in range(len(args.texts))}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            e = json.loads(line)
            if e["text"] in args.texts and e.get("board") == args.board:
                c = args.texts.index(e["text"])
                attempt[c] = max(attempt[c], e["attempt"] + 1)

    def log(row: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"  board   {args.board}")
    print("  table   " + "  ".join(
        f"{g}:" + "/".join(LEVEL_NAMES[i][0] for i in EXPECTED[g]) for g in GROUPS))
    print("  levels  xy " + " / ".join(f"{k:.0f}" for k in levels.xy)
          + "   z " + " / ".join(f"{k:.0f}" for k in levels.z) + " N/m"
          + "   K_R " + " / ".join(f"{k:g}" for k in levels.kr) + " Nm/rad")
    for c, t in enumerate(args.texts):
        print(f"  case {c} {t!r:10} {done[c]}/{args.per_case} collected")

    if args.headless and args.workers > 1:
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(
                args.workers, initializer=_worker_init,
                initargs=(args.board, args.image_size, not args.no_cameras,
                          args.video > 0)) as pool:
            def _room(c):
                n = args.per_case - done[c]
                return n if args.attempts is None else min(n, args.attempts - attempt[c])

            while any(_room(c) > 0 for c in done):
                jobs = []
                for c, text in enumerate(args.texts):
                    for _ in range(max(0, _room(c))):
                        jobs.append((c, text, attempt[c], args))
                        attempt[c] += 1
                for row in pool.imap_unordered(_run_job, jobs):
                    done[row["case"]] += int(row["kept"])
                    log(row)
                    print(f"  case {row['case']} seed {row['seed']}  "
                          + verdict_line(row, done[row["case"]], args.per_case), flush=True)
        print(f"\n  {sum(done.values())} demos in {(time.time() - t0) / 60:.1f} min")
        print("  " + "  ".join(f"{t!r}: {done[c]}/{args.per_case}"
                               for c, t in enumerate(args.texts)))
        return

    sim = (CurvedWipingSim if args.board == "curved" else WipingSim)(
        cameras=not args.no_cameras, image_size=args.image_size,
        render_mode=None if args.headless else "human")
    viewer = None if args.headless else sim.u.render_human()

    def next_case():
        todo = [c for c in range(len(args.texts))
                if done[c] < args.per_case
                and (args.attempts is None or attempt[c] < args.attempts)]
        return None if not todo else min(todo, key=lambda c: (done[c], c))

    # WHO OWNS WHICH AXIS is worked out before anything describes it.  The key map below
    # prints only the axes the person sets, so computing these afterwards made the first
    # line of a session a NameError -- on the viewer path, which nothing headless runs.
    auto_ax = () if args.auto_axes == ["none"] else tuple(AXIS_OF[a] for a in args.auto_axes)
    person_ax = tuple(i for i in range(3) if i not in auto_ax)
    helper_ax = tuple(AXIS_OF[a] for a in args.helper_axes)

    if viewer is not None:
        print(__doc__.split("Usage:")[0].split("THE PROTOCOL")[1].split("A level change")[0])
        print("  levels  " + "   ".join(
            n + " " + k for n, k, i in (("xy", "1/2/3", 0), ("z", "8/9/0", 1),
                                        ("K_R", "4/5/6", 2)) if i in person_ax)
            + ("   (the table drives " + "/".join(("xy", "z", "K_R")[i]
                                                  for i in range(3) if i not in person_ax)
               + ")" if len(person_ax) < 3 else "")
            + "   G go   N abort   ESC quit")
        if args.motion == "human":
            print("  hand    " + ("I/J/K/L tap to sweep (tap again to stop)   U press   "
                                  "O lift   F faster   X stop all   R demo finished"
                                  if args.keys == "latch" else
                                  "IJKL slide   U press   O lift   SHIFT faster   "
                                  "R demo finished"))
        if args.arms:
            print(f"  arms    {' '.join(args.arms)}  interleaved, "
                  f"{'ANNOUNCED' if args.show_arm else 'not announced'}"
                  f" (schedule seed {args.arm_seed})")
        print()

    # The console is built AFTER the simulator, so a scene that fails to load is a plain
    # traceback on stderr and not a half-open pipe the console has to time out.
    con = Console(sim) if args.console else None
    args._console = con

    if args.auto_user:
        table = None
    elif auto_ax:
        table = SplitLevels(KeyboardLevels(viewer.window),
                            AutoUser(args.seed_base, tuple(args.reaction)), person_ax)
    else:
        table = KeyboardLevels(viewer.window)
    user = table
    arms = Arms(args.arms, args.arm_seed) if args.arms else None
    hide = tuple(i for i in range(3) if i not in person_ax)
    try:
        while (c := next_case()) is not None:
            text = args.texts[c]
            spec, style, seed = episode_spec(text, c, attempt[c], args.seed_base)
            if args.time_limit:
                spec = dataclasses.replace(spec, time_limit=args.time_limit)
            operator = None
            if arms is not None:
                arm = arms.next()
                user, operator = arm_user(sim, levels, args, arm, table)
                # The axes the person does NOT own go quiet -- no cue, no read-out --
                # because a read-out of the helper's choices is both a hint and a tell.
                # What the PERSON owns this episode: their axes, minus the ones a
                # helper took.  Everything else goes quiet -- a cue for an axis you
                # cannot set is noise, and a read-out of a helper's choices is a tell.
                owned = (person_ax if arm == "manual"
                         else tuple(i for i in person_ax if i not in helper_ax))
                hide = () if args.show_arm else tuple(i for i in range(3)
                                                      if i not in owned)
            elif args.auto_user:
                style, user, operator = tiered(sim, style, seed, levels, args)
            print(f"\n=== case {c} {text!r}  demo {done[c] + 1}/{args.per_case}  (seed {seed})")
            if arms is not None:
                print("    you set: " + (", ".join(("xy", "z", "K_R")[i] for i in owned)
                                         or "nothing -- just move the pad")
                    + (f"    [arm {arm}]" if args.show_arm else ""))
            if (viewer is not None or args.console) and not args.auto_next:
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
                    if args.console:
                        # NOT `c`.  This loop is inside `while (c := next_case())`, and
                        # binding the console's command to the same name destroyed the
                        # case index: starting a demo from the browser saved it under
                        # case "collect" and then raised KeyError on `attempt[c]`.
                        # Pressing G in the window broke out before this ran, so only
                        # the console path was affected -- and only after the demo.
                        cmd = con.poll()
                        if cmd == "collect":
                            break
                        if cmd == "stop":
                            raise KeyboardInterrupt
                    if viewer is None:
                        time.sleep(0.02)
            rec = WipeRecorder(sim, images=not args.no_cameras)
            res = run_episode(sim, spec, style, user, levels, args, recorder=rec,
                              viewer=viewer, cues=viewer is not None, hide=hide)
            if res is None:
                if arms is not None:
                    arms.undo(arm)
                if args.console:
                    # A DISCARDED EPISODE IS STILL AN EVENT.  Both N and the console's
                    # PASS land here, and without this the console would sit in
                    # "recording" for the rest of the session waiting for an episode
                    # that was thrown away.
                    con.episode(dict(case=c, text=text, board=args.board, seed=seed,
                                     attempt=attempt[c], kept=False, scored=False,
                                     success=False,
                                     reason="passed", arm=(operator or {}).get("arm"),
                                     motion=args.motion))
                continue
            if operator is not None and hasattr(user, "record"):
                # AFTER the episode.  `record()` taken at arm-selection time reported
                # calls=0 every time -- reset() had just zeroed it -- so the log could
                # not answer the first question anyone asks of a blinded session: did
                # the helper run at all.
                operator = dict(operator, **user.record())
            keep, path, row = save_episode(
                sim, rec, res, spec, style, levels, args, text, c, seed,
                "protocol-human" if args.motion == "human" else
                "protocol-scripted" if operator else
                ("protocol-auto" if args.auto_user else "protocol-keyboard"), out,
                operator=operator)
            row["attempt"] = attempt[c]
            row["arm"] = (operator or {}).get("arm")
            row["calls"] = (operator or {}).get("calls")
            row["motion"] = args.motion
            # COUNT WHAT IS ON DISK, which is how `done` was INITIALISED: by globbing
            # ep_*.h5.  Incrementing on `keep` instead meant that with --keep-failed
            # every demo wrote a file and the counter never moved -- it read 3/60 for a
            # whole session while files piled up, so it looked like nothing was being
            # saved -- and then jumped on the next run, when the glob counted them all.
            done[c] += int(bool(path))
            attempt[c] += 1
            log(row)
            if args.console:
                # AFTER save_episode: the console's loop advances on this event, so it
                # must not arrive before the file it refers to exists on disk.
                #
                # `kept` IS "A FILE EXISTS", NOT "THE TASK SUCCEEDED".  `keep` above means
                # the episode met the success criteria AND the protocol compliance, which
                # is a quality verdict and belongs in `success`, where it already is.  A
                # demo the operator chose to keep, and whose file is on disk, must count
                # toward the generation -- this project exists to collect imperfect demos
                # and recover their labels in hindsight, so scoring them as discards is
                # backwards, and it made the target unreachable while an operator was
                # still learning.  Only PASS discards, and that path never gets here.
                con.episode(dict(row, compliance=row["compliance"]["overall"]))
            print("\n" + verdict_line(row, done[c], args.per_case))
    except KeyboardInterrupt:
        print("\n[quit] episode in progress discarded")
    print("\n" + "  ".join(f"{t!r}: {done[c]}/{args.per_case}" for c, t in enumerate(args.texts)))
    sim.close()


if __name__ == "__main__":
    main()
