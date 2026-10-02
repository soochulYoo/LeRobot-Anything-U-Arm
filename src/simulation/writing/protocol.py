"""Stiffness-protocol collection: you move the pen and you set the stiffness.

Who moves the pen is `--motion`:

    human    (default) the keys push on the master and the slave follows, the
             same bilateral loop as interactive.py: you travel, press, write
             and lift yourself
    writer   the synthetic writer (teleop.SyntheticWriter) does all of that,
             and you only set the stiffness

Either way the Case 1 stiffness comes from the keyboard, in three levels per
axis group, translation only:

              xy (along the paper)     z (into the paper)
    low            500 N/m                 500 N/m
    mid           1000 N/m                1000 N/m
    high          3000 N/m                3000 N/m

Why these numbers (a scripted person following the protocol, 3 texts x 12
randomizations each):
    xy low 300 failed coverage on high-friction paper (3/36): mu 0.48 x 3.7 N
    of friction drags a 300 N/m pen ~6 mm behind its reference, past the 3 mm
    tolerance on curves.  At 500 it wrote 36/36.
    z mid at touch-down lands harder than the writer's own soft landing: peaks
    reached 10 N of the 12 N tear limit at the writer's 15-20 mm/s descent.
    The writer therefore descends at `--v-desc` = 10 mm/s here -- slower, as a
    careful hand does before contact -- rather than changing the protocol.
    The approach lasted 0.5-1 s at the writer's own travel speed, so a person
    reacting in 0.5-1 s held "z high" for only 33-43% of it.  The writer
    travels at `--v-travel` = 30 mm/s here, which makes the approach 1-3 s.

THE PROTOCOL, repeated for every stroke of the text:

    0. start of episode      xy mid,  z mid
    1. APPROACH              xy mid,  z HIGH    the pen travels to the stroke's start
    2. PRE-CONTACT           xy LOW,  z mid     the pen hovers, then descends
    3. CONTACT               xy LOW,  z mid     the pen presses and writes
    4. AFTER CONTACT         xy mid,  z mid     the pen lifts (and, at the end, rises)

A level change is not a step in K: it ramps with a `--ramp` time constant, like
a muscle, and the tank pays for it.

WHICH STEP IT IS.  The writer hovers `--dwell` seconds over each stroke's start
before it descends, so there is time for step 2.  When you move the pen, the
step is read off your hand: holding "press" above the paper is PRE-CONTACT,
ink is CONTACT, letting go is AFTER CONTACT, anything else is APPROACH.

Every episode is checked against the protocol: the fraction of time the levels
matched the table above, ignoring `--grace` seconds after each phase change for
reaction time.  Only episodes that SUCCEED and COMPLY are kept.

Keys (SAPIEN's viewer owns WASD and F for its camera):
    1 / 2 / 3     xy  low / mid / high
    Z / X / C     z   low / mid / high      (8 / 9 / 0 do the same)
    M or V        everything to mid
    G             start the next episode
    N             abort this episode, retry the same randomization
    ESC           quit (the episode in progress is discarded)
  with --motion human, the other hand:
    J / L         push left / right along the writing line
    I / K         push up / down the letter (away from / toward you)
    U (hold)      press the pen into the paper (3.5 N at rest)
    O             lift
    SHIFT         faster along the paper and up -- never harder into it
    R             finished writing: score, save, next episode

A keyboard cannot push back, so the force a haptic master would reflect is
drawn in a floating window in the viewer (panel.py; `--no-panel` removes it).

Usage:
    python3 protocol.py --texts S 7 "<star>" --per-case 50 --out demos/human
    # the pen writes by itself, you set only the stiffness
    python3 protocol.py --texts S 7 "<star>" --per-case 50 --motion writer --out demos/protocol
    # no person: a scripted operator at the keys, good / normal / bad (scripted.py)
    python3 protocol.py --texts S 7 "<star>" --scripted bad --attempts 20 --keep-failed \
                        --min-compliance 0 --headless --workers 12 --out demos/scripted/bad
    # no person: stiffness switches exactly at each phase, 12 episodes at a time
    python3 protocol.py --texts S 7 "<star>" --per-case 50 --auto-user --reaction 0 0 \
                        --headless --workers 12 --video 2 --out demos/protocol_auto
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sim as SM  # noqa: E402
import teleop as T  # noqa: E402

LEVEL_NAMES = ("LOW", "MID", "HIGH")
LOW, MID, HIGH = 0, 1, 2

# writer phase -> protocol phase
PHASE_GROUP = {"travel": "approach", "settle": "pre-contact", "descend": "pre-contact",
               "press": "contact", "stroke": "contact", "finish": "contact",
               "lift": "after-contact", "done": "after-contact", "finished": "after-contact"}
GROUPS = ["approach", "pre-contact", "contact", "after-contact"]
# protocol phase -> (xy level, z level)
EXPECTED = {"approach": (MID, HIGH), "pre-contact": (LOW, MID),
            "contact": (LOW, MID), "after-contact": (MID, MID)}


@dataclasses.dataclass
class Levels:
    xy: tuple = (500.0, 1000.0, 3000.0)
    z: tuple = (500.0, 1000.0, 3000.0)

    def k(self, level) -> np.ndarray:
        """(xy, z) level indices -> stiffness along (u, v, n)."""
        return np.array([self.xy[level[0]], self.xy[level[0]], self.z[level[1]]])


# --------------------------------------------------------------------------- #
# who sets the levels
# --------------------------------------------------------------------------- #
class KeyboardLevels:
    # z has two sets of keys: Z/X/C sit under the hand that is not moving the
    # pen, and 8/9/0 are the ones the writer-only mode started with.  For the
    # same reason V does what M does -- and M is one key from N, which aborts.
    KEYS = {"1": (0, LOW), "2": (0, MID), "3": (0, HIGH),
            "z": (1, LOW), "x": (1, MID), "c": (1, HIGH),
            "8": (1, LOW), "9": (1, MID), "0": (1, HIGH)}

    def __init__(self, window):
        self.win = window
        self.level = [MID, MID]

    def reset(self) -> None:
        self.level = [MID, MID]

    def poll(self, t: float, group: str) -> bool:
        changed = False
        for key, (axis, lv) in self.KEYS.items():
            if self.win.key_press(key):
                self.level[axis] = lv
                changed = True
        if self.win.key_press("m") or self.win.key_press("v"):
            self.level = [MID, MID]
            changed = True
        return changed


class AutoUser:
    """A person who follows the protocol perfectly, a reaction time late.  For
    dry runs and for checking that the protocol can be done at all."""

    def __init__(self, seed: int, reaction=(0.20, 0.45)):
        self.rng = np.random.default_rng(90_000 + seed)
        self.reaction = reaction
        self.reset()

    def reset(self) -> None:
        self.level = [MID, MID]
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


# --------------------------------------------------------------------------- #
class Compliance:
    """Fraction of time the levels matched the protocol, per protocol phase,
    not counting `grace` seconds after each phase change."""

    def __init__(self, grace: float):
        self.grace = grace
        self.t_change = 0.0
        self.group = None
        self.hit = {g: 0 for g in GROUPS}
        self.n = {g: 0 for g in GROUPS}

    def step(self, t: float, group: str, level) -> None:
        if group != self.group:
            self.group, self.t_change = group, t
        if t - self.t_change >= self.grace:
            self.n[group] += 1
            self.hit[group] += int(tuple(level) == EXPECTED[group])

    def summary(self) -> dict:
        per = {g: (self.hit[g] / self.n[g] if self.n[g] else float("nan")) for g in GROUPS}
        tot = sum(self.n.values())
        per["overall"] = sum(self.hit.values()) / tot if tot else float("nan")
        return per


class KeyboardWriter:
    """A person's hand in the synthetic writer's place.  The keys push on the
    master (interactive.KeyboardHuman), the slave follows, and run_episode reads
    the attributes it reads from teleop.SyntheticWriter.

    Nobody plans the strokes here, so the phase is read off the hand:

        travel  -> descend   press is held
        descend -> stroke    the pen is down (pressure >= ink_force)
        descend -> travel    press let go before the paper was reached
        stroke  -> lift      press let go, or lift held
        lift    -> travel    the paper is no longer felt: one stroke done
        lift    -> stroke    press held again before the pen came off

    "Let go" is RELEASE seconds without press: a finger re-seating itself on
    the key is not a lift.  The phases are a subset of teleop.PHASES, so the
    recorded `phase` reads the same in both kinds of demo.
    """
    RELEASE = 0.15     # s
    F_CLEAR = 0.2      # N of pressure, below which the pen is off the paper

    def __init__(self, window, sim, latch: bool = False):
        import interactive as I
        self.win, self.sim = window, sim
        # `latch` is the hand that needs no two keys at once: the same forces, tapped
        # instead of held.  See interactive.LatchedHand for why that is not a preference.
        self.hand = (I.LatchedHand if latch else I.KeyboardHuman)(window, sim.W, sim.K0)
        self.strokes = sim.target.strokes
        self.phase, self.k, self.done = "travel", 0, False
        self._idle = 0.0

    @staticmethod
    def params() -> dict:
        import interactive as I
        return dict({k: getattr(I.KeyboardHuman, k)
                     for k in ("F_PLANE", "F_WRITE", "F_NORMAL", "B_PLANE", "B_WRITE",
                               "B_NORMAL", "F_IDLE")},
                    RELEASE=KeyboardWriter.RELEASE, F_CLEAR=KeyboardWriter.F_CLEAR)

    def act(self, t: float, x_m, v_m, f_fb, tip) -> tuple[np.ndarray, None]:
        """(f_h, None): the stiffness is not the hand's to return here."""
        f_h, _ = self.hand.wrench(t, x_m, v_m, f_fb)
        # The hand answers these, not the keys: a latched hand means something different
        # by "pressing" than a held one, and only it knows which.
        lift, press = self.hand.lift, self.hand.press
        self._idle = 0.0 if press else self._idle + self.sim.dt
        let_go = lift or self._idle > self.RELEASE
        last = self.sim.last
        if self.phase == "travel":
            if press:
                self.phase = "descend"
        elif self.phase == "descend":
            if last["pen_down"]:
                self.phase = "stroke"
            elif let_go:
                self.phase = "travel"
        elif self.phase == "stroke":
            if let_go:
                self.phase = "lift"
        elif self.phase == "lift":
            if press:
                self.phase = "stroke"
            elif last["f_n"] < self.F_CLEAR:
                self.k += 1
                self.phase = "travel"
        return f_h, None


def group_of(wr) -> str:
    """The protocol phase the writer is in.  After the last stroke the writer
    passes through "travel" for one step on its way to "done"; that is not an
    approach, and cueing it as one sent a person reaching for "z high"."""
    if wr.phase == "travel" and wr.k >= len(wr.strokes):
        return "after-contact"
    return PHASE_GROUP[wr.phase]


def case_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", text) or "case"


def episode_spec(text: str, case: int, attempt: int, base: int) -> tuple[SM.TaskSpec, T.WriterStyle, int]:
    """Randomization for one attempt: the paper, the size/slant/placement and
    the writer vary; the text does not.  The stiffness schedule in the style is
    unused -- the levels replace it."""
    seed = base + 1000 * case + attempt
    spec = dataclasses.replace(SM.TaskSpec.sample(seed), text=text)
    return spec, T.WriterStyle.sample(seed), seed


def run_episode(sim, spec, style, user, levels: Levels, args, recorder=None,
                viewer=None, cues: bool = True, window=None, panel=None) -> dict | None:
    """One episode with levels from `user`.  The synthetic writer moves the pen
    in `style`; with style=None a person does, through the keys of `window`
    (the viewer's unless given), and ends the episode with R.  Real time when a
    viewer is given, with the reflected force drawn in `panel` (panel.ForcePanel).
    Returns None if aborted (N) or quit (ESC)."""
    sim.reset(spec)
    sess = T.TeleopSession(sim)
    win = window if window is not None else getattr(viewer, "window", None)
    human = style is None
    if human:
        wr = KeyboardWriter(win, sim)
    else:
        wr = T.SyntheticWriter(sim, dataclasses.replace(style, hover_dwell=args.dwell,
                                                        v_desc=args.v_desc, v_travel=args.v_travel))
    user.reset()
    comp = Compliance(args.grace)
    k_cmd = levels.k(user.level)
    dt = sim.dt
    i = 0
    last_group = None
    last, acc, last_print = time.time(), 0.0, -1.0
    while not wr.done and sim.t < spec.time_limit:
        n_steps = 1
        # the viewer's own window too: with a scripted operator `win` is the script
        for w in filter(None, (win, getattr(viewer, "window", None))):
            if w.should_close or w.key_down("esc"):
                raise KeyboardInterrupt
        if win is not None:
            if win.key_press("n"):
                print("\n[aborted] same randomization again")
                return None
            if human and win.key_press("r"):
                wr.done = True
                break
        if viewer is not None:
            now = time.time()
            acc += min(now - last, 0.1)
            last = now
            n_steps = int(acc / dt)
            acc -= n_steps * dt
            n_steps = min(n_steps, 50)
        for _ in range(n_steps):
            group = group_of(wr)
            if user.poll(sim.t, group) and viewer is not None:
                print(f"\n    xy {LEVEL_NAMES[user.level[0]]:<4}  z {LEVEL_NAMES[user.level[1]]:<4}")
            if cues and group != last_group:
                want = EXPECTED[group]
                print(f"\n  [{sim.t:5.1f}s] stroke {min(wr.k + 1, len(wr.strokes))}/{len(wr.strokes)} "
                      f"{group.upper():<13} -> xy {LEVEL_NAMES[want[0]]}, z {LEVEL_NAMES[want[1]]}")
            last_group = group
            f_h, _ = wr.act(sim.t, sess.x_m, sess.v_m, sess.f_fb, sim.last["p"])
            k_cmd = k_cmd + (levels.k(user.level) - k_cmd) * min(1.0, dt / args.ramp)
            comp.step(sim.t, group, user.level)
            rec = sess.step(f_h, k_cmd)
            rec.update(phase=T.PHASES.index(wr.phase), stroke=wr.k,
                       k_level=np.array(user.level))
            if recorder is not None:
                recorder.on_step(i, rec, sess, wr)
            i += 1
            if wr.done:
                break
        if viewer is not None:
            if panel is not None:
                group = group_of(wr)
                want = EXPECTED[group]
                panel.update(sim, sess.f_fb, [
                    "STIFFNESS",
                    f" now   xy {LEVEL_NAMES[user.level[0]]:<4} {k_cmd[0]:5.0f}    z {LEVEL_NAMES[user.level[1]]:<4} {k_cmd[2]:5.0f}  N/m",
                    f" step  {group.upper():<13} -> xy {LEVEL_NAMES[want[0]]}, z {LEVEL_NAMES[want[1]]}"
                    + ("" if tuple(user.level) == want else "   <- change")])
            sim.env.render_human()
            if sim.t - last_print > 0.5:
                last_print = sim.t
                print(f"\r   t {sim.t:5.1f}s  xy {LEVEL_NAMES[user.level[0]]:<4} z {LEVEL_NAMES[user.level[1]]:<4}"
                      f"  paper {sim.last['f_n']:4.1f} N  ink {len(sim.ink_uv):4d}  {group_of(wr):<13}",
                      end="", flush=True)
    res = sim.score()
    res["finished"] = bool(wr.done)
    if not wr.done:
        res["success"] = False
        res["fail_reason"] = ",".join(filter(None, [res["fail_reason"], "timeout"]))
    res["compliance"] = comp.summary()
    return res


# --------------------------------------------------------------------------- #
def save_episode(sim, rec, res, spec, style, levels, args, text: str, c: int, seed: int,
                 source: str, out: pathlib.Path, extra: dict | None = None):
    """Keep or drop one finished episode; returns (kept, path, log row).
    `extra` attributes are stored with it."""
    comp = res["compliance"]
    # --min-compliance 0 asks for no protocol check at all, and an episode too
    # short to be scored has a NaN compliance that would fail any comparison.
    ok_comp = args.min_compliance <= 0 or comp["overall"] >= args.min_compliance
    keep = bool(res["success"] and ok_comp)
    path = None
    if keep or args.keep_failed:
        d = out / case_name(text)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ep_{seed:05d}.h5"
        rec.save(path, dict(
            text=text, case=c, success=bool(res["success"]), compliant=bool(ok_comp), source=source,
            spec=dataclasses.asdict(spec),
            # who moved the pen: the writer's style, or the keyboard hand's constants
            style=dataclasses.asdict(style) if style is not None else None,
            hand=KeyboardWriter.params() if style is None else None,
            protocol=dict(levels=dataclasses.asdict(levels), expected=EXPECTED,
                          phase_group=PHASE_GROUP, dwell=args.dwell, ramp=args.ramp,
                          grace=args.grace, v_desc=args.v_desc, v_travel=args.v_travel,
                          reaction=list(args.reaction), level_names=LEVEL_NAMES,
                          motion="human" if style is None else "writer"),
            compliance=comp, master=dataclasses.asdict(T.MasterParams()),
            criteria=dataclasses.asdict(sim.crit),
            gains={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in dataclasses.asdict(sim.ctl.g).items()},
            metrics={k: v for k, v in res.items() if k not in ("checks", "compliance")},
            phases=T.PHASES, sim_dt=sim.dt, log_hz=100.0, policy_hz=10.0,
            writing_frame=sim.W.tolist(), **(extra or {})))
        if rec.video:
            import imageio.v2 as imageio
            imageio.mimsave(path.with_suffix(".mp4"), rec.video, fps=5, quality=7, macro_block_size=1)
    reason = ", ".join(filter(None, [res["fail_reason"], "" if ok_comp else "protocol"]))
    row = dict(case=c, text=text, seed=seed, kept=keep, saved=str(path) if path else None,
               success=bool(res["success"]), reason=reason, compliance=comp,
               coverage=res["coverage"], precision=res["precision"], in_band=res["in_band"],
               peak_force=res["peak_force"], chamfer_mm=res["chamfer_mm"], t=res["t"])
    return keep, path, row


def verdict_line(row: dict, done: int, per_case: int) -> str:
    comp = row["compliance"]
    v = "KEPT" if row["kept"] else f"not kept: {row['reason']}"
    return (f"[{v}] coverage {row['coverage']:.2f} precision {row['precision']:.2f} "
            f"in-band {row['in_band']:.2f} peak {row['peak_force']:.1f} N | protocol "
            + " ".join(f"{g} {100 * comp[g]:.0f}%" for g in GROUPS if comp[g] == comp[g])
            + f" | {done}/{per_case}")


def scripted_episode(sim, spec, levels: Levels, args, seed: int, **kw):
    """--scripted: one episode with a scripted operator at the keys, moving the
    pen and setting the levels.  Its delays are the preset's (or the two given),
    +-20% per episode.  Returns (spec, result, attributes to save)."""
    import scripted
    name = args.scripted[0] if len(args.scripted) == 1 else "custom"
    preset = scripted.SKILLS[name] if name != "custom" else scripted.Skill(*map(float, args.scripted))
    op = scripted.ScriptedOperator(sim, preset.sample(np.random.default_rng(80_000 + seed)))
    spec = dataclasses.replace(spec, time_limit=scripted.TIME_LIMIT)
    res = run_episode(sim, spec, None, KeyboardLevels(op), levels, args, window=op, **kw)
    return spec, res, dict(operator=dict(skill=name, preset=dataclasses.asdict(preset),
                                         **dataclasses.asdict(op.skill)))


_SIM = None


def _worker_init(image_size: int, cameras: bool, video: bool) -> None:
    global _SIM
    _SIM = SM.WritingSim(cameras=cameras, image_size=image_size,
                         render_mode="rgb_array" if video else None)


def _run_job(job) -> dict:
    """One autonomous episode in a worker process (headless, parallel)."""
    from collect import Recorder
    c, text, attempt, args = job
    sim = _SIM
    levels = Levels(xy=tuple(args.k_xy), z=tuple(args.k_z))
    spec, style, seed = episode_spec(text, c, attempt, args.seed_base)
    rec = Recorder(sim, images=not args.no_cameras,
                   video_every=2 if attempt < args.video else 0)
    source, extra = "protocol-auto", None
    if args.scripted:
        source, style = "protocol-scripted", None
        spec, res, extra = scripted_episode(sim, spec, levels, args, seed, recorder=rec, cues=False)
    else:
        res = run_episode(sim, spec, style, AutoUser(seed, tuple(args.reaction)), levels, args,
                          recorder=rec, cues=False)
    _, _, row = save_episode(sim, rec, res, spec, style, levels, args, text, c, seed,
                             source, pathlib.Path(args.out), extra)
    row["attempt"] = attempt
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", nargs="+", default=["S", "7", "<star>"],
                    help='the cases, e.g. S 7 "<star>" (shapes in <>, quote them)')
    ap.add_argument("--per-case", type=int, default=50, help="successful, compliant demos per case")
    ap.add_argument("--out", default="demos/protocol")
    ap.add_argument("--order", choices=["round-robin", "block"], default="round-robin")
    ap.add_argument("--seed-base", type=int, default=60_000)
    ap.add_argument("--k-xy", type=float, nargs=3, default=[500.0, 1000.0, 3000.0], metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--k-z", type=float, nargs=3, default=[500.0, 1000.0, 3000.0], metavar=("LOW", "MID", "HIGH"))
    ap.add_argument("--dwell", type=float, default=1.0, help="s hovering before each descent")
    ap.add_argument("--v-desc", type=float, default=0.010, help="m/s, the writer's descent before contact")
    ap.add_argument("--v-travel", type=float, default=0.030, help="m/s, the writer's approach speed")
    ap.add_argument("--ramp", type=float, default=0.15, help="s, time constant of a level change")
    ap.add_argument("--grace", type=float, default=0.6, help="s of reaction time not scored")
    ap.add_argument("--min-compliance", type=float, default=0.8)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--auto-next", action="store_true", help="don't wait for G between episodes")
    ap.add_argument("--motion", choices=["human", "writer"], default=None,
                    help="who moves the pen: you, with the keys, or the synthetic writer "
                         "(default: human, or writer with --auto-user)")
    ap.add_argument("--auto-user", action="store_true",
                    help="no person: the levels switch by protocol phase, --reaction late")
    ap.add_argument("--reaction", type=float, nargs=2, default=[0.20, 0.45], metavar=("MIN", "MAX"),
                    help="s, --auto-user's delay after each phase change; 0 0 = exactly per phase")
    ap.add_argument("--scripted", nargs="+", default=None, metavar="SKILL",
                    help="no person: a scripted operator at the keys moves the pen and sets the levels. "
                         "good, normal or bad (scripted.SKILLS), or its two delays in seconds: MOTION STIFFNESS")
    ap.add_argument("--attempts", type=int, default=None,
                    help="stop a case after this many attempts, kept or not")
    ap.add_argument("--headless", action="store_true", help="no window (needs --auto-user or --scripted)")
    ap.add_argument("--workers", type=int, default=1, help="parallel episodes (headless only)")
    ap.add_argument("--video", type=int, default=0, help="mp4 for the first N attempts per case (headless)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--no-panel", action="store_true",
                    help="no floating force-feedback window in the viewer")
    args = ap.parse_args()
    if args.scripted:
        import scripted
        if not (args.scripted[0] in scripted.SKILLS if len(args.scripted) == 1 else len(args.scripted) == 2):
            ap.error(f"--scripted takes one of {', '.join(scripted.SKILLS)}, or two delays in seconds")
        if args.auto_user or args.motion == "writer":
            ap.error("--scripted moves the pen and sets the levels itself: no --auto-user, no --motion writer")
    if args.motion is None:
        args.motion = "writer" if args.auto_user else "human"
    human = args.motion == "human"
    if args.headless and not (args.scripted or (args.auto_user and not human)):
        ap.error("--headless needs --scripted, or --auto-user with --motion writer: "
                 "nobody can press keys without a window")
    if (args.workers > 1 or args.video) and not args.headless:
        ap.error("--workers and --video need --headless")

    levels = Levels(xy=tuple(args.k_xy), z=tuple(args.k_z))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "attempts.jsonl"

    # resume: kept demos per case, and the next attempt number per case.  The
    # kept ones are counted from the log, not from the files: with
    # --keep-failed the folders hold the failed attempts too.
    done = {c: 0 for c in range(len(args.texts))}
    attempt = {c: 0 for c in range(len(args.texts))}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            e = json.loads(line)
            if e["text"] in args.texts:
                c = args.texts.index(e["text"])
                attempt[c] = max(attempt[c], e["attempt"] + 1)
                done[c] += int(e["kept"])
    else:
        done = {c: len(list((out / case_name(t)).glob("ep_*.h5"))) for c, t in enumerate(args.texts)}

    def log(row: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print("  levels  xy " + " / ".join(f"{k:.0f}" for k in levels.xy)
          + "   z " + " / ".join(f"{k:.0f}" for k in levels.z) + " N/m")
    for c, t in enumerate(args.texts):
        print(f"  case {c} {t!r:10} {done[c]}/{args.per_case} collected")

    def left(c: int) -> int:
        """Attempts case c may still start: until per_case are kept, or --attempts are made."""
        n = args.per_case - done[c]
        return n if args.attempts is None else min(n, args.attempts - attempt[c])

    # ---- headless and parallel: rounds until every case has per_case kept ----
    if args.headless and args.workers > 1:
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                          initargs=(args.image_size, not args.no_cameras,
                                                    args.video > 0)) as pool:
            while todo := {c: left(c) for c in done if left(c) > 0}:
                jobs = []
                for c, n in todo.items():
                    for _ in range(n):
                        jobs.append((c, args.texts[c], attempt[c], args))
                        attempt[c] += 1
                for row in pool.imap_unordered(_run_job, jobs):
                    done[row["case"]] += int(row["kept"])
                    log(row)
                    print(f"  case {row['case']} seed {row['seed']}  "
                          + verdict_line(row, done[row["case"]], args.per_case), flush=True)
        print(f"\n  {sum(done.values())} demos in {(time.time() - t0) / 60:.1f} min")
        print("  " + "  ".join(f"{t!r}: {done[c]}/{args.per_case}" for c, t in enumerate(args.texts)))
        return

    # ---- one episode at a time: with a window, or headless on one core ----
    from collect import Recorder
    sim = SM.WritingSim(cameras=not args.no_cameras, image_size=args.image_size,
                        render_mode=None if args.headless else "human")
    viewer = None if args.headless else sim.u.render_human()
    panel = None
    if viewer is not None and not args.no_panel:
        from panel import ForcePanel
        panel = ForcePanel(viewer)

    def next_case():
        todo = [c for c in range(len(args.texts)) if left(c) > 0]
        if not todo:
            return None
        if args.order == "block":
            return todo[0]
        return min(todo, key=lambda c: (done[c], c))

    if viewer is not None:
        print("THE PROTOCOL" + __doc__.split("THE PROTOCOL")[1].split("WHICH STEP")[0])
        if human:
            print("  The step is read off your hand: U held above the paper is PRE-CONTACT, ink is\n"
                  "  CONTACT, letting go of U is AFTER CONTACT, anything else is APPROACH.\n")
            print("  pen     J/L left/right   I/K up/down the letter   U (hold) press   O lift   SHIFT faster")
        print("  keys    xy 1/2/3   z Z/X/C (or 8/9/0)   V (or M) all mid   G go   "
              + ("R finished   " if human else "") + "N abort   ESC quit\n")

    # (who moves the pen, who sets the levels)
    source = {("writer", False): "protocol-keyboard", ("writer", True): "protocol-auto",
              ("human", False): "protocol-human", ("human", True): "protocol-human-auto"}[
                  args.motion, args.auto_user]
    if args.scripted:
        source = "protocol-scripted"
    user = None if args.auto_user or args.scripted else KeyboardLevels(viewer.window)
    try:
        while (c := next_case()) is not None:
            text = args.texts[c]
            spec, style, seed = episode_spec(text, c, attempt[c], args.seed_base)
            if human:
                # no writer, and no time limit: the person says when it is finished
                spec, style = dataclasses.replace(spec, time_limit=1e9), None
            if args.auto_user:
                user = AutoUser(seed, tuple(args.reaction))
            print(f"\n=== case {c} {text!r}  demo {done[c] + 1}/{args.per_case}  (seed {seed})")
            if viewer is not None and not args.auto_next and not args.scripted:
                print("    press G to start" + (", R when you have finished writing" if human else ""))
                sim.reset(spec)
                while not viewer.window.key_press("g"):
                    if viewer.window.should_close or viewer.window.key_down("esc"):
                        raise KeyboardInterrupt
                    if panel is not None:
                        panel.update(sim, None, [f"{text!r}  demo {done[c] + 1}/{args.per_case}:  press G to start"])
                    sim.env.render_human()
            rec = Recorder(sim, images=not args.no_cameras)
            kw = dict(recorder=rec, viewer=viewer, cues=viewer is not None, panel=panel)
            extra = None
            if args.scripted:
                spec, res, extra = scripted_episode(sim, spec, levels, args, seed, **kw)
            else:
                res = run_episode(sim, spec, style, user, levels, args, **kw)
            if res is None:                        # aborted: same randomization again
                continue
            keep, path, row = save_episode(sim, rec, res, spec, style, levels, args, text, c, seed,
                                           source, out, extra)
            row["attempt"] = attempt[c]
            done[c] += int(keep)
            attempt[c] += 1
            log(row)
            print("\n" + verdict_line(row, done[c], args.per_case))
    except KeyboardInterrupt:
        print("\n[quit] episode in progress discarded")
    print("\n" + "  ".join(f"{t!r}: {done[c]}/{args.per_case}" for c, t in enumerate(args.texts)))
    sim.close()


if __name__ == "__main__":
    main()
