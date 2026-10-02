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
    def step(self, t: float, group: str, level) -> None:
        if group != self.group:
            self.group, self.t_change = group, t
        if t - self.t_change >= self.grace:
            self.n[group] += 1
            self.hit[group] += int(tuple(level) == EXPECTED[group])


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
                viewer=None, cues: bool = True) -> dict | None:
    """One autonomous episode with levels from `user`."""
    sim.reset(spec)
    sess = T.TeleopSession(sim)
    wr = WT.SyntheticWiper(sim, dataclasses.replace(style, hover_dwell=args.dwell,
                                                    v_desc=args.v_desc,
                                                    v_travel=args.v_travel))
    user.reset()
    comp = Compliance(args.grace)
    k_cmd = levels.k(user.level)
    kr_cmd = levels.k_r(user.level)
    dt = sim.dt
    i = 0
    last_group = None
    last, acc, last_print = time.time(), 0.0, -1.0
    while not wr.done and sim.t < spec.time_limit:
        n_steps = 1
        if viewer is not None:
            w = viewer.window
            if w.should_close or w.key_down("esc"):
                raise KeyboardInterrupt
            if w.key_press("n"):
                print("\n[aborted] same randomization again")
                return None
            now = time.time()
            acc += min(now - last, 0.1)
            last = now
            n_steps = min(int(acc / dt), 50)
            acc -= n_steps * dt
        for _ in range(n_steps):
            group = WP.group_of(wr)
            if user.poll(sim.t, group) and viewer is not None:
                print(f"\n    xy {LEVEL_NAMES[user.level[0]]:<4} z {LEVEL_NAMES[user.level[1]]:<4}"
                      f" K_R {LEVEL_NAMES[user.level[2]]:<4}")
            if cues and group != last_group:
                want = EXPECTED[group]
                print(f"\n  [{sim.t:5.1f}s] row {min(wr.k + 1, len(wr.strokes))}/{len(wr.strokes)} "
                      f"{group.upper():<13} -> xy {LEVEL_NAMES[want[0]]}, z {LEVEL_NAMES[want[1]]},"
                      f" K_R {LEVEL_NAMES[want[2]]}")
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
                print(f"\r   t {sim.t:5.1f}s  xy {LEVEL_NAMES[user.level[0]]:<4}"
                      f" z {LEVEL_NAMES[user.level[1]]:<4} K_R {LEVEL_NAMES[user.level[2]]:<4}"
                      f"  board {sim.last['f_n']:4.1f} N  left {(~sim.gone).sum():3d}   ",
                      end="", flush=True)
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
    row = dict(case=c, text=text, board=args.board, seed=seed, kept=keep,
               saved=str(path) if path else None, success=bool(res["success"]),
               reason=reason, compliance=comp, erased=res["erased"],
               in_band=res["in_band"], peak_force=res["peak_force"], t=res["t"])
    return keep, path, row


def verdict_line(row: dict, done: int, per_case: int) -> str:
    comp = row["compliance"]
    v = "KEPT" if row["kept"] else f"not kept: {row['reason']}"
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

    if viewer is not None:
        print(__doc__.split("Usage:")[0].split("THE PROTOCOL")[1].split("A level change")[0])
        print("  keys    xy 1/2/3   z 8/9/0   K_R 4/5/6   M all mid   G go   N abort   ESC quit\n")

    user = None if args.auto_user else KeyboardLevels(viewer.window)
    try:
        while (c := next_case()) is not None:
            text = args.texts[c]
            spec, style, seed = episode_spec(text, c, attempt[c], args.seed_base)
            operator = None
            if args.auto_user:
                style, user, operator = tiered(sim, style, seed, levels, args)
            print(f"\n=== case {c} {text!r}  demo {done[c] + 1}/{args.per_case}  (seed {seed})")
            if viewer is not None and not args.auto_next:
                print("    press G to start")
                sim.reset(spec)
                while not viewer.window.key_press("g"):
                    if viewer.window.should_close or viewer.window.key_down("esc"):
                        raise KeyboardInterrupt
                    sim.env.render_human()
            rec = WipeRecorder(sim, images=not args.no_cameras)
            res = run_episode(sim, spec, style, user, levels, args, recorder=rec,
                              viewer=viewer, cues=viewer is not None)
            if res is None:
                continue
            keep, path, row = save_episode(
                sim, rec, res, spec, style, levels, args, text, c, seed,
                "protocol-scripted" if operator else
                ("protocol-auto" if args.auto_user else "protocol-keyboard"), out,
                operator=operator)
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
