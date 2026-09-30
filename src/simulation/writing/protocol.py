"""Stiffness-protocol collection: the pen writes by itself, you set the stiffness.

The synthetic writer (teleop.SyntheticWriter) does everything a hand does --
travels, feels for the paper, presses, writes, lifts.  The one thing it does
not do here is choose the Case 1 stiffness.  That comes from the keyboard, in
three levels per axis group, translation only:

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

The writer hovers `--dwell` seconds over each stroke's start before it descends,
so there is time for step 2.  A level change is not a step in K: it ramps with
a `--ramp` time constant, like a muscle, and the tank pays for it.

Every episode is checked against the protocol: the fraction of time the levels
matched the table above, ignoring `--grace` seconds after each phase change for
reaction time.  Only episodes that SUCCEED and COMPLY are kept.

Keys (SAPIEN's viewer owns WASD/QE for its camera):
    1 / 2 / 3     xy  low / mid / high
    8 / 9 / 0     z   low / mid / high
    M             everything to mid
    G             start the next episode
    N             abort this episode, retry the same randomization
    ESC           quit (the episode in progress is discarded)

Usage:
    python3 protocol.py --texts S 7 "<star>" --per-case 50 --out demos/protocol
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
    KEYS = {"1": (0, LOW), "2": (0, MID), "3": (0, HIGH),
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
        if self.win.key_press("m"):
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
                viewer=None, cues: bool = True) -> dict | None:
    """One autonomous episode with levels from `user`.  Real time when a viewer
    is given.  Returns None if aborted (N) or quit (ESC)."""
    sim.reset(spec)
    sess = T.TeleopSession(sim)
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
            sim.env.render_human()
            if sim.t - last_print > 0.5:
                last_print = sim.t
                print(f"\r   t {sim.t:5.1f}s  xy {LEVEL_NAMES[user.level[0]]:<4} z {LEVEL_NAMES[user.level[1]]:<4}"
                      f"  paper {sim.last['f_n']:4.1f} N  ink {len(sim.ink_uv):4d}   ", end="", flush=True)
    res = sim.score()
    res["finished"] = bool(wr.done)
    if not wr.done:
        res["success"] = False
        res["fail_reason"] = ",".join(filter(None, [res["fail_reason"], "timeout"]))
    res["compliance"] = comp.summary()
    return res


# --------------------------------------------------------------------------- #
def save_episode(sim, rec, res, spec, style, levels, args, text: str, c: int, seed: int,
                 source: str, out: pathlib.Path):
    """Keep or drop one finished episode; returns (kept, path, log row)."""
    comp = res["compliance"]
    ok_comp = comp["overall"] >= args.min_compliance
    keep = bool(res["success"] and ok_comp)
    path = None
    if keep or args.keep_failed:
        d = out / case_name(text)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"ep_{seed:05d}.h5"
        rec.save(path, dict(
            text=text, case=c, success=bool(res["success"]), compliant=bool(ok_comp), source=source,
            spec=dataclasses.asdict(spec), style=dataclasses.asdict(style),
            protocol=dict(levels=dataclasses.asdict(levels), expected=EXPECTED,
                          phase_group=PHASE_GROUP, dwell=args.dwell, ramp=args.ramp,
                          grace=args.grace, v_desc=args.v_desc, v_travel=args.v_travel,
                          reaction=list(args.reaction), level_names=LEVEL_NAMES),
            compliance=comp, master=dataclasses.asdict(T.MasterParams()),
            criteria=dataclasses.asdict(sim.crit),
            gains={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in dataclasses.asdict(sim.ctl.g).items()},
            metrics={k: v for k, v in res.items() if k not in ("checks", "compliance")},
            phases=T.PHASES, sim_dt=sim.dt, log_hz=100.0, policy_hz=10.0,
            writing_frame=sim.W.tolist()))
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
    res = run_episode(sim, spec, style, AutoUser(seed, tuple(args.reaction)), levels, args,
                      recorder=rec, cues=False)
    _, _, row = save_episode(sim, rec, res, spec, style, levels, args, text, c, seed,
                             "protocol-auto", pathlib.Path(args.out))
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
    ap.add_argument("--auto-user", action="store_true",
                    help="no person: the levels switch by protocol phase, --reaction late")
    ap.add_argument("--reaction", type=float, nargs=2, default=[0.20, 0.45], metavar=("MIN", "MAX"),
                    help="s, --auto-user's delay after each phase change; 0 0 = exactly per phase")
    ap.add_argument("--headless", action="store_true", help="no window (needs --auto-user)")
    ap.add_argument("--workers", type=int, default=1, help="parallel episodes (headless only)")
    ap.add_argument("--video", type=int, default=0, help="mp4 for the first N attempts per case (headless)")
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()
    if args.headless and not args.auto_user:
        ap.error("--headless needs --auto-user: nobody can press keys without a window")
    if (args.workers > 1 or args.video) and not args.headless:
        ap.error("--workers and --video need --headless")

    levels = Levels(xy=tuple(args.k_xy), z=tuple(args.k_z))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "attempts.jsonl"

    # resume: kept demos per case, and the next attempt number per case
    done = {c: len(list((out / case_name(t)).glob("ep_*.h5"))) if (out / case_name(t)).exists() else 0
            for c, t in enumerate(args.texts)}
    attempt = {c: 0 for c in range(len(args.texts))}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            e = json.loads(line)
            if e["text"] in args.texts:
                c = args.texts.index(e["text"])
                attempt[c] = max(attempt[c], e["attempt"] + 1)

    def log(row: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print("  levels  xy " + " / ".join(f"{k:.0f}" for k in levels.xy)
          + "   z " + " / ".join(f"{k:.0f}" for k in levels.z) + " N/m")
    for c, t in enumerate(args.texts):
        print(f"  case {c} {t!r:10} {done[c]}/{args.per_case} collected")

    # ---- headless and parallel: rounds until every case has per_case kept ----
    if args.headless and args.workers > 1:
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                          initargs=(args.image_size, not args.no_cameras,
                                                    args.video > 0)) as pool:
            while any(done[c] < args.per_case for c in done):
                jobs = []
                for c, text in enumerate(args.texts):
                    for _ in range(args.per_case - done[c]):
                        jobs.append((c, text, attempt[c], args))
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

    def next_case():
        todo = [c for c in range(len(args.texts)) if done[c] < args.per_case]
        if not todo:
            return None
        if args.order == "block":
            return todo[0]
        return min(todo, key=lambda c: (done[c], c))

    if viewer is not None:
        print(__doc__.split("Usage:")[0].split("THE PROTOCOL")[1].split("Every episode")[0])
        print("  keys    xy 1/2/3   z 8/9/0   M all mid   G go   N abort   ESC quit\n")

    user = None if args.auto_user else KeyboardLevels(viewer.window)
    try:
        while (c := next_case()) is not None:
            text = args.texts[c]
            spec, style, seed = episode_spec(text, c, attempt[c], args.seed_base)
            if args.auto_user:
                user = AutoUser(seed, tuple(args.reaction))
            print(f"\n=== case {c} {text!r}  demo {done[c] + 1}/{args.per_case}  (seed {seed})")
            if viewer is not None and not args.auto_next:
                print("    press G to start")
                sim.reset(spec)
                while not viewer.window.key_press("g"):
                    if viewer.window.should_close or viewer.window.key_down("esc"):
                        raise KeyboardInterrupt
                    sim.env.render_human()
            rec = Recorder(sim, images=not args.no_cameras)
            res = run_episode(sim, spec, style, user, levels, args, recorder=rec, viewer=viewer,
                              cues=viewer is not None)
            if res is None:                        # aborted: same randomization again
                continue
            keep, path, row = save_episode(
                sim, rec, res, spec, style, levels, args, text, c, seed,
                "protocol-auto" if args.auto_user else "protocol-keyboard", out)
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
