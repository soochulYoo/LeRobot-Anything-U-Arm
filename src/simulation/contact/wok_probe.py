"""A WOK, tossed by the Panda under the Case 1 controller: can it be done at all, and
does the stiffness decide it?  A feasibility PROBE, before any task is built on it.

    python wok_probe.py --variant lift slide --levels low mid high --seeds 4 \
           --out ../../../../../runs/wok-probe --video

A VR TRIAL, NOT A TASK YET.  The lift wok can be driven in VR through the contact
collector (`protocol.py --task wok`, which imports this file) and the console's
`sim-wok` station, with the person setting the levels.  There is no schedule, no
task_fit entry and no scripted hand for the collector: those come after the operator has
tried tossing and chosen a variant.

THE WOK IS PART OF THE ARM.  It is a link of the arm's own URDF, fixed to the hand, so
ManiSkill's disabled link gravity makes it a COMPENSATED payload -- as a real arm
compensates a known tool -- while its inertia stays in the controller's model.  The
marbles are separate bodies and keep their weight.  (Uncompensated, LOW would fail by
the wok drooping, which is a gravity-compensation problem and not a stiffness one.)

THE BOWL IS FLAT PANELS on a spherical cap, not a mesh: a moving body cannot have a
concave collision shape, and panels need no convex decomposition.

TWO VARIANTS, scripted, three cycles each:
    lift    the wok rests on the burner ring, HOPS straight up and stops dead -- a
            marble leaves when the wok decelerates faster than g -- then comes back
            down onto the ring at the height the hand believes it is at.
    slide   the wok stays on the ring, leaned on: push forward, then a sharp pull back
            with a flick of the far edge up.  THE FLICK IS SET ON THE CONTROLLER'S
            ORIENTATION REFERENCE DIRECTLY: Case1Proposal has no angular velocity, so
            the flick is not metered by the energy tank.  Acceptable in a probe only.

THE STIFFNESS IS HELD for the episode, isotropic, at one of the contact tasks' ALONG
levels (300 / 1000 / 3000 N/m), with the wrist at K_R 30 (HIGH) so the wok stays level:
the probe asks about translation only.

WHAT COUNTS, fixed before the first run:
    tossed    a marble out of contact with the wok for >= 0.15 s that rises >= 3 cm
              above where it left it
    spilled   a marble more than 2 cm below the wok's inner bottom, or more than 15 cm
              from its axis
    grate     the wok-on-stove force, low-passed at 5 Hz as the contact tasks'
              sustained force is: <= 30 N
    success   every cycle tosses >= 2 marbles, none spills, the grate force stays under
              its limit
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import multiprocessing
import pathlib
import re
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contact_tasks as CT  # noqa: E402  sets the path to writing/ and registers its envs
import contact_scene as SC  # noqa: E402
import controller as C  # noqa: E402
import sapien  # noqa: E402
import torch  # noqa: E402
from mani_skill import PACKAGE_ASSET_DIR  # noqa: E402
from mani_skill.agents.registration import register_agent  # noqa: E402
from mani_skill.agents.robots.panda.panda_stick import PandaStick  # noqa: E402
from mani_skill.utils.registration import register_env  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

Array = np.ndarray

# ---- the wok, in its own frame: z up, origin at the inner bottom (the TCP) ----------
WOK_R = 0.13            # m: rim radius
WOK_DEPTH = 0.07        # m
WOK_RC = (WOK_R ** 2 + WOK_DEPTH ** 2) / (2 * WOK_DEPTH)    # the cap's sphere, 0.156
WOK_T = 0.005           # m: panel thickness
WOK_MASS = 0.8          # kg
BANDS = np.radians([12.0, 24.0, 36.0, 46.0, np.degrees(np.arcsin(WOK_R / WOK_RC))])
SEGS = 16
# where the wok's inner bottom is on the hand, in the hand's frame (z down the hand's
# axis): 20 cm in front of it and 13 cm below, so the hand is above and behind the rim
HAND_TO_WOK = np.array([0.20, 0.0, 0.13])
HANDLE_FROM = np.array([-0.17, 0.0, 0.11])   # wok frame: by the hand ...
HANDLE_TO = np.array([-WOK_R, 0.0, WOK_DEPTH])  # ... to the near rim

LEVELS = {"low": 300.0, "mid": 1000.0, "high": 3000.0}


def _wok_shapes() -> list[tuple]:
    """(kind, xyz, rpy, size) of every piece of the wok, in the wok's frame."""
    out = [("cylinder", (0.0, 0.0, -WOK_T / 2), (0.0, 0.0, 0.0),
            (WOK_RC * np.sin(BANDS[0]) + 0.002, WOK_T))]
    centre = np.array([0.0, 0.0, WOK_RC])
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = 0.5 * (lo + hi)
        for k in range(SEGS):
            th = 2 * np.pi * k / SEGS
            n_out = np.array([np.sin(m) * np.cos(th), np.sin(m) * np.sin(th), -np.cos(m)])
            x = np.array([np.cos(m) * np.cos(th), np.cos(m) * np.sin(th), np.sin(m)])
            y = np.array([-np.sin(th), np.cos(th), 0.0])
            M = np.column_stack([x, y, -n_out])       # local z points into the bowl
            # a little longer and wider than the facet, so neighbours overlap and a
            # marble finds no seam to fall through
            half = (WOK_RC * np.tan(0.5 * (hi - lo)) + 0.002,
                    WOK_RC * np.sin(hi) * np.tan(np.pi / SEGS) + 0.002, WOK_T / 2)
            out.append(("box", tuple(centre + (WOK_RC + WOK_T / 2) * n_out),
                        tuple(Rotation.from_matrix(M).as_euler("xyz")),
                        tuple(2 * np.array(half))))
    d = HANDLE_TO - HANDLE_FROM
    out.append(("box", tuple(0.5 * (HANDLE_FROM + HANDLE_TO)),
                (0.0, float(np.arctan2(-d[2], d[0])), 0.0),
                (float(np.linalg.norm(d)) + 0.01, 0.02, 0.012)))
    return out


def _wok_urdf() -> str:
    """panda_stick.urdf with its flat stick taken off, the TCP moved to the wok's inner
    bottom, and the wok fixed to the hand -- built as contact_scene builds the stick."""
    src_dir = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda"
    txt = (src_dir / "panda_stick.urdf").read_text()
    txt = txt.replace('filename="franka_description/', f'filename="{src_dir}/franka_description/')
    txt = re.sub(r'\s*<visual>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</visual>', "",
                 txt, count=1, flags=re.S)
    txt = re.sub(r'\s*<collision>\s*<origin xyz="0 0 0.1" rpy="0 0 0"/>.*?</collision>', "",
                 txt, count=1, flags=re.S)
    old = ('<origin rpy="0 0 0" xyz="0 0 0.15"/>\n    <parent link="panda_hand"/>\n'
           '    <child link="panda_hand_tcp"/>')
    assert old in txt, "the TCP joint was not where it was expected"
    x, y, z = HAND_TO_WOK
    txt = txt.replace(old, old.replace('xyz="0 0 0.15"', f'xyz="{x} {y} {z}"'))

    def geom(kind, size):
        if kind == "cylinder":
            return f'<cylinder radius="{size[0]:.5f}" length="{size[1]:.5f}"/>'
        return '<box size="{:.5f} {:.5f} {:.5f}"/>'.format(*size)

    vis, col = [], []
    for kind, xyz, rpy, size in _wok_shapes():
        o = '<origin xyz="{:.5f} {:.5f} {:.5f}" rpy="{:.5f} {:.5f} {:.5f}"/>'.format(*xyz, *rpy)
        vis.append(f'<visual>{o}<geometry>{geom(kind, size)}</geometry>'
                   f'<material name="steel"><color rgba="0.22 0.22 0.24 1"/></material></visual>')
        col.append(f'<collision>{o}<geometry>{geom(kind, size)}</geometry></collision>')
    wok = f"""
  <link name="wok">
    {chr(10).join(vis)}
    {chr(10).join(col)}
    <inertial>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <mass value="{WOK_MASS}"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.007"/>
    </inertial>
  </link>
  <joint name="wok_joint" type="fixed">
    <origin rpy="{np.pi:.6f} 0 0" xyz="{x} {y} {z}"/>
    <parent link="panda_hand"/>
    <child link="wok"/>
  </joint>
"""
    return txt.replace("</robot>", wok + "</robot>")


def _ensure_assets() -> str:
    out = HERE / "assets"
    out.mkdir(exist_ok=True)
    urdf = out / "panda_wok.urdf"
    txt = _wok_urdf()
    if not urdf.exists() or urdf.read_text() != txt:
        SC._write_atomic(urdf, txt)
    src = pathlib.Path(PACKAGE_ASSET_DIR) / "robots" / "panda" / "panda_stick.srdf"
    s = src.read_text().replace(
        "</robot>", '    <disable_collisions link1="panda_hand" link2="wok" reason="Adjacent"/>\n'
                    '    <disable_collisions link1="panda_link7" link2="wok" reason="Adjacent"/>\n</robot>')
    srdf = out / "panda_wok.srdf"
    if not srdf.exists() or srdf.read_text() != s:
        SC._write_atomic(srdf, s)
    return str(urdf)


@register_agent()
class PandaWok(PandaStick):
    uid = "panda_wok"
    urdf_path = _ensure_assets()


@register_env("TeleopWok-v1", max_episode_steps=1_000_000)
class WokEnv(SC.ContactEnv):
    """robot --- wok --- stove.  A burner ring on the desk, the wok held over it."""
    SUPPORTED_ROBOTS = ["panda_wok"]
    ROBOT = "panda_wok"
    LOOK_AT = np.array([0.55, 0.0, 0.27])
    RENDER_EYE = np.array([-0.10, -0.62, 0.42])       # from the side and above: into the wok
    STOVE_AT = np.array([0.55, 0.0])
    RING_R = 0.055           # m: where the wok's bottom sits on the ring
    RING_TOP = SC.TABLE_TOP + 0.05
    RING_BAR = 0.005         # m: the ring's bars are 10 mm round
    N_MARBLES = 4
    MARBLE_R = 0.011
    MARBLE_MASS = 0.015      # kg
    COLOURS = ([0.90, 0.25, 0.15, 1], [0.95, 0.75, 0.10, 1],
               [0.20, 0.65, 0.25, 1], [0.95, 0.95, 0.90, 1])

    def _load_task(self) -> None:
        iron = sapien.physx.PhysxMaterial(0.3, 0.3, 0.0)
        dark = sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.13, 1])
        b = self.scene.create_actor_builder()
        cx, cy = self.STOVE_AT
        top = self.RING_TOP - 0.03                     # the burner body, under the ring
        half = [0.09, 0.09, (top - SC.TABLE_TOP) / 2]
        pose = sapien.Pose([cx, cy, SC.TABLE_TOP + half[2]])
        b.add_box_visual(pose=pose, half_size=half, material=dark)
        b.add_box_collision(pose=pose, half_size=half, material=iron)
        n = 12
        for k in range(n):
            a = 2 * np.pi * k / n
            q = Rotation.from_euler("z", a + np.pi / 2).as_quat()     # x, y, z, w
            pose = sapien.Pose([cx + self.RING_R * np.cos(a), cy + self.RING_R * np.sin(a),
                                self.RING_TOP - self.RING_BAR], [q[3], q[0], q[1], q[2]])
            hl = self.RING_R * np.tan(np.pi / n)
            b.add_capsule_visual(pose=pose, radius=self.RING_BAR, half_length=hl, material=dark)
            b.add_capsule_collision(pose=pose, radius=self.RING_BAR, half_length=hl,
                                    material=iron)
        self.stove = b.build_static(name="stove")

        r = self.MARBLE_R
        food = sapien.physx.PhysxMaterial(0.4, 0.4, 0.0)    # no bounce: food does not
        self.marbles = []
        for i in range(self.N_MARBLES):
            m = self.scene.create_actor_builder()
            m.add_sphere_collision(radius=r, material=food,
                                   density=self.MARBLE_MASS / (4 / 3 * np.pi * r ** 3))
            m.add_sphere_visual(radius=r, material=sapien.render.RenderMaterial(
                base_color=self.COLOURS[i % len(self.COLOURS)]))
            m.initial_pose = sapien.Pose([0.40 + 0.04 * i, 0.30, SC.TABLE_TOP + r])
            self.marbles.append(m.build(name=f"marble{i}"))


@dataclasses.dataclass
class WokSpec:
    seed: int = 0
    variant: str = "lift"
    belief_err: float = 0.0       # m, up: where the hand believes the ring is, minus true
    time_limit: float = 30.0      # s, for the collector; the probe's hand sets its own

    @staticmethod
    def sample(seed: int, variant: str = "lift") -> "WokSpec":
        rng = np.random.default_rng(seed)
        return WokSpec(seed=seed, variant=variant,
                       belief_err=float(rng.uniform(-0.008, 0.008)))


class TossHand:
    """The scripted cook: the reference's position and orientation at time t.

    It knows where it BELIEVES the ring is (`told`), never where it is, and it does not
    react to anything it feels -- every cycle is the same motion.
    """
    SETTLE = 0.6          # s: come down onto the ring and let the marbles settle
    PERIOD = 1.5          # s: one toss
    CYCLES = 3
    TAIL = 0.8            # s after the last toss
    LEAN = 0.005          # m below where it believes the ring is: leaning on it
    # lift.  1.4 m/s reached at a constant 9.3 m/s^2 -- under g, so the marbles stay
    # down while it speeds up -- then stop dead.  The first cut, 0.9 m/s in 0.12 s,
    # tossed nothing at any stiffness: the wok peaked at 0.35 / 0.48 / 0.60 m/s at
    # LOW / MID / HIGH and the marbles rose 0.2 / 0.6 / 2.1 cm off it.
    V_HOP, T_ACC = 1.4, 0.15
    T_HOLD, V_DOWN = 0.30, 0.20
    # slide
    PUSH, PULL, FLICK = 0.04, 0.03, np.radians(15.0)

    def __init__(self, variant: str, start: Array, told: Array, R0: Array):
        self.variant, self.start, self.R0 = variant, start, R0
        self.base = told - np.array([0.0, 0.0, self.LEAN])
        self.duration = self.SETTLE + self.CYCLES * self.PERIOD + self.TAIL

    def cycle_of(self, t: float) -> int | None:
        k = int(np.floor((t - self.SETTLE) / self.PERIOD))
        return k if 0 <= k < self.CYCLES else None

    @staticmethod
    def _ease(s: float) -> float:
        s = min(max(s, 0.0), 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * s)

    def target(self, t: float) -> tuple[Array, Array]:
        if t < self.SETTLE:
            return self.start + self._ease(t / 0.3) * (self.base - self.start), self.R0
        k = self.cycle_of(t)
        if k is None:
            return self.base.copy(), self.R0
        tau = t - self.SETTLE - k * self.PERIOD
        p, pitch = self.base.copy(), 0.0
        if self.variant == "lift":
            rise = 0.5 * self.V_HOP * self.T_ACC
            if tau < self.T_ACC:
                p[2] += 0.5 * (self.V_HOP / self.T_ACC) * tau ** 2
            elif tau < self.T_ACC + self.T_HOLD:
                p[2] += rise
            else:
                p[2] += max(0.0, rise - self.V_DOWN * (tau - self.T_ACC - self.T_HOLD))
        else:
            if tau < 0.30:
                p[0] += self.PUSH * self._ease(tau / 0.30)
            elif tau < 0.42:
                s = self._ease((tau - 0.30) / 0.12)
                p[0] += self.PUSH - (self.PUSH + self.PULL) * s
                pitch = self.FLICK * s
            elif tau < 0.62:
                p[0] -= self.PULL
                pitch = self.FLICK * (1.0 - self._ease((tau - 0.42) / 0.20))
            else:
                p[0] -= self.PULL * (1.0 - self._ease((tau - 0.62) / 0.30))
        # the far edge (+x) UP: a turn about world y by -pitch carries +x towards +z
        return p, Rotation.from_euler("y", -pitch).as_matrix() @ self.R0


class WokSim(CT.ContactSim):
    ENV_ID = "TeleopWok-v1"
    TASK = "wok"
    TOOL_LINK = "wok"
    LANDS = False
    # HELD FROM THE FIRST FRAME, as the drawer's handle is: the collector opens in
    # `work`, and the VR trigger pushes nothing (protocol.run_episode)
    GRASPED = True
    START_KR = 30.0
    TOSS_AIR, TOSS_RISE = 0.15, 0.03
    SPILL_DROP, SPILL_R = 0.02, 0.15
    GRATE_MAX = 30.0
    MIN_TOSSED = 2
    # WITHOUT THE PROBE'S CYCLES -- a person tossing in VR -- a TOSS is MIN_TOSSED or
    # more marbles leaving within EVENT_GAP of each other, and the episode wants TOSSES
    EVENT_GAP, TOSSES = 0.3, 3
    # THE VR HAND, for this task only (protocol.run_episode applies it).  The default
    # hand pulls the handle with at most 12 N through 30 Ns/m of damping, which caps
    # the handle at 12 / 30 = 0.4 m/s -- the right hand for wiping a board, and one
    # that cannot toss: a toss needs over 1 m/s, and the operator also carries the
    # arm's reflected inertia (5.4 kg up and down at the wok, measured).  The reflected
    # spring puts the hand IN SERIES with the arm's, so the hand has to be stiff for a
    # HIGH arm to stop hard.  Measured with a scripted controller doing the probe's hop
    # through this whole chain (VRHand -> handle -> arm), tosses at LOW / MID / HIGH:
    #     default hand (300 N/m, 12 N)        0 / 0 / 0   the wok peaks at 0.25 m/s
    #     3000 N/m, 75 Ns/m, 150 N            0 / 0 / 0   0.69 / 0.89 / 1.00 m/s
    #     10000 N/m, 140 Ns/m, 300 N          0 / 0 / 3   0.79 / 1.05 / 1.22 m/s
    # (and the last with the reflection switched off, 0 / 0 / 3: so it is kept on).
    # A hand this stiff passes the controller's jitter through; if the wok buzzes in a
    # person's hands, this is the knob.
    VR_MAP = dict(k_hand=10000.0, b_hand=140.0, max_force=300.0, press_force=0.0)
    # THE ENERGY TANK, FOR THE VR TRIAL ONLY.  The controller's 20 J was sized for
    # writing (1-3 J an episode) and is never refilled; three scripted tosses take 16 /
    # 35 / 70 J at LOW / MID / HIGH, and through the VR chain it emptied at MID and HIGH
    # and froze the wok mid-toss.  1000 J is the tank switched off in all but name.
    # Whether a dynamic task should have a bigger tank, or one refilled by what the
    # damping dissipates, is undecided; the probe's own runs pass their gains explicitly.
    TRIAL_E0 = 1000.0

    def __init__(self, *args, gains: C.Case1Gains | None = None, **kwargs):
        super().__init__(*args, gains=gains or C.Case1Gains(E0=self.TRIAL_E0), **kwargs)
        # oiled steel, not the stick's rubber ball that ContactSim gives every tool
        steel = sapien.physx.PhysxMaterial(0.3, 0.3, 0.0)
        for s in self.tool._objs[0].get_collision_shapes():
            s.set_physical_material(steel)

    def _work(self):
        return self.u.stove

    def _force(self, rec) -> Array:
        return self.u.scene.get_pairwise_contact_forces(self.tool, self.u.stove)[0].cpu().numpy()

    def reset(self, spec: WokSpec, k: float | None = None) -> dict:
        """`k`: the probe holds one stiffness; the collector leaves it to the levels."""
        self.spec = spec
        if k is not None:
            self.K0 = np.full(3, float(k))
        self.env.reset(seed=spec.seed)
        u = self.u
        z_rest = u.RING_TOP + np.sqrt((WOK_RC + WOK_T) ** 2 - u.RING_R ** 2) - WOK_RC
        self.home = np.array([*u.STOVE_AT, z_rest])         # the wok's bottom, resting
        self.told = self.home + np.array([0.0, 0.0, spec.belief_err])
        self._begin(self.home + np.array([0.0, 0.0, 0.003]), CT.FWD)
        p = self.ctl.tip_state()[1]
        rng = np.random.default_rng(spec.seed + 7)
        for i, m in enumerate(u.marbles):
            a = np.pi / 4 + i * 2 * np.pi / len(u.marbles) + rng.uniform(-0.3, 0.3)
            r = 0.035
            z = WOK_RC - np.sqrt(WOK_RC ** 2 - r ** 2) + u.MARBLE_R + 0.002
            m.set_pose(sapien.Pose(p + np.array([r * np.cos(a), r * np.sin(a), z])))
            m.set_linear_velocity(torch.zeros(1, 3))
            m.set_angular_velocity(torch.zeros(1, 3))
        n = len(u.marbles)
        self.flight = [None] * n          # (t left, z left, highest z) while in the air
        self.tosses: list[tuple[float, int, float, float]] = []   # (t, marble, air, rise)
        self.spilled = [False] * n
        return self.observe(images=False)

    def _task_step(self, rec, touching: bool) -> None:
        u, p = self.u, rec["p"]
        for i, m in enumerate(u.marbles):
            pos = m.pose.p[0].cpu().numpy().astype(float)     # float32 will not go in a log
            on = float(np.linalg.norm(u.scene.get_pairwise_contact_forces(
                m, self.tool)[0].cpu().numpy())) > 0.0
            fl = self.flight[i]
            if on:
                if fl is not None:
                    air, rise = self.t - fl[0], fl[2] - fl[1]
                    if air >= self.TOSS_AIR and rise >= self.TOSS_RISE:
                        self.tosses.append((fl[0], i, air, rise))
                self.flight[i] = None
            elif fl is None:
                self.flight[i] = [self.t, pos[2], pos[2]]
            else:
                fl[2] = max(fl[2], pos[2])
            rel = pos - p
            if rel[2] < -self.SPILL_DROP or np.hypot(rel[0], rel[1]) > self.SPILL_R:
                self.spilled[i] = True

    def toss_events(self) -> int:
        """Tosses of MIN_TOSSED or more marbles together -- the count without cycles."""
        n, group, t_last = 0, 0, None
        for t0, _, _, _ in sorted(self.tosses):
            if t_last is not None and t0 - t_last > self.EVENT_GAP:
                n, group = n + int(group >= self.MIN_TOSSED), 0
            group, t_last = group + 1, t0
        return n + int(group >= self.MIN_TOSSED)

    def seated(self) -> bool:
        return False          # nothing to come to rest against: the collector stays in work

    def _task_fields(self) -> dict:
        n = self.toss_events() if hasattr(self, "tosses") else 0
        return dict(progress=float(min(1.0, n / self.TOSSES)), toss_events=n)

    def privileged(self) -> dict:
        return {"belief_err": self.spec.belief_err, "home": self.home}

    def score(self, hand: TossHand | None = None) -> dict:
        """With the probe's hand: every one of its cycles tosses.  Without one (a person
        in VR): TOSSES tosses in the episode."""
        res, common = self._common_score()
        n_ev = self.toss_events()
        per = None
        if hand is not None:
            per = [0] * hand.CYCLES
            for t0, _, _, _ in self.tosses:
                k = hand.cycle_of(t0)
                if k is not None:
                    per[k] += 1
        checks = {"tossed": (all(c >= self.MIN_TOSSED for c in per) if per is not None
                             else n_ev >= self.TOSSES),
                  "no_spill": not any(self.spilled),
                  "grate": self.peak <= self.GRATE_MAX}
        res.update(progress=float(min(1.0, n_ev / self.TOSSES)), toss_events=n_ev,
                   tossed_per_cycle=per, spilled=int(sum(self.spilled)),
                   tosses=[dict(t=round(a, 3), marble=b, air=round(c, 3), rise=round(d, 4))
                           for a, b, c, d in self.tosses],
                   common_checks=common)
        return self._finish(res, checks)


# the contact collector (protocol.py) finds its tasks here: `--task wok`
CT.SIMS["wok"] = (WokSim, WokSpec)


def run(sim: WokSim, spec: WokSpec, level: str, video: pathlib.Path | None) -> dict:
    sim.reset(spec, LEVELS[level])
    hand = TossHand(spec.variant, sim.ctl.x_d.copy(), sim.told, sim.ctl.R_d.copy())
    every = max(1, int(round(1.0 / (60.0 * sim.dt))))   # 60 frames per simulated second
    frames, alpha, E, sat, tilt, err = [], 1.0, np.inf, 0, 0.0, 0.0
    while sim.t < hand.duration:
        x_next, R_next = hand.target(sim.t + sim.dt)
        sim.ctl.R_d = R_next
        rec = sim.step(C.Case1Proposal((x_next - sim.ctl.x_d) / sim.dt))
        alpha, E = min(alpha, rec["alpha"]), min(E, rec["E"])
        sat += int(np.any(np.abs(rec["d_tau"]) > 1e-9))
        c = (np.trace(sim.ctl.R_d.T @ rec["R"]) - 1.0) / 2.0
        tilt = max(tilt, float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        err = max(err, float(np.linalg.norm(rec["p"] - rec["x_d"])))
        if video is not None and sim.step_i % every == 0:
            img = sim.env.render()
            img = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
            frames.append(img[0] if img.ndim == 4 else img)
    res = sim.score(hand)
    res.update(variant=spec.variant, level=level, k=LEVELS[level], seed=spec.seed,
               belief_err=spec.belief_err, alpha_min=alpha, tank_min=E,
               saturated_frac=sat / max(1, sim.step_i), tilt_max_deg=tilt,
               track_err_max=err)
    if frames:
        import imageio.v2 as imageio
        # written at 30 fps from 60 a simulated second: HALF SPEED, so a toss can be seen
        imageio.mimsave(str(video), frames, fps=30, macro_block_size=8)
        res["video"] = str(video)
    return res


_SIM = None


def _job(args: tuple) -> dict:
    global _SIM
    spec, level, video, e0 = args
    if _SIM is None:
        gains = C.Case1Gains() if e0 is None else C.Case1Gains(E0=e0)
        _SIM = WokSim(cameras=False, gains=gains, render_mode="rgb_array")
    t0 = time.time()
    res = run(_SIM, spec, level, None if video is None else pathlib.Path(video))
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", nargs="+", default=["lift", "slide"], choices=("lift", "slide"))
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=950_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", action="store_true", help="an mp4 of each condition's first seed")
    ap.add_argument("--e0", type=float, default=None,
                    help="the energy tank's start, J (default: the controller's own)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for v in args.variant:
        for lv in args.levels:
            for i in range(args.seeds):
                spec = WokSpec.sample(args.seed0 + i, v)
                video = out / f"{v}_{lv}_seed{spec.seed}.mp4" if args.video and i == 0 else None
                jobs.append((spec, lv, None if video is None else str(video), args.e0))
    with concurrent.futures.ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        results = list(ex.map(_job, jobs))
    with open(out / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=float) + "\n")
    print(f"{'variant':7} {'level':5} {'seed':>6} {'belief':>7} {'tossed/cycle':>12} "
          f"{'spill':>5} {'grate N':>8} {'fast N':>7} {'alpha':>6} {'tank J':>7} "
          f"{'tilt':>5} {'err mm':>6}  result")
    for r in results:
        print(f"{r['variant']:7} {r['level']:5} {r['seed']:>6} {1000 * r['belief_err']:>+6.1f}m "
              f"{str(r['tossed_per_cycle']):>12} {r['spilled']:>5} {r['peak_force']:>8.1f} "
              f"{r['peak_force_fast']:>7.1f} {r['alpha_min']:>6.2f} {r['tank_min']:>7.1f} "
              f"{r['tilt_max_deg']:>5.1f} {1000 * r['track_err_max']:>6.1f}  "
              f"{'OK' if r['success'] else r['fail_reason']}")


if __name__ == "__main__":
    main()
