"""Bilateral teleoperation onto Case 1: the human pushes the master, the slave's
GIC equilibrium follows the master, and the master pushes back with the force
the slave's spring is exerting.

    human --f_h--> master (Mm, Bm) --x_m--> x_d request --gate--> slave GIC --> paper
      ^                                                             |
      +------------- f_fb = kf * K_p (p - x_d) <---------------------+

There is no admittance anywhere.  The master's position IS the requested
equilibrium (Case 1: "the command is x_d"), and the reflected force is the GIC
spring force, which at rest in contact equals the contact force: the operator
feels the paper through the same spring that presses on it.  A stiffer K_p
makes the pen follow more tightly AND makes the paper feel harder, which is the
coupling a human exploits when they stiffen their grip to write a sharp corner.

THE OPERATOR ALSO COMMANDS K_p.  On a real rig this is a grip-force sensor, a
trigger or an EMG co-contraction estimate; in the interactive mode it is two
keys; in the synthetic writer below it is a schedule the operator follows:

    travelling         moderate everywhere
    landing            soft along the normal -- cushions the unknown paper
    writing            stiff along the paper (the letter's shape), soft along
                       the normal (the paper's height is unknown, and a soft
                       normal turns a height error into a small force error),
                       stiffer still on tight curves and corners
    lifting            normal stiffens again

It is requested as a TARGET, low-passed with the operator's own co-contraction
time constant, and turned into a stiffness rate U_p for the gate.

THE SYNTHETIC WRITER.  Built the way ../cascade/spiral_task.py builds its
operator, for the same reason: what makes an operator an operator is delay.

    visual loop    sees the pen `visual_delay` late and INTEGRATES the
                   CROSS-TRACK error away -- keeping the pen on the line, not
                   on schedule.  Two simpler loops failed: a proportional
                   correction of a 200 ms-stale error (spiral_task.py's, on a
                   slower task) rings at ~1 Hz, and a 2-D integral stores a
                   correction that is right for one direction of travel and
                   wrong after the path turns, pushing the pen outside curves.
                   It also decides when a stroke is FINISHED: the writer lifts
                   when they see the pen reach the end, not when their hand
                   does.  Lifting on the hand's schedule cut the last 4 mm off
                   short strokes, because friction makes the pen trail by ~3 mm.
    haptic loop    feels the master `haptic_delay` late, adjusts how hard it
                   presses until the felt force matches its intent
    arm            a spring-damper about the INTENDED hand motion, i.e. the
                   damping acts on (v_hand - v_master).  Damping against the
                   world would make the hand lag its own plan by Bh v / Kh,
                   1.5 mm at writing speed, which no writer does.
    motor noise    signal-dependent, plus physiological tremor

Touch is detected only after the pen has SETTLED over the start point.  The
master feels the slave's inertia while travelling (1-2 N at 8 cm/s), so a
writer who took any felt force as "touch" would decide they had landed while
still hovering -- and then press, find nothing, press harder, and hit the paper
at 25 N.  The first version of this file did exactly that.

It does NOT know where the paper is: it plans on the nominal paper, finds the
real surface by feeling for it, and never learns the tilt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import controller as C
import glyphs as G

Array = np.ndarray


class DelayLine:
    """Fixed transport delay (same semantics as cascade/core.DelayLine)."""

    def __init__(self, delay_s: float, dt: float, n: int = 3):
        self.n = max(0, int(round(delay_s / dt)))
        self.buf = [np.zeros(n) for _ in range(self.n)]

    def push_and_get(self, value) -> Array:
        value = np.asarray(value, dtype=float)
        if self.n == 0:
            return value.copy()
        self.buf.append(value.copy())
        return self.buf.pop(0)

    def fill(self, value) -> None:
        self.buf = [np.asarray(value, dtype=float).copy() for _ in range(self.n)]


# --------------------------------------------------------------------------- #
# master device
# --------------------------------------------------------------------------- #
@dataclass
class MasterParams:
    Mm: float = 1.0          # kg, apparent mass of the handle
    Bm: float = 4.0          # Ns/m
    scale: float = 1.0       # slave motion per master motion
    kf: float = 1.0          # force-feedback gain (1 = the slave spring, unscaled)
    delay_fwd: float = 0.0   # s, master -> slave
    delay_bwd: float = 0.0   # s, slave -> master
    # The handle's travel: beyond +- workspace (m, per axis, about the start)
    # a stiff virtual wall pushes back.  A real device has one; without it a
    # keyboard "lift" held for two seconds floats the pen 15 cm off the desk.
    workspace: float = 0.12
    k_wall: float = 3000.0


class TeleopSession:
    """The bilateral loop for one episode, one physics step at a time.

    Shared by the synthetic writer and the interactive keyboard mode: both hand
    over a human wrench and a stiffness target, and everything downstream of
    that is identical.
    """

    def __init__(self, sim, mp: MasterParams | None = None):
        self.sim = sim
        self.mp = mp or MasterParams()
        dt = sim.dt
        self.x_m = sim.ctl.x_d.copy()        # master starts where the slave's reference is
        self.x_m0 = self.x_m.copy()
        self.x_d0 = sim.ctl.x_d.copy()
        self.v_m = np.zeros(3)
        self.f_fb = np.zeros(3)
        self.fwd = DelayLine(self.mp.delay_fwd, dt)
        self.bwd = DelayLine(self.mp.delay_bwd, dt)
        self.fwd.fill(self.x_d0)
        self.K_cmd = sim.k_diag()

    def step(self, f_h: Array, k_target: Array) -> dict:
        sim, mp, dt = self.sim, self.mp, self.sim.dt
        # ---- master: Mm a = f_h + f_fb + f_wall - Bm v ----
        d = self.x_m - self.x_m0
        f_wall = -mp.k_wall * (d - np.clip(d, -mp.workspace, mp.workspace))
        a = (np.asarray(f_h, float) + self.f_fb + f_wall - mp.Bm * self.v_m) / mp.Mm
        self.v_m = self.v_m + a * dt
        self.x_m = self.x_m + self.v_m * dt
        x_d_req = self.fwd.push_and_get(self.x_d0 + mp.scale * (self.x_m - self.x_m0))

        # ---- Case 1 proposal: deadbeat, so with the gate open applied == requested ----
        self.K_cmd = np.asarray(k_target, dtype=float)
        K_req = C.k_world(self.K_cmd, sim.W)
        prop = C.Case1Proposal(Vd=(x_d_req - sim.ctl.x_d) / dt,
                               Up=(K_req - sim.ctl.K) / dt)
        rec = sim.step(prop)

        # ---- reflected force: the spring the slave is stretching ----
        self.f_fb = self.bwd.push_and_get(mp.kf * rec["f_G"] / mp.scale)
        rec.update(x_m=self.x_m.copy(), v_m=self.v_m.copy(), f_h=np.asarray(f_h, float).copy(),
                   f_fb=self.f_fb.copy(), x_d_req=x_d_req, k_req=self.K_cmd.copy())
        return rec


# --------------------------------------------------------------------------- #
# synthetic writer
# --------------------------------------------------------------------------- #
@dataclass
class WriterStyle:
    f_intent: float = 3.0        # N, how hard they mean to press
    v_write: float = 0.030       # m/s along the stroke
    v_travel: float = 0.070
    v_desc: float = 0.015
    v_lift: float = 0.040
    hover: float = 0.012         # m above the BELIEVED paper
    # Seconds to hover over each stroke's start before descending.  Zero for the
    # synthetic writer; protocol.py uses it to give a person at the keyboard
    # time to set the pre-contact stiffness before the pen goes down.
    hover_dwell: float = 0.0
    # the operator's arm on the master, writing frame
    Kh_plane: float = 900.0
    Kh_normal: float = 350.0
    Bh: float = 40.0
    visual_gain: float = 2.5     # 1/s, integral gain on the seen in-plane error
    visual_delay: float = 0.20
    haptic_gain: float = 0.010   # m of press per N of felt error, per second
    haptic_delay: float = 0.06
    motor_noise: float = 0.03    # fraction of |f_h|
    tremor: float = 0.04         # N, ~10 Hz physiological tremor
    f_touch: float = 0.5         # N felt before they believe they touched
    # ---- the stiffness they command, writing frame (u, v, n), N/m ----
    k_travel: float = 800.0
    k_land_n: float = 300.0
    k_write_t: float = 2000.0
    k_press_n: float = 400.0
    corner_boost: float = 0.6    # extra in-plane stiffness on tight curves, fraction
    k_tau: float = 0.15          # s, co-contraction time constant
    k_wobble: float = 0.06       # slow random modulation, fraction
    seed: int = 0

    @staticmethod
    def sample(seed: int) -> "WriterStyle":
        r = np.random.default_rng(20_000 + seed)
        return WriterStyle(
            f_intent=float(r.uniform(2.0, 4.0)),
            v_write=float(r.uniform(0.020, 0.040)),
            v_travel=float(r.uniform(0.05, 0.09)),
            Kh_plane=float(r.uniform(700, 1200)), Kh_normal=float(r.uniform(250, 450)),
            Bh=float(r.uniform(30, 50)),
            v_desc=float(r.uniform(0.010, 0.020)),
            visual_gain=float(r.uniform(1.5, 3.5)),
            visual_delay=float(r.uniform(0.15, 0.25)),
            haptic_delay=float(r.uniform(0.04, 0.08)),
            motor_noise=float(r.uniform(0.01, 0.05)),
            k_travel=float(r.uniform(600, 1200)),
            k_land_n=float(r.uniform(200, 400)),
            k_write_t=float(r.uniform(1500, 3000)),
            k_press_n=float(r.uniform(250, 600)),
            corner_boost=float(r.uniform(0.2, 0.8)),
            seed=seed)


PHASES = ["travel", "settle", "descend", "press", "stroke", "finish", "lift", "done", "finished"]


def _curvature(poly: Array, window: float = 0.003, spacing: float = 0.001) -> tuple[Array, Array]:
    """Resampled stroke and a smoothed |curvature| (1/m) at each point.  Corners
    of a polyline are spread over +-window so the operator slows before them."""
    P = G.resample(poly, spacing)
    if len(P) < 3:
        return P, np.zeros(len(P))
    d = np.gradient(P, axis=0)
    dd = np.gradient(d, axis=0)
    num = np.abs(d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0])
    den = np.maximum(np.linalg.norm(d, axis=1) ** 3, 1e-12)
    kappa = num / den
    w = max(1, int(round(window / spacing)))
    kappa = np.convolve(np.pad(kappa, w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), "same")[w:-w]
    return P, kappa


class SyntheticWriter:
    """A delayed human that writes `target` on a paper it cannot see exactly."""

    def __init__(self, sim, style: WriterStyle):
        self.sim, self.st = sim, style
        dt = sim.dt
        self.dt = dt
        self.B = sim.belief                      # the nominal paper: all it knows
        self.n = self.B.normal
        W = sim.W
        self.Kh = C.k_world([style.Kh_plane, style.Kh_plane, style.Kh_normal], W)
        self.strokes = [_curvature(s) for s in sim.target.strokes]
        self.rng = np.random.default_rng(30_000 + style.seed)
        self.vis_tip = DelayLine(style.visual_delay, dt)
        self.vis_plan = DelayLine(style.visual_delay, dt, 4)   # (u, v, normal_u, normal_v)
        self.hap = DelayLine(style.haptic_delay, dt)
        # state
        start = self.B.to_canvas(sim.ctl.x_d)
        self.plan = start.copy()                 # (u, v, h) on the believed paper
        self.vis_tip.fill(sim.ctl.x_d)
        self.vis_plan.fill(np.r_[start[:2], 0.0, 0.0])
        self.phase = "travel"
        self.k = 0
        self.press = 0.0
        self.h_contact = 0.0
        self.s = 0.0
        self.t_phase = 0.0
        self.ok_time = 0.0
        self._travel_from = start.copy()
        self._travel_T = None
        self.K_cmd = sim.k_diag()
        self._noise = np.zeros(3)
        self.corr = np.zeros(2)
        self.c_ct = 0.0
        self._hand_prev = None
        self._wob_phase = self.rng.uniform(0, 2 * np.pi, 3)
        self.done = False

    # ---- helpers ----
    def _goto(self, phase: str) -> None:
        self.phase, self.t_phase, self.ok_time = phase, 0.0, 0.0

    def _k_target(self, kappa: float) -> Array:
        st = self.st
        boost = 1.0 + st.corner_boost * min(1.0, kappa * 0.004)
        kt_w = st.k_write_t * boost
        return {
            "travel": [st.k_travel, st.k_travel, st.k_travel],
            "settle": [st.k_write_t, st.k_write_t, st.k_land_n],
            "descend": [st.k_write_t, st.k_write_t, st.k_land_n],
            "press": [kt_w, kt_w, st.k_press_n],
            "stroke": [kt_w, kt_w, st.k_press_n],
            "finish": [kt_w, kt_w, st.k_press_n],
            "lift": [st.k_write_t, st.k_write_t, st.k_travel],
            "done": [st.k_travel, st.k_travel, st.k_travel],
            "finished": [st.k_travel, st.k_travel, st.k_travel],
        }[self.phase]

    # ---- one step: returns (f_h, K target) ----
    def act(self, t: float, x_m: Array, v_m: Array, f_fb: Array, tip: Array) -> tuple[Array, Array]:
        st, dt = self.st, self.dt
        self.t_phase += dt
        felt = self.hap.push_and_get(f_fb)
        felt_n = float(felt @ self.n)
        seen_tip = self.vis_tip.push_and_get(tip)
        kappa = 0.0

        if self.phase == "travel":
            if self.k >= len(self.strokes):
                self._goto("done")
            else:
                goal = np.r_[self.strokes[self.k][0][0], st.hover]
                if self._travel_T is None:
                    self._travel_from = self.plan.copy()
                    dist = np.linalg.norm(goal - self._travel_from)
                    self._travel_T = max(0.25, dist / st.v_travel * 1.6)
                tau = min(1.0, self.t_phase / self._travel_T)
                sj = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5          # minimum jerk
                self.plan = self._travel_from + sj * (goal - self._travel_from)
                if tau >= 1.0:
                    self._travel_T = None
                    self._goto("settle")
        elif self.phase == "settle":
            # hover until nothing is felt: travel drag is not contact
            self.ok_time = self.ok_time + dt if float(np.linalg.norm(felt)) < 0.3 else 0.0
            if (self.ok_time > 0.08 or self.t_phase > 0.6) and self.t_phase >= st.hover_dwell:
                self._goto("descend")
        elif self.phase == "descend":
            self.plan[2] -= st.v_desc * dt
            if felt_n > st.f_touch and self.t_phase > 0.1:
                self.h_contact = self.plan[2]
                self.press = 0.0
                self._goto("press")
            elif self.plan[2] < -0.03:            # never found it: give up on this stroke
                self._goto("lift")
        elif self.phase in ("press", "stroke", "finish"):
            self.press += st.haptic_gain * (st.f_intent - felt_n) * dt
            self.press = float(np.clip(self.press, -0.005, 0.05))
            self.plan[2] = self.h_contact
            P, kap = self.strokes[self.k]
            if self.phase == "press":
                self.ok_time = self.ok_time + dt if abs(felt_n - st.f_intent) < 0.3 * st.f_intent else 0.0
                if self.ok_time > 0.12 or self.t_phase > 1.0:
                    self.s = 0.0
                    self._goto("stroke")
            elif self.phase == "stroke":
                i = min(int(self.s / 0.001), len(P) - 1)
                kappa = float(kap[i])
                speed = st.v_write / (1.0 + kappa * 0.004)
                self.s += speed * dt
                L = (len(P) - 1) * 0.001
                j = min(self.s / 0.001, len(P) - 1)
                j0 = int(np.floor(j)); j1 = min(j0 + 1, len(P) - 1)
                self.plan[:2] = P[j0] + (j - j0) * (P[j1] - P[j0])
                if self.s >= L:
                    self._goto("finish")
            if self.phase == "finish":
                # Hold at the end, still pressing, until the pen is SEEN there.
                # 1.5 mm, not closer: static friction parks the tip ~1 mm short
                # of a stationary reference (mu f_n / k_t), and the writer
                # cannot see the difference.
                end = self.strokes[self.k][0][-1]
                seen_end = float(np.linalg.norm(self.B.to_canvas(seen_tip)[:2] - end))
                if seen_end < 0.0015 or self.t_phase > 0.6:
                    self._goto("lift")
        elif self.phase == "lift":
            self.press *= np.exp(-dt / 0.08)
            self.plan[2] += st.v_lift * dt
            if self.plan[2] >= self.h_contact + st.hover and felt_n < 0.2:
                self.k += 1
                self.press = 0.0
                self._goto("travel")
        elif self.phase == "done":
            self.plan[2] = min(self.plan[2] + st.v_lift * dt, 0.04)
            if self.t_phase > 1.2:
                self._goto("finished")
                self.done = True

        # ---- visual loop: cross-track error seen `visual_delay` ago ----
        nrm = np.zeros(2)
        if self.phase in ("stroke", "finish"):
            P = self.strokes[self.k][0]
            j = min(int(self.s / 0.001), len(P) - 2)
            tang = P[j + 1] - P[j]
            tang = tang / max(float(np.linalg.norm(tang)), 1e-12)
            nrm = np.array([-tang[1], tang[0]])
        delayed = self.vis_plan.push_and_get(np.r_[self.plan[:2], nrm])
        ideal_seen, nrm_seen = delayed[:2], delayed[2:]
        seen_uv = self.B.to_canvas(seen_tip)[:2]
        if self.phase in ("stroke", "finish") and np.any(nrm_seen):
            e_ct = float((ideal_seen - seen_uv) @ nrm_seen)
            self.c_ct += st.visual_gain * e_ct * dt
            self.c_ct *= np.exp(-dt / 1.0)        # a slow leak: no stale memory
        else:
            self.c_ct *= np.exp(-dt / 0.3)
        self.c_ct = float(np.clip(self.c_ct, -0.004, 0.004))
        self.corr = self.c_ct * nrm
        hand_c = self.plan.copy()
        hand_c[:2] += self.corr
        if self.phase in ("press", "stroke", "finish", "lift"):
            hand_c[2] -= self.press
        hand = self.B.to_world(hand_c)
        v_hand = (hand - self._hand_prev) / dt if self._hand_prev is not None else np.zeros(3)
        self._hand_prev = hand

        # ---- the arm on the master, about the intended motion ----
        f_h = self.Kh @ (hand - x_m) + st.Bh * (v_hand - v_m)
        sig = st.motor_noise * float(np.linalg.norm(f_h)) + st.tremor
        a = dt / (dt + 1.0 / (2 * np.pi * 10.0))
        # first-order low-pass of white noise, scaled so its stationary std is sig
        self._noise += a * (self.rng.normal(0.0, sig * np.sqrt((2.0 - a) / a), 3) - self._noise)
        f_h = f_h + self._noise

        # ---- stiffness: target by phase, low-passed like a muscle ----
        wob = 1.0 + st.k_wobble * np.sin(2 * np.pi * 0.3 * t + self._wob_phase)
        k_t = np.asarray(self._k_target(kappa)) * wob
        self.K_cmd = self.K_cmd + (k_t - self.K_cmd) * min(1.0, dt / st.k_tau)
        return f_h, self.K_cmd.copy()


def run_synthetic(sim, spec, style: WriterStyle, mp: MasterParams | None = None,
                  on_step=None, max_time: float | None = None) -> dict:
    """One teleoperated episode.  `on_step(i, rec, session, writer)` is called
    after every physics step (the recorder hooks in here)."""
    sim.reset(spec)
    sess = TeleopSession(sim, mp)
    wr = SyntheticWriter(sim, style)
    T = spec.time_limit if max_time is None else max_time
    i = 0
    rec = None
    while sim.t < T and not wr.done:
        f_h, k = wr.act(sim.t, sess.x_m, sess.v_m, sess.f_fb, sim.last["p"])
        rec = sess.step(f_h, k)
        rec.update(phase=PHASES.index(wr.phase), stroke=wr.k)
        if on_step is not None:
            on_step(i, rec, sess, wr)
        i += 1
    out = sim.score()
    out["finished"] = bool(wr.done)
    out["timeout"] = not wr.done
    if not wr.done:
        out["success"] = False
        out["fail_reason"] = ",".join(filter(None, [out["fail_reason"], "timeout"]))
    return out
