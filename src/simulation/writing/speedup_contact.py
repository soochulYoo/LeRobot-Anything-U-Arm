"""S3: what retiming does in CONTACT, and what the gains have to do about it.

speedup.py settled the free-motion half: Theorem 1 is exact here, and the
admissible uniform-scaling set of Eq. (11) is small because the stiffness
ceiling binds first.  Contact is where the theory stops being a symmetry.

THE DEFECT THAT DOES NOT VANISH.  Theorem 4's R_C is the one term the gain
schedule cannot cancel, because the environment is not retimed with the robot:

    f_n = k_e delta(s) + b_e r delta'(s)

has an elastic part that does not scale like r^2 and a viscous part that scales
like r, while the controller's own spring scales like r^2.  So an unchanged
environment breaks similarity, and the manuscript's point is that it pinpoints
WHERE rather than hand-waving that contact is hard.

IN THIS SIMULATOR THE PAPER IS RIGID.  `--probe` measures it: at 5 N the pen
penetrates under 0.1 um, and f_n = k_n * d to 0.5% over the whole working range,
so k_e ~ 5e7 N/m against a controller stiffness of at most 4000.  That limit
makes Section VII sharp rather than generic.  With k_e >> k,

    sigma_x = k_e Delta / (k + k_e)   ->   Delta          (any k)
    sigma_f = k k_e Delta / (k + k_e) ->   k Delta

so the NORMAL-direction position uncertainty is the surface uncertainty itself
and no gain reduces it -- you cannot know where a rigid surface is without
touching it -- while the force uncertainty is proportional to the gain.  The
force requirement therefore caps the normal stiffness at k_n <= eps_f / Delta,
a bound with no speed in it at all.

AND THE RETIMED FORCE.  On a rigid wall the achieved orbit is pinned by the
surface, so the contact force is k_n times the depth the equilibrium is
commanded below it.  Retiming an ACHIEVED demonstration holds x_d(s) fixed and
scales K, so

    f  ->  c^2 f        the writing force, from the spring
    f_impact ~ v_0 sqrt(m k_e)  ->  c f_impact    the landing, from the speed

Two different exponents on the same axis.  Neither is a choice; both follow from
holding the demonstration and changing the clock.

WHAT THIS FILE RUNS.

    --probe     the paper's contact law and its landing impulse, measured
    --affine    Propositions 7 and 8 instantiated on the measured environment
                and the task's real limits, with the exact affine enclosure
                checked against independent stiff ODE solves
    --retime    the decisive one: recorded demonstrations replayed at rate c
                under three gain rules, scored by the simulator's own ink,
                force-band and tear criteria

    fixed       K unchanged -- the naive speed-up
    uniform     K -> c^2 K on every axis -- Theorem 1, applied where it does
                not hold
    tangential  k_t -> c^2 k_t, k_n held -- the directional reading of
                Section VIII: the tangential axes carry a position requirement
                and may be scaled, the normal axis carries a force requirement
                and may not

Usage (from src/simulation/writing):
    python3 speedup_contact.py --probe
    python3 speedup_contact.py --affine
    python3 speedup_contact.py --retime --episodes 4 --workers 12
"""
from __future__ import annotations

import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "policy"))   # rollout.FlowPolicy lives there

Array = np.ndarray
SIM_HZ = 500
POLICY_HZ = 10.0


# --------------------------------------------------------------------------- #
# the gain rules under test
# --------------------------------------------------------------------------- #
def r_fixed(k: Array, c: float) -> Array:
    return k.copy()


def r_uniform(k: Array, c: float) -> Array:
    return c * c * k


def r_tangential(k: Array, c: float) -> Array:
    return np.array([c * c * k[0], c * c * k[1], k[2]])


RULES = {"fixed": r_fixed, "uniform": r_uniform, "tangential": r_tangential}


def capped_rates(x_d: Array, hz: float, c: float, normal: Array, smooth: int = 5) -> Array:
    """A NONUNIFORM execution clock that never lands harder than the demonstration.

    The sweep says the thing that stops this task is not the writing force but
    the landing: with the gains held, the peak contact force grows like c^0.73,
    which is the filtered version of Section IV-A's v_0 sqrt(m k_e), and the
    paper tears before the ink ever goes wrong.  That is a constraint on one
    component of one axis during one phase -- exactly the kind of thing a
    uniform clock cannot express and Theorem 4's r(s) can.

    So cap the approach: at rate r the pen descends at r * v_n(s), and we ask
    that it never exceed the fastest descent the demonstration itself used.

        r(s) = min(c, v_cap / v_n(s)),   v_cap = max_s v_n(s) at 1x

    Everything that is not descending onto the paper still runs at c.  The
    result is smoothed, because Corollary 5 prices |d log r / ds| and not the
    speed factor -- an abrupt clock is expensive even when it is slow.
    """
    v = np.diff(x_d, axis=0, prepend=x_d[:1]) * hz          # m/s at 1x
    v_n = np.maximum(-(v @ normal), 0.0)                     # downward only
    v_cap = float(v_n.max())
    r = np.full(len(x_d), float(c))
    if v_cap > 1e-9:
        r = np.minimum(c, v_cap / np.maximum(v_n, v_cap / c))
    if smooth > 1:                                           # bound d log r / ds
        k = np.ones(smooth) / smooth
        r = np.convolve(np.r_[[r[0]] * smooth, r, [r[-1]] * smooth], k, "same")[smooth:-smooth]
    return np.clip(r, 1.0, c)


CLOCKS = {"uniform": None, "normal-capped": capped_rates}
RULE_DOC = {
    "fixed": "K unchanged: the naive speed-up",
    "uniform": "K -> c^2 K everywhere: Theorem 1 applied where it does not hold",
    "tangential": "k_t -> c^2 k_t, k_n held: the directional reading of Section VIII",
}


# --------------------------------------------------------------------------- #
# --probe: what the paper actually is
# --------------------------------------------------------------------------- #
def probe(depths=(0.0005, 0.001, 0.002, 0.003, 0.004, 0.005),
          speeds=(0.005, 0.010, 0.020, 0.040, 0.080), k_n=1000.0, k_t=500.0) -> dict:
    """Measure the contact law and the landing impulse of the paper.

    The press measures k_e: command the equilibrium a known depth below the
    surface and read the force and the penetration.  The landing measures the
    impact, which Section IV-A says goes as v_0 sqrt(m k_e) -- linear in the
    approach speed, so linear in the clock, not quadratic like the spring.
    """
    import sim as SM
    import controller as C
    spec = SM.TaskSpec(text="I", time_limit=1e9, seed=0, canvas_dz=0.0,
                       tilt_x=0.0, tilt_y=0.0, show_template=False, friction=0.3)
    sim = SM.WritingSim(cameras=False)
    F = None

    def drive(target_h, kn, kt, T, shape="smooth"):
        nonlocal F
        dt = sim.dt
        n = max(2, int(T / dt))
        x0 = sim.ctl.x_d.copy()
        uvh0 = F.to_canvas(x0)
        goal = F.to_world(np.r_[uvh0[:2], target_h])
        K = C.k_world([kt, kt, kn], sim.W)
        rows = []
        for i in range(1, n + 1):
            w = min(1.0, i / (0.6 * n)) if shape == "smooth" else min(1.0, i / n)
            if shape == "smooth":
                w = 10 * w ** 3 - 15 * w ** 4 + 6 * w ** 5
            xn = x0 + w * (goal - x0)
            rec = sim.step(C.Case1Proposal((xn - sim.ctl.x_d) / dt, (K - sim.ctl.K) / dt))
            rows.append((F.to_canvas(sim.contact_point(rec, F.normal))[2],
                         rec["f_n"], rec["f_sensor_n"], float(np.linalg.norm(rec["f_raw"]))))
        return np.asarray(rows)

    # ---- press ----
    sim.reset(spec)
    F = sim.frame
    press = []
    drive(0.004, k_n, k_t, 1.5)
    for d in depths:
        a = drive(-d, k_n, k_t, 1.2)
        tail = a[-int(0.25 * len(a)):]
        press.append(dict(depth_m=float(d), f_n=float(tail[:, 1].mean()),
                          penetration_m=float(-tail[:, 0].mean()),
                          k_n_times_d=float(k_n * d)))
    # one secant stiffness over the whole press, in case it is not rigid
    dd = np.array([p["penetration_m"] for p in press])
    ff = np.array([p["f_n"] for p in press])
    k_e = float(np.polyfit(dd, ff, 1)[0]) if float(dd.max() - dd.min()) > 1e-12 else float("inf")

    # ---- landing ----
    land = []
    for v in speeds:
        sim.reset(spec)
        F = sim.frame
        drive(0.006, k_n, k_t, 1.5)                       # hover 6 mm up, settled
        a = drive(-0.003, k_n, k_t, 0.009 / v, shape="ramp")   # descend at ~v
        land.append(dict(v_mps=float(v), peak_pressure_N=float(a[:, 1].max()),
                         peak_sensor_N=float(a[:, 2].max()), peak_raw_N=float(a[:, 3].max())))
    sim.close()
    vv = np.array([l["v_mps"] for l in land])
    pp = np.array([l["peak_pressure_N"] for l in land])
    ss = np.array([l["peak_sensor_N"] for l in land])
    slope = lambda y: float(np.polyfit(np.log(vv), np.log(y), 1)[0])
    return dict(press=press, k_e_fit=k_e, k_n=k_n, land=land,
                pressure_exponent=slope(pp), sensor_exponent=slope(ss))


# --------------------------------------------------------------------------- #
# --affine: Propositions 7 and 8 on the measured environment
# --------------------------------------------------------------------------- #
def h5(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def rigid_limit(k_list, Delta: float, f_star: float = 3.0) -> list:
    """Section VII when k_e >> k, which --probe says is this simulator.

    The wall pins the tip, x = delta, so the state carries no dynamics at all
    and the specialization collapses to algebra:

        f(t) = k(t) (x_d(t) - delta),   sigma_f = k(t) Delta,   sigma_x = Delta.

    Proposition 7's machinery is still correct, it is just not needed here:
    with Z -> (1, 0) the exact support bound is k Delta.  What the machinery
    buys on a COMPLIANT surface is what --affine's second half shows.
    """
    return [dict(k=float(k), sigma_x_m=float(Delta), sigma_f_N=float(k * Delta),
                 depth_m=float(f_star / k), f_nominal_N=float(f_star)) for k in k_list]


def prop8(k_e: float, Delta: float, eps_x: float, eps_f: float) -> dict:
    """The fixed-gain feasibility interval of Eq. (34), and its rigid-wall limit."""
    lo = k_e * (Delta / eps_x - 1.0) if eps_x < Delta else 0.0
    hi = (eps_f * k_e / (k_e * Delta - eps_f)) if k_e * Delta > eps_f else np.inf
    return dict(eps_x=eps_x, eps_f=eps_f, k_min=float(lo), k_max=float(hi),
                feasible=bool(lo <= hi), k_max_rigid=float(eps_f / Delta))


def paper_case(c: float = 4.0, m: float = 1.0, k_e: float = 4000.0, b_e: float = 10.0,
               Delta: float = 0.001, F: float = 10.0, kA: float = 20000.0,
               kB: float = 200.0, S: float = 4.0) -> dict:
    """Section IX-D reproduced, as a check that this implementation is the one
    the manuscript describes.

    A COMPLIANT surface, where the affine enclosure is not trivial: the
    sensitivity Z really moves during the gain ramp, and the exact support
    bound is what gives the force interval.  The published numbers to hit are
    an early position radius of 0.1667 mm, a final force radius of 0.1908 N, a
    force range of [6.667, 13.333] N, peak effort 13.333 N and a peak gain rate
    of about 241679 N/(m s) at clock rate four.

    The clock enters as X' = (1/r)(A X + b), the manuscript's nonsingular
    coordinate choice that keeps PHYSICAL velocity in the state.
    """
    from scipy.integrate import solve_ivp
    k = lambda s: kA * np.exp(np.log(kB / kA) * h5(s - 1.0))
    u = lambda s: np.clip(s - 1.0, 0.0, 1.0)
    dk = lambda s: k(s) * np.log(kB / kA) * 30.0 * u(s) ** 2 * (1.0 - u(s)) ** 2
    d = lambda s: 1.4 * np.sqrt(m * k(s))
    x_star = F / k_e
    x_d = lambda s: x_star + F / k(s)
    A = lambda s: np.array([[0.0, 1.0], [-(k(s) + k_e) / m, -(d(s) + b_e) / m]])
    b = lambda s: np.array([0.0, k(s) * x_d(s) / m])
    E = np.array([0.0, k_e / m])
    X0 = np.array([x_star, 0.0])
    Z0 = np.array([k_e / (kA + k_e), 0.0])
    kw = dict(method="Radau", rtol=1e-12, atol=1e-15, dense_output=True)
    nom = solve_ivp(lambda s, y: (A(s) @ y + b(s)) / c, (0, S), X0, **kw)
    sen = solve_ivp(lambda s, y: (A(s) @ y + E) / c, (0, S), Z0, **kw)
    ss = np.linspace(0, S, 4001)
    Xb, Zt = nom.sol(ss), sen.sol(ss)
    err = 0.0
    for dl in np.linspace(-Delta, Delta, 5):
        full = solve_ivp(lambda s, y: (A(s) @ y + b(s) + E * dl) / c, (0, S),
                         X0 + Z0 * dl, **kw)
        err = max(err, float(np.abs(full.sol(ss) - (Xb + Zt * dl)).max()))
    cf = np.array([k_e, b_e])
    f_bar = cf @ Xb
    rad_f = Delta * np.abs(cf @ Zt - k_e)
    rad_x = Delta * np.abs(Zt[0])
    kk = np.array([k(s) for s in ss])
    dd = np.array([d(s) for s in ss])
    xdv = np.array([x_d(s) for s in ss])
    eff = np.abs(kk * (xdv - Xb[0]) - dd * Xb[1]) + kk * rad_x + dd * Delta * np.abs(Zt[1])
    early, final = ss <= 1.0, ss >= 3.0
    return dict(c=c, early_pos_radius_mm=float(1000 * rad_x[early].max()),
                final_force_radius_N=float(rad_f[final].max()),
                force_lo_N=float((f_bar - rad_f).min()), force_hi_N=float((f_bar + rad_f).max()),
                peak_effort_N=float(eff.max()),
                peak_gain_rate=float(c * np.abs(np.array([dk(s) for s in ss])).max()),
                penetration_min_m=float((Xb[0] - rad_x).min()),
                affine_vs_ode=float(err))


# --------------------------------------------------------------------------- #
# --retime: demonstrations replayed faster, scored by the simulator
# --------------------------------------------------------------------------- #
_ENV = None


def run_recorded(env, policy, spec, n_subs=None) -> dict:
    """evaluate.run_episode, plus what the pen was actually DOING.

    The point of a speed-up is the clock on the wall, so the episode has to
    report seconds and metres per second and not only the rate factor it was
    asked for: completion time, the time the pen spent down, and the in-plane
    speed it wrote at.  Writing speed is measured while the pen is down, which
    is the only part of the episode the force band and the ink rule judge.
    """
    obs = env.reset(spec)
    policy.reset(env.goal(), obs)
    rows = []
    i = 0
    while not env.time_up:
        a = policy.act(obs)
        if a is None:
            break
        if n_subs is not None:                 # a per-frame clock: r(s) frame by frame
            env.n_sub = int(n_subs[min(i, len(n_subs) - 1)])
        obs = env.step(a)
        i += 1
        v = np.asarray(obs["tcp_vel"], float)
        rows.append((env.sim.t, float(np.linalg.norm(v[:2])), float(np.linalg.norm(v)),
                     bool(env.sim.last["pen_down"])))
    res = env.score()
    L = np.asarray(rows, float)
    down = L[:, 3] > 0.5
    res.update(v_write_mean=float(L[down, 1].mean()) if down.any() else float("nan"),
               v_write_peak=float(L[down, 1].max()) if down.any() else float("nan"),
               v_tip_peak=float(L[:, 2].max()))
    return res


def _init():
    global _ENV
    import evaluate as EV
    _ENV = EV.WritingPolicyEnv(control_hz=POLICY_HZ, cameras=False)


def _job(job):
    import evaluate as EV
    path, c, rule_name, clock_name = job
    env = _ENV
    rule = RULES[rule_name]
    pol = EV.ReplayPolicy(path)
    base = 1.0 / (pol.hz * env.sim.dt)                       # substeps per frame at 1x
    n_subs = None
    if clock_name == "uniform":
        n_sub = max(1, int(round(base / c)))
        c_eff = base / n_sub
        h_bar = 0.0
    else:
        normal = np.asarray(env.sim.belief.normal, float)
        r = CLOCKS[clock_name](pol.x_d, pol.hz, c, normal)
        n_subs = np.maximum(1, np.round(base / r)).astype(int)
        r_eff = base / n_subs
        c_eff = len(n_subs) * base / n_subs.sum()            # overall, = T_1x / T
        n_sub = int(n_subs[0])
        # Corollary 5's quantity, in demonstration-time units
        h_bar = float(np.abs(np.diff(np.log(r_eff))).max() * pol.hz)
    env.n_sub = n_sub
    # the retimer carries the applied-stiffness ceiling, as S1 established; the
    # STOCK ceiling is reported separately as the Eq. (11) resource limit
    k_peak = float(rule(np.abs(pol.k).max(0), c).max())
    env.sim.ctl.g.k_hi = max(4000.0, 1.05 * k_peak)

    class Retimed(EV.ReplayPolicy):
        def __init__(self):
            self.__dict__.update(pol.__dict__)

        def act(self, obs):
            a = EV.ReplayPolicy.act(self, obs)
            if a is None:
                return None
            return {"x_d": a["x_d"], "k_diag": rule(np.asarray(a["k_diag"], float), c)}

    t0 = time.time()
    r = run_recorded(env, Retimed(), pol.spec, n_subs)
    r.pop("checks", None)
    r.update(c=float(c), c_eff=float(c_eff), rule=rule_name, clock=clock_name,
             h_bar=h_bar, n_sub=n_sub,
             ep=pathlib.Path(path).stem, text=pol.spec.text, seed=pol.spec.seed,
             k_peak=k_peak, wall_s=time.time() - t0)
    return r


def retime(files, cs, rules, clocks, workers: int) -> list:
    import multiprocessing as mp
    jobs = [(f, c, r, k) for f in files for c in cs for r in rules for k in clocks]
    out = []
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for r in pool.imap_unordered(_job, jobs):
            out.append(r)
            print(f"  {r['ep']:>10} {r['rule']:<11}{r['clock']:<15} c={r['c_eff']:<5.3g} "
                  f"{'OK  ' if r['success'] else 'FAIL'} "
                  f"peak {r['peak_force']:5.2f} N  band {r['in_band']:.3f}  "
                  f"cov {r['coverage']:.3f}  prec {r['precision']:.3f}"
                  f"{'  TORN' if r['torn'] else ''}  {r['fail_reason']}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# --policy: the same retiming, but with the trained policy in the loop
# --------------------------------------------------------------------------- #
TEXTS = ["S", "7", "<star>"]
DATA_HIST = 10          # data.HIST: frames of force history the policy is given


OBS_ABLATIONS = ("none", "vel", "k", "ft", "all")


class Retimed:
    """A trained policy whose chunk is consumed `c` times faster, with its own
    stiffness put through a gain rule.

    The policy emits set-points spaced one demonstration frame apart and the
    environment walks to each of them over `n_sub` physics substeps, so dividing
    `n_sub` by `c` is exactly the retiming of Theorem 4 applied to a plan that
    is being generated online.  Nothing else changes: the policy still sees the
    cameras, its own pose, its own stiffness and the last second of force, and
    still replans every `n_exec` set-points -- at `c` times the physical rate,
    which is itself off the distribution it was trained on.

    `ablate` doctors ONE input back to what the policy was trained on, to find
    out which one the retiming actually breaks:

        vel   tcp_vel / c      -- the twist the demonstration clock would show
        k     k_diag / c^2     -- undo the gain rule in the OBSERVATION only
        ft    a force history covering the same PHYSICAL window as 1x, by
              keeping c times more samples and handing the policy every c-th

    None of these is a controller a robot could run: `k` lies about the arm's
    own stiffness and `vel` about its own speed.  They are a diagnosis, and the
    point is to learn whether a rate-conditioned policy needs to be told the
    rate (if one input explains the failure) or retrained on retimed data (if
    they all contribute).
    """

    def __init__(self, pol, rule, c, ablate=()):
        import collections
        self.pol, self.rule, self.c = pol, rule, float(c)
        self.ablate = set(ablate) - {"none"}
        if "all" in self.ablate:
            self.ablate = {"vel", "k", "ft"}
        self.stride = max(1, int(round(self.c)))
        self.long = collections.deque(maxlen=DATA_HIST * self.stride)

    def reset(self, goal, obs):
        self.pol.reset(goal, obs)
        self.long.clear()

    def act(self, obs):
        o = obs
        if self.ablate & {"vel", "k"}:
            o = dict(obs)
            if "vel" in self.ablate:
                o["tcp_vel"] = np.asarray(obs["tcp_vel"], float) / self.c
            if "k" in self.ablate:
                o["k_diag"] = np.asarray(obs["k_diag"], float) / (self.c ** 2)
        # FlowPolicy.act, unrolled, because the force history has to be built
        # here rather than appended to by the policy
        f = np.asarray(obs["f_contact"], np.float32)
        if "ft" in self.ablate:
            import collections
            self.long.append(f)
            back = list(self.long)[::-1][::self.stride][:DATA_HIST][::-1]
            while len(back) < DATA_HIST:
                back.insert(0, back[0])
            self.pol.hist = collections.deque(back, maxlen=DATA_HIST)
        else:
            self.pol.hist.append(f)
        if not self.pol.queue:
            self.pol._plan(o)
        a = self.pol.queue.pop(0)
        return {"x_d": a["x_d"],
                "k_diag": self.rule(np.asarray(a["k_diag"], float), self.c)}


def run_policy(env, policy, spec, frames_every: int = 0, cap=None, c: float = 1.0) -> dict:
    """One closed-loop episode, recording what the pen did and when it stopped.

    A learned policy does not end its own episode, so "completion" is read off
    the ink: the physical time of the LAST mark it laid.  That is the number a
    speed-up has to move, and it is not the same as the time budget.
    """
    obs = env.reset(spec)
    policy.reset(env.goal(), obs)
    rows, shots, n_ink, t_last = [], [], 0, 0.0
    ink_t, base = [], 1.0 / (POLICY_HZ * env.sim.dt)
    n_sub0, normal = env.n_sub, np.asarray(env.sim.belief.normal, float)
    subs = 0
    while not env.time_up:
        a = policy.act(obs)
        if a is None:
            break
        if cap:
            # the nonuniform clock, online: the plan is being generated as it is
            # retimed, so r(s) is read off the set-point the policy just emitted
            v_n = max(0.0, -float((np.asarray(a["x_d"], float)
                                   - env.sim.ctl.x_d) @ normal) * POLICY_HZ)
            r_s = min(c, cap / max(v_n, cap / c)) if v_n > 0 else c
            env.n_sub = max(1, int(round(base / r_s)))
        subs += env.n_sub
        obs = env.step(a)
        sim = env.sim
        v = np.asarray(obs["tcp_vel"], float)
        if int(obs["n_ink"]) > n_ink:
            n_ink, t_last = int(obs["n_ink"]), float(sim.t)
            ink_t.append((float(sim.t), n_ink))
        rows.append((sim.t, float(np.linalg.norm(v[:2])), float(np.linalg.norm(v)),
                     bool(sim.last["pen_down"]), float(sim.last["f_n"]),
                     *np.asarray(sim.k_diag(), float)))
        if frames_every and len(rows) % frames_every == 0 and env.frames:
            shots.append(env.frames[-1])
    res = env.score()
    L = np.asarray(rows, float)
    down = L[:, 3] > 0.5
    # t95: when 95% of the ink was down.  t_last_ink is a max over marks and one
    # stray late touch moves it by seconds; the task is effectively finished at t95.
    t95 = 0.0
    if ink_t:
        need = 0.95 * ink_t[-1][1]
        t95 = next(t for t, k in ink_t if k >= need)
    res.update(t_last_ink=t_last, t95_ink=t95,
               c_eff_run=(len(rows) * n_sub0 / max(subs, 1)) * (base / max(n_sub0, 1)),
               v_write_mean=float(L[down, 1].mean()) if down.any() else float("nan"),
               v_write_peak=float(L[down, 1].max()) if down.any() else float("nan"),
               v_tip_peak=float(L[:, 2].max()),
               trace=dict(t=L[:, 0].tolist(), f=L[:, 4].tolist(), k=L[:, 5:8].tolist()))
    return res


def demo_descent_cap(data: str, normal) -> float:
    """The fastest the DEMONSTRATIONS ever drove the equilibrium at the paper.

    The online clock caps the approach against this, so a retimed policy may
    land no harder than the behaviour it was trained on -- which is the one
    place S3 found the contact force actually binding.
    """
    import h5py
    n, hz, v = np.asarray(normal, float), POLICY_HZ, 0.0
    for f in sorted(glob.glob(f"{data}/*/ep_*.h5")):
        with h5py.File(f) as h:
            x = h["action/x_d"][:].astype(float)
        d = np.diff(x, axis=0, prepend=x[:1]) * hz
        v = max(v, float(np.maximum(-(d @ n), 0.0).max()))
    return v


_PENV = _POL = None


def _pinit(ckpt, n_exec, video):
    global _PENV, _POL
    from evaluate import WritingPolicyEnv
    from rollout import FlowPolicy
    _PENV = WritingPolicyEnv(control_hz=POLICY_HZ, cameras=True,
                             render_mode="rgb_array" if video else None)
    _PENV.record_video = bool(video)
    _POL = FlowPolicy(ckpt, n_exec=n_exec)


def _pjob(job):
    import dataclasses
    import protocol as P
    case, attempt, duration, c, rule_name, clock_name, n_exec, abl = job
    env, pol = _PENV, _POL
    pol.n_exec = int(n_exec)         # set-points executed before the policy replans
    rule = RULES[rule_name]
    spec, _, seed = P.episode_spec(TEXTS[case], case, attempt, 60_000)
    # the TIME BUDGET shrinks with the rate: finishing c times faster has to mean
    # finishing, not being given the same wall clock and writing more slowly
    spec = dataclasses.replace(spec, time_limit=duration / c)
    base = 1.0 / (POLICY_HZ * env.sim.dt)
    env.n_sub = max(1, int(round(base / c)))
    # the retimer carries the applied-stiffness ceiling, as S1 established
    env.sim.ctl.g.k_hi = 4000.0 if rule_name == "fixed" else max(4000.0, 1.05 * c * c * 4000.0)
    t0 = time.time()
    r = run_policy(env, Retimed(pol, rule, c, abl), spec,
                   cap=(_VCAP if clock_name == "normal-capped" else None), c=c)
    r.pop("checks", None)
    r.pop("trace", None)
    r.update(c=float(c), c_eff=float(r.pop("c_eff_run", base / env.n_sub)),
             rule=rule_name, clock=clock_name, h_bar=0.0, n_exec=int(n_exec),
             obs_ablate="+".join(sorted(abl)) if abl else "none",
             n_sub=env.n_sub, ep=f"{TEXTS[case].strip('<>')}{attempt}", text=TEXTS[case],
             seed=seed, k_peak=float(env.sim.ctl.g.k_hi), wall_s=time.time() - t0)
    return r


_VCAP = None


def _pinit2(ckpt, n_exec, video, vcap):
    global _VCAP
    _VCAP = vcap
    _pinit(ckpt, n_exec, video)


def policy_sweep(ckpt, cases, attempts, durations, cs, rules, clocks, n_execs,
                 workers, vcap=None, ablations=(("none",),)) -> list:
    import multiprocessing as mp
    jobs = [(c_, a, durations[c_], cc, r, k, ne, ab)
            for c_ in cases for a in attempts for cc in cs
            for r in rules for k in clocks for ne in n_execs for ab in ablations]
    out = []
    with mp.get_context("spawn").Pool(workers, initializer=_pinit2,
                                      initargs=(str(ckpt), n_execs[0], False, vcap)) as pool:
        for r in pool.imap_unordered(_pjob, jobs):
            out.append(r)
            print(f"  {r['ep']:>9} {r['rule']:<11}{r['clock']:<15}"
                  f"obs={r['obs_ablate']:<12}n_exec={r['n_exec']} "
                  f"c={r['c']:<5.3g} {'OK  ' if r['success'] else 'FAIL'} "
                  f"t95 {r['t95_ink']:5.2f}s  peak {r['peak_force']:5.2f} N  "
                  f"band {r['in_band']:.3f}  cov {r['coverage']:.3f}"
                  f"{'  TORN' if r['torn'] else ''}  {r['fail_reason']}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# --video: the same paper, written at different rates, side by side in REAL TIME
# --------------------------------------------------------------------------- #
def record(ckpt, case, attempt, duration, configs, n_exec, out_mp4, fps=30, shrink=2):
    """One episode per configuration, composed on a shared PHYSICAL time axis.

    Real time is the whole point, so the panels start together and each one
    freezes when its own episode ends: the faster run simply stops moving first,
    and the gap at the end IS the speed-up.  Underneath, the contact force of
    every configuration on one axis, with the band the score requires and the
    tear threshold it must not cross.
    """
    import dataclasses
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    import protocol as P
    from evaluate import WritingPolicyEnv
    from rollout import FlowPolicy

    env = WritingPolicyEnv(control_hz=POLICY_HZ, cameras=True, render_mode="rgb_array")
    env.record_video = True
    pol = FlowPolicy(ckpt, n_exec=n_exec)
    spec, _, seed = P.episode_spec(TEXTS[case], case, attempt, 60_000)
    runs = []
    for c, rule_name in configs:
        env.n_sub = max(1, int(round(1.0 / (POLICY_HZ * env.sim.dt) / c)))
        env.sim.ctl.g.k_hi = (4000.0 if rule_name == "fixed"
                              else max(4000.0, 1.05 * c * c * 4000.0))
        r = run_policy(env, Retimed(pol, RULES[rule_name], c),
                       dataclasses.replace(spec, time_limit=duration / c))
        fr = [np.asarray(f)[::shrink, ::shrink] for f in env.frames]
        tr = r["trace"]
        runs.append(dict(c=c, rule=rule_name, frames=fr, t=np.asarray(tr["t"], float),
                         f=np.asarray(tr["f"], float), res=r))
        print(f"  {rule_name:<11} c={c:<4g} {'SUCCESS' if r['success'] else 'fail: ' + r['fail_reason']}"
              f"  last ink {r['t_last_ink']:.2f}s  peak {r['peak_force']:.1f} N"
              f"{'  TORN' if r['torn'] else ''}  ({len(fr)} frames)", flush=True)

    T = max(float(r["t"][-1]) for r in runs)
    grid = np.arange(0.0, T + 1e-9, 1.0 / fps)
    h, w = runs[0]["frames"][0].shape[:2]
    n = len(runs)
    fig = plt.figure(figsize=(n * w / 100, h / 100 + 2.6), dpi=100)
    gs = fig.add_gridspec(2, n, height_ratios=[h / 100, 2.4], hspace=0.28, wspace=0.03)
    ims, titles = [], []
    for i, r in enumerate(runs):
        ax = fig.add_subplot(gs[0, i])
        ims.append(ax.imshow(r["frames"][0]))
        ax.axis("off")
        lab = ("1x, as demonstrated" if r["c"] == 1 else
               f"{r['c']:g}x, " + {"fixed": "gains unchanged",
                                   "uniform": "K -> c^2 K (Theorem 1)",
                                   "tangential": "k_t -> c^2 k_t"}[r["rule"]])
        titles.append(ax.set_title(lab, fontsize=10))
    axf = fig.add_subplot(gs[1, :])
    axf.axhspan(1.0, 6.0, color="#1baf7a", alpha=0.10, lw=0)
    axf.axhline(0.8, color="#9CA3AF", lw=0.8, ls=":")
    axf.axhline(12.0, color="#d62a2a", lw=1.0, ls="--")
    axf.text(0.01, 12.0, " tear", color="#d62a2a", fontsize=8, va="bottom")
    col = ["#2a78d6", "#8a8880", "#eb6834", "#1baf7a"]
    lns = []
    for i, r in enumerate(runs):
        (ln,) = axf.plot([], [], color=col[i % len(col)], lw=1.6,
                         label=titles[i].get_text())
        lns.append(ln)
    cur = axf.axvline(0, color="#6B7280", lw=0.8)
    top = max(13.0, max(float(r["f"].max()) for r in runs) * 1.1)
    axf.set_xlim(0, T), axf.set_ylim(-0.3, top)
    axf.set_xlabel("physical time (s)"), axf.set_ylabel("contact force (N)")
    axf.grid(alpha=0.18, lw=0.5), axf.legend(fontsize=8, frameon=False, ncol=n, loc="upper left")
    fig.suptitle(f"{TEXTS[case]}  seed {seed}: the same paper, written at different rates",
                 fontsize=11)
    pathlib.Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_mp4, fps=fps, macro_block_size=1, quality=8) as vid:
        for ts in grid:
            for i, r in enumerate(runs):
                j = int(np.searchsorted(r["t"], ts))
                j = min(max(j, 0), len(r["frames"]) - 1)
                ims[i].set_data(r["frames"][j])
                m = r["t"] <= ts
                lns[i].set_data(r["t"][m], r["f"][m])
            cur.set_xdata([ts, ts])
            fig.canvas.draw()
            vid.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    print(f"  wrote {out_mp4}  ({len(grid)} frames at {fps} fps, {T:.2f} s real time)")
    return [dict(c=r["c"], rule=r["rule"],
                 **{k: r["res"][k] for k in ("success", "fail_reason", "coverage",
                                             "precision", "in_band", "peak_force",
                                             "torn", "t_last_ink", "v_write_mean")})
            for r in runs]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--affine", action="store_true")
    ap.add_argument("--retime", action="store_true")
    ap.add_argument("--policy", default=None, metavar="CKPT",
                    help="retime a TRAINED policy instead of a recorded demonstration")
    ap.add_argument("--n-exec", nargs="*", type=int, default=[3],
                    help="set-points executed before the policy replans; lowering it "
                         "raises the feedback rate PER UNIT PHASE, which retiming cuts")
    ap.add_argument("--obs-ablate", nargs="*", default=["none"],
                    help="doctor one observation back to its 1x value: "
                         "none, vel, k, ft, all (or e.g. vel+ft)")
    ap.add_argument("--video", action="store_true",
                    help="record a side-by-side comparison on one paper")
    ap.add_argument("--video-c", type=float, default=3.0)
    ap.add_argument("--video-text", default="S", choices=TEXTS)
    ap.add_argument("--video-attempt", type=int, default=500)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--attempts", nargs="*", type=int, default=None,
                    help="randomization indices; 500+ are unseen by training")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the report from contact_retime.json")
    ap.add_argument("--data", default="demos/protocol_v1")
    ap.add_argument("--episodes", type=int, default=4, help="per text")
    ap.add_argument("--c", nargs="*", type=float, default=[1, 1.25, 1.5, 2, 2.5, 3, 4])
    ap.add_argument("--rules", nargs="*", default=list(RULES), choices=list(RULES))
    ap.add_argument("--clocks", nargs="*", default=["uniform"], choices=list(CLOCKS),
                    help="uniform r(s) = c, or normal-capped: a nonuniform clock that "
                         "never lands harder than the demonstration did")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--delta", type=float, default=0.004,
                    help="surface-offset uncertainty, m (TaskSpec.canvas_dz is +-4 mm)")
    ap.add_argument("--out", default="logs/speedup")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.probe:
        p = probe()
        print("The paper, measured\n")
        print(f"  press at k_n = {p['k_n']:g} N/m")
        print(f"  {'depth (mm)':>12}{'f_n (N)':>10}{'k_n d (N)':>11}{'penetration (um)':>19}")
        for r in p["press"]:
            print(f"  {1000 * r['depth_m']:>12.2f}{r['f_n']:>10.3f}{r['k_n_times_d']:>11.3f}"
                  f"{1e6 * r['penetration_m']:>19.4f}")
        print(f"\n  -> k_e ~ {p['k_e_fit']:.3g} N/m against a controller ceiling of 4000:")
        print("     the wall is RIGID, so f_n = k_n * (depth of x_d below the surface),")
        print("     sigma_x -> Delta for every gain and sigma_f -> k_n Delta.")
        print(f"\n  landing, descending onto the paper at k_n = {p['k_n']:g} N/m")
        print(f"  {'v (mm/s)':>10}{'pressure (N)':>14}{'sensor (N)':>12}{'raw (N)':>10}")
        for r in p["land"]:
            print(f"  {1000 * r['v_mps']:>10.0f}{r['peak_pressure_N']:>14.3f}"
                  f"{r['peak_sensor_N']:>12.3f}{r['peak_raw_N']:>10.2f}")
        print(f"\n  peak pressure ~ v^{p['pressure_exponent']:.2f},  "
              f"peak sensor ~ v^{p['sensor_exponent']:.2f}")
        print("  Section IV-A's v_0 sqrt(m k_e): the landing is LINEAR in the clock,")
        print("  while the writing force is quadratic.  Two exponents on one axis.")
        (out / "contact_probe.json").write_text(json.dumps(p, indent=1))
        return

    if args.affine:
        pr = json.loads((out / "contact_probe.json").read_text()) if (
            out / "contact_probe.json").exists() else None
        k_e = pr["k_e_fit"] if pr else 2.7e7
        D = args.delta
        band = (1.0, 6.0)
        eps_f = 0.5 * (band[1] - band[0])
        print("Section VII on the environment this simulator actually has\n")
        print(f"  k_e ~ {k_e:.3g} N/m (--probe), Delta {1000 * D:g} mm (TaskSpec.canvas_dz),")
        print(f"  force band {band} N -> eps_f {eps_f:g} N, position tolerance 3 mm\n")
        print("  k_e >> k, so the wall pins the tip and the specialization is algebra:")
        print("  f = k (x_d - delta),  sigma_x = Delta for EVERY gain,  sigma_f = k Delta.\n")
        ks = [250.0, 500.0, 625.0, 1000.0, 2000.0, 4000.0]
        rows = rigid_limit(ks, D)
        print(f"  {'k_n (N/m)':>11}{'depth for 3 N (mm)':>21}{'sigma_x (mm)':>14}"
              f"{'sigma_f (N)':>13}{'in band?':>10}")
        for r in rows:
            ok = r["sigma_f_N"] <= eps_f
            print(f"  {r['k']:>11.0f}{1000 * r['depth_m']:>21.3f}{1000 * r['sigma_x_m']:>14.3f}"
                  f"{r['sigma_f_N']:>13.3f}{'yes' if ok else 'NO':>10}")
        p8 = prop8(k_e, D, eps_x=0.003, eps_f=eps_f)
        print(f"\n  Proposition 8, Eq. (34)")
        print(f"    k >= {p8['k_min']:.4g} N/m   from the 3 mm position radius")
        print(f"    k <= {p8['k_max']:.4g} N/m   from the {eps_f:g} N force radius")
        print(f"    -> {'feasible' if p8['feasible'] else 'EMPTY: no fixed gain in the declared class'}")
        print(f"\n  The position bound diverges because sigma_x = Delta whatever the gain:")
        print(f"  a blind controller cannot find a rigid surface, and no stiffness helps.")
        print(f"  What survives is a force cap with NO SPEED IN IT,")
        print(f"      k_n <= eps_f / Delta = {p8['k_max_rigid']:.0f} N/m.")
        print(f"  The demonstrations write at k_n = 1000 N/m, where sigma_f would be "
              f"{1000 * D * 1.0:.1f} N:")
        print(f"  they are in band only because the operator FELT for the paper, which is")
        print(f"  the observer escape the manuscript names right after Proposition 8.")
        print(f"\n  Retiming multiplies k_n by c^2 and sigma_f with it, so a demonstration")
        print(f"  already at the cap has NO uniform headroom at all.")

        print("\n\nSection IX-D reproduced, as a check that this is the manuscript's method")
        print("(a COMPLIANT surface, k_e 4000 N/m, where the enclosure is not trivial)\n")
        pc = paper_case(c=4.0)
        pub = dict(early_pos_radius_mm=0.1667, final_force_radius_N=0.1908,
                   force_lo_N=6.667, force_hi_N=13.333, peak_effort_N=13.333,
                   peak_gain_rate=241679.0)
        print(f"  {'quantity':<28}{'here':>14}{'published':>14}")
        for key, lab in (("early_pos_radius_mm", "early position radius (mm)"),
                         ("final_force_radius_N", "final force radius (N)"),
                         ("force_lo_N", "force range low (N)"),
                         ("force_hi_N", "force range high (N)"),
                         ("peak_effort_N", "peak effort (N)"),
                         ("peak_gain_rate", "peak gain rate (N/m/s)")):
            print(f"  {lab:<28}{pc[key]:>14.4f}{pub[key]:>14.4f}")
        print(f"  {'penetration stays positive':<28}{str(pc['penetration_min_m'] > 0):>14}")
        print(f"  {'affine vs independent ODE':<28}{pc['affine_vs_ode']:>14.2e}")
        (out / "contact_affine.json").write_text(json.dumps(
            dict(rigid=rows, prop8=p8, paper_case=pc, k_e=k_e, Delta=D), indent=1))
        return

    if args.replot:
        d = json.loads((out / "contact_retime.json").read_text())
        report(d["episodes"], d.get("delta", args.delta), out)
        return

    if args.policy and args.video:
        rows_j = [json.loads(l) for l in
                  (pathlib.Path(args.data) / "attempts.jsonl").read_text().splitlines()]
        case = TEXTS.index(args.video_text)
        duration = 1.3 * max(r["t"] for r in rows_j if r["text"] == args.video_text)
        cfg = [(1.0, "fixed"), (args.video_c, "fixed"), (args.video_c, "uniform")]
        print(f"recording {args.video_text} attempt {args.video_attempt}: "
              + ", ".join(f"{c:g}x {r}" for c, r in cfg))
        res = record(args.policy, case, args.video_attempt, duration, cfg, args.n_exec[0],
                     out / f"speedup_{args.video_text.strip('<>')}_{args.video_c:g}x.mp4",
                     fps=args.fps)
        (out / f"speedup_{args.video_text.strip('<>')}_{args.video_c:g}x.json").write_text(
            json.dumps(res, indent=1))
        return

    if args.policy:
        import protocol as P  # noqa: F401  (imported in the workers too)
        rows_j = [json.loads(l) for l in
                  (pathlib.Path(args.data) / "attempts.jsonl").read_text().splitlines()]
        dur = {i: 1.3 * max(r["t"] for r in rows_j if r["text"] == t)
               for i, t in enumerate(TEXTS)}
        attempts = args.attempts or list(range(500, 500 + args.episodes))
        n = (len(TEXTS) * len(attempts) * len(args.c) * len(args.rules)
             * len(args.clocks) * len(args.n_exec) * len(args.obs_ablate))
        print(f"{pathlib.Path(args.policy).parent.parent.name}/"
              f"{pathlib.Path(args.policy).parent.name}: {len(TEXTS)} texts x "
              f"{len(attempts)} unseen papers x {len(args.c)} rates x "
              f"{len(args.rules)} rules x {len(args.clocks)} clocks x "
              f"{len(args.n_exec)} n_exec x {len(args.obs_ablate)} obs = {n} episodes")
        print(f"  the time budget is 1.3x the longest demonstration DIVIDED BY c, so a "
              f"faster\n  run has to finish, not merely start.\n")
        for r in args.rules:
            print(f"  {r:<11} {RULE_DOC[r]}")
        print()
        t0 = time.time()
        vcap = None
        if "normal-capped" in args.clocks:
            import scene as S
            vcap = demo_descent_cap(args.data, S.CanvasFrame(np.zeros(3)).normal)
            print(f"  demonstration descent cap: {1000 * vcap:.1f} mm/s\n")
        rows = policy_sweep(args.policy, range(len(TEXTS)), attempts, dur, args.c,
                            args.rules, args.clocks, args.n_exec, args.workers, vcap,
                            [tuple(a.split("+")) for a in args.obs_ablate])
        print(f"\n({(time.time() - t0) / 60:.1f} min)\n")
        report(rows, args.delta, out, tag="policy")
        return

    if args.retime:
        files = []
        for d in sorted({pathlib.Path(f).parent for f in glob.glob(f"{args.data}/*/ep_*.h5")}):
            files += sorted(glob.glob(f"{d}/ep_*.h5"))[:args.episodes]
        if not files:
            raise SystemExit(f"no demonstrations under {args.data}")
        n = len(files) * len(args.c) * len(args.rules) * len(args.clocks)
        print(f"{len(files)} demonstrations x {len(args.c)} rates x {len(args.rules)} rules "
              f"x {len(args.clocks)} clocks = {n} episodes\n")
        for r in args.rules:
            print(f"  {r:<11} {RULE_DOC[r]}")
        for k in args.clocks:
            print(f"  {k:<11} " + ("r(s) = c" if k == "uniform" else
                                   "r(s) = min(c, v_cap / v_n(s)): never land harder than 1x"))
        print()
        t0 = time.time()
        rows = retime(files, args.c, args.rules, args.clocks, args.workers)
        print(f"\n({(time.time() - t0) / 60:.1f} min)\n")
        report(rows, args.delta, out)
        return

    ap.error("pick one of --probe, --affine, --retime")


def report(rows: list, Delta: float, out: pathlib.Path, tag: str = "retime") -> None:
    for x in rows:
        t = x["rule"] if x["clock"] == "uniform" else f"{x['rule']}+capped"
        if len({r.get("n_exec", 3) for r in rows}) > 1:
            t += f" x{x.get('n_exec', 3)}"
        if len({r.get("obs_ablate", "none") for r in rows}) > 1:
            t += f" [{x.get('obs_ablate', 'none')}]"
        x["tag"] = t
    tags, seen = [], set()
    for x in rows:
        if x["tag"] not in seen:
            seen.add(x["tag"]); tags.append(x["tag"])
    # a per-frame clock gives each episode its own overall rate, so bin by request
    cs = sorted({x["c"] for x in rows})
    sel = lambda tag, c: [x for x in rows if x["tag"] == tag and x["c"] == c]
    key = "t_last_ink" if "t_last_ink" in rows[0] else "t"
    for x in rows:
        x["t"] = x[key]
    t_1x = float(np.mean([x["t"] for x in rows if x["c"] == min(cs)]))
    print(f"  The demonstration takes {t_1x:.2f} s.  What each rule BUYS, and whether it\n"
          f"  still writes:\n")
    print(f"  {'rule':<18}{'c':>5}{'time (s)':>10}{'faster':>8}{'v_write':>9}{'success':>9}"
          f"{'coverage':>10}{'precision':>11}{'in band':>9}{'peak f':>9}{'torn':>7}")
    print(f"  {'':<18}{'req':>5}{'':>10}{'actual':>8}{'(mm/s)':>9}{'':>9}{'':>10}{'':>11}{'':>9}"
          f"{'(N)':>9}{'':>7}")
    summary = {}
    for rule in tags:
        for c in cs:
            g = sel(rule, c)
            if not g:
                continue
            m = lambda k: float(np.nanmean([x[k] for x in g]))
            s = dict(success=m("success"), coverage=m("coverage"), precision=m("precision"),
                     in_band=m("in_band"), peak_force=m("peak_force"),
                     peak_force_fast=m("peak_force_fast"), t=m("t"),
                     speedup=(t_1x / m("t") if m("t") > 1e-9 else float("nan")),
                     pen_down_s=m("pen_down_s"), v_write_mean=m("v_write_mean"),
                     c_eff=m("c_eff"), h_bar=m("h_bar"),
                     v_write_peak=m("v_write_peak"), v_tip_peak=m("v_tip_peak"),
                     torn=m("torn"), k_peak=m("k_peak"), n=len(g))
            summary[f"{rule}@{c:g}"] = s
            sp = (f"{s['speedup']:>7.2f}x" if np.isfinite(s["speedup"]) else f"{'--':>8}")
            vw = (f"{1000 * s['v_write_mean']:>9.1f}"
                  if np.isfinite(s["v_write_mean"]) else f"{'--':>9}")
            print(f"  {rule:<18}{c:>5.3g}{s['t']:>10.2f}{sp}{vw}"
                  f"{100 * s['success']:>8.0f}%"
                  f"{100 * s['coverage']:>9.1f}%{100 * s['precision']:>10.1f}%"
                  f"{100 * s['in_band']:>8.1f}%{s['peak_force']:>9.2f}{100 * s['torn']:>6.0f}%")
    # The honest speed statistic: per demonstration, the fastest rate that still
    # completes, then the median over demonstrations.  "every episode passes" is
    # a min over 12 draws and reads whichever demonstration happened to be the
    # most fragile; the manuscript asks for each method's fastest ACCEPTED
    # COMPLETED execution, which is this.
    demos = sorted({x["ep"] for x in rows})
    print(f"\n  Per demonstration, the fastest rate that still writes:")
    print(f"  {'rule':<20}{'median':>8}{'mean':>8}{'min':>7}{'max':>7}{'time at the median (s)':>24}")
    per = {}
    order = []
    for rule in tags:
        b = []
        for dm in demos:
            ok = [x["c_eff"] for x in rows if x["tag"] == rule and x["ep"] == dm and x["success"]]
            b.append(max(ok) if ok else 0.0)
        b = np.asarray(b, float)
        per[rule] = dict(median=float(np.median(b)), mean=float(b.mean()),
                         lo=float(b.min()), hi=float(b.max()), per_demo=b.tolist())
        order.append((float(np.median(b)), rule))
    for _, rule in sorted(order, reverse=True):
        q = per[rule]
        print(f"  {rule:<20}{q['median']:>7.2f}x{q['mean']:>7.2f}x{q['lo']:>6.2f}x{q['hi']:>6.2f}x"
              f"{t_1x / max(q['median'], 1e-9):>24.2f}")
    print(f"\n  Tearing, the failure that cannot be retried, by requested rate:")
    print(f"  {'rule':<20}" + "".join(f"{c:>8.3g}x" for c in cs))
    for rule in tags:
        line = f"  {rule:<20}"
        for c in cs:
            g = sel(rule, c)
            line += f"{100 * np.mean([x['torn'] for x in g]) if g else np.nan:>7.0f}%"
        print(line)

    print(f"\n  The fastest execution each rule COMPLETES -- all episodes passing:")
    print(f"  {'rule':<18}{'c':>6}{'time (s)':>11}{'faster than the demo':>23}{'v_write (mm/s)':>17}")
    best = {}
    for rule in tags:
        ok = [c for c in cs if sel(rule, c) and all(x["success"] for x in sel(rule, c))]
        if not ok:
            print(f"  {rule:<18}{'--':>6}{'--':>11}{'nothing passes':>23}{'--':>17}")
            continue
        c = max(ok)
        s = summary[f"{rule}@{c:g}"]
        best[rule] = dict(c=c, **{k: s[k] for k in ("t", "speedup", "v_write_mean")})
        print(f"  {rule:<18}{c:>6.3g}{s['t']:>11.2f}{s['speedup']:>22.2f}x"
              + (f"{1000 * s['v_write_mean']:>17.1f}"
                 if np.isfinite(s["v_write_mean"]) else f"{'--':>17}"))
    base = [x for x in rows if x["c"] == min(cs)]
    f1 = float(np.mean([x["peak_force"] for x in base])) if base else np.nan
    print(f"\n  The force law, against the 1x peak of {f1:.2f} N:")
    print(f"  {'rule':<18}" + "".join(f"{c:>9.3g}x" for c in cs))
    for rule in tags:
        line = f"  {rule:<18}"
        for c in cs:
            g = sel(rule, c)
            line += f"{(np.mean([x['peak_force'] for x in g]) / f1 if g else np.nan):>10.2f}"
        print(line)
    print(f"  {'c^2':<18}" + "".join(f"{c * c:>10.2f}" for c in cs))
    (out / f"contact_{tag}.json").write_text(json.dumps(
        dict(summary=summary, fastest_passing=best, per_demo_fastest=per,
             demo_time_s=t_1x, delta=Delta,
             episodes=[{k: v for k, v in r.items() if k not in ("trace",)} for r in rows]),
        indent=1))
    plot(rows, out / f"contact_{tag}.png")


def plot(rows, png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = sorted({r["c"] for r in rows})
    tags, seen = [], set()
    for x in rows:
        if x["tag"] not in seen:
            seen.add(x["tag"]); tags.append(x["tag"])
    pal = ["#8a8880", "#eb6834", "#1baf7a", "#2a78d6", "#d6b32a", "#8a5ad6",
           "#d62a2a", "#2ad6c0", "#9c6b3f"]
    col = {t: pal[i % len(pal)] for i, t in enumerate(tags)}
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 3.9))
    for rule in tags:
        m = lambda k: [float(np.nanmean([x[k] for x in rows
                                         if x["tag"] == rule and x["c"] == c])) for c in cs]
        st = dict(color=col[rule], marker="o", ms=4, label=rule)
        ax[0].plot(cs, m("peak_force"), **st)
        ax[0].set_yscale("log")
        ax[1].plot(cs, [100 * v for v in m("in_band")], **st)
        ax[2].plot(cs, [100 * v for v in m("success")], **st)
        ok = [c for c in cs if all(x["success"] for x in rows
                                   if x["tag"] == rule and x["c"] == c)]
        t = m("t")
        ax[3].plot(cs, t, **st)
        if ok:
            i = cs.index(max(ok))
            ax[3].plot([cs[i]], [t[i]], marker="*", ms=15, color=col[rule], ls="none")
    f1 = float(np.mean([x["peak_force"] for x in rows if x["c"] == min(cs)]))
    ax[0].plot(cs, [f1 * c * c for c in cs], color="#2a78d6", ls=":", label="$c^2$ from 1x")
    ax[0].axhspan(1, 6, color="#1baf7a", alpha=0.08)
    ax[0].axhline(12, color="#d62a2a", lw=1.0, ls="--")
    ax[0].annotate("tear", (cs[0], 12), fontsize=8, color="#d62a2a", va="bottom")
    ax[0].set_ylabel("peak contact force (N)")
    ax[1].axhline(85, color="#8a8880", lw=0.8, ls=":")
    ax[1].set_ylabel("time in the force band (%)")
    ax[2].set_ylabel("task success (%)")
    ax[3].set_ylabel("completion time (s)")
    ax[3].annotate("star = fastest that still writes", (0.5, 0.92), xycoords="axes fraction",
                   ha="center", fontsize=8, color="#555")
    for a in ax:
        a.set_xlabel("execution-rate factor c")
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    fig.suptitle("Retiming an achieved demonstration in contact", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
