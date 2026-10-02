# The retiming identities, verified in this simulator

`speedup.py` tests Theorem 1, Proposition 2, Corollary 3, Theorem 4 and
Corollary 5 of *Accelerated Demonstration Reproduction on SE(3)* against the
Case 1 geometric impedance controller this repository runs. `speedup_contact.py`
(S3, below) takes it into contact.

```
python3 speedup.py --audit                        # Assumption 1, checked
python3 speedup.py --c 2 --ablate Kr null bounds tank tau
python3 speedup.py --c 2 --sweep                  # Proposition 2, numerically
python3 speedup.py --c 2 --fixed-rate             # the sampling defect instead
python3 speedup.py --c 2 --keep-link-damping      # price the unmatched drag
python3 speedup.py --clock 1.4 --clock-n 3        # Theorem 4 and Eq. (15)
python3 speedup.py --corollary5 --clock 1.4 --clock-n 3
```

**Plant version.** Every number here was measured after `controller.py`
promoted the rotational stiffness to a controller state (`ctl.kr`, its own rate
slot `Ur`, bounds `kr_lo`/`kr_hi`, and `Dr = 2 zeta sqrt(kr) Lambda_r^1/2` from
the rotational operational-space inertia instead of a declared
`wrist_inertia`). That change moved the absolute tracking errors and the usable
physics rate; it did not touch any of the identities.

## What the theorem asks for, and what this controller already gives

Theorem 1: run the demonstration clock `c` times faster and set

```
K_p, K_R -> c^2 K     D, D_q -> c D     U_N -> c^2 U_N     g_d,c(t) = g_d,0(ct)
```

and the closed loop reproduces the baseline orbit exactly, `q_c(t) = q_0(ct)`,
`v_c(t) = c v_0(ct)`.

Two of those four lines are **free** in `controller.py`, because both damping
designs are built on a square root of the stiffness and the inertias are
unchanged at a matched configuration:

```
D  = zeta (Lambda^1/2 K^1/2 + K^1/2 Lambda^1/2)   ->  K  -> c^2 K  gives  D  -> c D
Dr = 2 zeta sqrt(kr) Lambda_r^1/2                 ->  kr -> c^2 kr gives  Dr -> c Dr
```

The test reads `D` back from the log rather than assuming it, and finds
`D_c = c D_0` to a relative `0.000e+00`. This survived the rotational redesign:
the exponent is carried by the square root, whatever inertia sits beside it.

The coefficients a retimer must carry itself, because they are **not** in the
policy's action `(x_d, k_diag)`: `ctl.kr`, `null_kp`/`null_kd`, the bounds
`k_lo`/`k_hi` and `kr_lo`/`kr_hi`, the tank `E0`/`Ec`, and `tau_limit`.

## Assumption 1, checked rather than assumed (`--audit`)

| hypothesis | measurement | verdict |
|---|---|---|
| exact gravity compensation | zero-torque fall over 0.8 s: `0.000e+00 rad` | **granted**. ManiSkill disables link gravity for a fixed-base arm, so `g_q = 0` and Corollary 3's torque law is the clean `b T_0` |
| no unmatched dissipation | coasting kinetic energy loses **11.8 %/s** | **broken by SAPIEN's default link damping of 0.05**. That is a `b qdot` torque: it scales as `c`, not `c^2`. Zeroed, the loss falls to 2.6 %/s, which is the integrator's own and not a force in the model |
| the discrete loop can be conjugate | PhysX alone, no controller, `(dt, v0)` vs `(dt/c, c v0)`: `max abs(q_c(t) - q_0(ct)) = 0.000e+00 rad` | **granted, bit for bit**. The integrator's numerical dissipation is itself conjugate, so it cannot break the identity |

`disable_joint_drives()` is doing its job: after it, every joint reads
`stiffness = damping = friction = armature = 0`. The URDF's `damping="0.003"` is
not the culprit; SAPIEN's per-link drag is.

Theorem 1 is a **continuous-time** statement, so the implemented loop is
conjugate step for step only if the faster execution also steps `c` times
faster. `speedup.py` scales `sim_freq` by `c` by default (this is why
`WritingSim` grew a `sim_freq` argument) and `--fixed-rate` keeps 500 Hz to
measure what is left.

## S1 -- Theorem 1 (the Table I / Fig. 1 analogue)

6 s SE(3) free-motion reference: 7 cm travel along the paper's `u` axis with
lateral and vertical excursions, noncommuting roll/pitch/yaw at three
frequencies, and a stiffness schedule that rises and returns. Contact force is
asserted to stay at `0.000e+00 N` throughout.

| execution | peak position | peak orientation |
|---|---|---|
| 1x, baseline gains | 1.1166 mm | 0.9614 deg |
| 2x, fixed gains | 4.2850 mm | 1.9467 deg |
| 2x, `c^2 K`, `c D` | **1.1166 mm** | **0.9614 deg** |
| 4x, fixed gains | 12.6222 mm | 4.2286 deg |
| 4x, `c^2 K`, `c D` | **1.1166 mm** | **0.9614 deg** |

| c | `max abs(dq)` | `max abs(dv/c)` | tip | `K_c = c^2 K_0` | `D_c = c D_0` | `tau_c = c^2 tau_0` |
|---|---|---|---|---|---|---|
| 2 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| 4 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |

The one column that is not identically zero is the geodesic rotation
discrepancy, `1.032e-06 rad`, and that is the float32 pose the physics returns,
not a disagreement: the joint angles behind it are bit-identical. (Measured
through the chordal norm. `arccos(tr/2 - 1/2)` has an infinite slope at identity
and reports `8e-04 rad` for the same bit-identical rotations, which would have
hidden the result this file exists to show.)

Corollary 3, measured: speed `2.00000` / `4.00000`, torque `4.00000` /
`16.00000`, power `8.00000` / `64.00000`, stored energy `4.00000` / `16.00000`
-- that is `c`, `c^2`, `c^3`, `c^2` exactly.

## Proposition 2, numerically (`--sweep`)

The restricted converse says that within this controller class -- no
acceleration feedforward -- the exponents `(2, 1)` are also *necessary* for the
wrench operator to be preserved. The implemented controller can only sit on the
curve `(p, p/2)`, since both damping designs take a square root of the
stiffness, so one sweep over `p` decides it:

| p | D exponent | `max abs(dq)` (rad) | tip (m) |
|---|---|---|---|
| 1.0 | 0.50 | `1.134e-02` | `1.189e-03` |
| 1.5 | 0.75 | `5.125e-03` | `4.927e-04` |
| 1.75 | 0.875 | `2.432e-03` | `2.253e-04` |
| 1.9 | 0.95 | `9.441e-04` | `9.071e-05` |
| **2.0** | **1.0** | **`0.000e+00`** | **`0.000e+00`** |
| 2.1 | 1.05 | `9.012e-04` | `8.027e-05` |
| 2.25 | 1.125 | `2.174e-03` | `1.887e-04` |
| 2.5 | 1.25 | `4.148e-03` | `3.459e-04` |
| 3.0 | 1.5 | `7.496e-03` | `5.817e-04` |

A single exact zero with no flat bottom: `p = 1.9` and `p = 2.1` already miss by
`9e-05 m`. This is the strongest defensible version of "faster needs stiffer" --
necessity *for operator covariance in this class* -- and says nothing about a
controller with acceleration feedforward, which can track an accelerated path
without raising feedback stiffness at all.

## What each coefficient is worth (`--ablate`, c = 2, tip-position defect)

| left unscaled | defect | what it does |
|---|---|---|
| nothing (Theorem 1) | `0.000e+00` | -- |
| `bounds` | `3.21e-04 m` | `k_hi = 4000` clips the `c^2`-scaled schedule, which peaks at 9600 |
| `Kr` | `3.13e-04 m` | orientation error 0.961 -> 1.913 deg |
| `null` | `1.15e-05 m` | the posture potential `U_N` |
| `tank` | `0.000e+00` | **not binding on this reference**, see below |
| `tau` | `0.000e+00` | **not binding**: saturation is `0.000e+00 N m` |
| *everything* (fixed gains) | `3.49e-03 m` | the comparison, ~10x larger |

A coefficient left unscaled costs nothing *until it binds*, which is why the
tank and the torque limit are reported as margins:

| execution | `alpha_min` | tank left (J) | max saturation |
|---|---|---|---|
| 1x, baseline | 1.000000 | 19.434 | `0.000e+00 N m` |
| 2x, Theorem 1 | 1.000000 | 77.736 | `0.000e+00 N m` |
| 2x, Theorem 1 without `tank` | 1.000000 | **17.736** | `0.000e+00 N m` |

The unscaled tank holds a quarter of the energy it should and still never gates
here. The baseline drains 0.5660 J and the drain scales as `c^2`, so a 20 J tank
with a 1 J knee first closes at **`c = 5.79`** on this reference -- after which
`alpha < 1` rescales both `V_d` and `Kdot`, and the symmetry is gone without
anything in the controller reporting an error.

## The two defects that remain in the implementation

| source | tip defect at c = 2 |
|---|---|
| exact (matched rate, drag removed) | `0.000e+00 m` |
| SAPIEN link drag kept (`b qdot`, scales as `c`) | `1.77e-05 m` |
| fixed 500 Hz instead of 1000 Hz (`beta_imp`) | `3.57e-05 m` |
| wrong gains (the thing being tested against) | `3.49e-03 m` |

## Corollary 3's admissible uniform-scaling set, Eq. (11)

```
velocity   b <= 35.817
torque     b <= 296.883
stiffness  b <=  1.667        <- binding
-> b_max 1.667, c_max 1.291
```

The same test on the recorded demonstrations (`approach_kn ~ 2936 N/m` against
`k_hi = 4000`) gives `c_max ~ 1.17`. **Uniform Theorem 1 scaling is nearly
useless for this task on its own** -- not because the identity fails, but
because the stiffness ceiling is reached first.

---

# S2 -- a general execution clock: the defect and its correction

Under a general clock `ds/dt = r(s)`, Theorem 4 splits the departure from the
baseline orbit into named pieces,

```
R = (G_0 - r^-2 G_k) + (B_0 - r^-1 B_k) p   +   J^T[f_0 - r^-2 f_th]   -   (r'/r) M p   +   (u + dtau)/r^2
    `------------- gain schedule ---------'       `---- contact ----'       `- clock -'
```

and in free motion with the gains scheduled by the similarity law **pointwise in
phase**, `K = r(s)^2 K_0`, only the clock term is left. Eq. (15)'s
`u_clock = (rdot/r) M v = r'(s) M(q) qdot` is the active torque that cancels it.

This needed one slot in the controller: `Case1Proposal.u_tau`, a joint-space
active torque. It is joint-space because the theorem's `M` is the joint-space
inertia, not the operational-space `Lambda`. It is **active, not passive** --
while the clock accelerates it supplies power -- so it goes through Case 1's
gate, its power `qdot . u` is metered into `p_prop`, and it is clipped at the
torque limit with everything else. Nothing in the writing task uses it.

## The floor, and why this test is built around it

The constant-clock test could compare bit for bit, because a `c`-times faster
execution at `c` times the physics rate takes the *same number* of float32 steps
with exactly scaled values. A nonuniform clock cannot: its phase grid is
nonuniform by construction, so the two runs take different numbers of steps and
their roundoff no longer cancels. Measured directly, two runs of the **identical
system** (`r = 1` in both, only the step size differing) disagree by 0.045 mm
(500/1000 Hz), 0.132 mm (1000/2000) and 0.262 mm (2000/4000). It *grows* with
the rate, so it is float32 accumulation, not truncation.

So the study carries its own floor control: a **uniform** clock at the same mean
rate, on the same physics rate. Theorem 1 says its true defect is exactly zero,
so whatever it reports is the floor, and nothing below it means anything.

Two consequences, both the opposite of the obvious:

* **The sharpest test is the lowest physics rate that still resolves the loop**
  -- and the *scaled* loop, not the baseline one. With the rotational redesign
  the rotational stiffness peaks at `r_peak^2 Kr = 461 N m/rad`, and 500 Hz no
  longer resolves it: the nonuniform runs there report millimetres and the
  correction makes them worse. That is a sampled-controller artefact, not a
  defect of the model, and 1000 Hz is the sweet spot.
* **Raise the signal, not the resolution.** `--clock-n` puts `n` half-periods of
  the clock in the same horizon, multiplying `r'` by `n` while leaving the rate
  range, the duration *and* the floor alone. This is the manuscript's own point
  -- the condition bounds `|d log r/ds| = |a|/r^2`, not the speed factor.

## Result (6 s demonstration, `r` in [1, 2.4], 3.873 s execution, 1.549x overall)

| clock | `sup abs(d log r/ds)` | Hz | floor | scaled gains only | **+ `u_clock`** | signal / floor |
|---|---|---|---|---|---|---|
| `1 + 1.4 sin^2(pi s/6)` (the manuscript's) | 0.4732 | 1000 | 0.0470 mm | 0.2681 mm | **0.0450 mm** | 5.7 |
| | | 2000 | 0.1281 mm | 0.2850 mm | 0.1897 mm | 2.2 |
| `1 + 1.4 sin^2(3pi s/6)` | 1.4195 | 1000 | 0.0470 mm | **0.8615 mm** | **0.0520 mm** | 18.3 |
| | | 2000 | 0.1281 mm | 0.8664 mm | 0.1770 mm | 6.8 |

**The correction takes the defect to the floor.** Two checks that it is the
clock term and not something else:

* The uncorrected defect is **independent of the physics rate** (0.2681 mm at
  1000 Hz against 0.2850 at 2000; 0.8615 against 0.8664), so it is a term of the
  continuous model. The corrected residual instead tracks the floor.
* Tripling `r'` at a **fixed rate range and fixed duration** multiplies the
  uncorrected defect by 3.21 against the 3.00 in `sup abs(d log r/ds)`. The
  defect is driven by how fast the clock changes, not by how fast it runs.

`K = r(s)^2 K_0` is verified from the log at a relative `1.3e-15`.

## What the correction costs

It is not free and Case 1 does not pretend it is. At `n = 3` the tank drains
1.599 J against 0.566 J on the baseline, and the peak correction torque is
0.460 N m -- a real actuator limit would have to hold it, and no passivity
argument supplies it. In contact this is exactly what one does not want to do,
which is what the next result is for.

## Corollary 5: scheduled gains *without* the active correction

Instantiated on a scalar GIC, as the manuscript does and for the same reason:
the corollary needs a contraction rate for the baseline in a declared metric,
and contraction is an extra hypothesis, not something GIC passivity hands you on
a 7-DOF arm.

```
m 1 kg, k 400 N/m, zeta 1.4 -> d 56 N s/m,  r = 1 + 1.4 sin^2(3 pi s / 6)
eigenvector metric of the baseline A:  lam = 8.4041 1/s,  L_Pi = 1.7385,  V_P = 0.04612
hbar = sup |d log r/ds| = 1.4195 1/s
lam* = lam - hbar L_Pi = 5.9363 > 0   -> the bound holds

peak metric error   1.757e-03
comparison bound    1.103e-02        conservative by 6.3x
```

`lam*` is what decides feasibility, and `hbar` is what enters it -- so the
retimer's knob in contact is the *smoothness* of the clock, not its speed.

## Four traps this study walked into, all invisible under a constant clock

1. **`k_lo` scaled by the peak rate.** The applied-stiffness bounds must
   *bracket* the whole schedule, so the floor scales with the clock's minimum
   and the ceiling with its maximum. Scaling the floor by the peak makes the
   applied stiffness `max(r^2 K_0, k_lo)`, which is not the similarity law. A
   constant clock hides this completely. This was the whole of a 0.13 mm
   residual that survived the correction.
2. **A partial final phase step.** `ds = r(s) dt` is the identity the scheme
   rests on; clamping the last step to land exactly on `S` breaks it for one
   step and leaves an endpoint spike that dominates a max-over-phase.
3. **`R_d` read one sample ahead of `x_d`.** The rotational equilibrium then
   advances by `dt` while the clock advances everything else by `r dt`.
4. **A positional argument that moved.** `Case1Proposal` gained `Ur` ahead of
   `u_tau` in the concurrent rotational redesign, so the correction torque was
   silently being passed as a rotational stiffness rate. Keyword arguments now.
   The same redesign moved the rotational stiffness from `gains.Kr` to the
   controller state `ctl.kr`, which is where the schedule has to be written.

`--sweep` also found the rotation metric itself: `arccos(tr/2 - 1/2)` reports
`8e-04 rad` for bit-identical float32 rotations. `rot_angle` uses the chordal
norm instead.

## Files

```
speedup.py
logs/speedup/audit.json
logs/speedup/theorem1_c*.{png,json}
logs/speedup/proposition2_c*.{png,json}
logs/speedup/theorem4_a*.{png,json}
logs/speedup/corollary5.{png,json}
logs/speedup/S1_battery.txt, S2_battery.txt
```

`sim.py` gained one argument (`sim_freq`) and `controller.py` one optional
proposal field (`u_tau`); nothing else in the simulator changed on our account.

---

# S3 -- contact: where the symmetry stops, and how fast the task actually goes

`speedup_contact.py`. Free motion is settled; contact is where Theorem 1 stops
being a symmetry, because the environment is not retimed with the robot.
Theorem 4's `R_C` is the one term no gain schedule can cancel.

```
python3 speedup_contact.py --probe
python3 speedup_contact.py --affine
python3 speedup_contact.py --retime --episodes 4 --c 1 1.5 2 3 4 5 6.25 8 \
        --rules fixed uniform tangential --clocks uniform normal-capped --workers 12
```

## The paper, measured (`--probe`)

| commanded depth | `f_n` | `k_n d` | penetration |
|---|---|---|---|
| 0.50 mm | 0.496 N | 0.500 N | 0.028 um |
| 2.00 mm | 1.990 N | 2.000 N | 0.057 um |
| 5.00 mm | 4.976 N | 5.000 N | 0.029 um |

`k_e ~ 2.7e7 N/m` against a controller ceiling of 4000: the wall is **rigid**,
and `f_n = k_n x (depth of x_d below the surface)` to 0.5%.

The landing is a different law. Descending onto the paper at 5 to 80 mm/s, the
peak sensor force scales as `v^0.99` -- Section IV-A's `v_0 sqrt(m k_e)`, exactly
linear -- and the 5 Hz pressure signal the tear criterion is judged on scales as
`v^0.59`. **Two exponents on one axis**: the writing force is quadratic in the
clock, the landing is not.

## Section VII, on a rigid wall (`--affine`)

With `k_e >> k` the specialization collapses to algebra: `sigma_x = Delta` and
`sigma_f = k Delta`.

| `k_n` | depth for 3 N | `sigma_f` | inside the 2.5 N force radius? |
|---|---|---|---|
| 500 | 6.0 mm | 2.0 N | yes |
| **625** | 4.8 mm | **2.5 N** | the cap |
| 1000 (the demonstrations) | 3.0 mm | 4.0 N | **no** |
| 4000 | 0.75 mm | 16.0 N | no |

Proposition 8's interval is **empty**: the 3 mm position radius asks for
`k >= 9.1e6 N/m` and the force radius for `k <= 625 N/m`. The position bound
diverges because `sigma_x = Delta` whatever the gain -- a blind controller cannot
find a rigid surface, and no stiffness helps. What survives is a cap **with no
speed in it**,

```
k_n <= eps_f / Delta = 625 N/m.
```

The demonstrations write at `k_n = 1000 N/m`, over the cap, and are in band only
because the operator **felt** for the paper. That is the observer escape the
manuscript names immediately after Proposition 8, and it is the reason a
force-conditioned policy is not bound by this number.

**Implementation check.** Section IX-D reproduced on its own compliant surface:

| | here | published |
|---|---|---|
| early position radius | 0.1667 mm | 0.1667 mm |
| final force radius | 0.1908 N | 0.1908 N |
| force range | [6.667, 13.333] N | [6.667, 13.333] N |
| peak effort | 13.3333 N | 13.333 N |
| peak gain rate | 241679.18 N/(m s) | 241679 N/(m s) |
| affine vs independent ODE | `4.40e-15` | `1.36e-12` |

## Retiming an achieved demonstration (`--retime`)

12 demonstrations, 8 rates, 3 gain rules, 2 clocks, 576 episodes, scored by the
simulator's own ink, force-band and tear criteria. **The demonstration takes
11.50 s and writes at 15.9 mm/s.**

Per demonstration, the fastest rate that still writes (median over the 12):

| rule | median | mean | range | time at the median |
|---|---|---|---|---|
| **fixed** (`K` unchanged) | **5.00x** | 4.79x | 2.94–6.25x | **2.30 s, 75.8 mm/s** |
| tangential (`k_t -> c^2 k_t`) | 4.17x | 3.84x | 1.00–5.00x | 2.76 s |
| fixed + landing-capped clock | 4.05x | 4.30x | 2.63–6.10x | 2.84 s |
| tangential + capped clock | 3.50x | 3.18x | 1.00–4.26x | 3.28 s |
| **uniform (`K -> c^2 K`, Theorem 1)** | **1.00x** | 1.17x | 1.00–1.52x | 11.50 s |

**Theorem 1's prescription is the worst rule tested.** Doing nothing to the
gains is the best. The force law says why, against the 1x peak of 3.64 N:

| rule | 1.5x | 2x | 3x | 4x | 6.25x | 8x |
|---|---|---|---|---|---|---|
| uniform | 2.20 | 3.85 | 8.52 | 40.26 | 98.74 | 105.61 |
| **`c^2`** | **2.25** | **4.00** | **9.00** | 16.00 | 39.06 | 64.00 |
| fixed | 1.13 | 1.41 | 1.85 | 2.82 | 3.83 | 4.86 |
| fixed + capped | 1.13 | 1.37 | 2.01 | 2.13 | 2.26 | 2.60 |

`uniform` tracks `c^2` to within 5% through `c = 3` and then runs past it as the
contact turns impact-dominated (359 N at 6.25x). This is exactly the predicted
mechanism: on a rigid wall the orbit is pinned by the surface, so scaling `K`
scales the contact force directly and buys nothing in tracking, because the
surface was already doing the tracking. `fixed` instead grows as `c^0.75`, which
is the filtered landing law from `--probe`.

**What actually limits the task is the landing, not the writing force.**
Coverage and precision stay above 96% out to 5x with the gains unchanged; what
fails is tearing.

## The landing-capped clock: a safety result, not a speed result

Section VII says the constraint sits on one component of one axis during one
phase, which is what a nonuniform `r(s)` is for. `--clocks normal-capped` sets
`r(s) = min(c, v_cap / v_n(s))` so the pen never approaches the paper faster
than the demonstration did, and runs at `c` everywhere else.

| tear rate | 1.5x | 2x | 3x | 4x | 5x | 6.25x | 8x |
|---|---|---|---|---|---|---|---|
| fixed | 0% | 0% | 0% | 17% | 17% | 92% | 100% |
| **fixed + capped** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** | **0%** |
| uniform | 0% | 67% | 100% | 100% | 100% | 100% | 100% |

It removes tearing **entirely, at every rate tested**, and its peak force
saturates near 9 N. But it does **not** make the task faster: the median fastest
passing rate falls from 5.00x to 4.05x, because the time spent slowing the
descent is not recovered and the binding failure simply moves from tearing to
coverage. The honest reading is that the clock changes the *character* of
failure -- a missed glyph can be retried, a torn page cannot -- and that is worth
having, but it is not a speed lever here.

The directional rule is a second honest negative: `tangential` does not beat
`fixed` either, because the tangential axes were never binding. The manuscript
retains a negative control of the same kind.

## What S3 establishes

1. The free-motion symmetry is exact (S1) and **actively harmful** in contact:
   its gain prescription multiplies the contact force by `c^2`, confirmed to
   within 5%, and tears the paper from `c = 2`.
2. The achieved speed-up of this task under retiming is a median **5.0x**
   (11.50 s to 2.30 s, 15.9 to 75.8 mm/s) with the gains left alone.
3. The binding constraint is the landing impulse, which scales as `c^0.75` after
   the pressure filter, not the writing force.
4. Capping the approach speed with a nonuniform clock eliminates tearing at
   every rate but does not raise the passing rate.
5. `k_n <= eps_f / Delta = 625 N/m` has no speed in it at all. The only way past
   it is to sense the surface, which is what the policy in `policy/` does and
   what S4 is for.

## Files

```
speedup_contact.py
logs/speedup/contact_probe.json
logs/speedup/contact_affine.json
logs/speedup/contact_retime.{json,png}
logs/speedup/S3_retime.txt
```

---

# S4 -- the trained policy, retimed

`speedup_contact.py --policy`. S3 retimed a *recorded* demonstration, which is
an open-loop plan. This retimes the **closed-loop policy trained on those
demonstrations** (`act_ft_input`, Comp-ACT unchanged, the best structure in
`policy/runs/COMPARISON.md` at 100% success): the chunk it generates is consumed
`c` times faster, and its own stiffness goes through the same gain rules.

```
python3 speedup_contact.py --policy policy/runs/act_ft_input/seed0/model.pkl \
        --attempts 500 501 502 --c 1 1.25 1.5 1.75 2 2.5 3 \
        --rules fixed uniform tangential --workers 10
python3 speedup_contact.py --policy ... --video --video-c 2 --video-text S
```

The time budget is `1.3 x` the longest demonstration **divided by c**, so
finishing `c` times faster has to mean finishing, not being handed the same wall
clock and writing more slowly.

## Result (3 glyphs x 3 unseen papers, 189 episodes)

Per paper, the fastest rate the policy still writes at:

| rule | median | mean | range |
|---|---|---|---|
| **fixed** (`K` unchanged) | **1.72x** | 1.65x | 1.25–2.00x |
| uniform (`K -> c^2 K`, Theorem 1) | **1.00x** | 1.00x | all 1.00 |
| tangential (`k_t -> c^2 k_t`) | **1.00x** | 1.00x | all 1.00 |

| success | 1x | 1.25x | 1.52x | 1.72x | 2x | 2.5x | 2.94x |
|---|---|---|---|---|---|---|---|
| fixed | 100% | 100% | 89% | 67% | 11% | 0% | 0% |
| uniform | 100% | **0%** | 0% | 0% | 0% | 0% | 0% |
| tangential | 100% | **0%** | 0% | 0% | 0% | 0% | 0% |

Completion time of the successful `fixed` runs: 10.79 s at 1x, 7.01 s at 1.25x
(9/9 papers), 5.46 s at 1.52x (8/9), 4.77 s at 1.72x (6/9).

## The two findings

**1. The policy retimes far worse than its own demonstrations.** Open loop, the
recorded plan survives a median **5.0x** (S3). Closed loop, the policy trained
on exactly those demonstrations manages **1.72x**. The plant is not the limit
here; the policy's distribution is. At 2x it still presses correctly -- the
force stays in band -- and then **lifts off early**, writing half the glyph:
the failures are `coverage` and `precision`, not force.

**2. Any gain rescaling kills it instantly, and not by tearing.** Both scaling
rules drop from 100% to **0% at c = 1.25**, and their mean peak contact force at
1.25x through 2x is **0.00 N**: the pen never reaches the paper at all. The
trace shows why -- the pen descends to 19 mm, then climbs back to 34 mm, while
the stiffness the policy commands stays at its travel value instead of dropping
for contact.

The mechanism is specific and worth stating plainly: `data.py`'s state vector
contains the policy's **own log stiffness** and the **spring deflection**
`tcp - x_d`. Multiplying `K` by `c^2` puts both off the distribution it was
trained on -- the deflection for a given force shrinks by `c^2` -- so the policy
never recognises contact and never commits to the descent. Theorem 1's
prescription does not merely hurt the contact physics (S3); in closed loop it
**corrupts the policy's observation of that physics**, and it does so at a rate
factor where the physics would still have been fine.

## The comparison video

`logs/speedup/speedup_S_2x.mp4` -- the same paper and target written three ways,
on a **shared physical time axis**. Each panel freezes when its own episode
ends, so the gap at the end is the speed-up; underneath, all three contact
forces on one axis with the 1-6 N band and the 12 N tear line.

| panel | outcome | last ink | peak |
|---|---|---|---|
| 1x, gains unchanged | SUCCESS | 10.60 s | 2.9 N |
| 2x, gains unchanged | fail: coverage, precision | 4.10 s | 6.5 N |
| 2x, `c^2 K` (Theorem 1) | fail: coverage, precision, force | -- | **0.0 N** |

## Files

```
speedup_contact.py --policy
logs/speedup/S4_policy.txt
logs/speedup/speedup_S_2x.mp4
```
