# cascade — inner Cartesian impedance + outer admittance

A dependency-light simulator for the cascade under study, and the label study
built on it. Everything here runs on numpy alone; the SAPIEN/FR3 version lives
in `../haptic_teleop_fr3_bilateral.py` and is validated against this package.

## Why a second simulator

The FR3 script runs the cascade through SAPIEN contact, IK and a redundant arm,
so a wrong number there could come from the controller, the IK or the contact
model and there is no way to tell which. Here the same controller chain runs
against a scalar spring whose steady state is known in closed form, so the
controller is proven correct first and ported second. That is how the sign bug
in the FR3 admittance was found.

## Files

| file | what it is |
|---|---|
| `analytic.py` | closed-form steady states — the answer the integrator must match |
| `case1.py` | **Case 1**: direct GIC on a torque interface + the energy tank (Theorem 1) |
| `case2.py` | **Case 2**: geometric admittance driving an opaque motion servo (Prop. 1–2) |
| `tests_cases.py` | 44 checks for the two cases: `python3 tests_cases.py` |
| `case_spiral.py` | the spiral wipe driven by all three executors, one operator |
| `study_case_spiral.py` | the three-way comparison → `case_spiral.png` |
| `core.py` | the simulator: human → master → channel → coupling → **outer admittance → inner impedance** → slave → environment |
| `labels.py` | seven label rules, each declaring the quantity the analysis predicts it returns |
| `tests.py` | 87 checks: `python3 tests.py` |
| `study_labels.py` | the Kh sweep and the prior-sensitivity sweep → `label_study.png`, `label_study.csv` |
| `study_layers.py` | what each cascade layer does, and at what frequency → `layer_roles.png` |
| `wipe.py` | the tilted-surface wiping task: 3-D, Coulomb friction, unknown surface orientation |
| `study_anisotropy.py` | T1, the three-factor label study → `anisotropy.png`, `anisotropy_raw.npz` |
| `study_transplant.py` | which frame each published rule actually picks → `transplant.png` |
| `../fr3_wipe_scene.py` | the wipeable FR3 scene: tilted plate, welded pad → `fr3_wipe_frames.png` |
| `../peg_insertion_cascade.py` | PegInsertionSide under the cascade, pre-grasped |
| `../record_demo.py` | video of both demonstrations |

## Sign convention

Positive = **into the wall**. `f_e >= 0` is the environment reaction, so every
equation reads `M xdd = (drive) - f_e`. The FR3 script uses world vectors where
`f_e` already points opposite to the penetration, so there it is **added** —
the two agree along the penetration axis.

## Results this package establishes

**The inner stiffness is a series compliance.** At rest the whole chain is
springs in series:

    f_e = [f_d + Ko (x_ref - x_wall)] / (1 + Ko/ke + Ko/Ki)

`Ki` enters exactly like the environment stiffness, a term absent from any
single-layer analysis. At `Ki=2000, Ko=1000, ke=5000` a commanded 10 N arrives
as 5.88 N, because `Ko/Ki = 0.50` exceeds the wall's own `Ko/ke = 0.20`.
**`Ki >> Ko` is a requirement, not a preference.**

**The (x_ref, f_d, Ko) action is over-parameterised, so the label rule must fix a
gauge.** Labelling `f_d := f_ch` *and* `x_ref := x_c` double-counts the coupling
force: replay gives 1.93x the demonstrated force. Fixing `x_ref := x_r`
reproduces it exactly. `Action.spring_share()` is the deploy-time diagnostic.

**Stiffness labels do not identify the operator.** Sweeping the true `Kh` 16x
with the controller gains fixed, the log-log sensitivity `d log(K_hat)/d log(Kh)` is:

| rule | returns | sensitivity |
|---|---|---|
| R3 Imp-ACT controller rule | `Ki` | −0.01 |
| R4 Compliance-for-Free | `series(Ka,Ki)` | −0.00 |
| R4b same, vs `x_r` | `Ka` | −0.00 |
| R5 particle filter | its prior | −0.22 |
| **R6 probe identification** | **`Kh`** | **1.00** |
| R7 naive regression | `Kh` only if intent is constant | n/a (goes negative) |
| R1 ACP heuristic | designer constants | 0.18 (spurious) |
| R2 Comp-ACT toggle | designer constants | 0.00 |

R1's 0.18 is the trap: a stiffer operator pushes harder, so a force-keyed
heuristic drifts upward and looks like partial success without estimating
anything.

**Identifiability is frequency separation, not just excitation.** With a
*constant* operator intent, a regression carrying one intercept recovers `Kh`
exactly with no probe — the entire unobservable intent is one unknown number.
Once the intent varies (0.2 Hz here) that rule collapses to −82 N/m while the
probe-based rule still returns 400.0 N/m to 1e-13. Probe off, excitation is
5e-18 m and R6 refuses to answer rather than returning a plausible number.

**The particle filter's answer is set by a nuisance hyperparameter.**
`sigma_xeq` — the random-walk width on the unobservable equilibrium point, which
says nothing about the operator — moves `K_hat` across 46–1235 N/m (27x) while
the filter's own one-step predictive error only moves 14–17% of signal. The
prior *mean* barely matters (5%), because the likelihood constrains the product
`K(x_eq - x)`; the assumption about how fast `x_eq` wanders is what sets `K`.
Some settings land near the true 400 N/m and nothing in the data says which.

## T1: which factor of a stiffness label does a task care about?

**No suitable task existed.** ManiSkill's drawing tasks are explicitly kinematic
-- `draw.py` says so itself, "we do not actually check if the robot contacts the
table" -- so stiffness cannot affect their success. `PegInsertionSide-v1` is the
only tabletop task with real contact, but it randomises the box about z only,
needs a grasp phase first, and scores success geometrically. Nothing anywhere in
ManiSkill does sustained surface contact. `wipe.py` is the task built for this.

**A label carries three things, and papers report one.** `K = exp(s) R diag(w) R^T`:
a magnitude, an anisotropy ratio, and a FRAME -- which axis is the soft one. No
rule states the frame as a choice, yet "stiffness along the direction of motion"
and "stiffness along the surface normal" are 90 degrees apart on a wipe.

Each factor swept alone over 5 surfaces at +-25 deg tilt, scored by a single
failure margin `max(force_error/tol, path_error/tol)`:

| factor | tolerated | published rules disagree by | verdict |
|---|---|---|---|
| magnitude | 0.74-4.7x (**6x**) | 4-6x in the literature's sweeps; 27x measured above | marginal |
| ratio | 1.9-100 (**52x**, one-sided) | — | forgiving |
| frame | **0-25 deg** | up to **90 deg** | exceeded |

**Threshold sensitivity, because the success thresholds were a choice.** Raw
metrics are cached so thresholds can be varied without re-running:

- the ratio window is the widest in **5/5** feasible settings;
- the frame tolerance **never reaches 40 deg**, let alone the 90 deg that
  separates the two frame conventions;
- the magnitude window, however, spans **3x to 59x** across those settings. At
  strict tolerances it is the size of the published disagreement; at loose ones
  it comfortably contains it.

So the frame result is robust and the magnitude result is conditional on how
demanding the task is. **The original hypothesis for T1 -- "anisotropy carries
the intention, magnitude is nearly irrelevant" -- is wrong as stated.** Magnitude
is bounded on both sides: too soft and friction drags the tool off the stroke,
too stiff and the unknown surface orientation turns into force error. What
survives is narrower and sharper: of the three factors, the one nobody writes
down is the one the task cannot absorb.

Run: `python3 study_anisotropy.py` (`--replot` redraws from the cache).

## The wipe on the real arm: a scene that can actually be wiped

`../fr3_wipe_scene.py`. The first FR3 attempt (`../fr3_tilted_wipe.py`) confirmed
only the direction of the frame effect, by 1.3x, because the contact was two
thin Panda fingers grazing a small box: pressing worked perfectly, and the
moment tangential motion started the contact collapsed to 60% and spiked to
150 N. Box friction made no difference at any coefficient, so the failure was
geometric. The scene was replaced with a **320 x 320 mm tilted plate** and a
**15 mm sphere welded to the hand** -- a sphere on a plane has no edge to catch,
and the weld removes the grasp slip that moved the peg 44 mm in the insertion
study (it holds to 0.024 mm over a second).

| | fingers on a box | welded pad on a plate |
|---|---|---|
| contact held | 60% | **100%** |
| force error | 10.35 N (173%) | **0.64 N (8%)** |
| peak | 36.9 N | **7.9 N** |

Same label numbers read in four frames, 18/-10 degree tilt, 50 mm stroke:

| label frame | soft axis off the normal | K along the normal | force error | peak |
|---|---|---|---|---|
| task frame | 0° | 259 | **8%** | 7.9 N |
| world axes | 21° | 544 | 16% | 9.4 N |
| end-effector rest axes | 21° | 544 | 16% | 9.4 N |
| **motion-aligned** | **90°** | **2585** | **46%** | 4.8 N |

The measured stiffness along the normal matches `Kn cos^2(th) + Kt sin^2(th)` to
**0.0%** at every angle, so the frame error IS a stiffness error and nothing
else. A motion-aligned reading -- "stiffness along the direction of motion" --
puts the soft axis 90 degrees from the surface normal and makes the surface
**10x stiffer than intended**.

It also under-delivers force, to 4.3 N against a commanded 8 N, which is the
series-compliance factor `1/(1 + Ko/ke + Ko/Ki)` from the top of this file
evaluated at `Ko = 2585, Ki = 3000`: 0.54. The whole chain of results closes on
the same formula.

Two of the four readings coincide: the Panda's rest TCP has its z along -world z
and the two stiff axes are degenerate, so "world axes" and "end-effector rest
axes" are the same matrix. The stroke is 50 mm because the arm tracks that to
3.4 mm at this pose while 90 mm gives 24 mm -- a reach limit, not a stiffness one.

## Insertion: the same question on a task where motion and force are ALIGNED

`../peg_insertion_cascade.py` runs PegInsertionSide under the same cascade, from
a pre-grasped peg. The contrast with the wipe is the point. On a wipe the motion
is tangential and the contact force is normal, so a motion-aligned rule and a
force-aligned rule end up **orthogonal**. During an insertion both lie along the
hole axis, so those two rules and the correct one **collapse onto the same
frame**, and only the frame-free conventions can be wrong.

Same label numbers, read in three frames, 8 episodes with the hole orientation
randomised:

| label frame | success | axial K | stiff axis vs hole | peak force |
|---|---|---|---|---|
| hole axis (= motion = force) | 100% | 5570 | 0° | 26.0 N |
| end-effector axes | 100% | 5149 | 15° | 24.1 N |
| world / per-axis axes | 75% | 955 | 75° | 48.1 N |

A frame **75 degrees wrong**, with 5.8x less stiffness along the insertion axis,
still inserts 75% of the time — where the wipe tolerated only 25 degrees.

**The design rule is therefore specific, not universal: frame ambiguity bites
when the motion and the contact force are orthogonal, and an insertion-heavy
literature would never have met it.**

Two caveats, both measured. `_build_box_with_hole` makes four plain boxes with
**no chamfer**, so there is no lead-in to convert an axial push into a lateral
correction: sweeping the anisotropy ratio from 10 (soft laterally) to 0.1 (stiff
laterally) changes nothing, and a lateral aim error of 3 mm inserts while 6 mm
fails regardless. The classic RCC argument for lateral compliance needs a
chamfer this task does not have. And 8 episodes is a small sample.

### What it took to get a reliable baseline

Five things, none of them the controller:

1. **Robot.** The env defaults to `panda_wristcam`; the wrist camera's collision
   body hits the box at 168 N while the peg is still 8 mm short. Plain `panda`.
2. **Simulation rate.** PegInsertionSide's default SimConfig runs at 100 Hz. An
   inner Cartesian impedance cannot live there — `study_layers.py` put the
   requirement at 200 Hz minimum. 500 Hz.
3. **Contact force.** `peg.get_net_contact_forces()` includes the gripper
   squeezing the peg. Feeding the grasp force into the admittance drove the arm
   to 8 m/s in 10 ms. Use `scene.get_pairwise_contact_forces(peg, box)`.
4. **Grasp offset.** Held at its centre, the hand strikes the box face at 107 N
   before the head reaches the success threshold — a geometric limit of the
   grasp. Grip 40 mm behind the centre.
5. **Aim the head, not the centre.** `goal_pose` gives the peg-centre position
   that works only for a perfectly aligned peg; a residual angle displaces the
   head by L sin(theta), which at 124 mm and 12 degrees is 27 mm against 3 mm of
   clearance. Closing the loop on the measured peg axis, with Kr raised from 120
   to 400 Nm/rad, took the success rate from 2/6 to 6/6.

And one that WAS the controller, shared with the FR3 script: **rotational
damping matched to the wrong inertia.** `Dr = 2 zeta sqrt(Kr)` assumes
1 kg m^2; a wrist with a gripped peg is ~0.005. The 200x overestimate put the
loop past the explicit limit 2I/D, and the arm sat still for 0.24 s before
locking into a step-to-step sign flip at 142 rad/s. Both scripts now use
`Dr = 2 zeta sqrt(I Kr)`.

## What this found in the FR3 script

1. **Admittance sign error.** `Vr_dot = solve(Ma, f_ch - f_e - Br@Vr)` subtracted
   a reaction that already points against the penetration, so contact force drove
   the reference *into* the wall. Contact ran away instantly and stopped only at
   the workspace clamp, at 200+ N with `pr_z` pinned to `workspace_lo`. Fixed to
   `f_ch + f_e`; contact forces are now 1.9–7.5 N and `f_ch == f_e` at rest, the
   quasi-static identity this package tests.
2. **Gravity compensation must NOT be applied.** Every link carries
   `disable_gravity=True`. Zero torque drifts 0.00 mm in 0.5 s; applying
   `compute_passive_force()` throws the end effector 998 mm.
3. **The IK + joint PD baseline is not a rigid position source.** Its effective
   Cartesian stiffness measures ~1970 N/m — fitted on one point, it predicts the
   `ik`-vs-`impedance` force ratio at `Ki` = 800/2000/4000 to within 1.6%.

## What each layer buys: the ablation

`../study_ablation.py` → `../ablation.png`. Three controllers, identical
otherwise, on both tasks:

| | outer (force feedback) | inner (compliance) |
|---|---|---|
| cascade | admittance | Ki = 3000 |
| outer only | admittance | Ki = 20000 (rigid tracking) |
| inner only | none, pose straight through | Ki = 3000 |

**Wipe** — outward spiral over 8 bumps on a tilted plate, 8 N commanded:

| | surface height error | contact | mean force | peak |
|---|---|---|---|---|
| cascade | 0 mm | 96% | 7.74 N | 76 N |
| cascade | **3 mm** | 96% | **7.74 N** | 76 N |
| outer only | 0 / 3 mm | **62%** | 18.82 N | **230 N** |
| inner only | 0 mm | 97% | 12.45 N | 17 N |
| inner only | **3 mm** | 98% | **21.06 N** | 26 N |

**Insertion** — 4 episodes each:

| | lateral error | success | mean force |
|---|---|---|---|
| cascade | 0 / 3 mm | 100% / 50% | 1.62 / **6.47 N** |
| outer only | 0 / 3 mm | 100% / 25% | 1.24 / **22.49 N** |
| inner only | 0 / 3 mm | **25% / 25%** | 1.22 / 1.31 N |

Three different failures:

- **Remove the inner compliance** and the tool cannot ride a surface. Over the
  bumps it bounces: contact drops to 62%, the peak triples to 230 N, the mean
  force is 2.4x the command. On insertion it is fine when perfectly aimed and
  3.5x more forceful when not (22.5 N against the cascade's 6.5 N).
- **Remove the force feedback** and the tool cannot find a surface. The cascade's
  numbers are *identical* to the digit with a 3 mm height error, because the
  admittance integrates until the force matches and does not care where the
  surface is; the inner-only run's mean force goes 12.45 → 21.06 N. On insertion
  it barely inserts at all (25%) -- a pose command alone does not push a peg
  through a hole.
- **The cascade is not uniformly best.** With perfect aim its peak insertion
  force is higher than outer-only's (25 vs 9 N): a soft inner lets the outer
  layer wind further before the contact releases.

One number worth keeping: inner-only delivers 12.45 N for a commanded 8 N on a
perfectly known surface. The commanded penetration came from
`f_target / series(Ki, ke)` and is **55% off**, because the real contact involves
the pad's sphere geometry and the plate, not a 1-D spring. Open-loop force needs
a contact model, and the contact model is wrong.

## Demonstrations

`spiral_task.py` generates them, `collect_demos.py` collects them.

An operator who INTENDS a constant-pitch spiral at a constant normal force, on a
random surface they cannot see. Every other study in this package was blocked by
the same thing -- the operator's intent is unobservable, so every label rule
substitutes a proxy and none can be checked. Here the intent is known by
construction, so what separates intent from execution is measurable.

**What makes an operator an operator is delay, not degrees of freedom.** A
zero-delay model tracks any surface perfectly and leaves no gap at all. This one
has a visual loop (200 ms, corrects the in-plane path -- what they can see), a
haptic loop (60 ms, adjusts how hard they press -- what they can feel, sooner),
and signal-dependent motor noise. Two loops, different delays, different axes:
the operator is itself a cascade.

**The surface must tilt the contact normal, not only move up and down.** The
first version varied only the height, so a spiral over bumps tracked as well as
over glass and the scenario measured nothing. What matters is the SLOPE,
`amp / length_scale`: at 2.5 mm RMS a 45 mm length scale costs 1.2x the flat-plate
path error, and a 12 mm one costs 3.0x.

300 episodes, each a different surface, tilt, operator and intended pitch:

| | mean | range |
|---|---|---|
| path error from the ideal spiral | 2.19 ± 0.79 mm | 0.79 – 6.13 |
| force RMSE (commanded 6–10 N) | 0.66 ± 0.40 N | 0.21 – 3.22 |
| contact held | 99.97% | 98.65 – 100 |

113,700 policy steps at 20 Hz. All 300 passed the filter, which is a statement
about the operator rather than about the filter -- the worst episode sits at
6.13 mm against an 8 mm threshold.

### The labels, and how they are checked

Actions use the fixed gauge: `x_ref := x_r`, `f_d := f_ch`. The check that
matters is not the shapes or a gauge identity that holds by construction, it is
whether replaying the stored labels reproduces the demonstration:

| replayed with | resulting force |
|---|---|
| `x_ref := x_r` (stored) | the demonstration, **RMSE 0.01–0.26 N** |
| `x_ref := x_c` (the gauge this package warns about) | **1.86–1.88x** the demonstration |

The 1.87x matches the 1.93x measured in the 1-DOF test at the top of this file,
from a completely different setup. The verifier can fail; the labels pass. The
residual 0.26 N is the cost of decimating the action to 100 Hz, not a labelling
error -- replaying at the stored rate instead of the simulation rate doubles it.

Everything is seeded, so the 69 MB of `demos/spiral_v1` is regenerable and can
be gitignored:

```bash
python3 collect_demos.py --episodes 300 --workers 14 --out demos/spiral_v1
```

## Video

`../record_demo.py` renders the two demonstrations.

```bash
cd .. && python3 record_demo.py --what wipe        # wipe_demo.mp4 / .gif
cd .. && python3 record_demo.py --what insertion   # insertion_demo.mp4 / .gif
```

```bash
cd .. && python3 record_demo.py --what wipe --single --path spiral \
        --v-limit 0.20 --duration 26 --out spiral_demo
```

The spiral is the one to watch. The tool winds outward from the centre to a
100 mm radius over 3.5 turns, covering the whole tilted plate, and the normal
force stays at 7.3-7.4 N against a commanded 8 N the entire way: **contact held
100%, force RMSE 0.68 N (8%), path error 1.09 mm** at a 0.12 m/s rim speed.
Holding a force constant while sweeping an unknown tilted surface is the thing
the cascade is for, and a spiral shows it in one shot where a back-and-forth
stroke retraces one line.

One correction to an earlier note in this file: the growing path error of longer
strokes was blamed on the arm's reach. It was the outer layer's `v_limit`. At
0.03 m/s a 90 mm stroke tracks to 24 mm; at 0.10 m/s the same stroke tracks to
0.97 mm, and a 220 mm-diameter spiral tracks to 1.17 mm. The clamp is there to
bound the APPROACH speed, where it cuts the impact peak 3.6x -- during the
stroke it should sit above the path's own speed.

The other wipe recording is a split screen of the SAME label numbers read in two
frames, stepped in lockstep: left the task frame (259 N/m along the surface
normal, holding 7.4 N against a commanded 8), right the motion-aligned reading
(2585 N/m, delivering 4.3 N). The two arms look alike; the force readouts do
not, which is the point. The insertion recording shows the contrasting task,
where motion and force share the hole axis.

Neither environment ships a usable render camera -- the demo env's
`_default_human_render_camera_configs` computes a pose and then returns an
empty dict -- so `record_demo.py` installs one. Note that the camera is built by
`reset()`, not by `gym.make()`: restoring a monkeypatched camera config in
between leaves the original in place and the reframing silently does nothing.

## Running

```bash
python3 tests.py                    # 87 checks, ~2 min
python3 study_labels.py             # figure + csv  (--quick for a short run)
python3 study_layers.py             # layer roles and bandwidths
python3 study_anisotropy.py         # T1  (--replot redraws from the cache)

cd .. && python3 haptic_teleop_fr3_bilateral.py --inner impedance --Ki 2000
cd .. && python3 haptic_teleop_fr3_bilateral.py --inner ik          # baseline
```

`core.safe_dt(params)` picks a stable timestep; `CascadeSim` refuses to
construct at an unstable `dt` and names the element that set the limit.


## The two single-interface cases

`case1.py` and `case2.py` implement the manuscript's two execution cases against
the **same** wall (`core.unilateral_wall`) and the **same** reference ramp
(`core.quintic_pos`) as the cascade, so a force difference between them is
attributable to controller structure and nothing else.

**The cascade is neither case — it is both stacked.** Case 1 has no outer
admittance state (`(x_r, v_r)` simply does not exist); Case 2 has no inner law
at all, only a vendor servo. `haptic_teleop_fr3_bilateral.py` integrates an
outer admittance *and* applies inner GIC torque, so it pays a series compliance
neither single-interface case incurs:

| | series chain | delivered at a 35 mm command, `ke`=1000 |
|---|---|---|
| Case 1 | `1 + Kp/ke` | 5.308 N |
| Case 2 | `1 + Ka/ke` | 5.308 N |
| cascade, `Ki`=2000 | `1 + Ko/ke + Ko/Ki` | 4.759 N (−10.3%) |
| cascade, `Ki`=`Ko` | same | 3.000 N (−43%) |

`Ki >> Ko` is a requirement *created by stacking the layers*. It is not a fact
about impedance control.

**Case 1 is the only one of the three that enforces anything.** The tank gate
(`alpha = min(1, E/Ec)`) scales active torque, reference velocity and gain rates
together while leaving the passive baseline `tau_0` untouched. Under a 200 N
active-torque proposal it admits 0.960 J of the 1.0 J budget and drives `alpha`
to 0.0019; the same proposal ungated injects 32.7 J — **34x**. The cascade's
`E_ch`/`H_ch` comparison and Case 2's `r_M` are both retrospective and throttle
nothing.

### Two traps this package now makes numerical

**Requested is not applied.** With a small tank the gate throttles `V_d`, so the
run tracks 34.18 mm of a requested 35.00 mm. `contact_force_case1` still matches
to 1e-13 — *against the applied reference*. Labelling the requested one is wrong
by 3.6%.

**Case 2's steady state is not determined by its gains.** The servo invariant
`c = x_r − x_s − Ts*v_s` selects which equilibrium is reached, and 66 clipped
steps leave `c = −0.72 mm`. Predicted force at `c=0` is 5.308 N; the run settles
at 5.474 N, which `contact_force_case2(..., c=c)` reproduces to 5e-8. No gain
changed. The cascade has no equivalent free parameter.

### Verified against the manuscript

- **Theorem 1** energy identity: residual first-order in `dt` (ratio 2.00 over
  four refinements). The dissipation inequality closes; note that `H_T + E` is
  *not* monotone — `−f_e·V_s` is positive on retreat, when the wall returns
  stored energy.
- **Prop. 1** port defect: identity first-order; `r_M → 0` exactly (0.0 W) under
  `Ts=0` with perfect sensing, and is 5.5e-2 W for the realistic servo.
- **Prop. 2** Routh: agrees with the reduced cubic's poles on **250/250**
  randomised parameter sets.
- **Table I**: all 15 numbers reproduced to 2 decimals.

Two notes for the manuscript arising from this:

1. The Table I caption states the 3–6 s window for the force column only. The
   **RMS column is the full 0–6 s run** — 3–6 s gives 0.72/1.26/97.83/1.53/61.49
   against the published 0.65/1.31/79.69/1.43/44.38, and 0–6 s reproduces all
   five exactly. Worth stating in the caption.
2. Eq. 5's rotational gradient omits the 1/2 that
   `haptic_teleop_fr3_bilateral.geometric_impedance_torque` carries. Numerically,
   `d/dt tr{K_R(I−R_de)}` matches `vee(...)·w` without the 1/2, so Eq. 4–5 are
   self-consistent and the **code's `Kr` is half the manuscript's convention**.


## Three executors on the spiral wipe

`python3 study_case_spiral.py --seeds 8` → `case_spiral.png`

Same delayed operator, same random surface, same seed, same `dt`, same
commanded 8 N. Only the follower changes. `Ka = Kp = 300` is held fixed across
all three, so "the gain that faces the environment" is constant and the
structure is the variable.

| executor | path RMS | force RMS | f mean | f std | contact |
|---|---|---|---|---|---|
| cascade | 2.66 mm | 0.90 N | 8.37 N | 0.82 N | 100.0% |
| Case 1 | **2.41 mm** | **0.88 N** | 8.29 N | 0.82 N | 100.0% |
| Case 2, `da`=150 | 4.31 mm | 1.66 N | 8.36 N | 1.62 N | 99.4% |
| Case 2, `da`=65 | 28.00 mm | 65.42 N | 63.80 N | 34.08 N | 98.7% |

**The static comparison cannot see this.** On a quasistatic push Case 1 and
Case 2 are indistinguishable — both are one series compliance, both deliver
5.308 N. On a moving, undulating, stiff surface they separate completely, and
the reason is not the admittance:

```
ke = 5000 N/m, Ts = 20 ms, ma = 3  ->  Eq. 15 requires  da >= 67.20 Ns/m
the cascade's own gains give                            da  = 65.00 Ns/m
```

**Case 2 misses the stability boundary by 2.2 Ns/m — 3% — on gains the cascade
runs happily**, because the cascade terminates in a Cartesian impedance rather
than a lag servo. The same numbers are safe in one structure and divergent in
the other. `analytic.routh_da_min` gives the boundary in closed form, and the
useful thing to read off it is that the `ka*Ts^2` coefficient is negligible:
the requirement tracks `ma*Ts*ke`, so **softening the coupling buys nothing**
and what matters is the workpiece stiffness times the servo lag — neither of
which is a gain the designer chose.

**Case 1 edges out the cascade here** (2.41 vs 2.66 mm, 0.88 vs 0.90 N). With
one operator and eight seeds that gap is inside the seed spread, so the honest
reading is *no penalty for dropping the outer layer on this task*, not a win.
Its own force error is the smallest of the three because it has the fewest
series compliances between command and tool.

**Routh stability is necessary, not sufficient.** `Ts = 5 ms, da = 65` satisfies
Eq. 15 and still lands at 21 N with the limiter active 55% of the time. The
condition is local, linearized and excludes saturation — exactly as the
manuscript states. Clearing it is not the same as behaving.

Two implementation notes worth keeping:

- The operator lives in `spiral_task.make_operator_target` so all three
  executors are driven by the *identical* feedback loops. If the operator
  differed between runs, none of the numbers above would mean anything.
- `common_dt` gives all three one step size. This is not tidiness: the
  operator's haptic loop integrates a rate, so a different `dt` is a different
  operator.
