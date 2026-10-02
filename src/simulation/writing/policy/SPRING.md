# The spring-consistent action, and what it is measured to buy

Four changes, on the writing task.  Everything here is reproducible from this
directory; `tests.py` is 45/45 (was 35/35) and the eight legacy runs are
untouched -- `--layout legacy` is the default and every existing checkpoint
still loads.

## 1. The action layout (steps 2 and the "future_ft derived" half)

GIC ties three quantities with one equation, `F = K (x_d - x)`.  Which two the
policy names decides what a stiffness error costs.

| | names | A | the controller then |
|---|---|---|---|
| `legacy` | target `x_d`, stiffness `K` | 6 | applies `K (x_d - x)`, whatever that is |
| `spring` | expected pose `x_ref`, wrench `f_d`, stiffness `K` | 9 | places the target at `x_ref + K^-1 f_d` |

Both spring labels are **measurements already in every recorded episode** --
`obs/tcp_pos` and `obs/f_contact` -- so no demonstrations were re-collected.
`data.episode_samples(..., layout="spring")` derives them.

**The future wrench is now inside the action.** On the demonstration `x = x_ref`,
so `F(t+1+j) = f_d(t+1+j)` exactly.  Verified on a real episode: the wrench
implied by the `f_d` columns matches the stored `fut_ft` to **0.0 N**.  Cases b,
c and d exist to reconcile an action head with a force head; under this layout
they have nothing left to reconcile, which is what makes the 4 x 2 grid the
measurement rather than a new case.

### What it buys, measured

`evaluate.py --k-err` is the scripted writer with a stiffness head that is wrong
by `k_err` while its pose head reproduced the demonstration.  6 episodes each:

| `k_err` | legacy force | legacy success | spring force | spring success |
|---|---|---|---|---|
| 0.3 | **1.26 N** | 0/6 (force, coverage) | 3.27 N | 6/6 |
| 1.0 | 3.14 N | 6/6 | 3.03 N | 6/6 |
| 2.0 | **6.02 N** | 0/6 (force, tear) | 3.01 N | 5/6 |

The legacy force tracks the stiffness error one for one.  The spring force does
not move.  At `k_err = 1.0` the two layouts agree to three decimals on every
metric, which is the decode check: the same controller input with the division
moved from the policy to the controller.

**Where the error went instead.**  It did not vanish, it moved into the
engagement transient.  Commanding `f_d` as a step at `k_n = 120 N/m` moves the
target by 25 mm at once: peak 13.3 N and the paper torn 6/6.  Ramping `f_d` over
0.2 s at both ends fixes it (peak 7.0 N, 6/6) and also improves the nominal case
-- peak 8.5 -> 6.0 N, in-band 0.968 -> 0.991 -- because the release now spends
the spring before lifting instead of flicking the pen.  **A force command has to
be ramped; a depth command never had to be.**  The remaining 5/6 at `k_err = 2.0`
is the honest residue: K no longer moves the force, it moves the sensitivity to
surface deviation, which is what K is supposed to be choosing.

## 2. K's only gradient (step 3)

Under `spring`, `x = x_ref` on every training sample, so the spring term is
identically zero there and **imitation gives K no gradient at all**.  That is the
identifiability problem stated honestly rather than a defect: K is not in
`(x, F)`, and the legacy layout only looked like it learned K because K was
absorbing gradient that belonged to the force.

`--w-robust` supplies the missing gradient from the one thing known without any
human label -- the surface is not where `x_ref` said.  Two opposing sizing rules
(`model.robust_loss`), both dimensionless before they are added:

    soft enough    |f_d| +/- k_n delta stays in the force band -> k_n <~ (f - F_lo)/delta
    stiff enough   friction mu|f_d| does not drag the tip off the line -> k_t >~ mu f / tol

`delta` is a **requirement**, not an estimate: "keep the force in band when the
surface is up to delta from where the policy expected it".  Default 2 mm, half
the +/- 4 mm canvas height error the task randomises.

**Calibration, which is the reason to believe it.**  At `f = 3 N`, band
`(1, 6) N`, `delta = 2 mm`, `mu = 0.5`, `tol = 3 mm` the rules give
`k_n <~ 1000 N/m` and `k_t >~ 500 N/m` -- exactly the MID and LOW levels the
demonstration protocol uses (`tests.py`).  The term explains the demonstrated
stiffnesses instead of fighting them.

`W_ROBUST=0` vs `W_ROBUST=1` is the ablation that says whether K was ever being
learned.

## 3. Force as surprise (step 5)

`policy/surprise.py` fits a cross-fitted, heteroscedastic expectation model
`g(images, where I am, where I am going) -> (mu, sigma)` of the wrench, and
`--surprise` widens `ft_hist` from 3 to 9 channels:
`[measured | expected | z]`, with `z = (F - mu)/sigma` gated to zero out of
contact.

**The leak that makes this feature useless if you miss it.**  `data.state_vec`
carries `tcp - x_d` and `log K`, and the spring law makes those two
*algebraically* the force.  An expectation model given the full state does not
predict the wrench, it computes it, `mu` comes out equal to `F`, and `z` is
identically zero -- a feature that trains without error and carries nothing.
`surprise.S_KEEP` is the force-free slice; the deflection, `log K` and `qpos` are
all excluded (`qpos` because FK plus `x_d` rebuilds the deflection).

### Out-of-fold R^2, 150 episodes, 4 folds, 2500 steps/fold

| axis | R^2 | sigma |
|---|---|---|
| world x (tangential) | **-0.196** | 0.065 N |
| world y (tangential) | **0.002** | 0.069 N |
| world z (normal) | **0.867** | 0.362 N |

The split *is* the result.  The camera predicts the pressing force from geometry
and knows **nothing** about friction -- `mu` is randomised 0.2-0.5 and invisible
-- so the tangential residual is pure "can be felt, cannot be seen", which is the
role the feature exists for.  Residual `|z|` mean 0.70, p95 2.37: calibrated.

An R^2 near 1.0 on every axis would have meant the residual was noise and the
six extra channels were not worth their cost.  Check it before building on it.

## 4. Rotational stiffness: the felt / `K_R` distinction (step 4)

`controller.py` now takes a **3x3 `K_R` in the tool body frame** (a scalar still
works and is what every current caller passes).  The generalization is exact:
45/45 tests, and the scalar path reduces to the old law term for term.

The elastic moment is `1/2 vee(K_R R_d^T R - R^T R_d K_R)`, whose small-angle
gradient is

    K_eff = 1/2 (tr(K_R) I - K_R) = 1/2 diag(b+c, a+c, a+b)

so **the stiffness felt about an axis does not depend on that axis's own `K_R`
entry at all.**  Measured by probing the controller's own law (`tests.py`):

| `K_R` | felt | |
|---|---|---|
| `diag(10, 30, 100)` | `diag(65, 55, 20)` | the axis set to 100 is the softest |
| `diag(50, 5, 50)` | `diag(27.5, 50, 27.5)` | "softening" y makes y the **stiffest** |
| `diag(5, 95, 5)` | `diag(50, 5, 50)` | what you must pass to get felt `(50, 5, 50)` |
| `kr I` | `kr I` | isotropic hides all of it -- which is why writing never saw it |

Conversions are `felt_from_KR` / `KR_from_felt` (`K_R = tr(K_eff) I - 2 K_eff` in
this file's `1/2` convention -- derive it for yours, the factor depends on it).
`felt_reachable` is the test to run first: no felt axis may be stiffer than the
other two together, so `felt (5, 5, 50)` -- "resist twist, comply in tilt", the
peg-insertion profile -- needs `K_R = diag(50, 50, -40)` and is **unreachable**.
Asking anyway is refused by the eigenvalue clip, not approximated: you get felt
`(21.8, 21.8, 43.4)`.

**Scope.** This is the controller, the conversion and the tests.  The writing
task cannot show a task-level `K_R` effect and nothing here claims it does -- the
pen tip is a ball at the compliance centre, so no contact moment exists.  The
task-level demonstration belongs to `../wiping`, which is where anisotropy is
demanded; this is the plumbing it will import, with the trap closed before it
gets there.

## Known inconsistency, not changed

`compute()` meters the tank with `tr(U_r (I - R_d^T R))`, which corresponds to a
potential `tr(K_R (I - R_d^T R))`, while `e_R` carries a `1/2` and so
corresponds to `1/2 tr(...)`.  The tank therefore **over-charges stiffening by
2x**.  Over-charging drains the tank faster and closes the gate sooner, so it is
conservative and passivity is not at risk; it is left alone because it is an
approved design and `tests.py` pins the value.  Worth deciding deliberately.

## Running it

    # once: the expectation model (a prerequisite, not a variant)
    mkdir -p logs && sbatch policy/slurm/surprise.sbatch

    # the three sweeps worth running against the existing eight
    LAYOUT=spring W_ROBUST=0 bash policy/slurm/submit_all.sh
    LAYOUT=spring W_ROBUST=1 bash policy/slurm/submit_all.sh
    LAYOUT=spring W_ROBUST=1 SURPRISE=demos/protocol_v1/_surprise_64_k4.npz \
        bash policy/slurm/submit_all.sh

Checkpoints land in `policy/runs/<[spring_]name[_sur]>/seed<k>/`; the names come
from `model.run_name()` and `_train.sh` composes the same ones, so no two
combinations share a directory.

New open-loop diagnostics under `spring`: `fd_N` / `fd_n_N`, the commanded force
error in newtons.  It is directly comparable with `ft_N`, which is the same
quantity from a separate head -- and that comparison is the point of the grid.

---

# A and B: making the comparison possible at all

`policy/label_check.py` on the demonstrations the eight runs were trained on:

    demos/protocol_v1   150 episodes, 170008 labelled frames
      xy level used        LOW  70.5%  MID  29.5%  HIGH   0.0%
      xy level per episode LOW 100.0%  MID   0.0%  HIGH   0.0%
      BEST PREDICTOR OF THE xy LABEL, by what it is allowed to see:
        phase only          100.0%   <- the lookup table
        drag only            79.5%
      I(friction ; xy level) 0.000 bits        (max 1.58)
      z level: phase only 100.0%

**The stiffness label carries zero information about the environment.** A phase
classifier reproduces it exactly, and in contact the xy level is LOW in 100% of
episodes -- a constant, not a decision.  That is why all eight structures command
the demonstrated K to 0.0-0.7% and sit at `acc xy` 97.7-97.9% in
`policy/runs/COMPARISON.md`: the stiffness axis of that table cannot distinguish
policies, and no architecture change -- including everything above -- can show
anything on it.  Which saturated is worth being precise about: SUCCESS is not
(the flow family spans 52.1-93.8%), K is.

Two ways out, at very different cost.

## A -- change the evaluation, re-collect nothing

Stop asking the policy to reproduce a K and ask what its K is FOR.  The
demonstrations were collected at `dz = 4 mm, tilt = 5 deg` and
`eval_policy.sbatch` evaluates at the same numbers, so there is no distribution
shift at all.  `policy/rollout.py --tilt-sweep` raises it; the demos are
untouched.

**dz is the wrong axis, and this was measured rather than assumed.** dz is a
CONSTANT height offset, and the policy absorbs it exactly as the operator does --
it descends until it feels contact, then presses.  Swept to 20 mm,
`act_ft_input/seed0` still succeeded and in-band moved only 99.8 -> 96.6%.  A flat
curve.  `--dz-sweep` is kept because showing that it is flat is worth one run.

**Tilt is deviation DURING contact.**  Over a 37 mm glyph the surface height
ranges `w tan(theta)`, so the sizing rule `k_n <~ (f - F_lo)/delta` caps the
stiffness at

    tilt      5 deg      8 deg     12 deg     15 deg     20 deg
    swing     3.2 mm     5.2 mm     7.9 mm     9.9 mm    13.5 mm
    k_n <~    1236       769        509        403        297     N/m

and the demonstrations command **1000 N/m at every tilt**.  So the memorised
constant is already at its limit at the tilt it was collected at and should start
losing the band around 8-12 deg.  The band is 1-6 N and the paper tears at 12 N.

### Measured, 2 episodes, `act_ft_input/seed0`

| tilt | success | coverage | in band | contact N | K_n |
|---|---|---|---|---|---|
| 5 deg (the demos') | 100% | 100.0% | 99.8% | 3.22 | 1001 |
| 20 deg | **50%** | **64.7%** | 99.3% | 2.74 | 1002 |

Two episodes, so the number is indicative; the MECHANISM is not.  Note which
metric moved: `in_band` held at 99.3% while coverage collapsed, because `in_band`
is averaged over PEN-DOWN steps only.  An over-stiff `k_n` does not press wrong --
**it lifts the pen off the paper**, and the glyph simply never gets written:

    tilt 12 deg, k_n 1000 N/m: 7.9 mm range -> 3 N +- 3.9 N = -0.9 .. +6.9 N  loses contact
    tilt 12 deg, k_n  400 N/m: 7.9 mm range -> 3 N +- 1.6 N = +1.4 .. +4.6 N  holds
    tilt 20 deg, k_n 1000 N/m:              -> -3.7 .. +9.7 N
    tilt  5 deg, k_n 1000 N/m:              -> +1.4 .. +4.6 N   (the demos, just inside)

So watch **coverage**, and `K_n` held at 1000 across the sweep is the point: the
policy kept commanding the memorised constant while the paper went out from under
it.

**And the two layouts are trapped differently, which is the prediction worth
testing.**  Legacy fixes the DEPTH, so `force = k_n x depth`: soft means weak
means no ink, stiff means the force swings off the paper on a tilt.  It is
squeezed from both sides and has an interior optimum it must find per episode.
Spring commands `f_d` directly, so soft holds 3 N at every tilt and the low-end
cost disappears -- the spring arm should be able to go soft and hold coverage
across the whole sweep where the legacy arm cannot.

That makes the **delta x tilt experiment** the one falsifiable test of
`--w-robust` that runs on existing demonstrations:

> a policy trained at `delta = 8 mm` should outlive one trained at `delta = 2 mm`
> at tilt 12-20 deg, and should be slightly WORSE at 5 deg, because it gives up
> force precision at nominal to buy tolerance.

Both curves lying on top of each other falsifies the term.  Four arms, so the
layout and the term are separable:

    bash policy/slurm/robust_experiment.sh          # trains + sweeps all four
    python3 policy/plot_stress.py --runs policy/runs

| arm | run directory |
|---|---|
| legacy (control, already trained) | `act_ft_input` |
| spring, K unsupervised | `spring_act_ft_input` |
| spring + robust, delta 2 mm | `spring_act_ft_input_d2` |
| spring + robust, delta 8 mm | `spring_act_ft_input_d8` |

`robust_delta` is in the run name, so two deltas cannot share a directory.

## B -- change the protocol, re-collect (needs your sign-off)

This module's own docstring had the answer in it:

> *xy low 300 failed coverage on high-friction paper (3/36): mu 0.48 x 3.7 N of
> friction drags a 300 N/m pen ~6 mm behind its reference, past the 3 mm
> tolerance on curves.  At 500 it wrote 36/36.*

The xy demand **does** depend on friction.  It was handled by picking the one
level that covers the worst friction in a narrow range -- i.e. by removing the
variable-impedance problem from the task by design.

`protocol.py --adaptive-xy` makes the operator do what a hand does: feel the pen
dragging and stiffen along the paper.  The lag of a spring against friction is
`mu f_n / k_t`, so the level needed is `k_t >~ drag / tol_eff` with
`tol_eff ~ 3.5 mm` back-calculated from the measurement quoted above:

    LOW  (500 N/m)  holds while drag <~ 1.75 N
    MID  (1000)     holds while drag <~ 3.50 N
    HIGH (3000)     above that

With `--mu 0.15 1.6` all three rungs are genuinely used, and the level now
depends on friction, which is randomised per episode and **invisible**: measured
out of fold, vision explains R^2 = -0.20 and 0.00 of the two tangential force
axes against 0.87 of the normal one.  The camera predicts the pressing force from
geometry and knows nothing whatever about friction, so the xy label becomes
inferable **only from the force history**.  `acc xy` becomes a measurement
instead of a memorised constant, and "force is a modality vision cannot replace"
stops being an assertion.

Two details that are load-bearing:

* **drag starts at zero.**  There is no friction until the pen slides, so the
  operator begins a stroke at LOW and stiffens as it feels the pull.  The label
  is causally downstream of the force, which is what makes it learnable from
  force and not from the phase.
* **hysteresis** (`DRAG_HYST`), or a drag sitting on a threshold chatters every
  step and neither the operator nor the compliance check is reproducible.  The
  compliance check scores against the same `expected_levels()` the operator
  follows, so an adaptive episode is not dropped for doing what the adaptive
  protocol asks.

    python3 protocol.py --texts S 7 "<star>" --per-case 50 --auto-user --reaction 0 0 \
        --headless --workers 12 --adaptive-xy --mu 0.15 1.6 --out demos/protocol_adaptive
    python3 policy/label_check.py demos/protocol_v1 demos/protocol_adaptive

Both flags default to the original behaviour, so running `protocol.py` without
them reproduces `demos/protocol_v1` exactly and `tests.py` is unchanged at 45/45.

### What the validation found, and it is not all good

8 kept of 12 attempts (67%) at `--mu 0.15 1.6`, `--texts S`.  Every drop was
**coverage/precision**, with protocol compliance 100% and in-band 99% -- i.e. the
operator followed the adaptive protocol perfectly and the pen still lagged off the
line.  That is the friction lag: at `mu = 1.6` the drag is 4.8 N and xy LOW
(500 N/m) trails the reference by 9.6 mm against a 3 mm tolerance, and friction
cannot be felt until the pen is already sliding.

Two consequences, one fixable and one not.

**Fixable -- selection bias.**  The drops are concentrated at high friction, so the
kept set is the low-friction half and the variation the change exists to add gets
filtered back out.  Cap the range where the STARTING level is already inside
tolerance: contact begins at MID, lag = `3 mu` mm there, so `mu <= 0.9` is safe.

    python3 protocol.py --texts S 7 "<star>" --per-case 50 --auto-user --reaction 0 0         --headless --workers 12 --adaptive-xy --mu 0.15 0.9 --drag-up 1.5 3.0         --out demos/protocol_adaptive

**Not fixable by a protocol tweak -- xy stiffness has no upper-side cost in this
task.**  Nothing in `sim.Criteria` penalises a stiff pen along the paper: the
lower bound is the friction lag, and there is no upper bound at all.  So a policy
could emit xy HIGH always, score full marks, and be wrong about the label.
`--adaptive-xy` makes the label INFORMATIVE (hard to predict without force); it
cannot make it TASK-RELEVANT.  `acc xy` on an adaptive dataset measures imitation
fidelity, not competence -- which is worth having, and worth not overclaiming.

The same is true in the other direction for z under the spring layout: `f_d` is
commanded, so soft holds the force at every tilt and soft is simply better.  An
interior optimum in writing exists only in the LEGACY layout, squeezed between
"too soft = no ink" and "too stiff = loses contact".  This is the writing twin of
what `../wiping/ROTATIONAL_STIFFNESS.md` found for K_R on a uniformly curved
board -- "a constant 0.3 beat the oracle" -- and the way out there was
`patchy_surface`: alternating regimes WITHIN one episode.  Writing has no such
alternation, and that, more than the lookup-table labels, is why its eight
structures tie.

**The protocol change itself is proposed, not adopted.**  `../wiping/CURVED_BOARD.md`
treats a protocol edit as "a change to an approved protocol and so not mine to
make", and the same applies here: the thresholds, the friction range and whether
to re-collect at all are yours to decide.  What is in the repo is the mechanism
behind a default-off flag, plus the audit that says whether it worked.

---

# The per-episode best constant: the question, closed

`policy/best_constant.py`, 6 stiffnesses x 24 episodes x 2 layouts, text 'S',
tilt +-15 deg (per-episode tilts 0-21 deg, against the demos' 5), k_t held at
2000 N/m so only k_n varies.  Raw results in `best_constant_{legacy,spring}.json`.

This exists because "variable stiffness cannot win" and "that oracle RULE is bad"
are different claims, and a hand-written oracle (`tol_deg = 0`) cannot separate
them.  So the rule is removed: run the SAME episodes at every stiffness and compare
`argmax_k mean_episodes` (the global best constant) against
`mean_episodes max_k` (what a policy knowing the episode perfectly would get while
still holding one stiffness).  The second is an upper bound on any within-episode
schedule too, and there is no rule in it to criticise.

    k_n     success  coverage   in_band  precision   contact N    peak N
                                                              legacy / spring
    200       1.000     1.000     0.959      1.000        3.42   10.83 / 7.26
    400       0.708     0.961     0.921      1.000        3.48    8.32 / 6.04
    700       0.333     0.877     0.819      1.000        3.95    7.50 / 6.34
   1200       0.083     0.789     0.718      1.000        4.82    8.22 / 7.87
   2000       0.000     0.715     0.585      0.995        6.25   10.43 / 11.44
   3500       0.000     0.661     0.473      0.981        8.64   14.65 / 17.47
   (success/coverage/in_band/precision shown for legacy; spring agrees to <0.05)

| metric | gap (legacy) | gap (spring) | episodes differing | argmax correlation |
|---|---|---|---|---|
| success | **+0.000** | **+0.000** | **0/24** | -- |
| coverage | **+0.000** | **+0.000** | **0/24** | -- |
| precision | **+0.000** | **+0.000** | **0/24** | -- |
| in_band | +0.014 | +0.014 | 20/24, 17/24 | all \|r\| <= 0.40 |

**THE ORACLE RULES ARE EXONERATED.**  Over a tilt range where the sizing rule moves
k_n's ceiling from ~1236 to ~300 N/m, not one of 24 episodes wants a different
stiffness from any other.  The rule-free upper bound EQUALS the global constant, so
no schedule of any kind could have won and `tol_deg = 0` was never the problem.
The only gap is 1.4 pp of `in_band` whose argmax correlates with nothing -- ladder
selection noise, not a signal a policy could collect.

Caveat against ourselves: 200 N/m is the ladder floor and it won every metric, so
the shared optimum is "<= 200", not "= 200".  That does not touch the 0/24.

## What the run did find

At the operating point the spring layout cuts peak force **10.83 -> 7.26 N**, i.e.
the margin against the 12 N tear limit goes 1.2 -> 4.7 N, about 4x.  Success was
already 100% at that stiffness, so the spring layout does not buy success here --
it buys SAFETY MARGIN AT THE STIFFNESS THE TASK WANTS, which is the kind of thing
that decides whether a real robot survives.

Both `peak_force` columns are non-monotone (interior minimum at 700 legacy, 400
spring), which locates the only two-sided cost in the task: the ENGAGEMENT
TRANSIENT, and it is not in the score function.

## Why tightening `tear_force` is not the next experiment

It would create an interior optimum in `success`, but the landing impact is set by
the descent speed and the commanded press, and neither varies with tilt, friction,
dz or letter height.  The optimum would sit at the same k_n in every episode and
0/24 would differ again.  A predictable experiment is not worth the cores.

## The one hidden variable that would move the per-episode optimum

**Environment compliance.**  Randomise the surface stiffness K_e -- paper on granite
vs paper on foam.  It moves the optimal k_n per episode through two independent
channels: the impact peak scales as `v sqrt(m K_e)`, and in contact the deflection
splits as `K_e / (K_e + k_n)` so the same k_n yields a different force.  And it is
invisible to a camera: two identical grey sheets.

Every environment tested so far -- writing, wiping (flat / patchy / attitude /
dome), peg insertion -- has a RIGID surface.  That is the untested axis, and it is
the "feel it, cannot see it" case the project set out to study.  It needs a
compliant canvas in the scene, which is a contained change.  If it also comes back
flat, the negative result is strong enough to be the finding rather than the
obstacle.

---

# Two bugs and one confound, from running the arms

Recorded because each one produces a result that looks valid.

## 1. A training path with no inference path (`*_sur`, 32/32 evaluations dead)

    ValueError: operands could not be broadcast together with shapes (1,10,3) (9,)

`surprise.py` fitted K fold models, cached their out-of-fold (mu, sigma) into the
npz, and **threw the models away**.  Cross-fitting requires that for the targets;
it leaves a policy that trains on nine channels and cannot be rolled out at all,
because at inference there is nothing to produce mu.  Training succeeded for all
32 checkpoints and every evaluation died on the first feature build.

Fixed with `--deploy-only`: one expectation model fitted on ALL episodes, saved
beside the npz, loaded by `surprise.Deploy` and called per frame in the rollout.
Three things that matter:

* the npz is NOT regenerated, because 32 policies were already fitted to those
  targets and re-rolling the folds would move them;
* the deploy model carries its own `Stats`, since the folds are fitted with
  `val_per_case=0` statistics while a policy checkpoint holds train-split ones;
* the nine channels are assembled in ONE function, `data.surprise_channels`,
  called by both the offline path and the rollout.  If those drift, a policy is
  fed a different input than it was fitted to and nothing raises.

The mismatch to state in any write-up: a training sample saw mu from a model
blind to its episode; at rollout mu comes from a model fitted on every
demonstration.  Evaluation episodes are unseen randomizations, so it is still
out-of-sample, but it is a different predictor.

Its in-sample R^2 is worth reporting next to the folds' out-of-fold numbers,
because the gap is the point:

| axis | out-of-fold | in-sample |
|---|---|---|
| tangential x | **-0.196** | 0.283 |
| tangential y | **0.002** | 0.264 |
| normal z | 0.867 | 0.896 |

The normal force is genuinely learned (0.867 -> 0.896, almost no gap).  The
tangential force is MEMORISED and generalises to nothing.  Without cross-fitting
the residual would have looked partly predictable and been quietly destroyed.

## 2. The spring decode divided by an unclamped network output (3/32 evaluations dead)

    numpy.linalg.LinAlgError: Eigenvalues did not converge

`x_d = x_ref + K^-1 f_d` used the stiffness the POLICY asked for.  `k` is `exp()`
of a network output and unbounded, while `Case1Controller.advance` projects K onto
`[k_lo, k_hi]`.  One outlier frame with `k -> 0` put the target at infinity, the
force followed, and the controller's `eigh` raised on the resulting NaN.

Clamping to the controller's own bounds before dividing is also the physically
honest decode: the spring that gets applied is the clamped one, so that is the K
the target must be placed against.  Verified at k = 1e-3 and 1e+6 (finite target,
K clipped to 100 / 4000).

The 29 evaluations that had SURVIVED were re-run rather than pooled with the
fixed ones.  Measured afterwards, the clamp changes the scores by well under a
seed's spread -- it only bites on frames outside the bounds -- but a comparison
cannot mix two decodes.

## 3. An inference-only change tested by retraining (the (d) schedule fix)

`d_lead` alters (d)'s sampling schedule and nothing in its loss, so retraining
with `D_LEAD=3` should have produced identical weights at each seed.  It did not:

    WEIGHTS_IDENTICAL_PER_SEED [False, False, False, False, False, False, False, False]

GPU training is not bit-reproducible across nodes and hardware.  So the paired
result -- **+2.3 +- 6.0 pp, t = 1.07, better 4/8, worse 2/8, tied 2/8** -- mixes
the schedule change with retraining noise and cannot test the fix at all.

`policy/rollout.py --d-lead N` now overrides the schedule on an EXISTING
checkpoint (`FlowPolicy.__init__`, which also sets `model.cfg` because
`CrossCond.sample` reads the module's own config), so one set of weights can be
run both ways.

**The general rule this implies for the whole study**: any config change that
only affects inference -- `d_lead`, `n_steps`, `n_refine`, `n_exec` -- must be
measured by overriding at evaluation time, never by retraining.  Retraining
nondeterminism is worth about +-6 pp here, which swamps every effect of that kind.

---

# The spring layout as a LEARNED policy: it fails, 32/32 seeds

The claim under test was the one measurement left open: the force invariance and
the 4x tear margin were both measured on the SCRIPTED writer, where `x_ref` is
exact by construction.  Four structures x 8 seeds, paired per seed, all 32
evaluations re-run with the clamped decode:

| structure | legacy | spring | paired diff | coverage | in_band | peak N |
|---|---|---|---|---|---|---|
| a ft_input | 93.8 +- 6.1 | 15.2 +- 3.6 | **-78.5 +- 6.7** (8/8 worse) | 97.0 -> 64.1 | 98.9 -> 96.5 | 4.2 -> 10.0 |
| b uni_dir | 86.2 +- 2.8 | 13.5 +- 3.6 | **-72.7 +- 2.9** (8/8 worse) | 95.8 -> 68.5 | 99.1 -> 98.0 | 4.3 -> 5.8 |
| c unified | 52.1 +- 13.2 | 6.9 +- 5.9 | **-45.2 +- 12.6** (8/8 worse) | 83.8 -> 53.8 | 97.9 -> 88.8 | 4.7 -> **25.1** |
| d cross_cond | 70.4 +- 14.1 | 8.3 +- 5.6 | **-62.1 +- 14.6** (8/8 worse) | 88.9 -> 59.1 | 97.2 -> 91.5 | 3.8 -> **19.0** |

## The mechanism, from the signature

`in_band` holds at 89-98% while coverage HALVES.  When the pen is down the force
is right; it is not down where the glyph is.  That is `x_ref` prediction error
arriving as position error, and it is the direct cost of the open-loop decode
chosen in `evaluate.WritingPolicyEnv.step`:

    x_ref is used OPEN LOOP on purpose.  Re-anchoring it on the measured tip
    every substep would hold the force at f_d whatever the surface did -- that
    is force control, and it would make K do nothing.

With a scripted `x_ref` that choice is free.  With a learned one it is not, and
nothing in the mechanism tests could have shown it: the scripted writer's `x_ref`
is the surface it just touched, exact to the millimetre.

Peak force degrades too -- 4.7 -> 25.1 N and 3.8 -> 19.0 N on the two weaker
structures, far past the 12 N tear limit.  Same cause: a bad `x_ref` places the
target deep below the surface and `f_d + K (x_ref - x)` then delivers whatever
that error implies.  So the layout loses force control as well as position once
`x_ref` is unreliable, not position alone.

## What this does to the measured claims

They stand, with their scope stated exactly:

* the applied force IS invariant to a stiffness error (3.0 N across a 6x K
  sweep, against legacy's 1.26 / 6.02 N) -- **given an accurate `x_ref`**;
* the tear margin at the operating point IS 4x better (7.26 vs 10.83 N peak)
  -- **given an accurate `x_ref`**.

Both reverse with a learned `x_ref`.  That is a limitation of the contribution,
not a tuning problem, and it has to be reported with the claims rather than
after them.

## The repair, if the layout is worth keeping

Anchor the POSITION on measurement and leave only the spring offset open loop:

    x_d = p_measured + delta_x_ref + K^-1 f_d

The offset `K^-1 f_d` stays open-loop, so the force remains K-independent and K
still governs the response to surface deviation -- the property the open-loop
choice was protecting.  What changes is that position error no longer
accumulates across a chunk.  This is a different algorithm from the one measured
above and needs its own run; it is not a reinterpretation of these numbers.

---

# All four arms, 8 seeds each, paired per seed

Full table in `policy/runs/COMPARISON_arms.md`
(`compare.py --family flow --arms "" nf sur spring`).  Success, %:

| structure | as trained | force input ZEROED | + surprise (9 ch) | spring action |
|---|---|---|---|---|
| a ft_input | 93.8 +- 6.1 | 87.3 +- 5.6 | 69.6 +- 8.9 | 15.2 +- 3.6 |
| b uni_dir | 86.2 +- 2.8 | 73.5 +- 6.9 | 61.5 +- 14.4 | 13.5 +- 3.6 |
| c unified | 52.1 +- 13.2 | 39.0 +- 13.2 | 40.4 +- 11.9 | 6.9 +- 5.9 |
| d cross_cond | 70.4 +- 14.1 | 64.8 +- 13.5 | 56.9 +- 11.4 | 8.3 +- 5.6 |

## 1. The structure ranking is NOT about force conditioning

a > b > d > c holds with the force input zeroed, and the a-c gap WIDENS from
41.7 to 48.3 pp.  Paired, force removed costs -6.5 / -12.7 / -13.1 / -5.6 pp.
So force is informative, but it is not what separates the four structures: the
ranking in `policy/runs/COMPARISON.md` measures optimisation difficulty.  The
clue was there already -- K error is 0.0-0.7% for every structure and the
ranking tracks `pos1_mm` and the action loss.

**What force IS for here, measured**: removing it takes `ft_input`'s peak force
from 4.2 to 15.1 N, with 5 of 8 seeds past the 12 N tear limit and one at 45.6 N.
The policy can see where the paper is but only force tells it WHEN it touched,
so without it the landing is uncontrolled.  That is the "phase" role of force
(before / during / after contact), not evidence and not modality.

## 2. Force as surprise HURTS, and that is a verdict on the implementation

| | force removed | surprise added |
|---|---|---|
| a | -6.5 | **-24.2** (t = -6.16, 8/8 worse) |
| b | -12.7 | **-24.8** (t = -4.89, 8/8 worse) |
| c | -13.1 | -11.7 (t = -2.85, 6/8) |
| d | -5.6 | -13.5 (t = -2.83, 7/8) |

For a and b, nine channels carrying the measured force PLUS an expectation PLUS
a residual are worse than three channels AND worse than zero.  Extra information
cannot do that; something in the input is actively misleading.  Measured cause:

    RMS(mu_deploy - mu_fold) is 0.52-0.77 of sigma, and the z-channel the policy
    consumes differs by RMS 2.0 on the normal axis against a training-time
    |z| of 0.66.

The policy was fitted against fold-model mu and rolled out against deploy-model
mu.  It is being fed a different variable than it learned on.  `in_band` and peak
force are untouched (98.9 -> 99.1, 4.2 -> 4.8), so this is an input-distribution
failure, not a force-regulation one.

So: **"feeding the residual helps" remains UNTESTED.**  The out-of-fold R^2
decomposition (0.867 normal, ~0 tangential) is unaffected -- no policy is involved
in it.  The repair is to make the predictor identical on both sides: ENSEMBLE the
K fold models at training and at rollout, which keeps "never saw this episode" at
training time while removing the mismatch.  That is a change to `surprise.py`,
not a hyperparameter.

## 3. The (d) schedule fix is null, measured cleanly

Same 8 checkpoints, sync clock vs wrench 3 steps ahead
(`rollout.py --d-lead 3`):

    paired +1.0 +- 6.4 pp, t = +0.46, better 5/8, worse 2/8

The retrained comparison gave +2.3 +- 6.0 (t = +1.07) -- same conclusion, smaller
effect, but it could have gone either way, which is why it had to be redone.
The DIAGNOSIS stands (sampling on lamA = lamF is a measure-zero slice of (d)'s
training distribution); fixing it does not help on this task.

---

# Round two: fixing the two negatives that were mine

## Surprise, repaired: half the damage was the sigma division

Two changes, both measured before retraining anything:

| | change | measured effect on the train/deploy gap |
|---|---|---|
| residual units | `z = (F-mu)/sigma` -> **`F-mu` in newtons** | perturbation falls from ~2.5-3x the signal (z-shift 1.69-2.00 vs training \|z\| 0.66) to **0.33-0.47x** |
| deploy predictor | all-data single fit -> **ensemble of the K folds** | RMS(mu) disagreement only 10-20% lower |

The units change did the work; the ensemble barely moved it.  Retrained 32
policies, 8 seeds, paired against baseline:

| structure | surz (standardized z) | **sur (newtons + ensemble)** |
|---|---|---|
| a ft_input | -24.2 +- 11.1 | **-12.9 +- 9.6** (t = -3.80, better 1/8) |
| b uni_dir | -24.8 +- 14.3 | **-15.6 +- 7.9** (t = -5.62, better 0/8) |
| c unified | -11.7 +- 11.6 | **-10.6 +- 9.8** |
| d cross_cond | -13.5 +- 13.6 | **-2.1 +- 7.3** |

**About half the -24 pp was my bug.  The remaining ~-13 pp is unresolved** and
this experiment cannot resolve it: with 0.33-0.47 sigma of train/deploy
perturbation still present, "leftover mismatch" and "the extra channels genuinely
cost accuracy" predict the same thing.  That ambiguity was stated before the run,
not after it.

What would separate them: train on channels built from the DEPLOY predictor
itself (in-sample mu, no cross-fitting on the policy's input).  Zero mismatch by
construction.  Cross-fitting's job here is to make the REPORTED R^2 honest, and
that is measured separately, so spending it on the input is acceptable.

### And the comparison that makes the encoding look worst

Paired against the NO-FORCE arm rather than baseline:

| structure | sur vs nf | baseline vs nf |
|---|---|---|
| a ft_input | **-6.5 +- 10.0** (3/8) | +6.5 +- 9.1 (6/8) |
| b uni_dir | **-2.9 +- 11.4** (3/8) | +12.7 +- 8.4 (8/8) |
| c unified | +2.5 +- 21.7 (3/8) | +13.1 +- 21.4 (6/8) |
| d cross_cond | +3.5 +- 10.1 (5/8) | +5.6 +- 11.7 (5/8) |

The plain 3-channel input beats no-force on all four structures.  The 9-channel
encoding does not -- on `ft_input` it is **worse than having no force at all**.
A nine-channel input losing to a zero-channel input is evidence about the
encoding, not about force.

### Within one architecture, the cleanest result of the day

Same structure, same training, only the force input differs:

    ft_input   3 ch plain   93.8
               0 ch          87.3
               9 ch sur      80.8
               9 ch surz     69.6

Monotone in how much was done to the force channel.  But note the scope: `a` with
the plain input is at **93.8%, near ceiling**, so on this task no elaboration can
WIN, only lose.  "Nothing I added beat feeding it raw" is the honest claim;
"simple force input is best in general" needs a task where plain is not at the
ceiling.

## spring_rel: my repair was wrong, and the gate caught it

The proposed fix was `x_d = p_measured + delta_x_ref + K^-1 f_d`.  Gate, seed 0,
3 episodes per text:

| | success | coverage | in_band | peak N |
|---|---|---|---|---|
| legacy | 80.0% | 91.0% | 96.6% | 3.82 |
| spring | 13.3% | 61.1% | 89.4% | 29.20 |
| **spring_rel** | **0.0%** | 53.7% | **44.6%** | **76.73** |

Worse than the layout it was meant to repair.  Force climbs to 74.8 N by frame 94
of 145, 28 frames past the tear limit.

**The mechanism is accumulation, from a semantics error I introduced.**  The label
is `delta_x_ref = x_ref(t+1+j) - tcp(t)`, a displacement from the tip AT THE
PLANNING INSTANT.  The decode re-anchors on the CURRENT measured tip every policy
frame and adds delta again.  In free space the anchor advances with the tip and
nothing accumulates; IN CONTACT THE TIP IS BLOCKED, so the anchor stops advancing
while the decode keeps adding the downward component -- a few mm deeper per frame,
and `f_d + K (x_ref - x)` delivers the result.  The original spring layout does
not have this because `x_d(t)` already contains the accumulated offset, so its
labels are self-consistent.

So: **re-anchoring on measurement is incompatible with a displacement label
whenever the tool can be blocked.**  A correct repair has to stop achieved
penetration being re-added -- anchor the in-plane components on measurement while
keeping the normal component on `x_d`, or label `x_ref` absolutely in the paper
frame.  That is a different design, not a flag, and it is NOT what the earlier
section of this file proposed.

31 queued evaluations (~6 GPU-hours) were cancelled rather than left to confirm a
deterministic mechanism failure.  The 32 trainings are reusable.

What caught it was the gate's NUMBERS, not its pass/fail: it did not crash.  A
crash-only gate would have green-lit six hours of 0% runs.
