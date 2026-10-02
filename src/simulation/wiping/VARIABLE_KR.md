# Does variable rotational stiffness beat a constant? — the gate, and its answer

The question the policy work rests on: a wiping pad on a surface whose curvature
changes from place to place, is a K_R the robot CHOOSES better than the best K_R
we could ship?  An estimator cannot beat the oracle it approximates, so the gate
is oracle-vs-best-constant and nothing is collected or trained until it passes.

Criterion, fixed before the run: the oracle must clear the best constant by
**5 points of coverage AND twice the seed spread**.

## The answer

Corrugated board (the only family inside the window, below), 8 seeds, coverage
not saturated, zero episodes dropped on the force band, `corr_gate.py`:

| | coverage | mis_flank | yld_plateau |
|---|---|---|---|
| constant K_R 60 | 38.2% ± 0.8 | 11.8° | 0.1° |
| constant K_R 3 | 51.7% ± 1.5 | 10.9° | 0.7° |
| constant K_R 1 | 69.9% ± 1.0 | 9.0° | 1.5° |
| **best constant, K_R 0.3** | **73.6% ± 1.8** | 3.6° | 3.8° |
| best constant PER EPISODE | 73.6% | | |
| **within-episode ORACLE** | **76.1% ± 1.2** | 4.6° | 2.1° |

**+2.5 pp, about 1.4 seed sd. Below both halves of the criterion.**

1. **The per-episode gap is +0.0 pp, exactly, for the third time.**  It was 0.000
   on the dome at three (pad, radius) cells and it is 0.0 here.  A policy handed
   one latent per episode has nothing to encode: the hidden variables do not move
   the best stiffness.  **An RMA-shaped design is ruled out by this line alone**,
   independent of everything else in this note -- whatever a schedule can win has
   to be won WITHIN an episode.
2. **The oracle's schedule does work, per band.**  It sits near the soft constant
   on the flanks (4.6 against 3.6°) and near the stiff one on the plateaus (2.1
   against 3.8° of yield) -- a combination no constant reaches, since stiff is
   11.8° on the flanks and soft is 3.8° on the plateaus.  The mechanism is real.
   It converts into 2.5 pp.

## Why it loses, in three measured steps

**Contact does the aligning, not the spring.**  At K_R 0.3 the pad sheds 8.6° of
a 12° flank demand within the 1.5 s it spends on the next plateau.  A pad that
offers no resistance is laid flat by the surface itself, and the contact
stiffness doing it is orders above any K_R we would command.  So wherever FLUSH
is the right attitude, "as soft as stable" is already the answer and it is a
constant.  This is what every negative result in this directory has been
measuring.

**The erasure model discounts small misalignment.**  The contact patch is set by
`squash = pen + pad_give`, so a tilt whose height change across the pad is small
against `pad_give` barely narrows the patch.  `atan(give / 2r)` = atan(4/30) =
7.6° at the 15 mm pad -- larger than the soft wrist's entire error, so the thing
the schedule fixes is nearly free to get wrong.

**And when the metric IS made sensitive, the schedule loses outright.**  The
prediction was that a harder pad would expose the oracle's advantage.  It
reversed it, in all six cells (`corr_give.sbatch`, single seed):

| pad_give | blind | erase_work | best constant | oracle |
|---|---|---|---|---|
| 1 mm | 1.9° | 0.06 / 0.10 / 0.14 | 96.0 / 71.3 / 38.3 | 92.7 / 64.4 / 33.9 |
| 2 mm | 3.8° | 0.06 / 0.10 / 0.14 | 100.0 / 98.1 / 76.8 | 99.8 / 95.0 / 66.3 |

The oracle is 1.0° worse on the flanks, and the flanks are 63% of the path.  Once
coverage can see 2-4° the flank cost dominates the plateau gain.  The oracle is
worse there because it stiffens on the plateau (correctly) and then arrives at
the next flank stiff: re-softening is rate-limited (`Ur` through the tank) and
the rotational mode has to settle again, so the schedule pays a transition at
every flank entry that the always-soft constant never pays.  **"The cost of
switching eats the gain", measured.**

## The window, and the boundary that was inverted

The pad's face is a PLANE.  Fit one over the footprint and the surface splits
into the TILT of that plane, which a rotation can take out, and the RESIDUAL to
it, which no stiffness can at any value.  `pad_give` has to absorb the residual.
So a board only asks the wrist for something when

    residual  <=  pad_give  <  2 r tan(tilt)

**This inverts the boundary `dome_boundary.sbatch` was built on.**  That run read
`sagitta > give` as the regime adaptation should pay in; it is the regime where
rotation is POWERLESS, because the sagitta IS the residual.  A constant winning
all five of those cells is what the corrected condition predicts.  For the same
reason `oracle_kr.py` steered stiffness by the normal SWING across the pad --
also the part rotation cannot fix.  `corr_gate.oracle` uses the fitted plane's
tilt instead, `K_R = f_n r / theta`, which also goes stiff on a plateau for free
(theta -> 0 clips to `kr_hi`), the half a swing rule cannot express.

`gate_feasible.py` screens geometries against the full set, in numpy, before any
of them costs CPU hours:

| | condition | what it is for |
|---|---|---|
| A1 | `2 r tan(theta_flank) > give` | the demand survives the felt |
| A2 | `residual <= give` | a rotation is what answers it |
| A3 | `2 r tan(theta_plateau) <= give` | the other band asks for nothing |
| B | `t90 < flank / v` | the wrist can acquire the tilt while on the flank |
| C | >= 4 band crossings | an episode visits several facets |
| D | `plateau / v < 2 t90` | **and cannot recover -- so softness COSTS** |

with `t90 = 2.3 / (zeta sqrt(K_R* / Lambda_r))` and `K_R* = f_n r / theta`.

**Smooth bumps and domes fail A1 and A2 simultaneously** -- a Gaussian bump about
the pad's own size has its sagitta for a residual -- so 0 of 540 cells of the
`PatchySurface` family pass.  What passes is a locally FLAT but tilted patch:
`CorrugatedSurface`, trapezoidal, 734 of 2304 cells.  **The shipped patchy board
also failed B by 2-3x** (0.17 m bands at 0.12 m/s is 1.4 s against a 3.1 s
settling time) and C (a 0.15 m glyph inside a 0.34 m period sees one transition),
which is why no schedule could be tracked on it.

D is the condition every earlier environment was missing.  Without it the soft
wrist settles flush again before the next band's marks are reached, softness is
free, and a low constant wins.

## What was ruled out on the way

* **Metric saturation.**  `erase_work` is a CLIFF, not a knob: 0.12 gives 78-98%
  and 0.24 gives 3-4%, because each mark is passed a fixed number of times so the
  wear it collects is concentrated.  The gate sits at 0.16, on the knee, where
  coverage runs 38-76% across conditions.
* **Geometry.**  Plateau/flank 30/100 and 50/80 mm, slope 12°, both inside the
  window; the gap is -1.8 to +5.4 pp across six (geometry, erase_work) cells,
  and the one +5.4 was the seed -- 8 seeds put it at +2.5.
* **Pad hardness.**  1, 2 and 4 mm of give.  Harder makes it worse, above.
* **Force as a confound.**  The normal-force loop holds 4 N; in-band is 1.00 in
  every condition and no episode was dropped, so nothing here is the oracle
  pressing harder.
* **The rule, as opposed to the ceiling.**  Separately, on the dome: 270 searched
  5-knot swing->K_R rules with every constant NESTED in the search space beat the
  best constant by 0.002, -0.002 and 0.012 N of force rmse, on the fit seed.

## Where this leaves it

The direction is dead for a FLAT pad on a surface where lying flush is always
correct, and that is not a tuning failure -- it is step 1 above.  What is left is
a task where flush is NOT the right attitude, so that the contact moment pushes
AWAY from what the task wants and stiffness is the only thing that can resist it:
a squeegee, a scraper, a blade, a chamfer, any tool with an attack angle.  Then
the stiffness that holds the angle depends on the local geometry, there is a cost
on both sides, and the soft-is-free argument does not apply.  It needs the
contact and erasure models changed -- both assume the pad's face lies on the
surface -- so it is a new task, not a parameter.

The negative is also publishable as it stands: two inequalities bracketing the
regime where rotational VIC can pay at all, four surface families measured
against them, and the estimability already shown (local swing regressed to 1.07°
rmse against 2.40° for predicting the mean).  It answers the question a reviewer
asks first, which is why not just set the stiffness low and leave it.

## Conditions to report with any of this

* Scripted raster, not a policy.  8 seeds with the glyph moved per seed; seed
  spread 0.5-1.8 pp here, against the ±10-17% the policy study showed.
* Corrugation 30/100 mm at 12°, 15 mm pad, 0.02 m/s, 4 N, `erase_work` 0.16,
  `pad_give` 4 mm.  The speed is 2.5x slower than the shipped 0.05 m/s and B
  requires it.
* `Lambda_r` = 0.38 kg m², the worst of the measured [0.005, 0.18, 0.38], so B
  is screened conservatively.
* No six-axis F/T in these recordings: the contact moment reaches a model only
  through `tau`.
