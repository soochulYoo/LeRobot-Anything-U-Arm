# Wiping on a curved board — design note

## What step 1 found

Step 1 of the previous version of this note was: before reworking the canvas,
put OUR pad and OUR controller on the `../curved` slab, drive one straight
traverse, and measure pad misalignment against K_R.  `curved_probe.py` is that
experiment.  It answered the question it was built for and it overturned the
premise that motivated the rest of this note.

**The premise was wrong.**  A flat tilt exercises K_R just as well as a curved
slab does.  What was stopping it were two bugs and one mis-chosen decade of
K_R, none of them anything to do with the surface:

1. **The eraser agent had no SRDF, so no self-collision was filtered.**
   ManiSkill looks for `<stem>.srdf` beside the URDF and, finding none,
   disables nothing.  `panda_hand` and `panda_link7` are rigidly connected
   through the two fixed joints at `panda_link8` but are not parent and child,
   so PhysX does not filter them, and their collision meshes overlap by 23 mm
   by construction.  The solver can never resolve that contact: it pinned the
   wrist.  `../writing/scene.py` writes a pen SRDF for exactly this reason; the
   eraser was copied without it.  Fixed in `wipe_scene.py:_ensure_assets`.

   Everything the pad's ORIENTATION did in this task was measured in that
   state, including "a flat tilt cannot exercise K_R".  What the pin did not
   affect was translation on a flat board, which is why erasing still worked
   and `smoke.py` still passed: 100% erased, 98% in band, before and after.

2. **A sharp-edged box pad cannot be dragged over a triangle mesh.**  PhysX
   generates contacts against the mesh's internal edges with normals pointing
   sideways, so the pad meets what is effectively a vertical wall.  On a FLAT
   mesh slab, where the only correct answer is a smooth slide, the box pad
   stalled dead: 17 N mean and a 74 N peak against a 2.4 N command, the
   traverse stopping 90 mm short while the lateral spring loaded up.  Box-on-box
   contact is analytic and never showed this, which is why the flat CANVAS
   never revealed it and the curved board does — a curved board has to be a
   mesh.  The pad's rim is now rounded by 2 mm (`wipe_scene.PAD_ROUND`), outer
   dimensions and `ERASER_R` unchanged, so the erasure model is untouched.

3. **K_R = 60 vs 3 is rigid vs rigid for this tool.**  A pad of half-width r
   pressing with f can generate at most about f·r of contact moment, so an
   impedance K_R yields f·r/K_R and no more.  Our eraser is r = 15 mm at
   f ≈ 3.5 N: 0.05 Nm, so 8 deg of yield needs K_R ≲ 0.3 Nm/rad.
   `../curved/wipe.py` welded a 25 mm pad and pressed with 8 N — 0.2 Nm, four
   times more — which is why K_R = 3 was already at its knee and is rigid here.
   The sweep below runs down to 0.03 and prints f·r/K_R beside the measurement.

### The sweep

Curved slab, one straight traverse, commanded orientation vertical throughout,
translation given the surface height as an oracle so that nothing but K_R
stands between the pad and the surface.  `ask` is the rotation that would lie
flush, `yield` is how far the pad moved off its vertical command, `pred` is
f·r/K_R, `align` is what is left between the pad's face and the true local
normal, `left` is align/ask.  Force stayed in the 1-6 N band 100% of the time
in every row; no torque saturation and no tank gating anywhere (α = 1.00).

    K_R      ask   yield   pred   align   left        K_R    align   left
     60      8.9     0.1    0.1     8.8   0.99         60      8.0   1.00
     10      8.9     0.4    0.3     8.6   0.97         10      7.8   0.97
      3      8.9     1.3    1.0     7.9   0.89          3      7.2   0.89
      1      8.9     3.9    2.9     5.8   0.65          1      5.1   0.64
    0.3      8.9     7.1    8.9     4.0   0.45        0.3      4.1   0.51
    0.1      8.9     8.9   28.0     5.1   0.57        0.1      5.1   0.64
   0.03      8.9     9.8   98.0     6.4   0.72       0.03      6.4   0.80
    curved: amp 10 mm, sigma 35 mm,            flat, tilted 8 deg (--amp 0
    seed 3, 8.9 deg of tilt asked,             --tilt 8), 8.0 deg asked,
    3.9 deg of swing across the pad            no swing at all

Three things to read off it:

* **K_R separates, and the curve is the one the moment argument predicts.**
  Measured yield tracks f·r/K_R to within 30% all the way down the stiff side
  (0.1/0.1, 0.4/0.3, 1.3/1.0, 3.9/2.9).  The old K_R = 60 vs 3 comparison would
  have shown 8.8 vs 7.9 deg even on the curved board — 0.9 deg, which is the
  1.1 deg the flat board gave.  The pair was the problem, not the board.
* **The flat tilt behaves almost identically.** 8.0 → 4.1 deg against the
  curved slab's 8.8 → 4.0.  Rotational compliance does NOT need a turning
  normal to show up; it needs a demand and a K_R in the right decade.  The
  earlier claim in this note, that a constant normal gives K_R nothing to do,
  is false for a tool the controller holds at the WRONG constant orientation.
* **Compliance stops at about half the demand — because it runs out of
  moment, not because it runs out of compliance.**  As the pad rotates toward
  flush its pressure centroid moves in from the rim, so the moment that was
  turning it collapses: f·r is an upper bound reached only at full edge
  contact.  Pressing harder buys the rest of it.  At 8 mm of commanded depth
  instead of 4 (5.9 N instead of 3.6, same everything else):

      K_R    ask   yield   pred   align   left    f      in band
        3    8.9     2.0    1.7     7.3   0.82   5.88       58%
        1    8.9     5.6    4.8     3.9   0.44   5.64       81%
      0.3    8.9     7.6   15.7     1.8   0.20   5.48       80%

  1.8 deg left of a 8.9 deg demand: the pad very nearly lies flush.  The knob
  is the ratio K_R/(f·r), and the two ways to turn it are equivalent on the
  stiff side — but force is bounded by the task (the band is 1-6 N, and 5.9 N
  already spends it: 58-81% in band against 100% at 3.6 N), so K_R is the knob
  with room in it.  Below K_R ≈ 0.3 at 3.6 N the pad instead yields MORE than
  it was asked (9.8 deg of an 8.9 deg demand at K_R = 0.03) and wanders; that
  end is set by the rotational damping, not by the moment.

### What was ruled out

* **The tank, the null-space term and torque saturation: all three, directly.**
  `--no-tank --null-kp 0 --null-kd 0 --tau-scale 100` gives align 8.8 / 5.9 /
  3.8 deg at K_R = 60 / 1 / 0.3 against the stock controller's 8.8 / 5.8 / 4.0.
  Nothing.  Consistent with the logs: α = 1.00 and 0% saturation in every run,
  so there was nothing for the ablation to remove.
* **`Case1Gains.wrist_inertia` is wrong but not the blocker.**  The true
  operational-space rotational inertia at the pad is [0.005, 0.18, 0.38] kg m²,
  against the 0.01 the controller assumes, so `Dr = 2ζ√(I·K_R)` puts the real
  rotational damping ratio at 0.13-0.19 on two of three axes instead of 0.8 --
  the same defect `../curved/wipe.py:op_space_rot_inertia` was written to fix,
  with a MATRIX Dr.  Re-running the sweep with the inertia corrected
  (`--wrist-inertia 0.2`, true ζ 0.58-0.84) moves align by at most 1 deg and
  changes no conclusion: 8.8 / 8.6 / 8.0 / 6.7 / 4.9 / 4.7 / 5.2 down the same
  K_R column.  It does soften the degradation below the knee, which is what a
  correctly damped wrist should do.  Worth fixing in `../writing/controller.py`
  on its own merits; not worth blaming for anything here.

## Why a curved board, now that the premise is corrected

A flat tilt and a curved slab ask the pad for the same thing — ONE static
rotation — and the sweep shows they get the same answer.  What curvature adds
is what a single rotation cannot satisfy:

* **The normal turns as the pad travels.**  Along the traverse used above the
  along-path slope runs -7 deg → 0 → +9.5: the pad has to keep rotating, so
  the rotational loop's SPEED matters and not only its stiffness.  On a tilt,
  one yield at the start is the whole story.
* **The normal swings across the pad itself** — 3.9 deg mean, 5.4 p95 across
  30 mm here.  No rigid flat pad can lie flush on that at any K_R, which is the
  regime where pad compressibility (`WipeCriteria.pad_give`) and the erasure
  model's contact-patch share start to carry real weight.
* **Translation stops being free.**  With the height oracle removed
  (`--press-mode flat`, i.e. what the task actually commands) the same curved
  slab holds the force in band only 55-63% of the time, peaks at 7 N against a
  2.4 N command, and loses contact 20% of the traverse at K_R = 0.3, while the
  flat board stays at 100% in band.  That is a separate, real problem that the
  curved rework is what creates, and it is the one the policy work would face.

So the rework is still worth doing — but as a HARDER task, not as the thing
that makes K_R measurable.  K_R is measurable on a tilt, today.

## The curved board as built, and what it took to measure anything on it

It is a subclass, not a flag: `wipe_scene.CurvedWipingEnv` (env id
`TeleopWipingCurved-v1`), `wipe_scene.CurvedFrame`, `wipe_sim.CurvedWipingSim`.
Nothing in the flat path runs any differently -- `smoke.py` on the flat board
still gives 100% erased, 98.3% in band, 9.1 N peak, 6.2 s, to the digit.

* **`CurvedFrame` keeps `CanvasFrame`'s contract.**  `(u, v)` are still the
  board's in-plane coordinates and the third coordinate is still "off the
  board", only measured from the LOCAL SURFACE instead of the plane.  So
  `to_world(np.r_[uv, -penetration])` still means "press in by this much", and
  a driver written for the flat board presses correctly on the curve without
  being told.  `height(0,0) = 0`, so the board's centre, the hover height and
  every episode's start pose are unchanged.
* **The slab is kinematic and nonconvex.**  Shape is part of the SCENE (baked
  into a collision mesh at reconfigure, `CurvedWipingEnv.SURF`), pose is part
  of the EPISODE (height error and tilt still re-randomised every reset, as on
  the paper).  Several shapes in one run means several envs.
* **`self.canvas` is rebound to the slab**, with the flat box parked below the
  floor as `self.flat_canvas`.  Every reader of the board -- the contact force
  in `writing/sim.py:step`, the collision check on the other links, the
  friction material -- goes through that name, so rebinding it is what makes
  all of them work on the curve untouched.
* **The erasure model needed one term.**  The pad's face is a plane and the
  board is not, so a mark sitting `dh` higher than the contact point is pressed
  `dh` harder: `pen = dh - uvh[2] - d @ (-a[:2]/a[2])`, with `dh` from
  `wipe_sim.board_height`, which is identically zero for a flat `CanvasFrame`.
  That is the whole of it, and it is exact in the surface rather than
  first-order in its slope.
* **The board asks for something now.**  Over the raster area of a 35 mm glyph
  at the default offset: 9.0 deg of tilt on average, 15.4 worst case, and 5.5
  deg of normal swing across the pad's own 30 mm.  No rigid flat pad can lie
  flush on that, which is the point.

Two further things had to be right before a K_R comparison on it meant
anything, and `record_rot.py` had neither:

**The FORCE has to be held, not the depth.**  At a fixed penetration how much
force the pad makes depends on how it is lying: 2.51 N at K_R = 60 against
3.73 N at K_R = 0.3, a 1.5x difference from rotational stiffness alone.
Erasure is work, so the compliant pad then erases faster for a reason that has
nothing to do with lying flush -- with a hard glyph it read as 58% against
100%, a force effect reported under a rotational name.  `smoke.py` now carries
one integrator along the local normal (force control along it, position control
across it, no outer position spring in that direction because it would split
the target between the two springs).  Held at 3 N, every condition below
measures 3.06-3.17 N.

**The rotational DAMPING has to be real.**  `Dr = 2 zeta sqrt(I K_R)` with a
guessed `I` gives a true damping ratio of `zeta sqrt(I_guess/Lambda)`, which is
independent of K_R and equals 0.13-0.19 for the stock 0.01 against the measured
[0.005, 0.18, 0.38] kg m^2 -- a Q of about 4.  A raster REVERSES, every 1.06 s
here, and that rings the rotational mode.  On the curved board at K_R = 1 the
task collapsed outright: 0% of the glyph off and 10.0 deg of misalignment,
WORSE than holding the pad rigid.  With the inertia corrected the same
condition erases 66%.  The single smooth traverse in `curved_probe.py` had
missed this entirely -- it has no reversals -- which is why both measurements
are kept.

### What rotational compliance buys, with both of those right

Force held at 3 N, the rotational damping corrected, `erase_work` 0.12 N m (the default
0.040 lets the raster cover the glyph twice over, so both conditions finish at
100% and the question cannot be asked).  `mis` is the pad's face against the
true local normal; `yield` is how far it moved off the vertical it was
commanded; the curved board asks for 9.1 deg and the flat one for 0.

               curved board                    flat board
    K_R    erased   mis  yield          erased   mis  yield
     60     50.0    8.9    0.1           95.2    0.1    0.1
      3     53.2    8.3    1.6           96.8    1.2    1.2
      1     66.1    7.5    3.0           96.8    2.7    2.7
    0.3     83.9    5.5    6.0           98.4    4.4    4.4

* **On the curved board compliance erases 1.7x more of the glyph** (50 -> 84%)
  and takes the misalignment from 8.9 to 5.5 deg, monotonically in K_R.  That
  is the result the task was built to show, and it took the SRDF fix, the
  rounded rim, the force loop and the damping fix to see it.
* **On the flat board it is nearly free but buys nothing** -- 95 -> 98% erased,
  within the noise of this measure.  Note that `mis` and `yield` are equal
  there to the decimal: a flat board asks for zero, so every degree of
  misalignment on it IS wander.  About 4 deg of wander costs nothing; the 7.5
  deg the under-damped wrist produced cost a third of the glyph.
(The table was measured with `wrist_inertia = 0.2`, the scalar band-aid that
preceded the matrix Dr now in the controller.  The probe sweep re-run with the
real fix agrees to within 0.2 deg at every K_R, so the table stands.)

* **Which means K_R is not a knob to turn down.**  Soft is right when the board
  asks and wrong when it does not, and the task's own board is randomly tilted
  per episode, so the stiffness that erases the most is a function of something
  the robot would have to sense.  That is a variable-impedance problem, not a
  tuning one -- which is the whole premise of the cascade work this task feeds.

## Where the planar assumption was

DONE, all of it inside `wiping/` so that `writing/` was not touched: the
planar assumption was load-bearing in four places, and each got a
surface-aware version in a subclass rather than an edit.

| where | flat | curved |
|---|---|---|
| `place_canvas` | a box actor, pose from `CanvasFrame` | `CurvedWipingEnv.place_canvas`: kinematic nonconvex mesh, and it builds the `CurvedFrame` |
| `_dot_pose` | dot on a plane at `(u, v, lift)` | `CurvedWipingEnv._dot_pose`: `lift` above the local surface, standing on the local normal |
| `CanvasFrame.to_canvas` | one global `(u, v, h)` frame | `CurvedFrame`: `(u, v)` unchanged, `h` becomes height above the local surface |
| `wipe_sim.on_contact` | one fixed normal under the pad | the `dh` term above: exact in the surface, not just in its slope |

The erasure model transferred as predicted: one added term, no restructuring.
What the note did NOT predict is that the pad geometry and the driver both had
to change too -- a mesh board needs the rounded rim, and a constant-height
command loses the force band on curvature (55-63% in band against 100%), which
is what the normal-force loop is for.

## Order to do it in

1. ~~Confirm the premise~~ — done, premise corrected.  `curved_probe.py` stays
   as the regression: it is the cheapest thing in this directory that can tell
   a pinned wrist from a compliant one.
2. ~~Curved board behind a flag~~ — done, as a subclass.
3. ~~A normal-force loop~~ — done, in `smoke.py`, and it turned out to be
   required rather than nice to have.
4. ~~Marks on the surface, with per-mark local normals~~ — done.
5. ~~Surface-aware `on_contact`~~ — done, the `dh` term.
6. ~~Fix `Dr` properly~~ — done, in `../writing/controller.py`.  Dr is now the
   matrix `zeta (Lambda_r^1/2 K_R^1/2 + K_R^1/2 Lambda_r^1/2)` with Lambda_r
   the rotational block of the operational-space inertia, in the body frame,
   recomputed alongside the translational one.  The damping ratio is 0.8 on
   all three axes by construction where it had been 0.13-0.19 on two of them.
   `Case1Gains.wrist_inertia` is now None by default and a number restores the
   old guessed scalar, so the difference stays reproducible: a 10 deg step in
   the orientation reference overshoots by 0.37 deg fixed and 4.12 deg
   guessed, which `../writing/tests.py` now checks in both directions.
   The writing suite is 35/35 with the change.
7. ~~A K_R the robot chooses~~ — done.  `Case1Proposal.Ur` is a rotational
   stiffness RATE in the same slot as `Up`: metered into `p_prop` as
   `U_r tr(I - R_d^T R)` (= `U_r 2(1 - cos theta)`, nonnegative when
   stiffening, the rotational copy of `1/2 p_de^T U_p p_de`), gated by the same
   alpha, bounded by `kr_lo`/`kr_hi`.  `TeleopSession.step` carries it as a
   target, so an operator sets it exactly as they set K_p.  Tested: the tank
   pays `dkr tr(I - R_d^T R)`, the gate scales the rate by alpha, and K_R
   clips at the bound.
8. ~~Collect demonstrations under the protocol~~ — done, 150 per board, and
   it has something to say about the table (above).  The `--kr-land LOW` A/B
   is run and says the `pre-contact` cell should be LOW.
9. Per-episode SHAPE randomisation, if the policy work needs it: the height
   field is baked into a collision mesh, so a new shape is a reconfigure.
   Several envs, or several parked slabs, or accept one shape per worker.

## Collecting demonstrations

`protocol.py` is `../writing/protocol.py` with a third axis group.  The
machinery underneath is the writer's, unchanged: a wiper IS the synthetic
writer handed a raster instead of a glyph (`wipe_teleop.SyntheticWiper`), which
means the visual loop that keeps the tool on its line, the haptic loop that
presses until the felt force matches the intent, the master, the delays and the
phase machine all transfer.  Measured with none of it modified: 100% erased,
94-95% of pad-down time in the force band, 3.6-4.9 N peak -- its own soft
landing, against the 9.1 N of the scripted raster in `smoke.py`.

Two things follow from using the operator rather than the script:

* **It finds the surface by feel.**  `smoke.py` is handed `frame.to_world` and
  therefore the board's shape; the haptic loop is not.  On the curved board
  that is the difference between a demonstration a policy could imitate from
  force and vision and one it could not.
* **It chooses K_R**, which is the decision this whole task exists to teach.

The protocol, per raster row, with the measurements that set it in the module
docstring:

    phase          xy     z      K_R
    APPROACH       mid    HIGH   HIGH     the pad travels to the row's start
    PRE-CONTACT    LOW    mid    HIGH     it hovers, then descends -- land FLAT
    CONTACT        LOW    LOW    LOW      it presses and wipes
    AFTER-CONTACT  mid    mid    HIGH     it lifts

    levels    xy  500 / 1000 / 3000 N/m    z  300 / 600 / 1500 N/m
              K_R  0.3 / 3 / 30 Nm/rad

K_R LOW only in contact is the measured half (50% -> 84% of the glyph off on
the curve).  K_R HIGH before contact is the hypothesis: a wrist that is already
soft lands on a corner of the pad instead of flat on its face.  The data
collected here is what would test it.

One board per run -- the shape is baked into the collision mesh when the scene
is built -- and the case name carries the board, so a flat and a curved
dataset can share an `--out` directory.

### What the first 300 demonstrations say

`S`, `7` and `<star>`, fifty kept episodes of each, on both boards, 16-18
minutes per board on one GPU (`summary.py` prints this):

                        curved                  flat
    kept                150 of 166 (90%)        150 of 150 (100%)
    dropped             16, all "erased"        none
    erased              0.99 +- 0.02            1.00 +- 0.00
    in band             0.97 +- 0.01            0.97 +- 0.01
    peak                5.2 +- 1.0 N            4.5 +- 0.7 N
    the board asks      9.3 +- 2.2 deg          3.8 +- 1.5 deg
    LANDING  K_R 30     misaligned 9.0 deg      3.7 deg    = 1.00 of the ask
    WIPING   K_R 0.44   misaligned 7.5 deg      2.7 deg    = 0.80 / 0.76

Nothing was dropped for the force band, for tearing, for collisions or for
protocol compliance -- the whole 10% loss on the curved board is episodes that
did not get enough of the glyph off, which is the right failure for this task
to have.  The protocol is legible in the record: K_R is exactly 30 while
landing and 0.44 while wiping, and the pad holds its commanded vertical to
1.00 of the demand while landing, which is what HIGH is for.

**And the record exposes the cost of that cell.**  While wiping, the pad ends
up 0.80 of the way back to where it started -- against 0.45 for the same
stiffness on a continuous traverse (the sweep above).  The reason is a time
constant, not a stiffness: at K_R = 0.44 and Lambda_r = 0.38 kg m^2 the
rotational mode has a period of about 6 s, and one raster leg is 1.2 s.  A pad
that only begins to yield when it touches down spends the whole stroke still
rotating, and then lifts for the next row.  The protocol's one unmeasured cell
-- K_R HIGH to land flat -- is what delays the yield, so the A/B is to set it
LOW (`--kr-land LOW`) and compare `left` while wiping.  That is the next run,
and it is 18 minutes.

### The A/B: landing stiff protects nothing

Same everything, 150 more episodes, with the one cell moved to LOW
(`--kr-land LOW`):

                             K_R HIGH to land      K_R LOW to land
                             (the protocol)        (the A/B)
    kept                     150 of 166 (90%)      150 of 152 (99%)
    dropped                  16, all "erased"      2
    WIPING   K_R             0.44 +- 0.18          0.30 +- 0.00
             misaligned      7.5 deg, left 0.80    6.1 deg, left 0.62
    TOUCHDOWN misaligned     9.0 of 9.0 asked      9.1 of 9.2 asked
             force in 0.2 s  2.5 +- 0.3 N          2.5 +- 0.3 N
    in band / peak           0.97 / 5.2 N          0.97 / 5.3 N

The cell was there to stop the pad arriving on a corner, so the touchdown is
measured two ways -- the angle it arrives at, and the force in the 0.2 s after
it, since an edge strike concentrates the load.  Neither moves: 9.0 against
9.1 deg, 2.5 against 2.5 N.  In hindsight it could not have: BEFORE contact
there is no contact moment, so there is nothing for a soft K_R to yield to and
the wrist holds whatever orientation it was commanded whichever stiffness it
has.  The hypothesis was about a moment that does not exist yet.

What landing soft buys is the ramp: K_R is already settled at 0.30 when the
stroke starts instead of still falling through 0.44, the pad ends up 0.62 of
the way back instead of 0.80, and the drop rate falls from 10% to 1%.

**So the table's `pre-contact` K_R should be LOW, not HIGH** -- one word in
`EXPECTED`, and `--kr-land LOW` already runs it.

0.62 is still short of the 0.45 a continuous traverse reaches at this
stiffness, and the rest of the gap is the lifting: every row re-lands and the
wrist starts settling again.  Wiping in longer strokes -- not lifting between
rows, as `smoke.py` does not -- is what would close it, at the cost of a
demonstrator that no longer looks like the writer.

## Known approximations

* The force projected onto "into the board" uses the board's MEAN-plane normal
  (`CurvedFrame.normal`), because that is what `writing/sim.py:step` asks for.
  The local normal is within 10-16 deg of it, so the normal force is right to
  a few per cent and the tangential leak is of order `mu f sin(16 deg)`.  The
  place where the local normal is not optional -- the contact geometry in
  `on_contact` -- uses `normal_at`.
* `on_contact` tests a mark as under the pad with `|d| <= ERASER_R`, a disc,
  for a pad that is a square of half-width `ERASER_R`.  Pre-existing, and it
  under-counts the corners by about 20% of the area.
* `smoke.py` is GIVEN the surface (`frame.to_world` follows it), which is why
  it is a measurement driver and not the demonstrator: `protocol.py` uses the
  synthetic operator, whose haptic loop finds the board by feel.

## Reusable as-is

- `curved_probe.py` — the pad/controller/surface probe, with the demand along
  the commanded path printed beside the result; `--amp 0 --tilt 8` is the flat
  control, `--press-mode flat` is what the task commands today
- `curved/surface.py` — `CurvedSurface.height/grad/normal/mesh`, closed form, so
  alignment is measured against the true surface rather than mesh facets.  Note
  `sigma = 0.035`, not the default 0.050: a bump of amplitude a and width s has
  radius of curvature ~s²/a, and our 30 mm pad needs a proportionally sharper
  surface than the 50 mm one `../curved` welded to see the same swing.
- `curved/wipe.py:op_space_rot_inertia` — the measured rotational inertia and
  matrix Dr that `writing/controller.py` still guesses at
- `curved/record.py` — the side-by-side recording
- `wiping/record_rot.py` — the 3-row comparison layout

## Status

Done: the wiping task on both boards, the eraser SRDF, the rounded pad rim, the
normal-force loop, the curved board with marks on the surface and a
surface-aware erasure model, `curved_probe.py` with its ablations, and the
K_R comparison above.

One driver for all of it: `smoke.py:run`.  `kr_task.py` measures with it and
`record_rot.py` renders with it, so the video cannot drift away from the
numbers the way it had (it was still commanding a fixed depth on a tilted board
at K_R = 60 vs 3 after everything else had moved on).

Superseded recordings:

* `wipe_rot_pinned_wrist.mp4` — the original, with the wrist pinned by the
  missing SRDF.  Kept deliberately: next to the re-recording it is the
  before/after of that bug.
* `wipe_rot.mp4` — re-recorded on the flat 8 deg tilt at K_R 60 vs 0.3 with the
  wrist free.  It shows 8.0 vs 7.9 deg of misalignment, i.e. still nothing, and
  that is now a result rather than a bug: on a RASTER the compliant pad's
  wander is as large as the demand, and an unsigned misalignment cannot tell
  "held the wrong vertical" from "drifted somewhere else".  The `yield` column
  is what separates them, and the force loop and the damping fix are what make
  the comparison fair.  Use `record_rot.py --curved` for the comparison that
  shows the effect.
* `wipe_S.mp4` — translation only, unaffected by any of this.

Also done: the rotational impedance (item 6, matrix Dr) and K_R as a gated
action (item 7) in `../writing/controller.py`, with `../writing/tests.py` at
35/35; `wipe_teleop.py`, `protocol.py`, `verify.py` and `summary.py`; and 300
demonstrations, 150 per board.

The replay check on a collected episode, which is what says the stored labels
ARE the demonstration:

    mode        erased   in band   misaligned   force RMSE
    recorded    100.0%    98.4%      9.1 deg      0.08 N     (the board asks 10.6)
    K const     100.0%    23.4%      7.4 deg      2.60 N     fails: force, tear
    K_R const   100.0%    96.6%     11.6 deg      0.27 N

Freezing K_p at its episode mean loses the force band outright; freezing K_R at
its mean leaves the pad WORSE aligned than the demonstration did, which is the
evidence that the rotational label is carrying something.

Not done: item 9, and the one-word change to `EXPECTED["pre-contact"]` the A/B
argues for, which is a change to an approved protocol and so not mine to make.
