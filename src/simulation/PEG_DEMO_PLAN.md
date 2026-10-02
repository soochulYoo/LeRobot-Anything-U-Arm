# Peg insertion: demo-collection plan

Modelled on `writing/protocol.py` and `wiping/protocol.py`. Both share one
shape, and it is the shape this task should take too:

> A scripted executor does the whole motion. The one thing it does not decide is
> the Case 1 stiffness. That comes from a person (or `--auto-user`) choosing one
> of three levels per axis group, a reaction time late, ramped rather than
> stepped. Every episode is scored for *success* and for *protocol compliance*,
> and only episodes that pass both are kept.

The point of that design is that the demonstrator never has to be good at
stiffness — they only have to follow a table. The table is the hypothesis, and
the dataset is the evidence for or against it.

> ## VERDICT: do not run this plan.
>
> Step 2 is done and the gate in §8 closed against collection. A single
> constant stiffness — axial 400 N/m held for the whole episode, lateral
> 300 N/m, `K_R` 20 Nm/rad, 4 N press — matches the phase schedule's success
> rate at every aim error tested: **8/8 at 4 mm and 8/8 at 8 mm, against the
> schedule's 8/8 and 8/8**, for 2.2 N more peak force (47.0 vs 44.8 N). There
> is no region where switching stiffness earns anything, so there is nothing
> for a stiffness demonstration to teach. §10 records what was measured.
>
> The plan below is kept as written, because the design is reusable for a
> harder instance of the task (§10 argues this one is too forgiving) and
> because the reasoning that led here should stay legible.

---

## 1. What makes this task different from writing and wiping

| | writing | wiping | peg insertion |
|---|---|---|---|
| contact | pen tip on paper, point | pad face on board, patch | peg flank in a hole, two-sided |
| what must be found | nothing, the paper is where it looks | nothing | **the hole, within 3 mm** |
| motion vs force | orthogonal (drag along, press down) | orthogonal | **parallel** (push along the axis, react along the axis) |
| failure | smudge, miss a stroke | leave ink behind | **jam** — stuck with the force climbing |
| tolerance | ~3 mm on curves | ~10 mm of board swing | **3 mm, with no chamfer** |

Two consequences matter for collection.

The frame question that dominated wiping goes away here. `wiping/` found that a
stiffness label's *frame* is what the task cannot absorb, because on a wipe the
motion-aligned and force-aligned frames end up orthogonal. In insertion they
coincide: both the motion and the reaction run along the hole axis, which after
alignment is the end-effector's own x. So the policy can name `K` in the
**end-effector frame** — observable, no privileged hole pose — and still mean the
right thing. `peg_insertion_cascade.py` exists to confirm exactly that, and it
should be re-run under Case 1 to close it out.

The failure mode is new. Writing and wiping fail by *doing too little*: a stroke
is missed, ink is left. Insertion fails by **doing too much and getting stuck**,
with contact force rising and no progress. A demonstration therefore has to show
not just how to find the hole but how to *stop* pressing and retry — and that is
the behaviour the keep rule in §4 has to be built around.

## 2. Measured geometry (ManiSkill `PegInsertionSide-v1`)

Everything here was read out of the env, not assumed:

- Peg: length 170–250 mm, **square** cross-section, half-width 15–25 mm, mass
  ≈ 0.54 kg. Randomised per seed.
- Hole: **square** aperture of half-width (peg half-width + 3 mm), cut by four
  boxes — so the clearance is **3 mm per side and there is no chamfer**. The
  entrance edge is sharp. Nothing leads the peg in; the tip must actually
  overlap the opening.
- The hole's centre is offset randomly across the box face by up to
  0.5·(length − half-width), so its position changes every episode.
- The box is as deep as the peg is long. In the hole frame (origin at the box
  **centre**) the entrance face is at `x = -half_len`, and success is
  `head_x >= -15 mm` — the head must travel nearly the whole peg length inside.
- Relative clearance therefore varies 3/25 = 12% to 3/15 = 20% with peg size,
  which is a real difficulty axis and a candidate for a case split.

## 3. The protocol

### Phases

The scripted executor (`peg_insertion_case1.py`) runs five phases. Unlike
writing's per-stroke loop and wiping's per-row loop, insertion is a **single
pass**, so the protocol runs once per episode — fewer level changes per episode,
which means fewer decisions and so more episodes needed for the same coverage.

| # | phase | what the executor does |
|---|---|---|
| 0 | travel | turn the peg onto the hole axis, carry it to a stand-off |
| 1 | calibrate | hold at the stand-off, measure how far the peg hangs below its command |
| 2 | touch | ramp onto the face until it presses |
| 3 | search | spiral outward from the believed hole centre until the head drops in |
| 4 | insert | push along the axis to seated |

### The table (PLACEHOLDER — step 2 fills it in)

Three axis groups, as writing had two and wiping three:

| | axial (along the hole) | lateral (across it) | K_R (wrist) |
|---|---|---|---|
| low | 800 N/m | 200 N/m | 2 Nm/rad |
| mid | 2000 N/m | 600 N/m | 10 Nm/rad |
| high | 4000 N/m | 1500 N/m | 80 Nm/rad |

| phase | axial | lateral | K_R |
|---|---|---|---|
| 0 travel | mid | **high** | **high** |
| 1 calibrate | **low** | low | mid |
| 2 touch | **low** | mid | mid |
| 3 search | **low** | **low** | **low** |
| 4 insert | **high** | low | low |

What is already justified, and what is not:

- **axial low = 800 N/m is forced, not chosen.** Case 1 has no active-wrench
  slot, so the press is `k_axial × (how far the reference sinks past the face)`.
  A pure impedance also cannot pass its own reference, so the reference must sit
  *deeper* than the depth that counts as a catch. 800 N/m × 10 mm = 8 N of press
  with 10 mm of reachable depth against a 5 mm catch threshold. Raise `k_axial`
  and the press rises with it unless the depth is cut, and cutting the depth
  makes the catch unreachable. **These two knobs are not independent** — this is
  the single most important difference from the cascade.
- **axial high = 4000 N/m is the ceiling**, `Case1Gains.k_hi`. `advance()` clips
  K's eigenvalues, so a magnitude/ratio parameterisation silently saturates: the
  cascade's magnitude 1200 / ratio 10 asks for 5565 N/m. Name K by eigenvalues.
- **lateral low trades against the sag.** The peg's weight pulls the hand down
  by `m g / k_lat` — 27 mm at 200 N/m, 9 mm at 600, 3.5 mm at 1500, against
  3 mm of clearance. §7 explains how that is handled; the point for the table is
  that the softer the lateral axis, the more work the calibration has to do.
- **K_R low is a prediction that may well be wrong.** The moment available to
  pivot the peg into the aperture is about `press × peg half-width` ≈ 8 N ×
  20 mm = 0.16 Nm, so a felt stiffness below ~3 Nm/rad is needed for a few
  degrees of yield — hence low = 2. But the peg is a *long lever gripped off
  centre*: its weight alone applies 0.54 kg × 9.81 × 40 mm = 0.21 Nm about the
  grasp, which at 2 Nm/rad is a **6° droop, or 11 mm at the head** — again far
  past the clearance. So unlike wiping's pad, a soft wrist before the hole
  constrains the peg is actively harmful. Wiping found `K_R` high before contact
  "protects nothing"; **insertion should find the opposite**, and that is a
  falsifiable claim to test before collecting anything.
- **Nothing else in the table is measured.** Treat it as a starting point for the
  sweep, not a result.

### Level changes ramp

As in both existing protocols: a level change is a stiffness *rate*, with a time
constant (`--ramp`, 0.15 s), and the energy tank pays for it — for `K_R` too,
since `Case1Controller` meters a rotational stiffness rate as
`U_r tr(I - R_d^T R)` through the same gate. Measured: a 16 s episode at an 8 N
press draws ≈ 5 J of the 20 J tank and `alpha` never leaves 1.0, so the tank is
**not** what limits a search. Worth re-checking once the level changes are added,
because each ramp is metered.

## 4. Scoring and the keep rule

Success is the env's own `has_peg_inserted` — head within 15 mm of the hole
centre. That is necessary but far from sufficient for a demonstration worth
imitating, so an episode is kept only if all of:

| gate | why |
|---|---|
| `success` | it went in |
| `compliance >= 0.8` | the levels matched the table, ignoring `--grace` s after each phase change — identical to writing and wiping |
| `peak_force <= F_max` | **a jammed insertion that eventually succeeds is not a demo.** Writing's analogue was the 12 N paper-tear limit. `F_max` has to come from the sweep |
| no jam-and-eject | depth exceeded the catch threshold and then retreated past it — observed at 4 mm of aim error under the cascade, with force pinned at the clip |

The last two are new relative to writing and wiping, and they exist because of
§1: this task's failure mode is force, not coverage.

Recorded per episode regardless of keeping (`attempts.jsonl`, as both protocols
do): success, compliance, time-to-catch, peak and mean contact force, deepest
depth, jam count, `alpha_min`, tank energy drawn, the calibrated sag, and the
realised peg/hole geometry.

## 5. Cases and volume

`protocol_v1` is 3 texts × 50 kept = 150 episodes; wiping matched it per board.
The analogue of "text" here — the axis the policy must generalise over — is the
**aim error**, because that is what makes a search necessary at all:

| case | aim error | why |
|---|---|---|
| `e0` | 0 mm | the hole is where it looks; the search should cost almost nothing |
| `e2` | 2 mm | inside the clearance |
| `e4` | 4 mm | just outside it — **the jam was observed here** |
| `e6` | 6 mm | twice the clearance |
| `e8` | 8 mm | the largest the cascade's scrub recovered |

5 cases × 50 kept = **250 episodes**. The direction of the error is randomised
per seed; peg size and hole position are randomised by the env. The error is
injected as a *belief*, so the executor genuinely does not know it — only the
scorer does.

Two things to decide after step 1, not now: whether 10 mm and 12 mm are
reachable at all (if they are not, the ceiling is the dataset's edge and should
be documented), and whether peg size needs its own case split given that
relative clearance varies 12–20%.

## 6. What gets recorded

Same container as writing and wiping — one `.h5` per episode with a rich meta
dict (spec, protocol table, compliance breakdown, gains, criteria, metrics) plus
an optional `.mp4` — so the existing loaders and `policy/data.py` can read it
with the obvious field renames.

- **Observation**: wrist and/or side camera, proprioception, measured wrench,
  and the peg pose. The peg pose is kinematics, not force feedback, and a real
  system would get it from vision; it is needed because the schedule is written
  in *head* coordinates (§7).
- **Action**: `(Vd, K, K_R)` — the Case 1 proposal exactly. `K` as 3 eigenvalues
  in the **end-effector** frame (§1 is why that is legitimate here), `K_R` as a
  scalar unless the sweep shows an anisotropy is needed. Note that
  `felt_from_KR` means a commanded `K_R` is not the stiffness the tool feels, and
  "resist twist, comply in tilt" is *unreachable* — so an anisotropic `K_R`
  needs `felt_reachable()` checked before it goes in the table.

## 7. Known obstacles, already measured

These came out of porting the search from the cascade to Case 1, and all four
are fixed in `peg_insertion_case1.py`. They are listed because they constrain
the protocol, not just the code.

1. **The peg's weight sags the hand 13.8 mm** on a 300 N/m lateral axis — 4.6×
   the clearance. ManiSkill disables link gravity for a fixed-base arm, so the
   *arm* is weightless, but the grasped peg is not. The cascade hid this behind a
   3000 N/m isotropic inner loop. Stiffening it away would discard the lateral
   compliance the task is about, so the sag is **calibrated**: the reference
   holds at the stand-off with the search stiffness already applied and
   integrates the head error into a frozen bias. That correction is a reference
   motion, so it rides in `Vd` and the gate meters it.
   **Consequence for the protocol: the calibration is only valid at the
   stiffness it was measured at.** Change a level and the sag changes with it, so
   the calibration must happen *after* the level the search will use is in
   effect. This couples §3's table to the executor and is the main reason the
   table cannot simply be swapped at will.
2. **A phase boundary is a step.** Jumping the reference 32 mm in one 2 ms tick
   asks for `Vd` = 16 m/s; the damping feed-forward `u = D Vd` then demands
   2129 Nm against an 87 Nm limit and slams the peg into the face at 65 N. Every
   target ramps with a cosine ease and `|Vd|` is clipped.
3. **A pure impedance cannot pass its own reference** — see §3 on the axial
   level.
4. **Aiming is not commanding.** `x_d` names the TCP; success is judged on the
   peg head, ~107 mm away through a grasp that sits 4.5° off the hole axis. The
   schedule is written in head coordinates and converted through the *measured*
   peg axis each step.

Also: `Case1Controller.disable_joint_drives()` must **not** be called here. It
kills the drives on every active joint including the two fingers, and those
drives are the only thing holding the peg. `tau_limit` is zeroed on the finger
entries instead.

## 8. Order of work, with gates

The plan above is step 3. Do not start it until step 2 says there is something to
learn.

- **Step 1 — does the search work at all?** Port the scrub to Case 1 (done) and
  measure success vs aim error, with the insert/jam/miss split. Under the cascade
  the scrub recovered 8 mm — 2.7× the clearance — while aim-and-push failed past
  ~2 mm. Reproduce that under Case 1. *Gate: if the Case 1 search cannot beat
  aim-and-push, fix the executor before anything else.*
- **Step 2 — does one constant stiffness suffice?** Sweep the three axis groups
  as constants, per aim-error case, and record insert/jam/miss and peak force.
  This is what fills in §3's table and sets `F_max`. *Gate: if some single
  (axial, lateral, K_R) is at least as good as any phase-dependent schedule
  everywhere, stop. Say so, and do not collect.* Six wiping environments already
  went that way, so this is the likeliest outcome and the cheapest place to find
  it out.
- **Step 3 — collect and train**, per §§3–6, but only for the region step 2
  shows a constant cannot handle. The 4 mm jam is currently the only positive
  evidence that such a region exists: the peg got 17 mm in, the force saturated,
  and it was ejected. One episode is not evidence; step 2 is how it becomes
  evidence.

## 9. Open questions

- Does a jam need a *momentary* lateral stiffening to break, or a retreat and
  retry? The first is a stiffness policy; the second is a motion policy, and a
  stiffness dataset would not teach it.
- Is the spiral the right search? It is open-loop. A demonstrator feels the edge
  and follows it, which would make the demos much more informative — and much
  harder to script.
- The success criterion is deep (nearly the whole peg length). Is a partial
  insertion a useful demonstration, or noise?


---

## 10. Outcome of steps 1 and 2 (measured)

### Step 1 — the search works, in a band

8 seeds per cell, scrub against aim-and-push. The scrub turns 2/4 into 4/4 at
4 mm and removes every jam; at 6 mm it turns 2/4 into 3/4. At 0–2 mm the peg is
already in from the touch phase and the search does nothing; at 8 mm both arms
are equal. So the motion is real, and it is worth about 4–6 mm of aim error.

One case runs the other way (6 mm, seed 2): aim-and-push wedges 8.2 mm in while
the scrub never finds the hole at all. The search is not uniformly better, and
`videos/peg_search_6mm_counterexample.mp4` is that episode.

### Step 2 — the press is the knob, and one constant is enough

| | 4 mm | 8 mm | 12 mm |
|---|---|---|---|
| press 2 N | 7/8 | 1/8 | 0/8 |
| **press 4 N** | **8/8** | **7/8** | 2/8 |
| press 8 N | 7/8 | 3/8 | 0/8 |

Press dominates every stiffness effect, and it is not monotonic: 2 N is worse
than 4 N. The mechanism is friction at the sharp edge — a light normal force
lets the head slide over the aperture, a heavy one lets the edge bite and wedge.
Above ~50 N the wedge pulls the peg out of the fingers entirely, which is the
number the §4 keep rule wanted for `F_max`.

Schedule against a true constant, 8 seeds, press held at 4 N:

| axial | 4 mm | 8 mm | peak force at 8 mm |
|---|---|---|---|
| schedule 3000 → 400 → 3000 | 8/8 | 8/8 | 44.8 N |
| **constant 400** | **8/8** | **8/8** | 47.0 N |
| constant 800 | 8/8 | 6/8 | 51.3 N |
| constant 1600 | 8/8 | 4/8 | 51.2 N |
| constant 3000 | 8/8 | 2/8 | 46.4 N |

The task does care about the stiffness *magnitude* — a stiff constant collapses
to 2/8 — but not about *switching* it. Note also that a constant 400 N/m pushes
the peg home perfectly well: once the head is in the aperture the hole guides
it, so the stiff push the schedule provides is not doing any work.

(Peak force does not rank across rows with different success rates: a run that
never engages records a low peak because nothing happened.)

### What actually limits it, and why it is not a stiffness problem

At 12 mm the search fails, and widening the spiral from 13 mm to 25 mm of reach
made it **worse** — 0/8 against 2/8 — because the same 4 s over a longer path
skates the head across the opening without dwelling long enough to drop in. The
binding constraint at large misalignment is **search dwell time**, which is a
motion-policy question. A stiffness dataset could not teach it.

### Why this instance is a weak testbed for variable impedance

- Motion and contact force are **parallel**, so the stiffness-frame question
  that dominated wiping collapses to one dimension.
- The clearance is 3 mm on a 15–25 mm half-width peg: **12–20%**. Industrial
  precision insertion runs 0.1–1%. A passive RCC device solved this class of
  problem in the 1980s.
- There is no chamfer and the hole axis is known after alignment, so the
  multi-contact wedging that motivates impedance scheduling barely arises.

The honest claim is therefore "**fixed stiffness suffices for this instance**",
not "insertion does not need variable impedance". Tightening the clearance to
~0.3 mm, adding a chamfer, or tilting the hole axis would test the difference.


### Tightening the clearance does not rescue the schedule

The §10 claim above was "fixed stiffness suffices for THIS instance", with the
3 mm gap (12-20% of the peg half-width) named as the likely reason. So the gap
was cut to 2 mm and 1 mm and the comparison rerun, 8 seeds, aim errors 0-6 mm.
The spiral's radial pitch was rescaled to the clearance each time, because a
pitch wider than the capture window steps over the hole and would have made the
script's coarseness look like the task's difficulty.

Totals over the four aim-error levels (out of 32):

| axial | 3 mm gap | 2 mm gap | 1 mm gap |
|---|---|---|---|
| schedule | 32 | 30 | 21 |
| **constant 400** | **32** | 30 | **22** |
| constant 800 | 30 | **31** | 19 |
| constant 1600 | 28 | 27 | 13 |
| constant 3000 | 26 | 27 | 12 |

Tightening the gap makes the task much harder -- the best arm falls from 32/32
to 22/32 -- but it does **not** open a gap for the schedule. At every clearance
the best constant ties or beats it. The ranking is also unchanged: softer wins,
and the stiffest constant is worst everywhere. The task cares about the
stiffness MAGNITUDE and not about switching it.

One caveat that limits the 1 mm row. At 1 mm the zero-aim-error cell is only
7/8, and with no aim error there is nothing to search for -- that miss is the
sag calibration's residual, which is on the order of the clearance itself once
the clearance is 1 mm. So the 1 mm block is partly measuring this executor's
calibration precision rather than the task. The 2 mm block is clean, and it
says the same thing.
