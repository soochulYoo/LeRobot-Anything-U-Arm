# writing — teleoperated handwriting under Case 1 with a variable K_p

A SAPIEN/ManiSkill environment for collecting and evaluating demonstrations of
writing text or drawing shapes with a pen, where the force matters. The loop is
bilateral teleoperation: a human pushes a master, and the slave's Case 1
geometric impedance follows it into contact with the paper. The operator also
sets the Case 1 stiffness K_p while writing.

```
human --f_h--> master (Mm, Bm) --x_m--> x_d request --tank gate--> Case 1 GIC (K_p) --> pen --> paper
  ^                                          K_p request ------^                          |
  +---------------- f_fb = K_p (p - x_d)  <-------------------------------------------------+
```

It reuses `../cascade` as its reference, not as a dependency. `controller.py` is
`cascade/case1.py` moved from a point mass onto the Panda. It keeps the same
gate, the same tank, the same proposal power and the same record-then-integrate
step order. The synthetic operator is built the way `cascade/spiral_task.py`
builds one, with delayed vision and delayed touch.

## Files

| file | what it is |
|---|---|
| `glyphs.py` | single-stroke font (A–Z, 0–9) and six shapes; text layout in paper coordinates |
| `scene.py` | `TeleopWriting-v1`: Panda with a ball-point pen link, kinematic paper, ink and template dots, top and wrist cameras |
| `controller.py` | Case 1 on the arm: GIC with time-varying K_p, common power gate, energy tank, torque saturation logged as δτ |
| `sim.py` | `WritingSim`: one episode at 500 Hz; ink from measured force, tear and collision checks, observations, the score |
| `teleop.py` | virtual master, `TeleopSession` (the bilateral loop), `SyntheticWriter` (the delayed human) |
| `collect.py` | parallel demo collection → one HDF5 per episode + `index.json`; replays stored labels to verify them |
| `evaluate.py` | `WritingPolicyEnv` (policy-rate env), replay and scripted reference policies, pluggable learned policy |
| `interactive.py` | collect demos yourself: keyboard force teleop in the viewer, live K_p keys, recording (see below) |
| `protocol.py` | the pen writes by itself, you set stiffness in three levels by a fixed protocol (see below) |
| `policy/` | force-aware flow policies (CoFA a–d) whose action includes stiffness; training and Slurm scripts, see `policy/README.md` |
| `plot_episode.py` | one recorded episode on one page: ink vs. target, force, K_p, camera frames |
| `tests.py` | 30 checks: `python3 tests.py` (~20 s) |

## Collecting demonstrations with the keyboard

You write in the 3-D viewer. The keys **push** on the master, and the pen
follows through Case 1. The same recorder as `collect.py` saves each successful
attempt in the same HDF5 format. It needs a display and runs in real time.

```bash
cd src/simulation/writing
python3 interactive.py --record demos/keyboard
```

| option | effect |
|---|---|
| `--record DIR` | save demonstrations into `DIR` (omit it to just practise) |
| `--text HELLO` | write this every time; the paper pose is still random. Otherwise the task is a random 1–3 characters or a shape |
| `--keep-failed` | also save unsuccessful attempts (their `success` attribute says so) |
| `--no-template` | hide the grey guide on the paper |
| `--no-wrist`, `--image-size N` | camera setup, as in `collect.py` |
| `--seed N` | first task seed (default 50 000; synthetic demos use 0…, evaluation 10 000…) |

The viewer looks over your right shoulder, so the text reads left to right and
the keys match the screen. The terminal prints the task, then a status line:

```
[task 50000] write 'K7'   (paper height and tilt are hidden)
 t   12.5s  felt  3.41 N  paper  3.38 N (band 1-6)  ink  183  K u,v 1500 n 400
```

`felt` is the force the master pushes back with; `paper` is the pen's pressure,
the value the score uses.

### Keys

| key | does |
|---|---|
| **J / L** | push left / right along the line (about 5 cm/s) |
| **I / K** | push up / down the letter (away from you / toward you) |
| **U** (hold) | press the pen into the paper: it lands at about 15 mm/s, and at rest presses with 3.5 N, whatever K_p is |
| **O** | lift |
| **SHIFT** | twice as fast along the paper and up, never harder into it |
| **1 / 2** | in-plane stiffness K_u, K_v down / up (×1.25) |
| **3 / 4** | normal stiffness K_n down / up (×1.25) |
| **R** | finish: score it, save it if successful, next task |
| **N** | discard, next task |
| **T** | discard, try the same task again |
| **P** | print the score so far |
| **ESC** | quit; the attempt in progress is discarded |

WASD/QE and the mouse belong to the viewer's own camera, which is why the pen
uses IJKL/UO.

### Writing one character

1. The pen starts 15 mm above the middle of the text. Steer to the start of the
   first stroke with **J/L/I/K**. Stroke order and direction are not scored.
2. **Hold U.** The pen lands after about 1 s, and ink appears once the pressure
   passes 0.8 N.
3. **Keep U held** and steer along the stroke with J/L/I/K.
4. **Release U at the end of the stroke and pause half a second**, or tap O.
   Releasing lets the pen come off the paper. Moving the instant you release can
   leave a dot or two, and travelling with the pen down fails *precision*.
5. Repeat for every stroke, then press **R**. The terminal prints the verdict
   and where the file went:

```
[SUCCESS] coverage 0.97 precision 0.95 in-band 0.93 peak 4.1 N  ->  demos/keyboard/ep_50000.h5   (1 successful saved this session)
```

**Stiffness is part of the label.** The Case 1 stiffness you set is recorded
every step and becomes the action `k_diag` a policy learns. It starts at
K_u = K_v = 1500 and K_n = 400 N/m, and stays within 100–4000 N/m. Change it
while you write the way you'd want a policy to:

- **Higher in-plane stiffness** makes the pen hug your hand around corners.
- **A softer normal** forgives the paper's unknown height and tilt. With the
  keyboard, your 3.5 N push sets the resting force, not K_n; K_n sets how much
  the force swings when the paper's height changes under a moving pen. Moving at
  10 cm/s across a 5° tilt, the force error averaged 0.9 N at K_n = 400 and
  1.3 N at 3000.
- **It matters more for the policy than for you.** A policy that outputs
  (x_d, K) has no hand pushing on the master; the K_n in your demos is what it
  will write with. In the scripted comparison under *Results*, soft K_n succeeds
  30/30 and stiff K_n 5/30.

When an attempt fails, the verdict names the check (criteria under *Success
criterion* below):

| check failed | usual cause |
|---|---|
| coverage | a stroke missed, or started or lifted early (ink needs about 1 s of U to start) |
| precision | the pen travelled while still down; pause after releasing U |
| force | too much pen-down time spent landing or lifting instead of writing steadily |
| no_tear | unlikely with U alone: landing peaks measured 3.4–4.4 N, and 5 N moving fast across a tilt |

### Where the demonstrations go

- `demos/keyboard/ep_<seed>.h5`: same layout as the synthetic demos (see *Data*
  below), with `attrs["source"] = "keyboard"`. `/full/phase` and `/full/stroke`
  are −1, since there is no scripted operator. There is no time limit.
- `demos/keyboard/attempts.jsonl`: one line per finished attempt, saved or
  discarded, with its metrics.
- **Stop and resume freely.** Re-running the same command continues at the next
  unused seed. It never repeats a task or overwrites a file.

Check them the way the synthetic demos are checked:

```bash
python3 plot_episode.py demos/keyboard/ep_50000.h5                           # ink, force, K_p, frames
python3 evaluate.py --policy replay --demos demos/keyboard --episodes 1000   # labels reproduce the attempt
```

Keyboard and synthetic demos share one format, so they can be mixed in one
training set and told apart by `attrs["source"]`.

**How this was tested.** The keyboard human, landing, release and the recording
workflow run headless in `tests.py` and in a scripted-keyboard session:
R/N/T/ESC, saving, `attempts.jsonl`, resume at the next seed, and replay of a
saved attempt to 0.06 N. The viewer opens with the view above on this machine.
No person has written a full word with it yet, so expect to tune `F_PLANE`,
`F_NORMAL` and the damping in `KeyboardHuman` to taste.

## Collecting with the stiffness protocol

The pen writes by itself, and you set only the translational stiffness, in
three levels. The synthetic writer does the motion and the pressing; your
keyboard replaces its stiffness schedule. The script is `protocol.py`.

```bash
cd src/simulation/writing
python3 protocol.py --texts S 7 "<star>" --per-case 50 --out demos/protocol
```

`--texts` are your cases (shapes go in `<>`, so quote them). Each case collects
`--per-case` kept demos. Within a case the text is fixed and everything else is
randomized per demo: the paper's height (±4 mm), tilt (±5°) and friction (μ
0.2–0.5); the letter size, slant and placement; and the writer's speed, pressing
force and reaction delays.

### Levels and keys

| level | xy (along the paper) | z (into the paper) |
|---|---|---|
| low | 500 N/m | 500 N/m |
| mid | 1000 N/m | 1000 N/m |
| high | 3000 N/m | 3000 N/m |

| key | does |
|---|---|
| **1 / 2 / 3** | xy low / mid / high |
| **8 / 9 / 0** | z low / mid / high |
| **M** | everything to mid |
| **G** | start the next demo (the paper and template are shown first) |
| **N** | abort this demo and retry the same randomization |
| **ESC** | quit; the demo in progress is discarded |

A level change is a command, not a step in K. The applied stiffness follows it
with a 0.15 s time constant (`--ramp`), and the energy tank pays for the
change. Change the values with `--k-xy LOW MID HIGH` and `--k-z LOW MID HIGH`.
Rotational stiffness stays fixed at 80 Nm/rad.

### The protocol

Repeat it for **every stroke**. A single-stroke text like `S`, `7` or `<star>`
goes through it once; `H` would go through it three times.

| step | writer is… | set | terminal cue |
|---|---|---|---|
| 0 | starting | all mid (automatic at G) | |
| 1. approach | travelling to the stroke's start | xy mid, **z HIGH** | `APPROACH -> xy MID, z HIGH` |
| 2. pre-contact | hovering 1 s over the start, then descending | **xy LOW**, z mid | `PRE-CONTACT -> xy LOW, z MID` |
| 3. contact | pressing and writing | keep **xy LOW**, z mid | `CONTACT -> xy LOW, z MID` |
| 4. after contact | lifting (and rising at the end) | **all mid** (M) | `AFTER-CONTACT -> xy MID, z MID` |

A cue prints the moment each phase starts. The keys for one stroke are
**0**, then **1 9**, then **M**.

### What is kept

Each demo is checked against the protocol. **Compliance** is the fraction of
time your levels matched the table, not counting the first 0.6 s after each
phase change (`--grace`, your reaction time). A demo is kept only if it
**succeeds** on the task criteria **and** complies ≥ 80 % (`--min-compliance`).
Otherwise it moves on to a new randomization. The terminal shows both:

```
[KEPT] coverage 1.00 precision 1.00 in-band 0.97 peak 3.4 N | protocol approach 93% pre-contact 100% contact 100% after-contact 100% | 1/50
```

- `demos/protocol/<case>/ep_<seed>.h5`, one folder per case (`S/`, `7/`,
  `star/`), in the same format as all other demos, plus:
  - `k_level` (xy, z; 0 = low, 1 = mid, 2 = high) in `/full` (100 Hz), `/obs`
    and `/action` (10 Hz), next to the applied `k_diag`, `K` (3×3) and damping
    `D` already there;
  - attributes `case`, `source = "protocol-keyboard"`, `compliance` (per phase),
    and `protocol` (levels, the expected table, dwell, ramp, grace).
- `demos/protocol/attempts.jsonl`: every finished demo, kept or not.
- **Resume** by re-running the same command. It counts what is in each folder
  and continues with new randomizations.

By default the cases rotate, one demo each in turn (`--order round-robin`);
`--order block` does all 50 of one case first. Each demo takes 10–14 s of
writing in real time, so 3 × 50 is roughly 30–40 minutes plus retries.

### Without a person

`--auto-user` replaces the keyboard. The levels switch by protocol phase,
`--reaction MIN MAX` seconds late (a person's reaction; `0 0` switches exactly
at each phase change). Headless, it runs in parallel:

```bash
python3 protocol.py --texts S 7 "<star>" --per-case 50 --auto-user --reaction 0 0 \
                    --headless --workers 12 --video 2 --out demos/protocol_v1
python3 plot_episode.py demos/protocol_v1/S/ep_60000.h5
```

`--video N` also writes an mp4 of the first N attempts per case. These files
carry `source = "protocol-auto"` and the reaction used in `attrs["protocol"]`.

**`demos/protocol_v1`, collected with exactly that command:** 150/150 kept on
the first attempt, in 2.3 min on 12 workers, 98 MB total. Each file has
128×128 top and wrist images at 10 Hz (17 070 frames per camera in all).

| case | coverage mean [min] | force in band | pressure peak | chamfer | length |
|---|---|---|---|---|---|
| `S` | 99.96 % [98.9] | 99.0 % | 3.8 N [max 5.0] | 0.58 mm | 10.9 s |
| `7` | 99.87 % [96.7] | 98.5 % | 3.6 N [max 4.7] | 0.49 mm | 9.3 s |
| `<star>` | 99.46 % [**91.7**] | 99.4 % | 4.1 N [max 5.3] | 0.61 mm | 13.8 s |

- **Levels:** the recorded levels equal the protocol table in 99.9 % of samples;
  the rest is the one sample at each switch.
- **Settled stiffness per phase (u, v, n):**
  - approach 1000/1000/2992
  - pre-contact 501/501/1006
  - contact 500/500/1000
  - after contact 998/998/1000
- **Replay:** the stored labels reproduce the pressure to 0.03–0.08 N RMSE;
  with K frozen at its mean, 0.4–0.8 N.
- **Corners:** xy LOW while writing rounds the star's points. The worst star
  demo is at 91.7 % coverage, just above the 90 % threshold.

### How the numbers were chosen

A scripted person followed the protocol on `S`, `7` and `<star>`, 20
randomizations each (`tests.py` repeats one of them):

| setting | result |
|---|---|
| first try: xy low = 300, writer's own descent and travel speed | 33/36; coverage failed on high-friction paper, peaks up to 10.6 N |
| xy low = 500 | coverage fixed |
| descent 10 mm/s before contact (`--v-desc`) | landing peaks from ≤10 N to ≤5.3 N (tear limit 12 N) |
| approach at 30 mm/s (`--v-travel`) | a slow reactor holds "z high" 84–90 % of the approach instead of 33–43 % |
| **final, quick person (0.2–0.45 s reaction)** | **60/60 succeed, 100 % compliance, peak ≤ 5.0 N** |
| **final, slow person (0.5–1.0 s reaction)** | **60/60 succeed, compliance ≥ 86 %** |

Why xy low needed to be 500: with μ = 0.48 and a 3.7 N press, friction drags a
300 N/m pen about 6 mm behind its reference, past the 3 mm tolerance on
curves.

The descent and approach are slower here than in `collect.py`. That is a
property of the writer in this dataset, recorded in the style attributes of
each file.

## The task

The robot writes a random 1–3 character string (A–Z, 0–9), or one of six
shapes, 30–40 mm tall, with random slant and placement. A faint grey template
is drawn on the paper, so vision says what to write; `show_template=False`
turns it off. The goal strokes are also stored with every episode.

**The paper is not where anyone thinks it is.** Each episode randomizes its
height (±4 mm), tilt (±5° per axis) and friction (μ 0.2–0.5). The operator and
any policy plan on the nominal paper. `WritingSim.privileged()` has the truth,
and it is stored under `/privileged`, never in observations.

**Ink comes from force, not proximity.** ManiSkill's `DrawSVG` inks whenever
the TCP is within 5 mm of the canvas, so force can't matter there. Here the
paper is a collision body, and ink is laid at the contact point only while the
measured pen/paper force is above 0.8 N. Above 12 N the paper tears.

### Success criterion (`sim.Criteria`)

| check | threshold |
|---|---|
| coverage: target points with ink within `tol` | ≥ 90 %, `tol` = 3 mm |
| precision: ink within `tol` of the target (catches an unlifted pen) | ≥ 90 % |
| force: pen-down time with pressure in 1–6 N | ≥ 85 % |
| no tear: pressure never above | 12 N |
| no collision: any non-pen link on the paper above | 1 N |

**Two force signals, because PhysX contact is rigid.** A landing at 15 mm/s
puts 0.14 N·s into a single 2 ms step: 68 N raw, 16 N at 25 Hz. That implies
about 7 kg of apparent mass at the tip, measured below. Light touches chatter,
and each re-contact is a one-step impulse. When the 25 Hz sensor signal judged
the task, a 2 mm/s touch at 0.3 N laid 8 dots of ink, and gentle landings
counted as tears. The fix:

- **Sensor, 25 Hz:** observed by a policy and logged.
- **Pressure, 5 Hz:** decides ink, pen-down, the force band and tearing.

On the pressure signal a 15 mm/s landing reads 4 N and a 0.4 N touch leaves no
ink. Ink lags the pen by the filter's 32 ms, well under a millimetre at writing
speed.

## Teleoperation onto Case 1

**There is no admittance.** The master's position is the requested equilibrium
x_d ("the command is x_d"). The reflected force is the GIC spring force
K_p(p − x_d), which at rest in contact equals the contact force. The operator
feels the paper through the same spring that presses on it. A stiffer K_p makes
the pen follow more tightly and makes the paper feel harder.

**The keyboard is a force input.** In `interactive.py`, holding "press" applies
3.5 N to the master, and the pen settles where the paper pushes back as hard.
The tests measure 3.3–3.6 N on the paper (friction and tilt take the rest),
independent of K_n. A real haptic
master, such as the U-Arm this repository builds, plugs in at the same place:
anything with `wrench(t, x_m, v_m, f_fb) -> (f_h, k_target)`.

### Variable K_p, and why it needs the tank

The operator requests a stiffness target along the writing frame (u, v along
the paper, n off it). It is low-passed with a co-contraction time constant and
turned into a stiffness rate U_p. Case 1's gate scales that rate by the same α
as the reference velocity. Stiffening a stretched spring injects ½ p_deᵀ ΔK p_de,
and without the tank a stiffness schedule is an energy source. The test
measures the tank paying 0.01490 J against a predicted 0.01490 J. A stiffness
request past `k_hi` is clipped (eigenvalue-wise, so K stays SPD).

The synthetic writer's schedule:

| phase | k along the paper | k off the paper |
|---|---|---|
| travel | 600–1200 | 600–1200 |
| settle / descend | 1500–3000 | **200–400**, cushions the unknown paper |
| press / stroke | 1500–3000, **up to +80 % on tight curves** | **250–600**, turns a height error into a small force error |
| lift | 1500–3000 | back to travel |

Each writer draws its own values, with ±6 % slow random modulation. Across the
24 recorded demos k_n spans 190–1240 N/m and k_t 590–4000 N/m. k_t reaches the
`k_hi` = 4000 bound on tight curves, where the gate's clip takes over.

**Damping follows K_p and the arm's inertia:**
D = ζ(Λ^½K^½ + K^½Λ^½), with Λ the operational-space inertia at the tip. The
first version assumed a 1.5 kg tip. The real one is about 7 kg vertically and
1–7 kg sideways at the writing pose, so the normal axis ran at ζ ≈ 0.3, the pen
overshot its hover height by 5 mm, and it hit the paper at travel speed.
Halving the impact peaks took this fix, not a gain change.

The damping feed-forward D·V_d, which removes the moving-reference lag, is not
dissipative. It therefore rides in Case 1's active slot u, where the gate scales
it and the tank pays for it. Hiding it inside the "passive" baseline would be
an unaccounted energy source.

**Requested vs. applied.** Both are logged. With the gate open they match every
step to 1e-9 (tested). In all demos collected so far the gate never closed
(min α = 1.00, reported in `index.json`), and the verifier replays the applied
values to prove the label.

## The synthetic writer

Its delays make it an operator, as in `spiral_task.py`:

- **Visual loop:** 150–250 ms delay. It integrates the cross-track error, and
  it decides when a stroke is finished.
- **Haptic loop:** 40–80 ms delay. It adjusts how hard to press until the felt
  force matches the intended 2–4 N.
- **Arm:** a spring-damper about the intended hand motion, plus
  signal-dependent motor noise and 10 Hz tremor.

It never learns the tilt. It finds the paper by feeling for it.

Four things had to be fixed before it could write; each one is a property a
real operator has:

1. **Settle before feeling for the paper.** While travelling, the master feels
   the slave's inertia (1–2 N), which the writer took as "touched". It then
   pressed, found nothing, pressed harder, and hit the paper at 25 N.
2. **Damp about the intended motion, not the world.** World damping made the
   hand lag its own plan by Bh·v/Kh, about 1.5 mm.
3. **Correct cross-track error only.** A proportional correction of a
   200 ms-stale error rang at about 1 Hz. A 2-D integral stored a correction
   that became wrong once the path turned, and pushed the pen outside curves.
4. **Lift when the pen is seen at the end, not when the hand is.** Friction
   makes the pen trail by about 3 mm, and lifting on the hand's schedule cut
   the last 4 mm off short strokes.

## Results

**Synthetic writer, 60 random episodes:** 60/60 successful. Coverage and
precision were 100 % on all of them. Force in band averaged 98.6 % (minimum
90 %). Mean pressure peak was 4.4 N (maximum 7.0 N), and episodes averaged
20.5 s. It runs about 4× real time without cameras.

**Collection, 24 episodes with both cameras at 10 Hz:** 24/24 kept, 1.5 min on
8 workers, 30 MB.

**Replaying the stored labels** (x_d, K) through the controller on the same
paper:

| replayed with | pressure RMSE vs. demo |
|---|---|
| the recorded, time-varying K | **0.04–0.05 N**, every episode still succeeds |
| K frozen at its episode mean | 0.22–0.50 N (5–10× worse), still succeeds |

So the stiffness label carries real information about the force. On these
three episodes, though, dropping it did not cost success: x_d already encodes a
press depth that works. The next table is where K decides the outcome.

**Does the task reward choosing K?** A scripted, privileged writer knows the
strokes but not the paper. It feels for the paper, presses to depth
f_target / k_n, and writes. Only its stiffness varies. 30 held-out seeds each:

| K_p (u, v / n) | success | mean force in band | chamfer |
|---|---|---|---|
| 2000 / 400 (demo style) | **30/30** | 97 % | 0.29 mm |
| 400 / 400 | **30/30** | 97 % | 0.46 mm |
| 2000 / 2000 | 13/30 | 84 % | 0.88 mm |
| 3000 / 3000 | 5/30 | 69 % | 1.27 mm |

**Normal stiffness decides success.** A 5° tilt changes the paper's height by
3 mm along a 35 mm stroke; through a stiff k_n that becomes 9 N of force error
or loss of contact. In-plane stiffness only moves accuracy (chamfer 0.29 → 0.46
mm). The 3 mm tolerance doesn't make it decisive, because friction lag is
mostly along the stroke. Tighten `Criteria.tol` if in-plane stiffness should
matter for success.

## Data (HDF5, one file per episode)

| group | rate | contents |
|---|---|---|
| `/full` | 100 Hz | `qpos qvel tau tcp_pos tcp_quat tcp_vel tcp_angvel`; `f_contact` (sensor), `f_raw_mean` (raw impulse per window), `f_n` (pressure), `pen_down`; Case 1: `x_d_req x_d k_req k_diag K D alpha E`; master: `x_m v_m f_h f_fb`; operator `phase stroke` |
| `/obs` | 10 Hz | `rgb_top_camera rgb_wrist_camera` (128², uint8) + the proprio/force fields above |
| `/action` | 10 Hz | `x_d` (world) and `k_diag` (N/m along u, v, n): the applied values at the next frame |
| `/goal` | — | `strokes_uv` (32×128×2) + `mask`, `strokes_world`, the believed paper frame |
| `/privileged` | — | true paper frame, height error, tilt, friction, the ink laid down |

Attributes carry the text, `source` (`synthetic` or `keyboard`), task spec,
writer style, master, gains, criteria and metrics as JSON. Time convention: every state is at the start of its 2 ms step,
so `x_d_req` is the request for the next step.

## Running

```bash
python3 tests.py                                            # 30 checks
python3 collect.py --episodes 200 --workers 8 --out demos/writing_v1 [--video 4]
python3 plot_episode.py demos/writing_v1/ep_00004.h5
python3 evaluate.py --policy scripted --k-t 2000 --k-n 400 --episodes 30 --no-cameras
python3 evaluate.py --policy replay --demos demos/writing_v1 --episodes 10
python3 evaluate.py --policy my_pkg.my_policy:make_policy --episodes 50   # your policy
python3 interactive.py --record demos/keyboard              # your own demos; needs a display
python3 protocol.py --texts S 7 "<star>" --per-case 50      # stiffness protocol; needs a display
```

Evaluation seeds start at 10 000 and demo seeds at 0, so they don't overlap.
`TaskSpec.sample(seed, max_len, dz, tilt_deg)` sets the difficulty. `collect.py`
exposes the same knobs as `--max-len --dz --tilt-deg`.

## What this environment does not do

- **Translation only on the master.** The pen is held vertical by a fixed
  rotational stiffness. Orientation teleoperation is out of scope, as it is in
  `haptic_teleop_fr3_bilateral.py`.
- **Rigid paper.** SAPIEN doesn't expose PhysX compliant contact, so paper
  compliance is not modelled. The impact behaviour above follows from that; the
  pressure filter is the workaround, not a contact model.
- **The tank never closed in the demos.** `E0` = 20 J against 0.9–5.1 J drained
  per episode, almost all of it friction work. The gate and the
  requested-vs-applied split are exercised in `tests.py`, not in the data.
- **The keyboard mode has not been driven by a person yet.** Everything except a
  human at the keys has been tested (see *How this was tested* above).
- The success thresholds are choices. With the synthetic writer, coverage and
  precision never bind; force does, through K_n.
