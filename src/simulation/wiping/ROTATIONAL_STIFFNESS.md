# Rotational stiffness on a curved surface — status

All numbers below are a **scripted raster**, single seed, curved board, 40 mm pad.
They are mechanism measurements, not policy results.

## What K_R does, and when it matters

`K_R` is how hard the wrist resists a moment. For a flat tool on a surface:

- **high** — the pad keeps the orientation it was commanded, meets a curved
  surface on an edge, puts its force through a sliver of contact: erases badly,
  presses too hard.
- **low** — the contact moment lays the pad flat on the surface: erases well.
- **too low** — the pad rocks as it travels and the contact force becomes
  unsteady.

So it has an optimum rather than a direction, which is what separates it from
translational stiffness.

**It only matters on a curved surface.** A tilted plane has a constant normal, so
a rigid tool aligned once stays aligned. Measured: on a 10° tilted flat board,
K_R 60 → 0.3 (200x) moved pad misalignment only 9.9° → 8.8° and changed no task
metric at all.

**The existence condition is the tool-to-curvature ratio.** At a 15 mm pad the
normal swings 5.3° across the pad and nothing happens; at 40 mm it swings 13.1°
(p95 21.3°) and everything does.

## Measured

| K_R | 60 | 30 | 10 | **3** | 1 | 0.3 |
|---|---|---|---|---|---|---|
| erased, damping as shipped | 12.9% | 17.7% | 41.9% | **72.6%** | 67.7% | 67.7% |
| erased, damping corrected | 14.5% | 17.7% | 30.6% | 87.1% | 90.3% | **93.5%** |
| force rmse, as shipped | 1.49 | 1.43 | 1.19 | **0.91** | 1.28 | 1.29 |
| force rmse, corrected | 1.47 | 1.40 | 1.15 | **0.92** | 1.16 | 1.31 |
| misalignment | 9.4° | 9.1° | 8.1° | 6.1° | 4.7° | 5.2° |

1. **The effect is large** — one scalar moves erasure from 14.5% to 93.5%.
2. **The force optimum at K_R ≈ 3 is robust.** Correcting the damping
   (`wrist_inertia` 0.01 → 0.20, the true operational-space value; rotational
   damping x4.5) leaves the minimum exactly where it was. The trade-off is real.
3. **The erasure dip at low K_R was an artifact.** With the damping corrected it
   disappears. So "too low and it stops erasing" is wrong; "too low and the
   force gets unsteady" is right.

> **Superseded on the boundary, see `VARIABLE_KR.md`.**  The oracle below steers
> K_R by the normal SWING across the pad, which is the part of the surface a
> rotation CANNOT take out; the demand is the tilt of the best-fit plane under
> the pad.  Read with that correction, and note that the gate the ceiling was
> meant to open -- oracle beats best constant by 5 pp -- came out at +2.5 pp on
> 8 seeds, below criterion, with the per-episode gap at exactly 0.0.

### Ceiling (oracle with the true surface)

`K_R = f_n · r / swing`, per step:

| | erased | force rmse | misalignment |
|---|---|---|---|
| best constant (K_R = 3) | 72.6% | **0.91** | 6.5° |
| **oracle** | **83.9%** | 1.37 | 5.3° |

A ceiling exists: **+11.3 pp of erasure**, paid for in force stability. Letting
the oracle stay stiffer (tolerating 1–3° of misalignment) lost erasure without
buying the force stability back, so a single K_R cannot satisfy both.

### Estimating the surface from observation

Target is the local normal swing in degrees — a property of the environment, not
of the pad's pose. 150 episodes, 45 284 frames (17 223 in contact), split by
episode (30 held out).

| | deg rmse |
|---|---|
| predict the mean | 2.40 |
| **regressor, all inputs** | **1.07** |

Both cameras, contact force, joint torques, proprioception. 55% of the target's
variance is explained, so the surface **is** estimable from observation.

## Bugs found on the way

- **K_R was not settable at runtime.** `reset_state` copies `g.Kr` into
  `ctl.kr` and `advance` integrates it from a rate, so assigning `g.Kr` does
  nothing. The real knob is `Case1Proposal.Ur`, which the gate scales and the
  tank pays for. Every oracle measurement before this fix was void.
- **The erasure model was blind to pad geometry** — it used a disc around the
  pad centre. Now it uses the contact patch implied by the pad's tilt, with the
  felt's compressibility setting the patch width.
- **The pad was a cylinder**, which PhysX resolves to a point contact that
  transmits no moment. It is a box now.

## Next

1. **Plug the regressor into the oracle slot** and measure how much of the
   +11.3 pp it recovers. This is the number the whole direction rests on.
2. **Input ablation** — torque+proprioception only / vision only / all. If
   torque alone suffices, the vision-distillation story weakens, and that is
   worth knowing before building on it.
3. **Mechanism of the low-K_R instability** — torque saturation `δτ`, the tank
   gate `α`, commanded vs achieved impedance. Connect to the **chatter**
   literature: "too much compliance destabilises grinding" is long established,
   and not citing it invites the rediscovery charge.
4. **A multi-objective oracle** that also targets force stability, to see
   whether the trade-off is really irreducible.
5. **The spatially-constrained case** — grinding, where tool attitude is bounded
   and K_R *cannot* be lowered. The more interesting of the two failure modes.
6. **Latent distillation** only if (1) falls short. If the direct regressor
   recovers the ceiling, not needing it is itself the result.

## Conditions that must be reported with any claim

- Single seed, scripted raster. Our policy study showed ±10–17% seed spread.
- The damping design, since conclusions at low K_R depend on it.
- **No six-axis F/T in these recordings** — the contact moment, the most direct
  evidence of how the pad is sitting, reaches the model only through `tau`.
  Re-recording with a wrench sensor would likely make estimation easier and is
  closer to real hardware.
- The pad-to-curvature ratio, which is what makes the task non-trivial at all.
