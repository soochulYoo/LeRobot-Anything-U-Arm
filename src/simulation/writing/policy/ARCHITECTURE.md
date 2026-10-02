# Inputs, outputs and architecture of the eight variants

Two families, four structures each.  The structures are the ones in the CoFA
figure; the families are what the structures are built on.

| family | built on | file |
|---|---|---|
| `flow` | CoFA rectified-flow velocity fields | `model.py` |
| `act` | Comp-ACT's CVAE transformer (arXiv:2406.14990); case **a is Comp-ACT unmodified** | `act_model.py` |

Everything below is per policy frame `t` (the policy runs at 10 Hz).

## What every variant reads and writes

Identical across all eight, so the comparison is about structure alone.

### Input

| name | shape | what it is |
|---|---|---|
| `top` | (64, 64, 3) | third-person camera, 2x2 box-downsampled from 128 |
| `wrist` | (64, 64, 3) | wrist camera |
| `state` | (19,) | `x_d - origin` (3), spring deflection `tcp - x_d` (3), tcp velocity (3), `log K` (3), `qpos` (7) |
| `ft_hist` | (10, 3) | contact force over the last 10 frames -- the only place the paper's hidden height and tilt appear |

The target path is **not** given as a vector.  It used to be (`goal`, 32 points
relative to `x_d`), but handing the policy the path lets it write without ever
reading a camera, and the grey template is legible in the 64x64 input, so the
policy reads the glyph off the paper like the operator did.  `--goal` puts the
vector back for an ablation.

### Output

| name | shape | what it is |
|---|---|---|
| `action` | (10, 6) | per frame `j = 0..9`: `x_d(t+1+j) - x_d(t)` in metres (3) **and `log k_diag(t+1+j)` along the paper u, v, n (3)** |
| `future_ft` | (10, 3) | contact force at frames `t+1 .. t+10`, aligned with the chunk. Case (a) does not produce it |

**Stiffness is half the action.** That is the whole point: the policy commands
where to go *and* how hard to insist on it. At run time the first `n_exec = 3`
set-points (0.3 s) are executed, then the policy samples again.

## The shared front end

```
  top   (64,64,3) ──► CNN ─────┐
  wrist (64,64,3) ──► CNN ─────┼──► h (256)  ─┐
  state (19)      ──► MLP ─────┘              ├──► c = [h | f]   (512)
  ft_hist (10,3)  ──► GRU ──────────► f (256) ┘
```

In the `act` family the same encoders feed a transformer instead: each camera
becomes a 4x4 grid of 16 tokens, and state, wrench and the CVAE latent one token
each -- 35 tokens of width 256, then 4 encoder layers.  The wrench token is
Comp-ACT's `include_ft=True`.

## Family 1 -- `flow` (CoFA)

Rectified flow: `x_tau = (1-tau)*noise + tau*data`, the field predicts
`data - noise`, and sampling is Euler from `tau = 0` to `1` in 10 steps.

### (a) `ft_input` -- the wrench is only an input

```
                 ┌───────────────────────┐
  c ────────────►│                       │
  A_tau (10,6) ─►│   action field  v_A   │──► velocity (10,6)
  tau ──────────►│                       │        │
                 └───────────────────────┘        │
                          ▲                       │
                          └──── Euler x10 ────────┘──► action (10,6)
```
Out: action. No future wrench anywhere.

### (b) `uni_dir` -- force to action, one way

```
                ┌─────────────┐
  c ───────────►│  predictor  │──► Z_hat (10,3) ──► future_ft
                └─────────────┘         │ stop-grad
                                        ▼
                 ┌───────────────────────┐
  c ────────────►│   action field  v_A   │──► velocity ──Euler x10──► action
  A_tau ────────►│                       │
  tau ──────────►└───────────────────────┘
```
The predictor never sees the action, so the action can never correct the force.

### (c) `unified` -- one flow over both

```
                 ┌───────────────────────┐
  c ────────────►│                       │
  [A|Z]_tau ────►│    joint field  v     │──► velocity (10,9) ──Euler x10──► [A|Z]
  tau ──────────►│      (10, 6+3)        │                                    │  │
                 └───────────────────────┘                             action ◄┘  └► future_ft
```
One noise vector and **one flow time** for both.  The two are tied at every step
and cannot be denoised at different rates -- which is what hurts it in practice.

### (d) `cross_cond` -- each field reads the other

```
        ┌──────────────────────────── step k of 10 ────────────────────────────┐
        │                                                                      │
        │   c, Z^k, lam_F, lam_A ──►┌──────────────────┐                        │
        │                           │ action field v_A │──► A^(k+1) ────────────┼──► action
        │   c, A^k, lam_A, lam_F ──►└──────────────────┘                        │
        │            │                      ▲                                  │
        │            │   cross-conditioned  │                                  │
        │            ▼                      │                                  │
        │                           ┌──────────────────┐                        │
        │                           │ wrench field v_F │──► Z^(k+1) ────────────┼──► future_ft
        │                           └──────────────────┘                        │
        └──────────────────────────────────────────────────────────────────────┘
```
Two fields, **independent flow times** `lam_A`, `lam_F` during training so each
learns to read the other anywhere between noise and data; one shared clock at
inference, updated synchronously.

Not implemented from the paper figure: the future-F/T-conditioned
**mixture-of-experts** action field, and the **EMA-encoder latent** as the wrench
target.  `cross_cond` here is the bare cross-conditioned pair.

## Family 2 -- `act` (Comp-ACT)

A CVAE whose decoder emits the whole chunk in one shot.  Comp-ACT's published
configuration: hidden 512, FFN 3200, 4 encoder / 7 decoder layers, 8 heads,
latent 32, `L1 + 100 x KL`, ImageNet-pretrained ResNet18, lr 1e-5, warmup 1000,
grad clip 10.  Case (a) is Comp-ACT unmodified.

### The shared trunk

```
  TRAINING ONLY -- the CVAE style encoder
  ┌──────────────────────────────────────────────────────────┐
  │  [cls] + state(19) + action chunk(10,6)                   │
  │        │                                                   │
  │        ▼  2 encoder layers                                 │
  │   mu, log var  ──►  z ~ N(mu, sigma)   (32)                │
  └──────────────────────────────────────────────────────────┘
        at inference:  z = 0                     │
                                                 ▼
  top   (64,64,3) ─► ResNet18 ─► 2x2x512 ─► 4 tokens ┐
  wrist (64,64,3) ─► ResNet18 ─► 2x2x512 ─► 4 tokens │
  state   (19)    ─► Linear  ──────────────► 1 token │
  goal    (64)    ─► Linear  ──────────────► 1 token ├─► 12 tokens x 512
  ft_hist (10,3)  ─► GRU     ──────────────► 1 token │        │
  z       (32)    ─► Linear  ──────────────► 1 token ┘        │
                                                              ▼
                                                  4 encoder layers
                                                              │
                                                              ▼
                                                        MEMORY (12, 512)
```

`ft_hist` as its own token is Comp-ACT's `include_ft=True, ft_as_obs=False` --
the wrench is given to the transformer, not concatenated onto the state.

### (a) `act_ft_input` -- Comp-ACT, unmodified

```
                 10 learned queries
                          │
   MEMORY ────────────────▼──────────────┐
                 ┌────────────────────┐  │
                 │  action decoder    │  │ 7 layers: self-attn,
                 │  cross-attends the │◄─┘ cross-attn to memory, FFN
                 │  memory            │
                 └─────────┬──────────┘
                           ▼
                     head -> (10, 6)      x_d offset (3) + log K (3)
```
Nothing predicts the future wrench.  Force enters once, through the GRU token.

### (b) `act_uni_dir` -- force to action, one way

```
   MEMORY ──┬──────────────────────────────────────────────┐
            │         10 queries                            │
            ▼    ┌──────────────────┐                       │
                 │  wrench decoder  │──► Z_hat (10, 3) ──► future_ft
                 └──────────────────┘          │
                                               │ stop-grad
                                               │ + frame position
                                               ▼
            │                          10 extra memory tokens
            │         10 queries               │
            ▼    ┌──────────────────┐          │
                 │  action decoder  │◄─────────┘
                 │  memory ++ tokens│
                 └────────┬─────────┘
                          ▼
                    head -> (10, 6)
```
The wrench decoder never sees the action, so the action can never correct it.

### (c) `act_unified` -- one decoder, one head

```
                 10 learned queries
                          │
   MEMORY ────────────────▼──────────────┐
                 ┌────────────────────┐  │
                 │   one decoder      │◄─┘
                 └─────────┬──────────┘
                           ▼
                    head -> (10, 6+3)
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
             action (10,6)     future_ft (10,3)
```
Action and wrench leave the same head, so they cannot be decoded at different
rates -- the one-shot analogue of sharing a flow time.

### (d) `act_cross_cond` -- each decoder reads the other

ACT emits in one shot, so it has no integration axis on which two fields can
meet.  We give it one: in training each stream sees the other's target **blended
with noise at an independently drawn fidelity `lambda`**, embedded alongside it
exactly as the flow time is.  At inference the pair is unrolled from noise, with
`lambda` rising to 1 on the last pass.

```
   ┌───────────────────── pass k of n_refine = 4 ─────────────────────────┐
   │                                                                      │
   │   Z^k (10,3) ──► proj + lambda + frame position ──► 10 tokens        │
   │                                    │                                 │
   │   MEMORY ++ those tokens ──► ┌──────────────────┐                    │
   │                              │  action decoder  │──► A^(k+1) (10,6) ─┼──► action
   │                              └──────────────────┘                    │
   │                                    ▲        cross-conditioned        │
   │                                    │                                 │
   │   A^k (10,6) ──► proj + lambda + frame position ──► 10 tokens        │
   │                                    │                                 │
   │   MEMORY ++ those tokens ──► ┌──────────────────┐                    │
   │                              │  wrench decoder  │──► Z^(k+1) (10,3) ─┼──► future_ft
   │                              └──────────────────┘                    │
   │                                                                      │
   │   both updated synchronously, lambda_k = k / (n_refine - 1)          │
   └──────────────────────────────────────────────────────────────────────┘
      k = 0: both streams are pure noise, lambda = 0
      k = 3: both are near-clean, lambda = 1
```

The frame position embedding matters: without it the 10 wrench tokens arrive as
an unordered set and the decoder cannot read the wrench at frame j against the
action at frame j.

## Sizes

Parameter counts are matched within each family, so a ranking is about structure
and not capacity.  In `act`, case (a) keeps Comp-ACT's width and (b) and (d) --
which carry a second decoder -- are narrowed to match it.

| structure | `flow` | `act` (width) |
|---|---|---|
| a `ft_input` | 2.67 M | 9.30 M (256) |
| b `uni_dir` | 2.67 M | 9.04 M (192) |
| c `unified` | 2.71 M | 9.30 M (256) |
| d `cross_cond` | 2.65 M | 9.05 M (192) |

The flow family now carries the two pieces of the CoFA figure the first version
left out -- a 4-expert MoE action field and a future-F/T latent with an EMA
target encoder -- which is why it is larger than the 1.1-1.4 M of the first run.

The two families are **not** matched to each other: a 10-step flow over a 60-dim
chunk needs far less than a transformer with a 36-token memory.  Compare within
a family; across families, compare behaviour, not parameter efficiency.
