# Four ways force can talk to a policy

Notation: `f` is the force feature (a GRU summary of the F/T history, or a
predicted future wrench), `h` a hidden feature of the policy, `a` the action.
What separates the mechanisms is **where f enters the computation** -- and each
has an exact counterpart in classical force control.

## 1. Concatenate / extra token

*Force is one more input beside the image and state.*

```
   image tokens  ─┐
   state token   ─┼──►  [ policy trunk ]  ──►  a
   force token f ─┘
        ▲
        └─ f sits in the INPUT SET; the trunk must learn, on its own,
           to route it to whatever depends on it
```

Controls analogy: **a feedforward term** -- force is added to what the
controller knows, but nothing about the controller's structure changes.
Examples: FACTR, TA-VLA (1 token).

Cheapest to add, weakest inductive bias: force competes for attention with every
other token, and nothing forces the policy to use it.

## 2. FiLM / adaLN(-Zero)

*Force rescales and shifts the policy's own features.*

```
                 ┌──────────────┐
   f ───────────►│     MLP      │──► gamma (scale), beta (shift)
                 └──────────────┘        │        │
                                         ▼        ▼
   h ────────────────────────────►  h' = gamma ⊙ h + beta  ──► ... ──► a
                                    (applied at EVERY layer it is inserted)

   adaLN-Zero: gamma, beta start at identity, so the policy begins
               force-agnostic and learns how much to be modulated
```

Controls analogy: **gain scheduling** -- exactly `K(F)` in variable impedance,
where the measured force sets the controller's gains rather than its setpoint.
Examples: FAWAM, Bi-MoDe.

Force cannot be ignored: it multiplies the features rather than sitting beside
them.

## 3. Gate / MoE

*Force picks which sub-policy runs.*

```
   f ──►┌──────────┐
        │  router  │──► w = softmax(...)   (M weights)
        └──────────┘         │
                             │
   h ──┬──►[ expert 1 ]──► v1│
       ├──►[ expert 2 ]──► v2├──► a = Σ_m  w_m · v_m
       └──►[ expert M ]──► vM┘

   (hard gate: one-hot w -- a switch rather than a blend)
```

Controls analogy: **the selection matrix S in hybrid force/position control** --
force decides which directions, or which regime, are under force control at all.
Examples: FoAR, ForceVLA.

Suits contact tasks because they are genuinely regime-switched: free space,
landing, and in-contact want different behaviour from the same observation, and
one network has to average them.

## 4. Cross-attention

*Action queries look up a force memory.*

```
   F/T history (T frames) ──► force memory:  T tokens  (K, V)
                                                 ▲   ▲
                                                 │   │
   action queries ──► Q ─────────────────────────┘───┘
                      │
                      ▼
              softmax(QKᵀ/√d) V  ──► ... ──► a

   the query for chunk frame j can attend to whichever moment of the
   force trace matters -- the impact, the slip, the settling
```

Controls analogy: **a controller that can re-read the recent force trace**,
rather than one that sees only a running summary of it.
Examples: LIFT, ForceVLA2.

The only mechanism that keeps force *resolved in time*.  The other three compress
the history to a single vector before it ever meets the policy.

## Where this repository's eight variants sit

| mechanism | used by | how |
|---|---|---|
| 1. extra token / concat | **all eight** | `ft_hist` (10,3) → GRU → one feature: a token in the ACT memory, a slice of `c = [h \| f]` in the flow trunk.  Also how the *predicted* wrench reaches the flow action field, flattened and concatenated |
| 2. FiLM / adaLN | **none** | -- |
| 3. gate / MoE | flow `a`-`d` | CoFA's `MoEField`: a router reads the wrench state and mixes 4 expert velocity fields |
| 4. cross-attention | `act_uni_dir`, `act_cross_cond` | the predicted wrench becomes 10 tokens (one per chunk frame, with frame positions) appended to the decoder memory, which the action queries attend over |

Two gaps worth noting.

**FiLM is missing, and it is the one with the tightest fit to this task.** The
task *is* variable impedance: the action's second half is `log K`, and the
controls analogy for FiLM is gain scheduling `K(F)`.  A variant that lets the
measured wrench scale and shift the decoder features -- rather than sit beside
them -- is the most direct expression of "force sets the gains" that the four
mechanisms offer.

**The force memory is only one token deep.** Even in mechanism 4 above, the
*measured* history is compressed by the GRU to a single token before it reaches
the transformer; only the *predicted* future wrench is kept per-frame.  Letting
the 10 measured frames enter as 10 tokens would make the cross-attention row
true of the observation as well, which is what LIFT actually does.
