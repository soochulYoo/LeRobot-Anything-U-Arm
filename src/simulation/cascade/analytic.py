"""Closed-form steady states for the cascade, used as the ground truth that
core.py's integrator is tested against.

SIGN / FRAME CONVENTION (used everywhere in this package, 1-DOF per axis):
  - positive direction = "into the wall" (the pushing direction)
  - x_wall is the surface position; penetration is delta = x - x_wall > 0
  - f_e >= 0 is the environment REACTION, i.e. the magnitude that resists
    penetration.  Every dynamics equation therefore reads  M xdd = (drive) - f_e,
    matching haptic_teleop_fr3_bilateral.py's `Vr_dot = solve(Ma, f_ch - f_e - Br@Vr)`.

THE CHAIN (deploy / "policy" mode):
      policy -> outer admittance -> inner Cartesian impedance -> slave -> env
  outer:  Ma xdd_r = f_drive - f_e - Br xd_r,   f_drive = f_d + Ko (x_ref - x_r)
  inner:  f_cmd = Ki (x_r - x) + Di (xd_r - xd)
  slave:  Ms xdd   = f_cmd - f_e
  env:    f_e      = ke (x - x_wall)   for x > x_wall, else 0

At rest every derivative vanishes, so the chain is four springs in SERIES and
the algebra below is exact.  This is the whole point of having this file: the
integrator has an answer to be wrong against.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "contact_force_policy",
    "contact_force_teleop",
    "contact_force_case1",
    "contact_force_case2",
    "routh_case2",
    "routh_da_min",
    "series_stiffness",
    "equilibrium_state_policy",
]


def series_stiffness(*stiffnesses: float) -> float:
    """Stiffness of springs in series: 1/K = sum(1/k_i).  A k_i of inf (rigid)
    contributes nothing; any k_i of 0 makes the series 0."""
    inv = 0.0
    for k in stiffnesses:
        if k == 0.0:
            return 0.0
        if np.isinf(k):
            continue
        inv += 1.0 / k
    return np.inf if inv == 0.0 else 1.0 / inv


def contact_force_policy(f_d, x_ref, Ko, Ki, ke, x_wall):
    """Steady-state contact force when a policy commands (x_ref, f_d, Ko).

    Derivation (all velocities zero):
        slave:  Ki (x_r - x) = f_e          ->  x_r = x + f_e/Ki
        env:    f_e = ke (x - x_wall)       ->  x   = x_wall + f_e/ke
        outer:  f_d + Ko (x_ref - x_r) = f_e
    Substituting:
        f_e * (1 + Ko/ke + Ko/Ki) = f_d + Ko (x_ref - x_wall)

    Note the inner stiffness Ki enters EXACTLY like the environment stiffness:
    a soft inner impedance is another series compliance, so it degrades outer
    force tracking the same way a soft environment does.  That term is absent
    from any single-layer (impedance-only) analysis.

    Returns 0.0 if the solution would not be in contact (f_e <= 0).
    """
    denom = 1.0 + (Ko / ke if np.isfinite(ke) else 0.0) + (Ko / Ki if np.isfinite(Ki) else 0.0)
    f_e = (f_d + Ko * (x_ref - x_wall)) / denom
    return max(0.0, float(f_e))


def equilibrium_state_policy(f_d, x_ref, Ko, Ki, ke, x_wall):
    """Full steady state (f_e, x_slave, x_ref_admittance) for the policy mode."""
    f_e = contact_force_policy(f_d, x_ref, Ko, Ki, ke, x_wall)
    if f_e <= 0.0:
        return 0.0, np.nan, np.nan  # free space: no unique rest position
    x = x_wall + f_e / ke
    x_r = x + f_e / Ki
    return f_e, float(x), float(x_r)


def contact_force_teleop(Kh, reach, Ka, Ki, ke, x_wall, x_c0=0.0, x_m0=0.0):
    """Steady-state contact force for the FULL bilateral teleop chain.

    The human is a spring reaching for a target:  f_h = Kh (x_m0 + reach - x_m).
    With an identity, zero-delay channel the decoded command tracks the master
    one-for-one,  x_c - x_c0 = x_m - x_m0,  and at rest every force in the
    series chain is equal (f_h = f_ch = f_cmd = f_e).  Walking the chain back
    from the wall:
        x     = x_wall + f/ke
        x_r   = x      + f/Ki
        x_c   = x_r    + f/Ka
        x_m   = x_m0 + (x_c - x_c0)
    and substituting into the human spring gives

        f = Kh (reach + x_c0 - x_wall) / (1 + Kh (1/ke + 1/Ki + 1/Ka))

    i.e. the human stiffness in series with EVERYTHING downstream.  This is the
    formula behind the label study: any rule that regresses force against an
    observable displacement recovers some series combination, never Kh alone.
    """
    downstream = 1.0 / series_stiffness(ke, Ki, Ka)  # = 1/ke + 1/Ki + 1/Ka
    f = Kh * (reach + x_c0 - x_wall) / (1.0 + Kh * downstream)
    return max(0.0, float(f))


# --------------------------------------------------------------------------- #
# The two single-interface execution cases (manuscript Sec. III)
#
# Both are deliberately expressed in the SAME convention as the cascade forms
# above, because the one result worth reading off them is a COUNT: how many
# series compliances sit between the commanded quantity and the wall.
#
#     cascade (policy) :  1 + Ko/ke + Ko/Ki          -> two
#     cascade (teleop) :  1 + Kh(1/ke + 1/Ki + 1/Ka) -> three
#     Case 1           :  1 + Kp/ke                  -> ONE
#     Case 2           :  1 + Ka/ke                  -> ONE
#
# Case 1 has one because it has no outer layer to put in series (the
# manuscript's "no independent virtual mass or admittance state"); Case 2 has
# one because it has no inner stiffness -- the vendor servo is a velocity
# source, not a spring.  The cascade pays for having both.  This is the
# quantitative form of the "Ki >> Ko is a requirement" result: it is a
# requirement created by stacking the two layers, and neither single-interface
# case incurs it.
# --------------------------------------------------------------------------- #
def contact_force_case1(u, x_d, Kp, ke, x_wall):
    """Steady-state contact force for Case 1, direct GIC on a torque interface.

    Derivation (all velocities zero, so the damping terms of tau_0 drop out):
        plant:  0 = tau_appl - f_e,   tau_appl = tau_0 + u = -Kp (x - x_d) + u
        env:    f_e = ke (x - x_wall)  ->  x = x_wall + f_e/ke
    Substituting:
        f_e (1 + Kp/ke) = Kp (x_d - x_wall) + u

    `u` is the intentional active torque (the tank-gated term); at u = 0 this is
    pure impedance and the force is set by the equilibrium offset alone.  There
    is NO Ko/Ki term: the GIC stiffness talks to the environment directly.

    Returns 0.0 if the solution would not be in contact.
    """
    denom = 1.0 + (Kp / ke if np.isfinite(ke) else 0.0)
    f_e = (Kp * (x_d - x_wall) + u) / denom
    return max(0.0, float(f_e))


def contact_force_case2(x_c, Ka, ke, x_wall, c=0.0):
    """Steady-state contact force for Case 2, geometric admittance + motion servo.

    At rest v_r = v_s = 0, so the admittance (Eq. 12) reduces to a force balance
    between the coupling spring and the measured contact, and the servo's own
    invariant c = x_r - x_s - Ts v_s fixes the reference/actual offset:
        admittance:  0 = Ka (x_c - x_r) - f_e
        servo:       x_r = x_s + c        (at rest)
        env:         f_e = ke (x_s - x_wall)
    Substituting:
        f_e (1 + Ka/ke) = Ka (x_c - x_wall - c)

    NOTE ON c.  The manuscript is explicit that c "need not be zero after
    saturation or command delay", and that is not a footnote -- it is the whole
    reason a Case 2 steady state is not predictable from the gains alone.  A run
    that clipped its velocity command lands on a DIFFERENT leaf of the
    equilibrium family and settles at a different force, with every gain
    unchanged.  c = 0 is the clean-run default only.

    The servo lag Ts does not appear: it sets whether the equilibrium is
    reached (Prop. 2), never where it is.
    """
    denom = 1.0 + (Ka / ke if np.isfinite(ke) else 0.0)
    f_e = Ka * (x_c - x_wall - c) / denom
    return max(0.0, float(f_e))


def routh_case2(ma, da, ka, ke, Ts):
    """Manuscript Eq. 15: the Case 2 lag-dependent contact-stability condition,

        (ma + da Ts)(da + ka Ts) > ma Ts (ka + ke),

    with da = ba + br.  Returns (stable, margin), where margin is lhs - rhs and
    is negative exactly when the reduced cubic has a right-half-plane pair.

    This is the condition that has no counterpart anywhere in the cascade: the
    cascade's own stability limit (core.stability_limit) is an INTEGRATOR step
    bound, whereas this is a property of the continuous plant.  Positive virtual
    mass, damping and stiffness do not imply it.
    """
    lhs = (ma + da * Ts) * (da + ka * Ts)
    rhs = ma * Ts * (ka + ke)
    return bool(lhs > rhs), float(lhs - rhs)


def routh_da_min(ma, ka, ke, Ts):
    """The smallest total damping da = ba + br that satisfies Eq. 15.

    Solving (ma + da Ts)(da + ka Ts) = ma Ts (ka + ke) for da collapses the
    ka Ts term:

        Ts da^2 + (ma + ka Ts^2) da - ma Ts ke = 0

    so the boundary is a plain quadratic root.  Two things are worth reading off
    it.  The +ka*Ts^2 coefficient is tiny, so the COUPLING STIFFNESS barely
    moves the requirement -- softening Ka does not buy stability.  What the
    requirement tracks is ma*Ts*ke: the ENVIRONMENT stiffness and the servo lag,
    multiplied.  A stiffer workpiece and a slower servo are the same problem, and
    neither is a controller gain the designer chose.
    """
    if Ts <= 0.0:
        return 0.0
    b = ma + ka * Ts ** 2
    return float((-b + np.sqrt(b ** 2 + 4.0 * Ts ** 2 * ma * ke)) / (2.0 * Ts))
