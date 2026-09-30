"""The published stiffness-label rules as 3-D rules, so the FRAME each one picks
can be measured rather than argued about.

Every rule returns a diagonal matrix in some frame.  Which frame is never stated
as a choice, but it is a choice, and the rules do not make the same one:

    world / end-effector axes   ACP, Comp-ACT, Impedance Cloning (per-DOF)
    the direction of MOTION     Imp-ACT ("stiffness along the direction of motion")
    the direction of FORCE      EquiContact-style surface frames
    whatever a fit returns      Compliance-for-Free as a full-matrix regression
    the excited frame           probe identification

On a wipe the motion is tangential and the force is normal, so the second and
third are orthogonal.  This module measures that instead of asserting it.

SCORING.  "Angle between the soft axis and the normal" is degenerate for a
motion-aligned rule -- one distinguished axis leaves the perpendicular plane
with no preferred direction -- so the primary score is the effective stiffness
ratio the rule assigns along the TRUE task directions:

    anisotropy(K) = log10( t_hat . K t_hat  /  n_hat . K n_hat )

positive means correctly stiff along the stroke and soft into the surface;
negative means inverted.  It is well defined for every rule, degenerate or not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def frame_from_axis(u: Array, first: bool = True) -> Array:
    """An orthonormal frame with `u` as its first (or last) column."""
    u = np.asarray(u, dtype=float)
    u = u / max(np.linalg.norm(u), 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = np.cross(u, a); v /= max(np.linalg.norm(v), 1e-12)
    w = np.cross(u, v)
    R = np.column_stack([u, v, w]) if first else np.column_stack([v, w, u])
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1.0
    return R


def spd_project(K: Array, floor: float = 1.0) -> Array:
    """Symmetrise and push eigenvalues up to `floor`.

    A least-squares fit of a stiffness has no reason to come back symmetric or
    positive definite; using it unprojected would hand the controller a matrix
    that can inject energy.  The floor is reported by the caller, because how
    much projection was needed is itself a measure of how ill-posed the fit was.
    """
    Ks = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(Ks)
    return V @ np.diag(np.maximum(w, floor)) @ V.T


def anisotropy(K: Array, n_hat: Array, t_hat: Array) -> float:
    """log10(stiffness along the stroke / stiffness into the surface)."""
    kn = float(n_hat @ K @ n_hat)
    kt = float(t_hat @ K @ t_hat)
    if kn <= 0 or kt <= 0:
        return np.nan
    return float(np.log10(kt / kn))


def stroke_leak(K: Array, t_hat: Array) -> float:
    """How much of K's soft axis points along the STROKE, |u_soft . t_hat|.

    The angle to the normal is the wrong score on its own, and this package's
    own numbers show why: a rule whose soft axis sits 44 deg off the normal can
    still pass, because the tolerance is direction-dependent.  Swinging the soft
    axis toward the stroke trades away exactly the stiffness that holds the tool
    on its path, and fails.  Swinging it toward the CROSS-stroke tangent costs
    almost nothing -- nothing is being dragged that way.  So it is the leak into
    the stroke, not the angle to the normal, that predicts failure.
    """
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    return float(abs(V[:, 0] @ t_hat))


def soft_axis_angle(K: Array, n_hat: Array) -> tuple[float, bool]:
    """Angle (deg) between K's least-stiff eigenvector and `n_hat`, plus a flag
    saying whether that eigenvector is degenerate (the two smallest eigenvalues
    within 10%), in which case the angle is not meaningful on its own."""
    w, V = np.linalg.eigh(0.5 * (K + K.T))
    u = V[:, 0]
    ang = np.rad2deg(np.arccos(np.clip(abs(float(u @ n_hat)), 0.0, 1.0)))
    degenerate = bool(w[1] <= 1.1 * w[0])
    return float(ang), degenerate


# --------------------------------------------------------------------------- #
# demo view
# --------------------------------------------------------------------------- #
@dataclass
class Demo3D:
    log: dict
    settle_s: float = 3.0
    contact_thresh: float = 0.5

    @property
    def dt(self) -> float:
        return float(self.log["_dt"])

    def sig(self, key: str) -> Array:
        a = np.asarray(self.log[key])
        return a[int(round(self.settle_s / self.dt)):]

    @property
    def contact(self) -> Array:
        return self.sig("f_normal")[:, 0] > self.contact_thresh


@dataclass
class Est3D:
    name: str
    K: Array
    frame_source: str            # where this rule's frame came from
    note: str = ""
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# the rules
# --------------------------------------------------------------------------- #
def r1_acp(d: Demo3D, K_min=50.0, K_max=1200.0, f_scale=10.0) -> Est3D:
    """ACP-style per-axis force heuristic, diagonal in the WORLD/EE axes."""
    f = np.abs(d.sig("f_e")[d.contact])
    frac = np.clip(np.median(f, axis=0) / f_scale, 0.0, 1.0)
    return Est3D("R1 ACP force heuristic", np.diag(K_min + (K_max - K_min) * frac),
                 "world / EE axes", "per-axis heuristic; the frame is never chosen, it is inherited")


def r2_compact(d: Demo3D, K_low=120.0, K_high=1500.0) -> Est3D:
    """Comp-ACT-style toggle: preset gains, diagonal in the WORLD/EE axes."""
    f = np.abs(d.sig("f_e")[d.contact])
    loaded = np.median(f, axis=0) > 0.3 * np.median(f, axis=0).max()
    return Est3D("R2 Comp-ACT toggle", np.diag(np.where(loaded, K_low, K_high)),
                 "world / EE axes", "binary preset per axis")


def r3_impact_motion(d: Demo3D) -> Est3D:
    """Imp-ACT-style rule, read along the DIRECTION OF MOTION.

    The frame is built from the mean stroke direction; the stiffness along it
    and across it come from the controller's own force-over-deflection ratio.
    On a wipe the stroke is tangential, so this frame is orthogonal to the
    surface normal by construction.
    """
    v = d.sig("v")[d.contact]
    f = d.sig("f_e")[d.contact]
    dx = (d.sig("x_r") - d.sig("x"))[d.contact]
    speed = np.linalg.norm(v, axis=1)
    fast = speed > 0.3 * np.median(speed[speed > 0]) if np.any(speed > 0) else speed > -1
    u = v[fast].T @ np.sign(v[fast] @ v[fast][np.argmax(speed[fast])])
    u = u / max(np.linalg.norm(u), 1e-12)
    R = frame_from_axis(u, first=True)

    def ratio(axis):
        num = np.abs(f @ axis); den = np.abs(dx @ axis)
        ok = den > 1e-7
        return float(np.median(num[ok] / den[ok])) if ok.sum() > 10 else np.nan

    k_along = ratio(R[:, 0])
    k_perp = float(np.nanmean([ratio(R[:, 1]), ratio(R[:, 2])]))
    k_along = k_along if np.isfinite(k_along) else k_perp
    K = R @ np.diag([k_along, k_perp, k_perp]) @ R.T
    return Est3D("R3 Imp-ACT along motion", spd_project(K), "direction of motion",
                 "the stroke direction, orthogonal to the normal on a wipe",
                 {"motion_axis": u, "k_along": k_along, "k_perp": k_perp})


def r4_compliance_for_free(d: Demo3D) -> Est3D:
    """Compliance-for-Free as a FULL-matrix regression of f_ch on (x_c - x).

    The frame is not chosen at all here: it is whatever the fit's eigenvectors
    come out as, which makes its conditioning part of the result.
    """
    f = d.sig("f_ch")[d.contact]
    dx = (d.sig("x_c") - d.sig("x"))[d.contact]
    K, *_ = np.linalg.lstsq(dx, f, rcond=None)      # dx @ K ~ f  ->  K is 3x3
    cond = float(np.linalg.cond(dx))
    Kp = spd_project(K.T)
    return Est3D("R4 Compliance-for-Free", Kp, "whatever the fit returns",
                 "no frame is chosen; conditioning decides it",
                 {"cond": cond, "asymmetry": float(np.linalg.norm(K - K.T) / max(np.linalg.norm(K), 1e-12))})


def r5_particle_per_axis(d: Demo3D, n_particles=1500, sigma_logK=0.5,
                         sigma_xeq=0.02, sigma_f=0.2, seed=0, decimate=20) -> Est3D:
    """Impedance-Cloning-style filter run PER WORLD AXIS, as the method is posed.

    Running it per axis is not an approximation of a matrix estimator -- it IS
    the published formulation, and it fixes the frame to the world axes without
    ever saying so.
    """
    rng = np.random.default_rng(seed)
    f_all = d.sig("f_ch")[::decimate]
    x_all = d.sig("x")[::decimate]
    ks = []
    for ax in range(3):
        f_obs, x_obs = f_all[:, ax], x_all[:, ax]
        logK = rng.normal(np.log(300.0), 1.0, n_particles)
        x_eq = x_obs[0] + rng.normal(0.0, sigma_xeq, n_particles)
        sq = np.sqrt(d.dt * decimate)
        trace = []
        for f_t, x_t in zip(f_obs, x_obs):
            logK = logK + rng.normal(0.0, sigma_logK * sq, n_particles)
            x_eq = x_eq + rng.normal(0.0, sigma_xeq * sq, n_particles)
            resid = f_t - np.exp(logK) * (x_eq - x_t)
            w = np.exp(-0.5 * (resid / sigma_f) ** 2)
            tot = w.sum()
            if not np.isfinite(tot) or tot <= 1e-300:
                logK = rng.normal(np.log(300.0), 1.0, n_particles)
                x_eq = x_t + rng.normal(0.0, sigma_xeq, n_particles)
                continue
            idx = rng.choice(n_particles, n_particles, p=w / tot)
            logK, x_eq = logK[idx], x_eq[idx]
            trace.append(float(np.exp(np.mean(logK))))
        ks.append(np.mean(trace[len(trace) // 2:]) if trace else np.nan)
    return Est3D("R5 particle filter (per axis)", np.diag(np.nan_to_num(ks, nan=100.0)),
                 "world / EE axes", "per-DOF filtering fixes the frame silently")


def _tone(sig: Array, t: Array, f_hz: float, f_cut_ratio=0.4) -> Array:
    """Complex amplitude of each column of `sig` at f_hz, fitted against a
    low-frequency nuisance basis (see labels._tone_fit for why not a raw DFT)."""
    tt = t - t[0]
    dt = float(t[1] - t[0]); T = float(tt[-1] + dt)
    cols = [np.ones_like(tt), tt]
    for k in range(1, int(np.floor(f_cut_ratio * f_hz * T)) + 1):
        fk = k / T
        if abs(fk - f_hz) < 1.5 / T:
            continue
        cols += [np.cos(2 * np.pi * fk * tt), np.sin(2 * np.pi * fk * tt)]
    cols += [np.cos(2 * np.pi * f_hz * tt), np.sin(2 * np.pi * f_hz * tt)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, sig, rcond=None)
    return beta[-2] - 1j * beta[-1]


def r6_probe(d: Demo3D, freqs=(2.7, 3.9, 5.3)) -> Est3D:
    """Probe identification of the FULL operator stiffness matrix.

    f_h = -Kh x_m - Bh v_m holds at every probe frequency, so three distinct
    frequencies give 18 real equations for the 18 unknowns of (Kh, Bh).  This is
    the only rule that both excites the operator and measures the handle force,
    and the only one whose frame comes from the data rather than from a
    convention.
    """
    t = d.sig("t")
    F, X, V = [], [], []
    for fh in freqs:
        F.append(_tone(d.sig("f_h"), t, fh))
        X.append(_tone(d.sig("x_m"), t, fh))
        V.append(_tone(d.sig("v_m"), t, fh))
    exc = float(np.mean([np.linalg.norm(x) for x in X]))
    if exc < 1e-7:
        return Est3D("R6 probe identification", np.full((3, 3), np.nan), "the data",
                     f"no excitation (|x_m(w)| = {exc:.1e} m)", {"excitation": exc})
    # rows: [Re; Im] of  -F = Kh X + Bh V   for each frequency
    A = np.zeros((6 * len(freqs), 18)); b = np.zeros(6 * len(freqs))
    for k, (Fk, Xk, Vk) in enumerate(zip(F, X, V)):
        for r in range(3):
            for c in range(3):
                A[6 * k + r, 3 * r + c] = Xk[c].real
                A[6 * k + r, 9 + 3 * r + c] = Vk[c].real
                A[6 * k + 3 + r, 3 * r + c] = Xk[c].imag
                A[6 * k + 3 + r, 9 + 3 * r + c] = Vk[c].imag
            b[6 * k + r] = -Fk[r].real
            b[6 * k + 3 + r] = -Fk[r].imag
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    Kh = sol[:9].reshape(3, 3)
    return Est3D("R6 probe identification", spd_project(Kh), "the data (excited)",
                 "excitation + handle sensor",
                 {"excitation": exc, "cond": float(np.linalg.cond(A)),
                  "Bh_hat": sol[9:].reshape(3, 3)})


def r8_force_aligned(d: Demo3D) -> Est3D:
    """A surface-frame rule: the distinguished axis follows the measured FORCE.

    EquiContact-style label sources place compliance relative to the contact
    frame.  On a wipe the mean contact force is the surface normal, so this
    lands orthogonal to the motion-aligned rule above -- from the same record.
    """
    f = d.sig("f_e")[d.contact]
    dx = (d.sig("x_r") - d.sig("x"))[d.contact]
    u = f.mean(axis=0); u = u / max(np.linalg.norm(u), 1e-12)
    R = frame_from_axis(u, first=False)          # force axis LAST = the soft one

    def ratio(axis):
        num = np.abs(f @ axis); den = np.abs(dx @ axis)
        ok = den > 1e-7
        return float(np.median(num[ok] / den[ok])) if ok.sum() > 10 else np.nan

    k_n = ratio(R[:, 2])
    k_t = float(np.nanmean([ratio(R[:, 0]), ratio(R[:, 1])]))
    K = R @ np.diag([k_t, k_t, k_n if np.isfinite(k_n) else k_t]) @ R.T
    return Est3D("R8 force-aligned (surface frame)", spd_project(K), "direction of force",
                 "the contact normal, orthogonal to the motion frame on a wipe",
                 {"force_axis": u})


ALL_RULES: list[tuple[str, Callable[[Demo3D], Est3D]]] = [
    ("R1", r1_acp), ("R2", r2_compact), ("R3", r3_impact_motion),
    ("R4", r4_compliance_for_free), ("R5", r5_particle_per_axis),
    ("R6", r6_probe), ("R8", r8_force_aligned),
]
