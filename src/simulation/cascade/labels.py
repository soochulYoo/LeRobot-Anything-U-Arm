"""The six published sources of "operator stiffness" labels, reimplemented so
they can be run against a demo whose true Kh is known.

The question is NOT "which rule is most accurate" -- it is whether any of them
is identifying the operator at all, or whether each is reporting a fixed
combination of controller gains that happens to look plausible.  So every rule
carries `recovers`: the closed-form quantity this package PREDICTS it will
return.  study_labels.py checks the prediction, which turns the study into a
test of the analysis rather than an unanchored measurement.

Observability, stated once because it is the crux:
    observable in any bilateral rig : x_m, v_m, x_c, x_r, x, v, f_ch, f_e
    observable only with a handle F/T sensor : f_h
    NEVER observable : the operator's intended target (`reach`)
Every practical rule substitutes an observable proxy for that unobservable
target, and it is the substitution -- not the estimator -- that decides what
gets measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import analytic as A


@dataclass
class Demo:
    """A settled teleop log plus the ground truth that generated it."""
    log: dict
    dt: float
    Kh_true: float
    Ka: float
    Ki: float
    ke: float
    has_handle_sensor: bool = True
    settle_s: float = 5.0
    contact_thresh: float = 0.5

    def _tail(self) -> slice:
        return slice(int(round(self.settle_s / self.dt)), None)

    def sig(self, key: str) -> np.ndarray:
        return np.asarray(self.log[key])[self._tail()].ravel().astype(float)

    @property
    def contact(self) -> np.ndarray:
        return self.sig("f_e") > self.contact_thresh


@dataclass
class Estimate:
    name: str
    value: float
    recovers: str          # what this package predicts the rule returns
    predicted: float       # the closed-form value of `recovers`
    note: str = ""
    diagnostics: dict = field(default_factory=dict)


def _lstsq_1d(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with an explicit conditioning report.  A rule whose
    regressors are collinear returns a number that is decided by floating-point
    noise, so the condition number is part of the result, not a footnote."""
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < X.shape[1] + 2:
        return np.full(X.shape[1], np.nan), np.inf
    cond = float(np.linalg.cond(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, cond


# --------------------------------------------------------------------------- #
# R1 -- ACP-style offline force heuristic
# --------------------------------------------------------------------------- #
def rule_acp_force_heuristic(d: Demo, K_min=50.0, K_max=1000.0, f_scale=10.0) -> Estimate:
    """A monotone hand-tuned map from measured force magnitude to stiffness.

    This is not an estimator: its output is a function of the designer's three
    constants and the force level, and the data enters only through |f_e|.  It
    is included because it is one of the six published label sources, and
    stating plainly that it cannot track Kh is part of the result.
    """
    f = np.abs(d.sig("f_e"))[d.contact]
    if f.size == 0:
        return Estimate("R1 ACP force heuristic", np.nan, "designer constants", np.nan, "no contact")
    frac = float(np.clip(np.median(f) / f_scale, 0.0, 1.0))
    val = K_min + (K_max - K_min) * frac
    return Estimate("R1 ACP force heuristic", val, "designer constants", val,
                    "output is a hyperparameter, not an estimate",
                    {"median_f": float(np.median(f)), "frac": frac})


# --------------------------------------------------------------------------- #
# R2 -- Comp-ACT-style operator toggle
# --------------------------------------------------------------------------- #
def rule_compact_toggle(d: Demo, K_low=100.0, K_high=1500.0) -> Estimate:
    """Operator presses a button for "compliant"; the label is then one of two
    preset gains.  Like R1 the value is a hyperparameter; the only thing the
    data decides is WHEN to switch."""
    frac_contact = float(d.contact.mean())
    val = K_low if frac_contact > 0.5 else K_high
    return Estimate("R2 Comp-ACT toggle", val, "designer constants", val,
                    "binary preset; data only picks the switch time",
                    {"contact_fraction": frac_contact})


# --------------------------------------------------------------------------- #
# R3 -- Imp-ACT-style online controller rule
# --------------------------------------------------------------------------- #
def rule_impact_controller_rule(d: Demo) -> Estimate:
    """Stiffness read off the controller's own tracking error during the demo:
    K = f / (reference - actual).

    In a cascade that ratio is the INNER gain by construction, so this rule
    returns Ki no matter what the operator does.  Predicted: Ki.
    """
    f = d.sig("f_e")[d.contact]
    err = (d.sig("x_r") - d.sig("x"))[d.contact]
    ok = np.abs(err) > 1e-9
    if ok.sum() < 10:
        return Estimate("R3 Imp-ACT controller rule", np.nan, "Ki", d.Ki, "insufficient deflection")
    val = float(np.median(f[ok] / err[ok]))
    return Estimate("R3 Imp-ACT controller rule", val, "Ki", d.Ki,
                    "reads back the inner gain", {"n": int(ok.sum())})


# --------------------------------------------------------------------------- #
# R4 -- Compliance-for-Free regression, x_eq := leader pose
# --------------------------------------------------------------------------- #
def rule_compliance_for_free(d: Demo, against: str = "x") -> Estimate:
    """Regress the commanded force on (leader pose - follower pose).

    Walking the chain at rest:  x_c - x = f/Ka + f/Ki,  so the slope is the
    SERIES stiffness of the coupling and the inner loop -- a pure controller
    quantity with no dependence on Kh at all.  Regressing against x_r instead
    returns Ka exactly.
    """
    f = d.sig("f_ch")[d.contact]
    dx = (d.sig("x_c") - d.sig(against))[d.contact]
    beta, cond = _lstsq_1d(f, dx)
    pred = A.series_stiffness(d.Ka, d.Ki) if against == "x" else d.Ka
    what = "series(Ka, Ki)" if against == "x" else "Ka"
    return Estimate(f"R4 Compliance-for-Free (vs {against})", float(beta[0]), what, pred,
                    "controller gains only; Kh does not enter", {"cond": cond})


# --------------------------------------------------------------------------- #
# R5 -- Impedance-Cloning-style particle filter over (x_eq, K)
# --------------------------------------------------------------------------- #
def rule_particle_filter(
    d: Demo, n_particles: int = 3000, log_K_prior: tuple[float, float] = (np.log(300.0), 1.0),
    sigma_logK: float = 0.5, sigma_xeq: float = 0.02, sigma_f: float = 0.2,
    seed: int = 0, decimate: int = 20,
) -> Estimate:
    """Joint (x_eq, K) estimation with a slow-variation prior on K.

    The measurement is the scalar  f = K (x_eq - x),  so at every step the pair
    (K, x_eq) lies on a 1-D manifold and only their PRODUCT is observed.  Without
    excitation the likelihood is flat along that manifold, the posterior is the
    prior projected onto it, and the filter can fit the force beautifully while
    reporting almost any stiffness.

    `fit_rms` is the filter's OWN one-step-ahead predictive error, not a post-hoc
    replay with the final parameters.  That distinction decides whether the
    result means anything: a badly tuned filter also disagrees across priors, and
    a reviewer would rightly call that a tuning artefact.  The claim only stands
    if every prior setting fits the measured force about equally WELL and still
    disagrees about K, so the fit quality has to be measured the way the filter
    actually predicts.
    """
    rng = np.random.default_rng(seed)
    f_obs = d.sig("f_ch")[::decimate]
    x_obs = d.sig("x")[::decimate]
    if f_obs.size < 10:
        return Estimate("R5 particle filter", np.nan, "the prior", np.nan, "too few samples")

    logK = rng.normal(log_K_prior[0], log_K_prior[1], n_particles)
    x_eq = x_obs[0] + rng.normal(0.0, sigma_xeq, n_particles)
    dt_eff = d.dt * decimate
    sq = np.sqrt(dt_eff)

    resid_sq: list[float] = []
    K_trace: list[float] = []
    collapses = 0
    for f_t, x_t in zip(f_obs, x_obs):
        # slow-variation prior: both states random-walk between observations
        logK = logK + rng.normal(0.0, sigma_logK * sq, n_particles)
        x_eq = x_eq + rng.normal(0.0, sigma_xeq * sq, n_particles)
        K = np.exp(logK)

        f_pred = K * (x_eq - x_t)                 # one-step-ahead prediction
        resid_sq.append(float(np.mean(f_pred) - f_t) ** 2)

        w = np.exp(-0.5 * ((f_t - f_pred) / sigma_f) ** 2)
        tot = w.sum()
        if not np.isfinite(tot) or tot <= 1e-300:  # collapse -> reinitialise and count it
            collapses += 1
            logK = rng.normal(log_K_prior[0], log_K_prior[1], n_particles)
            x_eq = x_t + rng.normal(0.0, sigma_xeq, n_particles)
            continue
        idx = rng.choice(n_particles, n_particles, p=w / tot)
        logK, x_eq = logK[idx], x_eq[idx]
        K_trace.append(float(np.exp(np.mean(logK))))

    if not K_trace:
        return Estimate("R5 particle filter (x_eq, K)", np.nan, "the prior", np.nan,
                        "filter collapsed at every step")
    # report the posterior mean over the settled second half of the record
    K_hat = float(np.mean(K_trace[len(K_trace) // 2:]))
    return Estimate("R5 particle filter (x_eq, K)", K_hat, "the prior", np.nan,
                    "unidentifiable without excitation: only K*(x_eq-x) is observed",
                    {"fit_rms": float(np.sqrt(np.mean(resid_sq))),
                     "signal_rms": float(np.sqrt(np.mean(f_obs ** 2))),
                     "collapses": collapses,
                     "logK_std": float(np.std(logK))})


# --------------------------------------------------------------------------- #
# R6 -- probe-based identification (needs excitation AND a handle sensor)
# --------------------------------------------------------------------------- #
def _tone_fit(signals: dict[str, np.ndarray], t: np.ndarray, f_hz: float,
              f_cut_ratio: float = 0.5) -> tuple[dict[str, complex], float, int]:
    """Least-squares fit of a known-frequency tone against a low-frequency
    nuisance basis, returning each signal's complex amplitude at f_hz.

    A raw DFT is wrong here and was the first version's bug: 1/(f_hz*dt) is not
    an integer sample count, so trimming to "whole cycles" does not put f_hz on
    a bin, and the DC level -- two orders of magnitude larger than the probe
    amplitude -- leaks straight into the probe bin.  Fitting instead projects
    the probe tone against {1, t} plus every harmonic of 1/T below
    f_cut_ratio*f_hz, so DC, drift and the operator's slow intent are removed by
    construction.  Only a CUTOFF is assumed, not the intent frequency, which is
    what makes this usable on real data.

    Convention: y ~ a cos(wt) + b sin(wt)  ->  amplitude (a - 1j b).
    """
    tt = t - t[0]
    dt = float(t[1] - t[0])
    T = float(tt[-1] + dt)
    cols = [np.ones_like(tt), tt]
    f_cut = f_cut_ratio * f_hz
    n_harm = 0
    for k in range(1, int(np.floor(f_cut * T)) + 1):
        f_k = k / T
        if abs(f_k - f_hz) < 1.5 / T:      # never let a nuisance column collide with the probe
            continue
        w_k = 2.0 * np.pi * f_k
        cols += [np.cos(w_k * tt), np.sin(w_k * tt)]
        n_harm += 1
    w = 2.0 * np.pi * f_hz
    cols += [np.cos(w * tt), np.sin(w * tt)]   # the probe tone: always the last two columns
    X = np.column_stack(cols)
    cond = float(np.linalg.cond(X))
    out: dict[str, complex] = {}
    for name, y in signals.items():
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[name] = complex(beta[-2], -beta[-1])
    return out, cond, n_harm


def rule_probe_identification(d: Demo, probe_hz: float = 3.0) -> Estimate:
    """Identify the operator's arm impedance from the device-side probe.

    Model:  f_h = Kh (target(t) - x_m) - Bh v_m.  At the probe frequency the
    intent contributes nothing, so with complex amplitudes F, X, V at that
    frequency
        F = -Kh X - Bh V
    which is one complex equation, i.e. two real ones, in the two real unknowns
    (Kh, Bh) -- exactly determined, solved as a 2x2 system below.

    V is taken from the LOGGED v_m rather than assumed equal to j w X.  The two
    differ: the log's x_m and v_m satisfy the discrete relation x_m(t) =
    x_m(t-dt) + dt v_m(t), so V/X = (1 - exp(-j w dt))/dt, which carries a real
    part of about 0.36 at a 3 Hz probe and would bias Kh upward through Bh.
    Using the logged V removes that approximation entirely.

    This is the only rule of the six that uses independent excitation, and the
    only one needing f_h -- a force sensor on the handle.  Both requirements are
    precisely what the published label sources lack, which is the finding.
    """
    if not d.has_handle_sensor:
        return Estimate("R6 probe identification", np.nan, "Kh", d.Kh_true,
                        "requires a handle force sensor")
    t = d.sig("t")
    amps, cond, n_harm = _tone_fit(
        {"f": d.sig("f_h"), "x": d.sig("x_m"), "v": d.sig("v_m")}, t, probe_hz)
    F, X, V = amps["f"], amps["x"], amps["v"]
    diag = {"excitation_amp_m": abs(X), "design_cond": cond, "n_nuisance_harmonics": n_harm}
    if abs(X) < 1e-6:
        return Estimate("R6 probe identification", np.nan, "Kh", d.Kh_true,
                        f"no excitation at {probe_hz} Hz (|x_m(w)|={abs(X):.2e} m)", diag)
    M = np.array([[X.real, V.real], [X.imag, V.imag]])
    if abs(np.linalg.det(M)) < 1e-18:
        return Estimate("R6 probe identification", np.nan, "Kh", d.Kh_true,
                        "x_m and v_m collinear at the probe frequency", diag)
    Kh_hat, Bh_hat = np.linalg.solve(M, np.array([-F.real, -F.imag]))
    diag["Bh_hat"] = float(Bh_hat)
    diag["probe_hz"] = probe_hz
    return Estimate("R6 probe identification", float(Kh_hat), "Kh", d.Kh_true,
                    "excitation + handle sensor", diag)


def rule_naive_regression_with_intercept(d: Demo) -> Estimate:
    """The control case that shows WHY a probe is needed.

    Regress f_h on (1, -x_m, -v_m): the intercept stands in for the operator's
    intent.  With a CONSTANT intent this is exact and needs no probe at all.
    Once the intent varies in time the single intercept cannot represent it, the
    intent leaks into the x_m column, and the estimate degrades -- which is the
    identifiability claim stated as an experiment rather than an assertion.
    """
    if not d.has_handle_sensor:
        return Estimate("R7 naive regression + intercept", np.nan, "Kh", d.Kh_true,
                        "requires a handle force sensor")
    f, xm, vm = d.sig("f_h"), d.sig("x_m"), d.sig("v_m")
    X = np.column_stack([np.ones_like(xm), -xm, -vm])
    beta, cond = _lstsq_1d(f, X)
    return Estimate("R7 naive regression + intercept", float(beta[1]), "Kh", d.Kh_true,
                    "exact only if the operator's intent is constant",
                    {"cond": cond, "intercept": float(beta[0])})


ALL_RULES: list[tuple[str, Callable[[Demo], Estimate]]] = [
    ("R1", rule_acp_force_heuristic),
    ("R2", rule_compact_toggle),
    ("R3", rule_impact_controller_rule),
    ("R4", rule_compliance_for_free),
    ("R4b", lambda d: rule_compliance_for_free(d, against="x_r")),
    ("R5", rule_particle_filter),
    ("R6", rule_probe_identification),
    ("R7", rule_naive_regression_with_intercept),
]
