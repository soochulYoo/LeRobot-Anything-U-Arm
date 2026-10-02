"""A surface whose curvature changes from BAND to BAND along the wipe.

`../curved/surface.py` makes a uniformly bumpy slab: the normal swings about the
same amount everywhere, so one well-chosen constant K_R serves the whole
traverse.  Measured, and it does -- a constant 0.3 beat the oracle there.  That
is the wrong environment for asking whether stiffness should ADAPT.

Here the slab alternates flat bands and curved ones along u, the long axis of
the raster, so a single traverse crosses several regimes:

    |  flat  |  curved  |  flat  |  curved  |      <- u, the wiping direction
      K_R high  K_R low   high     low             <- what each band wants

A constant has to pick one and be wrong half the time: stiff enough to stay
steady on the flat bands rides the curved ones on an edge, and soft enough to
conform on the curved bands wanders on the flat ones.

Implementation keeps the parent's closed form exactly.  The bands are applied to
the bump AMPLITUDES, not to the height field, so the result is still a plain sum
of Gaussians -- no mask gradient, no crease, and `grad` and `normal` are the
parent's unchanged.

    python3 patchy_surface.py            # print what the bands look like
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from curved.surface import CurvedSurface          # noqa: E402


def band(x, period: float, sharp: float = 0.45):
    """0 on the flat bands, 1 on the curved ones, smooth in between."""
    return np.clip(0.5 + np.sin(2.0 * np.pi * np.asarray(x) / period) / max(sharp, 1e-6),
                   0.0, 1.0)


@dataclasses.dataclass
class PatchySurface(CurvedSurface):
    # Chosen by sweeping for CONTRAST at a 40 mm pad (see __main__):
    # flat bands 1.3 deg of swing, curved bands 12.5 deg -- 9.7x.  The two sit
    # on either side of the thresholds we measured earlier: 5.3 deg was where
    # nothing happened at all, 13.1 deg was where K_R moved erasure from 14% to
    # 94%.  So one traverse crosses a regime where K_R is irrelevant and one
    # where it decides the task, which is what a constant cannot serve.
    #
    # The bumps have to stay ABOUT THE PAD'S SIZE.  Narrower ones were tried and
    # are worse than useless: a pad wider than the bump bridges it and feels no
    # curvature at all.  That forces the bands to be wide too -- a band narrower
    # than the bumps is just the bumps spilling across it.
    sigma: float = 0.045
    amp: float = 0.014
    n_bumps: int = 16
    period: float = 0.34        # m, one flat + one curved band
    sharp: float = 0.25         # smaller -> squarer bands
    floor: float = 0.0          # residual amplitude on the flat bands

    def __post_init__(self) -> None:
        super().__post_init__()
        # Scale each bump by the band its CENTRE falls in.  Doing it here and
        # not in height() is what keeps the sum-of-Gaussians form.
        w = band(self.c[:, 0], self.period, self.sharp)
        self.a = self.a * (self.floor + (1.0 - self.floor) * w)

    def key(self, n: int = 161) -> str:
        return (f"patchy_{super().key(n)}_{self.period:.3f}"
                f"_{self.sharp:.2f}_{self.floor:.2f}")

    def band_at(self, x) -> np.ndarray:
        return band(x, self.period, self.sharp)


def swing(surf, xy, r):
    n0 = surf.normal(xy[..., 0], xy[..., 1])
    out = np.zeros(len(xy))
    for d in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
        q = xy + np.asarray(d)
        n1 = surf.normal(q[..., 0], q[..., 1])
        out = np.maximum(out, np.degrees(np.arccos(
            np.clip(np.einsum("ij,ij->i", n0, n1), -1.0, 1.0))))
    return out


if __name__ == "__main__":
    r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.040
    for name, s in (("uniform", CurvedSurface(seed=0)), ("patchy", PatchySurface(seed=0))):
        u = np.linspace(-0.12, 0.12, 241)
        xy = np.column_stack([u, np.zeros_like(u)])
        sw = swing(s, xy, r)
        b = s.band_at(u) if hasattr(s, "band_at") else np.ones_like(u)
        flat, curved = b < 0.2, b > 0.8
        print(f"{name:8s} swing over the traverse: mean {sw.mean():5.2f} deg,"
              f" p5 {np.percentile(sw, 5):5.2f}, p95 {np.percentile(sw, 95):5.2f}")
        if flat.any() and curved.any():
            print(f"         flat bands {sw[flat].mean():5.2f} deg"
                  f"   curved bands {sw[curved].mean():5.2f} deg"
                  f"   ratio {sw[curved].mean()/max(sw[flat].mean(), 1e-9):.1f}x")


@dataclasses.dataclass
class DomeSurface(CurvedSurface):
    """One centred dome -- pottery, not an elephant's back.

    A single Gaussian whose crest radius is `R = sigma^2 / amp`, so the curvature
    is strong, smooth and everywhere the same sign.  That makes it the HARDEST
    case for adaptation and the easiest for a constant: a uniform curvature asks
    for one stiffness everywhere, so there is nothing for a schedule to switch
    between.  It is also where a flat pad stops being able to conform at all --
    lying on a dome needs the pad to deform by the sagitta r^2/2R, and the felt
    only gives so much.
    """
    radius: float = 0.25        # m, crest radius of curvature

    def __post_init__(self) -> None:
        super().__post_init__()
        self.c = np.zeros((1, 2))
        self.s = np.array([self.sigma])
        self.a = np.array([self.sigma ** 2 / self.radius])

    def key(self, n: int = 161) -> str:
        return f"dome_{self.radius:.3f}_{self.sigma:.3f}_{n}"


@dataclasses.dataclass
class CorrugatedSurface(CurvedSurface):
    r"""Flat facets that alternate tilt -- the only shape inside the window.

    Decomposing the surface under the pad into the TILT of the best-fit plane
    and the RESIDUAL to it splits it into the part a rotation can take out and
    the part no stiffness can.  The felt absorbs `pad_give`, so a board is only
    asking the wrist for something when

        residual  <=  pad_give  <  2 r tan(tilt)

    -- the residual small enough that the pad can seat at the right angle, the
    tilt big enough that the felt cannot swallow it.  A Gaussian bump about the
    pad's own size fails this on BOTH sides at once (its residual is the
    sagitta, which is what exceeds `pad_give` first), which is why every bumpy
    and domed board measured here was won by a constant.  What satisfies it is a
    locally FLAT but tilted patch, so the board is a trapezoidal corrugation:

        _____            _____            plateau: asks for 0, and a wrist still
             \          /                   carrying the last flank's tilt
              \        /                     erases less of it -- the cost of
               \______/                      softness, in the task's own metric
         plateau  flank  plateau

    A constant cannot serve both: soft enough to acquire the flank's tilt is
    still rotating when it reaches the plateau, stiff enough to hold the plateau
    rides the flank on an edge.  The profile is one-dimensional (no y
    dependence) deliberately -- an across-track demand would be a second
    mechanism and this gate is testing one.

    Corners are rounded by `corner` (a tanh, so the surface is C-infinity and
    PhysX sees no crease).  `height(0, 0) = 0` and the board centre is flat, so
    the hover height and every episode's start pose are what they are on the
    flat board, as `CurvedFrame` requires.
    """
    plateau: float = 0.060      # m, flat run
    flank: float = 0.040        # m, tilted run
    slope_deg: float = 12.0     # deg, the flank's tilt
    corner: float = 0.004       # m, tanh rounding at each slope change

    def __post_init__(self) -> None:
        super().__post_init__()                 # c/a/s are unused: height is overridden
        self.pitch = self.plateau + self.flank
        # Enough periods to cover the slab with room for the footprint probes.
        k = int(np.ceil((self.half + 4 * self.pitch) / self.pitch))
        j = np.arange(-k, k + 1)
        self.b = self.plateau / 2 + j * self.pitch          # flank starts
        self.sgn = np.where(j % 2 == 0, 1.0, -1.0) * np.tan(np.radians(self.slope_deg))
        self._h0 = float(self._raw(np.zeros(1))[0])

    # ---- profile ---------------------------------------------------------
    def _step(self, x):
        """Smoothed unit step, (1 + tanh(x/w))/2."""
        return 0.5 * (1.0 + np.tanh(x / self.corner))

    def _ramp(self, x):
        """Its integral, so that d/dx _ramp = _step.  Stable for large |x|."""
        z = x / self.corner
        return 0.5 * (x + self.corner * (np.abs(z) + np.log1p(np.exp(-2 * np.abs(z)))))

    def _raw(self, u):
        u = np.asarray(u, float)[..., None]
        return np.sum(self.sgn * (self._ramp(u - self.b)
                                  - self._ramp(u - self.b - self.flank)), axis=-1)

    def _slope(self, u):
        u = np.asarray(u, float)[..., None]
        return np.sum(self.sgn * (self._step(u - self.b)
                                  - self._step(u - self.b - self.flank)), axis=-1)

    # ---- the CurvedSurface interface -------------------------------------
    def height(self, x, y):
        x = np.asarray(x, float)
        return self._raw(x) - self._h0 + 0.0 * np.asarray(y, float)

    def grad(self, x, y):
        x = np.asarray(x, float)
        return self._slope(x), np.zeros_like(x) + 0.0 * np.asarray(y, float)

    def key(self, n: int = 161) -> str:
        return (f"corr_{self.half:.3f}_{self.plateau:.3f}_{self.flank:.3f}"
                f"_{self.slope_deg:.2f}_{self.corner:.4f}_{self.base:.3f}_{n}")

    def band_at(self, x) -> np.ndarray:
        """1 on the flanks (tilt asked), 0 on the plateaus (tilt zero)."""
        return np.abs(self._slope(x)) / max(np.tan(np.radians(self.slope_deg)), 1e-9)
