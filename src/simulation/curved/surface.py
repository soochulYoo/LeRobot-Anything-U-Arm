"""A surface with random CURVATURE -- an elephant's back, not a tilted plane.

The distinction matters for the question being asked.  A tilted plane has a
constant normal: a rigid tool aligned once stays aligned forever, and rotational
stiffness never has to do anything.  What makes rotational compliance necessary
is the normal TURNING as the tool travels, and turning appreciably across the
width of the tool itself, so that a rigid pad cannot lie flat on it.

Height is a sum of Gaussian bumps,

    h(x, y) = sum_i  a_i exp( -|| (x,y) - c_i ||^2 / (2 s_i^2) ),

which is smooth (so the normal is well defined and PhysX sees no creases) and
has a curvature scale set by s_i: a bump of amplitude a and width s has radius of
curvature ~ s^2 / a at its crest.  With s = 50 mm and a = 10 mm that is 250 mm,
so across a 50 mm pad the normal swings about 50/250 = 0.2 rad = 11 deg.

The mesh is exported for PhysX; `height` and `normal` stay available in closed
form so alignment can be measured against the true surface rather than against
the mesh's facets.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class CurvedSurface:
    half: float = 0.18          # the patch is 2*half on a side, m
    n_bumps: int = 14
    amp: float = 0.010          # bump amplitude, m
    amp_jitter: float = 0.4     # +-fraction on each bump's amplitude
    sigma: float = 0.050        # bump width, m
    sigma_jitter: float = 0.3
    base: float = 0.015         # slab thickness under the lowest point, m
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Centres are drawn beyond the patch edge too, so the rim is not
        # systematically flatter than the middle.
        m = self.half + self.sigma
        self.c = rng.uniform(-m, m, size=(self.n_bumps, 2))
        self.a = self.amp * (1.0 + self.amp_jitter * rng.uniform(-1, 1, self.n_bumps))
        self.a *= rng.choice([-1.0, 1.0], self.n_bumps)      # dents as well as bumps
        self.s = self.sigma * (1.0 + self.sigma_jitter * rng.uniform(-1, 1, self.n_bumps))

    def height(self, x: Array, y: Array) -> Array:
        x, y = np.asarray(x, float), np.asarray(y, float)
        dx = x[..., None] - self.c[:, 0]
        dy = y[..., None] - self.c[:, 1]
        return np.sum(self.a * np.exp(-(dx ** 2 + dy ** 2) / (2 * self.s ** 2)), axis=-1)

    def grad(self, x: Array, y: Array) -> tuple[Array, Array]:
        x, y = np.asarray(x, float), np.asarray(y, float)
        dx = x[..., None] - self.c[:, 0]
        dy = y[..., None] - self.c[:, 1]
        g = self.a * np.exp(-(dx ** 2 + dy ** 2) / (2 * self.s ** 2)) / self.s ** 2
        return np.sum(-g * dx, axis=-1), np.sum(-g * dy, axis=-1)

    def normal(self, x: Array, y: Array) -> Array:
        """Unit outward normal (+z side) of z = h(x, y), shape (..., 3)."""
        hx, hy = self.grad(x, y)
        n = np.stack([-hx, -hy, np.ones_like(hx)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def key(self, n: int = 161) -> str:
        """Identity of the GEOMETRY, for caching the exported mesh.

        Keying the cache on the seed alone silently served a previously written
        CURVED mesh to a flat-surface run -- same seed, different amplitude --
        and the flat control came out looking like a curved one measured against
        the wrong normal.  Every field that moves a vertex belongs in here.
        """
        import hashlib
        f = (self.half, self.n_bumps, self.amp, self.amp_jitter, self.sigma,
             self.sigma_jitter, self.base, self.seed, n)
        return hashlib.sha1(repr(f).encode()).hexdigest()[:16]

    def mesh(self, n: int = 161):
        """Closed watertight slab whose top face is the height field."""
        import trimesh
        g = np.linspace(-self.half, self.half, n)
        X, Y = np.meshgrid(g, g, indexing="ij")
        Z = self.height(X, Y)
        z_floor = float(Z.min()) - self.base

        def grid_faces(idx, flip):
            f = []
            for i in range(n - 1):
                for j in range(n - 1):
                    a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                    f += [[a, c, b], [a, d, c]] if flip else [[a, b, c], [a, c, d]]
            return f

        top = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        bot = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_floor)], axis=1)
        it = np.arange(n * n).reshape(n, n)
        ib = it + n * n
        faces = grid_faces(it, False) + grid_faces(ib, True)
        # Skirt, so the solid is closed: PhysX treats an open shell as one-sided
        # and the pad can fall through it from below.
        for i in range(n - 1):
            for (p, q) in ((it[i, 0], it[i + 1, 0]), (it[i + 1, -1], it[i, -1]),
                           (it[0, i + 1], it[0, i]), (it[-1, i], it[-1, i + 1])):
                faces += [[p, q, q + n * n], [p, q + n * n, p + n * n]]
        m = trimesh.Trimesh(vertices=np.vstack([top, bot]), faces=np.asarray(faces))
        m.fix_normals()
        return m

    def stats(self, pad: float) -> dict:
        """How hard is this surface for a RIGID pad of half-width `pad`?"""
        rng = np.random.default_rng(self.seed + 991)
        r = self.half - pad - 0.01
        p = rng.uniform(-r, r, size=(4000, 2))
        n0 = self.normal(p[:, 0], p[:, 1])
        worst = np.zeros(len(p))
        for dx, dy in [(pad, 0), (-pad, 0), (0, pad), (0, -pad)]:
            nk = self.normal(p[:, 0] + dx, p[:, 1] + dy)
            worst = np.maximum(worst, np.degrees(np.arccos(
                np.clip(np.sum(n0 * nk, axis=1), -1, 1))))
        tilt = np.degrees(np.arccos(np.clip(n0[:, 2], -1, 1)))
        return {"tilt_mean": tilt.mean(), "tilt_p95": np.percentile(tilt, 95),
                "swing_mean": worst.mean(), "swing_p95": np.percentile(worst, 95),
                "h_ptp": float(np.ptp(self.height(p[:, 0], p[:, 1])))}
