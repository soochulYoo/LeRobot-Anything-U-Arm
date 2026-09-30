"""Single-stroke glyphs: what the operator intends to write.

A writing task needs its target as STROKES -- ordered polylines with the pen
lifted between them -- not as filled outlines.  Outline fonts (TrueType,
matplotlib's TextPath) describe the boundary of the ink, which a pen tip cannot
draw; a stroke font describes the pen's path.  Nothing installed here ships one,
so a small Hershey-style set is defined below: A-Z, 0-9 and a few shapes, on a
4-wide x 6-tall grid with the baseline at y = 0.

Everything returned is in CANVAS coordinates, metres: (u, v) with u along the
writing direction and v up the letter.  scene.py owns the map from canvas to
world, so this module has no idea where the paper is -- which is the point: the
same target drives the synthetic operator, the template drawn on the paper and
the success check, and none of them can disagree about what was asked for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

GRID_W, GRID_H = 4.0, 6.0


def _arc(cx, cy, rx, ry, a0, a1, step_deg: float = 10.0):
    """Ellipse arc from angle a0 to a1 (degrees, either direction)."""
    n = max(2, int(np.ceil(abs(a1 - a0) / step_deg)) + 1)
    a = np.deg2rad(np.linspace(a0, a1, n))
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in a]


def _s(*parts):
    """Concatenate points and arc point-lists into one stroke."""
    out: list = []
    for p in parts:
        if isinstance(p, list):
            out.extend(p)
        else:
            out.append(p)
    return out


# Each glyph is a list of strokes; each stroke is a list of (x, y) grid points.
# Stroke ORDER and DIRECTION follow ordinary handwriting (top-down, left-right),
# since the operator writes them in this order and a policy imitates that.
GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "A": [[(0, 0), (2, 6), (4, 0)], [(0.8, 2.4), (3.2, 2.4)]],
    "B": [[(0, 6), (0, 0)],
          _s((0, 6), (2.4, 6), _arc(2.4, 4.5, 1.5, 1.5, 90, -90), (0, 3)),
          _s((0, 3), (2.5, 3), _arc(2.5, 1.5, 1.5, 1.5, 90, -90), (0, 0))],
    "C": [_arc(2, 3, 2, 3, 45, 315)],
    "D": [[(0, 6), (0, 0)], _s((0, 6), (1.5, 6), _arc(1.5, 3, 2.5, 3, 90, -90), (0, 0))],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [_s(_arc(2, 3, 2, 3, 60, 360), (2.2, 3))],
    "H": [[(0, 6), (0, 0)], [(4, 6), (4, 0)], [(0, 3), (4, 3)]],
    "I": [[(1, 6), (3, 6)], [(2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "J": [_s((3.5, 6), (3.5, 1.5), _arc(1.75, 1.5, 1.75, 1.5, 0, -180))],
    "K": [[(0, 6), (0, 0)], [(4, 6), (0, 2)], [(1.3, 3.3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 2), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [_arc(2, 3, 2, 3, 90, 450)],
    "P": [[(0, 6), (0, 0)], _s((0, 6), (2.5, 6), _arc(2.5, 4.5, 1.5, 1.5, 90, -90), (0, 3))],
    "Q": [_arc(2, 3, 2, 3, 90, 450), [(2.5, 1.5), (4, 0)]],
    "R": [[(0, 6), (0, 0)], _s((0, 6), (2.5, 6), _arc(2.5, 4.5, 1.5, 1.5, 90, -90), (0, 3)),
          [(1.8, 3), (4, 0)]],
    "S": [_s(_arc(2, 4.5, 2, 1.5, 20, 270), _arc(2, 1.5, 2, 1.5, 90, -160))],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [_s((0, 6), (0, 2), _arc(2, 2, 2, 2, 180, 360), (4, 6))],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1, 0), (2, 4), (3, 0), (4, 6)]],
    "X": [[(0, 6), (4, 0)], [(4, 6), (0, 0)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
    "0": [_arc(2, 3, 1.8, 3, 90, 450)],
    "1": [[(1, 4.8), (2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "2": [_s(_arc(2, 4.3, 1.8, 1.7, 160, -30), (0, 0), (4, 0))],
    "3": [_s(_arc(2, 4.5, 1.8, 1.5, 150, -90), _arc(2, 1.5, 1.9, 1.5, 90, -150))],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [_s((3.8, 6), (0.4, 6), (0.2, 3.2), _arc(2, 1.9, 1.9, 1.9, 120, -150))],
    "6": [_s((3.2, 6), (0.21, 2.55), _arc(2, 1.9, 1.9, 1.9, 160, -200))],
    "7": [[(0, 6), (4, 6), (1.5, 0)]],
    "8": [_arc(2, 4.5, 1.6, 1.5, -90, 270), _arc(2, 1.5, 1.9, 1.5, 90, 450)],
    "9": [_s(_arc(2, 4.1, 1.9, 1.9, 0, 360), (3.0, 0))],
}
# Shapes, drawn in the same 4 x 6 box so they lay out like characters.
SHAPES: dict[str, list[list[tuple[float, float]]]] = {
    "circle": [_arc(2, 3, 2, 2, 90, 450)],
    "square": [[(0, 1), (0, 5), (4, 5), (4, 1), (0, 1)]],
    "triangle": [[(2, 5.5), (0, 0.5), (4, 0.5), (2, 5.5)]],
    "star": [[(2 + 2 * np.cos(np.deg2rad(90 + 144 * k)), 3 + 2.6 * np.sin(np.deg2rad(90 + 144 * k)))
              for k in range(6)]],
    "spiral": [[(2 + 0.34 * t * np.cos(t), 3 + 0.34 * t * np.sin(t))
                for t in np.linspace(0, 3.2 * np.pi, 60)]],
    "wave": [[(x, 3 + 1.5 * np.sin(2 * np.pi * x / 2.0)) for x in np.linspace(0, 4, 40)]],
}


def available() -> list[str]:
    return sorted(GLYPHS) + [f"<{k}>" for k in sorted(SHAPES)]


def resample(poly: Array, spacing: float) -> Array:
    """Points every `spacing` metres along a polyline, endpoints kept."""
    poly = np.asarray(poly, dtype=float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-12:
        return poly[:1].copy()
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    q = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(q, s, poly[:, i]) for i in range(poly.shape[1])])


@dataclass
class Target:
    """The thing to be written, in canvas metres.

    `strokes` is ordered and directed: the operator writes stroke 0 first,
    from its first point to its last.  `points` is the union of all strokes
    resampled densely, which is what the success check covers.
    """
    text: str
    strokes: list[Array]
    letter_height: float
    spacing: float = 0.001

    @property
    def points(self) -> Array:
        return np.concatenate([resample(s, self.spacing) for s in self.strokes], axis=0)

    @property
    def length(self) -> float:
        return float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in self.strokes))

    def bbox(self) -> tuple[Array, Array]:
        p = np.concatenate(self.strokes, axis=0)
        return p.min(axis=0), p.max(axis=0)

    def padded(self, max_strokes: int = 32, max_pts: int = 128) -> tuple[Array, Array]:
        """Fixed-shape (max_strokes, max_pts, 2) array plus a validity mask, for
        storing the goal next to the observations."""
        out = np.zeros((max_strokes, max_pts, 2), np.float32)
        mask = np.zeros((max_strokes, max_pts), bool)
        for i, s in enumerate(self.strokes[:max_strokes]):
            r = resample(s, max(self.length / (max_strokes * max_pts), 0.002))
            if len(r) > max_pts:
                r = r[np.linspace(0, len(r) - 1, max_pts).round().astype(int)]
            out[i, :len(r)] = r
            mask[i, :len(r)] = True
        return out, mask


def tokenize(text: str) -> list[str]:
    """'HI<star>' -> ['H', 'I', '<star>'].  Unknown characters raise, so a typo
    in a task list fails at load time rather than as an empty episode."""
    toks, i = [], 0
    while i < len(text):
        if text[i] == "<":
            j = text.index(">", i)
            name = text[i + 1:j]
            if name not in SHAPES:
                raise KeyError(f"unknown shape <{name}>; have {sorted(SHAPES)}")
            toks.append(text[i:j + 1])
            i = j + 1
            continue
        c = text[i].upper()
        if c != " " and c not in GLYPHS:
            raise KeyError(f"no glyph for {text[i]!r}; have {''.join(sorted(GLYPHS))}")
        toks.append(c)
        i += 1
    return toks


def layout(text: str, letter_height: float = 0.035, tracking: float = 1.6,
           slant: float = 0.0) -> Target:
    """Lay out `text` on one line, centred on the canvas origin.

    `tracking` is the gap between characters in grid units (a character is 4
    wide).  `slant` shears the glyphs (radians, positive leans right), a cheap
    way to vary the demonstrations' style without changing what they spell.
    """
    unit = letter_height / GRID_H
    strokes: list[Array] = []
    x0 = 0.0
    for tok in tokenize(text):
        if tok != " ":
            src = SHAPES[tok[1:-1]] if tok.startswith("<") else GLYPHS[tok]
            for s in src:
                p = np.asarray(s, dtype=float)
                p = np.column_stack([p[:, 0] + np.tan(slant) * p[:, 1], p[:, 1]])
                strokes.append((p + [x0, 0.0]) * unit)
        x0 += GRID_W + tracking
    if not strokes:
        raise ValueError("nothing to write")
    lo = np.min([s.min(axis=0) for s in strokes], axis=0)
    hi = np.max([s.max(axis=0) for s in strokes], axis=0)
    c = 0.5 * (lo + hi)
    return Target(text=text, strokes=[s - c for s in strokes], letter_height=letter_height)
