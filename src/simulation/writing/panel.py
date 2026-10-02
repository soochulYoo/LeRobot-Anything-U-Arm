"""The force a keyboard cannot give back, drawn instead.

A haptic master pushes on the operator's hand with the slave's spring force,
f_fb = kf K_p (p - x_d) (teleop.py): at rest in contact that is the contact
force, and while writing its in-plane part is the paper's friction dragging the
pen back.  A keyboard has no motor, so the same vector is shown in a small
floating window inside the viewer, next to the pen's pressure -- the signal the
ink, the force band and tearing are judged on (sim.py).

The gauges are sliders because the viewer's font is proportional: a bar made
of characters does not line up.  Dragging one does nothing; it is set again on
the next frame.

Nothing here is new data: f_fb and the pressure are what the recorder already
stores as /full/f_fb and /full/f_n.
"""
from __future__ import annotations

import numpy as np

F_MAX = 8.0        # N, right end of the felt normal force
DRAG_MAX = 3.0     # N, either end of the in-plane gauges


def pressure_state(f_n: float, crit) -> str:
    """The pen's pressure in the words the score uses."""
    lo, hi = crit.force_band
    if f_n > crit.tear_force:
        return "TEARING"
    if f_n > hi:
        return "too hard"
    if f_n >= lo:
        return "in band"
    return "too light" if f_n >= crit.ink_force else "no ink"


class ForcePanel:
    """The floating window.  `update` is called once per rendered frame."""
    N_EXTRA = 4

    def __init__(self, viewer, pos=(404, 24), size=(472, 268)):
        from sapien import internal_renderer as R
        from sapien.utils.viewer.plugin import Plugin

        panel = self

        class _Plugin(Plugin):
            def get_ui_windows(self):
                return [panel.ui]

        def gauge(uid: str, label: str, lo: float, hi: float):
            return R.UISliderFloat().Id(uid).Label(label).Min(lo).Max(hi).Value(0.0).WidthRatio(0.5)

        self.normal = gauge("fb_n", "off the paper", 0.0, F_MAX)
        self.along = gauge("fb_u", "left  -  |  +  right", -DRAG_MAX, DRAG_MAX)
        self.across = gauge("fb_v", "toward you  -  |  +  away", -DRAG_MAX, DRAG_MAX)
        self.pressure = gauge("f_n", "pressure", 0.0, 12.0)
        self.band = R.UIDisplayText().Text("")
        self.extra = [R.UIDisplayText().Text("") for _ in range(self.N_EXTRA)]
        self.ui = R.UIWindow().Label("Force feedback").Pos(*pos).Size(*size).append(
            R.UIDisplayText().Text("FELT AT THE HAND  (the master's force feedback, N)"),
            self.normal, self.along, self.across,
            R.UIDisplayText().Text("AT THE PEN  (N)"),
            self.pressure, self.band, *self.extra)
        plugin = _Plugin()
        plugin.init(viewer)
        viewer.plugins.append(plugin)

    def update(self, sim, f_fb=None, extra=()) -> None:
        """`f_fb` in world coordinates (zero if None); `extra` lines go below."""
        u, v, n = np.zeros(3) if f_fb is None else np.asarray(f_fb, dtype=float) @ sim.W
        f_n, cr = float(sim.last["f_n"]), sim.crit
        self.normal.Value(float(n))
        self.along.Value(float(u))
        self.across.Value(float(v))
        self.pressure.Value(f_n).Max(cr.tear_force)
        self.band.Text(f" {pressure_state(f_n, cr)}   (band {cr.force_band[0]:.0f}-{cr.force_band[1]:.0f} N, "
                       f"ink above {cr.ink_force:.1f} N, tears above {cr.tear_force:.0f} N)")
        for row, text in zip(self.extra, list(extra) + [""] * self.N_EXTRA):
            row.Text(text)
