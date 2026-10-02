"""What is in a collected wiping dataset, and what the protocol did in it.

The attempt log says how many demonstrations were thrown away and why; the
episodes themselves say what the stiffness schedule bought.  The second is the
reason `mis` and `ask` are recorded per step rather than averaged: the protocol
holds K_R HIGH to land and LOW to wipe, so an average over all pad-down time is
half landing and reads as no effect.

    python3 summary.py demos/wipe_curved demos/wipe_flat
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "writing"))

import teleop as T  # noqa: E402  for PHASES

CONTACT = [T.PHASES.index(p) for p in ("press", "stroke", "finish")]
WIPE = [T.PHASES.index("stroke")]
LAND = [T.PHASES.index(p) for p in ("settle", "descend")]


def report(d: pathlib.Path) -> None:
    import h5py
    files = sorted(d.rglob("ep_*.h5"))
    print(f"\n=== {d}  ({len(files)} episodes)")
    log = d / "attempts.jsonl"
    if log.exists():
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        kept = sum(r["kept"] for r in rows)
        why = collections.Counter(r["reason"] for r in rows if not r["kept"])
        print(f"  attempts {len(rows)}, kept {kept} ({100*kept/max(1,len(rows)):.0f}%)"
              + ("" if not why else "  dropped: "
                 + ", ".join(f"{k or 'unknown'} x{v}" for k, v in why.most_common())))
    if not files:
        return

    agg = collections.defaultdict(list)
    for p in files:
        with h5py.File(p) as f:
            m = json.loads(f.attrs["metrics"])
            ph = f["full/phase"][:]
            mis, ask, kr = (f["full/mis"][:], f["full/ask"][:], f["full/kr"][:])
            dn = f["full/pen_down"][:].astype(bool)
            agg["erased"].append(m["erased"])
            agg["in_band"].append(m["in_band"])
            agg["peak"].append(m["peak_force"])
            agg["t"].append(m["t"])
            agg["steps"].append(len(ph))
            agg["ask"].append(ask[dn].mean() if dn.any() else np.nan)
            # TOUCHDOWN: the first step of each contact, which is the moment
            # `K_R HIGH to land` exists to protect.  A pad that arrives tilted
            # meets the board on an edge, and an edge strike shows up as a
            # force spike -- so both are measured here, not just the angle.
            td = np.flatnonzero(dn & ~np.r_[False, dn[:-1]])
            if len(td):
                agg["td_mis"].append(float(mis[td].mean()))
                agg["td_ask"].append(float(ask[td].mean()))
                fn = f["full/f_n"][:]
                w = min(20, len(fn))     # 0.2 s at the 100 Hz log rate
                agg["td_peak"].append(float(np.mean(
                    [fn[i:i + w].max() for i in td])))
                agg["td_n"].append(len(td))
            for name, sel in (("wipe", np.isin(ph, WIPE) & dn),
                              ("land", np.isin(ph, LAND) & dn),
                              ("contact", np.isin(ph, CONTACT) & dn)):
                if sel.any():
                    agg[f"mis_{name}"].append(mis[sel].mean())
                    agg[f"left_{name}"].append(mis[sel].mean() / max(ask[sel].mean(), 1e-6))
                    agg[f"kr_{name}"].append(kr[sel].mean())

    def s(k, f="{:.2f}"):
        v = np.asarray(agg[k], dtype=float)
        return f.format(np.nanmean(v)) + " +- " + f.format(np.nanstd(v))

    print(f"  erased {s('erased')}   in-band {s('in_band')}   peak {s('peak', '{:.1f}')} N"
          f"   {s('t', '{:.1f}')} s   {np.mean(agg['steps']):.0f} log rows")
    print(f"  the board asks {s('ask', '{:.1f}')} deg over the glyph")
    print(f"  touchdown ({np.mean(agg['td_n']):.1f} per episode):"
          f" misaligned {s('td_mis', '{:4.1f}')} deg of {s('td_ask', '{:4.1f}')} asked,"
          f" force within 0.2 s {s('td_peak', '{:4.1f}')} N")
    for name in ("land", "wipe"):
        print(f"  {name:5s}: K_R {s('kr_' + name, '{:5.2f}')} Nm/rad"
              f"   misalignment {s('mis_' + name, '{:4.1f}')} deg"
              f"   left {s('left_' + name, '{:.2f}')} of what was asked")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    for d in ap.parse_args().dirs:
        report(pathlib.Path(d))
    print()


if __name__ == "__main__":
    main()
