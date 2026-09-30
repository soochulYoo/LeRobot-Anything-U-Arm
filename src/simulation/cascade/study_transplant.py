"""T1, part 2: what frame does each published label rule actually pick, and does
the task survive it?

Part 1 (study_anisotropy.py) varied a synthesised label's three factors and
found the frame to be the one the task cannot absorb.  That left the size of the
real disagreement as an interpretation of the literature.  This measures it.

Each rule in labels3d.py is run on the SAME demonstration, over many surface
tilts, and scored two ways:

  anisotropy      log10( t.K t / n.K n ), against the operator's true value.
                  Positive = correctly stiff along the stroke, soft into the
                  surface.  Negative = inverted.
  transplant      the wipe re-run with that rule's stiffness, scored for success.
                  Run twice: NATIVE (the rule's K as it comes) and FRAME-ONLY
                  (the rule's eigenvector frame, with the expert's magnitude and
                  ratio).  The second isolates the frame, which is the factor
                  under study; the first says what a practitioner would get.

Usage:  python3 study_transplant.py [--tilts 12] [--quick] [--out FIG.png]
"""
from __future__ import annotations

import argparse

import numpy as np

import labels3d as L3
import wipe as W
from study_anisotropy import FORCE_TOL, PATH_TOL, MAG0, RATIO0

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

PRESS, PROBE = 0.06, 2.5


def margin_of(res: W.WipeResult) -> float:
    return max(res.force_rel / FORCE_TOL, res.path_error / PATH_TOL)


def run_one(task: W.WipeTask) -> dict:
    op = W.OperatorParams(press=PRESS, probe_amp=PROBE)
    log = W.demonstrate(task, op)
    d = L3.Demo3D(log=log, settle_s=task.settle_s)
    n_hat, t_hat = task.normal, task.tangent
    K_true = log["_Kh_true"]

    out = {"truth": {"aniso": L3.anisotropy(K_true, n_hat, t_hat)}}
    for tag, fn in L3.ALL_RULES:
        est = fn(d)
        K = est.K
        rec = {"name": est.name, "frame_source": est.frame_source}
        if not np.all(np.isfinite(K)):
            rec.update(aniso=np.nan, angle=np.nan, leak=np.nan, degenerate=True,
                       native=np.nan, frame_only=np.nan,
                       frame_only_lo=np.nan, frame_only_hi=np.nan)
            out[tag] = rec
            continue
        rec["aniso"] = L3.anisotropy(K, n_hat, t_hat)
        rec["angle"], rec["degenerate"] = L3.soft_axis_angle(K, n_hat)
        rec["leak"] = L3.stroke_leak(K, t_hat)
        rec["native"] = margin_of(W.rollout(task, K))
        # frame-only: keep the rule's eigenvectors, impose the expert's
        # magnitude and ratio, so only the frame choice is on trial.  For a
        # DEGENERATE rule the eigenvectors in the tied plane are arbitrary, so a
        # single draw would report luck as if it were a property of the rule.
        # Those get sampled around the tied plane and reported as a range.
        _, _, frame = W.decompose(K)
        if rec["degenerate"]:
            ms = []
            # Spin about the STIFFEST axis (column 0), which mixes columns 1 and
            # 2 and so actually moves the soft axis around the tied plane.
            # Spinning about column 2 leaves the soft axis exactly where it was,
            # which is why the first version reported a zero-width "range".
            for phi in np.linspace(0.0, np.pi, 5, endpoint=False):
                c, s_ = np.cos(phi), np.sin(phi)
                spin = np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])
                ms.append(margin_of(W.rollout(task, W.compose(MAG0, RATIO0, frame @ spin))))
            rec["frame_only"] = float(np.mean(ms))
            rec["frame_only_lo"] = float(np.min(ms))
            rec["frame_only_hi"] = float(np.max(ms))
        else:
            m = margin_of(W.rollout(task, W.compose(MAG0, RATIO0, frame)))
            rec["frame_only"] = rec["frame_only_lo"] = rec["frame_only_hi"] = m
        out[tag] = rec
    out["expert"] = {"native": margin_of(W.rollout(task, W.expert_stiffness(task, MAG0, RATIO0)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilts", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="transplant.png")
    args = ap.parse_args()
    n_tilt = 4 if args.quick else args.tilts

    rng = np.random.default_rng(1)
    tasks = [W.WipeTask(tilt_x=float(a), tilt_y=float(b))
             for a, b in rng.uniform(-0.44, 0.44, size=(n_tilt, 2))]
    print(f"{n_tilt} surfaces at +-25 deg tilt; operator ratio {8.0:.0f}:1 in the TRUE task frame")
    print(f"probe {PROBE} N per axis at {W.OperatorParams().probe_hz} Hz\n")

    runs = []
    for i, task in enumerate(tasks):
        runs.append(run_one(task))
        print(f"  surface {i+1}/{n_tilt} done")

    tags = [t for t, _ in L3.ALL_RULES]
    names = {t: runs[0][t]["name"] for t in tags}
    srcs = {t: runs[0][t]["frame_source"] for t in tags}
    truth = np.array([r["truth"]["aniso"] for r in runs])
    A = {t: np.array([r[t]["aniso"] for r in runs]) for t in tags}
    G = {t: np.array([r[t]["angle"] for r in runs]) for t in tags}
    Lk = {t: np.array([r[t]["leak"] for r in runs]) for t in tags}
    Dg = {t: np.array([bool(r[t]["degenerate"]) for r in runs]) for t in tags}
    Flo = {t: np.array([r[t]["frame_only_lo"] for r in runs]) for t in tags}
    Fhi = {t: np.array([r[t]["frame_only_hi"] for r in runs]) for t in tags}
    N = {t: np.array([r[t]["native"] for r in runs]) for t in tags}
    Fo = {t: np.array([r[t]["frame_only"] for r in runs]) for t in tags}
    ex = np.array([r["expert"]["native"] for r in runs])

    print("\n" + "=" * 104)
    print(f"WHAT EACH RULE RECOVERS   (operator truth: log10(kt/kn) = {np.nanmean(truth):+.2f})")
    print("=" * 104)
    print(f"  {'':4}{'rule':<28}{'frame from':<22}{'log10(kt/kn)':>15}"
          f"{'vs normal':>13}{'stroke leak':>13}{'inverted':>10}{'degen':>7}")
    print("  " + "-" * 108)
    for t in tags:
        a, g, lk = A[t], G[t], Lk[t]
        inv = 100.0 * np.nanmean(a < 0)
        print(f"  {t:<4}{names[t][:26]:<28}{srcs[t][:20]:<22}"
              f"{np.nanmean(a):+7.2f} +-{np.nanstd(a):4.2f}"
              f"{np.nanmean(g):8.0f} deg{np.nanmean(lk):13.2f}"
              f"{inv:9.0f}%{100*np.mean(Dg[t]):6.0f}%")
    print("  " + "-" * 108)
    print("  'inverted' = fraction of surfaces where the rule makes the NORMAL stiffer than")
    print("  the stroke -- the opposite of what the operator did.")
    print("  'stroke leak' = |soft axis . stroke direction|; this, not the angle to the")
    print("  normal, is what predicts failure, because swinging the soft axis toward the")
    print("  CROSS-stroke tangent costs almost nothing.")
    print("  'degen' = the two smallest eigenvalues are within 10%, so the rule does not")
    print("  determine a frame at all and the entry above is one arbitrary draw from a plane.")

    print("\n" + "=" * 104)
    print("TRANSPLANT: re-run the wipe with each rule's stiffness  (margin < 1 = success)")
    print("=" * 104)
    print(f"  {'':4}{'rule':<30}{'native K':>22}{'frame only':>22}")
    print("  " + "-" * 78)
    print(f"  {'--':<4}{'expert (true frame)':<30}{np.nanmean(ex):8.2f}  "
          f"{100*np.mean(ex < 1):3.0f}% ok{'':>12}")
    for t in tags:
        n_, f_ = N[t], Fo[t]
        spread = ("" if not Dg[t].any()
                  else f"   [degenerate: {np.nanmean(Flo[t]):.2f}-{np.nanmean(Fhi[t]):.2f}]")
        print(f"  {t:<4}{names[t][:28]:<30}"
              f"{np.nanmean(n_):8.2f}  {100*np.nanmean(n_ < 1):3.0f}% ok"
              f"{np.nanmean(f_):12.2f}  {100*np.nanmean(f_ < 1):3.0f}% ok" + spread)
    print("  " + "-" * 78)
    print("  'frame only' keeps the rule's frame but gives it the expert's magnitude and")
    print("  ratio, so a failure there is the frame choice alone and nothing else.")

    plot(tags, names, srcs, truth, A, Lk, N, Fo, ex, args.out)


def plot(tags, names, srcs, truth, A, Lk, N, Fo, ex, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(1, 3, figsize=(15.8, 5.2))
    y = np.arange(len(tags))[::-1]
    labels = [f"{t}  {names[t].split(' ', 1)[1][:22]}" for t in tags]

    # --- A: what anisotropy each rule recovers ---
    a = ax[0]
    a.axvline(float(np.nanmean(truth)), color=INK, ls="--", lw=1.6, zorder=2)
    a.annotate(" operator's true value", xy=(float(np.nanmean(truth)), 0.97),
               xycoords=("data", "axes fraction"), color=INK, fontsize=8, va="top")
    a.axvspan(-2, 0, color=INK3, alpha=0.16, lw=0, zorder=1)
    a.annotate("inverted:\nnormal made stiffer\nthan the stroke", xy=(0.02, 0.10),
               xycoords="axes fraction", color=INK2, fontsize=7.5, va="bottom")
    for i, t in enumerate(tags):
        v = A[t][np.isfinite(A[t])]
        if v.size:
            a.plot(v, np.full(v.size, y[i]), "o", color=SLOT[i % len(SLOT)], ms=6,
                   alpha=0.55, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
            a.plot([v.mean()], [y[i]], "|", color=SLOT[i % len(SLOT)], ms=20, mew=2.5, zorder=4)
    a.set_yticks(y); a.set_yticklabels(labels, fontsize=8)
    a.set_xlabel(r"$\log_{10}(\,\hat{k}_{tangent}\,/\,\hat{k}_{normal})$")
    a.set_title("What each rule recovers, over surfaces", color=INK, loc="left")
    a.set_xlim(-1.5, 1.5)
    a.grid(True, axis="x", alpha=0.9); a.set_axisbelow(True)

    # --- B: soft-axis angle ---
    b = ax[1]
    for i, t in enumerate(tags):
        v = Lk[t][np.isfinite(Lk[t])]
        if v.size:
            b.plot(v, np.full(v.size, y[i]), "o", color=SLOT[i % len(SLOT)], ms=6,
                   alpha=0.55, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
            b.plot([v.mean()], [y[i]], "|", color=SLOT[i % len(SLOT)], ms=20, mew=2.5, zorder=4)
    b.axvline(0.42, color=INK, ls=":", lw=1.6)      # sin(25 deg)
    b.annotate(" 0.42 = sin 25$\\degree$,\n the task's tolerance", xy=(0.42, 0.97),
               xycoords=("data", "axes fraction"), color=INK, fontsize=8, va="top")
    b.axvspan(0.42, 1.05, color=INK3, alpha=0.16, lw=0, zorder=1)
    b.set_yticks(y); b.set_yticklabels([])
    b.set_xlabel(r"stroke leak  $|\,u_{soft}\cdot\hat{t}\,|$   (0 = soft axis clear of the stroke)")
    b.set_title("How much soft axis leaks into the stroke", color=INK, loc="left")
    b.set_xlim(-0.04, 1.05)
    b.grid(True, axis="x", alpha=0.9); b.set_axisbelow(True)

    # --- C: transplant success ---
    c = ax[2]
    h = 0.36
    c.barh(y + h / 2, [100 * np.nanmean(N[t] < 1) for t in tags], height=h,
           color=SLOT[0], label="native K", zorder=3)
    c.barh(y - h / 2, [100 * np.nanmean(Fo[t] < 1) for t in tags], height=h,
           color=SLOT[1], label="frame only (expert magnitude + ratio)", zorder=3)
    c.axvline(100 * np.mean(ex < 1), color=INK, ls="--", lw=1.6)
    c.annotate(f" expert: {100*np.mean(ex < 1):.0f}%", xy=(100 * np.mean(ex < 1), 0.97),
               xycoords=("data", "axes fraction"), color=INK, fontsize=8, va="top", ha="left")
    c.set_yticks(y); c.set_yticklabels([])
    c.set_xlabel("wipe success rate over surfaces (%)")
    c.set_title("Transplanting each rule's stiffness", color=INK, loc="left")
    c.set_xlim(0, 108)
    c.grid(True, axis="x", alpha=0.9); c.set_axisbelow(True)
    c.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("T1, measured: the published rules do not agree on which axis should be soft, and the frame alone decides whether the task survives.",
                 color=INK, fontsize=10.5, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out_path, dpi=170)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
