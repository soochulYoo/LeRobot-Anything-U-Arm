"""Compare the four structures over their seeds: which one to use, and why.

Reads, for every structure and seed,
    policy/runs/<name>/seed<k>/train_log.json        open loop, held-out demos
    policy/runs/<name>/seed<k>/eval/results.json     closed loop, unseen papers
and prints one table with mean +- std over seeds.

The ranking is on CLOSED-LOOP SUCCESS, pooled over the three texts: it is the
only number that asks what the policy is for.  The open-loop columns are there
to explain a ranking, not to set it -- a chunk can be close to the recorded one
in millimetres and still fail, most often by choosing a normal stiffness that
does not forgive the paper's unknown height and tilt.  The stiffness columns say
what the policy actually chose against what the demonstrations chose.

Usage (from src/simulation/writing):
    python3 policy/compare.py                       # all four, all seeds
    python3 policy/compare.py --out policy/runs/COMPARISON.md
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict

STRUCTS = ["ft_input", "uni_dir", "unified", "cross_cond"]
# act_film and act_ft_tokens are mechanism arms, not CoFA structures: see
# policy/FORCE_CONDITIONING.md for the four ways force can reach a policy.
MECH = ["act_film", "act_ft_tokens"]
FAMILY = {"flow": STRUCTS, "act": [f"act_{n}" for n in STRUCTS] + MECH}
ORDER = FAMILY["flow"] + FAMILY["act"]
def case_of(name: str) -> str:
    """The structure letter for a run name, arm suffix/prefix stripped.  CASE is
    keyed by base names only, so a lookup on 'ft_input_nf' raises."""
    return CASE.get(split_arm(name)[0], "?")


CASE = {n: c for n, c in zip(STRUCTS, "abcd")}
CASE.update({f"act_{n}": c for n, c in zip(STRUCTS, "abcd")})
CASE.update({"act_film": "e", "act_ft_tokens": "a+"})
TITLE = {"flow": "CoFA rectified-flow fields", "act": "Comp-ACT CVAE transformer (a = Comp-ACT itself)"}
TASKS = ["S", "7", "<star>"]

# ARMS: run-name suffixes that are the SAME structure under a different
# condition, so they belong in the same table rather than a separate one.  The
# whole point of the control arms is that the RANKING is read across them:
#   _nf   the force input is zeroed.  If a still beats c by ~40 pp with no force
#         to condition on, the ranking was never about force handling -- it was
#         optimisation difficulty.  (K error is 0.0-0.7% for every structure, and
#         the ranking tracks pos1_mm and the action loss, which is what raised
#         the suspicion.)
#   _nft  the future-force target is zeroed too, so b/c/d's extra machinery has
#         nothing to predict and the comparison is purely architectural.
#   _leadN  (d) only, and not an ablation but a FIX: (d) trains with independent
#         flow times and used to sample on the diagonal lamA = lamF, a
#         measure-zero slice of its own training distribution.  Diffusion Forcing
#         (arXiv:2407.01392) avoids a shared clock for exactly this reason.
# name -> (prefix, suffix, label).  A prefix as well as a suffix because
# model.run_name() puts the layout in front ("spring_ft_input") and the
# input/sampling arms behind ("ft_input_nf"); a suffix-only reader silently
# dropped every spring run from the table.
ARMS = {
    "":       ("",        "",       "as trained"),
    "nf":     ("",        "_nf",    "CONTROL: force INPUT zeroed"),
    "nft":    ("",        "_nft",   "CONTROL: force input AND target zeroed"),
    "lead3":  ("",        "_lead3", "FIX: wrench field 3 steps ahead at inference"),
    "sur":    ("",        "_sur",   "force as SURPRISE: measured | expected | residual (N), ensemble deploy"),
    "surz":   ("",        "_surz",  "surprise, STANDARDIZED z + single-model deploy (the -24 pp version)"),
    "springrel": ("springrel_", "", "SPRING, position re-anchored on measurement"),
    "spring": ("spring_", "",       "SPRING-CONSISTENT action: x_ref + f_d + log K"),
}


def arm_key(x: str) -> str:
    """Accept '_nf' and 'nf' alike, so a stale command line still works."""
    x = x.strip()
    return x if x in ARMS else x.lstrip("_").rstrip("_")


def expand(base: list, arm: str) -> list:
    pre, suf, _ = ARMS[arm]
    return [f"{pre}{n}{suf}" for n in base]


def split_arm(name: str) -> tuple:
    """Longest-token match first, so 'spring_x_nf' is not read as plain 'nf'."""
    for k in sorted((k for k in ARMS if k), key=lambda k: -(len(ARMS[k][0]) + len(ARMS[k][1]))):
        pre, suf, _ = ARMS[k]
        if name.startswith(pre) and name.endswith(suf) and len(name) > len(pre) + len(suf):
            return name[len(pre): len(name) - len(suf) or None], k
    return name, ""


def mean_std(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0)


def fmt(m, s, scale=1.0, prec=1):
    if math.isnan(m):
        return f"{'-':>13}"
    return f"{scale * m:>7.{prec}f}+-{scale * s:<5.{prec}f}"


def read_runs(root: pathlib.Path, name: str):
    """One entry per seed: the last training row and the closed-loop summary."""
    out = []
    for d in sorted((root / name).glob("seed*")):
        row = {"seed": d.name}
        log = d / "train_log.json"
        if log.exists():
            try:
                rows = json.loads(log.read_text())
                row["train"] = rows[-1] if rows else None
            except json.JSONDecodeError:
                row["train"] = None          # a run still writing its log
        ev = d / "eval" / "results.json"
        if ev.exists():
            try:
                row["eval"] = json.loads(ev.read_text())
            except json.JSONDecodeError:
                row["eval"] = None
        out.append(row)
    return [r for r in out if r.get("train") or r.get("eval")]


def pooled(ev, key):
    """Mean over every episode of a run.  An episode that never touched the
    paper has no contact stiffness, so NaNs are dropped rather than poisoning
    the mean -- the success column already counts those episodes as failures."""
    eps = (ev or {}).get("episodes") or []
    vals = [float(e[key]) for e in eps
            if e.get(key) is not None and not math.isnan(float(e[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def per_task(runs):
    """Per-text means over seeds, straight out of each run's summary block."""
    out = {}
    for i, t in enumerate(TASKS):
        vals = defaultdict(list)
        for r in runs:
            s = ((r.get("eval") or {}).get("summary") or {}).get(t)
            if not s:
                continue
            for k, v in s.items():
                if isinstance(v, (int, float)) and not math.isnan(float(v)):
                    vals[k].append(float(v))
        ref = ((runs[0].get("eval") or {}).get("demo_reference") or {}).get(str(i), {})
        out[t] = ({k: mean_std(v) for k, v in vals.items()}, ref)
    return out


def demo_ref(runs):
    for r in runs:
        ref = (r.get("eval") or {}).get("demo_reference") or {}
        if ref:
            vals = list(ref.values())
            return {k: sum(v[k] for v in vals) / len(vals)
                    for k in ("contact_ku", "contact_kn", "approach_kn")}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="policy/runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--family", choices=["flow", "act", "all"], default="all")
    ap.add_argument("--arms", nargs="+", default=[""],
                    help='arms to include as extra sections, e.g. --arms "" nf spring '
                         '(see ARMS at the top; "_nf" is accepted for "nf")')
    a = ap.parse_args()
    root = pathlib.Path(a.runs)

    base = ORDER if a.family == "all" else FAMILY[a.family]
    a.arms = [arm_key(x) for x in a.arms]
    unknown = [x for x in a.arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")
    wanted = [n for arm in a.arms for n in expand(base, arm)]
    data = {n: read_runs(root, n) for n in wanted}
    data = {n: v for n, v in data.items() if v}
    if not data:
        raise SystemExit(f"no runs under {root}")

    L = []
    L.append("closed loop: 20 unseen paper randomizations per text, pooled over S / 7 / <star>")
    L.append("open loop:   held-out demonstrations (the last 5 episodes of each text)")
    L.append("note:        'val total' is each structure's own objective, so it does NOT compare")
    L.append("             across rows -- b, c and d add a future-force term.  'action loss' is")
    L.append("             the action-flow part alone, which does.")
    L.append("")
    head = (f"{'structure':<13}{'seeds':>6}{'success':>14}{'coverage':>14}{'precision':>14}"
            f"{'in band':>14}{'peak N':>14}{'K_n contact':>14}{'K_u contact':>14}"
            f"{'pos1 mm':>14}{'posH mm':>14}{'acc xy %':>14}{'acc z %':>14}{'ft N':>14}"
            f"{'action loss':>14}{'val total':>14}")
    L.append(head)
    L.append("-" * len(head))

    stats = {}
    shown = None
    for n, runs in data.items():
        nb, arm = split_arm(n)
        fam = "act" if nb.startswith("act_") else "flow"
        key = (fam, arm)
        if key != shown:
            t = TITLE[fam] + ("" if not arm else f"  --  {ARMS[arm][2]}")
            L.append(f"-- {t} " + "-" * max(0, len(head) - len(t) - 4))
            shown = key
        tr = [r["train"] for r in runs if r.get("train")]
        ev = [r["eval"] for r in runs if r.get("eval")]
        g = lambda rows, k: [r.get(k) for r in rows if r]
        s = {
            "success": mean_std([pooled(e, "success") for e in ev]),
            "coverage": mean_std([pooled(e, "coverage") for e in ev]),
            "precision": mean_std([pooled(e, "precision") for e in ev]),
            "in_band": mean_std([pooled(e, "in_band") for e in ev]),
            "peak": mean_std([pooled(e, "peak_force") for e in ev]),
            "kn": mean_std([pooled(e, "contact_kn") for e in ev]),
            "ku": mean_std([pooled(e, "contact_ku") for e in ev]),
            "pos1": mean_std(g(tr, "pos1_mm")), "posH": mean_std(g(tr, "posH_mm")),
            "acc_xy": mean_std(g(tr, "acc_xy")), "acc_z": mean_std(g(tr, "acc_z")),
            "ft": mean_std(g(tr, "ft_N")), "val": mean_std(g(tr, "val_loss")),
            # b, c and d carry a future-force term in their total, so only the
            # action-flow part compares across structures (a has nothing else).
            "l_a": mean_std([r.get("l_a", r.get("loss")) for r in tr if r]),
        }
        s["_success_seeds"] = [pooled(e, "success") for e in ev]
        stats[n] = s
        L.append(f"{case_of(n)} {n:<11}{len(runs):>6}"
                 + fmt(*s["success"], 100, 1) + fmt(*s["coverage"], 100, 1)
                 + fmt(*s["precision"], 100, 1) + fmt(*s["in_band"], 100, 1)
                 + fmt(*s["peak"], 1, 2) + fmt(*s["kn"], 1, 0) + fmt(*s["ku"], 1, 0)
                 + fmt(*s["pos1"], 1, 2) + fmt(*s["posH"], 1, 2)
                 + fmt(*s["acc_xy"], 100, 1) + fmt(*s["acc_z"], 100, 1)
                 + fmt(*s["ft"], 1, 2) + fmt(*s["l_a"], 1, 3) + fmt(*s["val"], 1, 3))

    ref = demo_ref([r for runs in data.values() for r in runs])
    if ref:
        L.append("-" * len(head))
        L.append(f"{'demonstrations':<19}{'':>14}{'':>14}{'':>14}{'':>14}{'':>14}"
                 f"{ref['contact_kn']:>14.0f}{ref['contact_ku']:>14.0f}"
                 + "   <- what the operator's protocol chose")

    # ---------------------------------------------------------------- per task
    tasks = {n: per_task(runs) for n, runs in data.items()}
    if any(any(v[0] for v in t.values()) for t in tasks.values()):
        L += ["", "per task: how often it works, how hard it presses, and how closely the",
              "stiffness it chose matches the operator's -- |K - K_demo| / K_demo over u and n",
              ""]
        th = (f"{'structure':<19}{'task':>8}{'success':>10}{'contact N':>11}{'K_u':>8}{'K_n':>8}"
              f"{'K_u demo':>10}{'K_n demo':>10}{'K error':>9}")
        L += [th, "-" * len(th)]
        for n, t in tasks.items():
            for task in TASKS:
                st, ref = t[task]
                if not st:
                    continue
                g = lambda k: st.get(k, (float("nan"), 0))[0]
                err = [abs(g(f"contact_{a_}") - ref[f"contact_{a_}"]) / ref[f"contact_{a_}"]
                       for a_ in ("ku", "kn") if ref.get(f"contact_{a_}")]
                L.append(f"{case_of(n) + ' ' + n:<19}{task:>8}{100 * g('success'):>9.0f}%"
                         f"{g('contact_f'):>11.2f}{g('contact_ku'):>8.0f}{g('contact_kn'):>8.0f}"
                         f"{ref.get('contact_ku', float('nan')):>10.0f}"
                         f"{ref.get('contact_kn', float('nan')):>10.0f}"
                         f"{100 * sum(err) / len(err) if err else float('nan'):>8.1f}%")

    ranked = [n for n in stats if not math.isnan(stats[n]["success"][0])]
    ranked.sort(key=lambda n: -stats[n]["success"][0])
    if ranked:
        for fam, arm in [(f, s) for s in a.arms for f in ("flow", "act")]:
            r = [n for n in ranked
                 if split_arm(n)[0] in FAMILY[fam] and split_arm(n)[1] == arm]
            if len(r) < 2:
                continue
            L += ["", f"ranking within {TITLE[fam]}"
                  + ("" if not arm else f"  [{ARMS[arm][2]}]") + ":"]
            for i, n in enumerate(r, 1):
                m, sd = stats[n]["success"]
                L.append(f"  {i}. ({case_of(n)}) {n:<16} {100 * m:5.1f}% +- {100 * sd:.1f}")
        L += ["", "ranking over everything by closed-loop success:"]
        for i, n in enumerate(ranked, 1):
            m, s = stats[n]["success"]
            L.append(f"  {i}. ({case_of(n)}) {n:<16} {100 * m:5.1f}% +- {100 * s:.1f}")
        best = ranked[0]
        if len(ranked) > 1:
            L += ["", f"paired per-seed difference in success against {best} "
                      "(negative = worse than the winner):"]
            for n in ranked[1:]:
                xs, ys = stats[n]["_success_seeds"], stats[best]["_success_seeds"]
                k = min(len(xs), len(ys))
                d = [xs[i] - ys[i] for i in range(k)]
                m, s = mean_std(d)
                L.append(f"  {n:<16} {100 * m:+6.1f} +- {100 * s:.1f}   "
                         f"lost {sum(1 for x in d if x < 0)}/{k} seeds")
    else:
        L += ["", "no closed-loop results yet -- the table above is open loop only, "
                  "which cannot rank these structures on its own."]

    text = "\n".join(L)
    print(text)
    if a.out:
        p = pathlib.Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("```\n" + text + "\n```\n")
        print(f"\nwritten to {p}")


if __name__ == "__main__":
    main()
