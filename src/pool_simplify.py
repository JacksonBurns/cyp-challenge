"""Phase 2 (NEXT_ITER_PLAN sec 5) - O1: simplify the 19-member cp17 pool to a
small (<=5) strong, decorrelated set, placement UNCHANGED.

Method: NESTED HONEST forward selection. For each scaffold fold (seed 0,
ft_data.csv 'fold', K=5), starting from the empty set, repeatedly add the
candidate member (from the 19 cp17 members) that best improves the TRAIN-fold
z-correlation, up to MAX_MEMBERS. Then score the selected blend on the VAL fold.
Per-isoform nested Pearson (Fisher-z avg) + macro R2 = mean(nested^2) - the exact
repo honest metric (regression_family_block.honest_nested / tabpfn_probe).

Selection is INSIDE each fold on train data only -> no test-set leakage. This is
the honest test of "does a small pool match the big one?".

Also reports, for the best <=5 pool: per-member LOFO gain (honest reselection)
and the paranoid family-block merge (tp_paranoid map) to confirm each member adds
honest, non-redundant signal.

GATE 2: the simplified (<=5) pool must beat cp17 (0.4682) on nested macro R2 to
be the new candidate. If it does not, the iteration is a no-op - keep cp17.

Run (cyp env): ~/miniforge3/envs/cyp/bin/python src/pool_simplify.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
import run_regression as R  # noqa: E402

ISO = R.ISOFORMS
K = 5
MAX_MEMBERS = 5
CP17_BAR = 0.4682
FAMILY_CAP = 0.5

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
folds = np.load(os.path.join(CACHE, "folds.npy"))


def zavg(files):
    dfs = [pd.read_csv(f) for f in files]
    df = dfs[0].copy()
    for iso in ISO:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    return df


def g(tag):
    return sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{tag}*.csv")))


# The 19 cp17 members (same as regression_family_block.POOLS["cp17"])
FAMS = {
    "gbm": [os.path.join(CACHE, "oof_blend.csv")],
    "emb": [os.path.join(CACHE, "oof_emb_concat.csv")],
    "ft_frozen": [os.path.join(CACHE, "ft_oof_frozen.csv")],
    "ft_fullft": g("fullft"),
    "ft_ext": g("ext"),
    "ft_pre": g("pre"),
    "ft_dmpnn": g("dmpnn"),
    "ft_cpmed": g("cpmed"),
    "ft_cpchm": g("cpchm"),
    "ft_chmridge": g("chmridge"),
    "ft_d2d6": g("d2d6"),
    "ft_admchm": g("admchm"),
    "ft_admmed": g("admmed"),
    "ft_unimol": g("unimol"),
    "ft_umtmL1": g("umtmL1"),
    "ft_umtmL2": g("umtmL2"),
    "ft_umtmL3": g("umtmL3"),
    "ft_umtmL4": g("umtmL4"),
    "ft_monridge": [os.path.join(CACHE, "ft_oof_monridge.csv")],
    "ft_tabpfn_cpmed": [os.path.join(CACHE, "ft_oof_tabpfn_cpmed.csv")],
}
CP17 = list(FAMS.keys())  # 19 members
oofs = {k: zavg(FAMS[k]) for k in CP17}
names = list(CP17)

ymaps = {}
for iso in ISO:
    y = targets[R.DIRECT[iso]].values
    ymaps[iso] = (y, ~np.isnan(y))

# precompute per-isoform, per-member z-arrays (labeled rows)
zs_by_iso = {}
for iso in ISO:
    y, m = ymaps[iso]
    zz = {}
    for k in names:
        v = oofs[k][iso].values.astype(float)
        z = np.full(len(v), np.nan)
        z[m] = (v[m] - v[m].mean()) / v[m].std()
        zz[k] = z
    zs_by_iso[iso] = zz


def forward_select(zs, y, m, trm, vam, max_members):
    """Honest per-fold forward selection: add members that improve TRAIN-fold
    z-corr up to max_members, then score on VAL fold. Returns (val_pearson,
    selected_members)."""
    # re-standardize on the train mask (honest, matches honest_nested)
    zt = {}
    for k in zs:
        v = zs[k].copy()
        mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0.0
        sd = v[trm].std()
        zt[k] = (v - mu) / (sd if sd > 0 else 1.0)
    w = {k: 0.0 for k in zt}
    tot = 0.0
    for _ in range(max_members):
        bestk, bestr = None, -2.0
        for k in zt:
            if w[k] > 0:
                continue  # each member added at most once (forward, no replacement)
            cand = sum((w[j] + (1.0 if j == k else 0.0)) * zt[j] for j in zt)
            r = float(pearsonr(y[trm], (cand / (tot + 1.0))[trm]).statistic)
            if r > bestr:
                bestr, bestk = r, k
        if bestk is None:
            break
        w[bestk] += 1.0
        tot += 1.0
    sel = [k for k in w if w[k] > 0]
    zf = sum(w[k] * zt[k] for k in sel)
    val_r = float(pearsonr(y[vam], zf[vam]).statistic)
    return val_r, sel


def nested_forward(max_members):
    """Per-isoform nested Pearson + macro R2 for the forward-selected pool."""
    rs = {iso: [] for iso in ISO}
    sel_counts = {k: 0 for k in names}
    sel_hist = {iso: [] for iso in ISO}
    for iso in ISO:
        y, m = ymaps[iso]
        for f in range(K):
            trm = m & (folds != f)
            vam = m & (folds == f)
            vr, sel = forward_select(zs_by_iso[iso], y, m, trm, vam, max_members)
            rs[iso].append(vr)
            sel_hist[iso].append(sel)
            for k in sel:
                sel_counts[k] += 1
    nested = {iso: float(np.tanh(np.mean(np.arctanh(np.clip(rs[iso], -0.999, 0.999)))))
              for iso in ISO}
    macro = float(np.mean([nested[i] ** 2 for i in ISO]))
    return nested, macro, sel_counts, sel_hist


def nested_fixed(pool_names):
    """Nested honest R2 for a FIXED member set (greedy reselect weights per fold,
    like honest_nested - for LOFO / comparison)."""
    rs = {iso: [] for iso in ISO}
    for iso in ISO:
        y, m = ymaps[iso]
        for f in range(K):
            trm = m & (folds != f)
            vam = m & (folds == f)
            zt = {}
            for k in pool_names:
                v = zs_by_iso[iso][k].copy()
                mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0.0
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1.0)
            w = {k: 0.0 for k in zt}
            tot = 0.0
            for _ in range(12):  # greedy with replacement (matches honest_nested)
                bestk, bestr = None, -2.0
                for k in zt:
                    cand = sum((w[j] + (1.0 if j == k else 0.0)) * zt[j] for j in zt)
                    r = float(pearsonr(y[trm], (cand / (tot + 1.0))[trm]).statistic)
                    if r > bestr:
                        bestr, bestk = r, k
                if bestk is None:
                    break
                w[bestk] += 1.0
                tot += 1.0
            zf = sum(w[k] * zt[k] for k in w)
            rs[iso].append(float(pearsonr(y[vam], zf[vam]).statistic))
    nested = {iso: float(np.tanh(np.mean(np.arctanh(np.clip(rs[iso], -0.999, 0.999)))))
              for iso in ISO}
    return nested, float(np.mean([nested[i] ** 2 for i in ISO]))


print("=== PHASE 2 (O1): nested HONEST forward selection over the 19 cp17 members ===", flush=True)
print(f"bar: cp17 = {CP17_BAR}  |  MAX_MEMBERS = {MAX_MEMBERS}\n", flush=True)

# size sweep 1..5
sweep = {}
for mm in range(1, MAX_MEMBERS + 1):
    nested, macro, sel_counts, sel_hist = nested_forward(mm)
    sweep[mm] = {"nested": {i: round(nested[i], 4) for i in ISO}, "macro": round(macro, 4),
                 "sel_counts": {k: c for k, c in sel_counts.items() if c > 0}}
    top = sorted(sel_counts.items(), key=lambda x: -x[1])[:mm]
    print(f"  size {mm}: macro {macro:.4f}  " + "  ".join(f"{i} {nested[i]:.4f}" for i in ISO) +
          f"  | top-picked: {', '.join(f'{k}({c})' for k, c in top)}", flush=True)

# the 5-member pool
nested5, macro5, sel5, hist5 = nested_forward(MAX_MEMBERS)
print(f"\n5-member pool: nested macro {macro5:.4f} vs cp17 {CP17_BAR}  "
      f"delta {macro5-CP17_BAR:+.4f}", flush=True)

# consensus 5-member set (most-picked across folds)
consensus = [k for k, c in sorted(sel5.items(), key=lambda x: -x[1]) if c > 0][:MAX_MEMBERS]
print(f"consensus 5-member set: {consensus}", flush=True)

# fixed-pool nested for the consensus set + cp17 (sanity: matches 0.4682)
nf_cons, mac_cons = nested_fixed(consensus)
nf_19, mac_19 = nested_fixed(names)
print(f"\nfixed greedy-reselect nested:  consensus-5 {mac_cons:.4f}  |  full-19 {mac_19:.4f}", flush=True)

# LOFO on the consensus 5-member set (honest reselection)
print(f"\n=== LOFO on consensus-5 (honest reselection) ===", flush=True)
lofo = {}
for drop in consensus:
    keep = [k for k in consensus if k != drop]
    _, mac_drop = nested_fixed(keep)
    lofo[drop] = {"gain": round(mac_cons - mac_drop, 4), "pool_wo": round(mac_drop, 4)}
    print(f"  drop {drop:16s}: pool {mac_cons:.4f} -> {mac_drop:.4f}  (gain {mac_cons-mac_drop:+.4f})", flush=True)

# paranoid family-block merge (tp_paranoid map) on consensus-5
TP_PARANOID = {"ft_cpmed": "cpridge", "ft_cpchm": "cpridge", "ft_tabpfn_cpmed": "cpridge",
               "ft_chmridge": "cpridge", "ft_admchm": "admecd", "ft_admmed": "admecd",
               "ft_umtmL2": "umtm", "ft_umtmL3": "umtm", "ft_umtmL1": "chemeleon",
               "ft_umtmL4": "dmpnn", "ft_dmpnn": "dmpnn", "ft_d2d6": "dmpnn",
               "ft_ext": "fullft", "ft_near": "fullft"}
print(f"\n=== paranoid family-block (tp_paranoid) LOFO on consensus-5 ===", flush=True)
fam_lofo = {}
for fam in sorted({TP_PARANOID.get(k, k) for k in consensus
                   if sum(1 for k2 in consensus if TP_PARANOID.get(k2, k2) == TP_PARANOID.get(k, k)) > 1}):
    keep = [k for k in consensus if TP_PARANOID.get(k, k) != fam]
    _, mac_drop = nested_fixed(keep)
    fam_lofo[fam] = {"gain": round(mac_cons - mac_drop, 4), "pool_wo": round(mac_drop, 4)}
    print(f"  drop family {fam:12s}: pool {mac_cons:.4f} -> {mac_drop:.4f}  (gain {mac_cons-mac_drop:+.4f})", flush=True)

gate2 = macro5 > CP17_BAR
print(f"\n=== GATE 2: simplified (<=5) pool vs cp17 ({CP17_BAR}) ===", flush=True)
print(f"  5-member nested macro: {macro5:.4f}  delta {macro5-CP17_BAR:+.4f}", flush=True)
print(f"  GATE 2: {'PASS' if gate2 else 'NOT MET'}", flush=True)
if not gate2:
    print("  -> no-op: cp17 stands, no regression resubmit (pool simplification does not beat the 19-member pool)", flush=True)

out = {
    "bar_cp17": CP17_BAR,
    "max_members": MAX_MEMBERS,
    "size_sweep": sweep,
    "five_member": {"nested": {i: round(nested5[i], 4) for i in ISO}, "macro": round(macro5, 4),
                    "sel_counts": sel5, "consensus_set": consensus},
    "fixed_greedy": {"consensus5": round(mac_cons, 4), "full19": round(mac_19, 4)},
    "lofo_consensus5": lofo,
    "paranoid_tp_lofo": fam_lofo,
    "gate2": {"five_macro": round(macro5, 4), "delta_vs_cp17": round(macro5 - CP17_BAR, 4),
              "pass": bool(gate2)},
    "recipe": "nested honest forward selection (per-fold, train-mask, up to MAX_MEMBERS, no replacement) + Fisher-z avg; macro = mean(nested^2); folds = ft_data seed 0 K=5",
}
with open(os.path.join(CACHE, "pool_simplify.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nsaved pool_simplify.json", flush=True)
print("DONE", flush=True)
