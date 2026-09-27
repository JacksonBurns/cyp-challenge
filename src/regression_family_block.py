"""Family-block audit for the REGRESSION blend pools (step 2b, NOTES sec 14 plan).

Rewrites the blend_nested_all pattern so entire MEMBER FAMILIES are held out
together (leave-one-family-out, honestly reselected from the rest), mirroring
src/tdi_blend_family_block2.py. Answers the sec 13/14 question: cpmed/cpchm
share the frozen-ridge lineage (and chmridge shares the frozen-encoder-ridge
mechanism + the CheMeLeon encoder lineage with emb/ft_frozen), so does the cp
gain survive holding out the WHOLE family?

Pools: cp7 (the file on the board) and cp8 (cp7 + ft_chmridge, step 2a).
Family maps reported per pool:
  cp:     cpridge={ft_cpmed,ft_cpchm} (the sec 13 root cause exactly)
  frozen: ridge={ft_cpmed,ft_cpchm,ft_chmridge} (all frozen-encoder ridge probes)
  chem:   chemeleon={emb,ft_frozen,ft_chmridge} (CheMeLeon encoder lineage)
For each pool: fold-nested honest macro R2 (free + family-cap-0.5 greedy),
LOFO honest reselection per multi-member family, family weight concentration.
Writes cache/regression_family_block_audit.json.

Usage (cyp env): python src/regression_family_block.py
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

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
folds = np.load(os.path.join(CACHE, "folds.npy"))
K = 5
FAMILY_CAP = 0.5


def zavg(files):
    dfs = [pd.read_csv(f) for f in files]
    df = dfs[0].copy()
    for iso in R.ISOFORMS:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    return df


def g(tag):
    return sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{tag}*.csv")))


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
    "ft_near": g("near"),
}
POOLS = {
    "cp7": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
            "ft_dmpnn", "ft_cpmed", "ft_cpchm"],
    "cp8": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
            "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge"],
    "cp8d": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
             "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge", "ft_d2d6"],
    "cp9": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
            "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge", "ft_d2d6", "ft_near"],
}
FAMAPS = {
    "cp": {"ft_cpmed": "cpridge", "ft_cpchm": "cpridge"},
    "frozen": {"ft_cpmed": "ridge", "ft_cpchm": "ridge", "ft_chmridge": "ridge"},
    "chem": {"emb": "chemeleon", "ft_frozen": "chemeleon", "ft_chmridge": "chemeleon"},
    "mpnn": {"ft_dmpnn": "dmpnn", "ft_d2d6": "dmpnn"},
    "ftext": {"ft_ext": "fullft", "ft_near": "fullft"},
}


def fam_of(name, amap):
    return amap.get(name, name)


def greedy(zs, y, m, fam=None, cap=None, rounds=12):
    """Greedy blend weights with replacement; optional per-family weight cap."""
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    tot = 0.0
    for _ in range(rounds):
        fsum = None
        if fam is not None and cap is not None and tot > 0:
            fsum = {}
            for k in keys:
                fsum[fam[k]] = fsum.get(fam[k], 0.0) + w[k]
        bestk, bestr = None, -2.0
        for k in keys:
            if fsum is not None and (fsum[fam[k]] + 1.0) / (tot + 1.0) > cap:
                continue
            cand = sum((w[j] + (1.0 if j == k else 0.0)) * zs[j] for j in keys)
            r = float(pearsonr(y[m], (cand / (tot + 1.0))[m]).statistic)
            if r > bestr:
                bestr, bestk = r, k
        if bestk is None:
            break
        w[bestk] += 1.0
        tot += 1.0
    return {k: v / tot for k, v in w.items() if v > 0}


def honest_nested(zs_by_iso, ymaps, cap=None, amap=None):
    """Fisher-averaged honest nested Pearson per iso over a pool.

    zs_by_iso: {iso: {member: z-array}}; cap: None free, else FAMILY_CAP with
    fam labels from amap. Returns (dict iso->r, weight history {iso: [w,folds]}).
    """
    rs = {}
    hist = {}
    for iso in R.ISOFORMS:
        y, m = ymaps[iso]
        ziso = zs_by_iso[iso]
        famd = {k: fam_of(k, amap) for k in ziso} if (cap and amap) else None
        rr = []
        hh = []
        for f in range(K):
            trm = m & (folds != f)
            vam = m & (folds == f)
            zt = {}
            for k in ziso:
                v = ziso[k].copy()
                mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0.0
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1.0)
            # full-length z into greedy with the train mask (like blend_nested_all)
            w = greedy(zt, y, trm, fam=famd, cap=cap)
            hh.append(w)
            zf = sum(w[k] * zt[k] for k in w)
            rr.append(float(pearsonr(y[vam], zf[vam]).statistic))
        rs[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rr, -0.999, 0.999)))))
        hist[iso] = hh
    return rs, hist


def macro(rs):
    return float(np.mean([rs[i] ** 2 for i in R.ISOFORMS]))


def concentration(hist, amap):
    per = []
    for hs in hist.values():
        for w in hs:
            s = {}
            for k, v in w.items():
                f = fam_of(k, amap)
                s[f] = s.get(f, 0.0) + v
            per.append(s)
    fams = sorted({f for s in per for f in s})
    return {f: {"mean": round(float(np.mean([s.get(f, 0.0) for s in per])), 3),
                "max": round(float(np.max([s.get(f, 0.0) for s in per])), 3)}
            for f in fams
            if max(s.get(f, 0.0) for s in per) > 0}


results = {}
for pool_name, pool in POOLS.items():
    names = [n for n in pool if FAMS.get(n)]
    oofs = {k: zavg(FAMS[k]) for k in names}
    ymaps = {}
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        ymaps[iso] = (y, ~np.isnan(y))
    zs_by_iso = {}
    for iso in R.ISOFORMS:
        y, m = ymaps[iso]
        zz = {}
        for k in names:
            v = oofs[k][iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zz[k] = z
        zs_by_iso[iso] = zz

    entry = {}
    base, hist0 = honest_nested(zs_by_iso, ymaps)
    entry["nested"] = {i: round(base[i], 4) for i in R.ISOFORMS}
    entry["nested_macro"] = round(macro(base), 4)

    capped, histc = honest_nested(zs_by_iso, ymaps, cap=FAMILY_CAP, amap={})
    entry["capped"] = {i: round(capped[i], 4) for i in R.ISOFORMS}
    entry["capped_macro"] = round(macro(capped), 4)

    # family-block-honest gate numbers: 0.5 cap on merged cp family, and on
    # the full frozen-ridge family (cpmed+cpchm+chmridge) at once
    cap_cp, _ = honest_nested(zs_by_iso, ymaps, cap=FAMILY_CAP, amap=FAMAPS["cp"])
    entry["capped_cp_macro"] = round(macro(cap_cp), 4)
    cap_frozen, _ = honest_nested(zs_by_iso, ymaps, cap=FAMILY_CAP, amap=FAMAPS["frozen"])
    entry["capped_frozen_macro"] = round(macro(cap_frozen), 4)

    for mapname, amap in FAMAPS.items():
        multi = sorted({fam_of(k, amap) for k in names
                        if sum(1 for k2 in names if fam_of(k2, amap) == fam_of(k, amap)) > 1})
        if not multi:
            continue
        lofo = {}
        for drop in multi:
            keep = [k for k in names if fam_of(k, amap) != drop]
            zsub = {iso: {k: zs_by_iso[iso][k] for k in keep} for iso in R.ISOFORMS}
            rsub, _ = honest_nested(zsub, ymaps)
            lofo[drop] = {
                "nested_macro": round(macro(rsub), 4),
                "gain_free": round(macro(base) - macro(rsub), 4),
                "gain_vs_capped": round(macro(capped) - macro(rsub), 4),
            }
        entry[f"lofo_{mapname}"] = lofo
        entry[f"concentration_{mapname}"] = concentration(hist0, amap)

    results[pool_name] = entry
    print(f"== {pool_name}: nested {entry['nested_macro']} | capped {entry['capped_macro']}")
    for mapname in FAMAPS:
        if f"lofo_{mapname}" in entry:
            print(f"   LOFO[{mapname}] {json.dumps(entry[f'lofo_{mapname}'])}")
            print(f"   conc[{mapname}]  {json.dumps(entry[f'concentration_{mapname}'])}")

with open(os.path.join(CACHE, "regression_family_block_audit.json"), "w") as fh:
    json.dump(results, fh, indent=2)
print("saved cache/regression_family_block_audit.json")
