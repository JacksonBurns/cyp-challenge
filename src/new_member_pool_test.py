"""Reusable honest pool test: does a NEW member lift the cp17 pool?

Usage:  ~/miniforge3/envs/cyp/bin/python src/new_member_pool_test.py <member_tag>
  e.g.  ... new_member_pool_test.py schnet
  e.g.  ... new_member_pool_test.py llmpharm

Loads the cp17 19-member pool + the new member (cache/ft_oof_<tag>.csv,
seed-averaged if <tag> is a family prefix), runs the EXACT repo honest greedy
blend (per scaffold fold, train-fit weights, val-score Pearson, Fisher-z
nested, macro = mean(nested^2)), and reports:
  - new member's standalone honest nested per isoform
  - cp17 macro vs cp17+new macro (delta)
  - per-isoform deltas (so we can see if it's 2D6-specific vs broad)
  - decorrelation: new member's avg |pearson| vs the cp17 OOF blend

Gate: delta > +0.004 (above fold noise ~0.004) AND no isoform regresses > 0.01.

Output: cache/new_member_<tag>_pooltest.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
folds = ft["fold"].values

# cp17 pool (19 members) - same as llm_pharmaco_eval CP17
FAMS = {
    "gbm": ["oof_blend.csv"],
    "emb": ["oof_emb_concat.csv"],
    "ft_frozen": ["ft_oof_frozen.csv"],
    "ft_fullft": "fullft",
    "ft_ext": "ext",
    "ft_pre": "pre",
    "ft_dmpnn": "dmpnn",
    "ft_cpmed": "cpmed",
    "ft_cpchm": "cpchm",
    "ft_chmridge": "chmridge",
    "ft_d2d6": "d2d6",
    "ft_near": "near",
    "ft_admchm": "admchm",
    "ft_admmed": "admmed",
    "ft_unimol": "unimol",
    "ft_umtmL1": "umtmL1",
    "ft_umtmL2": "umtmL2",
    "ft_umtmL3": "umtmL3",
    "ft_umtmL4": "umtmL4",
    "ft_monridge": ["ft_oof_monridge.csv"],
    "ft_tabpfn_cpmed": "tabpfn_cpmed",
}
CP17 = list(FAMS)


def zavg(files):
    dfs = [pd.read_csv(os.path.join(CACHE, f)) for f in files]
    df = dfs[0].copy()
    for iso in ISO:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    return df


def get_fam(tag):
    if tag in FAMS:
        spec = FAMS[tag]
    else:
        spec = tag  # treat as a family prefix / single file
    if isinstance(spec, str):
        fs = sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{spec}*.csv")))
        if not fs:
            cand = os.path.join(CACHE, f"ft_oof_{spec}.csv")
            if os.path.exists(cand):
                fs = [cand]
    else:
        fs = [os.path.join(CACHE, f) for f in spec]
    fs = [f for f in fs if os.path.exists(f)]
    return zavg(fs) if fs else None


def greedy_blend(pool):
    oofs = {}
    for tag in pool:
        df = get_fam(tag)
        if df is not None:
            oofs[tag] = df
    names = list(oofs)
    nested = {}
    for iso in ISO:
        y = ft[iso].values.astype(float)
        m = ~np.isnan(y)
        zs = {}
        for k in names:
            v = oofs[k][iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zs[k] = z
        rs = []
        for fd in range(5):
            trm = m & (folds != fd)
            vam = m & (folds == fd)
            zt = {}
            for k in names:
                v = zs[k].copy()
                mu = v[trm].mean()
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1)
            w = {k: 0.0 for k in names}
            for _ in range(12):
                bestk, bestr = None, -2
                for k in names:
                    cand = sum((w[j] + (1 if j == k else 0)) * zt[j] for j in names)
                    tot = sum(w.values()) + 1
                    r = pearsonr(y[trm], (cand / tot)[trm]).statistic
                    if np.isnan(r):
                        continue
                    if r > bestr:
                        bestr, bestk = r, k
                w[bestk] += 1
            zf = sum(w[k] * zt[k] for k in w)
            r = pearsonr(y[vam], zf[vam]).statistic
            if not np.isnan(r):
                rs.append(r)
        nested[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
    macro = float(np.mean([nested[i] ** 2 for i in ISO]))
    return nested, macro


def standalone_nested(tag):
    df = get_fam(tag)
    if df is None:
        return None
    nested = {}
    for iso in ISO:
        y = ft[iso].values.astype(float)
        m = ~np.isnan(y)
        v = df[iso].values.astype(float)
        z = np.full(len(v), np.nan)
        z[m] = (v[m] - v[m].mean()) / v[m].std()
        rs = []
        for fd in range(5):
            trm = m & (folds != fd)
            vam = m & (folds == fd)
            r = pearsonr(y[vam], z[vam]).statistic
            if not np.isnan(r):
                rs.append(r)
        nested[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
    macro = float(np.mean([nested[i] ** 2 for i in ISO]))
    return nested, macro


def decorrelation(tag, pool):
    """Avg |pearson| of new member's OOF vs each pool member's OOF, per isoform."""
    newdf = get_fam(tag)
    if newdf is None:
        return None
    corrs = {iso: [] for iso in ISO}
    for ptag in pool:
        pdf = get_fam(ptag)
        if pdf is None:
            continue
        for iso in ISO:
            a = newdf[iso].values.astype(float)
            b = pdf[iso].values.astype(float)
            r = pearsonr(a, b).statistic
            if not np.isnan(r):
                corrs[iso].append(abs(r))
    return {iso: float(np.mean(v)) if v else float("nan") for iso, v in corrs.items()}


def main():
    if len(sys.argv) < 2:
        print("usage: new_member_pool_test.py <member_tag>")
        return
    tag = sys.argv[1]
    print(f"=== POOL TEST: cp17 (19) vs cp17 + {tag} ===")
    stand = standalone_nested(tag)
    if stand is None:
        print(f"ERROR: no OOF for tag '{tag}' (cache/ft_oof_{tag}*.csv)")
        return
    sn, sm = stand
    print(f"  {tag} standalone: macro={sm:.4f}  "
          f"({ {k: round(v,4) for k,v in sn.items()} })")

    n17, m17 = greedy_blend(CP17)
    n18, m18 = greedy_blend(CP17 + [tag])
    print(f"  cp17  macro: {m17:.4f}  ({ {k: round(v,4) for k,v in n17.items()} })")
    print(f"  cp18  macro: {m18:.4f}  ({ {k: round(v,4) for k,v in n18.items()} })")
    print(f"  delta macro: {m18-m17:+.4f}")
    per_iso = {}
    for iso in ISO:
        d = n18[iso] - n17[iso]
        per_iso[iso] = d
        print(f"    {iso}: {n17[iso]:.4f} -> {n18[iso]:.4f} ({d:+.4f})")

    decor = decorrelation(tag, CP17)
    print(f"  |corr| vs pool (lower = more decorrelated): {decor}")

    # gate
    delta = m18 - m17
    worst = min(per_iso.values())
    gate = {
        "delta_macro": delta,
        "threshold": 0.004,
        "per_iso_delta": per_iso,
        "worst_iso_delta": worst,
        "regression_penalty": -0.01,
        "pass": (delta > 0.004) and (worst > -0.01),
    }
    res = {
        "tag": tag,
        "standalone": {"nested": sn, "macro": sm},
        "cp17": {"nested": n17, "macro": m17},
        "cp18": {"nested": n18, "macro": m18},
        "delta_macro": delta,
        "per_iso_delta": per_iso,
        "decorrelation_vs_pool": decor,
        "gate": gate,
    }
    with open(os.path.join(CACHE, f"new_member_{tag}_pooltest.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print()
    print(f"VERDICT: {'PASS - add to pool (cp18 candidate)' if gate['pass'] else 'no clean lift / regresses an isoform'}")
    print(f"saved cache/new_member_{tag}_pooltest.json")


if __name__ == "__main__":
    main()
