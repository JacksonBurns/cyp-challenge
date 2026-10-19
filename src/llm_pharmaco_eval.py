"""Honest nested eval of the LLM pharmacophore member (Avenue 2).

Anti-leak protocol (same as mordred_probe / pool_simplify):
  - features are SMILES-only (no target in X), so no direct leakage
  - per fold: impute + z-score with TRAIN-fold stats; fit learner on train;
    predict val; score Pearson on val
  - nested = Fisher-z avg per isoform; macro = mean(nested^2)

Learner: per-isoform ElasticNet (L1 selects which of my prior features matter,
L2 stabilizes). 18 features, ~300 pts/fold -> well-conditioned, no overfit risk
if alpha is sane.

Then the POOL test (the real question for blend value): add ft_llmpharm as a
20th member to the full 19-member cp17 pool and re-run the honest greedy blend
per isoform. If macro R2 goes up by > fold noise (~0.004) AND it's not just
2D6-chasing, the member is worth adding.

Output: cache/llm_pharmaco_eval.json, cache/ft_oof_llmpharm.csv,
        cache/ft_test_preds_llmpharm.csv
Run:  ~/miniforge3/envs/cyp/bin/python src/llm_pharmaco_eval.py
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
folds = ft["fold"].values
test = pd.read_csv(os.path.join(CACHE, "..", "data", "cyp-challenge-TEST-BLINDED.csv"))

with open(os.path.join(CACHE, "llm_pharmaco_feats.pkl"), "rb") as fh:
    F = pickle.load(fh)
Xtr_all, Xte, fnames = F["Xtr"], F["Xte"], F["names"]


def impute_scale(Xtr, Xva, Xte):
    """Per-column median impute + z-score using TRAIN-fold stats only."""
    Xtr_i = Xtr.copy()
    Xva_i = Xva.copy()
    Xte_i = Xte.copy()
    for j in range(Xtr.shape[1]):
        c = Xtr_i[:, j]
        med = np.nanmedian(c)
        med = 0.0 if np.isnan(med) else med
        Xtr_i[:, j] = np.where(np.isnan(c), med, c)
        Xva_i[:, j] = np.where(np.isnan(Xva_i[:, j]), med, Xva_i[:, j])
        Xte_i[:, j] = np.where(np.isnan(Xte_i[:, j]), med, Xte_i[:, j])
    mu = Xtr_i.mean(axis=0)
    sd = Xtr_i.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr_i - mu) / sd, (Xva_i - mu) / sd, (Xte_i - mu) / sd


def fit_predict(Xtr, ytr, Xva, alpha=1.0):
    m = ~np.isnan(ytr)
    # Ridge (not L1): 18 prior features on ~300 pts/fold. L1/ElasticNet
    # zeroed all weights -> constant collapse on 3/4 isoforms. Ridge keeps the
    # (weakly-informative) prior features and can't collapse to constant.
    rg = Ridge(alpha=alpha)
    rg.fit(Xtr[m], ytr[m])
    return rg.predict(Xva), rg


def nested_eval():
    """Per-isoform honest nested R2 of the LLM member alone."""
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    full_models = {}
    for iso in ISO:
        y = ft[iso].values.astype(float)
        m = ~np.isnan(y)
        rs = []
        for fd in range(5):
            trm = m & (folds != fd)
            vam = m & (folds == fd)
            a, b, _ = impute_scale(Xtr_all[trm], Xtr_all[vam], Xte)
            pred, _ = fit_predict(a, y[trm], b)
            oof.loc[vam, iso] = pred
            r = pearsonr(y[vam], pred).statistic
            if not np.isnan(r):
                rs.append(r)
        nested = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
        # full-train model for test preds
        mfull = m
        a, b, c = impute_scale(Xtr_all[mfull], Xtr_all, Xte)
        pred_full, model = fit_predict(a, y[mfull], c)
        full_models[iso] = (c, model)
        oof[iso] = oof[iso].fillna(0)  # ensure non-NaN (rare)
        print(f"  {iso}: nested={nested:.4f} R2={nested**2:.4f}")
    # recompute nested per isoform from oof (clean)
    nested_per = {}
    for iso in ISO:
        y = ft[iso].values.astype(float)
        m = ~np.isnan(y)
        rs = []
        for fd in range(5):
            trm = m & (folds != fd)
            vam = m & (folds == fd)
            r = pearsonr(y[vam], oof[iso].values[vam]).statistic
            if not np.isnan(r):
                rs.append(r)
        nested_per[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
    macro = float(np.mean([nested_per[i] ** 2 for i in ISO]))
    print(f"  MACRO nested R2 (LLM alone): {macro:.4f}")
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, "ft_oof_llmpharm.csv"), index=False)
    # test preds
    tdf = pd.DataFrame(index=range(len(test)), columns=ISO)
    for iso in ISO:
        c, model = full_models[iso]
        tdf[iso] = model.predict(c)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, "ft_test_preds_llmpharm.csv"), index=False)
    return nested_per, macro, tdf


# ---------------------------------------------------------------------------
# POOL test: add ft_llmpharm to the 19-member cp17 pool, honest greedy blend.
# ---------------------------------------------------------------------------
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
    "ft_llmpharm": ["ft_oof_llmpharm.csv"],  # NEW
}
CP17 = ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre", "ft_dmpnn",
        "ft_cpmed", "ft_cpchm", "ft_chmridge", "ft_d2d6", "ft_near",
        "ft_admchm", "ft_admmed", "ft_unimol", "ft_umtmL1", "ft_umtmL2",
        "ft_umtmL3", "ft_umtmL4", "ft_monridge", "ft_tabpfn_cpmed"]
CP18 = CP17 + ["ft_llmpharm"]


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
    spec = FAMS[tag]
    if isinstance(spec, str):
        fs = sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{spec}*.csv")))
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


def main():
    print("=== LLM pharmacophore member: standalone honest nested ===")
    nested_per, macro, tdf = nested_eval()
    print()
    print("=== POOL test: cp17 (19) vs cp18 (20, +ft_llmpharm) ===")
    n17, m17 = greedy_blend(CP17)
    n18, m18 = greedy_blend(CP18)
    print(f"  cp17 macro: {m17:.4f}   (nested { {k: round(v,4) for k,v in n17.items()} })")
    print(f"  cp18 macro: {m18:.4f}   (nested { {k: round(v,4) for k,v in n18.items()} })")
    print(f"  delta macro: {m18-m17:+.4f}")
    for iso in ISO:
        print(f"    {iso}: {n17[iso]:.4f} -> {n18[iso]:.4f} ({n18[iso]-n17[iso]:+.4f})")

    res = {
        "llm_standalone": {"nested": nested_per, "macro": macro},
        "cp17": {"nested": n17, "macro": m17},
        "cp18": {"nested": n18, "macro": m18},
        "delta_macro": m18 - m17,
        "gate": {"pass": (m18 - m17) > 0.004, "threshold": 0.004},
    }
    with open(os.path.join(CACHE, "llm_pharmaco_eval.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print()
    print(f"VERDICT: {'PASS - add to pool' if res['gate']['pass'] else 'no lift above fold noise'}")
    print("saved cache/llm_pharmaco_eval.json")


if __name__ == "__main__":
    main()
