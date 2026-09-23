"""Cheaper CheMeLeon feature transforms -> LGBM, and linear probes.

1. Ridge/KernelRidge linear probes on the 2048-d frozen embeddings (small-data regime).
2. PCA-compressed embeddings (32/64/128) concatenated with FP features -> full scaffold CV.

Same gate: per-isoform OOF Pearson vs cache/oof_pearson.json; blend MA-ST-RAE vs 0.743.
Run with cyp env:  python src/sweep_emb2.py [ridge|pca32|pca64|pca128]
Writes cache/cv_emb2_<name>.json + cache/oof_emb2_<name>.csv.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
R.S_ALL, R.STE_ALL = S, Ste
Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
folds = R.make_folds(R.scaffold_groups(Xtr["SMILES"].tolist()))

emb = pd.read_parquet(os.path.join(CACHE, "chemeleon_emb.parquet")).set_index("SMILES")
Etr = emb.reindex(Xtr["SMILES"]).fillna(0.0).values.astype(np.float32)

which = sys.argv[1:] or ["ridge", "pca32", "pca64", "pca128"]


def ridge_probe():
    """Per isoform, scaffold-CV RidgeCV on embeddings (log-scale standardized)."""
    oof = {}
    summary = {}
    E = np.log1p(np.maximum(Etr, 0)) - np.log1p(np.maximum(Etr, 0)).mean(0)
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        oof[iso] = np.full(len(y), np.nan)
        for f in range(R.K):
            trm = (folds != f) & ~np.isnan(y)
            vam = folds == f
            m = RidgeCV(alphas=np.logspace(-2, 4, 13))
            m.fit(E[trm], y[trm])
            oof[iso][vam] = m.predict(E[vam])
        lab = ~np.isnan(y)
        lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
        summary[iso] = {
            "st_rae": float(R.soft_rae(y[lab], oof[iso][lab], lo[lab], hi[lab])),
            "pearson": float(pearsonr(y[lab], oof[iso][lab]).statistic),
            "n": int(lab.sum()),
        }
    summary["MA_st_rae"] = float(np.mean([summary[i]["st_rae"] for i in R.ISOFORMS]))
    return summary, oof


def pca_variant(k):
    p = PCA(n_components=k, random_state=0)
    P = p.fit_transform(Etr).astype(np.float32)
    return np.hstack([Xbase, P])


results = {}
for name in which:
    t0 = time.time()
    if name == "ridge":
        s, oof = ridge_probe()
        print(json.dumps({"ridge": s}, indent=1), flush=True)
    else:
        k = int(name.replace("pca", ""))
        Xf = pca_variant(k)
        _, s, oof = R.cv_config(name, Xf, None, targets, bounds, folds, R.ISOFORMS, None, {})
    results[name] = s
    pd.DataFrame({iso: oof[iso] for iso in R.ISOFORMS}).to_csv(
        os.path.join(CACHE, f"oof_emb2_{name}.csv"), index=False)
    print(f"[{name}] {time.time()-t0:.0f}s", flush=True)

with open(os.path.join(CACHE, "cv_emb2.json"), "w") as fh:
    json.dump(results, fh, indent=2)
gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))
print("\nGATE:", {k: round(v, 3) for k, v in gate.items()}, "MA_st_rae 0.743", flush=True)
for name in which:
    print(name, {iso: round(results[name][iso]["pearson"], 3) for iso in R.ISOFORMS},
          "MA", round(results[name]["MA_st_rae"], 3), flush=True)
