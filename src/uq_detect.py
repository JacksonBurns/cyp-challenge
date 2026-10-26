"""UQ arm (Oct 25 directive): per-molecule uncertainty to DETECT
issue-causing molecules, and test UQ-weighted loss relevance.

Two uncertainty sources, same scaffold-5CV split convention:
 A) ensemble spread: per-fold ridge (alpha sweep) on Morgan-2048 + descriptor
    features -> per-fold residual matrix; spread = var-of-fold-preds + MAD.
 B) latent embedding geometry: PCA-reduced (256d, memory rule) chemprop_medium
    embeddings (emb_cpmed_all.parquet) -> kNN distance to train (k=10),
    Mahalanobis distance (PCA whitened).

Targets: the 4 pIC50 isoforms from cache/X_train.parquet labels (ft table).

Deliverables:
 1. Spearman(UQ, mean |residual|) per isoform - gate rho > 0.30.
 2. Top-5% issue molecules per isoform -> cache/uq_issue_top.csv.
 3. Correlation of UQ with test-set (blinded) neighbor distance - "is the
    blind set OOD relative to train?" external_features overlap audit exists;
    this gives the UQ view.
 4. UQ-weighted loss RELEVANCE test: refit ridge with weights 1/(eps+spread)
    vs unweighted: honest per-fold R2 change. (Cheap proxy for the loss-side
    use of UQ; the deep trainer version uses ft_oof spread as sample weights.)

Writes: cache/uq_scores.csv, cache/uq_issue_top.csv, cache/uq_analysis.json
CPU only.
"""
import json
import os

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid

RDLogger.disable()

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
K = 5
SEED = 7
ALPHAS = [1.0, 10.0, 100.0]
KNN = 10
PCA_DIM = 256


def scaffold_folds(smiles, k=K, seed=SEED):
    import sys
    sys.path.insert(0, os.path.join(HERE))
    from run_regression import scaffold_groups, make_folds
    return make_folds(scaffold_groups(smiles), k, seed)


def main():
    X = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    X = X.drop_duplicates("SMILES").reset_index(drop=True)
    desc = pd.read_csv(os.path.join(CACHE, "desc3d_train.csv"))
    desc = desc.drop_duplicates("SMILES").set_index("SMILES")
    emb = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet"))
    emb = emb.drop_duplicates("SMILES").set_index("SMILES")
    ytab = X[["SMILES"] + [i for i in ISO if i in X.columns]]
    feats = [c for c in X.columns if c not in ("SMILES", *ISO)]
    Xf = X.set_index("SMILES")
    common = ([s for s in ytab["SMILES"] if s in emb.index and s in desc.index])
    ytab = ytab.set_index("SMILES").loc[common].reset_index()
    Xm = Xf.loc[common, feats].values
    Dm = desc.loc[common].values.astype(float)
    Dm = np.nan_to_num(Dm, nan=0.0, posinf=0.0, neginf=0.0)
    Em = emb.loc[common].values.astype(np.float32)
    Em = np.nan_to_num(Em, nan=0.0)
    fold = scaffold_folds(ytab["SMILES"].tolist())
    n = len(ytab)

    # PCA embedding (memory rule: never run distances at 2048-d)
    pca = PCA(n_components=min(PCA_DIM, Em.shape[1]), random_state=0)
    Ep = pca.fit_transform(Em)
    Ep = Ep / (Ep.std(0, keepdims=True) + 1e-9)

    results = {}
    spread_store = np.zeros((n, len(ISO)))
    knn = np.zeros(n)
    maha = np.zeros(n)
    for f in range(K):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        # latent geometry: distance from val to train points
        d = cdist(Ep[va], Ep[tr], metric="euclidean")
        knn[va] = np.sort(d, axis=1)[:, :KNN].mean(1)
        mu = Ep[tr].mean(0)
        C = np.cov(Ep[tr].T) + 1e-4 * np.eye(Ep.shape[1])
        Ci = np.linalg.pinv(C)
        dc = Ep[va] - mu
        maha[va] = np.sqrt(np.einsum("ij,jk,ik->i", dc, Ci, dc))
    for j, iso in enumerate(ISO):
        y = ytab[iso].values.astype(float)
        lab = ~np.isnan(y)
        preds = np.full((n, K), np.nan)
        best_a, best_r2 = None, -np.inf
        for f in range(K):
            tr = np.where(fold != f)[0]
            va = np.where(fold == f)[0]
            trl = tr[lab[tr]]
            val = va[lab[va]]
            if len(val) == 0:
                continue
            for a in ALPHAS:
                r = Ridge(alpha=a).fit(np.c_[Xm[trl], Dm[trl]], y[trl])
                p = r.predict(np.c_[Xm[val], Dm[val]])
                preds[val, ALPHAS.index(a)] = p
                ss = 1 - ((y[val] - p) ** 2).sum() / ((y[val] - y[trl].mean()) ** 2).sum()
                if ALPHAS.index(a) == 0 and ss > best_r2:
                    best_a, best_r2 = a, ss
        # per-molecule spread: across-alpha + use mean |resid| as error target
        err = np.nanmean(np.abs(preds - y[:, None]), axis=1)
        sp_pred = np.nanvar(preds, axis=1)
        sp_mean = np.nanmean(preds, axis=1)
        rho_sp = spearmanr(sp_pred[lab], err[lab]).statistic
        rho_knn = spearmanr(knn[lab], err[lab]).statistic
        rho_ma = spearmanr(maha[lab], err[lab]).statistic
        # UQ-weighted refit: weights 1/(0.5+knn-z) style, honest per-fold R2
        w = 1.0 / (0.5 + np.clip(np.nan_to_num(sp_pred), 0, None)
                   + knn / knn.std())
        r2_w, r2_u = [], []
        for f in range(K):
            tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
            trl, val = tr[lab[tr]], va[lab[va]]
            if len(val) < 5:
                continue
            XX = np.c_[Xm, Dm]
            ru = Ridge(alpha=10.0).fit(XX[trl], y[trl])
            pu = ru.predict(XX[val])
            rw = Ridge(alpha=10.0).fit(XX[trl], y[trl], sample_weight=w[trl])
            pw = rw.predict(XX[val])
            r2_u.append(1 - ((y[val] - pu) ** 2).sum() / ((y[val] - y[trl].mean()) ** 2).sum())
            r2_w.append(1 - ((y[val] - pw) ** 2).sum() / ((y[val] - y[trl].mean()) ** 2).sum())
        results[iso] = {
            "rho_spread_err": round(float(rho_sp), 4),
            "rho_knn_err": round(float(rho_knn), 4),
            "rho_maha_err": round(float(rho_ma), 4),
            "mean_abs_err": round(float(np.nanmean(err[lab])), 4),
            "r2_unweighted": round(float(np.mean(r2_u)), 4),
            "r2_uqweighted": round(float(np.mean(r2_w)), 4),
        }
        spread_store[:, j] = np.nan_to_num(sp_pred)
        print(iso, json.dumps(results[iso]), flush=True)

    out = pd.DataFrame({"SMILES": common, "knn10": knn, "mahalanobis": maha,
                        "spread_1A2": spread_store[:, 0],
                        "spread_2C9": spread_store[:, 1],
                        "spread_2D6": spread_store[:, 2],
                        "spread_3A4": spread_store[:, 3]})
    out.to_csv(os.path.join(CACHE, "uq_scores.csv"), index=False)
    # top-5% issue molecules: per-iso rank of spread (mean of z-scores of
    # spread, knn, maha = composite OOD-issue score)
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    out["composite"] = (z(out["knn10"]) + z(out["mahalanobis"])
                        + z(out[[f"spread_{i}" for i in ("1A2", "2C9", "2D6", "3A4")]]).mean(1))
    top = out.sort_values("composite", ascending=False).head(int(0.05 * n))
    top.to_csv(os.path.join(CACHE, "uq_issue_top.csv"), index=False)
    with open(os.path.join(CACHE, "uq_analysis.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("WROTE uq_scores.csv, uq_issue_top.csv (n=%d), uq_analysis.json" % n)


if __name__ == "__main__":
    main()
