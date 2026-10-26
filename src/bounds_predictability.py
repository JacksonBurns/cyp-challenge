"""Is the per-row assay lower bound (lo) predictable from chemistry?

If yes (R2 > ~0.3), learned bound heads can enable test-side
clip-into-interval calibration (the +0.024..0.115 train lever).
If no, the clip lever dies: we cannot reconstruct the interval blind.
Ridge CV on Xtr descriptors -> lo, hi per isoform. CPU.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402


def ridge_cv_r2(X, yv, K=5, ridge=10.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yv))
    parts = np.array_split(idx, K)
    preds = np.zeros(len(yv))
    for k in range(K):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(K) if j != k])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt = (X[tr] - mu) / sd
        Xe = (X[te] - mu) / sd
        w = np.linalg.solve(Xt.T @ Xt + ridge * len(tr) * np.eye(X.shape[1]), Xt.T @ yv[tr])
        preds[te] = Xe @ w
    ss_res = np.sum((yv - preds) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    return 1 - ss_res / ss_tot


def main():
    X = H.Xtr[[c for c in H.Xtr.columns if c != "SMILES"]].values.astype(float)
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    X = np.where(np.isfinite(X), X, med)
    # rank transform = scale-robust (descriptors have 1e9-scale tails)
    X = pd.DataFrame(X).rank(pct=True).values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    smi = H.Xtr["SMILES"].values
    for iso in H.ISO:
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        m = np.isfinite(lo) & np.isfinite(hi)
        r2lo = ridge_cv_r2(X[m], lo[m])
        r2hi = ridge_cv_r2(X[m], hi[m])
        r2span = ridge_cv_r2(X[m], (hi - lo)[m])
        print(f"{iso}: n={m.sum()} CV R2 lo={r2lo:.3f} hi={r2hi:.3f} "
              f"span={r2span:.3f}", flush=True)


if __name__ == "__main__":
    main()
