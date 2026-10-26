"""Does the TDI-gate offset help the SHIP blend (zoo6), not just ft_pcgrad?

Uses the zoo6 hinge-fit blend OOF (cache/hinge_blend_zoo6.json member
weights applied to family OOFs) as P, then the same honest 5-fold
gate-offset as tdi_moe_offset. Focus: 2D6 (the +0.0069 4/5 signal).
Gate: 2D6 gain>0 and >=4/5 folds on the SHIP blend -> add to cp21 pipeline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    scale = np.maximum(hi - lo, 0.5)
    return float(np.nanmean(np.clip(err / scale, 0, 5)))


def platt(z, y):
    a, b = 1.0, 0.0
    for _ in range(300):
        pz = 1 / (1 + np.exp(-(a * z + b)))
        a -= np.sum((pz - y) * z) / (np.sum(pz * (1 - pz) * z * z) + 1e-8)
        b -= np.sum(pz - y) / (np.sum(pz * (1 - pz)) + 1e-8)
    return a, b


def fit_offset(X, resid, ridge=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(Xs.shape[1]), Xs.T @ resid)
    return w, mu, sd


def main():
    gate_df = pd.read_csv(os.path.join(CACHE, "joint_oof_tdi_pcgrad.csv"))
    gate_df = gate_df.drop_duplicates("SMILES").set_index("SMILES")
    smi = H.Xtr["SMILES"].values

    # ship blend OOF: zoo6 hinge-fit weights applied to family OOFs
    import json
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))

    dcols = [c for c in H.Xtr.columns if c != "SMILES"]
    desc = H.Xtr[dcols].values.astype(float)
    med = np.nanmedian(np.where(np.isfinite(desc), desc, np.nan), axis=0)
    desc = np.where(np.isfinite(desc), desc, med)
    dz = (desc - desc.mean(0)) / (desc.std(0) + 1e-9)
    dsel = dz[:, [0, 1]]

    for iso in ["CYP2D6", "CYP3A4", "CYP1A2", "CYP2C9"]:
        short = iso.split("CYP")[1]
        wmap = W.get(iso, {}).get("w", {})
        P = np.zeros(len(smi))
        tot = np.zeros(len(smi))
        for fam, w in wmap.items():
            if not isinstance(w, (int, float)) or abs(w) < 1e-9:
                continue
            try:
                fam_oof = H.raw_oof_fam(fam)[iso].reindex(smi).values
            except Exception:
                continue
            ok = np.isfinite(fam_oof)
            P[ok] += w * fam_oof[ok]
            tot[ok] += abs(w)
        # weighted blend; weights sum ~1 by construction (lam-regularized)

        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        g_raw = gate_df[f"is_TDI_{short}"].reindex(smi).values.astype(float)
        y_t = gate_df[f"is_TDI_{short}_truth"].reindex(smi).values.astype(float)
        pc = np.clip(np.nan_to_num(g_raw, nan=0.5), 1e-6, 1 - 1e-6)
        z = np.log(pc / (1 - pc))
        labm = np.isfinite(y_t)
        if labm.sum() > 50 and len(np.unique(y_t[labm])) > 2:
            a, b = platt(z[labm], y_t[labm])
            gi = np.nan_to_num(1 / (1 + np.exp(-(a * z + b))), nan=0.0)
        else:
            gi = np.nan_to_num(g_raw, nan=0.0)

        ok = np.isfinite(P) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        base_s, adj_s, gains = [], [], []
        for k in range(5):
            te = np.where(ok & (H.FOLDS == k))[0]
            tr = np.where(ok & (H.FOLDS != k))[0]
            Xtr = np.column_stack([gi[tr], P[tr] * gi[tr], dsel[tr, 0] * gi[tr],
                                   dsel[tr, 1] * gi[tr], gi[tr]])
            w, mu, sd = fit_offset(Xtr, y[tr] - P[tr])
            Xte = np.column_stack([gi[te], P[te] * gi[te], dsel[te, 0] * gi[te],
                                   dsel[te, 1] * gi[te], gi[te]])
            delta = ((Xte - mu) / sd) @ w
            base_s.append(soft_rae(y[te], P[te], lo[te], hi[te]))
            adj_s.append(soft_rae(y[te], P[te] + delta, lo[te], hi[te]))
            gains.append(base_s[-1] - adj_s[-1])
        gp = sum(1 for g in gains if g > 0)
        print(f"{iso} (zoo6): base={np.mean(base_s):.4f} adj={np.mean(adj_s):.4f} "
              f"gain={np.mean(gains):+.4f} folds_pos={gp}/5", flush=True)


if __name__ == "__main__":
    main()
