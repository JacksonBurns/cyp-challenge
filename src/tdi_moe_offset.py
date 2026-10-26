"""TDI gate -> pIC50 offset, honest nested-CV evaluation (CPU).

Pipeline (all quantities out-of-fold):
 1. Gate g_iso = Platt-calibrated P(is_TDI) from the pcgrad co-trained TDI
    head. Platt params fit on rows with TDI truth (sparse); transform
    applied to ALL rows (gate prob exists everywhere).
 2. Offset model fit on TRAIN folds only: blend residual (y - P) regressed
    (ridge) on [g, g*P, g*d1, g*d2, g].
 3. Apply to held-out fold; score soft_rae vs uncorrected blend.

This is the LEARNED TDI-conditioning the post-hoc tdi_calib test failed
at: offset fit out-of-sample, applied out-of-sample.
Gate: mean gain > 0 and >=4/5 folds positive per isoform.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
ISO = H.ISO


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    scale = np.maximum(hi - lo, 0.5)
    return float(np.nanmean(np.clip(err / scale, 0, 5)))


def calib_gate(p_raw, y):
    """Platt-scale all rows; fit params on labeled rows only."""
    pc = np.clip(np.nan_to_num(p_raw, nan=0.5), 1e-6, 1 - 1e-6)
    z = np.log(pc / (1 - pc))
    lab = np.isfinite(y)
    if lab.sum() < 50 or len(np.unique(y[lab])) < 2 or np.std(z) < 1e-9:
        return np.nan_to_num(p_raw, nan=0.0), 0.0
    zl, yl = z[lab], y[lab]
    a, b = 1.0, 0.0
    for _ in range(300):
        pz = 1 / (1 + np.exp(-(a * zl + b)))
        a -= np.sum((pz - yl) * zl) / (np.sum(pz * (1 - pz) * zl * zl) + 1e-8)
        b -= np.sum(pz - yl) / (np.sum(pz * (1 - pz)) + 1e-8)
    return 1 / (1 + np.exp(-(a * z + b))), float(a)


def fit_offset(X, resid, ridge=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(Xs.shape[1]), Xs.T @ resid)
    return w, mu, sd


def main():
    gate_df = pd.read_csv(os.path.join(CACHE, "joint_oof_tdi_pcgrad.csv"))
    gate_df = gate_df.drop_duplicates("SMILES").set_index("SMILES")
    pgrad = pd.read_csv(os.path.join(CACHE, "ft_oof_pcgrad.csv"))
    pgrad = pgrad.drop_duplicates("SMILES").set_index("SMILES")

    smi = H.Xtr["SMILES"].values
    n = len(smi)
    dcols = [c for c in H.Xtr.columns if c != "SMILES"]
    desc = H.Xtr[dcols].values.astype(float)
    med = np.nanmedian(np.where(np.isfinite(desc), desc, np.nan), axis=0)
    desc = np.where(np.isfinite(desc), desc, med)
    dz = (desc - desc.mean(0)) / (desc.std(0) + 1e-9)
    dsel = dz[:, [0, 1]]

    rows = []
    for iso in ISO:
        short = iso.split("CYP")[1]
        P = pgrad[iso].reindex(smi).values.astype(float)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        g_raw = gate_df[f"is_TDI_{short}"].reindex(smi).values.astype(float)
        y_t = gate_df[f"is_TDI_{short}_truth"].reindex(smi).values.astype(float)
        gi, a_slope = calib_gate(g_raw, y_t)
        gi = np.nan_to_num(gi, nan=0.0)

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
        rows.append(dict(iso=iso, slope=round(a_slope, 2),
                         base=round(float(np.mean(base_s)), 4),
                         adj=round(float(np.mean(adj_s)), 4),
                         gain=round(float(np.mean(gains)), 4),
                         folds_pos=int(sum(1 for g in gains if g > 0))))
        r = rows[-1]
        print(f"{iso}: slope={r['slope']} base={r['base']:.4f} adj={r['adj']:.4f} "
              f"gain={r['gain']:+.4f} folds_pos={r['folds_pos']}/5", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CACHE, "tdi_moe_offset.csv"), index=False)
    print(f"\nMACRO gain = {df.gain.mean():+.4f}")


if __name__ == "__main__":
    main()
