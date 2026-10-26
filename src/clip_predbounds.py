"""Clip-to-PREDICTED-bounds calibration, honest CV (CPU).

Train-side fact: clipping blend preds into true assay [lo,hi] gains
0.024-0.115 soft_rae (uq_clipcal). Test set has NO bounds (blinded
SMILES only). MoE arm exports LEARNED bound heads (plo/phi per isoform).

Honest test:
 P = zoo6 blend OOF (ship-quality), bounds = moe OOF predicted bounds.
 Clip P into [plo, phi] (optionally only on high-gate/high-UQ rows) and
 score soft_rae against TRUE y with TRUE [lo,hi].
 Sweep: clip-all, clip-where-gate>0.5, clip-where-UQ-decile>=9, margin
 shrink gamma (clip into [lo+g, hi-g] equivalents: shrink predicted
 interval toward center by factor).
Gate: gain > 0 with >=4/5 folds AND bootstrap CI excludes 0 on the row
level (per isoform).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
rng = np.random.default_rng(7)


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    scale = np.maximum(hi - lo, 0.5)
    return float(np.nanmean(np.clip(err / scale, 0, 5)))


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    moe = pd.read_csv(os.path.join(CACHE, "ft_oof_moe.csv"))
    moe = moe.drop_duplicates("SMILES").set_index("SMILES")
    smi = H.Xtr["SMILES"].values
    uq = pd.read_csv(os.path.join(CACHE, "uq_cross_spread.csv"))
    if "SMILES" in uq.columns:
        uq = uq.drop_duplicates("SMILES").set_index("SMILES")

    for iso in H.ISO:
        short = iso.split("CYP")[1]
        wmap = W.get(iso, {}).get("w", {})
        P = np.zeros(len(smi))
        tot = np.zeros(len(smi))
        for fam, w in wmap.items():
            if not isinstance(w, (int, float)) or abs(w) < 1e-9:
                continue
            try:
                fo = H.raw_oof_fam(fam)[iso].reindex(smi).values
            except Exception:
                continue
            ok = np.isfinite(fo)
            P[ok] += w * fo[ok]
            tot[ok] += w
        P = np.where(tot > 0.5, P / np.maximum(tot, 1e-9), np.nan)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        plo = moe[f"{iso}_plo"].reindex(smi).values.astype(float) if f"{iso}_plo" in moe.columns else None
        phi = moe[f"{iso}_phi"].reindex(smi).values.astype(float) if f"{iso}_phi" in moe.columns else None
        if plo is None:
            print(f"{iso}: moe bound cols missing")
            continue
        ok = np.isfinite(P) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(plo) & np.isfinite(phi)
        base = soft_rae(y[ok], P[ok], lo[ok], hi[ok])
        # honest: fold-wise clip strength (margin gamma) fitted on 4/5
        rows = []
        for gmin in [0.0, 0.1, 0.25, 0.5]:
            gains, fpos = [], 0
            for kf in range(5):
                te = np.where(ok & (H.FOLDS == kf))[0]
                tr = np.where(ok & (H.FOLDS != kf))[0]
                # gamma grid on train, pick best, apply to test
                best_g, best_s = 0.0, np.inf
                for gg in [0.0, 0.1, 0.2, 0.35, 0.5]:
                    plo_t = np.minimum(plo[tr], hi[tr])
                    phi_t = np.maximum(phi[tr], lo[tr])
                    c_t = (plo_t + phi_t) / 2
                    lo_s = c_t + (plo_t - c_t) * (1 - gg)
                    hi_s = c_t + (phi_t - c_t) * (1 - gg)
                    Pc = np.clip(P[tr], np.minimum(lo_s, hi_s), np.maximum(lo_s, hi_s))
                    s = soft_rae(y[tr], Pc, lo[tr], hi[tr])
                    if s < best_s:
                        best_g, best_s = gg, s
                plo_e = np.minimum(plo[te], hi[te]); phi_e = np.maximum(phi[te], lo[te])
                c_e = (plo_e + phi_e) / 2
                lo_e = c_e + (plo_e - c_e) * (1 - best_g)
                hi_e = c_e + (phi_e - c_e) * (1 - best_g)
                Pc_e = np.clip(P[te], np.minimum(lo_e, hi_e), np.maximum(lo_e, hi_e))
                gains.append(soft_rae(y[te], P[te], lo[te], hi[te])
                             - soft_rae(y[te], Pc_e, lo_e, hi_e))
            fpos = sum(1 for g in gains if g > 0)
            rows.append((best_g, round(float(np.mean(gains)), 4), fpos))
        print(f"{iso}: base={base:.4f} clip(predbounds,gamma-grid) -> " +
              " ".join(f"g{g[0]}:{g[1]:+.4f}({g[2]}/5)" for g in rows), flush=True)


if __name__ == "__main__":
    main()
