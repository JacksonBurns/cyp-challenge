"""When families disagree, WHERE is the error? (Oct 26 user question)

For deciles of cross-family UQ (spread across pool members):
 - signed residual (y - P): is the blend systematically LOW?
 - position of P relative to true assay interval [lo,hi]: below / inside /
   above. LLOD hypothesis: disagreements = predictions in the low tail
   (below lo, near assay LLOD).
Uses zoo6 blend OOF + true bounds. CPU.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE


def main():
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    smi = H.Xtr["SMILES"].values
    names = [n for n in H.POOLS["zoo6"] if H.FAMS.get(n)]
    oofs = {k: H.raw_oof_fam(k) for k in names}
    for iso in H.ISO:
        P = np.zeros(len(smi))
        tot = np.zeros(len(smi))
        for fam, w in W[iso]["w"].items():
            if not isinstance(w, (int, float)) or abs(w) < 1e-9:
                continue
            try:
                fo = oofs[fam][iso].reindex(smi).values
            except Exception:
                continue
            ok = np.isfinite(fo)
            P[ok] += w * fo[ok]
            tot[ok] += w
        P = np.where(tot > 0.5, P / np.maximum(tot, 1e-9), np.nan)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        # cross-family spread on labeled rows
        M = np.column_stack([oofs[k][iso].reindex(smi).values for k in names])
        spread = np.nanstd(M, axis=1)
        okm = np.isfinite(P) & np.isfinite(y) & np.isfinite(spread) & np.isfinite(lo) & np.isfinite(hi)
        q = pd.Series(spread[okm]).rank(pct=True).values
        Pv, yv, lov, hiv = P[okm], y[okm], lo[okm], hi[okm]
        pos = np.where(Pv < lov, "below_lo", np.where(Pv > hiv, "above_hi", "inside"))
        print(f"\n{iso} (n={okm.sum()}):")
        for dlo, dhi, lab in [(0.0, 0.5, "UQ low half"), (0.5, 0.7, "UQ 50-70"),
                              (0.7, 0.9, "UQ 70-90"), (0.9, 1.01, "UQ top 10%")]:
            m = (q >= dlo) & (q < dhi)
            frac_below = np.mean(pos[m] == "below_lo")
            frac_above = np.mean(pos[m] == "above_hi")
            frac_in = np.mean(pos[m] == "inside")
            resid = np.mean(yv[m] - Pv[m])
            # signed metric error: how much error comes from P below lo?
            e_below = np.clip(lov[m] - Pv[m], 0, None)
            e_above = np.clip(Pv[m] - hiv[m], 0, None)
            tot = e_below.sum() + e_above.sum()
            sh_below = e_below.sum() / tot if tot > 0 else 0.0
            print(f"  {lab:12s} n={m.sum():4d} mean(y-P)={resid:+.3f} "
                  f"below_lo={frac_below:.2f} inside={frac_in:.2f} "
                  f"above_hi={frac_above:.2f} | err-share from LOW side={sh_below:.2f}")


if __name__ == "__main__":
    main()
