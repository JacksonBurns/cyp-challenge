"""Near-neighbor label ambiguity: pairs with Tanimoto>=0.9 (ECFP4), per
isoform, spread of pIC50 labels. High spread => chemistry near-identical
but labels differ => the hardness is label-side, not feature-side.
Also: how often do near-identical pairs straddle a hard/extreme label?"""
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ecp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return GEN.GetFingerprint(m)


def main():
    tgt = H.targets
    smis = list(tgt.index)
    fp = {s: ecp(s) for s in smis}
    fp = {s: f for s, f in fp.items() if f is not None}
    keys = list(fp)
    fps = [fp[k] for k in keys]
    print("n fp:", len(keys))
    # nearest neighbor (excl self) per molecule, chunked
    nbr = {}
    B = 200
    for i in range(0, len(keys), B):
        for bi in range(i, min(i + B, len(keys))):
            sl = DataStructs.BulkTanimotoSimilarity(fps[bi], fps)
            best, bj = 0.0, -1
            for k, v in enumerate(sl):
                if k != bi and v > best:
                    best, bj = v, k
            nbr[keys[bi]] = (best, keys[bj])
    rows = []
    for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        col = f"{iso}_pIC50_direct_inhibition"
        for s in keys:
            y = tgt[col].get(s, np.nan)
            nn, nb = nbr.get(s, (np.nan, None))
            yn = tgt[col].get(nb, np.nan) if nb else np.nan
            if np.isfinite(y) and np.isfinite(nn) and np.isfinite(yn) and nn >= 0.9:
                rows.append({"iso": iso, "smi": s, "y": y, "ynb": yn,
                             "d": abs(y - yn)})
    df = pd.DataFrame(rows)
    print("== near-neighbor (Tanimoto>=0.9) |y - y_nnb| by isoform ==")
    for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        sub = df[df.iso == iso]
        if len(sub) == 0:
            print(f" {iso} n=0")
            continue
        print(f" {iso} n={len(sub)} median|d|={sub.d.median():.2f} "
              f"mean={sub.d.mean():.2f} frac|d|>1={np.mean(sub.d>1):.3f} "
              f"frac|d|>2={np.mean(sub.d>2):.3f} p90={sub.d.quantile(.9):.2f}")
    print("label sigma per isoform (for scale):")
    for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        col = f"{iso}_pIC50_direct_inhibition"
        v = tgt[col].dropna().values
        print(f" {iso} sd={np.nanstd(v):.2f}")


main()
