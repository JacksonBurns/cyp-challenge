"""Convert UMTM OOF is_TDI probabilities to tdi_blend_nested-style npz members.

  python tools/umtm_to_tdi_npz.py outname cache/umtm_oof_L2.csv [s1csv s2csv ...]

Writes cache/outname.npz with {"CYP2D6": arr, "CYP3A4": arr}, rows aligned to
X_train.parquet dedup-SMILES order (the contract tdi_blend_nested.py expects).
Multiple seed files are probability-averaged first (like umtm_eval_tdi --avg).
Caveat (asserted in NOTES sec 22): UMTM is_TDI preds are fold_reg-based OOF;
the blend nests weights on fold_tdi seed-7, same protocol as the go/no-go eval.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/cyp-challenge")
CACHE = os.path.join(ROOT, "cache")
ISO = {"CYP2D6": "is_TDI_2D6", "CYP3A4": "is_TDI_3A4"}  # npz key -> OOF column

out = sys.argv[1]
files = sys.argv[2:]
assert files, "usage: outname csv [csv...]"

dfs = [pd.read_csv(f) for f in files]
base = dfs[0].copy()
for iso, col in ISO.items():
    assert col in base.columns, f"{files[0]} missing {col}"
    base[col] = np.mean([d[col].values.astype(float) for d in dfs], axis=0)

Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
final = {}
for iso, col in ISO.items():
    p = base.set_index("SMILES")[col].reindex(Xtr["SMILES"]).values.astype(float)
    assert np.isfinite(p).all(), (
        f"{out} {iso}: unmatched X_train rows (UMTM OOF must cover all TDI train mols)")
    final[iso] = (p - p.mean()) / p.std()
dst = os.path.join(CACHE, f"{out}.npz")
np.savez(dst, **final)
print("wrote", dst, {k: v.shape for k, v in final.items()},
      f"(avg of {len(dfs)} seeds, X_train-aligned, all finite)")
