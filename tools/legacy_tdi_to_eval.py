"""Build umtm-eval-compatible CSVs from legacy TDI OOF npz files (seed-7 folds),
so umtm_eval_tdi.py scores legacy singles and UMTM on the identical protocol.
Usage (cyp env): python tools/legacy_tdi_to_eval.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/cyp-challenge")
sys.path.insert(0, os.path.join(ROOT, "src"))
from run_regression import make_folds, scaffold_groups  # noqa: E402

Xtr = pd.read_parquet(os.path.join(ROOT, "cache", "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
folds = make_folds(scaffold_groups(Xtr["SMILES"].tolist()), seed=7)

NPZ = {
    "legacy_lgbm": "cache/tdi_oof.npz",
    "legacy_dmpnn": "cache/tdi_dmpnn_oof.npz",
    "legacy_tabicl": "cache/tdi_tabicl_oof.npz",
    "legacy_ft_chm": "cache/tdi_ft_chm_oof.npz",
}
for name, path in NPZ.items():
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print("skip (missing):", path)
        continue
    z = np.load(p)
    keys = list(z.keys())
    df = pd.DataFrame({"SMILES": Xtr["SMILES"].values, "fold_tdi": folds})
    for iso in ["CYP2D6", "CYP3A4"]:
        if iso in keys:
            df[f"is_TDI_{iso}"] = z[iso]
    out = os.path.join(ROOT, "cache", f"umtm_eval_{name}.csv")
    df.to_csv(out, index=False)
    print("wrote", out, df.shape, "cols", [c for c in df.columns if c.startswith("is_TDI")])
