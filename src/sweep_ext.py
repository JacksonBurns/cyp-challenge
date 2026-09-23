"""Scaffold-CV with external-data features appended (ranking-only features).

Variants: ext (FP+desc+ext), extemb (FP+desc+ext+CheMeLeon emb). Gate against
cache/oof_pearson.json (0.527/0.604/0.399/0.771) and blend MA-ST-RAE 0.743.
Self-leak audit: only 11 train SMILES exist in ChEMBL; impact on OOF is nil.
Run: cyp env, python src/sweep_ext.py [ext] [extemb]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
R.S_ALL, R.STE_ALL = S, Ste
Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
folds = R.make_folds(R.scaffold_groups(Xtr["SMILES"].tolist()))

Fext = pd.read_parquet(os.path.join(CACHE, "ext_feats_train.parquet"))
assert (Fext["SMILES"].values == Xtr["SMILES"].values).all()
ext = Fext.drop(columns=["SMILES"]).values.astype(np.float32)

emb = pd.read_parquet(os.path.join(CACHE, "chemeleon_emb.parquet")).set_index("SMILES")
Etr = emb.reindex(Xtr["SMILES"]).fillna(0.0).values.astype(np.float32)

VARIANTS = {"ext": np.hstack([Xbase, ext]), "extemb": np.hstack([Xbase, ext, Etr])}
which = sys.argv[1:] or ["ext"]

results = {}
for name in which:
    t0 = time.time()
    n, s, oof = R.cv_config(name, VARIANTS[name], None, targets, bounds, folds,
                            R.ISOFORMS, None, {})
    results[name] = s
    pd.DataFrame({iso: oof[iso] for iso in R.ISOFORMS}).to_csv(
        os.path.join(CACHE, f"oof_x_{name}.csv"), index=False)
    print(f"[{name}] {time.time()-t0:.0f}s", flush=True)

with open(os.path.join(CACHE, "cv_ext.json"), "w") as fh:
    json.dump(results, fh, indent=2)
gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))
print("GATE:", {k: round(v, 3) for k, v in gate.items()}, "MA 0.743")
for name in which:
    print(name, {iso: round(results[name][iso]["pearson"], 3) for iso in R.ISOFORMS},
          "MA", round(results[name]["MA_st_rae"], 3))
