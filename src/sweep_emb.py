"""Scaffold-CV of LightGBM with CheMeLeon frozen embeddings, vs the FP baseline.

Variants (all reuse run_regression.cv_config, folds, nn_block self-exclusion):
  base  : FP+RDKit only (sanity re-run, should reproduce ~0.743 blend)
  concat: FP+RDKit + CheMeLeon 2048-d embeddings
  emb   : CheMeLeon embeddings only
Compares per-isoform OOF Pearson + blend MA-ST-RAE against the gate
(cache/oof_pearson.json: 0.527/0.604/0.399/0.771; blend 0.743).

Run with the cyp env. Writes cache/cv_emb.json + cache/oof_emb_<variant>.csv.
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
groups = R.scaffold_groups(Xtr["SMILES"].tolist())
folds = R.make_folds(groups)

emb = pd.read_parquet(os.path.join(CACHE, "chemeleon_emb.parquet"))
Etr = emb.set_index("SMILES").reindex(Xtr["SMILES"]).fillna(0.0).values.astype(np.float32)
n_missing_tr = int(emb.set_index("SMILES").reindex(Xtr["SMILES"]).isna().any(axis=1).sum())
print(f"emb matrix {Etr.shape}; train rows missing emb: {n_missing_tr}/{len(Xtr)}", flush=True)

VARIANTS = {
    "base": Xbase,
    "concat": np.hstack([Xbase, Etr]),
    "emb": Etr,
}

which = sys.argv[1:] or list(VARIANTS)
results = {}
for name in which:
    t0 = time.time()
    n, s, oof = R.cv_config(f"emb_{name}", VARIANTS[name], None, targets, bounds, folds,
                            R.ISOFORMS, None, {})
    results[name] = s
    pd.DataFrame({iso: oof[iso] for iso in R.ISOFORMS}).to_csv(
        os.path.join(CACHE, f"oof_emb_{name}.csv"), index=False)
    print(f"[{name}] done in {time.time()-t0:.0f}s", flush=True)

out_path = os.path.join(CACHE, "cv_emb.json")
old = {}
if os.path.exists(out_path):
    old = json.load(open(out_path))
old.update(results)
with open(out_path, "w") as fh:
    json.dump(old, fh, indent=2)

gate = json.load(open(os.path.join(CACHE, "oof_pearson.json")))
print("\nGATE CHECK (need blend of shipped configs to beat per-isoform pearson)", flush=True)
for name in which:
    r = results[name]
    line = {iso: round(r[iso]["pearson"], 3) for iso in R.ISOFORMS}
    print(name, "pearson", line, "MA_st_rae", round(r["MA_st_rae"], 3), flush=True)
print("gate  pearson", {k: round(v, 3) for k, v in gate.items()}, "MA_st_rae 0.743", flush=True)
