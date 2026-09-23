"""Regression submission v2: FP+RDKit GBM blend (4cfg x 3seed, shipped recipe)
mixture with CheMeLeon-embedding-augmented blend (4cfg x 3seed), placed with
reveal-calibrated spread factors.

Mixture weights w_gbm per isoform chosen on scaffold OOF (cache/emb_blend.json):
  1A2 0.6 / 2C9 0.4 / 2D6 0.4 / 3A4 0.7  (z-space, convex)
Placement: sd = F_ISO * clip(rho_oof_new * OOF_TO_BLIND) * sd_blind, moments as
shipped (BLIND_MOMENTS; 2D6 -> STRAE 3.57/0.90). F from the Sep 23 reveal +
ST-RAE sims (src/shrink_placement.py): 1A2 0.6, 2C9 0.6, 2D6 1.0, 3A4 0.7.
Output: cache/regression_v2_submission.csv (+ blend_test_preds.npz with RAW preds).
Does NOT touch cache/raw_regression_submission.csv (the scored interim file).

Run: cyp env, from repo root is NOT required (validation run separately).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402
from final_submit import BLIND_MOMENTS, STRAE_MOMENTS, OOF_TO_BLIND, FLOOR  # noqa: E402

W_GBM = {"CYP1A2": 0.6, "CYP2C9": 0.4, "CYP2D6": 0.4, "CYP3A4": 0.7}
F_SPREAD = {"CYP1A2": 0.6, "CYP2C9": 0.6, "CYP2D6": 1.0, "CYP3A4": 0.7}

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
R.S_ALL, R.STE_ALL = S, Ste
Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
Xte_base = Xte.drop(columns=["SMILES"]).values.astype(np.float32)

emb = pd.read_parquet(os.path.join(CACHE, "chemeleon_emb.parquet")).set_index("SMILES")
Etr = emb.reindex(Xtr["SMILES"]).fillna(0.0).values.astype(np.float32)
Ete = emb.reindex(Xte["SMILES"]).fillna(0.0).values.astype(np.float32)
print("test rows missing emb:", int(emb.reindex(Xte["SMILES"]).isna().any(axis=1).sum()), flush=True)

sweep = json.load(open(os.path.join(CACHE, "cv_sweep.json")))
param_sets = list(sweep["params"].values())

out_blend, out_cat = {}, {}
t00 = time.time()
for iso in R.ISOFORMS:
    y = targets[R.DIRECT[iso]].values
    cand = np.where(~np.isnan(y))[0]
    nb_tr = R.nn_block(S, cand, y, q_idx=np.arange(len(y))).astype(np.float32)
    nb_te = R.nn_block(Ste, cand, y).astype(np.float32)
    feats = {"blend": (np.hstack([Xbase, nb_tr]), np.hstack([Xte_base, nb_te])),
             "concat": (np.hstack([Xbase, Etr, nb_tr]), np.hstack([Xte_base, Ete, nb_te]))}
    for tag, (Xl, Xq) in feats.items():
        preds, n = 0.0, 0
        for params in param_sets:
            for s in range(3):
                m = R.lgbm(random_state=42 + s, **params)
                m.fit(Xl[cand], y[cand])
                preds = preds + m.predict(Xq)
                n += 1
        (out_blend if tag == "blend" else out_cat)[iso] = preds / n
    print(f"{iso}: blends done ({time.time()-t00:.0f}s)", flush=True)

np.savez(os.path.join(CACHE, "blend_test_preds.npz"),
         **{f"blend_{i}": out_blend[i] for i in R.ISOFORMS},
         **{f"concat_{i}": out_cat[i] for i in R.ISOFORMS})

sub = pd.DataFrame({"SMILES": test["SMILES"].values, "Molecule_Name": test["Molecule_Name"].values})
report = {}
for iso in R.ISOFORMS:
    b, c = out_blend[iso], out_cat[iso]
    zb, zc = (b - b.mean()) / b.std(), (c - c.mean()) / c.std()
    w = W_GBM[iso]
    z = w * zb + (1 - w) * zc
    z = (z - z.mean()) / z.std()
    # rho for placement: OOF pearson of the SAME z-mixture (from OOF files)
    oofb = pd.read_csv(os.path.join(CACHE, "oof_blend.csv"))[iso].values
    oofc = pd.read_csv(os.path.join(CACHE, "oof_emb_concat.csv"))[iso].values
    y = targets[R.DIRECT[iso]].values
    msk = ~np.isnan(y)
    zof_b = np.full(len(oofb), np.nan)
    zof_c = np.full(len(oofc), np.nan)
    zof_b[msk] = (oofb[msk] - oofb[msk].mean()) / oofb[msk].std()
    zof_c[msk] = (oofc[msk] - oofc[msk].mean()) / oofc[msk].std()
    zmix_oof = w * zof_b + (1 - w) * zof_c
    rho = float(pearsonr(y[msk], zmix_oof[msk]).statistic)
    rho_a = float(np.clip(rho * OOF_TO_BLIND[iso], 0.05, 0.95))
    mom = BLIND_MOMENTS[iso]
    mu, sd = mom["mean"], mom["sd"]
    if iso in STRAE_MOMENTS:
        mu, sd = STRAE_MOMENTS[iso]["mean"], STRAE_MOMENTS[iso]["sd"]
    sd = F_SPREAD[iso] * rho_a * mom["sd"] if iso not in STRAE_MOMENTS else F_SPREAD[iso] * sd
    placed = np.clip(mu + sd * z, FLOOR, None)
    sub[f"{iso}_pIC50_direct_inhibition"] = placed
    report[iso] = {"rho_oof_blend": None, "rho_mix_oof": rho, "w_gbm": w, "f_spread": F_SPREAD[iso],
                   "target_mean": mu, "target_sd": sd, "placed_mean": float(placed.mean()),
                   "placed_sd": float(placed.std())}
    print(iso, json.dumps(report[iso]), flush=True)

sub.to_csv(os.path.join(CACHE, "regression_v2_submission.csv"), index=False)
json.dump(report, open(os.path.join(CACHE, "regression_v2_report.json"), "w"), indent=2)
print("DONE", time.time() - t00, "s")
