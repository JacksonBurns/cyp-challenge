"""Task-5 expected-MCC fraction machinery on the 3-way blended TDI score.

Blend z-scores: 3A4 0.6*base+0.2*emb+0.2*tabicl, 2D6 0.9*base+0.1*emb
(weights from tdi_blend3.json). Reuses tdi_fraction_opt. Writes
cache/tdi_fraction_optima_v3.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.join(ROOT, "cache")
from tdi_fraction_opt import rank_conditional_curve, emcc, rescale_to_pi, ISO  # noqa: E402

base = np.load(os.path.join(CACHE, "tdi_oof.npz"))
emb = np.load(os.path.join(CACHE, "tdi_emb_oof.npz"))
tab = np.load(os.path.join(CACHE, "tdi_tabicl_oof.npz"))
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
tdi = pd.read_csv(os.path.join(ROOT, "data", "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(ROOT, "data", "cyp-challenge-TRAIN_Emax.csv"))
lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                        emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
Y = lab_by_smi.reindex(Xtr["SMILES"])

W = {"CYP3A4": (0.6, 0.2, 0.2), "CYP2D6": (0.9, 0.1, 0.0)}
probs = {}
for iso in ISO:
    y = Y[f"{iso}_is_TDI"].values
    lab = ~pd.isna(y)
    a, b, c = W[iso]
    zb = (base[iso][lab] - base[iso][lab].mean()) / base[iso][lab].std()
    ze = (emb[iso][lab] - emb[iso][lab].mean()) / emb[iso][lab].std()
    zt = (tab[iso][lab] - tab[iso][lab].mean()) / tab[iso][lab].std()
    probs[iso] = (a * zb + b * ze + c * zt, lab)

pi_post = {
    "CYP3A4": {0.25: 0.08, 0.30: 0.15, 0.35: 0.22, 0.40: 0.28, 0.45: 0.17, 0.50: 0.10},
    "CYP2D6": {0.216: 0.15, 0.26: 0.22, 0.30: 0.26, 0.34: 0.22, 0.38: 0.10, 0.42: 0.05},
}
fracs = np.arange(0.05, 0.61, 0.01)
out = {}
for iso in ISO:
    p, lab = probs[iso]
    y = Y[f"{iso}_is_TDI"].values
    yl = np.array([1 if bool(v) else 0 for v in y[lab]], dtype=int)
    rates, _ = rank_conditional_curve(p, yl)
    curves = {pi: [round(emcc(rescale_to_pi(rates, pi), f), 4) for f in fracs] for pi in pi_post[iso]}
    weighted = np.zeros(len(fracs))
    for pi, c in curves.items():
        weighted += pi_post[iso][pi] * np.array(c)
    worst = np.min(np.array(list(curves.values())), axis=0)
    ib, im = int(np.argmax(weighted)), int(np.argmax(worst))
    out[iso] = {
        "posterior_optimal_frac": float(fracs[ib]),
        "posterior_optimal_mcc": round(float(weighted[ib]), 4),
        "minimax_optimal_frac": float(fracs[im]),
        "weighted_mcc_at_f": {f"{f:.2f}": round(float(weighted[int(round((f - 0.05) / 0.01))]), 4)
                              for f in [0.08, 0.15, 0.25, 0.28, 0.33, 0.36, 0.40, 0.43, 0.50]},
    }
    print(iso, json.dumps(out[iso], indent=1), flush=True)

with open(os.path.join(CACHE, "tdi_fraction_optima_v3.json"), "w") as fh:
    json.dump(out, fh, indent=2)
