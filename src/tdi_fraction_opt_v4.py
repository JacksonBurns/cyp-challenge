"""Fraction machinery (v4) on the nested pooled TDI OOF (tdi_nested_pooled_oof_<iso>.npy).

Same expected-MCC posterior machinery as tdi_fraction_opt_v3, but on the honest
nested-blend score. Writes cache/tdi_fraction_optima_v4.json with a
'shipped_fraction' decision per iso.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")
from tdi_fraction_opt import rank_conditional_curve, emcc, rescale_to_pi, ISO  # noqa: E402

Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                 emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
Y = lab.reindex(Xtr["SMILES"])

pi_post = {
    "CYP3A4": {0.25: 0.08, 0.30: 0.15, 0.35: 0.22, 0.40: 0.28, 0.45: 0.17, 0.50: 0.10},
    "CYP2D6": {0.216: 0.15, 0.26: 0.22, 0.30: 0.26, 0.34: 0.22, 0.38: 0.10, 0.42: 0.05},
}
fracs = np.arange(0.05, 0.61, 0.01)
out = {}
for iso in ISO:
    p = np.load(os.path.join(CACHE, f"tdi_nested_pooled_oof_{iso}.npy"))
    y = Y[f"{iso}_is_TDI"].values
    lab_m = ~pd.isna(y)
    yl = np.array([1 if bool(v) else 0 for v in y[lab_m]], dtype=int)
    rates, _ = rank_conditional_curve(p, yl)
    curves = {pi: [round(emcc(rescale_to_pi(rates, pi), f), 4) for f in fracs] for pi in pi_post[iso]}
    weighted = np.zeros(len(fracs))
    for pi, c in curves.items():
        weighted += pi_post[iso][pi] * np.array(c)
    worst = np.min(np.array(list(curves.values())), axis=0)
    ib, im = int(np.argmax(weighted)), int(np.argmax(worst))
    out[iso] = {"posterior_optimal_frac": float(fracs[ib]),
                "posterior_optimal_mcc": round(float(weighted[ib]), 4),
                "minimax_optimal_frac": float(fracs[im]),
                "minimax_optimal_mcc": round(float(worst[im]), 4)}
    # shipped decision: posterior argmax, snapped to plateau center if flat
    cand = fracs[ib]
    plateau = fracs[weighted >= weighted[ib] - 0.005]
    if len(plateau) > 3:
        cand = float(np.median(plateau))
    out[iso]["shipped_fraction"] = float(cand)
    print(iso, json.dumps(out[iso]))

with open(os.path.join(CACHE, "tdi_fraction_optima_v4.json"), "w") as fh:
    json.dump(out, fh, indent=2)
