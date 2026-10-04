"""Fraction posterior v7: board-informed prior on the sec 22 UMTM-augmented pools.

Same machinery/prior as tdi_fraction_opt_v6.py (board-implied blind pi prior,
N=1500 rank-conditional trials), variants from tdi_family_block_audit4 caches
(tdi_fb4_*): v3 cap (continuity + incumbent v4_candidate machinery), v6a/v6b/
v6c merged free+capped. Reports argmax fraction + E[MCC]/E[acc] + precision
(E[P]) at each fraction so the 0.45 precision floor from the plan can be read.

Writes cache/tdi_fraction_optima_v7.json
Usage (cyp env): python src/tdi_fraction_opt_v7.py
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
from tdi_fraction_opt import rank_conditional_curve, rescale_to_pi, mcc_counts  # noqa: E402

ISO = ["CYP2D6", "CYP3A4"]
rng = np.random.default_rng(7)
PI_PRIOR = {0.08: 0.10, 0.11: 0.20, 0.14: 0.25, 0.17: 0.22, 0.213: 0.15, 0.25: 0.08}
fracs = np.arange(0.05, 0.46, 0.01)
N_TRIALS = 1500

VARIANTS = {
    "v3_nested_cap": "tdi_fb4_v3_merged_{iso}cap.npy",
    "v6a_merged": "tdi_fb4_v6a_merged_{iso}.npy",
    "v6a_cap": "tdi_fb4_v6a_merged_{iso}cap.npy",
    "v6b_merged": "tdi_fb4_v6b_merged_{iso}.npy",
    "v6b_cap": "tdi_fb4_v6b_merged_{iso}cap.npy",
    "v6c_cap": "tdi_fb4_v6c_merged_{iso}cap.npy",
}


def emcc_eacc_ep(probs_by_rank, f):
    n = len(probs_by_rank)
    k = max(1, min(n - 1, int(round(f * n))))
    tot_m, tot_a, tot_p = 0.0, 0.0, 0.0
    for _ in range(N_TRIALS):
        y = (rng.random(n) < probs_by_rank).astype(int)
        tp = int(y[:k].sum()); fp = k - tp
        P = int(y.sum()); fn = P - tp; tn = n - k - fn
        tot_m += mcc_counts(tp, fp, tn, fn)
        tot_a += (tp + tn) / n
        tot_p += tp / k if k else 0.0
    return tot_m / N_TRIALS, tot_a / N_TRIALS, tot_p / N_TRIALS


Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                 emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
Y = lab.reindex(Xtr["SMILES"])

out = {}
for iso in ISO:
    y = Y[f"{iso}_is_TDI"].values
    m = ~pd.isna(y)
    yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
    out[iso] = {"pi_prior": {str(k): v for k, v in PI_PRIOR.items()}}
    for nm, pref in VARIANTS.items():
        p = os.path.join(CACHE, pref.format(iso=iso))
        if not os.path.exists(p):
            continue
        rates, _ = rank_conditional_curve(np.load(p), yl)
        w_mcc = np.zeros(len(fracs)); w_acc = np.zeros(len(fracs)); w_p = np.zeros(len(fracs))
        for pi, pw in PI_PRIOR.items():
            pr = rescale_to_pi(rates, pi)
            for i, f in enumerate(fracs):
                v = emcc_eacc_ep(pr, f)
                w_mcc[i] += pw * v[0]; w_acc[i] += pw * v[1]; w_p[i] += pw * v[2]
        ib = int(np.argmax(w_mcc))
        floor_mask = w_p >= 0.45
        if floor_mask.any():
            ifloor = int(np.argmax(np.where(floor_mask, w_mcc, -9)))
        else:
            ifloor = -1
        rec = {"argmax_f": float(round(fracs[ib], 2)), "argmax_emcc": round(float(w_mcc[ib]), 4),
               "argmax_eacc": round(float(w_acc[ib]), 3), "argmax_ep": round(float(w_p[ib]), 3),
               "floor_argmax_f": (float(round(fracs[ifloor], 2)) if ifloor >= 0 else None),
               "floor_argmax_emcc": (round(float(w_mcc[ifloor]), 4) if ifloor >= 0 else None),
               "plateau": [float(round(fracs[i], 2)) for i in np.where(w_mcc >= w_mcc[ib] - 0.005)[0]],
               "grid": {f"{f:.2f}": [round(float(w_mcc[i]), 4), round(float(w_acc[i]), 3), round(float(w_p[i]), 3)]
                        for i, f in enumerate(fracs) if int(round(f * 100)) % 2 == 0}}
        out[iso][nm] = rec
        print(f"{iso} {nm}: argmax f={rec['argmax_f']:.2f} E[MCC]={rec['argmax_emcc']}"
              f" E[P]={rec['argmax_ep']} | P>=0.45 best f={rec['floor_argmax_f']}"
              f" E[MCC]={rec['floor_argmax_emcc']}", flush=True)

with open(os.path.join(CACHE, "tdi_fraction_optima_v7.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote cache/tdi_fraction_optima_v7.json")
