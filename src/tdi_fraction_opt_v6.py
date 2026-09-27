"""Fraction posterior v6: board-informed prior on the step-1b pool blends.

Same machinery/prior as tdi_fraction_opt_v5_board.py (board-implied blind pi
0.12-0.17, N=1500 rank-conditional trials), but the variant list comes from
the audit2 caches (tdi_fb2_*): v4 pool (v3 + tabcpchm + dmpnn), merged and
split family caps, plus the v3 merged numbers for continuity. Reports the
E[MCC]/E[acc] grid + argmax per iso, and each variant's shipped-fraction
value so the incumbent (v2 static 0.08/0.36) can be compared on equal terms.

Writes cache/tdi_fraction_optima_v6.json
Usage (cyp env): python src/tdi_fraction_opt_v6.py
"""
import glob
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
SHIPPED = {"CYP2D6": 0.08, "CYP3A4": 0.36}
N_TRIALS = 1500

VARIANTS = ["v2_static", "v3_nested_cap", "v4_nested", "v4_nested_cap"]


def emcc_eacc(probs_by_rank, f):
    n = len(probs_by_rank)
    k = max(1, min(n - 1, int(round(f * n))))
    tot_m, tot_a = 0.0, 0.0
    for _ in range(N_TRIALS):
        y = (rng.random(n) < probs_by_rank).astype(int)
        tp = int(y[:k].sum()); fp = k - tp
        P = int(y.sum()); fn = P - tp; tn = n - k - fn
        tot_m += mcc_counts(tp, fp, tn, fn)
        tot_a += (tp + tn) / n
    return tot_m / N_TRIALS, tot_a / N_TRIALS


Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                 emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
Y = lab.reindex(Xtr["SMILES"])
base = np.load(os.path.join(CACHE, "tdi_oof.npz"))
emb = np.load(os.path.join(CACHE, "tdi_emb_oof.npz"))
tab = np.load(os.path.join(CACHE, "tdi_tabicl_oof.npz"))
W_ST = {"CYP2D6": (0.9, 0.1, 0.0), "CYP3A4": (0.6, 0.2, 0.2)}


def z(v):
    return (v - v.mean()) / v.std()


out = {}
for iso in ISO:
    y = Y[f"{iso}_is_TDI"].values
    m = ~pd.isna(y)
    yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
    a, b, c = W_ST[iso]
    variants = {"v2_static": a * z(base[iso][m]) + b * z(emb[iso][m]) + c * z(tab[iso][m])}
    p3c = os.path.join(CACHE, f"tdi_fbnestcap_v3_{iso}.npy")
    if os.path.exists(p3c):
        variants["v3_nested_cap"] = np.load(p3c)
    for nm, pref in [("v4_nested", "tdi_fb2_v4_merged_{iso}.npy"),
                     ("v4_nested_cap", "tdi_fb2_v4_merged_{iso}cap.npy"),
                     ("v4_nested_split", "tdi_fb2_v4_split_{iso}.npy"),
                     ("v4_nested_cap_split", "tdi_fb2_v4_split_{iso}cap.npy")]:
        p = os.path.join(CACHE, pref.format(iso=iso))
        if os.path.exists(p):
            variants[nm] = np.load(p)
    out[iso] = {"pi_prior": {str(k): v for k, v in PI_PRIOR.items()}}
    for nm, p in variants.items():
        rates, _ = rank_conditional_curve(p, yl)
        w_mcc = np.zeros(len(fracs)); w_acc = np.zeros(len(fracs))
        for pi, pw in PI_PRIOR.items():
            pr = rescale_to_pi(rates, pi)
            for i, f in enumerate(fracs):
                v = emcc_eacc(pr, f)
                w_mcc[i] += pw * v[0]; w_acc[i] += pw * v[1]
        ib = int(np.argmax(w_mcc))
        i_sh = int(round((SHIPPED[iso] - 0.05) / 0.01))
        rec = {"argmax_f": float(round(fracs[ib], 2)), "argmax_emcc": round(float(w_mcc[ib]), 4),
               "argmax_eacc": round(float(w_acc[ib]), 3),
               "plateau": [float(round(fracs[i], 2)) for i in np.where(w_mcc >= w_mcc[ib] - 0.005)[0]],
               "shipped_f": SHIPPED[iso], "shipped_emcc": round(float(w_mcc[i_sh]), 4),
               "shipped_eacc": round(float(w_acc[i_sh]), 3),
               "grid": {f"{f:.2f}": [round(float(w_mcc[i]), 4), round(float(w_acc[i]), 3)]
                        for i, f in enumerate(fracs) if int(round(f * 100)) % 2 == 0}}
        out[iso][nm] = rec
        print(f"{iso} {nm}: argmax f={rec['argmax_f']:.2f} E[MCC]={rec['argmax_emcc']}"
              f" E[acc]={rec['argmax_eacc']} | shipped {SHIPPED[iso]}:"
              f" E[MCC]={rec['shipped_emcc']} E[acc]={rec['shipped_eacc']}", flush=True)

with open(os.path.join(CACHE, "tdi_fraction_optima_v6.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote cache/tdi_fraction_optima_v6.json")
