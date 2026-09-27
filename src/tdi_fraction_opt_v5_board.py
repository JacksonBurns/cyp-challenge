"""Board-informed fraction posterior (step 1c follow-up).

The shipped fraction machinery (tdi_fraction_opt*.py) centered the test
positive-rate prior at pi ~0.30-0.40. Our OWN two scored v2 points let us
measure the blind pi directly: pi_i = prec_i * f_i / rec_i. Macro:
  Sep 23 board (v2, f=0.08/0.36): prec 0.408 rec 0.537 -> implied pi ~0.167
  Sep 26 board (same file):       prec 0.359 rec 0.672 -> implied pi ~0.118
Both far below the prior's center of mass. This script re-runs the expected-MCC
posterior with a BOARD-INFORMED pi prior (0.10-0.22, center ~0.16) on every
honest score variant and reports the fraction grid, so the fraction call is
made against the pi the blind set actually has, not the one we guessed.

Writes cache/tdi_fraction_optima_v5_board.json
Usage (cyp env): python src/tdi_fraction_opt_v5_board.py
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

# Board-informed prior: implied pi from both scored v2 points is ~0.12-0.17;
# keep modest mass at train rate 0.213 and at 0.08/0.25 for tail robustness.
PI_PRIOR = {0.08: 0.10, 0.11: 0.20, 0.14: 0.25, 0.17: 0.22, 0.213: 0.15, 0.25: 0.08}
fracs = np.arange(0.05, 0.46, 0.01)
SHIPPED = {"CYP2D6": 0.08, "CYP3A4": 0.36}
N_TRIALS = 1500


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
    for nm, fn in [("v2_nested", "tdi_fbnest_v2_{iso}.npy"),
                   ("v2_nested_cap", "tdi_fbnestcap_v2_{iso}.npy"),
                   ("v3_nested", "tdi_fbnest_v3_{iso}.npy"),
                   ("v3_nested_cap", "tdi_fbnestcap_v3_{iso}.npy")]:
        p = os.path.join(CACHE, fn.format(iso=iso))
        if os.path.exists(p):
            variants[nm] = np.load(p)
    out[iso] = {"pi_prior": PI_PRIOR}
    for nm, p in variants.items():
        rates, _ = rank_conditional_curve(p, yl)
        w_mcc = np.zeros(len(fracs)); w_acc = np.zeros(len(fracs))
        per_pi = {}
        for pi, pw in PI_PRIOR.items():
            pr = rescale_to_pi(rates, pi)
            mc, ac = np.array([0.0] * len(fracs)), np.array([0.0] * len(fracs))
            for i, f in enumerate(fracs):
                v = emcc_eacc(pr, f)
                mc[i], ac[i] = v[0], v[1]
            per_pi[str(pi)] = {"emcc": [round(float(x), 3) for x in mc],
                               "eacc": [round(float(x), 3) for x in ac]}
            w_mcc += pw * mc; w_acc += pw * ac
        ib = int(np.argmax(w_mcc))
        i_sh = int(round((SHIPPED[iso] - 0.05) / 0.01))
        rec = {"argmax_f": float(fracs[ib]), "argmax_emcc": round(float(w_mcc[ib]), 4),
               "argmax_eacc": round(float(w_acc[ib]), 3),
               "shipped_f": SHIPPED[iso], "shipped_emcc": round(float(w_mcc[i_sh]), 4),
               "shipped_eacc": round(float(w_acc[i_sh]), 3),
               "grid": {f"{f:.2f}": [round(float(w_mcc[i]), 4), round(float(w_acc[i]), 3)]
                        for i, f in enumerate(fracs) if int(round(f * 100)) % 2 == 0}}
        out[iso][nm] = rec
        print(f"{iso} {nm}: argmax f={rec['argmax_f']:.2f} E[MCC]={rec['argmax_emcc']}"
              f" E[acc]={rec['argmax_eacc']} | shipped {SHIPPED[iso]}:"
              f" E[MCC]={rec['shipped_emcc']} E[acc]={rec['shipped_eacc']}", flush=True)

with open(os.path.join(CACHE, "tdi_fraction_optima_v5_board.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote cache/tdi_fraction_optima_v5_board.json")
