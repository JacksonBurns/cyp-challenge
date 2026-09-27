"""Step 1c (NEXT_AGENT_PROMPT): expected-MCC fraction posterior on ALL honest
TDI score variants, with shipped-fraction evaluation and accuracy diagnostics.

Variants analyzed (per iso):
  v2_static        - the weights make_tdi_submission_v2.py ships on OOF rows
                     (2D6 0.9base+0.1emb; 3A4 0.6/0.2/0.2)
  v2_nested        - per-fold greedy blend over the v2 pool (tdi_fbnest_v2)
  v2_nested_cap    - same, family weight capped at 0.5
  v3_nested        - per-fold greedy blend over the v3 pool incl tabcp(+ext)
  v3_nested_cap    - same, capped
The Sep 26 board (acc 0.725 / prec 0.359 vs top-10 acc >= 0.839 / prec >= 0.44)
says at least one isoform's shipped fraction is badly mis-tuned, so this reports
E[MCC] AND E[accuracy] at every fraction, per-pi sensitivity curves, and the
value of the shipped 0.08/0.36 vs the argmax on each variant.

Writes cache/tdi_fraction_optima_v5.json
Usage (cyp env): python src/tdi_fraction_opt_v5.py
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

pi_post = {
    "CYP3A4": {0.25: 0.08, 0.30: 0.15, 0.35: 0.22, 0.40: 0.28, 0.45: 0.17, 0.50: 0.10},
    "CYP2D6": {0.216: 0.15, 0.26: 0.22, 0.30: 0.26, 0.34: 0.22, 0.38: 0.10, 0.42: 0.05},
}
fracs = np.arange(0.05, 0.61, 0.01)
SHIPPED = {"CYP2D6": 0.08, "CYP3A4": 0.36}  # tdi_submission_v2.csv on the board
N_TRIALS = 1200


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


# ---- labels ------------------------------------------------------------------
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                        emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
Y = lab_by_smi.reindex(Xtr["SMILES"])

base = np.load(os.path.join(CACHE, "tdi_oof.npz"))
emb = np.load(os.path.join(CACHE, "tdi_emb_oof.npz"))
tab = np.load(os.path.join(CACHE, "tdi_tabicl_oof.npz"))
W_STATIC = {"CYP2D6": (0.9, 0.1, 0.0), "CYP3A4": (0.6, 0.2, 0.2)}


def z(v):
    return (v - v.mean()) / v.std()


variants_fn = {
    "v2_nested": "tdi_fbnest_v2_{iso}.npy",
    "v2_nested_cap": "tdi_fbnestcap_v2_{iso}.npy",
    "v3_nested": "tdi_fbnest_v3_{iso}.npy",
    "v3_nested_cap": "tdi_fbnestcap_v3_{iso}.npy",
}

out = {}
for iso in ISO:
    y = Y[f"{iso}_is_TDI"].values
    m = ~pd.isna(y)
    yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
    a, b, c = W_STATIC[iso]
    scores = {
        "v2_static": a * z(base[iso][m]) + b * z(emb[iso][m]) + c * z(tab[iso][m]),
    }
    for nm, fn in variants_fn.items():
        p = os.path.join(CACHE, fn.format(iso=iso))
        if os.path.exists(p):
            scores[nm] = np.load(p)
    out[iso] = {}
    for nm, p in scores.items():
        rates, _ = rank_conditional_curve(p, yl)
        curves = {pi: [] for pi in pi_post[iso]}
        w_mcc = np.zeros(len(fracs)); w_acc = np.zeros(len(fracs))
        for pi in pi_post[iso]:
            pr = rescale_to_pi(rates, pi)
            for i, f in enumerate(fracs):
                mval, aval = emcc_eacc(pr, f)
                curves[pi].append((round(mval, 4), round(aval, 3)))
                w_mcc[i] += pi_post[iso][pi] * mval
                w_acc[i] += pi_post[iso][pi] * aval
        ib = int(np.argmax(w_mcc))
        f_sh = SHIPPED[iso]
        i_sh = int(round((f_sh - 0.05) / 0.01))
        # fraction where E[acc] crosses 0.83 (the Sep 26 top-10 floor)
        acc83 = [float(f) for f, aq in zip(fracs, w_acc) if aq >= 0.83]
        rec = {
            "argmax_f": float(fracs[ib]), "argmax_emcc": round(float(w_mcc[ib]), 4),
            "argmax_eacc_at_argmax": round(float(w_acc[ib]), 3),
            f"shipped_f_{f_sh}": {"emcc": round(float(w_mcc[i_sh]), 4),
                                  "eacc": round(float(w_acc[i_sh]), 3),
                                  "gap_vs_argmax": round(float(w_mcc[ib] - w_mcc[i_sh]), 4)},
            "max_f_with_eacc_ge_0.83": max(acc83) if acc83 else None,
            "emcc_at_max_acc_f": round(float(w_mcc[[i for i, aq in enumerate(w_acc) if aq >= 0.83][-1]]), 4) if acc83 else None,
        }
        rec["grid"] = {f"{f:.2f}": [round(float(w_mcc[i]), 4), round(float(w_acc[i]), 3)]
                       for i, f in enumerate(fracs) if abs(f * 100 - round(f * 100)) < 1e-6 and int(round(f * 100)) % 3 == 0}
        out[iso][nm] = rec
        print(f"{iso} {nm}: argmax f={rec['argmax_f']:.2f} E[MCC]={rec['argmax_emcc']}"
              f" | shipped f={f_sh} E[MCC]={rec[f'shipped_f_{f_sh}']['emcc']}"
              f" gap={rec[f'shipped_f_{f_sh}']['gap_vs_argmax']}"
              f" | max f with E[acc]>=0.83: {rec['max_f_with_eacc_ge_0.83']}", flush=True)

with open(os.path.join(CACHE, "tdi_fraction_optima_v5.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote cache/tdi_fraction_optima_v5.json")
