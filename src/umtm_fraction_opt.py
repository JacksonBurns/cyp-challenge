"""Precision-floor fraction search for UMTM TDI probabilities (FINAL_PUSH_PLAN 1.5).

Maximize E[MCC] subject to E[precision] >= P_FLOOR (0.45), using v6's rank-
conditional posterior machinery: blind positive rate rescaled to the pi prior
{0.08..0.25}, N trials per (pi, f); reports the argmax, the constrained argmax,
and the E[MCC]/E[P] grid. Usage (cyp env):
  python src/umtm_fraction_opt.py cache/umtm_oof_L2.csv [more oof csvs -> averaged]
"""
import argparse
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
rng = np.random.default_rng(11)
PI_PRIOR = {0.08: 0.10, 0.11: 0.20, 0.14: 0.25, 0.17: 0.22, 0.213: 0.15, 0.25: 0.08}
FRACS = np.arange(0.05, 0.46, 0.01)
N_TRIALS = 1500
P_FLOOR = 0.45


def trials(probs_by_rank, f):
    n = len(probs_by_rank)
    k = max(1, min(n - 1, int(round(f * n))))
    tm = ta = tp_ = 0.0
    for _ in range(N_TRIALS):
        y = (rng.random(n) < probs_by_rank).astype(int)
        tp = int(y[:k].sum()); fp = k - tp
        P = int(y.sum()); fn = P - tp; tn = n - k - fn
        tm += mcc_counts(tp, fp, tn, fn)
        tp_ += tp / k if k else 0.0
    return tm / N_TRIALS, tp_ / N_TRIALS


def truth():
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    return pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                      emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]
                      ]).drop_duplicates("SMILES").set_index("SMILES")


LAB = truth()


def main(paths):
    dfs = [pd.read_csv(p) for p in paths]
    base = dfs[0].copy()
    for iso in ISO:
        short = iso[3:]  # master heads use is_TDI_2D6 / is_TDI_3A4
        p = np.mean([d[f"is_TDI_{short}"].values.astype(float) for d in dfs], axis=0)
        y = LAB[f"{iso}_is_TDI"].reindex(base["SMILES"]).values
        m = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
        rates, _ = rank_conditional_curve(p[m], yl)
        w_mcc = np.zeros(len(FRACS)); w_p = np.zeros(len(FRACS))
        for pi, pw in PI_PRIOR.items():
            pr = rescale_to_pi(rates, pi)
            for i, f in enumerate(FRACS):
                e, ep = trials(pr, f)
                w_mcc[i] += pw * e; w_p[i] += pw * ep
        ia = int(np.argmax(w_mcc))
        ok = w_p >= P_FLOOR
        ic = int(np.argmax(np.where(ok, w_mcc, -1.0))) if ok.any() else -1
        print(f"{iso}: unconstrained argmax f={FRACS[ia]:.2f} E[MCC]={w_mcc[ia]:.4f} "
              f"E[P]={w_p[ia]:.3f}", flush=True)
        if ic >= 0:
            print(f"  CONSTRAINED (E[P]>={P_FLOOR}): f={FRACS[ic]:.2f} E[MCC]={w_mcc[ic]:.4f} "
                  f"E[P]={w_p[ic]:.3f}  (cost vs argmax: {w_mcc[ia]-w_mcc[ic]:.4f})", flush=True)
        else:
            print(f"  NO f meets the floor; max E[P] = {w_p.max():.3f} @{FRACS[int(np.argmax(w_p))]:.2f}", flush=True)
        grid = {f"{f:.2f}": [round(float(w_mcc[i]), 4), round(float(w_p[i]), 3)]
                for i, f in enumerate(FRACS) if int(round(f * 100)) % 2 == 0}
        yield_iso = {"argmax_f": float(round(FRACS[ia], 2)), "argmax_emcc": float(w_mcc[ia]),
                     "constrained_f": float(round(FRACS[ic], 2)) if ic >= 0 else None,
                     "constrained_emcc": float(w_mcc[ic]) if ic >= 0 else None, "grid": grid}
        with open(os.path.join(CACHE, f"umtm_fraction_{'_'.join(iso)}".replace("CYP", "") + f"_{iso}.json"), "w") as fh:
            import json; json.dump(yield_iso, fh, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    main(a.paths)
