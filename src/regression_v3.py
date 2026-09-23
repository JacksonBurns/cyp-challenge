"""Regression submission v3: N-way z-space mixture per blendN.json weights.

Candidates: gbm (FP GBM 4cfg blend), emb (GBM+CheMeLeon emb), ft_frozen,
ft_fullft (CheMeLeon fine-tunes). OOF weights come from blendN.py (greedy);
placement = reveal-calibrated moments with F_SPREAD shrink (shrink_placement.py).

Raw TEST preds: cache/blend_test_preds.npz (built by regression_v2.py) and
cache/ft_test_preds_<tag>.csv (built by ft_chemeleon.py per run tag).

Output: cache/regression_v3_submission.csv + regression_v3_report.json.
Then run: python src/verify_submissions.py cache/regression_v3_submission.csv
Run with the cyp env.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")

FT_TAGS = ["frozen", "fullft"]
F_SPREAD = {"CYP1A2": 0.6, "CYP2C9": 0.6, "CYP2D6": 1.0, "CYP3A4": 0.7}
OOF_TO_BLIND = {"CYP1A2": 1.32, "CYP2C9": 1.23, "CYP2D6": 1.66, "CYP3A4": 1.07}
BLIND_MOMENTS = {"CYP1A2": (4.412, 1.553), "CYP2C9": (4.830, 1.101),
                 "CYP2D6": (3.107, 1.599), "CYP3A4": (4.880, 1.272)}
STRAE_MOMENTS = {"CYP2D6": (3.57, 0.90)}
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
FLOOR = 1.0

test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
blend = np.load(os.path.join(CACHE, "blend_test_preds.npz"))
ft_te = {t: pd.read_csv(os.path.join(CACHE, f"ft_test_preds_{t}.csv")).set_index("SMILES")
         for t in FT_TAGS}
bN = json.load(open(os.path.join(CACHE, "blendN.json")))

sub = pd.DataFrame({"SMILES": test["SMILES"].values, "Molecule_Name": test["Molecule_Name"].values})
report = {}
for iso in ISO:
    w = bN[iso]["weights"]
    zt = np.zeros(len(test))
    for k, wk in w.items():
        if wk == 0:
            continue
        if k == "gbm":
            v = blend[f"blend_{iso}"]
        elif k == "emb":
            v = blend[f"concat_{iso}"]
        else:
            v = ft_te[k.replace("ft_", "")][iso].reindex(test["SMILES"]).values
        zt += wk * (v - v.mean()) / v.std()
    zt = (zt - zt.mean()) / zt.std()
    rho = float(bN[iso]["pearson"])
    rho_a = float(np.clip(rho * OOF_TO_BLIND[iso], 0.05, 0.95))
    mu, sd = BLIND_MOMENTS[iso][0], BLIND_MOMENTS[iso][1] * rho_a
    if iso in STRAE_MOMENTS:
        mu, sd = STRAE_MOMENTS[iso]
    sd = F_SPREAD[iso] * sd
    placed = np.clip(mu + sd * zt, FLOOR, None)
    sub[f"{iso}_pIC50_direct_inhibition"] = placed
    report[iso] = {"weights": w, "oof_pearson": rho, "rho_blind_assumed": rho_a,
                   "f_spread": F_SPREAD[iso], "placed_mean": float(placed.mean()),
                   "placed_sd": float(placed.std())}
    print(iso, json.dumps(report[iso]), flush=True)

sub.to_csv(os.path.join(CACHE, "regression_v3_submission.csv"), index=False)
json.dump(report, open(os.path.join(CACHE, "regression_v3_report.json"), "w"), indent=2)
print("wrote cache/regression_v3_submission.csv")
