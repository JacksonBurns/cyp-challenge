"""Final regression submission builder: N-way z-space mixture + reveal placement.

Generalizes regression_v3.py: candidate test predictions resolve from
  - cache/blend_test_preds.npz keys blend_<iso> (gbm) / concat_<iso> (emb)
  - cache/ft_test_preds_<tag>.csv for ft_* candidates
  - "ft_fullft_avg" = z-mean of the per-seed full-FT runs (fullft, fullft_s1, ...)

Placement = reveal-calibrated moments x OOF_TO_BLIND x F_SPREAD (see NOTES
sec 10). Usage (cyp env):
  python src/regression_final.py --blend cache/blendN_seedavg.json \
      --out cache/regression_final_submission.csv
Then: python src/verify_submissions.py cache/regression_final_submission.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")

F_SPREAD = {"CYP1A2": 0.6, "CYP2C9": 0.6, "CYP2D6": 1.0, "CYP3A4": 0.7}
OOF_TO_BLIND = {"CYP1A2": 1.32, "CYP2C9": 1.23, "CYP2D6": 1.66, "CYP3A4": 1.07}
BLIND_MOMENTS = {"CYP1A2": (4.412, 1.553), "CYP2C9": (4.830, 1.101),
                 "CYP2D6": (3.107, 1.599), "CYP3A4": (4.880, 1.272)}
STRAE_MOMENTS = {"CYP2D6": (3.57, 0.90)}
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
FLOOR = 1.0
FT_PREFIX = "ft_"


def ft_test_pred(key, test_smiles):
    """Resolve a candidate's test predictions; seed-averaged in z-space."""
    tag = key[len(FT_PREFIX):]
    if tag.endswith("_avg"):
        base = tag[:-4]
        files = sorted(glob.glob(os.path.join(CACHE, f"ft_test_preds_{base}*.csv")))
        assert files, f"no test preds for {base}"
        z = None
        for f in files:
            p = pd.read_csv(f).set_index("SMILES")[ISO].reindex(test_smiles).values
            zz = (p - p.mean(0)) / p.std(0)
            z = zz if z is None else z + zz
        return z / len(files)
    p = pd.read_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv")).set_index("SMILES")
    return p[ISO].reindex(test_smiles).values


def main(blend_path, out_path):
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    blend = np.load(os.path.join(CACHE, "blend_test_preds.npz"))
    bN = json.load(open(blend_path if os.path.isabs(blend_path)
                        else os.path.join(CACHE, blend_path)))
    sub = pd.DataFrame({"SMILES": test["SMILES"].values,
                        "Molecule_Name": test["Molecule_Name"].values})
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
                M = ft_test_pred(k, test["SMILES"])
                col = ISO.index(iso)
                v = M[:, col]
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
    out = out_path if os.path.isabs(out_path) else os.path.join(CACHE, out_path)
    sub.to_csv(out, index=False)
    json.dump(report, open(out.replace(".csv", "_report.json"), "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default="blendN_seedavg.json")
    ap.add_argument("--out", default="regression_final_submission.csv")
    a = ap.parse_args()
    main(a.blend, a.out)
