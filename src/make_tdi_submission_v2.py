"""TDI submission v2: 3-way blend of test scores + fractions from v3 analysis.

Scores: z-space per iso: 3A4 0.6*base + 0.2*emb + 0.2*tabicl; 2D6 0.9*base +
0.1*emb (cache/tdi_blend3.json OOF optima). Fractions (expected-MCC machinery
on the blended OOF, cache/tdi_fraction_optima_v3.json): 2D6 f=0.08; 3A4 f=0.36
(center of the 0.36-0.50 plateau, minimax 0.38, v2 shipped at 0.43).

Usage (cyp env): python src/make_tdi_submission_v2.py [frac_2d6 frac_3a4]
Writes cache/tdi_submission_v2.csv; validates + verifies against BLINDED CSV.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

from validation.tdi_validation import validate_tdi_submission  # noqa: E402

ISO = ["CYP2D6", "CYP3A4"]
W = {"CYP2D6": {"tdi_test_probs.csv": 0.9, "tdi_emb_test_probs.csv": 0.1,
                "tdi_tabicl_test_probs.csv": 0.0},
     "CYP3A4": {"tdi_test_probs.csv": 0.6, "tdi_emb_test_probs.csv": 0.2,
                "tdi_tabicl_test_probs.csv": 0.2}}


def threshold_for_fraction(p, f):
    order = np.sort(p)
    k = int(round(f * len(p)))
    k = max(1, min(len(p) - 1, k))
    return float(order[-k])


def main(fracs=None):
    fr = fracs or {"CYP2D6": 0.08, "CYP3A4": 0.36}
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    prob = {}
    for iso in ISO:
        z = None
        for f, w in W[iso].items():
            if w == 0:
                continue
            p = pd.read_csv(os.path.join(CACHE, f)).set_index("SMILES")[f"{iso}_proba"]
            p = p.reindex(test["SMILES"]).values
            zz = w * (p - p.mean()) / p.std()
            z = zz if z is None else z + zz
        prob[iso] = z
    sub = test.copy()
    for iso in ISO:
        p = prob[iso]
        thr = threshold_for_fraction(p, fr[iso])
        pred = p >= thr
        sub[f"{iso}_is_TDI"] = pred
        print(f"{iso}: target fraction {fr[iso]}, threshold {thr:.4f}, "
              f"achieved {pred.mean():.4f} ({pred.sum()}/{len(p)})")
    out = os.path.join(CACHE, "tdi_submission_v2.csv")
    sub.to_csv(out, index=False)
    ok, errs = validate_tdi_submission(out, expected_ids=set(test["Molecule_Name"].astype(str)))
    print("VALID:", ok, errs)


if __name__ == "__main__":
    fr = None
    if len(sys.argv) == 3:
        fr = {"CYP2D6": float(sys.argv[1]), "CYP3A4": float(sys.argv[2])}
    main(fr)
