"""TDI submission v4 candidate: family-capped v3 blend + board-prior fractions.

Blend weights: mean of the per-fold greedy weights UNDER the 0.5 family cap
(cache/tdi_family_block_audit2.json, v3 pool, merged families,
fold_nested_capped_weights). The cap is the direct mitigation of shipped-v3's
failure mode (free greedy put mean 0.65 / max 0.83 on the chemprop cp family
on 2D6 and the blind refused to inherit it, ratio 0.94).
Fractions: 2D6 0.33 / 3A4 0.26 from cache/tdi_fraction_optima_v6.json
(board-informed pi prior; plateau 2D6 0.32-0.35, 3A4 0.23-0.27, and 0.33
dominates the 2D6 grid at EVERY pi in 0.08-0.213 on this blend).

Usage (cyp env): python src/make_tdi_submission_v4.py [frac_2d6 frac_3a4]
Writes cache/tdi_submission_v4_candidate.csv; validates + row-for-row verifies.
"""
import json
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
SRC = {"base": "tdi_test_probs.csv", "emb": "tdi_emb_test_probs.csv",
       "tab": "tdi_tabicl_test_probs.csv",
       "tabcp": "tdi_tabicl_cp_test_probs.csv",
       "tabcpext": "tdi_tabicl_cp_ext_test_probs.csv"}


def threshold_for_fraction(p, f):
    order = np.sort(p)
    k = int(round(f * len(p)))
    k = max(1, min(len(p) - 1, k))
    return float(order[-k])


def main(fracs=None):
    fr = fracs or {"CYP2D6": 0.33, "CYP3A4": 0.26}
    audit = json.load(open(os.path.join(CACHE, "tdi_family_block_audit2.json")))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    sub = test.copy()
    for iso in ISO:
        ws = audit[iso]["v3"]["merged"]["fold_nested_capped_weights"]
        names = sorted({k for w in ws for k in w})
        wavg = {k: float(np.mean([w.get(k, 0.0) for w in ws])) for k in names}
        z = None
        for k in names:
            p = pd.read_csv(os.path.join(CACHE, SRC[k])).set_index("SMILES")[f"{iso}_proba"]
            p = p.reindex(test["SMILES"]).values
            zz = wavg[k] * (p - p.mean()) / p.std()
            z = zz if z is None else z + zz
        thr = threshold_for_fraction(z, fr[iso])
        pred = z >= thr
        sub[f"{iso}_is_TDI"] = pred
        print(f"{iso}: weights {wavg}, fraction {fr[iso]}, achieved {pred.mean():.4f}")
    out = os.path.join(CACHE, "tdi_submission_v4_candidate.csv")
    sub.to_csv(out, index=False)
    ok, errs = validate_tdi_submission(out, expected_ids=set(test["Molecule_Name"].astype(str)))
    print("VALID:", ok, errs)


if __name__ == "__main__":
    fr = None
    if len(sys.argv) == 3:
        fr = {"CYP2D6": float(sys.argv[1]), "CYP3A4": float(sys.argv[2])}
    main(fr)
