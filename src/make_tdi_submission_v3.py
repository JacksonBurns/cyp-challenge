"""TDI submission v3: nested-weight blend incl. TabICL-on-chemprop embeddings.

Members z-scored on test; per-iso blend weights = mean of the per-fold greedy
weights from cache/tdi_blend_nested.json (honest selection machinery). Fraction
from cache/tdi_fraction_optima_v4.json (posterior machinery on the nested
pooled OOF). Usage (cyp env): python src/make_tdi_submission_v3.py
Writes cache/tdi_submission_v3.csv + validates.
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
       "tab": "tdi_tabicl_test_probs.csv", "extbase": "tdi_extbase_w03_test_probs.csv",
       "tabcp": "tdi_tabicl_cp_test_probs.csv", "tabcpext": "tdi_tabicl_cp_ext_test_probs.csv"}


def threshold_for_fraction(p, f):
    order = np.sort(p)
    k = int(round(f * len(p)))
    k = max(1, min(len(p) - 1, k))
    return float(order[-k])


def main():
    nested = json.load(open(os.path.join(CACHE, "tdi_blend_nested.json")))
    frac = json.load(open(os.path.join(CACHE, "tdi_fraction_optima_v4.json")))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    sub = test.copy()
    for iso in ISO:
        ws = nested[iso]["fold_weights"]
        names = sorted({k for w in ws for k in w})
        wavg = {k: float(np.mean([w.get(k, 0.0) for w in ws])) for k in names}
        z = None
        for k in names:
            p = pd.read_csv(os.path.join(CACHE, SRC[k])).set_index("SMILES")[f"{iso}_proba"]
            p = p.reindex(test["SMILES"]).values
            zz = wavg[k] * (p - p.mean()) / p.std()
            z = zz if z is None else z + zz
        f = frac[iso]["shipped_fraction"]
        thr = threshold_for_fraction(z, f)
        pred = z >= thr
        sub[f"{iso}_is_TDI"] = pred
        print(f"{iso}: weights {wavg}, fraction {f}, achieved {pred.mean():.4f}")
    out = os.path.join(CACHE, "tdi_submission_v3.csv")
    sub.to_csv(out, index=False)
    ok, errs = validate_tdi_submission(out, expected_ids=set(test["Molecule_Name"].astype(str)))
    print("VALID:", ok, errs)


if __name__ == "__main__":
    main()
