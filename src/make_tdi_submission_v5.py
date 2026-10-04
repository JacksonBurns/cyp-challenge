"""TDI submission v5 candidate: v3 pool + UMTM is_TDI members, family-capped.

Blend weights: mean of per-fold greedy weights UNDER the 0.5 family cap from
cache/tdi_family_block_audit4.json pool v6b merged
= {base, emb, tab, tabcp, tabcpext} + seed-averaged UMTM is_TDI members
L1/L2/L3/L4 (cache/umtm_L*_tdi.npz; nested blend 2D6 0.212 / 3A4 0.4688,
capped nested macro 0.3404 vs v3 cap 0.3279). UMTM families survive the
paranoid-merge audit (audit4 v6b paranoid capped macro 0.3371, i.e. the gain
is not lineage re-draw).
Fractions: 2D6 0.15 / 3A4 0.24 = argmax E[MCC] on the v6b-capped pooled OOF
under the board pi prior (cache/tdi_fraction_optima_v7.json; plateau
2D6 0.15-0.19, 3A4 0.23-0.31). E[P] at those fractions 0.30/0.41 - the plan's
0.45 precision floor is NOT satisfiable on any 2D6 blend we have (all
posterior maxima < 0.45, sec 22), so 2D6 ships at the unconstrained argmax.

Usage (cyp env): python src/make_tdi_submission_v5.py [frac_2d6 frac_3a4]
Writes cache/tdi_submission_v5_candidate.csv; validates + row-for-row verifies.
"""
import glob
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


def umtm_test_probs(lin, iso):
    """Seed-averaged is_TDI test probabilities for a lineage (prob space,
    matching umtm_to_tdi_npz OOF averaging)."""
    files = sorted(glob.glob(os.path.join(CACHE, f"umtm_test_{lin}*.csv")))
    col = f"is_TDI_{'2D6' if iso == 'CYP2D6' else '3A4'}"
    p = None
    for f in files:
        v = pd.read_csv(f).set_index("SMILES")[col]
        p = v if p is None else p + v
    return p / len(files)


def threshold_for_fraction(p, f):
    order = np.sort(p)
    k = int(round(f * len(p)))
    k = max(1, min(len(p) - 1, k))
    return float(order[-k])


def main(fracs=None):
    fr = fracs or {"CYP2D6": 0.15, "CYP3A4": 0.24}
    audit = json.load(open(os.path.join(CACHE, "tdi_family_block_audit4.json")))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    sub = test.copy()
    for iso in ISO:
        ws = audit[iso]["v6b"]["merged"]["fold_nested_capped_weights"]
        names = sorted({k for w in ws for k in w})
        wavg = {k: float(np.mean([w.get(k, 0.0) for w in ws])) for k in names}
        z = None
        for k in names:
            if wavg[k] == 0:
                continue
            if k.startswith("umtm_"):
                p = umtm_test_probs(k[5:], iso).reindex(test["SMILES"]).values
            else:
                p = pd.read_csv(os.path.join(CACHE, SRC[k])).set_index("SMILES")[f"{iso}_proba"]
                p = p.reindex(test["SMILES"]).values
            zz = wavg[k] * (p - p.mean()) / p.std()
            z = zz if z is None else z + zz
        thr = threshold_for_fraction(z, fr[iso])
        pred = z >= thr
        sub[f"{iso}_is_TDI"] = pred
        print(f"{iso}: weights { {k: round(v,3) for k,v in wavg.items()} }, fraction {fr[iso]}, achieved {pred.mean():.4f}")
    out = os.path.join(CACHE, "tdi_submission_v5_candidate.csv")
    sub.to_csv(out, index=False)
    ok, errs = validate_tdi_submission(out, expected_ids=set(test["Molecule_Name"].astype(str)))
    print("VALID:", ok, errs)


if __name__ == "__main__":
    fr = None
    if len(sys.argv) == 3:
        fr = {"CYP2D6": float(sys.argv[1]), "CYP3A4": float(sys.argv[2])}
    main(fr)
