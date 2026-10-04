"""Fix malformed TDI submission: the form requires BOTH the pIC50
direct-inhibition columns (1A2/2C9/2D6/3A4) AND the is_TDI columns (2D6/3A4).

Our tdi_submission_v5_candidate.csv only carried the two is_TDI columns.
Join the verified regression cp15 pIC50 predictions onto the TDI rows
(1:1 on Molecule_Name, SMILES-identical) to produce the combined file.

Writes cache/tdi_submission_v5_candidate.csv (in place, superseding the
malformed copy) and re-validates.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

PCOLS = ["CYP1A2_pIC50_direct_inhibition", "CYP2C9_pIC50_direct_inhibition",
         "CYP2D6_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition"]
IS_TDI = ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]


def main():
    reg = pd.read_csv(os.path.join(CACHE, "regression_final_cp15_submission.csv"))
    tdi = pd.read_csv(os.path.join(CACHE, "tdi_submission_v5_candidate.csv"))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))

    assert tdi["Molecule_Name"].is_unique
    assert reg["Molecule_Name"].is_unique
    # join 1:1
    merged = tdi.merge(reg[["Molecule_Name"] + PCOLS], on="Molecule_Name", how="left",
                       validate="one_to_one")
    assert merged["Molecule_Name"].is_unique, "join produced duplicates"
    assert merged[PCOLS].isna().sum().sum() == 0, "null pIC50 after join"

    # SMILES consistency
    sm = merged.set_index("Molecule_Name")["SMILES"]
    ref = test.set_index("Molecule_Name")["SMILES"]
    assert (sm.reindex(ref.index) == ref).all(), "SMILES mismatch vs test"

    # column order: ids, pIC50 (1A2,2C9,2D6,3A4), is_TDI (2D6,3A4)
    out_cols = ["SMILES", "Molecule_Name"] + PCOLS + IS_TDI
    merged = merged[out_cols]

    out = os.path.join(CACHE, "tdi_submission_v5_candidate.csv")
    merged.to_csv(out, index=False)
    print("rows", len(merged), "cols", list(merged.columns))
    print(merged.head(3).to_string(index=False))

    # re-validate is_TDI columns via the project validator
    sys.path.insert(0, ROOT)
    from validation.tdi_validation import validate_tdi_submission
    ok, errs = validate_tdi_submission(out, expected_ids=set(test["Molecule_Name"].astype(str)))
    print("TDI validator:", ok, errs)

    # explicit combined-column check (what the form complained about)
    need = PCOLS + IS_TDI
    missing = [c for c in need if c not in merged.columns]
    print("combined required cols present:", not missing, "missing:", missing)


if __name__ == "__main__":
    main()
