"""Q4 (CPU): 2D6 noise-floor / ceiling analysis.

Question: is 2D6 hard because the LABEL is noisy (inter-assay variance) or
because the FEATURES are blind (chemistry underdetermined)?

Method: ChEMBL pchembl_value gives per-molecule replicate pIC50 across
assays. For each isoform, within-molecule between-assay std over molecules
with >=3 assays = inter-lab noise floor sigma_lab. Model error floor:
best achievable RMSE ~ sigma_lab (labels are noisy draws). Compare across
isoforms: if sigma_2D6 >> sigma_others, the 2D6 resistance is substantially
IRREDUCIBLE label noise -> know-when-to-stop evidence.
Also: distribution of |pIC50 - molecule median| = practical RAE floor for
ST-RAE denominators.
"""
import json

import numpy as np
import pandas as pd

d = pd.read_csv("/home/jackson/cyp-challenge/external/chembl_cyp_ic50.csv")
d = d[d["pchembl_value"].notna()]
d["p"] = d["pchembl_value"].astype(float)
# clean: exact =, nM units, human, no flagged duplicates
d = d[(d["standard_relation"] == "=")]
d = d[d["target_organism"].astype(str).str.contains("sapiens", na=False)]

out = {}
for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4", "CYP2C19"]:
    s = d[d["isoform"] == iso]
    g = s.groupby("canonical_smiles")["p"]
    rep = g.agg(["count", "std", "mean"]).reset_index()
    rep = rep[(rep["count"] >= 3) & rep["std"].notna()]
    rep = rep[np.abs(rep["std"]) < 4]  # drop chemistry-gone-wild outliers
    n_mol = len(rep)
    pooled = float(np.sqrt((rep["std"] ** 2).mean())) if n_mol else np.nan
    # weighted by (n-1) dof
    w = (rep["count"] - 1).values
    pooled_w = float(np.sqrt(np.average(rep["std"] ** 2, weights=w))) \
        if n_mol else np.nan
    med = float(rep["std"].median()) if n_mol else np.nan
    # same-assay multiple rows = protocol replicate noise (tighter)
    g2 = s.groupby(["canonical_smiles", "assay_chembl_id"])["p"].agg(
        ["count", "std"]).reset_index()
    g2 = g2[(g2["count"] >= 2) & g2["std"].notna()]
    intra = float(np.sqrt((g2["std"] ** 2).mean())) if len(g2) else np.nan
    out[iso] = {"n_multi_assay_mols": int(n_mol),
                "inter_assay_std_pooled": round(pooled_w, 4),
                "inter_assay_std_median": round(med, 4),
                "intra_assay_std": round(intra, 4),
                "n_mols_total": int(s["canonical_smiles"].nunique()),
                "n_assays": int(s["assay_chembl_id"].nunique())}
    print(iso, json.dumps(out[iso]), flush=True)

r2d6 = out["CYP2D6"]["inter_assay_std_pooled"]
others = np.mean([out[i]["inter_assay_std_pooled"]
                  for i in ("CYP1A2", "CYP2C9", "CYP3A4")])
out["VERDICT"] = {
    "sigma_2D6_over_others": round(float(r2d6 / others), 3),
    "read": ("2D6 label noise is ELEVATED: much of the gap is irreducible"
             if r2d6 / others > 1.25 else
             "2D6 label noise ~ others: resistance is feature-side, "
             "not noise floor"),
}
print(json.dumps(out["VERDICT"]), flush=True)
json.dump(out, open("/home/jackson/cyp-challenge/cache/q4_noise.json", "w"),
          indent=1)
