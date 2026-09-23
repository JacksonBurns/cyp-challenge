import pandas as pd, numpy as np
pc = pd.read_csv("external/pubchem_cyp_qhts_aid1851.csv")
cols = [c for c in pc.columns if c.endswith("_pIC50")]
for c in cols:
    v = pd.to_numeric(pc[c], errors="coerce")
    print(c, "inf:", int(np.isinf(v).sum()), "min:", np.nanmin(v), "max:", np.nanmax(v))
ch = pd.read_csv("external/chembl_cyp_ic50.csv")
sv = pd.to_numeric(ch["standard_value"], errors="coerce")
print("chembl zeros:", int((sv == 0).sum()), "neg:", int((sv <= 0).sum()))
