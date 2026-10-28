"""Audit the user's IUPAC/pKa high-confidence dataset:
- unique molecule count, pKa usability (type distribution, aggregation)
- canonical overlap with challenge train/test
- IUPAC name corpus stats (tokens, stereo descriptors)
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

import hinge_blend as H  # noqa: E402


def canon(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


d = pd.read_csv("/home/jackson/cyp-challenge/external/user_supplied/iupac_highconf_v24.csv")
print("rows", len(d))
d["canon"] = d["SMILES"].map(canon)
print("unparseable SMILES:", d["canon"].isna().sum())
u = d.dropna(subset=["canon"])
print("unique canon molecules:", u["canon"].nunique())

pv = pd.to_numeric(d["pka_value"], errors="coerce")
print("usable pka_value rows:", pv.notna().sum())
print("pka_type:", d["pka_type"].astype(str).value_counts().head(8).to_dict())
print("acidity_label:", d["acidity_label"].astype(str).value_counts().head(6).to_dict())
print("pka value range:", pv.min(), pv.max(), "mean", round(pv.mean(), 2))

# per-molecule aggregation: strongest ionizable (acid: min pKa; base: max pKa)
u2 = u.assign(pka=pv[u.index])
ok = u2.dropna(subset=["pka"])
g = ok.groupby("canon")["pka"]
print("molecules with >=1 pka:", g.size().shape[0],
      "per-mol entries median:", int(g.size().median()))

# overlap
tr = H.Xtr["SMILES"].map(canon)
test = pd.read_csv("/home/jackson/cyp-challenge/data/cyp-challenge-TEST-BLINDED.csv")
te = test["SMILES"].map(canon)
molset = set(ok["canon"])
print("TRAIN overlap (exact, pka rows):", tr.map(lambda c: c in molset).sum() if tr.notna().all() else "n/a", "/", len(tr))
print("TEST overlap:", sum(c in molset for c in te.dropna()), "/", len(te))

# IUPAC names corpus
nm = d["original_IUPAC_names"].astype(str)
hasnm = nm[(nm != "nan") & (nm.str.len() > 3)]
print("name rows:", len(hasnm), "mean len", int(hasnm.str.len().mean()))
stereo = hasnm.str.contains(r"\(\d+\s*[RSEZ]", regex=True).sum()
print("names with stereo descriptors:", stereo)
tok = Counter()
for s in hasnm.sample(min(5000, len(hasnm))):
    for t in re.findall(r"[A-Za-z][a-z0-9,\-\[\]]+", s):
        tok[t] += 1
print("top tokens:", [t for t, _ in tok.most_common(15)])
