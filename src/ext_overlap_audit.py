"""Overlap audit: challenge train/test SMILES vs external DBs, + CV with external
features. Leak rule: a training row whose own SMILES is in the external DB would
peek its own label via the external feature; CV must blank the self-match.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")
sys.path.insert(0, HERE)

cb = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"), usecols=["canonical_smiles"])
pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"), usecols=["smiles"])
Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"), columns=["SMILES"])["SMILES"]
Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"), columns=["SMILES"])["SMILES"]
cbS = set(cb["canonical_smiles"].dropna())
pcS = set(pc["smiles"].dropna())
print(f"ChEMBL unique smiles {len(cbS)}; PubChem {len(pcS)}")
print("train overlap: chembl", len(set(Xtr) & cbS), "pubchem", len(set(Xtr) & pcS), f"/ {len(set(Xtr))}")
print("test  overlap: chembl", len(set(Xte) & cbS), "pubchem", len(set(Xte) & pcS), f"/ {len(set(Xte))}")
