"""ADME3b: IUPAC-name captioning distillation.

From the user's 24k high-confidence IUPAC names: build a token vocabulary
of chemically meaningful fragments (substituents, scaffolds, suffix classes).
Multi-label GBMs trained on 10.7k external molecules (name -> token labels,
features = desc+Morgan) predict token probabilities for all 6895 challenge
molecules = structural-semantic feature block (position-aware substructure
semantics the descriptor stack lacks). Output: cache/adme3/toks_feats_{tr,te}.npy
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
from adme2_build import canon, fit_predict  # noqa: E402

BASE = "/home/jackson/cyp-challenge"
CACHE = os.path.join(BASE, "cache")
OUTD = os.path.join(CACHE, "adme3")

# chemically meaningful token families to track (substring matches, lowercased)
TOKENS = [
    "acid", "amide", "anilide", "aniline", "anilino", "azetidine", "azetidin",
    "benzenesulfon", "benzimidazol", "benzofuran", "benzyl", "biphenyl",
    "bromo", "butyl", "carbamate", "carbamoyl", "carbonitrile", "carboxamide",
    "carboxyl", "chloro", "cyclohex", "cycloprop", "difluoro", "dimethyl",
    "diox", "ethoxy", "ethyl", "fluoro", "furan", "imidazol", "indazol",
    "indol", "isopropyl", "morpholin", "methyl", "naphthal", "nitro",
    "oxadiazol", "oxazol", "pyrazol", "pyridin", "pyrimidin", "pyrrolidin",
    "quinolin", "quinucl", "sulfonamide", "sulfanyl", "sulfoxid", "sulfonyl",
    "tetrazol", "tetrahydronaphth", "thiazol", "thiophen", "tolyl", "trifluoro",
    "trimethyl", "urea", "valer", "phenol", "hydroxy", "amino", "oxo",
    "spiro", "benzo", "propane", "ethane", "acetamide", "acetyl", "propyl",
    "pentyl", "methylsulfonyl", "carbonyl",
]


def tokenize(name):
    s = str(name).lower()
    y = np.zeros(len(TOKENS))
    for i, t in enumerate(TOKENS):
        if t in s:
            y[i] = 1.0
    return y


def main():
    d = pd.read_csv(os.path.join(BASE, "external/user_supplied/iupac_highconf_v24.csv"))
    d["canon"] = d["SMILES"].map(canon)
    nm = d["original_IUPAC_names"].astype(str)
    d = d.assign(nm=nm)
    d = d.dropna(subset=["canon"])
    d = d[(d["nm"] != "nan") & (d["nm"].str.len() > 3)]
    # one name per canonical molecule (first)
    d = d.drop_duplicates("canon")
    Y = np.vstack([tokenize(s) for s in d["nm"]])
    keep = Y.mean(axis=0) > 0.004  # >=~40 molecules
    Y = Y[:, keep]
    toks = [t for t, k in zip(TOKENS, keep) if k]
    print("tokens kept:", len(toks), "mols:", len(d), flush=True)

    epath = os.path.join(OUTD, "ext_desc3.pkl")
    import pickle
    with open(epath, "rb") as fh:
        Xext = pickle.load(fh)
    tpath = os.path.join(CACHE, "adme2", "challenge_desc2.pkl")
    with open(tpath, "rb") as fh:
        Xtr_d, Xte_d = pickle.load(fh)

    ext_canon = list(d["canon"])
    pos = {c: i for i, c in enumerate(Xext.index)}
    pairs = [(pos[c], i) for i, c in enumerate(ext_canon) if c in pos]
    xi = np.array([a for a, _ in pairs])
    yi = np.array([b for _, b in pairs])
    Xfit = Xext.values.astype(np.float32)[xi]
    Yfit = Y[yi]
    m = np.isfinite(Xfit).all(axis=1)
    Ftr, Fte = [], []
    for j in range(Yfit.shape[1]):
        y = Yfit[m][:, j]
        if y.sum() < 50:
            continue
        p1, p2 = fit_predict(Xfit[m], y, Xtr_d.values.astype(np.float32),
                             Xte_d.values.astype(np.float32))
        Ftr.append(p1)
        Fte.append(p2)
        print(f"  token {toks[j]}: pos {int(y.sum())}", flush=True)
    Atr = np.column_stack(Ftr)
    Ate = np.column_stack(Fte)
    np.save(os.path.join(OUTD, "toks_feats_tr.npy"), Atr)
    np.save(os.path.join(OUTD, "toks_feats_te.npy"), Ate)
    open(os.path.join(OUTD, "toks_names.txt"), "w").write("\n".join(toks))
    print("toks feats:", Atr.shape, Ate.shape)


if __name__ == "__main__":
    main()
