"""Robust overlap check: do the user ADME datasets share molecules with the
challenge train/test? Three keys: isomeric canonical SMILES, stereo-stripped
canonical, and InChIKey (skeleton layer).
"""
import sys

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "/home/jackson/cyp-challenge/src")

DATA = "/home/jackson/cyp-challenge/data/"
EXP = "/home/jackson/cyp-challenge/external/user_supplied/expansion_data_train.csv"
ROT = "/home/jackson/cyp-challenge/external/user_supplied/rotational_barrier_xinli.csv"


def keys(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None, None
    k_iso = Chem.MolToSmiles(m)
    m2 = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(m2)
    k_st = Chem.MolToSmiles(m2)
    try:
        from rdkit.Chem.inchi import MolToInchi
        from rdkit.Chem.rdInchi import InchiToInchiKey
        k_ik = InchiToInchiKey(MolToInchi(m2))
    except Exception:
        k_ik = None
    return k_iso, k_st, k_ik


def build(df, col):
    out = df[col].dropna().unique()
    rows = [keys(s) for s in out]
    return pd.DataFrame(rows, columns=["k_iso", "k_st", "k_ik"]).dropna(subset=["k_iso"])


def main():
    tr = pd.read_csv(DATA + "cyp-challenge-TRAIN_Emax.csv")
    te = pd.read_csv(DATA + "cyp-challenge-TEST-BLINDED.csv")
    allch = pd.concat([tr["SMILES"], te["SMILES"]])
    ck = pd.DataFrame([keys(s) for s in allch.dropna().unique()],
                      columns=["k_iso", "k_st", "k_ik"]).dropna(subset=["k_iso"])
    print("challenge molecules:", len(ck))

    exp = pd.read_csv(EXP)
    ec = [c for c in exp.columns if "smiles" in c.lower()][0]
    rot = pd.read_csv(ROT, encoding="utf-8-sig")
    rc = [c for c in rot.columns if "smiles" in c.lower() or "canonical" in c.lower()][0]

    for name, df, col in [("expansion", exp, ec), ("rotational", rot, rc)]:
        d = build(df, col)
        for key, label in [("k_iso", "isomeric"), ("k_st", "stereo-free"),
                           ("k_ik", "inchikey")]:
            if key == "k_ik" and d[key].isna().all():
                continue
            inter = set(d[key].dropna()) & set(ck[key].dropna())
            print(f"{name:11s} {label:11s} n={len(d):6d} overlap={len(inter)}")
        print()


if __name__ == "__main__":
    main()
