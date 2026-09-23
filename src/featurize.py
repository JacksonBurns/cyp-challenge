"""Featurize all train + test compounds; cache to parquet.

Features per compound:
  - RDKit descriptor list (2D)
  - Morgan count fingerprints r2 (2048) and r3 (1024), folded
  - MACCS keys
Targets assembled in train.py; this file only produces X matrices keyed by SMILES.
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
os.makedirs(CACHE, exist_ok=True)

MORGAN2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=False)
MORGAN3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)


def mol_of(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            if m is not None:
                try:
                    m.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(
                        m,
                        Chem.SanitizeFlags.SANITIZE_ALL
                        ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                    )
                except Exception:
                    return None
        return m
    except Exception:
        return None


_desc_names = [n for n, f in Descriptors._descList]


def features_for_mol(m):
    vals = []
    for n, f in Descriptors._descList:
        try:
            v = f(m)
            if not np.isfinite(v):
                v = 0.0
        except Exception:
            v = 0.0
        vals.append(float(v))
    return vals


def featurize_smiles_list(smiles_list):
    X = []
    names = []
    keep = []
    desc_vecs = []
    fps2 = []
    fps3 = []
    maccs = []
    for smi in smiles_list:
        m = mol_of(smi)
        if m is None:
            keep.append(False)
            continue
        keep.append(True)
        names.append(smi)
        d = np.array(features_for_mol(m), dtype=np.float64)
        fp2 = np.zeros(2048, dtype=np.float32)
        fp3 = np.zeros(1024, dtype=np.float32)
        a2 = MORGAN2.GetCountFingerprintAsNumPy(m).astype(np.float32)
        # downweight high counts (sqrt) - helps GBMs
        a2 = np.sqrt(a2)
        a3 = np.sqrt(MORGAN3.GetCountFingerprintAsNumPy(m).astype(np.float32))
        bv = MACCSkeys.GenMACCSKeys(m)
        mk = np.zeros(167, dtype=np.float32)
        mk[list(bv.GetOnBits())] = 1.0
        desc_vecs.append(d)
        fps2.append(a2)
        fps3.append(a3)
        maccs.append(mk)
    X = np.hstack([np.array(desc_vecs), np.array(fps2), np.array(fps3), np.array(maccs)])
    feat_names = _desc_names + [f"m2_{i}" for i in range(2048)] + [f"m3_{i}" for i in range(1024)] + [f"mk_{i}" for i in range(167)]
    return X, names, np.array(keep), feat_names


def main():
    train = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))

    all_train = pd.concat(
        [
            train[["Molecule_Name", "SMILES"]],
            tdi[["Molecule_Name", "SMILES"]],
        ]
    ).drop_duplicates("SMILES")
    names = all_train["SMILES"].tolist()
    X, names_out, keep, feat_names = featurize_smiles_list(names)
    assert (keep == True).all(), "some train SMILES failed"
    Xtr = pd.DataFrame(X, columns=feat_names)
    Xtr["SMILES"] = names_out
    Xtr.to_parquet(os.path.join(CACHE, "X_train.parquet"))
    print("train X:", Xtr.shape)

    Xt, names_t, keep_t, _ = featurize_smiles_list(test["SMILES"].tolist())
    print("test kept:", keep_t.sum(), "/", len(test))
    Xte = pd.DataFrame(Xt, columns=feat_names)
    Xte["SMILES"] = names_t
    Xte.to_parquet(os.path.join(CACHE, "X_test.parquet"))
    with open(os.path.join(CACHE, "feat_names.pkl"), "wb") as fh:
        pickle.dump(feat_names, fh)
    print("test X:", Xte.shape)


if __name__ == "__main__":
    main()
