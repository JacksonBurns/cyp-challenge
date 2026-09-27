"""External rows near the BLIND TEST set (stir_bar's retrieval lever, adapted).

stir_bar (rank 28 reg / 6 TDI, public write-up) admits the closest public
neighbours of the held-out compounds so every backbone "sees the neighbourhood
it will be scored in"; briford's specialists = the same mechanism sliced by
isoform. Adapted to our pool, no labels imported: the D-MPNN external aux pool
is restricted to the top-K Tanimoto-nearest rows to any TEST compound, keeping
the 8 aux heads; the primary challenge heads stay challenge-only.

Writes cache/ext_near_test_top.parquet: SMILES + tanimoto (sorted desc).
Run (cyp env): python src/ext_neighbors.py
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")
RDLogger.DisableLog("rdApp.*")

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
K_TOP = 9000


def fps(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        out.append(GEN.GetFingerprintAsNumPy(m) if m is not None else None)
    return out


def main():
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"),
                     usecols=["smiles", "CYP1A2_pIC50", "CYP2C9_pIC50", "CYP2D6_pIC50", "CYP3A4_pIC50"])
    ch = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    ch = ch[ch["standard_relation"] == "="]
    ch["standard_value"] = pd.to_numeric(ch["standard_value"], errors="coerce")
    u = ch["standard_units"].astype(str).str.strip().str.lower()
    ch = ch[u.isin(["nm"])].dropna(subset=["standard_value", "canonical_smiles"])
    ch = ch[ch["standard_value"] > 0]
    ch = ch[["canonical_smiles"]].rename(columns={"canonical_smiles": "smiles"})
    ext = pd.concat([pc[["smiles"]], ch[["smiles"]]]).drop_duplicates("smiles").reset_index(drop=True)
    # exclude challenge train + test SMILES
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    banned = set(ft["SMILES"]) | set(test["SMILES"])
    ext = ext[~ext["smiles"].isin(banned)].reset_index(drop=True)
    print("external candidates:", len(ext), flush=True)

    tfp = np.array([f for f in fps(list(test["SMILES"])) if f is not None], dtype=np.float32)
    nb = tfp.sum(1)
    print("test fps", tfp.shape, flush=True)
    best = np.zeros(len(ext), dtype=np.float32)
    B = 1024
    for i in range(0, len(ext), B):
        chunk = fps(list(ext["smiles"].iloc[i:i + B]))
        rows = [np.asarray(f, dtype=np.float32) for f in chunk if f is not None]
        if not rows:
            continue
        A = np.vstack(rows)
        na = A.sum(1)[:, None]
        inter = A @ tfp.T
        t = inter / np.maximum(na + nb[None, :] - inter, 1e-6)
        rows_idx = [j for j, f in enumerate(chunk) if f is not None]
        mx = t.max(1)
        for r, j in enumerate(rows_idx):
            best[i + j] = mx[r]
        print(f"  {i}/{len(ext)} max-so-far {best[:i + len(rows)].max():.2f}", flush=True)
    ext["tanimoto"] = best
    ext = ext.sort_values("tanimoto", ascending=False).head(K_TOP).reset_index(drop=True)
    ext.to_parquet(os.path.join(CACHE, "ext_near_test_top.parquet"))
    print("wrote top", len(ext), "nearest; tanimoto min", float(ext['tanimoto'].min()),
          "p50", float(ext['tanimoto'].median()), flush=True)


if __name__ == "__main__":
    main()
