"""External-data ranking features (ChEMBL CYP IC50 + PubChem AID1851 qHTS).

For each query compound (challenge train rows and test rows) and each isoform:
  nearest-neighbor features against EXTERNAL labeled compounds only:
    ext_top1_sim, ext_wavg_act (sim^2-weighted activity), ext_top10_act,
    ext_count_sim>0.7, ext_max_act_top5, ext_bin_rate_top5 (binary actives).
External assays are NOT DRC-calibrated: values are used only for RANKING
(GBM features), never for placement moments.

Activity scales:
  ChEMBL: standard_value in nM with standard_relation '=' (also '<' kept, flagged
  inactive-side censored? -> dropped to keep features honest), pIC50 = 9 - log10(nM).
  PubChem: Fit_LogAC50-derived pIC50_like columns; binary Active/Inactive columns.

Outputs: cache/ext_feats_{train,test}.parquet (SMILES + {iso}_{featname} cols)
Run: cyp env.
"""
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fps_for(smiles):
    arr, keep = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        if m is None:
            continue
        arr.append(np.asarray(GEN.GetFingerprintAsNumPy(m), dtype=np.float32))
        keep.append(i)
    return np.vstack(arr), np.array(keep)


def load_chembl():
    df = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    df = df[df["standard_relation"].isin(["=", "<", ">"])].copy()
    df = df[df["standard_relation"] == "="]  # drop censored rows for honest features
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    units = df["standard_units"].astype(str).str.strip().str.lower()
    df = df[units.isin(["nm"])]  # keep nM only
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df["pic50"] = 9.0 - np.log10(df["standard_value"])
    g = df.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    return g


def load_pubchem():
    df = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    frames = []
    for iso in ISO:
        pc, bc = f"{iso}_pIC50", f"{iso}_active"
        if pc not in df.columns:
            continue
        sub = df[["smiles", pc, bc]].dropna(subset=[pc]).copy()
        sub["isoform"] = iso
        sub = sub.rename(columns={pc: "pic50", bc: "active"})
        frames.append(sub[["isoform", "smiles", "pic50", "active"]])
    return pd.concat(frames, ignore_index=True)


def nn_features(qsmi, lab_df, iso):
    """qsmi: query smiles list; lab_df: external labeled (iso subset) smiles/pic50/active."""
    d = lab_df[lab_df["isoform"] == iso].dropna(subset=["pic50", "smiles"])
    d = d.drop_duplicates("smiles")
    Efp, k1 = fps_for(d["smiles"].tolist())
    Qfp, k2 = fps_for(qsmi)
    A = Qfp  # binary-ish counts; binarize both sides
    A = (A > 0).astype(np.float32)
    B = (Efp > 0).astype(np.float32)
    dmat = A @ B.T
    ca = A.sum(1)[:, None]
    cb = B.sum(1)[None, :]
    S = dmat / np.maximum(ca + cb - dmat, 1e-9)
    y = d["pic50"].values.astype(np.float64)
    act = d.get("active")
    yact = None if act is None else pd.to_numeric(act, errors="coerce").fillna(0).values.astype(float)
    n = S.shape[1]
    k10 = min(10, n)
    part = np.argpartition(-S, k10 - 1, axis=1)[:, :k10]
    rows = np.arange(S.shape[0])[:, None]
    topS = S[rows, part]
    topy = y[part]
    w = np.clip(topS, 0, None) ** 2
    wavg = (w * np.nan_to_num(topy)).sum(1) / np.maximum(w.sum(1), 1e-9)
    top5 = topy[:, :5]
    out = {
        f"{iso}_ext_top1sim": topS[:, 0],
        f"{iso}_ext_wavg": wavg,
        f"{iso}_ext_top10mean": np.nanmean(topy, axis=1),
        f"{iso}_ext_count7": (S > 0.7).sum(1).astype(np.float32),
        f"{iso}_ext_maxtop5": np.nanmax(top5, axis=1),
    }
    if yact is not None:
        topa = yact[part[:, :5]]
        out[f"{iso}_ext_binrate5"] = np.nanmean(topa, axis=1)
    return pd.DataFrame(out, index=range(len(qsmi)))


def main():
    chembl = load_chembl().rename(columns={"canonical_smiles": "smiles"})
    pub = load_pubchem()
    # merge: prefer ChEMBL median per smiles; PubChem adds pIC50 where absent + binary flags
    frames = []
    for iso in ISO:
        c = chembl[chembl["isoform"] == iso][["smiles", "pic50"]]
        p = pub[pub["isoform"] == iso].drop_duplicates("smiles").set_index("smiles")
        m = c.drop_duplicates("smiles").set_index("smiles")
        idx = m.index.union(p.index)
        pic = m["pic50"].reindex(idx)
        pic = pic.fillna(pd.Series(p["pic50"], index=idx).reindex(idx))
        active = pd.Series(p["active"], index=idx).reindex(idx) if len(p) else pd.Series(np.nan, index=idx)
        frames.append(pd.DataFrame({"isoform": iso, "smiles": idx, "pic50": pic.values, "active": active.values}))
    lab_all = pd.concat(frames, ignore_index=True).dropna(subset=["pic50"])
    print("external labeled rows per iso:", lab_all["isoform"].value_counts().to_dict(), flush=True)

    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"), columns=["SMILES"]).drop_duplicates("SMILES")
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"), columns=["SMILES"])
    for tag, X in (("train", Xtr), ("test", Xte)):
        cols = [nn_features(X["SMILES"].tolist(), lab_all, iso) for iso in ISO]
        f = pd.concat(cols, axis=1)
        f.insert(0, "SMILES", X["SMILES"].values)
        f.to_parquet(os.path.join(CACHE, f"ext_feats_{tag}.parquet"))
        print(tag, f.shape, flush=True)


if __name__ == "__main__":
    main()
