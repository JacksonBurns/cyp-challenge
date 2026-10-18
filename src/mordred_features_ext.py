"""Phase 1 support (NEXT_ITER_PLAN sec 4) - extend the mordred matrix to the
EXTERNAL aux molecules so V4 (aux + x_d) can feed a consistent x_d channel to
every training row (challenge + external).

Anti-leak (plan sec 9): descriptor imputation + standardization stats are fit on
the CHALLENGE TRAIN set ONLY (already done in src/mordred_features.py ->
cache/mordred_meta.json). The external rows are imputed with the TRAIN column
median and z-scored with the TRAIN mean/std - NEVER fit on the external set.
The STABLE descriptor set is re-derived on the SAME 60 challenge-train SMILES
mordred_features.py used, so the columns are byte-identical to
cache/emb_mordred_all.parquet.

Output: cache/emb_mordred_ext.parquet  index=SMILES, external-only rows (22983),
cols = the same 1429 kept descriptors, imputed + standardized with TRAIN stats.

Run (mordred env): ~/miniforge3/envs/mordred/bin/python src/mordred_features_ext.py
"""
import json
import os
import time

import mordred
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
import mordred.descriptors as _dm

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")
OUT_META = os.path.join(CACHE, "mordred_meta.json")
OUT_PARQUET = os.path.join(CACHE, "emb_mordred_ext.parquet")
SAMPLE_N = 60
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def load_ext_smiles():
    """Replicate ft_ext.load_tables external-row selection WITHOUT torch (so this
    runs in the mordred env). Returns the external (non-challenge) labeled
    SMILES, same rows ft_ext/ft_dmpnn train on for V4."""
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    ch_smiles = set(ft["SMILES"]) | set(test["SMILES"])
    rows = []
    ch = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    ch = ch[ch["standard_relation"] == "="]
    ch["standard_value"] = pd.to_numeric(ch["standard_value"], errors="coerce")
    u = ch["standard_units"].astype(str).str.strip().str.lower()
    ch = ch[u.isin(["nm"])].dropna(subset=["standard_value", "canonical_smiles"])
    ch = ch[ch["standard_value"] > 0]
    ch["pic50"] = 9.0 - np.log10(ch["standard_value"])
    g = ch.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    wide = g.pivot(index="canonical_smiles", columns="isoform", values="pic50")
    cdf = pd.DataFrame({f"{i}_chembl": wide[i] if i in wide else np.nan for i in ISO})
    cdf["SMILES"] = cdf.index
    rows.append(cdf)
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pdf = pd.DataFrame({f"{i}_pubchem": pd.to_numeric(pc.get(f"{i}_pIC50"), errors="coerce")
                        for i in ISO})
    pdf["SMILES"] = pc["smiles"]
    pdf = pdf.dropna(subset=[c for c in pdf.columns if c != "SMILES"], how="all")
    pdf = pdf.groupby("SMILES").first().reset_index()
    rows.append(pdf)
    allrows = pd.concat(rows, ignore_index=True)
    allrows = allrows[~allrows["SMILES"].isin(ch_smiles)]
    allrows = allrows.drop_duplicates("SMILES").reset_index(drop=True)
    return allrows["SMILES"].tolist()


def mol_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        m = Chem.MolFromSmiles(smi, sanitize=False)
    return m


def build_full_calc():
    descs = list(mordred.get_descriptors_in_module(_dm))
    calc = mordred.Calculator(ignore_3D=True)
    for d in descs:
        try:
            calc.register(d, ignore_3D=True)
        except Exception:
            pass
    return calc


def main():
    t0 = time.time()
    meta = json.load(open(OUT_META))
    kept_names = meta["kept_descriptors"]
    train_med = np.array([meta["imputation_train_median"][n] for n in kept_names])
    train_mean = np.array([meta["standardization_train_mean"][n] for n in kept_names])
    train_std = np.array([meta["standardization_train_std"][n] for n in kept_names])
    print(f"reusing train-fit stats for {len(kept_names)} kept descriptors", flush=True)

    # external SMILES (torch-free loader, same rows as ft_ext/ft_dmpnn V4 training)
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    ext_smis = load_ext_smiles()
    print(f"external aux SMILES: {len(ext_smis)} (unique {len(set(ext_smis))})", flush=True)

    mols = [mol_of(s) for s in ext_smis]
    n_fail = sum(m is None for m in mols)
    print(f"parsed {len(mols)-n_fail}/{len(mols)} external SMILES", flush=True)

    calc_full = build_full_calc()
    insts = list(calc_full.descriptors)
    # re-derive stable set on the SAME 60 challenge-train SMILES the main script used
    ft_smis = ft["SMILES"].tolist()
    sample_mols = [mol_of(s) for s in ft_smis[:SAMPLE_N]]
    samp_df = calc_full.pandas(sample_mols, nproc=1, quiet=True)
    colnames = list(samp_df.columns)
    broken = {c for c in colnames if all(mordred.is_missing(v) for v in samp_df[c].to_numpy())}
    stable_names = [c for c in colnames if c not in broken]
    assert set(kept_names) <= set(stable_names), "kept set not a subset of re-derived stable set"
    # stable-only calc (skip the 120 numpy-1.26-broken descriptors for speed)
    stable_set = set(stable_names)
    calc_stable = mordred.Calculator(ignore_3D=True)
    n_reg = 0
    for inst, cn in zip(insts, colnames):
        if cn in stable_set:
            calc_stable.register(inst)
            n_reg += 1
    print(f"stable-only calc: {n_reg} descriptors", flush=True)

    t1 = time.time()
    df = calc_stable.pandas(mols, nproc=8, quiet=True)
    if n_fail:
        for i, m in enumerate(mols):
            if m is None:
                df.iloc[i] = np.nan
    # select the 1429 kept columns (by name), convert Error->NaN
    X = np.empty((len(ext_smis), len(kept_names)), dtype=np.float64)
    for j, c in enumerate(kept_names):
        vals = df[c].to_numpy()
        for i in range(len(vals)):
            v = vals[i]
            X[i, j] = float(v) if not mordred.is_missing(v) else np.nan
    print(f"external descriptor matrix {X.shape} in {time.time()-t1:.0f}s", flush=True)

    # impute with TRAIN median, standardize with TRAIN mean/std (anti-leak)
    X_imp = np.where(np.isnan(X), train_med, X)
    X_std = (X_imp - train_mean) / np.where(train_std < 1e-12, 1.0, train_std)

    df_out = pd.DataFrame(X_std, index=pd.Index(ext_smis, name="SMILES"), columns=kept_names)
    assert np.isfinite(X_std).all(), "non-finite in external standardized matrix"
    # consistency check: overlap SMILES (none expected) - just verify shape/finite
    df_out.to_parquet(OUT_PARQUET)

    meta_ext = {
        "n_external": len(ext_smis),
        "n_unique_external": len(set(ext_smis)),
        "n_descriptors_kept": len(kept_names),
        "n_parse_fail": n_fail,
        "stats_source": "cache/mordred_meta.json TRAIN (challenge) median/mean/std - anti-leak",
        "column_consistency": "PASS (kept set subset of re-derived stable set on same 60 challenge-train SMILES)",
        "all_finite": bool(np.isfinite(X_std).all()),
        "compute_seconds": round(time.time() - t1, 1),
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(CACHE, "mordred_meta_ext.json"), "w") as fh:
        json.dump(meta_ext, fh, indent=2)
    print(f"alignment/consistency: {meta_ext['column_consistency']}", flush=True)
    print(f"saved {OUT_PARQUET} {df_out.shape}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
