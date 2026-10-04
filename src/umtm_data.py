"""UMTM master table: one row per unique SMILES across ALL competition sources.

22 NaN-masked heads (spec table sums to 22; plan's "~23" was approximate):
  direct  x4  {iso}_direct    challenge direct pIC50 (inhibition combine_first TDI)
  tdic    x4  {iso}_tdic      challenge TDI-condition pIC50
  is_TDI  x2  is_TDI_{2D6,3A4} binary (TDI union Emax, run_tdi.py convention)
  log2fc  x4  {iso}_log2fc    single-conc median(log2fc_estimate) pivot
  chembl  x4  {iso}_chembl    external ChEMBL CYP pIC50 ('=' relation, nM, median)
  qhts    x4  {iso}_pubchem   external PubChem AID1851 Fit_LogAC50 pIC50-like

Row kinds, in this exact order (subsample determinism depends on it):
  challenge: X_train order (= cache/ft_data.csv order, 6145 rows; carries BOTH
             fold vectors; regression fold_reg seed 0 matches ft_data fold col -
             the fold the ENTIRE legacy blend/ES machinery runs on; fold_tdi
             seed 7 matches the TDI track scripts)
  ext:       external ChEMBL+qHTS merged (one row per SMILES, union of the two
             sources' head columns), sorted SMILES, minus challenge+test collisions
  sc_only:   single-conc SMILES absent from X_train (aux view only, never val)
Test rows live separately in cache/umtm_test_smiles.csv (750).

Writes cache/umtm_master.csv. Run in the CYP ENV (numpy 1.26.4), NOT chemeleon:
fold tie-breaks are numpy-version-dependent (see cypfolds GOTCHA) and only cyp
reproduces the legacy ft_data.csv fold vector byte-for-byte.
  ~/miniforge3/envs/cyp/bin/python src/umtm_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
EXT = os.path.join(ROOT, "external")

sys.path.insert(0, HERE)
from cypfolds import make_folds, scaffold_groups  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

DIRECT = [f"{i}_direct" for i in ISO]
TDIC = [f"{i}_tdic" for i in ISO]
TDIB = [f"is_TDI_{i[3:]}" for i in ["CYP2D6", "CYP3A4"]]
SC = [f"{i}_log2fc" for i in ISO]
CHM = [f"{i}_chembl" for i in ISO]
PUB = [f"{i}_pubchem" for i in ISO]
HEADS = DIRECT + TDIC + TDIB + SC + CHM + PUB

# group -> (head list, loss, total group weight) per FINAL_PUSH_PLAN 1.2
def head_groups(aux_scale=1.0):
    return {
        "direct": (DIRECT, "mse", 1.0),
        "tdic": (TDIC, "mse", 0.5 * aux_scale),
        "tdib": (TDIB, "bce", 0.5 * aux_scale),
        "sc": (SC, "mse", 0.3 * aux_scale),
        "chembl": (CHM, "mse", 0.3 * aux_scale),
        "qhts": (PUB, "mse", 0.3 * aux_scale),
    }


def per_task_weights(df_fit, aux_scale=1.0):
    """Task_weights in HEADS order: direct = inverse-count mean-normalized to 1.0
    (ft_ext recipe), other groups = group_w / n_heads_in_group. NOTE tdib has 2
    heads (0.25 each), every other group 4."""
    counts = df_fit[HEADS].notna().sum().values.astype(float)
    prim = 1.0 / np.maximum(counts[:4], 1)
    prim = prim / prim.mean()
    w = [prim]
    for gname, gws, nh in [("tdic", 0.5, 4), ("tdib", 0.5, 2), ("sc", 0.3, 4),
                           ("chembl", 0.3, 4), ("qhts", 0.3, 4)]:
        w.append(np.full(nh, gws * aux_scale / nh))
    w = np.concatenate(w)
    assert len(w) == len(HEADS), f"weight vector {len(w)} != HEADS {len(HEADS)}"
    return w


def build_master():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    assert (Xtr["SMILES"].values == ft["SMILES"].values).all(), "X_train order != ft_data order"
    inh = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv")).drop_duplicates("SMILES")
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES")
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv")).drop_duplicates("SMILES")
    sc = pd.read_csv(os.path.join(DATA, "cyp-challenge-single-concentration-TRAIN.csv"))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv")).drop_duplicates("SMILES")
    assert len(tdi) == 6145 and len(inh) == 4905, (len(tdi), len(inh))

    base = pd.DataFrame({"SMILES": Xtr["SMILES"].values})
    inh_i, tdi_i = inh.set_index("SMILES"), tdi.set_index("SMILES")
    # GOTCHA: reindex(base.SMILES) yields a SMILES-indexed Series; assigning that
    # to base (integer index) aligns on labels -> all NaN. Always .values.
    for iso in ISO:
        d1 = inh_i[f"{iso}_pIC50_direct_inhibition"].reindex(base.SMILES)
        d2 = tdi_i[f"{iso}_pIC50_direct_inhibition"].reindex(base.SMILES)
        base[f"{iso}_direct"] = d1.combine_first(d2).values
        base[f"{iso}_tdic"] = tdi_i[f"{iso}_pIC50_TDI_condition"].reindex(base.SMILES).values
    # is_TDI: union of TDI + Emax (run_tdi.py), scored isoforms only
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ["CYP2D6", "CYP3A4"]]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ["CYP2D6", "CYP3A4"]]]]).drop_duplicates("SMILES")
    lab = lab.set_index("SMILES")
    for i in ["CYP2D6", "CYP3A4"]:
        v = lab[f"{i}_is_TDI"].reindex(base.SMILES)
        # HEADS/TDIB use is_TDI_2D6 (i[3:]); assigning is_TDI_CYP2D6 would silently
        # leave the real head column all-NaN via the missing-column filler.
        base[f"is_TDI_{i[3:]}"] = v.map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0}).values
        assert base[f"is_TDI_{i[3:]}"].notna().sum() > 0, f"{i} is_TDI all-NaN after assign"
    base["kind"] = "challenge"

    # fold vectors: reg (seed 0, legacy blend machinery) + tdi (seed 7, plan spec)
    groups = scaffold_groups(base["SMILES"].tolist())
    fold_reg = make_folds(groups, seed=0)
    assert (fold_reg == ft["fold"].values).all(), "fold_reg != ft_data fold (seed 0 broken?)"
    base["fold_reg"] = fold_reg
    base["fold_tdi"] = make_folds(groups, seed=7)

    # sc pivot: median of replicate log2fc_estimate per (SMILES, enzyme)
    scp = sc.groupby(["SMILES", "enzyme"])["log2fc_estimate"].median().unstack("enzyme")
    scp = scp.rename(columns={i: f"{i}_log2fc" for i in ISO})
    for c in SC:
        base[c] = scp[c].reindex(base.SMILES).values if c in scp.columns else np.nan

    ch_set = set(base.SMILES) | set(test.SMILES)
    # ChEMBL: '=' relation, nM units, median pIC50 per (isoform, smiles); sorted SMILES order
    ch = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    ch = ch[ch["standard_relation"] == "="]
    ch["standard_value"] = pd.to_numeric(ch["standard_value"], errors="coerce")
    u = ch["standard_units"].astype(str).str.strip().str.lower()
    ch = ch[u.isin(["nm"])].dropna(subset=["standard_value", "canonical_smiles"])
    ch = ch[ch["standard_value"] > 0]
    ch["pic50"] = 9.0 - np.log10(ch["standard_value"])
    g = ch.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    wide = g.pivot(index="canonical_smiles", columns="isoform", values="pic50")
    chdf = pd.DataFrame({f"{i}_chembl": wide[i] if i in wide.columns else np.nan for i in ISO})
    chdf["SMILES"] = chdf.index
    chdf = chdf[~chdf.SMILES.isin(ch_set)].sort_values("SMILES").reset_index(drop=True)
    chdf["kind"] = "chembl"
    # qHTS AID1851 (ft_ext recipe: first per SMILES), sorted SMILES order
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pubdf = pd.DataFrame({f"{i}_pubchem": pd.to_numeric(pc.get(f"{i}_pIC50"), errors="coerce") for i in ISO})
    pubdf["SMILES"] = pc["smiles"]
    pubdf = pubdf.dropna(subset=[c for c in pubdf.columns if c != "SMILES"], how="all")
    pubdf = pubdf.groupby("SMILES").first().reset_index()
    pubdf = pubdf[~pubdf.SMILES.isin(ch_set)].sort_values("SMILES").reset_index(drop=True)
    pubdf["kind"] = "qhts"
    # sc-only extras
    sc_only = scp[~scp.index.isin(set(base.SMILES))].reset_index()
    sc_only["kind"] = "sc_only"
    sc_only.columns = ["SMILES"] + [f"{i}_log2fc" for i in ISO] + ["kind"]

    # merge external sources into ONE frame (a SMILES can carry ChEMBL AND qHTS
    # labels -> same row, different head columns; union index, sorted)
    ext_idx = sorted(set(chdf.SMILES) | set(pubdf.SMILES))
    c_i, p_i = chdf.set_index("SMILES"), pubdf.set_index("SMILES")
    exdf = pd.DataFrame({"SMILES": ext_idx})
    for c in CHM:
        exdf[c] = c_i[c].reindex(ext_idx).values if c in c_i else np.nan
    for c in PUB:
        exdf[c] = p_i[c].reindex(ext_idx).values if c in p_i else np.nan
    exdf = exdf.dropna(subset=CHM + PUB, how="all")
    exdf["kind"] = "ext"
    # sc-only extras
    sc_only = scp[~scp.index.isin(set(base.SMILES))].reset_index()
    sc_only["kind"] = "sc_only"
    sc_only.columns = ["SMILES"] + [f"{i}_log2fc" for i in ISO] + ["kind"]

    blocks = [base, exdf, sc_only]
    master = pd.concat(blocks, ignore_index=True)
    for c in HEADS:
        if c not in master:
            master[c] = np.nan
    for c in ["fold_reg", "fold_tdi"]:
        master[c] = master[c].astype("float")
    master = master[["SMILES", "kind", "fold_reg", "fold_tdi"] + HEADS]
    # a SMILES must appear once
    assert master.SMILES.is_unique, "duplicate SMILES in master"
    return master, test


def main():
    master, test = build_master()
    master.to_csv(os.path.join(CACHE, "umtm_master.csv"), index=False)
    test[["SMILES"]].to_csv(os.path.join(CACHE, "umtm_test_smiles.csv"), index=False)
    ch = master[master.kind == "challenge"]
    print("master rows:", len(master), "| by kind:", master.kind.value_counts().to_dict())
    print("challenge labeled per head:", {c: int(ch[c].notna().sum()) for c in HEADS})
    print("tdi-only rows (no direct labels at all):",
          int((ch[DIRECT].isna().all(axis=1)).sum()))
    print("fold_reg dist:", np.bincount(ch.fold_reg.astype(int)), "fold_tdi dist:", np.bincount(ch.fold_tdi.astype(int)))


if __name__ == "__main__":
    main()
