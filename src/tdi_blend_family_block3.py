"""Family-block audit v3 (Task 1, sec 18 queue): + CheMeLeon fine-tuned on the
TDI binary task (src/tdi_ft_chemeleon.py) as a NEW family 'ftchm'.

Baseline gate: the v4_candidate machinery = v3 pool {base,emb,tab,tabcp,
tabcpext} with family cap 0.5 (capped macro nested 0.3279, sec 16). New pools:
  v5a  = v3 + ft_chm_fullft
  v5b  = v3 + ft_chm_frozen
  v5c  = v3 + both (both fine-tunes share the CheMeLeon lineage -> one family)
Three family maps:
  merged:  emb={emb,tab}, cp={tabcp,tabcpext}, ftchm separate (the prompt's
           claim: fine-tuning creates a new lineage)
  strict:  as merged but ft_chm_frozen joins emb (frozen CheMeLeon features are
           the emb/tab signal; conservative test for the frozen variant)
  paranoid: ftchm joins cp (worst case: one big CheMeLeon lineage)
Writes cache/tdi_family_block_audit3.json + pooled npys.

Usage (cyp env): python src/tdi_blend_family_block3.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

from run_regression import make_folds, scaffold_groups  # noqa: E402
from tdi_blend_family_block2 import best_by_frac, nested_blend, fam_share  # noqa: E402

ISO = ["CYP2D6", "CYP3A4"]
FAMILY_CAP = 0.5

SRC = {"base": "tdi_oof.npz", "emb": "tdi_emb_oof.npz",
       "tab": "tdi_tabicl_oof.npz", "tabcp": "tdi_tabicl_cp_oof.npz",
       "tabcpext": "tdi_tabicl_cp_ext_oof.npz",
       "ftchm": "tdi_ft_chm_oof.npz", "ftchmfz": "tdi_ft_chm_frozen_oof.npz"}

V3 = ["base", "emb", "tab", "tabcp", "tabcpext"]
POOLS = {"v3": V3, "v5a": V3 + ["ftchm"], "v5b": V3 + ["ftchmfz"],
         "v5c": V3 + ["ftchm", "ftchmfz"]}

FAM_MERGED = {"base": "base", "emb": "emb", "tab": "emb",
              "tabcp": "cp", "tabcpext": "cp", "ftchm": "ftchm", "ftchmfz": "ftchm"}
FAM_STRICT = {"base": "base", "emb": "emb", "tab": "emb",
              "tabcp": "cp", "tabcpext": "cp", "ftchm": "ftchm", "ftchmfz": "emb"}
FAM_PARANOID = {"base": "base", "emb": "emb", "tab": "emb",
                "tabcp": "cp", "tabcpext": "cp", "ftchm": "cp", "ftchmfz": "cp"}
MAPS = {"merged": FAM_MERGED, "strict": FAM_STRICT, "paranoid": FAM_PARANOID}


def audit_pool(zs, yl, fi, names, fam, save_prefix):
    res = {"members": names}
    pooled, wl = nested_blend(zs, yl, fi, names, fam, cap=None)
    mcc_n, frac_n = best_by_frac(pooled, yl)
    res["fold_nested"] = [round(mcc_n, 4), round(frac_n, 2)]
    res["family_share_free"] = fam_share(wl, fam)
    np.save(os.path.join(CACHE, f"{save_prefix}.npy"), pooled)
    pooled_c, wl_c = nested_blend(zs, yl, fi, names, fam, cap=FAMILY_CAP)
    mcc_c, frac_c = best_by_frac(pooled_c, yl)
    res["fold_nested_capped"] = [round(mcc_c, 4), round(frac_c, 2)]
    res["fold_nested_capped_weights"] = [{k: round(v, 2) for k, v in w.items()} for w in wl_c]
    res["family_share_capped"] = fam_share(wl_c, fam)
    np.save(os.path.join(CACHE, f"{save_prefix}cap.npy"), pooled_c)
    fams = sorted({fam[k] for k in names})
    lofo = {}
    for f_ in fams:
        rest = [k for k in names if fam[k] != f_]
        if not rest:
            continue
        pooled_f, _ = nested_blend({k: zs[k] for k in rest}, yl, fi, rest, fam, cap=None)
        mcc_f, _ = best_by_frac(pooled_f, yl)
        lofo[f"-{f_}"] = [round(mcc_f, 4)]
    res["lofo"] = lofo
    res["family_gain"] = {f_: round(mcc_n - lofo[f"-{f_}"][0], 4)
                          for f_ in fams if f"-{f_}" in lofo}
    return res


def main():
    members = {nm: np.load(os.path.join(CACHE, fn)) for nm, fn in SRC.items()
               if os.path.exists(os.path.join(CACHE, fn))}
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr["SMILES"])
    folds = make_folds(scaffold_groups(Xtr["SMILES"].tolist()), seed=7)

    out = {}
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        m = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[m]])
        fi = folds[m]
        zs = {}
        for nm, src in members.items():
            p = src[iso][m].astype(float)
            zs[nm] = (p - p.mean()) / p.std()
        out[iso] = {"singles": {k: round(best_by_frac(zs[k], yl)[0], 4) for k in zs}}
        for pool, names in POOLS.items():
            names = [k for k in names if k in zs]
            out[iso][pool] = {mn: audit_pool(zs, yl, fi, names, fam, f"tdi_fb3_{pool}_{mn}_{iso}")
                              for mn, fam in MAPS.items()}
            mm = out[iso][pool]["merged"]
            print(f"=== {iso} / {pool} (merged)", json.dumps(
                {k: mm[k] for k in ["fold_nested", "fold_nested_capped", "family_gain"]}), flush=True)

    macro = {}
    for pool in POOLS:
        for mn in MAPS:
            macro[f"{pool}_{mn}"] = {
                "fold_nested_macro": round(float(np.mean([out[i][pool][mn]["fold_nested"][0] for i in ISO])), 4),
                "capped_macro": round(float(np.mean([out[i][pool][mn]["fold_nested_capped"][0] for i in ISO])), 4),
            }
    out["_macro"] = macro
    print("MACRO:", json.dumps(macro, indent=1), flush=True)
    with open(os.path.join(CACHE, "tdi_family_block_audit3.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
