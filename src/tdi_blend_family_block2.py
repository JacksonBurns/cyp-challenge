"""Family-block TDI blend audit v2 (step 1b/1c follow-up, sec 16).

Extends src/tdi_blend_family_block.py with the step-1b diversified members:
  tabcpchm  TabICL on frozen chemprop_chemeleon embeddings (tdi_tabicl_cp2)
  dmpnn     D-MPNN binary classifier heads, challenge rows (tdi_dmpnn)

Two FAMILY maps are reported for every pool:
  merged: cp = {tabcp, tabcpext, tabcpchm} (conservative: cpmed/cpchm share
          the frozen-chemprop featurization + adme_pretrain-style lineage,
          exactly the sec 13 root cause; use this for the GO gate)
  split:  cpmed = {tabcp, tabcpext}, cpchm = {tabcpchm} (the prompt's
          diversification claim: different pretraining corpus = new family)
Per pool (v2, v3, v4 = v3 + tabcpchm + dmpnn):
  fold_nested (old optimistic), fold_nested_capped (family cap 0.5),
  LOFO reselection per family, family weight concentration.
Pooled blends saved as cache/tdi_fbnest{,cap}_<pool>_<iso>.npy for the
fraction posterior. Writes cache/tdi_family_block_audit2.json.

Usage (cyp env): python src/tdi_blend_family_block2.py
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

ISO = ["CYP2D6", "CYP3A4"]
K = 5
FAMILY_CAP = 0.5

FAM_MERGED = {"base": "base", "emb": "emb", "tab": "emb",
              "tabcp": "cp", "tabcpext": "cp", "tabcpchm": "cp", "dmpnn": "dmpnn"}
FAM_SPLIT = {"base": "base", "emb": "emb", "tab": "emb",
             "tabcp": "cpmed", "tabcpext": "cpmed", "tabcpchm": "cpchm", "dmpnn": "dmpnn"}

POOLS = {
    "v2": ["base", "emb", "tab"],
    "v3": ["base", "emb", "tab", "tabcp", "tabcpext"],
    "v4": ["base", "emb", "tab", "tabcp", "tabcpext", "tabcpchm", "dmpnn"],
}

SRC = {"base": "tdi_oof.npz", "emb": "tdi_emb_oof.npz",
       "tab": "tdi_tabicl_oof.npz", "extbase": "tdi_oof_extbase_w03.npz",
       "tabcp": "tdi_tabicl_cp_oof.npz", "tabcpext": "tdi_tabicl_cp_ext_oof.npz",
       "tabcpchm": "tdi_tabicl_cpchm_oof.npz", "dmpnn": "tdi_dmpnn_oof.npz"}


def mcc(p, y, t):
    pr = (p >= t).astype(int)
    tp = ((pr == 1) & (y == 1)).sum(); fp = ((pr == 1) & (y == 0)).sum()
    tn = ((pr == 0) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
    return (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)


def best_by_frac(z, yy, lo=0.05, hi=0.61):
    n = len(yy)
    rows = []
    for f in np.arange(lo, hi, 0.01):
        k = int(n * f)
        t = np.sort(z)[::-1][k]
        rows.append((float(mcc(z, yy, t)), float(f)))
    return max(rows)


def greedy_capped(zs, y, fam, rounds=12, cap=None):
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    wsum = 0.0
    cur = None
    for _ in range(rounds):
        fsum = {}
        for k in keys:
            fsum[fam[k]] = fsum.get(fam[k], 0.0) + w[k]
        bestk, bestr = None, -2.0
        for k in keys:
            if cap is not None and wsum > 0 and (fsum[fam[k]] + 1.0) / (wsum + 1) > cap:
                continue
            z = (wsum * cur + zs[k]) / (wsum + 1) if wsum else zs[k]
            r = best_by_frac(z, y)[0]
            if r > bestr:
                bestr, bestk = r, k
        if bestk is None:
            break
        w[bestk] += 1
        wsum += 1
        cur = (wsum * cur + zs[bestk]) / wsum if cur is not None else zs[bestk]
    return {k: v / wsum for k, v in w.items() if v > 0}


def nested_blend(zs, y, folds, names, fam, cap=None):
    pooled = np.zeros(len(y))
    w_hist = []
    for f in range(K):
        trn = folds != f
        va = folds == f
        w = greedy_capped({k: zs[k][trn] for k in names}, y[trn], fam, cap=cap)
        w_hist.append(w)
        pooled[va] = sum(w.get(k, 0.0) * zs[k][va] for k in names)
    return pooled, w_hist


def fam_share(w_list, FAMILIES):
    per = []
    for w in w_list:
        s = {}
        for k, v in w.items():
            s[FAMILIES[k]] = s.get(FAMILIES[k], 0.0) + v
        per.append(s)
    fams = sorted({f for s in per for f in s})
    return {f: {"mean": round(float(np.mean([s.get(f, 0.0) for s in per])), 3),
                "max": round(float(np.max([s.get(f, 0.0) for s in per])), 3)} for f in fams}


def audit_pool(zs, yl, fi, names, fam, save_prefix):
    res = {"members": names}
    pooled, wl = nested_blend(zs, yl, fi, names, fam, cap=None)
    mcc_n, frac_n = best_by_frac(pooled, yl)
    res["fold_nested"] = [round(mcc_n, 4), round(frac_n, 2)]
    res["fold_nested_family_weights"] = [{k: round(v, 2) for k, v in w.items()} for w in wl]
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
        mcc_f, frac_f = best_by_frac(pooled_f, yl)
        lofo[f"-{f_}"] = [round(mcc_f, 4), round(frac_f, 2)]
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
            out[iso][pool] = {
                "merged": audit_pool(zs, yl, fi, names, FAM_MERGED, f"tdi_fb2_{pool}_merged_{iso}"),
                "split": audit_pool(zs, yl, fi, names, FAM_SPLIT, f"tdi_fb2_{pool}_split_{iso}"),
            }
            mm = out[iso][pool]["merged"]
            print(f"=== {iso} / {pool} (merged families)", flush=True)
            print(json.dumps({k: mm[k] for k in ["fold_nested", "fold_nested_capped",
                                                 "family_gain", "family_share_free"]}, indent=1), flush=True)

    macro = {}
    for pool in POOLS:
        macro[pool] = {
            "fold_nested_macro": round(float(np.mean([out[i][pool]["merged"]["fold_nested"][0] for i in ISO])), 4),
            "capped_macro": round(float(np.mean([out[i][pool]["merged"]["fold_nested_capped"][0] for i in ISO])), 4),
        }
    out["_macro"] = macro
    print("MACRO:", json.dumps(macro), flush=True)
    with open(os.path.join(CACHE, "tdi_family_block_audit2.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
