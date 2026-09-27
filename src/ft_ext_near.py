"""ft_ext variant whose external pool = rows NEAREST TO THE BLIND TEST (step 2a, stir_bar lever).

stir_bar's write-up (rank 28 reg / 6 TDI): retrieve the closest public
neighbours of the held-out compounds so every backbone sees the neighbourhood
it will be scored in. Our adaptation imports no external labels into the
scored heads; it only re-slices which unlabeled-partial external rows join
training: cache/ext_near_test_top.parquet (top-9000 Tanimoto to any test
compound, built by src/ext_neighbors.py) instead of a random 9000 from the
whole external pool. Everything else = src/ft_ext.py --full-ft.

Output tags ft_oof_extnear.csv + ft_test_preds_extnear.csv - IMPORTANT: does
NOT match the ft_oof_ext*. glob? ft_oof_extnear DOES match ft_oof_ext*! To
keep the incumbent ext family untouched we write prefix 'near'.

Run (chemeleon env):
  ~/miniforge3/envs/chemeleon/bin/python src/ft_ext_near.py --full-ft --seed 12
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_ext as F  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache")


def main(full_ft, seed):
    F.EXT_CAP = 10 ** 9  # our pre-screened pool is already capped
    ft, test, allrows = F.load_tables()
    near = pd.read_parquet(os.path.join(CACHE, "ext_near_test_top.parquet"))
    near_set = set(near["smiles"])
    ch_mask = (allrows["_src"] == "challenge").values
    keep_mask = allrows["SMILES"].isin(near_set).values & ~ch_mask
    lab = allrows[F.COLS].notna().any(axis=1).values
    use_ext = keep_mask & lab
    print("external near-test rows w/ any label:", int(use_ext.sum()), flush=True)
    oof = pd.DataFrame(np.nan, index=ft.index, columns=F.ISO)
    folds = ft["fold"].values
    t0 = time.time()
    for f in range(5):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        tr_extra = allrows[use_ext & ~ch_mask]
        rng = np.random.default_rng(1000 * seed + f)
        lab_tr = tr_pool[F.COLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * F.VAL_FRAC), replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
        model = F.train_model(fit_df, tr_pool.iloc[vi], full_ft, f"fold{f}", seed=seed)
        P = F.predict(model, va_df)
        for j, iso in enumerate(F.ISO):
            oof.loc[va_idx, iso] = P[:, j]
        del model
        torch.cuda.empty_cache()
        print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, "ft_oof_near.csv"), index=False)

    tr_pool = allrows[ch_mask].reset_index(drop=True)
    rng = np.random.default_rng(99 + seed)
    l = tr_pool[F.COLS].notna().any(axis=1).values
    vi = rng.choice(np.where(l)[0], size=int(l.sum() * F.VAL_FRAC), replace=False)
    ext = allrows[use_ext]
    fit_df = pd.concat([tr_pool.drop(index=vi), ext], ignore_index=True)
    model = F.train_model(fit_df, tr_pool.iloc[vi], full_ft, "all", seed=seed)
    P = F.predict(model, test)
    tdf = pd.DataFrame(P[:, :4], columns=F.ISO)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, "ft_test_preds_near.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--seed", type=int, default=12)
    a = ap.parse_args()
    main(a.full_ft, a.seed)
