"""Phase 0 (NEXT_ITER_PLAN sec 3) - build the mordred descriptor matrix.

Computes mordred 2D descriptors for ALL molecules (train 6145 from ft_data.csv
+ test 750 from ft_test_smiles.csv, the SAME SMILES the rest of the repo uses)
and applies the ANTI-LEAK NaN policy:
  - drop any descriptor column whose NaN fraction on TRAIN > 0.15
  - impute remaining NaNs with the TRAIN column median
  - standardize (z-score) using TRAIN mean/std ONLY
3D descriptors are EXCLUDED UP FRONT via ignore_3D=True (2D pipeline, no
conformers): a 2D mol with no conformer leaves every 3D descriptor undefined,
so we never compute them (1826 -> 1613 descriptors).

STABLE-SUBSET (plan-sanctioned curated set, logged in meta): mordred 1.2.0 has
~125 descriptors that raise `np.float` (removed in numpy >=1.24; the repo pins
numpy 1.26.4) and return all-NaN. We detect them with a small sample (a column
is "broken" iff all-missing on the sample) and register ONLY the stable
descriptors, so the full 6895-mol run is fast and free of error-string cells.
The dropped set is logged in mordred_meta.json.

mordred 1.2.0 API (differs from the older get()/only_available):
  descs = list(mordred.get_descriptors_in_module(mordred.descriptors))  # classes
  calc  = mordred.Calculator(ignore_3D=True)
  for d in descs: calc.register(d, ignore_3D=True)   # 1613 expanded descriptors
  df    = calc.pandas(mols, nproc=1, quiet=True)     # failures -> Error objects
  mordred.is_missing(v) detects Error/NaN cells.
  NOTE: nproc=1 (multiprocessing path corrupts at 6895 scale).

Outputs:
  cache/emb_mordred_all.parquet   index=SMILES (6895 rows), cols=kept descriptors
                                  (imputed + standardized), 1:1 with train/test.
  cache/mordred_meta.json         n descriptors, broken/kept/dropped lists,
                                  imputation + standardization stats, versions,
                                  alignment check.

Repo invariant verified: 1:1 row alignment on SMILES and SMILES-identical to
ft_data.csv / ft_test_smiles.csv.

Run (mordred env): ~/miniforge3/envs/mordred/bin/python src/mordred_features.py
"""
import json
import os
import time

import mordred
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
import mordred.descriptors as _dm

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
os.makedirs(CACHE, exist_ok=True)

NAN_DROP_FRAC = 0.15
SAMPLE_N = 60
OUT_PARQUET = os.path.join(CACHE, "emb_mordred_all.parquet")
OUT_META = os.path.join(CACHE, "mordred_meta.json")


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
    train = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    train_smis = train["SMILES"].tolist()
    test_smis = test["SMILES"].tolist()
    all_smis = train_smis + test_smis
    n_train, n_test = len(train_smis), len(test_smis)
    print(f"train {n_train} + test {n_test} = {len(all_smis)} SMILES", flush=True)

    mols = [mol_of(s) for s in all_smis]
    n_fail = sum(m is None for m in mols)
    print(f"parsed {len(mols)-n_fail}/{len(mols)} SMILES "
          f"({n_fail} failures -> all-NaN rows)", flush=True)

    # --- full calc -> instances + column names (order matches) ---
    calc_full = build_full_calc()
    insts = list(calc_full.descriptors)
    # get column names from a 1-mol pandas call (order == instance order)
    probe_mols = [m for m in mols[:1] if m is not None] or [mol_of("C")]
    probe_df = calc_full.pandas(probe_mols, nproc=1, quiet=True)
    colnames = list(probe_df.columns)
    assert len(colnames) == len(insts), "col/instance count mismatch"
    n_desc_full = len(insts)
    print(f"mordred {mordred.__version__}: {n_desc_full} 2D descriptors "
          f"(ignore_3D=True; 3D set excluded up front)", flush=True)

    # --- detect broken (numpy-1.26 / undefined) descriptors on a sample ---
    sample_mols = [m for m in mols[:SAMPLE_N] if m is not None]
    samp_df = calc_full.pandas(sample_mols, nproc=1, quiet=True)
    broken_idx = []
    for j, c in enumerate(colnames):
        vals = samp_df[c].to_numpy()
        if all(mordred.is_missing(v) for v in vals):
            broken_idx.append(j)
    broken_names = [colnames[j] for j in broken_idx]
    stable_idx = [j for j in range(n_desc_full) if j not in set(broken_idx)]
    stable_insts = [insts[j] for j in stable_idx]
    stable_names = [colnames[j] for j in stable_idx]
    print(f"stable subset: {len(stable_insts)} descriptors "
          f"({len(broken_names)} broken/undefined dropped on {len(sample_mols)}-mol sample)", flush=True)

    # --- register stable-only calc, run full set ---
    calc_stable = mordred.Calculator(ignore_3D=True)
    for inst in stable_insts:
        calc_stable.register(inst)
    t1 = time.time()
    df = calc_stable.pandas(mols, nproc=1, quiet=True)
    if n_fail:
        for i, m in enumerate(mols):
            if m is None:
                df.iloc[i] = np.nan
    X = np.empty((len(all_smis), len(stable_names)), dtype=np.float64)
    for j, col in enumerate(stable_names):
        vals = df[col].to_numpy()
        for i in range(len(vals)):
            v = vals[i]
            X[i, j] = float(v) if not mordred.is_missing(v) else np.nan
    print(f"descriptor matrix {X.shape} (stable) in {time.time()-t1:.0f}s", flush=True)

    # --- ANTI-LEAK NaN policy (fit on TRAIN rows only) ---
    Xtr = X[:n_train]
    nan_frac_train = np.nanmean(np.isnan(Xtr), axis=0)
    keep = nan_frac_train <= NAN_DROP_FRAC
    n_dropped = int((~keep).sum())
    dropped_names = [stable_names[j] for j in range(len(stable_names)) if not keep[j]]
    X_kept = X[:, keep]
    kept_names = [stable_names[j] for j in range(len(stable_names)) if keep[j]]

    med = np.nanmedian(X_kept[:n_train], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X_imp = np.where(np.isnan(X_kept), med, X_kept)
    mean = X_imp[:n_train].mean(axis=0)
    std = X_imp[:n_train].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_std = (X_imp - mean) / std

    # --- save (index = SMILES, mirrors monroe_emb.parquet) ---
    df_out = pd.DataFrame(X_std, index=pd.Index(all_smis, name="SMILES"), columns=kept_names)
    assert list(df_out.index) == all_smis, "index mismatch vs SMILES order"
    assert set(df_out.index) == set(train_smis) | set(test_smis), "index set != train U test"
    assert len(df_out) == n_train + n_test, "row count mismatch"
    assert df_out.index.is_unique, "duplicate SMILES in index"
    assert df_out.loc[train_smis, kept_names].index.tolist() == train_smis, "train alignment broken"
    assert df_out.loc[test_smis, kept_names].index.tolist() == test_smis, "test alignment broken"
    assert np.isfinite(X_std).all(), "non-finite value in standardized matrix"
    df_out.to_parquet(OUT_PARQUET)

    meta = {
        "mordred_version": mordred.__version__,
        "rdkit_version": rdkit.__version__,
        "numpy_version": np.__version__,
        "n_train": n_train,
        "n_test": n_test,
        "n_smiles_total": len(all_smis),
        "ignore_3D": True,
        "n_descriptors_2d_total": n_desc_full,
        "n_descriptors_stable": len(stable_names),
        "n_descriptors_broken_numpy126": len(broken_names),
        "broken_numpy126_descriptors": broken_names,
        "n_descriptors_kept": len(kept_names),
        "n_descriptors_dropped_nan_policy": n_dropped,
        "nan_drop_frac_threshold": NAN_DROP_FRAC,
        "stable_sample_n": len(sample_mols),
        "drop_rule": "stable subset (all-missing on sample removed) -> drop column if NaN fraction on TRAIN > 0.15 -> impute with TRAIN median -> z-score with TRAIN mean/std",
        "dropped_nan_policy_descriptors": dropped_names,
        "dropped_nan_policy_frac_train": {n: round(float(nan_frac_train[stable_names.index(n)]), 4) for n in dropped_names},
        "kept_descriptors": kept_names,
        "imputation_train_median": {n: round(float(v), 6) for n, v in zip(kept_names, med)},
        "standardization_train_mean": {n: round(float(v), 6) for n, v in zip(kept_names, mean)},
        "standardization_train_std": {n: round(float(v), 6) for n, v in zip(kept_names, std)},
        "alignment_check": "PASS (1:1 SMILES, SMILES-identical to ft_data.csv + ft_test_smiles.csv, unique index, all-finite)",
        "n_parse_fail": n_fail,
        "compute_seconds": round(time.time() - t1, 1),
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_META, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"kept {len(kept_names)} descriptors, "
          f"dropped {len(broken_names)} (numpy-1.26 broken) + {n_dropped} (train-NaN>0.15)", flush=True)
    print(f"alignment: {meta['alignment_check']}", flush=True)
    print(f"saved {OUT_PARQUET} {df_out.shape}", flush=True)
    print(f"saved {OUT_META}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
