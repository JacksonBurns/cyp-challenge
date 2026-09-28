"""Uni-Mol 3D-conformer encoder embeddings for all train+test SMILES (Task 3).

GO gate passed Sep 28: dptech/Uni-Mol-Models weights download fine on this box
(190 MB in ~7 s), unimol-tools 0.1.6 from PyPI in the dedicated `unimol` conda
env (never into cyp). Uni-Mol v1 84m, remove_hs=False (all-hydrogens pretrain
weights mol_pre_all_h_220816.pt), RDKit ETKDG conformer generation per molecule
(1 conformer, the unimol-tools default).

Output: cache/unimol_emb.parquet, one row per SMILES + 512-d columns,
train (ft_data.csv order) then test (ft_test_smiles.csv), deduplicated on
SMILES - the same cache schema as chemeleon_emb.parquet.

Run (unimol env, GPU; ~6.9k mols):
  ~/miniforge3/envs/unimol/bin/python src/unimol_embeddings.py
"""
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    allsmi = pd.concat([ft[["SMILES"]], test[["SMILES"]]]).drop_duplicates("SMILES").reset_index(drop=True)
    print(f"{len(allsmi)} unique SMILES", flush=True)

    from unimol_tools import UniMolRepr
    t0 = time.time()
    repr_obj = UniMolRepr(data_type="molecule", remove_hs=False, batch_size=64)
    out = repr_obj.get_repr(allsmi)
    R = np.asarray(out, dtype=np.float32)
    assert R.shape == (len(allsmi), 512), R.shape
    print("repr shape", R.shape, f"{time.time()-t0:.0f}s", flush=True)

    edf = pd.DataFrame(R, columns=[f"umol_{i}" for i in range(R.shape[1])])
    edf.insert(0, "SMILES", allsmi["SMILES"])
    edf.to_parquet(os.path.join(CACHE, "unimol_emb.parquet"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
