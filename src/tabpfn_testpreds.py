"""TabPFN test predictions (all-data model per isoform) for the cp17 builder.

Reuses the exact recipe from src/tabpfn_probe.py (PCA-128 on frozen cpmed emb,
TabPFN-v2, same MODEL_PATH) but fits the ALL-DATA model per isoform and predicts
the 750 test SMILES -> cache/ft_test_preds_tabpfn_cpmed.csv (blendN_all format).
The OOF (ft_oof_tabpfn_cpmed.csv) is already saved; this only adds test preds.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
DIRECT = {i: f"{i}_pIC50_direct_inhibition" for i in ISO}
MODEL_PATH = "/home/jackson/.cache/tabpfn/tabpfn-v2-regressor-v2_default.ckpt"
PCA_DIM = 128
warnings.filterwarnings("ignore")


def main():
    from tabpfn import TabPFNRegressor
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    smi = Xtr["SMILES"].values
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    tesmi = test["SMILES"].values

    inh = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv")).drop_duplicates("SMILES").set_index("SMILES")
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")
    targets = pd.DataFrame(index=smi)
    for iso in ISO:
        targets[DIRECT[iso]] = inh[DIRECT[iso]].reindex(smi).combine_first(tdi[DIRECT[iso]].reindex(smi)).values

    cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet")).set_index("SMILES")
    emb_all = cpmed.loc[list(smi) + list(tesmi)].drop(columns=["SMILES"], errors="ignore").values.astype(np.float32)
    emb_all = np.nan_to_num(emb_all)
    # Fit PCA on TRAIN only (same as probe), transform train+test
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(emb_all[:len(smi)])
    Xp_tr = pca.transform(emb_all[:len(smi)]).astype(np.float32)
    Xp_te = pca.transform(emb_all[len(smi):]).astype(np.float32)

    tp_te = {}
    for iso in ISO:
        y = targets[DIRECT[iso]].values
        lab = ~np.isnan(y)
        m = TabPFNRegressor(model_path=MODEL_PATH, device=device, n_jobs=2,
                            ignore_pretraining_limits=True)
        m.fit(Xp_tr[lab], y[lab])
        tp_te[iso] = m.predict(Xp_te)
        print(f"  TabPFN {iso}: test preds done", flush=True)

    out = pd.DataFrame({"SMILES": tesmi, **tp_te})
    out.to_csv(os.path.join(CACHE, "ft_test_preds_tabpfn_cpmed.csv"), index=False)
    print(f"saved ft_test_preds_tabpfn_cpmed.csv {out.shape}", flush=True)


if __name__ == "__main__":
    main()
