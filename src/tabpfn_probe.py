"""R2: TabPFN (v2) probe on FROZEN chemprop_medium embeddings (jeremy's best recipe).

TabPFN is an in-context tabular learner (inference-only, no training). Per
isoform: PCA the frozen cpmed embeddings to a feature budget, then TabPFN OOF
with the regression scaffold seed-0 folds (folds.npy). Report nested Pearson
per isoform + macro R2, and add as a new family member to the cp16 pool
(nested + family-block paranoid).

Env: tabpfn (python 3.11). CPU or GPU (TabPFN uses torch).
"""
import os
import sys
import glob
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")

# Inline the constants (avoid importing run_regression -> rdkit, not in this env)
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
DIRECT = {i: f"{i}_pIC50_direct_inhibition" for i in ISO}
K = 5


def g(tag):
    return sorted(glob.glob(os.path.join(CACHE, f"ft_oof_{tag}*.csv")))


FAMS = {
    "gbm": [os.path.join(CACHE, "oof_blend.csv")],
    "emb": [os.path.join(CACHE, "oof_emb_concat.csv")],
    "ft_frozen": [os.path.join(CACHE, "ft_oof_frozen.csv")],
    "ft_fullft": g("fullft"),
    "ft_ext": g("ext"),
    "ft_pre": g("pre"),
    "ft_dmpnn": g("dmpnn"),
    "ft_cpmed": g("cpmed"),
    "ft_cpchm": g("cpchm"),
    "ft_chmridge": g("chmridge"),
    "ft_d2d6": g("d2d6"),
    "ft_near": g("near"),
    "ft_admchm": g("admchm"),
    "ft_admmed": g("admmed"),
    "ft_unimol": g("unimol"),
    "ft_umtmL2": g("umtmL2"),
    "ft_umtmL1": g("umtmL1"),
    "ft_umtmL3": g("umtmL3"),
    "ft_umtmL4": g("umtmL4"),
    "ft_monridge": [os.path.join(CACHE, "ft_oof_monridge.csv")],
}
CP16 = ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre", "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge", "ft_d2d6", "ft_admchm", "ft_admmed", "ft_unimol", "ft_umtmL2", "ft_umtmL1", "ft_umtmL3", "ft_umtmL4", "ft_monridge"]

warnings.filterwarnings("ignore")

MODEL_PATH = "/home/jackson/.cache/tabpfn/tabpfn-v2-regressor-v2_default.ckpt"
PCA_DIM = 128


def zavg(files):
    dfs = [pd.read_csv(f) for f in files]
    df = dfs[0].copy()
    for iso in ISO:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    return df


def greedy(zs, y, m, rounds=12):
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    for _ in range(rounds):
        bestk, bestr = None, -2
        for k in keys:
            cand = sum((w[j] + (1 if j == k else 0)) * zs[j] for j in keys)
            tot = sum(w.values()) + 1
            r = float(pearsonr(y[m], (cand / tot)[m]).statistic)
            if r > bestr:
                bestr, bestk = r, k
        w[bestk] += 1
    return {k: v / sum(w.values()) for k, v in w.items()}


def nested_pool(oofs, names, folds, targets, smi):
    nested = {}
    for iso in ISO:
        y = targets[DIRECT[iso]].values
        m = ~np.isnan(y)
        zs_all = {}
        for k, df in oofs.items():
            if k not in names:
                continue
            v = df[iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zs_all[k] = z
        rs = []
        for f in range(K):
            trm = m & (folds != f)
            vam = m & (folds == f)
            zt = {}
            for k in zs_all:
                v = zs_all[k].copy()
                mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1)
            w = greedy(zt, y, trm)
            zf = sum(w[k] * zt[k] for k in w)
            rs.append(float(pearsonr(y[vam], zf[vam]).statistic))
        nested[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
    return nested, float(np.mean([nested[i] ** 2 for i in ISO]))


def main():
    from tabpfn import TabPFNRegressor

    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    smi = Xtr["SMILES"].values
    folds = np.load(os.path.join(CACHE, "folds.npy"))
    targets = pd.DataFrame(index=smi)
    inh = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv")).drop_duplicates("SMILES").set_index("SMILES")
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")
    for iso in ISO:
        d = inh[DIRECT[iso]].reindex(smi)
        d2 = tdi[DIRECT[iso]].reindex(smi)
        targets[DIRECT[iso]] = d.combine_first(d2)

    # Frozen chemprop_medium embeddings (600-d) -> PCA 256
    cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet")).set_index("SMILES")
    emb = cpmed.loc[smi].drop(columns=["SMILES"], errors="ignore").values.astype(np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    print(f"cpmed emb: {emb.shape}, PCA -> {PCA_DIM}", flush=True)
    # Fit PCA on train emb only (no test leakage; test not needed for OOF)
    pca = PCA(n_components=PCA_DIM, random_state=0).fit(emb)
    Xp = pca.transform(emb).astype(np.float32)
    print(f"  explained var: {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    # TabPFN OOF per isoform
    device = "cuda" if torch_available() else "cpu"
    print(f"TabPFN device: {device}", flush=True)
    tp_oof = {iso: np.full(len(smi), np.nan) for iso in ISO}
    for iso in ISO:
        y = targets[f"{iso}_pIC50_direct_inhibition"].values
        lab = ~np.isnan(y)
        for f in range(K):
            tr = np.where(lab & (folds != f))[0]
            va = np.where(lab & (folds == f))[0]
            if len(tr) == 0 or len(va) == 0:
                continue
            m = TabPFNRegressor(model_path=MODEL_PATH, device=device, n_jobs=2,
                                ignore_pretraining_limits=True)
            m.fit(Xp[tr], y[tr])
            tp_oof[iso][va] = m.predict(Xp[va])
        print(f"  TabPFN {iso}: OOF done", flush=True)

    tp_df = pd.DataFrame({"SMILES": smi, **{iso: tp_oof[iso] for iso in ISO}})
    tp_path = os.path.join(CACHE, "ft_oof_tabpfn_cpmed.csv")
    tp_df.to_csv(tp_path, index=False)
    print(f"  saved {os.path.basename(tp_path)}", flush=True)

    # Standalone nested
    oofs = {k: zavg(FAMS[k]) for k in CP16 if FAMS.get(k)}
    oofs["ft_tabpfn_cpmed"] = tp_df
    names16 = [n for n in CP16 if FAMS.get(n)]
    names17 = names16 + ["ft_tabpfn_cpmed"]

    tp_nested, tp_macro = nested_pool({"ft_tabpfn_cpmed": tp_df}, ["ft_tabpfn_cpmed"], folds, targets, smi)
    print(f"\n  TabPFN-cpmed STANDALONE nested: " + "  ".join(f"{i} {tp_nested[i]:.4f}" for i in ISO) + f"  macro {tp_macro:.4f}")

    cp16_nested, cp16_macro = nested_pool(oofs, names16, folds, targets, smi)
    cp17_nested, cp17_macro = nested_pool(oofs, names17, folds, targets, smi)
    print(f"  cp16  nested: " + "  ".join(f"{i} {cp16_nested[i]:.4f}" for i in ISO) + f"  macro {cp16_macro:.4f}")
    print(f"  cp17 (cp16+tp) nested: " + "  ".join(f"{i} {cp17_nested[i]:.4f}" for i in ISO) + f"  macro {cp17_macro:.4f}")
    print(f"\n  Delta macro R2 (cp17 - cp16): {cp17_macro - cp16_macro:+.4f}")
    for iso in ISO:
        print(f"    {iso}: {cp16_nested[iso]:.4f} -> {cp17_nested[iso]:.4f}  ({cp17_nested[iso]-cp16_nested[iso]:+.4f})")

    out = {"tabpfn_standalone": tp_nested, "tabpfn_macro": tp_macro,
           "cp16": cp16_nested, "cp16_macro": cp16_macro,
           "cp17": cp17_nested, "cp17_macro": cp17_macro,
           "delta_macro": cp17_macro - cp16_macro}
    with open(os.path.join(CACHE, "r2_tabpfn_probe.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved r2_tabpfn_probe.json")


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
